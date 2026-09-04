//+------------------------------------------------------------------+
//| Study.mq5                                                        |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property strict

#include "Trajectory.mqh"
//---
input group                       "---- D2Skill training ----"
input ENUM_D2SKILL_STAGE          InpD2SkillStage = D2Skill_D2_STAGE_FORMATION; //Lifecycle stage
input ENUM_D2SKILL_MODE           InpD2SkillExecutionMode = D2_COLLECT; //Bank execution mode
input ED2SkillRepresentation      InpD2SkillRepresentation = D2SkillDirectionMagnitude; //Bank representation
input bool                        InpD2SkillResetBanksOnRepresentationMismatch = false; //Reset incompatible banks
input bool                        InpD2SkillRecreateIncompatibleCheckpoint = false; //Recreate incompatible policy checkpoint
//---
input group                       "---- Actor-Critic training ----"
input datetime                    Start         = D'2024.01.01'; //Training period start
input datetime                    End           = D'2026.01.01'; //Training period end
input int                         Iterations    = 1000000; //Training iterations
input int                         EpisodeBars   = 2 * StackSize; //Bars per episode
input double                      MinBalance    = 50.0; //Minimum account balance
input int                         UpdatePolicy  = 5; //Actor update interval
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int            Epochs        = 0;
CNet Actor;
CNet Q1;
CNet Q2;

CBufferFloat State;
CBufferFloat TimeState;
CBufferFloat Account;
CBufferFloat NextAccount;
CBufferFloat CurrentAction;
CBufferFloat BaselineAction;
CBufferFloat BaselineAccount;
CBufferFloat BaselineNextAccount;
CBufferFloat TeacherAction;
CBufferFloat RandomAction;
CBufferFloat CriticInput;
CBufferFloat ScalarTarget;
CBufferFloat ActorObjective;
ulong TeacherTransitions = 0;
ulong TeacherCriticTransitions = 0;
ulong RandomCriticTransitions = 0;
double TeacherRewardSum = 0;
ulong PolicyTransitions = 0;
ulong PolicySkippedInvalid = 0;
ulong PairedTransitions = 0;
double PairedDeltaSum = 0;
SD2SkillActorForwardState    ActorForwardState;
SD2SkillActorForwardState    BaselineActorForwardState;
SD2SkillEpisodeOutcome       BaselineEpisodeOutcome;
SD2SkillEpisodeOutcome       SkillEpisodeOutcome;
bool                          PairedEpisodeActive = false;
//+------------------------------------------------------------------+
//| Resets only D2Skill episode influence, preserving bank params.   |
//+------------------------------------------------------------------+
bool ResetD2SkillEpisodeInfluence(CNet &actor)
  {
   CNeuronBaseOCL *layer = actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill.Ready() || !skill.ResetEpisodeInfluence())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Adds one simulated account transition to its own episode result.|
//+------------------------------------------------------------------+
bool AccumulateEpisodeOutcome(SD2SkillEpisodeOutcome &outcome,
                              CBufferFloat *account,
                              const double reward)
  {
   if(!account || account.Total() != AccountDescr || !MathIsValidNumber(reward) ||
      (account.GetIndex() >= 0 && !account.BufferRead()))
      ReturnFalse;
   const double balance = MathMax(0.0, double(account[0]) * EtalonBalance);
   const double equity = MathMax(0.0, double(account[2]) * MathMax(balance, 1.0));
   const double drawdown = MathMin(0.0, double(account[1]));
   const double cost = MathMax(0.0, -reward);
   const double risk = MathMax(0.0, double(account[4]) + double(account[5]));
   return(outcome.Accumulate(reward, balance, equity, drawdown, cost, 1, risk));
  }
//+------------------------------------------------------------------+
//| Loads OrCreateD2SkillPolicies.                                     |
//+------------------------------------------------------------------+
bool LoadOrCreateD2SkillPolicies(void)
  {
   return D2SkillLoadOrCreatePolicySet(Actor, Q1, Q2, true);
  }
//+------------------------------------------------------------------+
//| Saves D2SkillPolicies.                                             |
//+------------------------------------------------------------------+
bool SaveD2SkillPolicies(void)
  {
//--- Forecast artifacts are immutable throughout Actor-Critic study. This
//--- bounded lifecycle check avoids hashing three large files every batch.
   return D2SkillSavePolicySet(Actor, Q1, Q2);
  }
//+------------------------------------------------------------------+
//| Implements PrepareHistory.                                       |
//+------------------------------------------------------------------+
bool PrepareHistory(int &first_position, int &last_position)
  {
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
      ReturnFalse;
   int wait = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars && CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars && MACD.BarsCalculated() >= bars);
      Sleep(100);
      wait++;
     }
   while(!calculated && wait < 100);
   if(!calculated)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
      ReturnFalse;
   first_position = end + 1;
//--- CreateBuffers(position,...,forecast) forms its state from
//--- position+NForecast and needs HistoryBars bars behind it.  position is
//--- therefore the end of the realized future window, not its first bar.
   last_position = start - HistoryBars - NForecast;
   return (last_position > first_position);
  }
//+------------------------------------------------------------------+
//| Implements PolicyBackward.                                       |
//+------------------------------------------------------------------+
bool PolicyBackward(void)
  {
//--- PolicyBackward builds Q+1 in this buffer.  ScalarTarget must remain the
//--- immutable TD target consumed by both Critic backward passes.
   return PolicyBackward(Actor, Q1, GetPointer(ActorObjective), GetPointer(D2SkillMarket), -1);
  }
//+------------------------------------------------------------------+
//| Builds one causal baseline/skill pair on the same state.         |
//+------------------------------------------------------------------+
bool BuildPairedTransition(const int position, const bool trace_td, bool &terminal,
                           double &reward, double &delta_j, bool &paired,
                           bool &baseline_terminal)
  {
   terminal = false;
   reward = 0;
   delta_j = 0;
   paired = false;
   baseline_terminal = false;
   CNeuronBaseOCL *layer = Actor.Layer(1);
   CD2Skill *skill = (layer && layer.Type() == defNeuronD2Skill ? (CD2Skill*)layer : NULL);
   paired = D2SkillUsePairedHindsight(skill, ActorForwardState);
   bool task_enabled = false;
   bool step_enabled = false;
   D2SkillD2BankFlags(task_enabled, step_enabled);
   static bool pair_setup_logged = false;
   if(!pair_setup_logged)
     {
      PrintFormat("D2Skill paired setup stage=%d layer_type=%d skill_ready=%s paired=%s",
                  D2SkillD2Stage, (layer ? layer.Type() : -1),
                  (skill != NULL && skill.Ready() ? "true" : "false"),
                  (paired ? "true" : "false"));
      pair_setup_logged = true;
     }
   double baseline_reward = 0;
   if(paired)
     {
      if(!D2SkillActorForwardStateReady(BaselineActorForwardState) ||
         !D2SkillRestorePairedActorState(BaselineActorForwardState, skill,
                                         false, false, "baseline_start"))
        {
         Print("TrainTransition stage=baseline_restore");
         ReturnFalse;
        }
      if(!Actor.feedForward(GetPointer(BaselineAccount), 1, false, GetPointer(D2SkillMarket), -1) ||
         !ReadAction(Actor, GetPointer(BaselineAction)) ||
         !AdvanceAccount(GetPointer(BaselineAccount), GetPointer(BaselineAction), position, MinBalance,
                         GetPointer(BaselineNextAccount), baseline_reward, baseline_terminal) ||
         !D2SkillCaptureActorForwardState(BaselineActorForwardState) ||
         !AccumulateEpisodeOutcome(BaselineEpisodeOutcome, GetPointer(BaselineNextAccount),
                                   baseline_reward))
        {
         D2SkillRestorePairedActorState(ActorForwardState, skill,
                                        task_enabled, step_enabled,
                                        "baseline_failure");
         Print("TrainTransition stage=baseline_transition");
         ReturnFalse;
        }
      if(!D2SkillRestorePairedActorState(ActorForwardState, skill,
                                         task_enabled, step_enabled,
                                         "baseline_to_skill"))
        { Print("TrainTransition stage=restore_skill_state"); ReturnFalse; }
     }
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(D2SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(CurrentAction)) ||
      !AdvanceAccount(GetPointer(Account), GetPointer(CurrentAction), position, MinBalance,
                      GetPointer(NextAccount), reward, terminal) ||
      (paired && (!D2SkillCaptureActorForwardState(ActorForwardState) ||
                  !AccumulateEpisodeOutcome(SkillEpisodeOutcome, GetPointer(NextAccount), reward))))
     {
      if(paired)
         D2SkillRestorePairedActorState(ActorForwardState, skill,
                                        task_enabled, step_enabled,
                                        "skill_failure");
      Print("TrainTransition stage=skill_transition");
      ReturnFalse;
     }
   if(paired)
     {
      delta_j = reward - baseline_reward;
      if(!MathIsValidNumber(delta_j))
        {
         D2SkillRestorePairedActorState(ActorForwardState, skill,
                                        task_enabled, step_enabled,
                                        "nonfinite_delta");
         Print("TrainTransition stage=paired_delta_nonfinite");
         ReturnFalse;
        }
      if(trace_td)
         PrintFormat("D2Skill paired iteration=%d JBase=%.8f JSkill=%.8f DeltaJ=%.8f baseline_terminal=%s skill_terminal=%s",
                     position, baseline_reward, reward, delta_j,
                     (baseline_terminal ? "true" : "false"), (terminal ? "true" : "false"));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements TrainTransition.                                      |
//+------------------------------------------------------------------+
bool TrainTransition(const int position, const int iteration, bool &skill_terminal,
                     bool &pair_terminal)
  {
   skill_terminal = false;
   pair_terminal = false;
   const bool trace_td = (iteration < 8 || iteration % 10000 == 0);
   if(!D2SkillForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
     { Print("TrainTransition stage=current_forecast"); ReturnFalse; }
   double reward = 0;
   double delta_j = 0;
   bool paired = false;
   bool baseline_terminal = false;
   if(!BuildPairedTransition(position, trace_td, skill_terminal, reward, delta_j, paired,
                             baseline_terminal))
      ReturnFalse;
   pair_terminal = (paired && D2SkillPairReachedTerminal(baseline_terminal,
                    skill_terminal));
   const double balance = MathMax(0.0, double(Account[0]) * EtalonBalance);
   const double buy_lot = MathMax(0.0, double(CurrentAction[0] - CurrentAction[3]));
   const double sell_lot = MathMax(0.0, double(CurrentAction[3] - CurrentAction[0]));
   if(trace_td)
      PrintFormat("D2Skill action iteration=%d balance=%.2f buy_lot=%.8f buy_tp=%.8f buy_sl=%.8f sell_lot=%.8f sell_tp=%.8f sell_sl=%.8f",
                  iteration, balance, buy_lot, double(CurrentAction[1]), double(CurrentAction[2]),
                  sell_lot, double(CurrentAction[4]), double(CurrentAction[5]));
//--- Utility is emitted only at the common episode boundary. The two branch
//--- outcomes are accumulated independently, never inferred from this reward.
   if(paired && trace_td)
      PrintFormat("D2Skill paired transition base_reward=%.8f skill_reward=%.8f",
                  reward - delta_j, reward);
   const double target_value = reward;
   if(trace_td)
      PrintFormat("D2Skill return iteration=%d reward=%.8f target=%.8f balance=%.2f equity=%.2f "
                  "floating_buy=%.8f floating_sell=%.8f next_buy=%.8f next_sell=%.8f terminal=%s",
                  iteration, reward, target_value, double(NextAccount[0])*EtalonBalance,
                  double(NextAccount[2])*balance, double(NextAccount[6])*MathMax(double(NextAccount[0])*EtalonBalance, 1.0),
                  double(NextAccount[7])*MathMax(double(NextAccount[0])*EtalonBalance, 1.0),
                  double(NextAccount[4]), double(NextAccount[5]),
                  (skill_terminal ? "true" : "false"));
   if(!MathIsValidNumber(target_value) || !ScalarTarget.BufferInit(1, 0) ||
      !ScalarTarget.Update(0, float(target_value)))
     { Print("TrainTransition stage=scalar_target"); ReturnFalse; }
   CNeuronBaseOCL *context_layer = Actor.Layer(0);
   CNeuronBaseOCL *actor_layer = Actor.Layer(3);
   if(!context_layer || !actor_layer)
     { Print("TrainTransition stage=actor_layer"); ReturnFalse; }
//--- Use the live device activation, not the CPU account object.
   if(!BuildCriticInput(context_layer.getOutput(), actor_layer.getOutput(), GetPointer(CriticInput)))
     { Print("TrainTransition stage=critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=q1_forward"); ReturnFalse; }
   if(!Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=q2_forward"); ReturnFalse; }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(trace_td)
     {
      CNeuronBaseOCL *q1_layer = Q1.Layer(3);
      CNeuronBaseOCL *q2_layer = Q2.Layer(3);
      CBufferFloat *q1_output = (q1_layer ? q1_layer.getOutput() : NULL);
      CBufferFloat *q2_output = (q2_layer ? q2_layer.getOutput() : NULL);
      if(!q1_output || !q2_output || q1_output.GetIndex() < 0 || q2_output.GetIndex() < 0 ||
         q1_output.Total() != 1 || q2_output.Total() != 1 ||
         !q1_output.BufferRead() || !q2_output.BufferRead())
        { Print("TrainTransition stage=current_q_read"); ReturnFalse; }
      PrintFormat("D2Skill Q iteration=%d q1=%.8f q2=%.8f td_q1=%.8f td_q2=%.8f",
                  iteration, double(q1_output[0]), double(q2_output[0]),
                  target_value - double(q1_output[0]), target_value - double(q2_output[0]));
     }
//--- A critic gradient has no valid direction on the discontinuous no-trade
//--- plateau.  Keep learning Q there, but let only the realized-future teacher
//--- move Actor back into the executable action manifold.
   const bool executable_action = (IsExecutableOrder(buy_lot, CurrentAction[1], CurrentAction[2]) ||
                                   IsExecutableOrder(sell_lot, CurrentAction[4], CurrentAction[5]));
//---
   if(iteration > 0 && UpdatePolicy > 0 && iteration % UpdatePolicy == 0)
     {
      if(executable_action)
        {
         if(!PolicyBackward())
           { Print("TrainTransition stage=policy_backward"); ReturnFalse; }
         PolicyTransitions++;
        }
      else
         PolicySkippedInvalid++;
     }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=critic_backward"); ReturnFalse; }
//--- The teacher uses realized future bars, not one selected Forecast scenario.
//--- It is deliberately a second Critic-only sample: Q1/Q2 are stateless,
//--- whereas Actor has already performed its only forward for this state.
   double teacher_reward = 0;
   if(!BuildTeacherAction(position, GetPointer(Account), GetPointer(TeacherAction), teacher_reward))
     { Print("TrainTransition stage=teacher_action"); ReturnFalse; }
//---
   if(teacher_reward > 0)
     {
      if(trace_td)
         PrintFormat("D2Skill teacher iteration=%d reward=%.8f buy_lot=%.8f sell_lot=%.8f",
                     iteration, teacher_reward,
                     MathMax(0.0, double(TeacherAction[0] - TeacherAction[3])),
                     MathMax(0.0, double(TeacherAction[3] - TeacherAction[0])));
      //--- Supervised Actor update reuses the current activation; it never advances
      //--- the history stack or invokes a second Actor forward.
      if(!Actor.backProp(GetPointer(TeacherAction), GetPointer(D2SkillMarket), -1))
        { Print("TrainTransition stage=teacher_actor_backward"); ReturnFalse; }
      TeacherTransitions++;
      TeacherRewardSum += teacher_reward;
     }
//--- Critic receives every oracle action, including a losing one.  Only Actor
//--- is restricted to profitable oracle supervision above.
   if(!ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(teacher_reward)) ||
      !BuildCriticInput(context_layer.getOutput(), GetPointer(TeacherAction), GetPointer(CriticInput)))
     { Print("TrainTransition stage=teacher_critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=teacher_critic_forward"); ReturnFalse; }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=teacher_critic_backward"); ReturnFalse; }
   TeacherCriticTransitions++;
//--- A second critic-only sample is uniformly random over executable lots and
//--- stop distances.  It never runs Actor backward or advances policy history.
   double random_reward = 0;
   if(!BuildRandomAction(position, GetPointer(Account), GetPointer(RandomAction), random_reward))
     { Print("TrainTransition stage=random_action"); ReturnFalse; }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(random_reward > 0)
     {
      if(!Actor.backProp(GetPointer(RandomAction), GetPointer(D2SkillMarket), -1))
        { Print("TrainTransition stage=random_actor_backward"); ReturnFalse; }
     }
   if(!ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(random_reward)) ||
      !BuildCriticInput(context_layer.getOutput(), GetPointer(RandomAction), GetPointer(CriticInput)))
     { Print("TrainTransition stage=random_critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=random_critic_forward"); ReturnFalse; }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1))
     { Print("TrainTransition stage=random_critic_backward"); ReturnFalse; }
   RandomCriticTransitions++;
//--- Only the three declared Forecast trainable buffers are read here. The
//--- complete persistent Codebook is checked at checkpoint/lifecycle bounds.
   if(!D2SkillVerifyFrozenWeightsExact())
     { Print("TrainTransition stage=frozen_forecast_check"); ReturnFalse; }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements TrainD2SkillActorCritic.                                |
//+------------------------------------------------------------------+
void TrainD2SkillActorCritic(void)
  {
   int first = 0, last = 0;
   if(!PrepareHistory(first, last))
     { PrintFormat("%s -> %d history unavailable", __FUNCTION__, __LINE__); return; }
   uint shown = GetTickCount();
   int completed_iterations = 0;
   int failed_iteration = -1;
   int failed_position = -1;
   int position = last;
   int episode = 0;
   bool episode_reset_failed = false;
   TeacherTransitions = 0;
   TeacherCriticTransitions = 0;
   RandomCriticTransitions = 0;
   TeacherRewardSum = 0;
   PolicyTransitions = 0;
   PolicySkippedInvalid = 0;
   PairedTransitions = 0;
   PairedDeltaSum = 0;
   for(int iteration = 0; iteration < Iterations && !IsStopped(); iteration++)
     {
      if(episode == 0)
        {
         if(!ResetD2SkillEpisodeInfluence(Actor) ||
            !D2SkillForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
            break;
         const vector<float> sampled = SampleAccount(GetPointer(State), Rates[position].time, EtalonBalance, MinBalance);
         if(sampled.Size() != AccountDescr || !Account.AssignArray(sampled) ||
            (Account.GetIndex() >= 0 && !Account.BufferWrite()) ||
            !BaselineAccount.AssignArray(GetPointer(Account)) ||
            (BaselineAccount.GetIndex() >= 0 && !BaselineAccount.BufferWrite()) ||
            !BaselineEpisodeOutcome.Reset() || !SkillEpisodeOutcome.Reset() || !Actor.Clear())
            break;
         CNeuronBaseOCL *layer = Actor.Layer(1);
         CD2Skill *skill = (layer && layer.Type() == defNeuronD2Skill ? (CD2Skill*)layer : NULL);
         PairedEpisodeActive = (D2SkillUsePairedHindsight(skill, ActorForwardState) &&
                                D2SkillActorForwardStateReady(BaselineActorForwardState));
         if(PairedEpisodeActive && (!D2SkillCaptureActorForwardState(ActorForwardState) ||
                                    !D2SkillCaptureActorForwardState(BaselineActorForwardState)))
            break;
        }
      bool skill_terminal = false;
      bool pair_terminal = false;
      //---
      if(!TrainTransition(position, iteration, skill_terminal, pair_terminal))
        {
         failed_iteration = iteration;
         failed_position = position;
         if(!ResetD2SkillEpisodeInfluence(Actor))
            episode_reset_failed = true;
         break;
        }
      completed_iterations++;
      //---
      if((NextAccount.GetIndex() >= 0 && !NextAccount.BufferRead()) || !Account.AssignArray(GetPointer(NextAccount)) ||
         (Account.GetIndex() >= 0 && !Account.BufferWrite()) ||
         (PairedEpisodeActive &&
          ((BaselineNextAccount.GetIndex() >= 0 && !BaselineNextAccount.BufferRead()) ||
           !BaselineAccount.AssignArray(GetPointer(BaselineNextAccount)) ||
           (BaselineAccount.GetIndex() >= 0 && !BaselineAccount.BufferWrite()))))
        {
         failed_iteration = iteration;
         failed_position = position;
         if(!ResetD2SkillEpisodeInfluence(Actor))
            episode_reset_failed = true;
         break;
        }
      position--;
      episode++;
      //---
      const bool episode_limit = (position < first ||
                                  episode >= MathMax(1, EpisodeBars));
      double terminal_delta = 0.0;
      bool utility_applied = false;
      bool influence_reset = false;
      bool episode_closed = false;
      if(!D2SkillCloseOfflineEpisode(Actor, PairedEpisodeActive, pair_terminal,
                                     episode_limit, BaselineEpisodeOutcome,
                                     SkillEpisodeOutcome, terminal_delta,
                                     utility_applied, influence_reset,
                                     episode_closed))
        {
         failed_iteration = iteration;
         failed_position = position;
         episode_reset_failed = true;
         break;
        }
      if(episode_closed && !influence_reset)
        {
         failed_iteration = iteration;
         failed_position = position;
         episode_reset_failed = true;
         break;
        }
      if(episode_closed)
        {
         if(PairedEpisodeActive)
           {
            PairedTransitions++;
            PairedDeltaSum += terminal_delta;
            PrintFormat("D2Skill paired terminal JBase=%.8f JSkill=%.8f DeltaJ=%.8f utility_mutated=%s base_balance=%.2f skill_balance=%.2f base_duration=%u skill_duration=%u",
                        BaselineEpisodeOutcome.Outcome(), SkillEpisodeOutcome.Outcome(), terminal_delta,
                        (utility_applied ? "true" : "false"),
                        BaselineEpisodeOutcome.Balance(), SkillEpisodeOutcome.Balance(),
                        BaselineEpisodeOutcome.Duration(), SkillEpisodeOutcome.Duration());
           }
         if(position < first)
            position = last;
         episode = 0;
        }
      //---
      if(GetTickCount() - shown > 500)
        {
         const double q1_rmse = MathSqrt(MathMax(0.0, double(Q1.getRecentAverageError()))) / MathMax(EtalonBalance, 1.0);
         const double q2_rmse = MathSqrt(MathMax(0.0, double(Q2.getRecentAverageError()))) / MathMax(EtalonBalance, 1.0);
         const double teacher_mean = (TeacherTransitions > 0 ? TeacherRewardSum / double(TeacherTransitions) : 0.0);
         Comment(StringFormat("D2Skill AC %6.2f%% Q1 rRMSE %.8f Q2 rRMSE %.8f teacherA %I64u teacherQ %I64u randomQ %I64u R %.2f policy %I64u skip %I64u",
                              100.0 * iteration / MathMax(Iterations, 1), q1_rmse, q2_rmse,
                              TeacherTransitions, TeacherCriticTransitions, RandomCriticTransitions,
                              teacher_mean, PolicyTransitions, PolicySkippedInvalid));
         shown = GetTickCount();
        }
     }
   if(!ResetD2SkillEpisodeInfluence(Actor))
      episode_reset_failed = true;
   Comment("");
//---
   if(completed_iterations != Iterations || episode_reset_failed)
     {
      if(episode_reset_failed)
         PrintFormat("%s -> %d publication aborted: episode influence reset failed completed=%d/%d",
                     __FUNCTION__, __LINE__, completed_iterations, Iterations);
      else
         if(failed_iteration >= 0)
            PrintFormat("%s -> %d publication aborted: TrainTransition failed iteration=%d position=%d completed=%d/%d",
                        __FUNCTION__, __LINE__, failed_iteration, failed_position, completed_iterations, Iterations);
         else
            PrintFormat("%s -> %d publication aborted: stop requested completed=%d/%d",
                        __FUNCTION__, __LINE__, completed_iterations, Iterations);
      return;
     }
   if(!SaveD2SkillPolicies())
      PrintFormat("%s -> %d save failed", __FUNCTION__, __LINE__);
   if(PairedTransitions > 0)
      PrintFormat("D2Skill paired summary transitions=%I64u mean_delta_j=%.8f",
                  PairedTransitions, PairedDeltaSum / double(PairedTransitions));
  }
//+------------------------------------------------------------------+
//| Function OnInit.                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
   if(!D2SkillConfigureRuntime(InpD2SkillStage, InpD2SkillMode,
                               InpD2SkillExecutionMode, InpD2SkillUtilityAware,
                               InpD2SkillMinUtility, InpD2SkillUtilityScale,
                               false, false, InpD2SkillRepresentation,
                               InpD2SkillResetBanksOnRepresentationMismatch,
                               InpD2SkillRecreateIncompatibleCheckpoint) ||
      !D2SkillInitIndicators() || !D2SkillLoadForecastInference() || !LoadOrCreateD2SkillPolicies() ||
      !D2SkillConfigureActorCriticUpdates(Actor, Q1, Q2) ||
      !D2SkillVerifyFrozenForecastExact())
     {
      PrintFormat("D2Skill Actor-Critic initialization failed at line %d error=%d", __LINE__, GetLastError());
      return INIT_FAILED;
     }
   if(!D2SkillConfigureD2UtilityMode(D2Skill_D2_UTILITY_PAIRED_HINDSIGHT) ||
      ((D2SkillD2ExecutionMode == D2_EVALUATE ||
        D2SkillD2ExecutionMode == D2_ONLINE_CALIBRATION) &&
       D2SkillD2Mode != D2Skill_D2_MODE_BASE &&
       (!D2SkillInitActorForwardState(ActorForwardState, Actor) ||
        !D2SkillInitActorForwardState(BaselineActorForwardState, Actor))))
     {
      PrintFormat("D2Skill paired snapshot initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return INIT_FAILED;
     }
   if(!EventSetMillisecondTimer(1))
     {
      PrintFormat("D2Skill Actor-Critic timer initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
//--- Only a completed explicit training loop persists all six artifacts and
//--- writes the hash manifest last. Deinit must not create a mixed generation.
   if(D2SkillFrozenBaselineReady && D2SkillForecast != NULL && !D2SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   D2SkillForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Runs the one-shot training lifecycle.                            |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   TrainD2SkillActorCritic();
   ExpertRemove();
  }
//+------------------------------------------------------------------+
