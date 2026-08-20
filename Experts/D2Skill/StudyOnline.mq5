//+------------------------------------------------------------------+
//| StudyOnline.mq5                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property strict
#include "Trajectory.mqh"
//---
input group                       "---- D2Skill online ----"
input ENUM_D2SKILL_STAGE          InpD2SkillStage = D2Skill_D2_STAGE_ONLINE; //Lifecycle stage
input ENUM_D2SKILL_MODE           InpD2SkillExecutionMode = D2_ONLINE_CALIBRATION; //Bank execution mode
//---
input group                       "---- Online learning ----"
input int                         UpdatePolicy  = ActorUpdate; //Actor update interval
input int                         UpdateTargets = TargetUpdate; //Target-network update interval
input float                       Tau           = tau; //Target-network smoothing
input int                         CheckpointTransitions = 256; //Transitions per checkpoint
input double                      MinBalance = 50.0; //Minimum account balance
datetime Start = 0;
datetime End = 0;
int Epochs = 0;
CNet Actor, TargetActor, Q1, Q2, TargetQ1, TargetQ2;
CBufferFloat State, TimeState, Account, Action, CriticInput, TargetCriticInput, ScalarTarget;
CBufferFloat PreviousAccount;
CBufferFloat BaselineVirtualAccount, SkillVirtualAccount;
CBufferFloat BaselineVirtualNext, SkillVirtualNext;
CBufferFloat BaselineVirtualAction, SkillVirtualAction;
CD2SkillActorForwardSnapshot ActorForwardSnapshot;
CD2SkillActorForwardSnapshot BaselineActorForwardSnapshot;
SD2SkillEpisodeOutcome BaselineEpisodeOutcome, SkillEpisodeOutcome;
double PreviousBalance = 0, PreviousEquity = 0;
double PreviousMarginPenalty = 0;
ulong Transition = 0;
bool HavePrevious = false;
bool PolicyCheckpointPending = false;
bool PolicyCheckpointSafe = true;
bool D2SkillOnlineInitialized = false;
bool OnlinePairActive = false;
bool OnlineVirtualTransitionPending = false;
bool OnlineBaselineTerminal = false;
bool OnlineSkillTerminal = false;
ulong OnlineEpisodeTransitions = 0;
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
//| Accumulates one causal online virtual account outcome.            |
//+------------------------------------------------------------------+
bool AccumulateOnlineOutcome(SD2SkillEpisodeOutcome &outcome,
                             CBufferFloat *account, const double reward)
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
//| Initializes the independent Base and Skill online trajectories.  |
//+------------------------------------------------------------------+
bool StartOnlinePair(CBufferFloat *account)
  {
   CNeuronBaseOCL *layer = Actor.Layer(1);
   CD2Skill *skill = (layer && layer.Type() == defNeuronD2Skill ? (CD2Skill*)layer : NULL);
   if(!account)
      ReturnFalse;
   if(!skill || !D2SkillUsePairedHindsight(skill, ActorForwardSnapshot))
     {
      OnlinePairActive = false;
      OnlineVirtualTransitionPending = false;
      return(true);
     }
   if(!BaselineActorForwardSnapshot.Ready() || !BaselineVirtualAccount.AssignArray(account) ||
      !SkillVirtualAccount.AssignArray(account) || !BaselineEpisodeOutcome.Reset() ||
      !SkillEpisodeOutcome.Reset() || !ActorForwardSnapshot.Capture() ||
      !BaselineActorForwardSnapshot.Capture())
      ReturnFalse;
   OnlinePairActive = true;
   OnlineVirtualTransitionPending = false;
   OnlineBaselineTerminal = false;
   OnlineSkillTerminal = false;
   OnlineEpisodeTransitions = 0;
   return(true);
  }
//+------------------------------------------------------------------+
//| Advances both virtual trajectories using their previous actions. |
//+------------------------------------------------------------------+
bool AdvanceOnlinePair(void)
  {
   if(!OnlinePairActive || !OnlineVirtualTransitionPending)
      return(true);
   OnlineVirtualTransitionPending = false;
   double baseline_reward = 0;
   double skill_reward = 0;
   if(Rates.Size() < 2 || !AdvanceAccount(GetPointer(BaselineVirtualAccount),
                                         GetPointer(BaselineVirtualAction), 1,
                                         MinBalance, GetPointer(BaselineVirtualNext),
                                         baseline_reward,
                                         OnlineBaselineTerminal))
      ReturnFalse;
   if(!AccumulateOnlineOutcome(BaselineEpisodeOutcome,
                               GetPointer(BaselineVirtualNext),
                               baseline_reward))
      ReturnFalse;
   if(!AdvanceAccount(GetPointer(SkillVirtualAccount), GetPointer(SkillVirtualAction), 1,
                     MinBalance, GetPointer(SkillVirtualNext),
                     skill_reward, OnlineSkillTerminal))
      ReturnFalse;
   if(!AccumulateOnlineOutcome(SkillEpisodeOutcome,
                               GetPointer(SkillVirtualNext),
                               skill_reward))
      ReturnFalse;
   if(!BaselineVirtualAccount.AssignArray(GetPointer(BaselineVirtualNext)) ||
      !SkillVirtualAccount.AssignArray(GetPointer(SkillVirtualNext)))
      ReturnFalse;
   OnlineEpisodeTransitions++;
   return(true);
  }
//+------------------------------------------------------------------+
//| Applies terminal paired DeltaJ and clears episode influence.     |
//+------------------------------------------------------------------+
bool FinishOnlinePair(void)
  {
   if(!OnlinePairActive)
      return(true);
   const double delta_j = SkillEpisodeOutcome.Outcome() - BaselineEpisodeOutcome.Outcome();
   if(!MathIsValidNumber(delta_j) || !D2SkillUpdateD2UtilityDelta(Actor, delta_j) ||
      !ResetD2SkillEpisodeInfluence(Actor))
      ReturnFalse;
   PrintFormat("D2Skill online paired terminal JBase=%.8f JSkill=%.8f DeltaJ=%.8f "
               "base_balance=%.2f skill_balance=%.2f base_drawdown=%.2f skill_drawdown=%.2f "
               "base_cost=%.2f skill_cost=%.2f base_risk=%.2f skill_risk=%.2f "
               "base_duration=%u skill_duration=%u transitions=%I64u",
               BaselineEpisodeOutcome.Outcome(), SkillEpisodeOutcome.Outcome(), delta_j,
               BaselineEpisodeOutcome.Balance(), SkillEpisodeOutcome.Balance(),
               BaselineEpisodeOutcome.Drawdown(), SkillEpisodeOutcome.Drawdown(),
               BaselineEpisodeOutcome.Cost(), SkillEpisodeOutcome.Cost(),
               BaselineEpisodeOutcome.Risk(), SkillEpisodeOutcome.Risk(),
               BaselineEpisodeOutcome.Duration(), SkillEpisodeOutcome.Duration(),
               OnlineEpisodeTransitions);
   OnlinePairActive = false;
   OnlineVirtualTransitionPending = false;
   OnlineBaselineTerminal = false;
   OnlineSkillTerminal = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Produces a Base action without disturbing the Skill recurrent state. |
//+------------------------------------------------------------------+
bool BuildOnlineBaselineAction(void)
  {
   if(!OnlinePairActive)
      return(true);
   CNeuronBaseOCL *layer = Actor.Layer(1);
   CD2Skill *skill = (layer && layer.Type() == defNeuronD2Skill ? (CD2Skill*)layer : NULL);
   if(!skill)
      ReturnFalse;
   bool task_enabled = false;
   bool step_enabled = false;
   D2SkillD2BankFlags(task_enabled, step_enabled);
   if(!D2SkillRestorePairedActorState(BaselineActorForwardSnapshot, skill,
                                      false, false, "online_baseline_start") ||
      !Actor.feedForward(GetPointer(BaselineVirtualAccount), 1, false,
                         GetPointer(D2SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(BaselineVirtualAction)) ||
      !D2SkillValidateAction(GetPointer(BaselineVirtualAction)) ||
      !BaselineActorForwardSnapshot.Capture() ||
      !D2SkillRestorePairedActorState(ActorForwardSnapshot, skill,
                                      task_enabled, step_enabled,
                                      "online_baseline_to_skill"))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for ClearPreviousTransit... |
//+------------------------------------------------------------------+
bool ClearPreviousTransition(void)
  {
   if(OnlinePairActive && !FinishOnlinePair())
      ReturnFalse;
   HavePrevious = false;
   PreviousAccount.Clear();
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   PreviousMarginPenalty = 0;
   if(!Actor.Clear() || !ResetD2SkillEpisodeInfluence(Actor) || !D2SkillMarket.Clear())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements StorePrevious.                                        |
//+------------------------------------------------------------------+
bool StorePrevious(void)
  {
   return (PreviousAccount.AssignArray(GetPointer(Account)) &&
           (PreviousAccount.GetIndex() < 0 || PreviousAccount.BufferWrite()));
  }
//+------------------------------------------------------------------+
//| Target objects are runtime-only. They start as exact copies o... |
//+------------------------------------------------------------------+
bool LoadOnlineTargets(void)
  {
   if(!D2SkillLoadPolicyNet(TargetActor, D2Skill_ACTOR_FILE) ||
      !D2SkillLoadPolicyNet(TargetQ1, D2Skill_Q1_FILE) || !D2SkillLoadPolicyNet(TargetQ2, D2Skill_Q2_FILE))
      ReturnFalse;
   TargetActor.SetOpenCL(D2SkillMarket.GetOpenCL());
   TargetQ1.SetOpenCL(D2SkillMarket.GetOpenCL());
   TargetQ2.SetOpenCL(D2SkillMarket.GetOpenCL());
   if(!D2SkillValidatePolicyShape(TargetActor, false) || !D2SkillValidatePolicyShape(TargetQ1, true) ||
      !D2SkillValidatePolicyShape(TargetQ2, true))
      ReturnFalse;
   TargetActor.TrainMode(false);
   TargetQ1.TrainMode(false);
   TargetQ2.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements TrainPreviousTransition.                              |
//+------------------------------------------------------------------+
bool TrainPreviousTransition(const bool terminal)
  {
//--- The live Forecast is at s_next.  Online reward is one step, so TD
//--- bootstrap uses runtime-only target objects initialized from live files.
   double q1_target = 0, q2_target = 0;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!terminal)
     {
      if(!TargetActor.feedForward(GetPointer(Account), 1, false, GetPointer(D2SkillMarket), -1))
         ReturnFalse;
      CNeuronBaseOCL *target_context = TargetActor.Layer(0);
      CNeuronBaseOCL *target_actor_layer = TargetActor.Layer(3);
      if(!target_context || !target_actor_layer ||
         !BuildCriticInput(target_context.getOutput(), target_actor_layer.getOutput(), GetPointer(TargetCriticInput)) ||
         !TargetQ1.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(D2SkillMarket), -1) ||
         !TargetQ2.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(D2SkillMarket), -1))
         ReturnFalse;
      CNeuronBaseOCL *target_q1_layer = TargetQ1.Layer(3), *target_q2_layer = TargetQ2.Layer(3);
      if(!target_q1_layer || !target_q2_layer)
         ReturnFalse;
      //--- TD target is consumed by CPU scalar arithmetic.  getResults is the
      //--- library CPU boundary and also supports a host-only final critic layer.
      CBufferFloat *qt1 = NULL, *qt2 = NULL;
      TargetQ1.getResults(qt1);
      TargetQ2.getResults(qt2);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!qt1 || !qt2 || qt1.Total() != 1 || qt2.Total() != 1)
        {
         DeleteObj(qt1);
         DeleteObjAndFalse(qt2);
        }
      q1_target = double(qt1[0]);
      q2_target = double(qt2[0]);
      DeleteObj(qt1);
      DeleteObj(qt2);
     }
   double observed = (AccountInfoDouble(ACCOUNT_EQUITY) - PreviousEquity +
                      AccountInfoDouble(ACCOUNT_BALANCE) - PreviousBalance) * PreviousBalance / EtalonBalance +
                     PreviousMarginPenalty;
//+------------------------------------------------------------------+
//| Match offline D2Skill: missing a move is penalized at one execu... |
//+------------------------------------------------------------------+
   if(MathMax(double(Account[4]), double(Account[5])) < Symb.LotsMin())
     {
      const double point_cost = Symb.TickValue() / Symb.TickSize();
      if(State.Total() != HistoryBars * BarDescr || Symb.LotsMin() <= 0 ||
         !MathIsValidNumber(point_cost))
         ReturnFalse;
      const double loss = MathAbs(State[0]) * point_cost * Symb.LotsMin();
      observed -= loss * PreviousBalance / EtalonBalance;
     }
//--- A realized online return has no same-state Base branch and therefore must
//--- not update D2 utility.  Utility is restricted to paired terminal delta J.
   const double target = observed + (terminal ? 0.0 : DiscFactor * MathMin(q1_target, q2_target));
   if(!MathIsValidNumber(target) || !ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(target)))
      ReturnFalse;
//--- Restore only the ordinary previous state. Forecast scenarios are rerun,
//--- never cached or serialized.
   if(!CreateBuffers(1, GetPointer(State), GetPointer(TimeState), (CBufferFloat*)NULL) ||
      !D2SkillMarket.Clear() ||
      !D2SkillForwardForecastState(GetPointer(State)))
      ReturnFalse;
   CNeuronBaseOCL *actor_context = Actor.Layer(0);
   CNeuronBaseOCL *actor_layer = Actor.Layer(3);
   if(!actor_context || !actor_layer || !BuildCriticInput(actor_context.getOutput(), actor_layer.getOutput(), GetPointer(CriticInput)) ||
      !Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(D2SkillMarket), -1))
      ReturnFalse;
   if(UpdatePolicy > 0 && Transition > 0 && Transition % (ulong)UpdatePolicy == 0 &&
      !PolicyBackward(Actor, Q1, GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1))
      ReturnFalse;
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(D2SkillMarket), -1))
      ReturnFalse;
   PreviousMarginPenalty = 0;
//--- Per-transition guard reads only the three small trainable weight buffers.
   if(!D2SkillVerifyFrozenWeightsExact())
      ReturnFalse;
   if(UpdateTargets > 0 && Transition > 0 && Transition % (ulong)UpdateTargets == 0)
      if(!TargetActor.WeightsUpdate(GetPointer(Actor), Tau) ||
         !TargetQ1.WeightsUpdate(GetPointer(Q1), Tau) ||
         !TargetQ2.WeightsUpdate(GetPointer(Q2), Tau))
         ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function OnInit.                                                 |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   D2SkillOnlineInitialized = false;
   if(!D2SkillConfigureRuntime(InpD2SkillStage, InpD2SkillMode,
                               InpD2SkillExecutionMode, InpD2SkillUtilityAware,
                               InpD2SkillMinUtility, InpD2SkillUtilityScale))
     { Print("D2Skill online init: runtime configuration=FAIL"); return INIT_FAILED; }
   if(!D2SkillInitIndicators())
     { Print("D2Skill online init: indicators=FAIL"); return INIT_FAILED; }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
     { Print("D2Skill online init: trade filling=FAIL"); return INIT_FAILED; }
   if(!D2SkillLoadForecastInference())
     { Print("D2Skill online init: forecast=FAIL"); return INIT_FAILED; }
   if(!D2SkillLoadOrCreatePolicySet(Actor, Q1, Q2, false))
     { Print("D2Skill online init: policy checkpoint=FAIL"); return INIT_FAILED; }
   if(!LoadOnlineTargets())
     { Print("D2Skill online init: runtime targets=FAIL"); return INIT_FAILED; }
   if(!D2SkillVerifyFrozenForecastExact())
     { Print("D2Skill online init: frozen forecast=FAIL"); return INIT_FAILED; }
   if(!D2SkillConfigureD2UtilityMode(D2Skill_D2_UTILITY_PAIRED_HINDSIGHT))
      { Print("D2Skill online init: utility source=FAIL"); return INIT_FAILED; }
   if(!ActorForwardSnapshot.Init(Actor) || !BaselineActorForwardSnapshot.Init(Actor) ||
      !Actor.Clear() || !D2SkillMarket.Clear() || !ResetD2SkillEpisodeInfluence(Actor) ||
      !ActorForwardSnapshot.Capture() || !BaselineActorForwardSnapshot.Capture())
      { Print("D2Skill online init: forecast clear=FAIL"); return INIT_FAILED; }
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   D2SkillOnlineInitialized = true;
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
 void OnDeinit(const int reason)
  {
   if(D2SkillOnlineInitialized && OnlinePairActive && !FinishOnlinePair())
      PolicyCheckpointSafe = false;
   if(D2SkillOnlineInitialized && !ResetD2SkillEpisodeInfluence(Actor))
     {
      PolicyCheckpointSafe = false;
      PrintFormat("%s -> %d episode influence reset failed", __FUNCTION__, __LINE__);
     }
//--- Checkpoint only a fully completed transition. The manifest is written last.
   if(PolicyCheckpointPending && PolicyCheckpointSafe)
      if(!D2SkillSavePolicySet(Actor, Q1, Q2))
         PrintFormat("%s -> %d save failed", __FUNCTION__, __LINE__);
   if(D2SkillOnlineInitialized && D2SkillForecast != NULL && !D2SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   D2SkillForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Function OnTick.                                                 |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillRefreshLiveMarket(GetPointer(State), GetPointer(TimeState)))
     {
      PrintFormat("%s -> %d market", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
//+------------------------------------------------------------------+
//| clears/reruns explicit states, so sequence boundaries remain ... |
//+------------------------------------------------------------------+
   if(!D2SkillForwardForecastState(GetPointer(State)))
     {
      PrintFormat("%s -> %d forecast", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   double buy_value = 0, sell_value = 0;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillBuildLiveAccount(PreviousBalance, PreviousEquity, Rates[0].time, GetPointer(Account), buy_value, sell_value))
     {
      PrintFormat("%s -> %d account", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   if(!OnlinePairActive && !StartOnlinePair(GetPointer(Account)))
     {
      PrintFormat("%s -> %d paired trajectory init", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   if(!AdvanceOnlinePair())
     {
      PrintFormat("%s -> %d paired trajectory advance", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
//--- A rejected order is recoverable.  Loss of the configured minimum
//--- deposit is the only financial terminal state for online training.
   const bool terminal = (AccountInfoDouble(ACCOUNT_BALANCE) <= MinBalance);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(HavePrevious)
     {
      PolicyCheckpointSafe = false;
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!TrainPreviousTransition(terminal))
        {
         PrintFormat("%s -> %d train: terminal recurrent-state failure", __FUNCTION__, __LINE__);
         ClearPreviousTransition();
         ExpertRemove();
         return;
        }
      PolicyCheckpointSafe = true;
      PolicyCheckpointPending = true;
      Transition++;
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(CheckpointTransitions > 0 && Transition % (ulong)CheckpointTransitions == 0)
        {
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(!FinishOnlinePair() || !D2SkillSavePolicySet(Actor, Q1, Q2))
           {
            PrintFormat("%s -> %d checkpoint failed", __FUNCTION__, __LINE__);
            ClearPreviousTransition();
            ExpertRemove();
           return;
          }
         PolicyCheckpointPending = false;
         if(!StartOnlinePair(GetPointer(Account)))
           {
            PrintFormat("%s -> %d paired boundary init", __FUNCTION__, __LINE__);
            ClearPreviousTransition();
            ExpertRemove();
            return;
           }
        }
      //--- Training ended at previous state. A terminal transition has no next
      //--- action, hence it must not rerun the current policy state.
      if(!terminal && (!CreateBuffers(0, GetPointer(State), GetPointer(TimeState), (CBufferFloat*)NULL) ||
                       //+------------------------------------------------------------------+
                       //| Function D2SkillForwardForecastState.                              |
                       //+------------------------------------------------------------------+
                       !D2SkillForwardForecastState(GetPointer(State))))
        {
         PrintFormat("%s -> %d restore: terminal recurrent-state failure", __FUNCTION__, __LINE__);
         ClearPreviousTransition();
         ExpertRemove();
         return;
        }
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(terminal)
     {
      PrintFormat("D2Skill online terminal: balance=%.2f min_balance=%.2f", AccountInfoDouble(ACCOUNT_BALANCE), MinBalance);
      if(!FinishOnlinePair())
         PrintFormat("%s -> %d paired terminal utility", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   if(OnlinePairActive && !BuildOnlineBaselineAction())
     {
      PrintFormat("%s -> %d paired baseline action", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(D2SkillMarket), -1) ||
//+------------------------------------------------------------------+
//| Function ReadAction.                                             |
//+------------------------------------------------------------------+
      !ReadAction(Actor, GetPointer(Action)) || !D2SkillValidateAction(GetPointer(Action)))
     {
      PrintFormat("%s -> %d action", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   if(OnlinePairActive && (!SkillVirtualAction.AssignArray(GetPointer(Action)) ||
                           (SkillVirtualAction.GetIndex() >= 0 && !SkillVirtualAction.BufferWrite())))
     {
      PrintFormat("%s -> %d paired skill action", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   double margin_penalty = 0;
   bool market_closed = false;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillExecuteAction(GetPointer(Action), buy_value, sell_value, margin_penalty, market_closed))
     {
      PrintFormat("%s -> %d execute: terminal partial-trade risk", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(market_closed)
     {
      Print("D2Skill online execution deferred: market closed");
      //--- The preceding transition was already trained above.  Do not replay it
      //--- on the first tradable bar, but preserve Actor history for this state.
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   if(margin_penalty < 0)
      PrintFormat("D2Skill online insufficient margin: penalty=%.2f", margin_penalty);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!StorePrevious())
     {
      PrintFormat("%s -> %d metadata: terminal post-trade failure", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   PreviousMarginPenalty = margin_penalty;
   HavePrevious = true;
   if(OnlinePairActive)
      OnlineVirtualTransitionPending = true;
  }
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
