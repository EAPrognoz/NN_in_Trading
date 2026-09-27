//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "Харнесс обучения политик ORION: формирование и калибровка пары актор-критик."
#property description "Проходит стадии жизненного цикла ORION и обучает актор, критиков Q1 и Q2 на исторических барах."
#property description "Только для тестов; ордера не выставляет."

#define Study
#include "Trajectory.mqh"
input datetime Start         = D'2024.01.01';
input datetime End           = D'2026.01.01';
input int      Iterations    = 1000000;
input int      EpisodeBars   = 2 * StackSize;
input double   MinBalance    = 50.0;
input int      UpdatePolicy  = 5;
CNet Actor;
CNet Q1;
CNet Q2;

CBufferFloat State;
CBufferFloat TimeState;
CBufferFloat Account;
CBufferFloat NextAccount;
CBufferFloat CurrentAction;
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
//+------------------------------------------------------------------+
//| Loads OrCreateORIONPolicies.                                     |
//+------------------------------------------------------------------+
bool LoadOrCreateORIONPolicies(void)
  {
   return ORIONLoadOrCreatePolicySet(Actor, Q1, Q2, true);
  }
//+------------------------------------------------------------------+
//| Saves ORIONPolicies.                                             |
//+------------------------------------------------------------------+
bool SaveORIONPolicies(void)
  {
//--- Forecast artifacts are immutable throughout Actor-Critic study. This
//--- bounded lifecycle check avoids hashing three large files every batch.
   return ORIONSavePolicySet(Actor, Q1, Q2);
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
   return PolicyBackward(Actor, Q1, GetPointer(ActorObjective), GetPointer(ORIONMarket), -1);
  }
//+------------------------------------------------------------------+
//| Implements TrainTransition.                                      |
//+------------------------------------------------------------------+
bool TrainTransition(const int position, const int iteration, bool &terminal)
  {
   terminal = false;
   const bool trace_td = (iteration < 8 || iteration % 10000 == 0);
   if(!ORIONForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
     { Print("TrainTransition stage=current_forecast"); ReturnFalse; }
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=actor_forward"); ReturnFalse; }
   if(!ReadAction(Actor, GetPointer(CurrentAction)))
     { Print("TrainTransition stage=read_action"); ReturnFalse; }
   const double balance = MathMax(0.0, double(Account[0]) * EtalonBalance);
   const double buy_lot = MathMax(0.0, double(CurrentAction[0] - CurrentAction[3]));
   const double sell_lot = MathMax(0.0, double(CurrentAction[3] - CurrentAction[0]));
   if(trace_td)
      PrintFormat("ORION action iteration=%d balance=%.2f buy_lot=%.8f buy_tp=%.8f buy_sl=%.8f sell_lot=%.8f sell_tp=%.8f sell_sl=%.8f",
                  iteration, balance, buy_lot, double(CurrentAction[1]), double(CurrentAction[2]),
                  sell_lot, double(CurrentAction[4]), double(CurrentAction[5]));
   double reward = 0;
   if(!AdvanceAccount(GetPointer(Account), GetPointer(CurrentAction), position, MinBalance,
                      //--- Gets Pointer.
                      GetPointer(NextAccount), reward, terminal))
     { Print("TrainTransition stage=advance_account"); ReturnFalse; }
   const double target_value = reward;
   if(trace_td)
      PrintFormat("ORION return iteration=%d reward=%.8f target=%.8f balance=%.2f equity=%.2f "
                  "floating_buy=%.8f floating_sell=%.8f next_buy=%.8f next_sell=%.8f terminal=%s",
                  iteration, reward, target_value, double(NextAccount[0])*EtalonBalance,
                  double(NextAccount[2])*balance, double(NextAccount[6])*MathMax(double(NextAccount[0])*EtalonBalance, 1.0),
                  double(NextAccount[7])*MathMax(double(NextAccount[0])*EtalonBalance, 1.0),
                  double(NextAccount[4]), double(NextAccount[5]), (terminal ? "true" : "false"));
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
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=q1_forward"); ReturnFalse; }
   if(!Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=q2_forward"); ReturnFalse; }
//--- Function if.
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
      PrintFormat("ORION Q iteration=%d q1=%.8f q2=%.8f td_q1=%.8f td_q2=%.8f",
                  iteration, double(q1_output[0]), double(q2_output[0]),
                  target_value - double(q1_output[0]), target_value - double(q2_output[0]));
     }
//--- A critic gradient has no valid direction on the discontinuous no-trade
//--- plateau.  Keep learning Q there, but let only the realized-future teacher
//--- move Actor back into the executable action manifold.
   const bool executable_action = (IsExecutableOrder(buy_lot, CurrentAction[1], CurrentAction[2]) ||
                                   IsExecutableOrder(sell_lot, CurrentAction[4], CurrentAction[5]));
//--- Function if.
   if(iteration > 0 && UpdatePolicy > 0 && iteration % UpdatePolicy == 0)
     {
//--- Function if.
      if(executable_action)
        {
         if(!PolicyBackward())
           { Print("TrainTransition stage=policy_backward"); ReturnFalse; }
         PolicyTransitions++;
        }
      else
         PolicySkippedInvalid++;
     }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=critic_backward"); ReturnFalse; }
//--- The teacher uses realized future bars, not one selected Forecast scenario.
//--- It is deliberately a second Critic-only sample: Q1/Q2 are stateless,
//--- whereas Actor has already performed its only forward for this state.
   double teacher_reward = 0;
   if(!BuildTeacherAction(position, GetPointer(Account), GetPointer(TeacherAction), teacher_reward))
     { Print("TrainTransition stage=teacher_action"); ReturnFalse; }
//--- Function if.
   if(teacher_reward > 0)
     {
      if(trace_td)
         PrintFormat("ORION teacher iteration=%d reward=%.8f buy_lot=%.8f sell_lot=%.8f",
                     iteration, teacher_reward,
                     MathMax(0.0, double(TeacherAction[0] - TeacherAction[3])),
                     MathMax(0.0, double(TeacherAction[3] - TeacherAction[0])));
      //--- Supervised Actor update reuses the current activation; it never advances
      //--- the history stack or invokes a second Actor forward.
      if(!Actor.backProp(GetPointer(TeacherAction), GetPointer(ORIONMarket), -1))
        { Print("TrainTransition stage=teacher_actor_backward"); ReturnFalse; }
      TeacherTransitions++;
      TeacherRewardSum += teacher_reward;
     }
//--- Critic receives every oracle action, including a losing one.  Only Actor
//--- is restricted to profitable oracle supervision above.
   if(!ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(teacher_reward)) ||
      !BuildCriticInput(context_layer.getOutput(), GetPointer(TeacherAction), GetPointer(CriticInput)))
     { Print("TrainTransition stage=teacher_critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=teacher_critic_forward"); ReturnFalse; }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=teacher_critic_backward"); ReturnFalse; }
   TeacherCriticTransitions++;
//--- A second critic-only sample is uniformly random over executable lots and
//--- stop distances.  It never runs Actor backward or advances policy history.
   double random_reward = 0;
   if(!BuildRandomAction(position, GetPointer(Account), GetPointer(RandomAction), random_reward))
     { Print("TrainTransition stage=random_action"); ReturnFalse; }
//--- Function if.
   if(random_reward > 0)
     {
      if(!Actor.backProp(GetPointer(RandomAction), GetPointer(ORIONMarket), -1))
        { Print("TrainTransition stage=random_actor_backward"); ReturnFalse; }
     }
   if(!ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(random_reward)) ||
      !BuildCriticInput(context_layer.getOutput(), GetPointer(RandomAction), GetPointer(CriticInput)))
     { Print("TrainTransition stage=random_critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=random_critic_forward"); ReturnFalse; }
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1))
     { Print("TrainTransition stage=random_critic_backward"); ReturnFalse; }
   RandomCriticTransitions++;
//--- Only the three declared Forecast trainable buffers are read here. The
//--- complete persistent Codebook is checked at checkpoint/lifecycle bounds.
   if(!ORIONVerifyFrozenWeightsExact())
     { Print("TrainTransition stage=frozen_forecast_check"); ReturnFalse; }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements TrainORIONActorCritic.                                |
//+------------------------------------------------------------------+
void TrainORIONActorCritic(void)
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
   TeacherTransitions = 0;
   TeacherCriticTransitions = 0;
   RandomCriticTransitions = 0;
   TeacherRewardSum = 0;
   PolicyTransitions = 0;
   PolicySkippedInvalid = 0;
   for(int iteration = 0; iteration < Iterations && !IsStopped(); iteration++)
     {
//--- Function if.
      if(episode == 0)
        {
         if(!ORIONForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
            break;
         const vector<float> sampled = SampleAccount(GetPointer(State), Rates[position].time, EtalonBalance, MinBalance);
         if(sampled.Size() != AccountDescr || !Account.AssignArray(sampled) ||
            (Account.GetIndex() >= 0 && !Account.BufferWrite()) || !Actor.Clear())
            break;
        }
      bool terminal = false;
//--- Function if.
      if(!TrainTransition(position, iteration, terminal))
        {
         failed_iteration = iteration;
         failed_position = position;
         break;
        }
      completed_iterations++;
      if((NextAccount.GetIndex() >= 0 && !NextAccount.BufferRead()) || !Account.AssignArray(GetPointer(NextAccount)) ||
         (Account.GetIndex() >= 0 && !Account.BufferWrite()))
        { failed_iteration = iteration; failed_position = position; break; }
      position--;
      episode++;
//--- Function if.
      if(position < first || terminal || episode >= MathMax(1, EpisodeBars))
        {
         if(position < first)
            position = last;
         episode = 0;
        }
//--- Function if.
      if(GetTickCount() - shown > 500)
        {
         const double q1_rmse = MathSqrt(MathMax(0.0, double(Q1.getRecentAverageError()))) / MathMax(EtalonBalance, 1.0);
         const double q2_rmse = MathSqrt(MathMax(0.0, double(Q2.getRecentAverageError()))) / MathMax(EtalonBalance, 1.0);
         const double teacher_mean = (TeacherTransitions > 0 ? TeacherRewardSum / double(TeacherTransitions) : 0.0);
         Comment(StringFormat("ORION AC %6.2f%% Q1 rRMSE %.8f Q2 rRMSE %.8f teacherA %I64u teacherQ %I64u randomQ %I64u R %.2f policy %I64u skip %I64u",
                              100.0 * iteration / MathMax(Iterations, 1), q1_rmse, q2_rmse,
                              TeacherTransitions, TeacherCriticTransitions, RandomCriticTransitions,
                              teacher_mean, PolicyTransitions, PolicySkippedInvalid));
         shown = GetTickCount();
        }
     }
   Comment("");
//--- Function if.
   if(completed_iterations != Iterations)
     {
      if(failed_iteration >= 0)
         PrintFormat("%s -> %d publication aborted: TrainTransition failed iteration=%d position=%d completed=%d/%d",
                     __FUNCTION__, __LINE__, failed_iteration, failed_position, completed_iterations, Iterations);
      else
         PrintFormat("%s -> %d publication aborted: stop requested completed=%d/%d",
                     __FUNCTION__, __LINE__, completed_iterations, Iterations);
      ExpertRemove();
      return;
     }
   if(!SaveORIONPolicies())
      PrintFormat("%s -> %d save failed", __FUNCTION__, __LINE__);
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Function OnInit.                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
   if(!ORIONInitIndicators() || !ORIONLoadForecastInference() || !LoadOrCreateORIONPolicies() ||
//--- Function ORIONVerifyFrozenForecastExact.
      !ORIONVerifyFrozenForecastExact())
     {
      PrintFormat("ORION Actor-Critic initialization failed at line %d error=%d", __LINE__, GetLastError());
      return INIT_FAILED;
     }
   return (EventChartCustom(ChartID(), 1, 0, 0, "Init") ? INIT_SUCCEEDED : INIT_FAILED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Only a completed explicit training loop persists all six artifacts and
//--- writes the hash manifest last. Deinit must not create a mixed generation.
   if(ORIONFrozenBaselineReady && ORIONForecast != NULL && !ORIONVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   ORIONForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Chart event callback.                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id == 1001)
      TrainORIONActorCritic();
  }
//+---+