//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|           ORION live direct-transition twin Actor-Critic trainer |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define StudyOnline
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input int   UpdatePolicy  = ActorUpdate;
input int   UpdateTargets = TargetUpdate;
input float Tau           = tau;
input int   CheckpointTransitions = 256;
input double MinBalance = 50.0;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CORIONNet Actor, TargetActor, Q1, Q2, TargetQ1, TargetQ2;
CBufferFloat State, TimeState, Account, Action, CriticInput, TargetCriticInput, ScalarTarget;
CBufferFloat PreviousAccount;
double PreviousBalance = 0, PreviousEquity = 0;
double PreviousMarginPenalty = 0;
ulong Transition = 0;
bool HavePrevious = false;
bool PolicyCheckpointPending = false;
bool PolicyCheckpointSafe = true;
bool ORIONOnlineInitialized = false;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ClearPreviousTransition(void)
  {
   HavePrevious = false;
   PreviousAccount.Clear();
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   PreviousMarginPenalty = 0;
   return ORIONMarket.Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool StorePrevious(void)
  {
   return (PreviousAccount.AssignArray(GetPointer(Account)) &&
           (PreviousAccount.GetIndex() < 0 || PreviousAccount.BufferWrite()));
  }
//+------------------------------------------------------------------+
//| Target objects are runtime-only.  They start as exact copies of |
//| their online counterparts and are never serialized separately.  |
//+------------------------------------------------------------------+
bool LoadOnlineTargets(void)
  {
   if(!ORIONLoadPolicyNet(TargetActor, ORION_ACTOR_FILE) ||
      !ORIONLoadPolicyNet(TargetQ1, ORION_Q1_FILE) || !ORIONLoadPolicyNet(TargetQ2, ORION_Q2_FILE))
      ReturnFalse;
   TargetActor.SetOpenCL(ORIONMarket.GetOpenCL());
   TargetQ1.SetOpenCL(ORIONMarket.GetOpenCL());
   TargetQ2.SetOpenCL(ORIONMarket.GetOpenCL());
   if(!ORIONValidatePolicyShape(TargetActor, false) || !ORIONValidatePolicyShape(TargetQ1, true) ||
      !ORIONValidatePolicyShape(TargetQ2, true))
      ReturnFalse;
   TargetActor.TrainMode(false);
   TargetQ1.TrainMode(false);
   TargetQ2.TrainMode(false);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TrainPreviousTransition(const bool terminal)
  {
//--- The live Forecast is at s_next.  Online reward is one step, so TD
//--- bootstrap uses runtime-only target objects initialized from live files.
   double q1_target = 0, q2_target = 0;
   if(!terminal)
     {
      if(!TargetActor.feedForward(GetPointer(Account), 1, false, GetPointer(ORIONMarket), -1))
         ReturnFalse;
      CNeuronBaseOCL *target_context = TargetActor.Layer(0);
      CNeuronBaseOCL *target_actor_layer = TargetActor.Layer(3);
      if(!target_context || !target_actor_layer ||
         !BuildCriticInput(target_context.getOutput(), target_actor_layer.getOutput(), GetPointer(TargetCriticInput)) ||
         !TargetQ1.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(ORIONMarket), -1) ||
         !TargetQ2.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(ORIONMarket), -1))
         ReturnFalse;
      CNeuronBaseOCL *target_q1_layer = TargetQ1.Layer(3), *target_q2_layer = TargetQ2.Layer(3);
      if(!target_q1_layer || !target_q2_layer)
         ReturnFalse;
      //--- TD target is consumed by CPU scalar arithmetic.  getResults is the
      //--- library CPU boundary and also supports a host-only final critic layer.
      CBufferFloat *qt1 = NULL, *qt2 = NULL;
      TargetQ1.getResults(qt1);
      TargetQ2.getResults(qt2);
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
   double observed = (PreviousEquity - AccountInfoDouble(ACCOUNT_EQUITY) +
                      PreviousBalance - AccountInfoDouble(ACCOUNT_BALANCE)) * PreviousBalance / EtalonBalance +
                     PreviousMarginPenalty;
//--- Match offline ORION: missing a move is penalized at one executable
//--- minimum lot, never at a balance-derived virtual position.
   if(MathMax(double(Account[4]), double(Account[5])) < Symb.LotsMin())
     {
      const double point_cost = Symb.TickValue() / Symb.TickSize();
      if(State.Total() != HistoryBars * BarDescr || Symb.LotsMin() <= 0 ||
         !MathIsValidNumber(point_cost))
         ReturnFalse;
      const double loss = MathAbs(State[0]) * point_cost * Symb.LotsMin();
      observed -= loss * PreviousBalance / EtalonBalance;
     }
   const double target = observed + (terminal ? 0.0 : DiscFactor * MathMin(q1_target, q2_target));
   if(!MathIsValidNumber(target) || !ScalarTarget.BufferInit(1, 0) || !ScalarTarget.Update(0, float(target)))
      ReturnFalse;
//--- Restore only the ordinary previous state. Forecast scenarios are rerun,
//--- never cached or serialized.
   if(!CreateBuffers(1, GetPointer(State), GetPointer(TimeState), (CBufferFloat*)NULL) ||
      !ORIONMarket.Clear() ||
      !ORIONForwardForecastState(GetPointer(State)))
      ReturnFalse;
   CNeuronBaseOCL *actor_context = Actor.Layer(0);
   CNeuronBaseOCL *actor_layer = Actor.Layer(3);
   if(!actor_context || !actor_layer || !BuildCriticInput(actor_context.getOutput(), actor_layer.getOutput(), GetPointer(CriticInput)) ||
      !Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(ORIONMarket), -1))
      ReturnFalse;
   if(UpdatePolicy > 0 && Transition > 0 && Transition % (ulong)UpdatePolicy == 0 &&
      !PolicyBackward(Actor, Q1, GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1))
      ReturnFalse;
   if(!Q1.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1) ||
      !Q2.backProp(GetPointer(ScalarTarget), GetPointer(ORIONMarket), -1))
      ReturnFalse;
   PreviousMarginPenalty = 0;
//--- Per-transition guard reads only the three small trainable weight buffers.
   if(!ORIONVerifyFrozenWeightsExact())
      ReturnFalse;
   if(UpdateTargets > 0 && Transition > 0 && Transition % (ulong)UpdateTargets == 0)
      if(!TargetActor.WeightsUpdate(GetPointer(Actor), Tau) ||
         !TargetQ1.WeightsUpdate(GetPointer(Q1), Tau) ||
         !TargetQ2.WeightsUpdate(GetPointer(Q2), Tau))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   ORIONOnlineInitialized = false;
   if(!ORIONInitIndicators())
     { Print("ORION online init: indicators=FAIL"); return INIT_FAILED; }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
     { Print("ORION online init: trade filling=FAIL"); return INIT_FAILED; }
   if(!ORIONLoadForecastInference())
     { Print("ORION online init: forecast=FAIL"); return INIT_FAILED; }
   if(!ORIONLoadOrCreatePolicySet(Actor, Q1, Q2, false))
     { Print("ORION online init: policy checkpoint=FAIL"); return INIT_FAILED; }
   if(!LoadOnlineTargets())
     { Print("ORION online init: runtime targets=FAIL"); return INIT_FAILED; }
   if(!ORIONVerifyFrozenForecastExact())
     { Print("ORION online init: frozen forecast=FAIL"); return INIT_FAILED; }
   if(!ORIONMarket.Clear())
     { Print("ORION online init: forecast clear=FAIL"); return INIT_FAILED; }
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   ORIONOnlineInitialized = true;
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Checkpoint only a fully completed transition. The manifest is written last.
   if(PolicyCheckpointPending && PolicyCheckpointSafe)
      if(!ORIONSavePolicySet(Actor, Q1, Q2))
         PrintFormat("%s -> %d save failed", __FUNCTION__, __LINE__);
   if(ORIONOnlineInitialized && ORIONForecast != NULL && !ORIONVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   ORIONForecast = NULL;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
   if(!ORIONRefreshLiveMarket(GetPointer(State), GetPointer(TimeState)))
     {
      PrintFormat("%s -> %d market", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
//--- Sequential current inference when possible. Training restoration below
//--- clears/reruns explicit states, so sequence boundaries remain deliberate.
   if(!ORIONForwardForecastState(GetPointer(State)))
     {
      PrintFormat("%s -> %d forecast", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   double buy_value = 0, sell_value = 0;
   if(!ORIONBuildLiveAccount(PreviousBalance, PreviousEquity, Rates[0].time, GetPointer(Account), buy_value, sell_value))
     {
      PrintFormat("%s -> %d account", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
//--- A rejected order is recoverable.  Loss of the configured minimum
//--- deposit is the only financial terminal state for online training.
   const bool terminal = (AccountInfoDouble(ACCOUNT_BALANCE) <= MinBalance);
   if(HavePrevious)
     {
      PolicyCheckpointSafe = false;
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
      if(CheckpointTransitions > 0 && Transition % (ulong)CheckpointTransitions == 0)
        {
         if(!ORIONSavePolicySet(Actor, Q1, Q2))
           {
            PrintFormat("%s -> %d checkpoint failed", __FUNCTION__, __LINE__);
            ClearPreviousTransition();
            ExpertRemove();
            return;
           }
         PolicyCheckpointPending = false;
        }
      //--- Training ended at previous state. A terminal transition has no next
      //--- action, hence it must not rerun the current policy state.
      if(!terminal && (!CreateBuffers(0, GetPointer(State), GetPointer(TimeState), (CBufferFloat*)NULL) ||
                       !ORIONForwardForecastState(GetPointer(State))))
        {
         PrintFormat("%s -> %d restore: terminal recurrent-state failure", __FUNCTION__, __LINE__);
         ClearPreviousTransition();
         ExpertRemove();
         return;
        }
     }
   if(terminal)
     {
      PrintFormat("ORION online terminal: balance=%.2f min_balance=%.2f", AccountInfoDouble(ACCOUNT_BALANCE), MinBalance);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(ORIONMarket), -1) ||
      !ReadAction(Actor, GetPointer(Action)) || !ORIONValidateAction(GetPointer(Action)))
     {
      PrintFormat("%s -> %d action", __FUNCTION__, __LINE__);
      if(!ClearPreviousTransition())
         ExpertRemove();
      return;
     }
   double margin_penalty = 0;
   bool market_closed = false;
   if(!ORIONExecuteAction(GetPointer(Action), buy_value, sell_value, margin_penalty, market_closed))
     {
      PrintFormat("%s -> %d execute: terminal partial-trade risk", __FUNCTION__, __LINE__);
      ClearPreviousTransition();
      ExpertRemove();
      return;
     }
   if(market_closed)
     {
      Print("ORION online execution deferred: market closed");
      //--- The preceding transition was already trained above.  Do not replay it
      //--- on the first tradable bar, but preserve Actor history for this state.
      HavePrevious = false;
      PreviousAccount.Clear();
      PreviousMarginPenalty = 0;
      return;
     }
   if(margin_penalty < 0)
      PrintFormat("ORION online insufficient margin: penalty=%.2f", margin_penalty);
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
  }
//+------------------------------------------------------------------+
