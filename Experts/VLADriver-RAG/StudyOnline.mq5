//+------------------------------------------------------------------+
//| StudyOnline.mq5                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#define StudyOnline

#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input group                "---- Other ----"
input int                  Agent = 1;
input group                "---- Base Memory ----"
input bool                 LoadBaseMemorySnapshot = true;
input string               BaseMemorySnapshotPath = "VLADriverBaseMemory.snapshot";
input bool                 PublishMemoryOnOnlineMemorySize = true;
input bool                 PublishMemoryOnEpisodeClosure = true;
CNet                 cActor[2];
CNet                 cCritic[3];
CNet                 cStateEncoder;

datetime             dtStudied;
CBufferFloat         bState;
CBufferFloat         bContext;
CBufferFloat         bTime;
CBufferFloat         bGradient;
CBufferFloat         *bAction;
CBufferFloat         *Result;

double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 bFirstRun = true;
bool                 bFillStack = true;
// The library owns records, centroids, snapshots and GPU retrieval.
CNeuronRAGMemory     RAGMemory;
bool                 BaseMemoryNullBound = true;
// These are execution facts only; pending events live in CNeuronRAGMemory.
ulong                OnlinePendingId = 0;
bool                 OnlinePendingExecuted = false;
datetime             OnlinePendingStartTime = 0;
double               OnlinePendingStartBalance = 0;
double               OnlinePendingStartEquity = 0;
double               OnlinePendingPeakEquity = 0;
double               OnlinePendingMaxDrawdown = 0;
double               OnlinePendingMaxMarginToEquity = 0;
bool                 OnlinePendingHasEquitySample = false;
bool                 OnlinePendingHasRiskSample = false;
ulong                OnlinePendingExecutionDeal = 0;
ulong                OnlinePendingPositionId = 0;
ulong                OnlinePendingPositionIds[];
uint                 OnlinePendingStartPosition = 0;
CBufferFloat           OnlinePendingAction;
CBufferFloat           OnlinePendingScenarioEmbedding;
bool                 OnlinePendingPrepared = false;
const float          RejectedExecutionPenaltyScale = 100.0f;
enum ENUM_VLADriverModifyResult
  {
   VLADriverModifyNoOp=0,
   VLADriverModifyAccepted=1,
   VLADriverModifyRejected=2
  };
void ClearOnlinePendingScenario(void)
  {
   OnlinePendingId = 0;
   OnlinePendingExecuted = false;
   OnlinePendingStartTime = 0;
   OnlinePendingStartBalance = 0;
   OnlinePendingStartEquity = 0;
   OnlinePendingPeakEquity = 0;
   OnlinePendingMaxDrawdown = 0;
   OnlinePendingMaxMarginToEquity = 0;
   OnlinePendingHasEquitySample = false;
   OnlinePendingHasRiskSample = false;
   OnlinePendingExecutionDeal = 0;
   OnlinePendingPositionId = 0;
   ArrayResize(OnlinePendingPositionIds,0);
   OnlinePendingStartPosition = 0;
   OnlinePendingAction.BufferInit(NActions, 0);
   OnlinePendingScenarioEmbedding.BufferInit(EmbeddingSize, 0);
   OnlinePendingPrepared = false;
  }
//+------------------------------------------------------------------+
//| Capture the observed state before the new market order is sent.  |
//+------------------------------------------------------------------+
bool AnchorOnlinePendingScenario(void)
  {
   if(OnlinePendingId != 0 || OnlinePendingPrepared || bAction == NULL)
      ReturnFalse;
   CBufferFloat *scenario_embedding = NULL;
   if(!cStateEncoder.GetLayerOutputDevice(StateScenarioLayer, scenario_embedding))
     {
      PrintFormat("%s -> %d pending scenario output failed", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   if(scenario_embedding.Total() != EmbeddingSize)
     {
      PrintFormat("%s -> %d pending scenario embedding size mismatch", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   float scenario_embedding_data[];
   uint replaced_values = 0;
   const bool scenario_read = RAGMemory.ReadScenarioEmbedding(scenario_embedding,
                              scenario_embedding_data, replaced_values);
   if(replaced_values > 0)
      PrintFormat("Online RAG scenario anchor replaced %u non-finite values", replaced_values);
   if(!scenario_read)
     {
      PrintFormat("%s -> %d pending scenario anchor read failed", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   for(int i = 0; i < NActions; i++)
      if(!OnlinePendingAction.Update(i, bAction[i]))
         ReturnFalse;
   if(!OnlinePendingScenarioEmbedding.BufferInit(EmbeddingSize,0))
      ReturnFalse;
   if(!OnlinePendingScenarioEmbedding.AssignArray(scenario_embedding_data))
      ReturnFalse;
   OnlinePendingStartTime = TimeCurrent();
   OnlinePendingStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   OnlinePendingStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   OnlinePendingPeakEquity=OnlinePendingStartEquity;
   OnlinePendingPrepared = true;
   return true;
   }
//+------------------------------------------------------------------+
//| Create library pending state only after a market order is known. |
//+------------------------------------------------------------------+
bool OpenConfirmedOnlinePendingScenario(const ulong deal)
  {
   if(!OnlinePendingPrepared || OnlinePendingId!=0 || deal==0 ||
      !HistoryDealSelect(deal) ||
      ulong(HistoryDealGetInteger(deal,DEAL_POSITION_ID))==0)
     {
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   float normalized_action[NActions];
   if(!NormalizeMemoryAction(RAGMemory,GetPointer(OnlinePendingAction),0,
                               OnlinePendingStartEquity,normalized_action))
     {
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   float scenario_embedding_data[EmbeddingSize];
   for(int i=0;i<EmbeddingSize;i++)
      scenario_embedding_data[i]=OnlinePendingScenarioEmbedding[i];
   OnlinePendingId=RAGMemory.OpenPending(scenario_embedding_data,normalized_action);
   if(OnlinePendingId==0 || !RegisterOnlinePendingExecution(deal))
     {
      if(OnlinePendingId!=0)
         RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Apply one uniform Critic-only penalty for rejected execution.    |
//+------------------------------------------------------------------+
bool PenalizeRejectedExecution(const double requested_volume)
  {
   if(requested_volume<=0)
      return true;
   Result.Clear();
   if(!Result.Add(float(-RejectedExecutionPenaltyScale*requested_volume)) ||
      !cCritic[0].backProp(Result,GetPointer(cStateEncoder),StateTokenLayer))
     {
      PrintFormat("%s -> %d rejected execution Critic update failed",__FUNCTION__,__LINE__);
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Classify terminal acknowledgement without treating no-op as      |
//| data.                                                            |
//+------------------------------------------------------------------+
ENUM_VLADriverModifyResult ClassifyOnlineTradeResult(const bool request_sent)
  {
   const uint retcode=Trade.ResultRetcode();
   if(request_sent && (retcode==TRADE_RETCODE_DONE || retcode==TRADE_RETCODE_DONE_PARTIAL))
      return VLADriverModifyAccepted;
   if(retcode==TRADE_RETCODE_NO_CHANGES)
      return VLADriverModifyNoOp;
   return VLADriverModifyRejected;
  }
//+------------------------------------------------------------------+
//| Check whether one position belongs to the current trade plan.   |
//+------------------------------------------------------------------+
bool OnlinePendingHasPosition(const ulong position)
  {
   if(position==0)
      ReturnFalse;
   const int total=ArraySize(OnlinePendingPositionIds);
   for(int i=0;i<total;i++)
      if(OnlinePendingPositionIds[i]==position)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Add a unique position identity to the current trade plan.       |
//+------------------------------------------------------------------+
bool AddOnlinePendingPosition(const ulong position)
  {
   if(position==0)
      ReturnFalse;
   if(OnlinePendingHasPosition(position))
      return true;
   const int total=ArraySize(OnlinePendingPositionIds);
   if(ArrayResize(OnlinePendingPositionIds,total+1)!=total+1)
      ReturnFalse;
   OnlinePendingPositionIds[total]=position;
   return true;
  }
//+------------------------------------------------------------------+
//| Undo a just-reserved position when event confirmation failed.    |
//+------------------------------------------------------------------+
bool RemoveOnlinePendingPosition(const ulong position)
  {
   const int total=ArraySize(OnlinePendingPositionIds);
   for(int i=0;i<total;i++)
      if(OnlinePendingPositionIds[i]==position)
        {
         for(int j=i+1;j<total;j++)
            OnlinePendingPositionIds[j-1]=OnlinePendingPositionIds[j];
         if(ArrayResize(OnlinePendingPositionIds,total-1)!=total-1)
            ReturnFalse;
         return true;
        }
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| A plan remains open while any of its netting/hedging positions   |
//| remains open.                                                    |
//+------------------------------------------------------------------+
bool OnlinePendingPositionsAreOpen(void)
  {
   if(ArraySize(OnlinePendingPositionIds)<=0)
      return false;
   for(int i=0;i<PositionsTotal();i++)
      if(PositionGetSymbol(i)==Symb.Name() &&
         OnlinePendingHasPosition(ulong(PositionGetInteger(POSITION_IDENTIFIER))))
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Compatibility helper for partial-close classification.          |
//+------------------------------------------------------------------+
bool OnlinePendingPositionIsOpen(void)
  {
   return OnlinePendingPositionsAreOpen();
  }
//+------------------------------------------------------------------+
//| Execute TP/SL change and classify no-op, accepted and rejected.  |
//+------------------------------------------------------------------+
ENUM_VLADriverModifyResult ModifyOnlinePosition(const ENUM_POSITION_TYPE type,
                                                const double sl,const double tp)
  {
   const int total=PositionsTotal();
   const datetime earliest_update=TimeCurrent()-5*PeriodSeconds(TimeFrame);
   uint result=uint(VLADriverModifyNoOp);
   for(int i=0;i<total;i++)
     {
      if(PositionGetSymbol(i)!=Symb.Name() || PositionGetInteger(POSITION_TYPE)!=type ||
         PositionGetInteger(POSITION_TIME_UPDATE)>earliest_update)
         continue;
      bool modify=false;
      double position_sl=PositionGetDouble(POSITION_SL);
      double position_tp=PositionGetDouble(POSITION_TP);
      if(type==POSITION_TYPE_BUY)
        {
         if((sl-position_sl)>=Symb.Point())
           { position_sl=sl; modify=true; }
         if(MathAbs(tp-position_tp)>=Symb.Point())
           { position_tp=tp; modify=true; }
        }
      else
        {
         if((position_sl-sl)>=Symb.Point())
           { position_sl=sl; modify=true; }
         if(MathAbs(tp-position_tp)>=Symb.Point())
           { position_tp=tp; modify=true; }
        }
      if(!modify)
         continue;
      const ENUM_VLADriverModifyResult update=ClassifyOnlineTradeResult(
         Trade.PositionModify(PositionGetInteger(POSITION_TICKET),position_sl,position_tp));
      result|=uint(update);
     }
   return (ENUM_VLADriverModifyResult)result;
  }
//+------------------------------------------------------------------+
//| Full close is terminal and therefore never creates RAG           |
//| correction.                                                      |
//+------------------------------------------------------------------+
ENUM_VLADriverModifyResult CloseOnlineByDirection(const ENUM_POSITION_TYPE type)
  {
   return ClassifyOnlineTradeResult(CloseByDirection(type));
  }
//+------------------------------------------------------------------+
//| Partial close is an action correction only while episode         |
//| remains.                                                         |
//+------------------------------------------------------------------+
ENUM_VLADriverModifyResult CloseOnlinePartial(const ENUM_POSITION_TYPE type,
                                              const double volume)
  {
   const ENUM_VLADriverModifyResult result=ClassifyOnlineTradeResult(ClosePartial(type,volume));
   if(result!=VLADriverModifyAccepted || !OnlinePendingPositionIsOpen())
      return (result==VLADriverModifyAccepted ? VLADriverModifyNoOp : result);
   return VLADriverModifyAccepted;
  }
//+------------------------------------------------------------------+
//| Sample actual account exposure for a confirmed pending           |
//| execution.                                                       |
//+------------------------------------------------------------------+
void SampleOnlinePendingScenario(void)
  {
   if(OnlinePendingId==0 || !OnlinePendingExecuted)
      return;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(MathIsValidNumber(equity) && equity > 0)
     {
      if(!OnlinePendingHasEquitySample || equity>OnlinePendingPeakEquity)
         OnlinePendingPeakEquity=equity;
      OnlinePendingMaxDrawdown=MathMax(OnlinePendingMaxDrawdown,OnlinePendingPeakEquity-equity);
      OnlinePendingHasEquitySample=true;
      double margin = AccountInfoDouble(ACCOUNT_MARGIN);
      if(MathIsValidNumber(margin) && margin >= 0)
        {
         OnlinePendingMaxMarginToEquity=MathMax(OnlinePendingMaxMarginToEquity,margin/equity);
         OnlinePendingHasRiskSample=true;
        }
     }
  }
//+------------------------------------------------------------------+
//| Bind only a confirmed market deal to the causal pending anchor.  |
//+------------------------------------------------------------------+
bool RegisterOnlinePendingExecution(const ulong deal)
  {
   if(OnlinePendingId==0 || deal==0 || !HistoryDealSelect(deal))
     {
      if(OnlinePendingId!=0)
         RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   ulong position_id = ulong(HistoryDealGetInteger(deal, DEAL_POSITION_ID));
   if(position_id == 0)
     {
      RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   if(!RAGMemory.ConfirmPendingExecution(OnlinePendingId,deal,position_id))
     {
      RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   OnlinePendingExecutionDeal=deal;
   OnlinePendingPositionId=position_id;
   if(!AddOnlinePendingPosition(position_id))
     {
      RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
      ReturnFalse;
     }
   OnlinePendingExecuted=true;
   SampleOnlinePendingScenario();
   return true;
  }
//+------------------------------------------------------------------+
//| Add only an accepted position correction to the pending episode. |
//+------------------------------------------------------------------+
bool RegisterOnlinePendingCorrection(const ulong deal=0)
   {
    if(OnlinePendingId==0 || !OnlinePendingExecuted || OnlinePendingPositionId==0 || bAction==NULL)
       ReturnFalse;
    ulong position=OnlinePendingPositionId;
    if(deal>0)
      {
       if(!HistoryDealSelect(deal))
          ReturnFalse;
       position=ulong(HistoryDealGetInteger(deal,DEAL_POSITION_ID));
       if(position==0 || HistoryDealGetInteger(deal,DEAL_ENTRY)!=DEAL_ENTRY_IN)
          ReturnFalse;
      }
    float normalized_action[NActions];
   for(int i=0;i<NActions;i++)
      OnlinePendingAction.Update(i,bAction[i]);
   if(!NormalizeMemoryAction(RAGMemory,GetPointer(OnlinePendingAction),0,OnlinePendingStartEquity,normalized_action) ||
       !RAGMemory.CorrectPending(OnlinePendingId,normalized_action))
      ReturnFalse;
   if(deal>0)
     {
      const bool already_added=OnlinePendingHasPosition(position);
      if(!already_added && !AddOnlinePendingPosition(position))
        {
         RAGMemory.CancelPendingCorrection(OnlinePendingId);
         ReturnFalse;
        }
      if(!RAGMemory.ConfirmPendingCorrection(OnlinePendingId,deal,position))
        {
         if(!already_added)
            RemoveOnlinePendingPosition(position);
         RAGMemory.CancelPendingCorrection(OnlinePendingId);
         ReturnFalse;
        }
      return true;
     }
   if(RAGMemory.ConfirmPendingCorrection(OnlinePendingId,OnlinePendingPositionId))
      return true;
   RAGMemory.CancelPendingCorrection(OnlinePendingId);
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Persist one validated snapshot/manifest publication pair.        |
//+------------------------------------------------------------------+
bool SaveOnlineBaseMemory(void)
  {
   if(RAGMemory.ScenarioCount()==0)
      return true;
   return RAGMemory.SavePublishedSnapshot(BaseMemorySnapshotPath);
   }
//+------------------------------------------------------------------+
//| Rebind both Actor copies before any later forward pass.          |
//+------------------------------------------------------------------+
bool RebindOnlineRAGMemory(void)
  {
   const bool actor0_bound=cActor[0].BindRAGMemory(2,GetPointer(RAGMemory));
   const bool actor1_bound=cActor[1].BindRAGMemory(2,GetPointer(RAGMemory));
   if(actor0_bound && actor1_bound)
      return true;
   PrintFormat("%s -> %d online RAG Actor rebind failed",__FUNCTION__,__LINE__);
   DetachOnlineRAGMemory();
   ExpertRemove();
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Detach both consumers before COW deletes the old device buffers. |
//+------------------------------------------------------------------+
bool DetachOnlineRAGMemory(void)
  {
   const bool actor0_detached=cActor[0].UnbindRAGMemory(2);
   const bool actor1_detached=cActor[1].UnbindRAGMemory(2);
   if(actor0_detached && actor1_detached)
      return true;
   PrintFormat("%s -> %d online RAG Actor detach failed",__FUNCTION__,__LINE__);
   ExpertRemove();
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Build and swap only an immutable Base Memory snapshot.           |
//+------------------------------------------------------------------+
bool PublishOnlineMemory(const bool episode_completed)
  {
   const uint pending=RAGMemory.PendingRecordCount();
   if(pending==0 ||
      (!(PublishMemoryOnOnlineMemorySize && pending>=OnlineMemorySize) &&
       !(PublishMemoryOnEpisodeClosure && episode_completed)))
      return true;
   if(!DetachOnlineRAGMemory())
      ReturnFalse;
   if(!RAGMemory.Publish(episode_completed))
     {
      if(!RebindOnlineRAGMemory())
         ReturnFalse;
      ReturnFalse;
     }
   if(!RebindOnlineRAGMemory())
      ReturnFalse;
   BaseMemoryNullBound=false;
   return SaveOnlineBaseMemory();
  }
//+------------------------------------------------------------------+
//| Store the actual, fully closed execution as one completed case.  |
//+------------------------------------------------------------------+
void CollectOnlineCompletedScenario(void)
  {
   if(OnlinePendingId==0 || !OnlinePendingExecuted)
      return;
   if(OnlinePendingPositionId==0 && OnlinePendingExecutionDeal>0)
      OnlinePendingPositionId=ulong(HistoryDealGetInteger(OnlinePendingExecutionDeal,DEAL_POSITION_ID));
   if(OnlinePendingPositionId==0)
      return;
   if(ArraySize(OnlinePendingPositionIds)==0)
      if(!AddOnlinePendingPosition(OnlinePendingPositionId))
         return;
   if(OnlinePendingPositionsAreOpen())
      return;
   SRAGTerminalOutcome evaluated_terminal;
   // The simulator is a base/fallback only; live account/deal data below wins.
   EvaluateTerminalOutcome(RAGMemory,GetPointer(OnlinePendingAction),OnlinePendingStartBalance,
                           OnlinePendingStartPosition,evaluated_terminal);
   double reward = 0;
   double costs = 0;
   bool has_deals = false;
   if(HistorySelect(OnlinePendingStartTime, TimeCurrent()))
      for(int i = 0; i < HistoryDealsTotal(); i++)
        {
         ulong deal = HistoryDealGetTicket(i);
         if(deal==0 || !OnlinePendingHasPosition(ulong(HistoryDealGetInteger(deal,DEAL_POSITION_ID))))
            continue;
         double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
         double swap = HistoryDealGetDouble(deal, DEAL_SWAP);
         double commission = HistoryDealGetDouble(deal, DEAL_COMMISSION);
         reward += profit + swap + commission;
         costs += -MathMin(swap, 0.0) - MathMin(commission, 0.0);
         has_deals = true;
        }
   bool has_actual_reward = has_deals;
   if(!has_actual_reward && OnlinePendingStartBalance>0)
     {
      double end_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      if(MathIsValidNumber(end_balance))
        {
         reward=end_balance-OnlinePendingStartBalance;
         has_actual_reward = true;
        }
     }
   bool partial_metrics=(!has_deals || !OnlinePendingHasEquitySample ||
                         !OnlinePendingHasRiskSample || OnlinePendingStartTime<=0);
   const int period_seconds = PeriodSeconds(TimeFrame);
   SRAGTerminalOutcome terminal=evaluated_terminal;
   if(has_actual_reward && period_seconds > 0)
     {
      const double risk_budget=OnlinePendingStartEquity*MemoryRiskBudgetFraction;
      RAGMemory.NormalizeTerminalOutcome(reward,OnlinePendingMaxDrawdown,costs,
                                         OnlinePendingMaxMarginToEquity*risk_budget,
                                         double(TimeCurrent()-OnlinePendingStartTime),
                                         OnlinePendingStartEquity,risk_budget,
                                         period_seconds*MathMax(NForecast,1),terminal);
     }
   else
      partial_metrics = true;
   if(partial_metrics)
      Print("Online RAG record has partial terminal metrics");
   if(!RAGMemory.ClosePending(OnlinePendingId,terminal))
     {
      PrintFormat("%s -> %d online RAG close failed", __FUNCTION__, __LINE__);
      return;
     }
   ClearOnlinePendingScenario();
   if(!PublishOnlineMemory(true))
      PrintFormat("%s -> %d online memory publication failed", __FUNCTION__, __LINE__);
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!MQLInfoInteger(MQL_TESTER))
      return INIT_FAILED;
   BaseMemoryNullBound = true;
   ClearOnlinePendingScenario();
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
   if(!RSI.BufferResize(StackSize + HistoryBars) || !CCI.BufferResize(StackSize + HistoryBars) ||
      !ATR.BufferResize(StackSize + HistoryBars) || !MACD.BufferResize(StackSize + HistoryBars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
// Load Models.
   float temp;
   CArrayObj *actor = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(actor, critic))
     {
      delete actor;
      delete critic;
      return INIT_FAILED;
     }
   if(!cStateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true))
     {
      PrintFormat("Error of load StateEncoder: %d", GetLastError());
      delete actor;
      delete critic;
      return INIT_FAILED;
     }
   if(!cActor[0].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !cActor[1].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor[0].Create(actor) ||
         !cActor[1].Create(actor))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      cActor[1].SetOpenCL(cActor[0].GetOpenCL());
      cActor[1].WeightsUpdate(GetPointer(cActor[0]), 1);
     }
   bool result = true;
   for(uint i = 0; (i < cCritic.Size() && result); i++)
      if(!cCritic[i].Load(FileName + CriticCheckpointFile, temp, temp, temp, dtStudied, true))
         result = false;
   if(!result)
     {
      Print("Create new Critic model");
      for(uint i = 0; i < cCritic.Size(); i++)
         if(!cCritic[i].Create(critic))
           {
            DeleteObj(actor);
            DeleteObj(critic);
            return INIT_FAILED;
           }
      for(uint i = 1; i < cCritic.Size(); i++)
        {
         cCritic[i].SetOpenCL(cCritic[0].GetOpenCL());
         if(!cCritic[i].WeightsUpdate(GetPointer(cCritic[0]), 1))
           {
            DeleteObj(actor);
            DeleteObj(critic);
            return INIT_FAILED;
           }
        }
     }
   DeleteObj(actor);
   DeleteObj(critic);
   for(int i = 0; i < 2; i++)
     {
      cActor[i].TrainMode(i == 0);
      cCritic[i].TrainMode(i == 0);
      cCritic[i].SetOpenCL(cActor[0].GetOpenCL());
      cActor[i].Clear();
      cCritic[i].Clear();
     }
   cActor[1].SetOpenCL(cActor[0].GetOpenCL());
   for(uint i = 2; i < cCritic.Size(); i++)
     {
      cCritic[i].TrainMode(false);
      cCritic[i].SetOpenCL(cActor[0].GetOpenCL());
      cCritic[i].Clear();
     }
   cStateEncoder.TrainMode(false);
   cStateEncoder.SetOpenCL(cActor[0].GetOpenCL());
   cStateEncoder.Clear();
   CLayerDescription memory_description;
   if(!CreateRAGMemoryDescription(memory_description) ||
      !RAGMemory.Init(0,0,cActor[0].GetOpenCL(),memory_description) ||
      !RAGMemory.SetOnlineMemorySize(OnlineMemorySize))
       return INIT_FAILED;
   if(LoadBaseMemorySnapshot && RAGMemory.LoadPublishedSnapshot(BaseMemorySnapshotPath))
      BaseMemoryNullBound=false;
   else
      if(!RAGMemory.BindNullInference())
         return INIT_FAILED;
   if(!RebindOnlineRAGMemory())
      return INIT_FAILED;
    if(!bGradient.BufferInit(EmbeddingSize, 0.0f) ||
       !bGradient.BufferCreate(cActor[0].GetOpenCL()))
       return INIT_FAILED;
   cActor[0].getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   cActor[0].GetLayerOutput(0, Result);
   if(Result.Total() != AccountDescr)
     {
      PrintFormat("Input size of Actor doesn't match context description (%d <> %d)", Result.Total(), AccountDescr);
      return INIT_FAILED;
     }
   cStateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of StateEncoder doesn't match market state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
   cStateEncoder.GetLayerOutput(StateTokenLayer, Result);
   if(Result.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("StateEncoder RankTCM layer doesn't match Critic context (%d <> %d)", Result.Total(), (BarDescr * EmbeddingSize));
      return INIT_FAILED;
     }
   cStateEncoder.GetLayerOutput(StateScenarioLayer, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("StateEncoder token layer doesn't match pooled scenario embedding (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   bFirstRun = true;
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   CollectOnlineCompletedScenario();
   if(OnlinePendingId!=0 && !OnlinePendingExecuted)
      RAGMemory.CancelPending(OnlinePendingId);
   if(reason!=REASON_INITFAILED && !SaveOnlineBaseMemory())
      Print("Error of save Base Memory snapshot");
   if(!(reason == REASON_RECOMPILE || reason == REASON_INITFAILED))
     {
      cActor[0].Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      cCritic[0].Save(FileName + CriticCheckpointFile, 0, 0, 0, TimeCurrent(), true);
     }
   DeleteObj(Result);
   DeleteObj(bAction);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(bFillStack)
     {
      if(!FillStack())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      bFillStack = false;
     }
   // Capture the live equity/margin observation on every tick while pending.
   SampleOnlinePendingScenario();
   if(!IsNewBar())
      return;
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), HistoryBars, Rates);
   if(bars < 0)
     {
      PrintFormat("%s -> %d CopyRates failed (%d)", __FUNCTION__, __LINE__, GetLastError());
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
   bTime.Clear();
   bTime.Reserve(HistoryBars);
   if(!CreateBuffers(0, GetPointer(bState), GetPointer(bTime), (CBufferFloat*)NULL))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   bool account_has_position = false;
   double position_discount = 0;
   double multiplier = 1.0 / (60.0 * 60.0 * 10.0);
   int total = PositionsTotal();
   datetime current = TimeCurrent();
   for(int i = 0; i < total; i++)
     {
      string position_symbol=PositionGetSymbol(i);
      if(position_symbol==NULL)
         continue;
      account_has_position=true;
      if(position_symbol != Symb.Name())
         continue;
      double profit = PositionGetDouble(POSITION_PROFIT);
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += profit;
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += profit;
            break;
        }
      position_discount += (current - PositionGetInteger(POSITION_TIME)) * multiplier * MathAbs(profit);
     }
// A bound position disappears only after its complete episode closes.
   CollectOnlineCompletedScenario();
   vector<float> account = vector<float>::Zeros(AccountDescr);
   account[0] = float(AccountInfoDouble(ACCOUNT_BALANCE) / EtalonBalance);
   account[1] = float((AccountInfoDouble(ACCOUNT_BALANCE) - PrevBalance) / MathMax(PrevBalance, 1.0));
   account[2] = float(AccountInfoDouble(ACCOUNT_EQUITY) / MathMax(PrevBalance, 1.0));
   account[3] = float((AccountInfoDouble(ACCOUNT_EQUITY) - PrevEquity) / MathMax(PrevEquity, 1.0));
   account[4] = (float)buy_value;
   account[5] = (float)sell_value;
   account[6] = float(buy_profit / MathMax(PrevBalance, 1.0));
   account[7] = float(sell_profit / MathMax(PrevBalance, 1.0));
   account[8] = float(position_discount / MathMax(PrevBalance, 1.0));
   double time = (double)Rates[0].time;
   double x = time / (double)(D'2024.01.01' - D'2023.01.01');
   account[9] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_MN1);
   account[10] = (float)MathCos(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_W1);
   account[11] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_D1);
   account[12] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
   if(!bContext.AssignArray(account))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   if(!cStateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   if(!RAGMemory.Retrieve(GetPointer(cStateEncoder),StateScenarioLayer))
     {
      PrintFormat("%s -> %d RAG retrieval failed", __FUNCTION__, __LINE__);
      return;
     }
   if(!bFirstRun)
     {
      // Target Nets.
      if(!cActor[1].feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                                GetPointer(cStateEncoder),StateScenarioLayer)
         || !cCritic[1].feedForward(GetPointer(cActor[1]), -1, GetPointer(cStateEncoder), StateTokenLayer)
         || !cCritic[2].feedForward(GetPointer(cActor[1]), -1, GetPointer(cStateEncoder), StateTokenLayer)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         bFirstRun = true;
         return;
        }
      // Critic.
      cCritic[1].getResults(Result);
      float forward = Result[0];
      cCritic[2].getResults(Result);
      forward = MathMin(forward, Result[0]);
      float reward = float(forward * DiscFactor +
                           (PrevEquity - AccountInfoDouble(ACCOUNT_EQUITY) +
                            PrevBalance - AccountInfoDouble(ACCOUNT_BALANCE)) * PrevBalance / EtalonBalance);
      if(MathMax(buy_value, sell_value) < Symb.LotsMin())
        {
         double marg = 0;
         if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
            marg = 200;
         double point_cost = Symb.TickValue() / Symb.TickSize();
         double loss = MathAbs(bState[0]) *
                       point_cost * PrevBalance / (10 * marg);
         reward -= float(loss * PrevBalance / EtalonBalance);
        }
      Result.Clear();
      if(!Result.Add(reward)
         || !cCritic[0].backProp(Result, GetPointer(cStateEncoder), StateTokenLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      if((PrevEquity - AccountInfoDouble(ACCOUNT_EQUITY) + PrevBalance - AccountInfoDouble(ACCOUNT_BALANCE)) <= 0 &&
         MathAbs(bState[0]) > Symb.StopsLevel()*Symb.Point())
        {
         double point_cost = Symb.TickValue() / Symb.TickSize();
         if(point_cost > 0)
           {
            vector<float> oracul = vector<float>::Zeros(NActions);
            double marg = 0;
            if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
               marg = 200;
            double lot = MathMax(PrevBalance / (10 * marg), Symb.LotsMin());
            if(bState[0] > 0)
              {
               lot = MathMin(lot, PrevBalance / ((-100 * bState[2 * HistoryBars] + Symb.Spread() * Symb.Point()) * point_cost));
               oracul[0] = (float)MathMin(lot, 1);
               oracul[1] = (float)MathMin(3 * bState[6] / (Symb.Point() * MaxTP), 1);
               oracul[2] = (float)MathMin(bState[6] / (Symb.Point() * MaxSL), 1);
              }
            else
              {
               lot = MathMin(lot, PrevBalance / ((100 * bState[HistoryBars] + Symb.Spread() * Symb.Point()) * point_cost));
               oracul[3] = (float)MathMin(lot, 1);
               oracul[4] = (float)MathMin(3 * bState[6] / (Symb.Point() * MaxTP), 1);
               oracul[5] = (float)MathMin(bState[6] / (Symb.Point() * MaxSL), 1);
              }
            reward = float(MathAbs(bState[0]) * point_cost * lot);
            reward = float(forward * DiscFactor + reward * PrevBalance / EtalonBalance);
            if(!Result.AssignArray(oracul) ||
               !cCritic[0].feedForward(Result, 1, false, (CNet*)GetPointer(cStateEncoder), StateTokenLayer))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               bFirstRun = true;
               return;
              }
            if(!Result.Update(0, reward)
               || !cCritic[0].backProp(Result, GetPointer(cStateEncoder), StateTokenLayer))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               bFirstRun = true;
               return;
              }
           }
        }
      if((MathRand() % ActorUpdate) == 0)
        {
         cCritic[0].TrainMode(false);
         if(!cCritic[0].feedForward(GetPointer(cActor[0]), -1, GetPointer(cStateEncoder), StateTokenLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
         cCritic[0].getResults(Result);
         CBufferFloat *actor_market_gradient=NULL;
         if(!cStateEncoder.GetLayerOutputDevice(StateScenarioLayer,actor_market_gradient))
           {
            PrintFormat("%s -> %d actor market gradient input failed", __FUNCTION__, __LINE__);
            return;
           }
         if(!Result.Update(0, float(MathMax(Result[0], 0) + (PrevBalance * 0.01 + PrevBalance) / EtalonBalance)) ||
            !cCritic[0].backProp(Result, GetPointer(cStateEncoder), StateTokenLayer) ||
            !cActor[0].backPropGradient(actor_market_gradient, GetPointer(bGradient), -1, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
         cCritic[0].TrainMode(true);
        }
      if(PrevBalance < 20)
         ExpertRemove();
     }
// New State.
   if(!cActor[0].feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                              GetPointer(cStateEncoder),StateScenarioLayer)
      || !cCritic[0].feedForward((CNet*)GetPointer(cActor[0]), -1, (CNet*)GetPointer(cStateEncoder), StateTokenLayer)
     )
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   cActor[0].getResults(bAction);
   if(bAction.Total() < NActions)
      bAction.BufferInit(NActions, 0);
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
   bool online_execution_attempted = false;
   if(bAction[0] >= bAction[3])
     {
      bAction.Update(0, (bAction[0] - bAction[3]));
      bAction.Update(3, 0);
     }
   else
     {
      bAction.Update(3, (bAction[3] - bAction[0]));
      bAction.Update(0, 0);
     }
// Buy Control.
   if(bAction[0] < min_lot ||
      (bAction[1] * MaxTP * Symb.Point()) <= 2 * stops ||
      (bAction[2] * MaxSL * Symb.Point()) <= stops
     )
      {
       if(buy_value > 0)
         {
          const ENUM_VLADriverModifyResult buy_close=CloseOnlineByDirection(POSITION_TYPE_BUY);
          if((buy_close&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(buy_value))
             return;
         }
     }
   else
     {
       double buy_lot = min_lot + MathRound((double)(bAction[0] - min_lot) / step_lot) * step_lot;
       double buy_tp = NormalizeDouble(Symb.Ask() + bAction[1] * MaxTP * Symb.Point(), Symb.Digits());
       double buy_sl = NormalizeDouble(Symb.Ask() - bAction[2] * MaxSL * Symb.Point(), Symb.Digits());
        bool buy_correction=false;
        ulong buy_correction_deal=0;
       ENUM_VLADriverModifyResult buy_modify=VLADriverModifyNoOp;
       if(buy_value>0)
          buy_modify=ModifyOnlinePosition(POSITION_TYPE_BUY,buy_sl,buy_tp);
       if((buy_modify&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(buy_lot))
          return;
       if((buy_modify&VLADriverModifyAccepted)!=0)
          buy_correction=true;
      if(buy_value != buy_lot)
        {
         if((buy_value - buy_lot) >= min_lot)
           {
            const ENUM_VLADriverModifyResult buy_partial=CloseOnlinePartial(POSITION_TYPE_BUY,buy_value-buy_lot);
            if((buy_partial&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(buy_value-buy_lot))
               return;
            if((buy_partial&VLADriverModifyAccepted)!=0)
               buy_correction=true;
           }
          else
             if((buy_lot-buy_value)>=min_lot)
               {
                // A same-direction volume increase is a correction of the current episode.
                if(buy_value>0 && sell_value<=0)
                  {
                   const ENUM_VLADriverModifyResult buy_increase=ClassifyOnlineTradeResult(
                      Trade.Buy(buy_lot-buy_value,Symb.Name(),Symb.Ask(),buy_sl,buy_tp));
                   if((buy_increase&VLADriverModifyRejected)!=0 &&
                      !PenalizeRejectedExecution(buy_lot-buy_value))
                      return;
                   if((buy_increase&VLADriverModifyAccepted)!=0)
                     {
                      buy_correction_deal=Trade.ResultDeal();
                      if(buy_correction_deal>0)
                         buy_correction=true;
                      else
                         Print("Accepted BUY volume correction has no deal and is not added to RAG memory");
                     }
                  }
                else
                   if(buy_value<=0 && sell_value<=0 && !account_has_position && OnlinePendingId==0 &&
                      !online_execution_attempted && AnchorOnlinePendingScenario())
                     {
                      online_execution_attempted = true;
                      if(Trade.Buy(buy_lot-buy_value,Symb.Name(),Symb.Ask(),buy_sl,buy_tp))
                        {
                         if(!OpenConfirmedOnlinePendingScenario(Trade.ResultDeal()))
                           {
                            if(!PenalizeRejectedExecution(buy_lot-buy_value))
                               return;
                            ExpertRemove();
                            return;
                           }
                        }
                      else
                        {
                         ClearOnlinePendingScenario();
                         if(!PenalizeRejectedExecution(buy_lot-buy_value))
                            return;
                        }
                     }
               }
        if(buy_correction && OnlinePendingId!=0 &&
           !RegisterOnlinePendingCorrection(buy_correction_deal))
         {
          PrintFormat("%s -> %d accepted BUY RAG correction registration failed",__FUNCTION__,__LINE__);
          ExpertRemove();
          return;
         }
      }
     }
// Sell Control.
   if(bAction[3] < min_lot ||
      (bAction[4] * MaxTP * Symb.Point()) <= 2 * stops ||
      (bAction[5] * MaxSL * Symb.Point()) <= stops
     )
      {
       if(sell_value > 0)
         {
          const ENUM_VLADriverModifyResult sell_close=CloseOnlineByDirection(POSITION_TYPE_SELL);
          if((sell_close&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(sell_value))
             return;
         }
     }
   else
     {
       double sell_lot = min_lot + MathRound((double)(bAction[3] - min_lot) / step_lot) * step_lot;
       double sell_tp = NormalizeDouble(Symb.Bid() - bAction[4] * MaxTP * Symb.Point(), Symb.Digits());
       double sell_sl = NormalizeDouble(Symb.Bid() + bAction[5] * MaxSL * Symb.Point(), Symb.Digits());
        bool sell_correction=false;
        ulong sell_correction_deal=0;
       ENUM_VLADriverModifyResult sell_modify=VLADriverModifyNoOp;
       if(sell_value>0)
          sell_modify=ModifyOnlinePosition(POSITION_TYPE_SELL,sell_sl,sell_tp);
       if((sell_modify&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(sell_lot))
          return;
       if((sell_modify&VLADriverModifyAccepted)!=0)
          sell_correction=true;
      if(sell_value != sell_lot)
        {
         if((sell_value - sell_lot) >= min_lot)
           {
            const ENUM_VLADriverModifyResult sell_partial=CloseOnlinePartial(POSITION_TYPE_SELL,sell_value-sell_lot);
            if((sell_partial&VLADriverModifyRejected)!=0 && !PenalizeRejectedExecution(sell_value-sell_lot))
               return;
            if((sell_partial&VLADriverModifyAccepted)!=0)
               sell_correction=true;
           }
          else
             if((sell_lot-sell_value)>=min_lot)
               {
                // A same-direction volume increase is a correction of the current episode.
                if(sell_value>0 && buy_value<=0)
                  {
                   const ENUM_VLADriverModifyResult sell_increase=ClassifyOnlineTradeResult(
                      Trade.Sell(sell_lot-sell_value,Symb.Name(),Symb.Bid(),sell_sl,sell_tp));
                   if((sell_increase&VLADriverModifyRejected)!=0 &&
                      !PenalizeRejectedExecution(sell_lot-sell_value))
                      return;
                   if((sell_increase&VLADriverModifyAccepted)!=0)
                     {
                      sell_correction_deal=Trade.ResultDeal();
                      if(sell_correction_deal>0)
                         sell_correction=true;
                      else
                         Print("Accepted SELL volume correction has no deal and is not added to RAG memory");
                     }
                  }
                else
                   if(buy_value<=0 && sell_value<=0 && !account_has_position && OnlinePendingId==0 &&
                      !online_execution_attempted && AnchorOnlinePendingScenario())
                     {
                      online_execution_attempted = true;
                      if(Trade.Sell(sell_lot-sell_value,Symb.Name(),Symb.Bid(),sell_sl,sell_tp))
                        {
                         if(!OpenConfirmedOnlinePendingScenario(Trade.ResultDeal()))
                           {
                            if(!PenalizeRejectedExecution(sell_lot-sell_value))
                               return;
                            ExpertRemove();
                            return;
                           }
                        }
                      else
                        {
                         ClearOnlinePendingScenario();
                         if(!PenalizeRejectedExecution(sell_lot-sell_value))
                            return;
                        }
                     }
               }
        if(sell_correction && OnlinePendingId!=0 &&
           !RegisterOnlinePendingCorrection(sell_correction_deal))
         {
          PrintFormat("%s -> %d accepted SELL RAG correction registration failed",__FUNCTION__,__LINE__);
          ExpertRemove();
          return;
         }
      }
     }
   if((OnlinePendingId!=0 && !OnlinePendingExecuted) ||
      (OnlinePendingPrepared && OnlinePendingId==0))
     {
      if(OnlinePendingId!=0)
         RAGMemory.CancelPending(OnlinePendingId);
      ClearOnlinePendingScenario();
     }
   bFirstRun = false;
   if((int(Rates[0].time / PeriodSeconds(TimeFrame)) % TargetUpdate) == 0)
     {
      if(MathRand() / 32767.0 > 0.5)
         cCritic[1].WeightsUpdate(GetPointer(cCritic[0]), tau);
      else
         cCritic[2].WeightsUpdate(GetPointer(cCritic[0]), tau);
      cActor[1].WeightsUpdate(GetPointer(cActor[0]), tau);
     }
  }
bool FillStack(void)
  {
   int start = StackSize + HistoryBars;
   int end = 0;
   int bars = CopyRates(Symb.Name(), TimeFrame, 2, start, Rates);
   if(bars < 0)
      ReturnFalse;
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
      ReturnFalse;
   if(RSI.BarsCalculated() < bars ||
      CCI.BarsCalculated() < bars ||
      ATR.BarsCalculated() < bars ||
      MACD.BarsCalculated() < bars)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
      ReturnFalse;
   bars -= end + HistoryBars;
   if(bars < 0)
      ReturnFalse;
   vector<float> result, target, neg_target;
   uint ticks = GetTickCount();
   uint tester_print = 0;
   for(uint i = 0; i < cActor.Size(); i++)
     {
      if(!cActor[i].Clear())
         ReturnFalse;
      cActor[i].TrainMode(false);
     }
   for(uint i = 0; i < cCritic.Size(); i++)
     {
      if(!cCritic[i].Clear())
         ReturnFalse;
      cCritic[i].TrainMode(false);
     }
   if(!cStateEncoder.Clear())
      ReturnFalse;
   cStateEncoder.TrainMode(false);
   for(int posit = start - HistoryBars - 1; (posit >= end && !IsStopped()); posit--)
     {
      if(!CreateBuffers(posit, GetPointer(bState), GetPointer(bTime), NULL))
         ReturnFalse;
      vector<float> account = vector<float>::Zeros(AccountDescr);
      account[0] = float(AccountInfoDouble(ACCOUNT_BALANCE) / EtalonBalance);
      account[2] = 1;
      double time = (double)bTime[0];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      account[9] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      account[10] = (float)MathCos(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_W1);
      account[11] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_D1);
      account[12] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      if(!bContext.AssignArray(account))
         ReturnFalse;
      if(!cStateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
         ReturnFalse;
      if(!RAGMemory.Retrieve(GetPointer(cStateEncoder),StateScenarioLayer))
        {
         PrintFormat("%s -> %d RAG retrieval failed", __FUNCTION__, __LINE__);
         ReturnFalse;
        }
      // Feed Forward.
      for(uint i = 0; (i < cActor.Size() && !IsStopped()); i++)
         if(!cActor[i].feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                                   GetPointer(cStateEncoder),StateScenarioLayer))
            ReturnFalse;
      for(uint i = 0; (i < cCritic.Size() && !IsStopped()); i++)
         if(!cCritic[i].feedForward(GetPointer(cActor[int(i > 0)]), -1,
                                    GetPointer(cStateEncoder), StateTokenLayer))
            ReturnFalse;
      if(GetTickCount() - ticks > 500)
        {
         double percent = (1.0 - double(posit - end) / (start - end - HistoryBars - NForecast))  * 100.0;
         string str = StringFormat("%-12s %6.2f%%", "Fill stack", percent);
         if(BaseMemoryNullBound)
            str += "\nRAG Null Memory";
         Comment(str);
         ticks = GetTickCount();
         if(MQLInfoInteger(MQL_TESTER) && percent >= tester_print)
           {
            Print(str);
            tester_print += 10;
           }
        }
     }
   cActor[0].TrainMode(true);
   cCritic[0].TrainMode(true);
   Comment("");
   if(MQLInfoInteger(MQL_TESTER))
      Print("Fill stack Done");
   return true;
  }
//+------------------------------------------------------------------+

