//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define Study
#define StudyOnline
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input datetime             Start          = D'2024.01.01';
input datetime             End            = D'2026.01.01';
input int                  Epochs         = 100;
input double               MinBalance     = 50;
input double               MaxBalance     = 150;
input bool                 LoadBaseMemorySnapshot = true;
input string               BaseMemorySnapshotPath = "VLADriverBaseMemory.snapshot";
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNet                 cActor;
CNet                 cCritic;
CNet                 cStateEncoder;
//---

datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bContext;
CBufferFloat         bTime;
CBufferFloat         bGradient;
CBufferFloat         *Result;
CBufferFloat         *Action;

//--- The library owns snapshot, records, consolidation and GPU retrieval.
CNeuronRAGMemory      RAGMemory;
bool                 OfflineHasLastStoredAction = false;
float                OfflineLastStoredAction[NActions];
const float          OfflineActionChangeThreshold = 0.01f;
const float          OfflineActionNonZeroThreshold = 0.001f;
ulong                OfflineCollectionChecks = 0;
ulong                OfflineTerminalFailures = 0;
ulong                OfflineRawLotRejections = 0;
ulong                OfflineScenarioAcquireFailures = 0;
ulong                OfflineScenarioShapeFailures = 0;
ulong                OfflineScenarioReadFailures = 0;
ulong                OfflineScenarioReplacedValues = 0;
ulong                OfflineNormalizationFailures = 0;
ulong                OfflineNormalizationInputFailures = 0;
ulong                OfflineNormalizationMarketFailures = 0;
ulong                OfflineNormalizationVolatilityFailures = 0;
ulong                OfflineNormalizationDescriptorFailures = 0;
ulong                OfflineNormalizationZeroFailures = 0;
ulong                OfflineDedupeRejections = 0;
ulong                OfflineRecordFailures = 0;
ulong                OfflineAcceptedRecords = 0;
ulong                OfflinePublishAttempts = 0;
ulong                OfflinePublishFailures = 0;
ulong                OfflinePublishSuccesses = 0;
bool                 OfflineFirstRejectionCaptured = false;
bool                 OfflineDiagnosticsPrinted = false;
string               OfflineFirstRejectionReason = "";
double               OfflineFirstRejectedAction0 = 0;
double               OfflineFirstRejectedAction3 = 0;
double               OfflineFirstRejectedATR = 0;
//+------------------------------------------------------------------+
//| Reset per-run offline RAG collection diagnostics.                |
//+------------------------------------------------------------------+
void ResetOfflineRAGDiagnostics(void)
  {
   OfflineCollectionChecks = 0;
   OfflineTerminalFailures = 0;
   OfflineRawLotRejections = 0;
   OfflineScenarioAcquireFailures = 0;
   OfflineScenarioShapeFailures = 0;
   OfflineScenarioReadFailures = 0;
   OfflineScenarioReplacedValues = 0;
   OfflineNormalizationFailures = 0;
   OfflineNormalizationInputFailures = 0;
   OfflineNormalizationMarketFailures = 0;
   OfflineNormalizationVolatilityFailures = 0;
   OfflineNormalizationDescriptorFailures = 0;
   OfflineNormalizationZeroFailures = 0;
   OfflineDedupeRejections = 0;
   OfflineRecordFailures = 0;
   OfflineAcceptedRecords = 0;
   OfflinePublishAttempts = 0;
   OfflinePublishFailures = 0;
   OfflinePublishSuccesses = 0;
   OfflineFirstRejectionCaptured = false;
   OfflineDiagnosticsPrinted = false;
   OfflineFirstRejectionReason = "";
   OfflineFirstRejectedAction0 = 0;
   OfflineFirstRejectedAction3 = 0;
   OfflineFirstRejectedATR = 0;
  }
//+------------------------------------------------------------------+
//| Retain one compact sample without changing collection decisions. |
//+------------------------------------------------------------------+
void CaptureOfflineRAGRejection(const string reason, const uint position)
  {
   if(OfflineFirstRejectionCaptured)
      return;
   OfflineFirstRejectionCaptured = true;
   OfflineFirstRejectionReason = reason;
   if(Action != NULL && Action.Total() > 3)
     {
      OfflineFirstRejectedAction0 = Action[0];
      OfflineFirstRejectedAction3 = Action[3];
     }
   if(position < Rates.Size())
      OfflineFirstRejectedATR = ATR.Main(position);
  }
//+------------------------------------------------------------------+
//| Print one final concise collection summary.                      |
//+------------------------------------------------------------------+
void PrintOfflineRAGDiagnostics(void)
  {
   if(OfflineDiagnosticsPrinted || (OfflineCollectionChecks == 0 && OfflinePublishAttempts == 0))
      return;
   OfflineDiagnosticsPrinted = true;
   PrintFormat("Offline RAG collection: checked=%I64u accepted=%I64u terminal=%I64u lot=%I64u acquire=%I64u shape=%I64u read=%I64u replaced=%I64u normalize=%I64u dedupe=%I64u add=%I64u",
               OfflineCollectionChecks, OfflineAcceptedRecords, OfflineTerminalFailures, OfflineRawLotRejections,
               OfflineScenarioAcquireFailures, OfflineScenarioShapeFailures, OfflineScenarioReadFailures,
               OfflineScenarioReplacedValues, OfflineNormalizationFailures, OfflineDedupeRejections,
               OfflineRecordFailures);
   PrintFormat("Offline RAG normalization: input=%I64u market=%I64u volatility=%I64u descriptor=%I64u zero=%I64u",
               OfflineNormalizationInputFailures, OfflineNormalizationMarketFailures,
               OfflineNormalizationVolatilityFailures, OfflineNormalizationDescriptorFailures,
               OfflineNormalizationZeroFailures);
   PrintFormat("Offline RAG publication: attempts=%I64u success=%I64u failures=%I64u scenarios=%u pending=%u",
               OfflinePublishAttempts, OfflinePublishSuccesses, OfflinePublishFailures,
               RAGMemory.ScenarioCount(), RAGMemory.PendingRecordCount());
   if(OfflineFirstRejectionCaptured)
      PrintFormat("Offline RAG first rejection: reason=%s action0=%.8f action3=%.8f atr=%.8f",
                  OfflineFirstRejectionReason, OfflineFirstRejectedAction0,
                  OfflineFirstRejectedAction3, OfflineFirstRejectedATR);
  }
//+------------------------------------------------------------------+
//| Persist the immutable library-owned Base Memory snapshot.        |
//+------------------------------------------------------------------+
bool SaveOfflineBaseMemory(void)
  {
// Do not replace a previously published image with an empty run.
   if(RAGMemory.ScenarioCount() == 0)
      return true;
   return RAGMemory.SavePublishedSnapshot(BaseMemorySnapshotPath);
  }
//+------------------------------------------------------------------+
//| Rebind the Actor immediately after COW snapshot publication.     |
//+------------------------------------------------------------------+
bool RebindOfflineRAGMemory(void)
  {
   if(cActor.BindRAGMemory(2, GetPointer(RAGMemory)))
      return true;
   PrintFormat("%s -> %d offline RAG Actor rebind failed", __FUNCTION__, __LINE__);
   ExpertRemove();
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Detach before the layer destroys an old COW inference buffer.    |
//+------------------------------------------------------------------+
bool DetachOfflineRAGMemory(void)
  {
   if(cActor.UnbindRAGMemory(2))
      return true;
   PrintFormat("%s -> %d offline RAG Actor detach failed", __FUNCTION__, __LINE__);
   ExpertRemove();
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Publish only records that were consumed into an immutable snapshot|
//+------------------------------------------------------------------+
bool PublishOfflineMemory(const bool episode_closed = false)
  {
   const uint pending = RAGMemory.PendingRecordCount();
   if(pending == 0)
      return true;
   if(!episode_closed && pending < OnlineMemorySize)
      return true;
   OfflinePublishAttempts++;
   if(!DetachOfflineRAGMemory())
     {
      OfflinePublishFailures++;
      ReturnFalse;
     }
   if(!RAGMemory.Publish(episode_closed))
     {
      if(!RebindOfflineRAGMemory())
        {
         OfflinePublishFailures++;
         ReturnFalse;
        }
      OfflinePublishFailures++;
      ReturnFalse;
     }
   if(!RebindOfflineRAGMemory())
     {
      OfflinePublishFailures++;
      ReturnFalse;
     }
   if(!SaveOfflineBaseMemory())
     {
      OfflinePublishFailures++;
      ReturnFalse;
     }
   OfflinePublishSuccesses++;
   return true;
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
//--- load models
   float temp;
   if(!cStateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true))
     {
      PrintFormat("Error of load StateEncoder: %d", GetLastError());
      return INIT_FAILED;
     }
   CArrayObj *actor = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(actor, critic))
     {
      DeleteObj(actor)
      DeleteObj(critic)
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor.Create(actor))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         DeleteObj(actor)
         DeleteObj(critic)
         return INIT_FAILED;
        }
     }
   if(!cCritic.Load(FileName + CriticCheckpointFile, temp, temp, temp, dtStudied, true))
     {
      Print("Create new Critic");
      if(!cCritic.Create(critic))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         DeleteObj(actor)
         DeleteObj(critic)
         return INIT_FAILED;
        }
     }
   DeleteObj(actor)
   DeleteObj(critic)
//---
   cActor.TrainMode(true);
   cCritic.TrainMode(true);
   cStateEncoder.TrainMode(false);
   COpenCL *opencl = cActor.GetOpenCL();
   cCritic.SetOpenCL(opencl);
   cStateEncoder.SetOpenCL(opencl);
   CLayerDescription memory_description;
   if(!CreateRAGMemoryDescription(memory_description) ||
      !RAGMemory.Init(0, 0, opencl, memory_description) ||
      !RAGMemory.SetOnlineMemorySize(OnlineMemorySize))
      return INIT_FAILED;
   const bool snapshot_loaded = (LoadBaseMemorySnapshot &&
                                 RAGMemory.LoadPublishedSnapshot(BaseMemorySnapshotPath));
   if(!snapshot_loaded && !RAGMemory.BindNullInference())
      return INIT_FAILED;
   if(!RebindOfflineRAGMemory())
      return INIT_FAILED;
   if(!bGradient.BufferInit(EmbeddingSize, 0.0f) ||
      !bGradient.BufferCreate(opencl))
      return INIT_FAILED;
//---
   if(!Symb.Name(_Symbol) || !Symb.Refresh())
      return INIT_FAILED;
//---
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
//---
   cActor.GetLayerOutput(0, Result);
   if(Result.Total() != AccountDescr)
     {
      PrintFormat("Input size of Actor doesn't match context description (%d <> %d)", Result.Total(), AccountDescr);
      return INIT_FAILED;
     }
//---
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
//---
   cStateEncoder.GetLayerOutput(StateScenarioLayer, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("StateEncoder token layer doesn't match pooled scenario embedding (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(reason != REASON_INITFAILED && !PublishOfflineMemory(true))
      Print("Error of publish Base Memory snapshot");
   PrintOfflineRAGDiagnostics();
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      if(!cActor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Actor");
      if(!cCritic.Save(FileName + CriticCheckpointFile, 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Critic");
     }
   delete Result;
   delete Action;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   switch(id)
     {
      case 1001:
         Train();
         break;
      case 1007:
         Print("Event 1007");
         if(!EventChartCustom(ChartID(), 1, 0, 0, "ChartEvent"))
           {
            PrintFormat("Error of create study event: %d", GetLastError());
            ExpertRemove();
           }
         break;
     }
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   ResetOfflineRAGDiagnostics();
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars < 0)
     {
      PrintFormat("%s -> %d CopyRates failed (%d)", __FUNCTION__, __LINE__, GetLastError());
      ExpertRemove();
      return;
     }
//---
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   int count = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars &&
                    CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars &&
                    MACD.BarsCalculated() >= bars
                   );
      Sleep(100);
      count++;
     }
   while(!calculated && count < 100);
   if(!calculated)
     {
      PrintFormat("%s -> %d The training data has not been loaded", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   vector<float> result, target, neg_target;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int epoch = 0; (epoch < Epochs && !IsStopped() && !Stop); epoch ++)
     {
      if(!cActor.Clear() ||
         !cCritic.Clear() ||
         !cStateEncoder.Clear())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      for(int posit = start - HistoryBars - NForecast - 1; posit >= end; posit--)
        {
         if(!CreateBuffers(posit, GetPointer(bState), GetPointer(bTime), Result))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         const vector<float> account = SampleAccount(GetPointer(bState), datetime(bTime[0]), MaxBalance, MinBalance);
         const vector<float> target_action = OraculAction(account, Result);
         if(!bContext.AssignArray(account))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         //--- Feed Forward
         if(!cStateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!RAGMemory.Retrieve(GetPointer(cStateEncoder), StateScenarioLayer))
           {
            PrintFormat("%s -> %d RAG retrieval failed", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                                GetPointer(cStateEncoder), StateScenarioLayer))
           {
            PrintFormat("%s -> %d Actor forward failed", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
#ifdef _DEBUG
         vector<float> res;
         cActor.getResults(res);
#endif
         if(!cCritic.feedForward(GetPointer(cActor), -1, GetPointer(cStateEncoder), StateTokenLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
#ifdef _DEBUG
         cCritic.getResults(res);
#endif
         //--- Study
         cActor.getResults(Action);
         double balance = account[0] * EtalonBalance;
         const uint terminal_position = uint(posit - NForecast + 1);
         SRAGTerminalOutcome terminal;
         OfflineCollectionChecks++;
         bool terminal_evaluated = EvaluateTerminalOutcome(RAGMemory, Action, balance, terminal_position, terminal);
         double reward = terminal.reward * balance / EtalonBalance;
         if(!terminal_evaluated)
           {
            OfflineTerminalFailures++;
            CaptureOfflineRAGRejection("terminal", terminal_position);
           }
         else
            if(MathAbs(Action[0] - Action[3]) < OfflineActionNonZeroThreshold)
              {
               OfflineRawLotRejections++;
               CaptureOfflineRAGRejection("lot", terminal_position);
              }
            else
              {
               CBufferFloat *scenario_embedding = NULL;
               if(!cStateEncoder.GetLayerOutputDevice(StateScenarioLayer, scenario_embedding))
                 {
                  OfflineScenarioAcquireFailures++;
                  CaptureOfflineRAGRejection("scenario_acquire", terminal_position);
                 }
               else
                  if(scenario_embedding.Total() != EmbeddingSize)
                    {
                     OfflineScenarioShapeFailures++;
                     CaptureOfflineRAGRejection("scenario_shape", terminal_position);
                    }
                  else
                    {
                     float normalized_action[NActions];
                     ENUM_MEMORY_ACTION_NORMALIZATION_STAGE normalization_stage;
                     if(NormalizeMemoryAction(RAGMemory, Action, terminal_position, balance, normalized_action,
                                              normalization_stage))
                       {
                        bool materially_changed = !OfflineHasLastStoredAction;
                        for(int i = 0; i < NActions && !materially_changed; i++)
                           materially_changed = (MathAbs(normalized_action[i] - OfflineLastStoredAction[i]) >= OfflineActionChangeThreshold);
                         if(materially_changed)
                           {
                            float scenario_values[];
                            uint replaced_values = 0;
                            const bool scenario_read = RAGMemory.ReadScenarioEmbedding(scenario_embedding,
                                                       scenario_values, replaced_values);
                            OfflineScenarioReplacedValues += replaced_values;
                            if(!scenario_read)
                              {
                               OfflineScenarioReadFailures++;
                               CaptureOfflineRAGRejection("scenario_read", terminal_position);
                               PrintFormat("%s -> %d scenario anchor read failed", __FUNCTION__, __LINE__);
                               Stop = true;
                               break;
                              }
                            if(!RAGMemory.AddCompletedRecord(scenario_values, normalized_action, terminal))
                              {
                               OfflineRecordFailures++;
                               CaptureOfflineRAGRejection("add_record", terminal_position);
                               PrintFormat("%s -> %d offline RAG record rejected", __FUNCTION__, __LINE__);
                               Stop = true;
                               break;
                              }
                            OfflineAcceptedRecords++;
                            ArrayCopy(OfflineLastStoredAction, normalized_action);
                            OfflineHasLastStoredAction = true;
                           }
                        else
                          {
                           OfflineDedupeRejections++;
                           CaptureOfflineRAGRejection("dedupe", terminal_position);
                          }
                       }
                     else
                       {
                        OfflineNormalizationFailures++;
                        switch(normalization_stage)
                          {
                           case MEMORY_ACTION_NORMALIZATION_INPUT:
                              OfflineNormalizationInputFailures++;
                              break;
                           case MEMORY_ACTION_NORMALIZATION_MARKET:
                              OfflineNormalizationMarketFailures++;
                              break;
                           case MEMORY_ACTION_NORMALIZATION_VOLATILITY:
                              OfflineNormalizationVolatilityFailures++;
                              break;
                           case MEMORY_ACTION_NORMALIZATION_DESCRIPTOR:
                              OfflineNormalizationDescriptorFailures++;
                              break;
                           case MEMORY_ACTION_NORMALIZATION_ZERO:
                              OfflineNormalizationZeroFailures++;
                              break;
                          }
                        CaptureOfflineRAGRejection("normalize_" + MemoryActionNormalizationStageName(normalization_stage),
                                                   terminal_position);
                       }
                    }
              }
         //---
         Result.Clear();
         if(!Result.Add(float(reward)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cCritic.backProp(Result, GetPointer(cStateEncoder), StateTokenLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Oracul
         if(!Action.AssignArray(target_action))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         reward = CheckAction(RAGMemory, Action, balance, posit - NForecast + 1) * balance / EtalonBalance;
         CBufferFloat *actor_market_gradient = NULL;
         if(!cStateEncoder.GetLayerOutputDevice(StateScenarioLayer, actor_market_gradient))
           {
            PrintFormat("%s -> %d actor market gradient input failed", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(reward > 0)
            if(!cActor.backProp(Action, actor_market_gradient, GetPointer(bGradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                                GetPointer(cStateEncoder), StateScenarioLayer) ||
            !cCritic.feedForward(Action, 1, false, GetPointer(cStateEncoder), StateTokenLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!Result.Update(0, float(reward)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cCritic.backProp(Result, GetPointer(cStateEncoder), StateTokenLayer) ||
            !cActor.backPropGradient(actor_market_gradient, GetPointer(bGradient), LatentLayer, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         cCritic.TrainMode(false);
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bContext), 1, false,
                                GetPointer(cStateEncoder), StateScenarioLayer) ||
            !cCritic.feedForward(GetPointer(cActor), -1, GetPointer(cStateEncoder), StateTokenLayer) ||
            !cCritic.backProp(Result, GetPointer(cStateEncoder), StateTokenLayer) ||
            !cActor.backPropGradient(actor_market_gradient, GetPointer(bGradient), -1, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         cCritic.TrainMode(true);
         if(RAGMemory.PendingRecordCount() >= OnlineMemorySize && !PublishOfflineMemory(false))
           {
            PrintFormat("%s -> %d offline memory publication failed", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (epoch + 1.0 - double(posit - end) / (start - end - HistoryBars - NForecast)) / Epochs * 100.0;
            string str = "";
            str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Actor", percent, cActor.getRecentAverageError());
            str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Critic", percent, cCritic.getRecentAverageError());
            str += StringFormat("RAG scenarios %u, pending %u\n", RAGMemory.ScenarioCount(), RAGMemory.PendingRecordCount());
            str += StringFormat("RAG collect accepted %I64u, lot %I64u, normalize %I64u, dedupe %I64u\n",
                                OfflineAcceptedRecords, OfflineRawLotRejections,
                                OfflineNormalizationFailures, OfflineDedupeRejections);
            Comment(str);
            ticks = GetTickCount();
           }
        }
      if(!Stop && !PublishOfflineMemory(true))
         PrintFormat("%s -> %d offline memory publication failed", __FUNCTION__, __LINE__);
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", cActor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", cCritic.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
