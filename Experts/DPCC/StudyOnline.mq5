//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property version   "3.00"
#property description "DPCC online Actor and twin Critic learning in Strategy Tester"
#property description "Updates the policy from one bar of reward and bootstrapped TD targets"
#property description "Continues through the configured tester interval"
#property strict

#define DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST
#include "Trajectory.mqh"
#undef DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST

input group                 "---- Общие параметры режима ----"
//--- Таймфрейм; PERIOD_CURRENT — таймфрейм графика/тестера.
input ENUM_TIMEFRAMES        InpTimeframe = PERIOD_CURRENT;
input ENUM_DPCC_RESUME_MODE  InpResumeMode = DPCC_RESUME_UNASSIGNED;    // Загрузить веса для продолжения обучения.
input ENUM_DPCC_MODE         InpMode = DPCC_P0;                         // Режим DPCC: P0, P1 или P2.
input long                   InpMagic = 33062;                          // Magic-номер заявок этого советника.
input bool                   InpMeasureLatency = false;                 // Измерять задержку принятия решения.

input group                 "---- Частота обновлений и TD ----"
input uint                   InpUpdatePolicy = 2;      // Обновлять Actor через каждые N переходов.
input uint                   InpUpdateTargets = 20;    // Обновлять Target-сети через каждые N переходов.
input double                 InpTauT = 1.0;            // Коэффициент Polyak-сглаживания Target-сетей.
input double                 InpGamma = 0.5;           // Коэффициент дисконтирования награды по барам.
input int                    InpSelector = 0;          // Источник градиента Actor: 0 — по хешу события, 1 — Q1, 2 — Q2.
input uint                   InpSeed = 20260920;       // Начальное значение для воспроизводимой случайной выборки.

input group                 "---- Распределительные Критики и мера риска ----"
input int                    InpRiskType = 2;           // Мера риска SRM, заданная значением defSRM_*.
input double                 InpRiskParameter = 0.1;    // Параметр выбранной меры SRM: alpha, lambda или nu.
input double                 InpMeanWeight = 0.0;       // Вес среднего результата в Mean-CVaR.
input double                 InpProbWeight = 1.0;       // Вес вероятностей квантильного распределения.
input double                 InpValueWeight = 1.0;      // Вес значений квантилей в распределении.
input uint                   InpQuantiles = 32;         // Число квантилей Critic; архитектура рассчитана на 32.

input group                 "---- Нижняя граница счёта ----"
input double                 InpMinBalance = 50.0;    // Минимальный баланс или эквити для торговли.

const string DPCC_PACKAGE_ROOT = "DPCC/StudyForecast_M05_20261017";
const bool   DPCC_COMMON_FILES = true;
//+------------------------------------------------------------------+
//| Shared Market and its six independent Trading roles              |
//+------------------------------------------------------------------+
CDPCCStudyForecastCore        EXTMarket;
CDPCCAction6ChartContextOwner EXTDecisionContext;
CNet                        *EXTActor = NULL;
CNet                        *EXTQ1 = NULL;
CNet                        *EXTQ2 = NULL;
CNet                        *EXTTargetActor = NULL;
CNet                        *EXTTargetQ1 = NULL;
CNet                        *EXTTargetQ2 = NULL;
SDPCCExecutionControl        EXTControl;
SDPCCAction6TradingRestore   EXTRestored;
CBufferFloat                 EXTCurrentZ;
CBufferFloat                 EXTActorAccount;
CBufferFloat                 EXTTargetAccount;
CBufferFloat                 EXTTargetQuery;
CBufferFloat                 EXTSample;
CBufferFloat                 EXTRandom;
SLayerReadState              EXTPendingMarketState;
SLayerReadState              EXTNextMarketState;
float                        EXTPendingAccount[13];
float                        EXTPendingAction[6];
bool                         EXTPending = false;
bool                         EXTPendingActorFrame = false;
bool                         EXTMutationFailed = false;
bool                         EXTFailed = false;
ulong                        EXTTransitions = 0;
ulong                        EXTPolicyEvents = 0;
ulong                        EXTTargetUpdates = 0;
ulong                        EXTActorNoPlan = 0;
ulong                        EXTTargetActorNoPlan = 0;
bool                         EXTTargetQ1First = true;
string                       EXTDonorSHA = "";
uint                         EXTUpdatePolicy = 0;
uint                         EXTUpdateTargets = 0;
uint                         EXTDonorSeed = 0;
int                          EXTSelector = -1;
float                        EXTGammaTD = 0;
float                        EXTTauTarget = 0;
float                        EXTProbabilityWeight = 0;
float                        EXTValueWeight = 0;
double                       EXTMinBalance = 0;
int                          EXTIndicators[4];
long                         EXTBarOpen = 0;
ulong                        EXTDecisions = 0;
ulong                        EXTPendingMarketGeneration = 0;
SDPCCDecisionMeta             EXTPendingDecision;
double                       EXTMarkBalance = 0;
double                       EXTMarkEquity = 0;
bool                         EXTReady = false;
bool                         EXTStopped = false;
bool                         EXTTrainingComplete = false;
bool                         EXTControlledFloorStop = false;
bool                         EXTModelSaveAttempted = false;
bool                         EXTModelSaved = false;
string                       EXTModelSavePath = "";
string                       EXTModelSaveSHA = "";
string                       EXTModelSaveReason = "";
ulong                        EXTLatencySamples = 0;
ulong                        EXTDecisionLatencyUs = 0;
ulong                        EXTDecisionLatencyMaxUs = 0;
//+------------------------------------------------------------------+
//| Stop before another model transaction or execution request       |
//+------------------------------------------------------------------+
void DPCCOnlineStop(const string phase, const string reason)
  {
   EXTStopped = true;
   EXTFailed = true;
   Print("DPCC_STUDYONLINE_BLOCKED phase=", phase, " reason=", reason, " bar=", EXTBarOpen);
   if(MQLInfoInteger(MQL_TESTER))
      TesterStop();
  }
//+------------------------------------------------------------------+
//| Configure the direct AC-SRM online-learning inputs               |
//+------------------------------------------------------------------+
bool DPCCOnlineConfigureLearning(string &reason)
  {
   const SDPCCConfig financial = EXTRestored.runtime.financial;
   if(!MathIsValidNumber(InpTauT) || !MathIsValidNumber(InpGamma) ||
      !MathIsValidNumber(InpRiskParameter) || !MathIsValidNumber(InpMeanWeight) ||
      !MathIsValidNumber(InpProbWeight) || !MathIsValidNumber(InpValueWeight) ||
      !MathIsValidNumber(InpMinBalance) || InpUpdatePolicy < 1 || InpUpdateTargets < 1 ||
      InpTauT <= 0 || InpTauT > 1 ||
      InpGamma < 0 || InpGamma > 1 || InpSelector < 0 || InpSelector > 2 ||
      InpSeed > INT_MAX ||
      InpRiskType != financial.risk_type || (float)InpRiskParameter != financial.risk_parameter ||
      (float)InpMeanWeight != financial.mean_weight || InpProbWeight < 0 || InpValueWeight < 0 ||
      InpQuantiles != 32 || InpMinBalance <= 0)
     {
      reason = "online.inputs.runtime_SRM_or_cadence_mismatch";
      return(false);
     }
   EXTUpdatePolicy = InpUpdatePolicy;
   EXTUpdateTargets = InpUpdateTargets;
   EXTTauTarget = (float)InpTauT;
   EXTGammaTD = (float)InpGamma;
   EXTSelector = InpSelector;
   EXTDonorSeed = InpSeed;
   EXTProbabilityWeight = (float)InpProbWeight;
   EXTValueWeight = (float)InpValueWeight;
   EXTMinBalance = InpMinBalance;
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocate reused input and sampling buffers on the Market context |
//+------------------------------------------------------------------+
bool DPCCOnlinePrepareBuffers(string &reason)
  {
   CNet *market = EXTMarket.ReadyMarketNetwork();
   COpenCLMy *context = EXTMarket.ReadyEncoderContext();
   CNeuronScenarioForecastPriceOCL *forecast = NULL;
   if(CheckPointer(market) != POINTER_INVALID)
      forecast = dynamic_cast<CNeuronScenarioForecastPriceOCL*>(market.Layer(6));
   ulong count = 0;
   if(!forecast || CheckPointer(context) == POINTER_INVALID || forecast.GetOpenCL() != context ||
      forecast.CodebookUpdate() || !DPCCRolloutProduct(forecast.Scenarios(), forecast.Variables(), count) ||
      !DPCCRolloutProduct(count, forecast.Horizon(), count) ||
      !DPCCRolloutProduct(count, forecast.Dimension(), count) || count == 0 ||
      count > (ulong)INT_MAX || count > (ulong)UINT_MAX / sizeof(float))
     {
      reason = "online.market.snapshot_axes_or_frozen_contract";
      return(false);
     }
   CBufferFloat *buffers[] = {GetPointer(EXTCurrentZ), GetPointer(EXTActorAccount),
                              GetPointer(EXTTargetAccount), GetPointer(EXTTargetQuery),
                              GetPointer(EXTSample), GetPointer(EXTRandom)
                             };
   const uint sizes[] = {(uint)count, 13, 13, 19, 1, 1};
   for(int i = 0; i < 6; i++)
     {
      if(!buffers[i].BufferInit(sizes[i], 0))
        {
         reason = StringFormat("online.buffer_init.%d", i);
         return(false);
        }
      if(!buffers[i].BufferCreate(context))
        {
         reason = StringFormat("online.buffer_create.%d", i);
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Save or restore only the frozen Market's three policy streams    |
//+------------------------------------------------------------------+
bool DPCCOnlineMarketState(SLayerReadState &state, const bool capture, const ulong decision,
                           const ulong generation, string &reason)
  {
   CNet *market = EXTMarket.ReadyMarketNetwork();
   CNeuronScenarioForecastPriceOCL *forecast = CheckPointer(market) == POINTER_INVALID ? NULL :
         dynamic_cast<CNeuronScenarioForecastPriceOCL*>(market.Layer(6));
   if(!forecast || (state.Context() != NULL && state.Context() != market.GetOpenCL()))
     {
      reason = "online.market_snapshot.context";
      return(false);
     }
   CBufferFloat *buffers[] = {forecast.GetZ(), forecast.GetLogU(), forecast.GetPi()};
   if(capture)
     {
      if(!state.Begin(market.GetOpenCL(), decision, generation))
        {
         reason = "online.market_snapshot.begin";
         return(false);
        }
      for(int i = 0; i < 3; i++)
         if(CheckPointer(buffers[i]) == POINTER_INVALID || !state.PutBuffer(buffers[i]))
           {
            reason = StringFormat("online.market_snapshot.capture.%d", i);
            return(false);
           }
     }
   else
     {
      state.Rewind();
      for(int i = 0; i < 3; i++)
        {
         CBufferFloat *original = buffers[i];
         if(!state.GetBuffer(buffers[i]) || buffers[i] != original)
           {
            reason = StringFormat("online.market_snapshot.restore.%d", i);
            return(false);
           }
        }
      if(!state.Complete())
        {
         reason = "online.market_snapshot.incomplete";
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Keep Market and targets frozen while enabling the online roles   |
//+------------------------------------------------------------------+
bool DPCCOnlineLearningPermissions(string &reason)
  {
   CNet *market = EXTMarket.ReadyMarketNetwork();
   CNet *networks[] = {EXTActor, EXTQ1, EXTQ2, EXTTargetActor, EXTTargetQ1, EXTTargetQ2};
   const int layers[] = {2, 5, 5, 2, 5, 5};
   COpenCLMy *context = EXTMarket.ReadyEncoderContext();
   if(CheckPointer(market) == POINTER_INVALID || CheckPointer(context) == POINTER_INVALID ||
      market.GetOpenCL() != context || market.WeightsUpdateEnabled())
     {
      reason = "online.market.frozen_context";
      return(false);
     }
   for(int i = 0; i < 6; i++)
     {
      const bool trainable = (i < 3);
      if(CheckPointer(networks[i]) == POINTER_INVALID || networks[i].GetOpenCL() != context ||
         !networks[i].IsOpenCLTransferReady() || networks[i].Layer(layers[i]) != NULL ||
         !networks[i].TrainMode(trainable) || !networks[i].SetWeightsUpdate(trainable))
        {
         reason = StringFormat("online.role.permission_or_context.%d", i);
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Restore the same Market and native Action6 bundle used by Study  |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   for(int i = 0; i < 4; i++)
      EXTIndicators[i] = INVALID_HANDLE;
   string reason = "";
   if(!DPCCConfigureExecutionControl(DPCC_PACKAGE_ROOT, InpTimeframe, DPCC_STAGE_STUDYONLINE, "",
                                     0, InpResumeMode, InpMode, DPCC_STAGE_STUDYONLINE, EXTControl, reason) ||
      !DPCCLoadMarketExecutionProfile(EXTControl, DPCC_COMMON_FILES, reason))
     {
      DPCCOnlineStop("execution_control", reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpMagic < 1 || !DPCCTesterOnly(reason) ||
      !DPCCRequireTesterOneDirection(reason))
     {
      DPCCOnlineStop("environment", reason == "" ? "H1_Tester_required" : reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(_Symbol == "" || AccountInfoString(ACCOUNT_CURRENCY) == "" ||
      PositionsTotal() != 0 || OrdersTotal() != 0)
     {
      DPCCOnlineStop("environment", "flat_Tester_start_required");
      return(INIT_FAILED);
     }
   if(!DPCCLoadAction6Trading(EXTControl, DPCC_COMMON_FILES, EXTMarket, EXTActor, EXTQ1, EXTQ2,
                              EXTTargetActor, EXTTargetQ1, EXTTargetQ2, EXTRestored, reason))
     {
      DPCCOnlineStop("D01_restore", reason);
      return(INIT_FAILED);
     }
   if(EXTRestored.diffusion_seed == 0)
     {
      DPCCOnlineStop("profile", "assigned_profile.diffusion.seed_must_be_nonzero_for_Action6");
      return(INIT_FAILED);
     }
   if(!DPCCOnlineConfigureLearning(reason) || !DPCCOnlinePrepareBuffers(reason) ||
      !DPCCOnlineLearningPermissions(reason))
     {
      DPCCOnlineStop("policy", reason);
      return(INIT_FAILED);
     }
   EXTIndicators[0] = iRSI(_Symbol, EXTControl.timeframe, RSIPeriod, RSIPrice);
   EXTIndicators[1] = iCCI(_Symbol, EXTControl.timeframe, CCIPeriod, CCIPrice);
   EXTIndicators[2] = iATR(_Symbol, EXTControl.timeframe, ATRPeriod);
   EXTIndicators[3] = iMACD(_Symbol, EXTControl.timeframe, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice);
   for(int i = 0; i < 4; i++)
      if(EXTIndicators[i] == INVALID_HANDLE)
        {
         DPCCOnlineStop("indicators", (string)GetLastError());
         return(INIT_FAILED);
        }
   EXTMarkBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   EXTMarkEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(!MathIsValidNumber(EXTMarkBalance) || !MathIsValidNumber(EXTMarkEquity) ||
      EXTMarkBalance < EXTMinBalance || EXTMarkEquity < EXTMinBalance)
     {
      DPCCOnlineStop("account", "actual_Tester_start_below_floor");
      return(INIT_FAILED);
     }
//--- Study supplies initial targets; an Online generation already owns its target parameters.
   if(EXTRestored.source_stage == "Study")
     {
      if(!EXTTargetActor.WeightsUpdate(EXTActor, 1.0f) ||
         !EXTTargetQ1.WeightsUpdate(EXTQ2, 1.0f) || !EXTTargetQ2.WeightsUpdate(EXTQ1, 1.0f))
        {
         DPCCOnlineStop("targets", "cold_Study_crossed_initialization");
         return(INIT_FAILED);
        }
     }
   else
      if(EXTRestored.source_stage != "StudyOnline")
        {
         DPCCOnlineStop("targets", "Trading_source_stage_required");
         return(INIT_FAILED);
        }
   MathSrand((int)EXTDonorSeed);
   EXTReady = true;
   Print("DPCC_STUDYONLINE_READY source_stage=", EXTRestored.source_stage,
         " donor=", EXTDonorSHA, " model_export=ON_COMPLETION_ONCE");
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Encode the next closed bar and build one pre-action account mark |
//+------------------------------------------------------------------+
bool DPCCOnlineObserve(const long decision_open, CBufferFloat *&context, float &account[],
                       string &reason)
  {
   context = NULL;
   SDPCCClosedHistoryRow closed;
   MqlTick tick;
   long quote_mode = -1;
   double tick_size = 0;
   if(!DPCCCaptureTesterClosedRow(_Symbol, EXTIndicators, 1, closed, reason) ||
      closed.successor_open_server != decision_open || !SymbolInfoTick(_Symbol, tick) ||
      decision_open > LONG_MAX / 1000 - 3600 || tick.time_msc < decision_open * 1000 ||
      tick.time_msc >= (decision_open + 3600) * 1000 ||
      !SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE, quote_mode) ||
      !SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE, tick_size) ||
      (quote_mode != SYMBOL_CHART_MODE_BID && quote_mode != SYMBOL_CHART_MODE_LAST))
     {
      if(reason == "")
         reason = "online.causal_H1_quote_binding";
      return(false);
     }
   if(EXTDecisions == 0 &&
      !DPCCPrepareReadyMarketLiveStart(EXTMarket, _Symbol, EXTIndicators,
                                       closed.bar.server_open, reason))
      return(false);
   const double first_quote = (quote_mode == SYMBOL_CHART_MODE_LAST ? tick.last : tick.bid);
   if(!EXTMarket.InferReadyMarketLive(closed, first_quote, tick.time_msc, tick_size,
                                      GetPointer(EXTCurrentZ), EXTPendingMarketGeneration, reason))
      return(false);
   if(!EXTDecisionContext.Build(EXTRestored.runtime, EXTRestored.limits,
                                EXTMarket.ReadyMarketNativeOHLC(), decision_open, context, account, reason))
      return(false);
   if(ArraySize(account) != 13 || CheckPointer(context) == POINTER_INVALID)
     {
      reason = "online.Account13_context_shape";
      return(false);
     }
   for(int i = 0; i < 13; i++)
      if(!MathIsValidNumber(account[i]))
        {
         reason = "online.Account13_nonfinite";
         return(false);
        }
   return(true);
  }
//+------------------------------------------------------------------+
//| Bind either Actor to this causal observation and its own input   |
//+------------------------------------------------------------------+
bool DPCCOnlineActorForward(CNet &network, CBufferFloat &account_buffer, CBufferFloat *context,
                            const long decision_open, const float &account[],
                            float &action[], SDPCCDecisionMeta &meta, bool &has_plan, string &reason)
  {
   has_plan = false;
   CNeuronDPCCOCL *actor = dynamic_cast<CNeuronDPCCOCL*>(network.Layer(1));
   if(!actor || ArraySize(account) != 13 || EXTPendingMarketGeneration == 0 ||
      EXTDecisions >= (ulong)LONG_MAX - 2)
     {
      reason = "online.Actor_input_or_decision_id";
      return(false);
     }
   for(int i = 0; i < 13; i++)
      if(!MathIsValidNumber(account[i]) || !account_buffer.Update(i, account[i]))
        {
         reason = "online.Actor_Account13_upload";
         return(false);
        }
   ZeroMemory(meta);
   meta.decision_id = EXTDecisions + 1;
   meta.market_state_id = EXTPendingMarketGeneration;
   meta.q1_state_id = meta.decision_id;
   meta.q2_state_id = meta.decision_id + 1;
   meta.noise_seed = EXTRestored.diffusion_seed;
   meta.decision_server_time = decision_open;
   meta.bar_index = (long)EXTDecisions;
   meta.mode = (uint)EXTControl.mode;
   meta.market_sha256 = EXTRestored.market_sha256;
   meta.profile_sha256 = EXTControl.profile_sha256;
   if(!account_buffer.BufferWrite() || !actor.BindDecisionContext(context, meta))
     {
      reason = "online.Action6_context";
      return(false);
     }
   const bool forwarded = network.feedForward(GetPointer(account_buffer), 1, false,
                          EXTMarket.ReadyMarketNetwork(), 6);
   if(!DPCCResolveAction6Output(actor, forwarded, action, reason))
     {
      reason = "online." + reason;
      return(false);
     }
   has_plan = actor.HasValidDecision();
   return(true);
  }
//+------------------------------------------------------------------+
//| Release Market generation without consuming the Actor unroll     |
//+------------------------------------------------------------------+
bool DPCCOnlineReleaseMarket(string &reason)
  {
   if(EXTPendingMarketGeneration == 0)
      return(true);
   if(!EXTMarket.ReleaseReadyEncoderZ(EXTPendingMarketGeneration, reason))
     {
      reason = "online.Market_release:" + reason;
      return(false);
     }
   EXTPendingMarketGeneration = 0;
   return(true);
  }
//+------------------------------------------------------------------+
//| Close one retained Actor frame after update, skip, or shutdown   |
//+------------------------------------------------------------------+
bool DPCCOnlineCancelActor(CNet *network, string &reason)
  {
   CNeuronDPCCOCL *actor = CheckPointer(network) == POINTER_INVALID ? NULL :
                           dynamic_cast<CNeuronDPCCOCL*>(network.Layer(1));
   if(!actor || !actor.CancelGeneration())
     {
      reason = "online.Actor_frame_release";
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Sample head 4 using the donor's open-interval random variate     |
//+------------------------------------------------------------------+
bool DPCCOnlineSample(CNet &critic, float &value, string &reason)
  {
   CNeuronACSRM *head = dynamic_cast<CNeuronACSRM*>(critic.Layer(4));
   if(!head)
     {
      reason = "online.target_Q.head4_required";
      return(false);
     }
   const float u = (float)((double)MathRand() + 0.5) / 32768.0f;
   if(!EXTRandom.Update(0, u) || !EXTRandom.BufferWrite() ||
      !head.Sample(GetPointer(EXTSample), GetPointer(EXTRandom), 1) ||
      !EXTSample.BufferRead() || !MathIsValidNumber(EXTSample.At(0)))
     {
      reason = "online.target_Q.sample";
      return(false);
     }
   value = EXTSample.At(0);
   return(true);
  }
//+------------------------------------------------------------------+
//| Bootstrap the next causal state with the frozen target models    |
//+------------------------------------------------------------------+
bool DPCCOnlineBootstrap(CBufferFloat *context, const long decision_open,
                         const float &account[], float &z1, float &z2, string &reason)
  {
   float action[];
   SDPCCDecisionMeta meta;
   bool has_plan = false;
   if(!DPCCOnlineActorForward(*EXTTargetActor, EXTTargetAccount, context, decision_open,
                              account, action, meta, has_plan, reason))
      return(false);
   if(!DPCCOnlineCancelActor(EXTTargetActor, reason))
      return(false);
   if(!has_plan)
      EXTTargetActorNoPlan++;
   for(int i = 0; i < 19; i++)
      if(!EXTTargetQuery.Update(i, i < 13 ? account[i] : action[i - 13]))
        {
         reason = "online.target_Q.Account13_Action6_upload";
         return(false);
        }
   if(!EXTTargetQuery.BufferWrite())
     {
      reason = "online.target_Q.query_write";
      return(false);
     }
   CNet *market = EXTMarket.ReadyMarketNetwork();
   if(!EXTTargetQ1.feedForward(GetPointer(EXTTargetQuery), 1, false, market, 6) ||
      !DPCCOnlineSample(*EXTTargetQ1, z1, reason))
     {
      reason = "online.target_Q1:" + reason;
      return(false);
     }
   if(!EXTTargetQ2.feedForward(GetPointer(EXTTargetQuery), 1, false, market, 6) ||
      !DPCCOnlineSample(*EXTTargetQ2, z2, reason))
     {
      reason = "online.target_Q2:" + reason;
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Refresh Actor and one crossed Critic on the donor target cadence |
//+------------------------------------------------------------------+
bool DPCCOnlineRefreshTargets(string &reason)
  {
   if(EXTTransitions == 0 || EXTTransitions % EXTUpdateTargets != 0)
      return(true);
   if(!EXTTargetActor.WeightsUpdate(EXTActor, EXTTauTarget))
     {
      reason = "online.target_Actor.weights_update";
      return(false);
     }
   if((EXTTargetUpdates & 1) == 0)
     {
      EXTTargetQ1First = ((MathRand() & 1) == 0);
     }
   const bool update_q1 = EXTTargetQ1First ? ((EXTTargetUpdates & 1) == 0) :
                          ((EXTTargetUpdates & 1) != 0);
   if(update_q1 ? !EXTTargetQ1.WeightsUpdate(EXTQ2, EXTTauTarget) :
      !EXTTargetQ2.WeightsUpdate(EXTQ1, EXTTauTarget))
     {
      reason = "online.crossed_target_Q.weights_update";
      return(false);
     }
   EXTTargetUpdates++;
   return(true);
  }
//+------------------------------------------------------------------+
//| Complete exactly one pending transition before a new action      |
//+------------------------------------------------------------------+
bool DPCCOnlineUpdatePending(CBufferFloat *context, const long decision_open,
                             const float &next_account[], const double next_equity,
                             string &reason)
  {
   if(!EXTPending)
      return(true);
   float reward = 0, z1 = 0, z2 = 0, y1 = 0, y2 = 0;
   if(!DPCCBuildOneBarReward(EXTMarkEquity, next_equity, reward, reason))
      return(false);
   if(!DPCCOnlineBootstrap(context, decision_open, next_account, z1, z2, reason))
      return(false);
   if(!DPCCBuildOneBarTDTargets(reward, EXTGammaTD, 1.0f, z1, z2, y1, y2, reason))
      return(false);
//--- Preserve s(t+1), train against s(t), then return Market to the new decision.
   if(!DPCCOnlineMarketState(EXTNextMarketState, true, EXTDecisions + 1,
                             EXTPendingMarketGeneration, reason))
      return(false);
   if(!DPCCOnlineMarketState(EXTPendingMarketState, false, 0, 0, reason))
     {
      string restore_reason = "";
      if(!DPCCOnlineMarketState(EXTNextMarketState, false, 0, 0, restore_reason))
         reason += ";next_state_restore:" + restore_reason;
      return(false);
     }
   float query[19];
   for(int i = 0; i < 13; i++)
      query[i] = EXTPendingAccount[i];
   for(int i = 0; i < 6; i++)
      query[13 + i] = EXTPendingAction[i];
   CNet *market = EXTMarket.ReadyMarketNetwork();
   bool updated = DPCCUpdateAction6Critic(*EXTQ1, *market, query, y1,
                                          EXTProbabilityWeight, EXTValueWeight, reason);
   if(updated)
      updated = DPCCUpdateAction6Critic(*EXTQ2, *market, query, y2,
                                        EXTProbabilityWeight, EXTValueWeight, reason);
   if(updated && EXTPendingActorFrame)
     {
      const ulong transition = EXTTransitions + 1;
      if(transition % EXTUpdatePolicy == 0)
        {
         const int selected = DPCCOnlineActorCritic(EXTPolicyEvents, EXTSelector, EXTDonorSeed);
         updated = selected >= 0 && DPCCUpdateAction6Actor(*EXTActor,
                   EXTPendingDecision.decision_id, (uint)selected, reason);
         if(updated)
            EXTPolicyEvents++;
        }
      else
         updated = DPCCOnlineCancelActor(EXTActor, reason);
     }
   if(updated)
     {
      EXTTransitions++;
      updated = DPCCOnlineRefreshTargets(reason);
     }
   string restore_reason = "";
   const bool restored = DPCCOnlineMarketState(EXTNextMarketState, false, 0, 0, restore_reason);
   if(!restored)
      reason += ";next_state_restore:" + restore_reason;
   EXTNextMarketState.Clear();
   if(!updated || !restored)
     {
      EXTMutationFailed = true;
      return(false);
     }
   EXTPendingMarketState.Clear();
   EXTPending = false;
   EXTPendingActorFrame = false;
   ZeroMemory(EXTPendingDecision);
   return(true);
  }
//+------------------------------------------------------------------+
//| Execute one causal operation and retain its Actor training frame |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!EXTReady || EXTStopped || IsStopped())
      return;
   const double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   string reason = "";
   if(!MathIsValidNumber(balance) || !MathIsValidNumber(equity))
     {
      DPCCOnlineStop("account", "actual_balance_or_equity_nonfinite");
      return;
     }
   if(balance < EXTMinBalance || equity < EXTMinBalance)
     {
      EXTStopped = true;
      EXTControlledFloorStop = true;
      Print("DPCC_STUDYONLINE_CONTROLLED_STOP phase=account_floor reason=actual_Tester_balance_or_equity_below_floor",
            " transitions=", EXTTransitions, " actor_updates=", EXTPolicyEvents);
      TesterStop();
      return;
     }
   const long open = (long)iTime(_Symbol, EXTControl.timeframe, 0);
   if(open <= 0 || open == EXTBarOpen)
      return;
   if(iVolume(_Symbol, EXTControl.timeframe, 0) != 1 ||
      (EXTBarOpen > 0 && (long)iTime(_Symbol, EXTControl.timeframe, 1) != EXTBarOpen))
     {
      DPCCOnlineStop("chronology", "first_H1_quote_required_no_hindsight_repair");
      return;
     }
   if(!DPCCTesterOnly(reason) || !DPCCRequireTesterOneDirection(reason))
     {
      DPCCOnlineStop("environment", reason);
      return;
     }
   const ulong started = GetMicrosecondCount();
   CBufferFloat *context = NULL;
   float account[], action[];
   if(!DPCCOnlineObserve(open, context, account, reason))
     {
      DPCCOnlineStop("causal_observation", reason);
      return;
     }
   if(!DPCCOnlineUpdatePending(context, open, account, equity, reason))
     {
      DPCCOnlineStop("one_bar_TD", reason);
      return;
     }
   SDPCCDecisionMeta meta;
   bool has_plan = false;
   if(!DPCCOnlineActorForward(*EXTActor, EXTActorAccount, context, open, account, action, meta, has_plan, reason))
     {
      DPCCOnlineStop("Action6_forward", reason);
      return;
     }
   if(!has_plan)
     {
      if(!DPCCOnlineCancelActor(EXTActor, reason))
        {
         DPCCOnlineStop("Actor_frame_release", reason);
         return;
        }
      EXTActorNoPlan++;
     }
   if(!DPCCOnlineMarketState(EXTPendingMarketState, true, meta.decision_id,
                             EXTPendingMarketGeneration, reason))
     {
      DPCCOnlineStop("pending_snapshot", reason);
      return;
     }
   for(int i = 0; i < 13; i++)
      EXTPendingAccount[i] = account[i];
   for(int i = 0; i < 6; i++)
      EXTPendingAction[i] = action[i];
   EXTPendingDecision = meta;
   EXTPendingActorFrame = has_plan;
   EXTMarkBalance = balance;
   EXTMarkEquity = equity;
   EXTPending = true;
   if(!DPCCExecuteTesterAction(_Symbol, InpMagic, action, (double)MaxTP, (double)MaxSL, reason))
     {
      EXTMutationFailed = true;
      DPCCOnlineStop("Tester_execution", reason);
      return;
     }
   EXTBarOpen = open;
   EXTDecisions++;
   if(!DPCCOnlineReleaseMarket(reason))
     {
      DPCCOnlineStop("Market_release", reason);
      return;
     }
   if(InpMeasureLatency)
     {
      const ulong elapsed = GetMicrosecondCount() - started;
      EXTLatencySamples++;
      EXTDecisionLatencyUs += elapsed;
      EXTDecisionLatencyMaxUs = MathMax(EXTDecisionLatencyMaxUs, elapsed);
     }
  }
//+------------------------------------------------------------------+
//| Mark a normal completed Tester history pass for final export     |
//+------------------------------------------------------------------+
double OnTester(void)
  {
   if(EXTReady && !EXTStopped && !EXTFailed && !EXTMutationFailed && EXTTransitions > 0)
      EXTTrainingComplete = true;
   return(0.0);
  }
//+------------------------------------------------------------------+
//| Release frames, save the last complete models once, then destroy |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   bool cleanup_ok = true;
   const bool pending_discarded = EXTPending;
   string release_reason = "";
   if(CheckPointer(EXTActor) != POINTER_INVALID && !DPCCOnlineCancelActor(EXTActor, release_reason))
      cleanup_ok = false;
   if(CheckPointer(EXTTargetActor) != POINTER_INVALID && !DPCCOnlineCancelActor(EXTTargetActor, release_reason))
      cleanup_ok = false;
   if(!DPCCOnlineReleaseMarket(release_reason))
      cleanup_ok = false;
   EXTPendingMarketState.Clear();
   EXTNextMarketState.Clear();
   EXTPending = false;
   EXTPendingActorFrame = false;
   ZeroMemory(EXTPendingDecision);
   if(EXTReady && cleanup_ok && !EXTFailed && !EXTMutationFailed && EXTTransitions > 0 &&
      (EXTTrainingComplete || EXTControlledFloorStop))
     {
      EXTModelSaved = DPCCSaveAction6OnlineTrading(EXTControl, DPCC_COMMON_FILES, EXTMarket,
                      *EXTActor, *EXTQ1, *EXTQ2, *EXTTargetActor, *EXTTargetQ1, *EXTTargetQ2,
                      EXTRestored, EXTControlledFloorStop, EXTModelSaveAttempted,
                      EXTModelSavePath, EXTModelSaveSHA, EXTModelSaveReason);
      if(!EXTModelSaved && EXTModelSaveReason == "")
         EXTModelSaveReason = "online.model_export_failed_without_reason";
     }
   else
      if(!cleanup_ok)
         EXTModelSaveReason = "online.cleanup_failed:" + release_reason;
      else
         if(EXTMutationFailed || EXTFailed)
            EXTModelSaveReason = "online.failed_or_partial_model_update";
         else
            if(!EXTTrainingComplete && !EXTControlledFloorStop)
               EXTModelSaveReason = "online.training_not_complete";
            else
               if(EXTTransitions == 0)
                  EXTModelSaveReason = "online.no_complete_transition";
   EXTDecisionContext.Reset();
   DPCCReleaseAction6Trading(EXTActor, EXTQ1, EXTQ2, EXTTargetActor, EXTTargetQ1, EXTTargetQ2);
   for(int i = 0; i < 4; i++)
      if(EXTIndicators[i] != INVALID_HANDLE)
         IndicatorRelease(EXTIndicators[i]);
   if(EXTReady)
      Print("DPCC_STUDYONLINE_FINAL transitions=", EXTTransitions, " actor_updates=", EXTPolicyEvents,
            " target_updates=", EXTTargetUpdates, " actor_no_plan=", EXTActorNoPlan,
            " target_actor_no_plan=", EXTTargetActorNoPlan, " mutation_failed=", EXTMutationFailed,
            " controlled_floor=", EXTControlledFloorStop,
            " pending_transition_discarded=", pending_discarded,
            " model_export=", (EXTModelSaved ? "PASS" : (EXTModelSaveAttempted ? "FAIL" : "SKIPPED")),
            " path=", EXTModelSavePath, " sha256=", EXTModelSaveSHA,
            " reason=", EXTModelSaveReason, " deinit_reason=", reason);
   if(InpMeasureLatency && EXTLatencySamples > 0)
      PrintFormat("DPCC_STUDYONLINE_DECISION_LATENCY samples=%I64u mean_us=%.2f max_us=%I64u",
                  EXTLatencySamples,
                  (double)EXTDecisionLatencyUs / (double)EXTLatencySamples,
                  EXTDecisionLatencyMaxUs);
  }
//+------------------------------------------------------------------+