//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property version   "3.00"
#property description "Evaluates a frozen DPCC Action6 policy in Strategy Tester"
#property description "Uses the chart symbol, configured timeframe, and current account context"
#property description "Runs across the configured tester interval"
#property strict

#define DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST
#include "Trajectory.mqh"
#undef DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST

input group                 "---- Таймфрейм данных ----"

input ENUM_TIMEFRAMES        InpTimeframe = PERIOD_CURRENT;    // Таймфрейм; PERIOD_CURRENT — таймфрейм графика/тестера.

input group                 "---- Режим оценки и исполнение ----"
input ENUM_DPCC_MODE         InpMode = DPCC_P0;    // Режим DPCC: P0, P1 или P2.
input long                   InpMagic = 33062;     // Magic-номер заявок этого советника.

input group                 "---- Замер производительности ----"
input bool                   InpMeasureLatency = false;    // Измерять задержку принятия решения.

const string DPCC_PACKAGE_ROOT = "DPCC/StudyForecast_M05_20261017";
const bool   DPCC_COMMON_FILES = true;
//+------------------------------------------------------------------+
//| Frozen policy owners used throughout the Tester interval         |
//+------------------------------------------------------------------+
CDPCCStudyForecastCore        EXTMarket;
CDPCCAction6ChartContextOwner EXTDecisionContext;
CNet                        *EXTActor = NULL;
CNet                        *EXTQ1 = NULL;
CNet                        *EXTQ2 = NULL;
SDPCCExecutionControl        EXTControl;
SDPCCRuntimeConfig           EXTRuntime;
float                        EXTTradingLimits[];
CBufferFloat                 EXTCurrentZ;
CBufferFloat                 EXTActorAccount;
int                          EXTIndicators[4];
long                         EXTBarOpen = 0;
long                         EXTFirstDecision = 0;
double                       EXTInitialBalance = 0;
double                       EXTInitialEquity = 0;
uint                         EXTDecisions = 0;
ulong                        EXTNoiseSeed = 0;
double                       EXTTPCapPoints = 0;
double                       EXTSLCapPoints = 0;
bool                         EXTReady = false;
bool                         EXTStopped = false;
bool                         EXTWindowCompleted = false;
string                       EXTMarketSHA = "";
string                       EXTTradingSHA = "";
ulong                        EXTLatencySamples = 0;
ulong                        EXTMarketLatencyUs = 0;
ulong                        EXTActorLatencyUs = 0;
ulong                        EXTExecutionLatencyUs = 0;
ulong                        EXTDecisionLatencyUs = 0;
ulong                        EXTDecisionLatencyMaxUs = 0;
//+------------------------------------------------------------------+
//| Stop before another decision or execution request                |
//+------------------------------------------------------------------+
void DPCCTestStop(const string phase, const string reason)
  {
   EXTStopped = true;
   Print("DPCC_TEST_BLOCKED phase=", phase, " reason=", reason, " bar=", EXTBarOpen);
   if(MQLInfoInteger(MQL_TESTER))
      TesterStop();
  }
//+------------------------------------------------------------------+
//| Shared D01 restore boundary for Market and Trading inference     |
//| roles                                                            |
//+------------------------------------------------------------------+
bool DPCCTestRestoreFrozenTrading(string &reason)
  {
   CNet *target_actor = NULL;
   CNet *target_q1 = NULL;
   CNet *target_q2 = NULL;
   SDPCCAction6TradingRestore restored;
   if(!DPCCLoadAction6Trading(EXTControl, DPCC_COMMON_FILES, EXTMarket, EXTActor, EXTQ1, EXTQ2,
                              target_actor, target_q1, target_q2, restored, reason))
      return(false);
//--- Test keeps only the frozen online policy and its ranking dependencies.
   delete target_actor;
   delete target_q1;
   delete target_q2;
   EXTRuntime = restored.runtime;
   EXTMarketSHA = restored.market_sha256;
   EXTTradingSHA = restored.trading_sha256;
   EXTNoiseSeed = restored.diffusion_seed;
   EXTTPCapPoints = MaxTP;
   EXTSLCapPoints = MaxSL;
   if(ArrayCopy(EXTTradingLimits, restored.limits) != 24)
     {
      reason = "test.restore.financial_limits_capacity";
      return(false);
     }
   if(EXTNoiseSeed == 0)
     {
      reason = "BLOCKED:test.assigned_profile.diffusion.seed_must_be_nonzero_for_Action6";
      return(false);
     }
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Shared physical-account and causal financial-context boundary    |
//+------------------------------------------------------------------+
bool DPCCTestDecisionContext(const long decision_open, CBufferFloat *native_ohlc,
                             CBufferFloat *&context, float &account13[], string &reason)
  {
   return(EXTDecisionContext.Build(EXTRuntime, EXTTradingLimits, native_ohlc,
                                   decision_open, context, account13, reason));
  }
//+------------------------------------------------------------------+
//| Verify frozen Market, Actor and the two inference-only critics   |
//+------------------------------------------------------------------+
bool DPCCTestFrozenPolicy(string &reason)
  {
   CNet *market = EXTMarket.ReadyMarketNetwork();
   CNet *networks[] = {market, EXTActor, EXTQ1, EXTQ2};
   const int layers[] = {8, 2, 5, 5};
   COpenCLMy *context = EXTMarket.ReadyEncoderContext();
   if(CheckPointer(context) == POINTER_INVALID)
     {
      reason = "test.frozen.market_context";
      return(false);
     }
   for(int i = 0; i < ArraySize(networks); i++)
     {
      if(CheckPointer(networks[i]) == POINTER_INVALID || !networks[i].IsOpenCLTransferReady() ||
         networks[i].GetOpenCL() != context || networks[i].WeightsUpdateEnabled() ||
         networks[i].Layer(layers[i]) != NULL)
        {
         reason = StringFormat("test.frozen.graph_or_permission.%d", i);
         return(false);
        }
      for(int j = 0; j < i; j++)
         if(networks[i] == networks[j])
           {
            reason = "test.frozen.network_alias";
            return(false);
           }
      for(int l = 0; l < layers[i]; l++)
        {
         CNeuronBaseOCL *layer = networks[i].Layer(l);
         if(CheckPointer(layer) == POINTER_INVALID || layer.GetOpenCL() != context || layer.TrainMode())
           {
            reason = StringFormat("test.frozen.layer.%d.%d", i, l);
            return(false);
           }
        }
     }
   CNeuronDPCCOCL *actor = dynamic_cast<CNeuronDPCCOCL*>(EXTActor.Layer(1));
   if(!actor || actor.Type() != defNeuronDPCCOCL || actor.ActorABI() != 3 ||
      !NNQCriticGraph(EXTQ1) || !NNQCriticGraph(EXTQ2))
     {
      reason = "test.frozen.Action6_graph";
      return(false);
     }
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocate the live Market Z snapshot from its actual frozen axes  |
//+------------------------------------------------------------------+
bool DPCCTestPrepareMarketSnapshot(string &reason)
  {
   CNet *market = EXTMarket.ReadyMarketNetwork();
   COpenCLMy *context = EXTMarket.ReadyEncoderContext();
   CNeuronScenarioForecastPriceOCL *forecast = NULL;
   if(CheckPointer(market) != POINTER_INVALID)
      forecast = dynamic_cast<CNeuronScenarioForecastPriceOCL*>(market.Layer(6));
   ulong count = 0;
   if(!forecast || CheckPointer(context) == POINTER_INVALID || forecast.GetOpenCL() != context ||
      forecast.Type() != defNeuronScenarioForecastPriceOCL || forecast.CodebookUpdate() ||
      !DPCCRolloutProduct(forecast.Scenarios(), forecast.Variables(), count) ||
      !DPCCRolloutProduct(count, forecast.Horizon(), count) ||
      !DPCCRolloutProduct(count, forecast.Dimension(), count) || count == 0 ||
      count > (ulong)INT_MAX || count > (ulong)UINT_MAX / sizeof(float))
     {
      reason = "test.market.snapshot_axes_or_frozen_contract";
      return(false);
     }
   if(!EXTCurrentZ.BufferInit((uint)count, 0) || !EXTCurrentZ.BufferCreate(context))
     {
      reason = "test.market.snapshot_allocation";
      return(false);
     }
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Close the Actor frame before releasing its Market generation     |
//+------------------------------------------------------------------+
bool DPCCTestReleaseGeneration(const ulong generation, string &reason)
  {
   if(generation == 0)
     {
      reason = "test.market.zero_generation";
      return(false);
     }
   CNeuronDPCCOCL *actor = CheckPointer(EXTActor) == POINTER_INVALID ? NULL :
                           dynamic_cast<CNeuronDPCCOCL*>(EXTActor.Layer(1));
   if(!actor || !actor.CancelGeneration())
     {
      reason = "test.Actor_frame_release";
      return(false);
     }
   return(EXTMarket.ReleaseReadyEncoderZ(generation, reason));
  }
//+------------------------------------------------------------------+
//| Initialize the assigned frozen policy for February 2026          |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   for(int i = 0; i < 4; i++)
      EXTIndicators[i] = INVALID_HANDLE;
   string reason = "";
//--- Configure the package and load its embedded Market profile before restoration.
   if(!DPCCConfigureExecutionControl(DPCC_PACKAGE_ROOT, InpTimeframe, DPCC_STAGE_TEST, "",
                                     0, DPCC_RESUME_NEW, InpMode, DPCC_STAGE_TEST, EXTControl, reason) ||
      !DPCCLoadMarketExecutionProfile(EXTControl, DPCC_COMMON_FILES, reason))
     {
      DPCCTestStop("execution_control", reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpMagic < 1)
     {
      DPCCTestStop("profile_control", "H1_frozen_Test_without_update_budget_required");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(!DPCCTesterOnly(reason) || !DPCCRequireTesterOneDirection(reason))
     {
      DPCCTestStop("environment", reason);
      return(INIT_FAILED);
     }
   if(_Symbol == "" || AccountInfoString(ACCOUNT_CURRENCY) == "" ||
      PositionsTotal() != 0 || OrdersTotal() != 0)
     {
      DPCCTestStop("environment", "flat_owned_Tester_start_required");
      return(INIT_FAILED);
     }
//--- D01 role validation and shared-context restoration belong to the shared loader.
   if(!DPCCTestRestoreFrozenTrading(reason))
     {
      DPCCTestStop("D01_frozen_restore", reason);
      return(INIT_FAILED);
     }
   if(!DPCCResolverLowerSha256(EXTMarketSHA) || !DPCCResolverLowerSha256(EXTTradingSHA) ||
      EXTNoiseSeed == 0 || !MathIsValidNumber(EXTTPCapPoints) || EXTTPCapPoints <= 0 ||
      !MathIsValidNumber(EXTSLCapPoints) || EXTSLCapPoints <= 0 ||
      EXTRuntime.financial.mode != EXTControl.mode || !DPCCTestFrozenPolicy(reason))
     {
      DPCCTestStop("frozen_policy", reason == "" ? "authenticated_policy_contract_required" : reason);
      return(INIT_FAILED);
     }
   CNeuronDPCCOCL *actor = dynamic_cast<CNeuronDPCCOCL*>(EXTActor.Layer(1));
   if(!actor.BindPolicyNetworks(EXTMarket.ReadyMarketNetwork(), EXTQ1, EXTQ2))
     {
      DPCCTestStop("frozen_policy", "Actor_inference_dependencies");
      return(INIT_FAILED);
     }
   if(!DPCCTestPrepareMarketSnapshot(reason))
     {
      DPCCTestStop("market_snapshot", reason);
      return(INIT_FAILED);
     }
   if(!EXTActorAccount.BufferInit(13, 0) ||
      !EXTActorAccount.BufferCreate(EXTMarket.ReadyEncoderContext()))
     {
      DPCCTestStop("account_input", "persistent_Account13_buffer");
      return(INIT_FAILED);
     }
   EXTIndicators[0] = iRSI(_Symbol, EXTControl.timeframe, RSIPeriod, RSIPrice);
   EXTIndicators[1] = iCCI(_Symbol, EXTControl.timeframe, CCIPeriod, CCIPrice);
   EXTIndicators[2] = iATR(_Symbol, EXTControl.timeframe, ATRPeriod);
   EXTIndicators[3] = iMACD(_Symbol, EXTControl.timeframe, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice);
   for(int i = 0; i < 4; i++)
      if(EXTIndicators[i] == INVALID_HANDLE)
        {
         DPCCTestStop("indicators", (string)GetLastError());
         return(INIT_FAILED);
        }
   EXTInitialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   EXTInitialEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(!MathIsValidNumber(EXTInitialBalance) || !MathIsValidNumber(EXTInitialEquity) ||
      EXTInitialBalance < 50 || EXTInitialEquity < 50)
     {
      DPCCTestStop("account", "actual_Tester_start_below_floor");
      return(INIT_FAILED);
     }
   EXTReady = true;
   Print("DPCC_TEST_READY policy=FROZEN_ACTION6 Q1_Q2=FROZEN_INFERENCE_ONLY tester_interval=FULL",
         " market_sha256=", EXTMarketSHA,
         " trading_sha256=", EXTTradingSHA, " profile_sha256=", EXTControl.profile_sha256);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Evaluate the first quote of each H1 bar in the Tester interval   |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!EXTReady || EXTStopped || IsStopped())
      return;
   const long open = (long)iTime(_Symbol, EXTControl.timeframe, 0);
   if(open <= 0 || open == EXTBarOpen)
      return;
   string reason = "";
   if(!DPCCRequireTesterOneDirection(reason))
     {
      DPCCTestStop("environment", reason);
      return;
     }
   if(iVolume(_Symbol, EXTControl.timeframe, 0) != 1 ||
      (EXTBarOpen > 0 && (long)iTime(_Symbol, EXTControl.timeframe, 1) != EXTBarOpen))
     {
      DPCCTestStop("chronology", "first_H1_quote_required_no_hindsight_repair");
      return;
     }
   const double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(!MathIsValidNumber(balance) || !MathIsValidNumber(equity) || balance < 50 || equity < 50)
     {
      DPCCTestStop("floor", "actual_balance_or_equity_below_50");
      return;
     }
//--- Capture the closed predecessor and only the observed first quote of its successor.
   const ulong decision_started = GetMicrosecondCount();
   SDPCCClosedHistoryRow closed;
   MqlTick tick;
   long quote_mode = -1;
   double tick_size = 0;
   if(!DPCCCaptureTesterClosedRow(_Symbol, EXTIndicators, 1, closed, reason) ||
      closed.successor_open_server != open || !SymbolInfoTick(_Symbol, tick) ||
      tick.time_msc < open * 1000 || tick.time_msc >= (open + 3600) * 1000 ||
      !SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE, quote_mode) ||
      !SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE, tick_size) ||
      (quote_mode != SYMBOL_CHART_MODE_BID && quote_mode != SYMBOL_CHART_MODE_LAST))
     {
      DPCCTestStop("closed_history", reason == "" ? "causal_H1_quote_binding" : reason);
      return;
     }
//--- Replay pre-start chart history before the first policy decision.
   if(EXTDecisions == 0 &&
      !DPCCPrepareReadyMarketLiveStart(EXTMarket, _Symbol, EXTIndicators,
                                       closed.bar.server_open, reason))
     {
      DPCCTestStop("market_warmup", reason);
      return;
     }
   const double first_quote = (quote_mode == SYMBOL_CHART_MODE_LAST ? tick.last : tick.bid);
   ulong generation = 0;
   const ulong market_started = GetMicrosecondCount();
   if(!EXTMarket.InferReadyMarketLive(closed, first_quote, tick.time_msc, tick_size,
                                      GetPointer(EXTCurrentZ), generation, reason))
     {
      DPCCTestStop("market_inference", reason);
      return;
     }
   const ulong market_us = GetMicrosecondCount() - market_started;
//--- Build the physical context without future bars and evaluate the frozen policy.
   CBufferFloat *context = NULL;
   float account[], action[];
   bool prepared = DPCCTestDecisionContext(open, EXTMarket.ReadyMarketNativeOHLC(), context, account, reason);
   if(prepared)
      prepared = (ArraySize(account) == 13 && CheckPointer(context) != POINTER_INVALID);
   if(prepared)
      for(int i = 0; i < 13; i++)
         if(!MathIsValidNumber(account[i]) || !EXTActorAccount.Update(i, account[i]))
           {
            reason = "test.Account13_input";
            prepared = false;
            break;
           }
   CNeuronDPCCOCL *actor = dynamic_cast<CNeuronDPCCOCL*>(EXTActor.Layer(1));
   SDPCCDecisionMeta meta;
   ZeroMemory(meta);
   meta.decision_id = (ulong)EXTDecisions + 1;
   meta.market_state_id = generation;
   meta.q1_state_id = meta.decision_id;
   meta.q2_state_id = meta.decision_id + 1;
   meta.noise_seed = EXTNoiseSeed;
   meta.decision_server_time = open;
   meta.bar_index = (long)EXTDecisions;
   meta.mode = (uint)EXTControl.mode;
   meta.market_sha256 = EXTMarketSHA;
   meta.profile_sha256 = EXTControl.profile_sha256;
   const ulong actor_started = GetMicrosecondCount();
   if(prepared)
      prepared = EXTActorAccount.BufferWrite() && actor.BindDecisionContext(context, meta);
   bool actor_forwarded = false;
   if(prepared)
      actor_forwarded = EXTActor.feedForward(GetPointer(EXTActorAccount), 1, false,
                                             EXTMarket.ReadyMarketNetwork(), 6);
   if(prepared)
     {
      string action_reason = "";
      prepared = DPCCResolveAction6Output(actor, actor_forwarded, action, action_reason);
      if(!prepared)
         reason = "test." + action_reason;
     }
   if(!prepared && reason == "")
      reason = "test.Action6_forward_or_output";
   const ulong actor_us = GetMicrosecondCount() - actor_started;
   string release_reason = "";
   const bool released = DPCCTestReleaseGeneration(generation, release_reason);
   if(!prepared || !released)
     {
      DPCCTestStop("policy_generation", !released ? "release:" + release_reason + ";" + reason : reason);
      return;
     }
//--- Execute only the published first Action6 through the existing Tester adapter.
   const ulong execution_started = GetMicrosecondCount();
   if(!DPCCExecuteTesterAction(_Symbol, InpMagic, action, EXTTPCapPoints, EXTSLCapPoints, reason))
     {
      DPCCTestStop("Tester_execution", reason);
      return;
     }
   const ulong execution_us = GetMicrosecondCount() - execution_started;
   const ulong decision_us = GetMicrosecondCount() - decision_started;
   EXTBarOpen = open;
   if(EXTFirstDecision == 0)
      EXTFirstDecision = open;
   EXTDecisions++;
   if(InpMeasureLatency)
     {
      EXTLatencySamples++;
      EXTMarketLatencyUs += market_us;
      EXTActorLatencyUs += actor_us;
      EXTExecutionLatencyUs += execution_us;
      EXTDecisionLatencyUs += decision_us;
      EXTDecisionLatencyMaxUs = MathMax(EXTDecisionLatencyMaxUs, decision_us);
     }
  }
//+------------------------------------------------------------------+
//| Mark normal completion of the Strategy Tester interval           |
//+------------------------------------------------------------------+
double OnTester(void)
  {
   if(EXTReady && !EXTStopped)
      EXTWindowCompleted = true;
   return(0.0);
  }
//+------------------------------------------------------------------+
//| Report observed results and release policy borrowers first       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(EXTReady)
      Print("DPCC_TEST_RESULT decisions=", EXTDecisions, " first_server_open=", EXTFirstDecision,
            " window_completed=", EXTWindowCompleted, " stopped=", EXTStopped,
            " initial_balance=", DoubleToString(EXTInitialBalance, 2),
            " final_balance=", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
            " initial_equity=", DoubleToString(EXTInitialEquity, 2),
            " final_equity=", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
            " market_sha256=", EXTMarketSHA, " trading_sha256=", EXTTradingSHA, " deinit_reason=", reason);
   if(InpMeasureLatency && EXTLatencySamples > 0)
      PrintFormat("DPCC_TEST_LATENCY samples=%I64u market_mean_us=%.2f " +
                  "actor_mean_us=%.2f execution_mean_us=%.2f " +
                  "total_mean_us=%.2f total_max_us=%I64u",
                  EXTLatencySamples, (double)EXTMarketLatencyUs / (double)EXTLatencySamples,
                  (double)EXTActorLatencyUs / (double)EXTLatencySamples,
                  (double)EXTExecutionLatencyUs / (double)EXTLatencySamples,
                  (double)EXTDecisionLatencyUs / (double)EXTLatencySamples, EXTDecisionLatencyMaxUs);
   EXTDecisionContext.Reset();
   if(CheckPointer(EXTActor) != POINTER_INVALID)
      delete EXTActor;
   EXTActor = NULL;
   if(CheckPointer(EXTQ2) != POINTER_INVALID)
      delete EXTQ2;
   EXTQ2 = NULL;
   if(CheckPointer(EXTQ1) != POINTER_INVALID)
      delete EXTQ1;
   EXTQ1 = NULL;
   for(int i = 0; i < 4; i++)
      if(EXTIndicators[i] != INVALID_HANDLE)
         IndicatorRelease(EXTIndicators[i]);
  }
//+------------------------------------------------------------------+