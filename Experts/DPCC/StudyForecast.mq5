//+------------------------------------------------------------------+
//|                                                StudyForecast.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property version   "1.00"
#property description "DPCC Market representation training in four selectable phases"
#property description "Uses terminal history to train the latent and price objectives"
#property description "Saves the latest Market model checkpoint in Common Files"
#property strict
#define DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST
#include "Trajectory.mqh"
#undef DPCC_TRAJECTORY_ENTRY_STUDY_FORECAST

input group "---- Источник и общие параметры запуска ----"
//--- Таймфрейм исторических баров; PERIOD_CURRENT — таймфрейм графика.
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_CURRENT;
//--- Максимум обновлений за запуск; 0 — без дополнительного ограничения.
input ulong InpStopAfterUpdates = 0;
//--- AUTO продолжает checkpoint; явное имя фазы запускает её на текущих весах.
input string InpM05TrainingPhase = "AUTO";
//--- Включить режим обучения Market по прямым параметрам M05.
input bool InpUseM05 = true;
input ENUM_DPCC_RESUME_MODE InpResumeMode = DPCC_RESUME_UNASSIGNED;    // Режим восстановления сохранённой модели.
input ENUM_DPCC_MODE InpMode = DPCC_P0;                                // Режим DPCC: P0, P1 или P2.
//--- 0 — весь доступный набор; больше 0 — указанное число обновлений.
input uint InpMaxUpdates = 0;

input group "---- Дополнительные режимы и параметры ценовой ошибки ----"
input bool InpPriceForwardOnly = false;     // Исторический H1 forward без обучения и сохранения моделей.
input bool InpPriceJoint = false;           // Совместная латентная и ценовая задача Market.
input float InpPriceLambda = 0.0f;          // Вес ценовой ошибки в дополнительном режиме.
input float InpPriceHuberDelta = 0.0f;      // Порог функции Huber для ошибки цены.
input float InpPriceRhoTolerance = 0.0f;    // Допуск суммы вероятностей rho.

//input group "---- Идентификатор и файлы задачи M05 ----"
/*input*/ string InpM05TaskId = "DPCC_M05_StudyForecast_20261017";    // Идентификатор запуска M05.
/*input*/ ulong InpM05NoiseSeed = 20261007;                           // Seed для воспроизводимой случайной части обучения.
//--- Максимальный размер читаемого или записываемого файла.
/*input*/ ulong InpM05MaxFileBytes = 1073741824;

input group "---- LATENT_PREPARE: период и число эпох ----"
input datetime InpM05LatentStart = D'2024.01.01';    // Время сервера; начало окна LATENT_PREPARE включительно.
input datetime InpM05LatentEnd = D'2026.01.01';      // Время сервера; исключительная граница окна LATENT_PREPARE.
input uint InpM05LatentEpochs = 100;                 // Число эпох обучения LATENT_PREPARE.

input group "---- OMPB_CALIBRATE: окна reference и calibration ----"
input datetime InpM05ReferenceStart = D'2024.01.01';      // Время сервера; начало reference-окна OMPB включительно.
input datetime InpM05ReferenceEnd = D'2024.10.01';        // Время сервера; исключительная граница reference-окна.
input datetime InpM05CalibrationStart = D'2024.10.01';    // Время сервера; начало calibration-окна включительно.
input datetime InpM05CalibrationEnd = D'2026.01.01';      // Время сервера; исключительная граница calibration-окна.
input uint InpM05OMPBTrainingEpochs = 30;                 // Число эпох калибровки OMPB.
input uint InpM05SourceAnchorPeriod = 10;                 // Интервал между исходными якорями OMPB.

input group "---- OMPB_CALIBRATE: параметры распределительного обучения ----"
input float InpM05Tau = 1.0f;              // Параметр tau калибровки OMPB.
input float InpM05LambdaDis = 0.01f;       // Вес штрафа за disagreement OMPB.
input float InpM05LambdaKL = 0.01f;        // Вес штрафа KL-дивергенции OMPB.
input float InpM05LambdaAlpha = 0.001f;    // Вес регуляризации alpha OMPB.
input float InpM05AlphaPrior = -6.0f;      // Исходное значение alpha prior OMPB.
input float InpM05MaxKL = 0.0f;            // Максимальный KL для калибровки OMPB.

input group "---- PRICE_JOINT и PRICE_ADAPT: цена и бюджет обучения ----"
input float InpM05PriceLambda = 1.0f;            // Вес ценовой ошибки Market.
input float InpM05PriceHuberDelta = 1.0f;        // Порог функции Huber в нормированной шкале цены.
input float InpM05PriceRhoTolerance = 0.001f;    // Допуск отклонения суммы rho от 1.
input uint InpM05PriceJointEpochs = 1;           // Число эпох PRICE_JOINT.
input ulong InpM05PriceJointMaxUpdates = 0;      // Лимит обновлений PRICE_JOINT; 0 — без отдельного лимита.
input uint InpM05PriceAdaptEpochs = 1;           // Число эпох PRICE_ADAPT.
input ulong InpM05PriceAdaptMaxUpdates = 0;      // Лимит обновлений PRICE_ADAPT; 0 — без отдельного лимита.

input group "---- Пороговые значения VERIFY ----"
input double InpM05VerifyAtol = 0.000001;               // Абсолютный допуск сравнения прогнозов.
input double InpM05VerifyRtol = 0.000001;               // Относительный допуск сравнения прогнозов.
input double InpM05VerifyMaxPriceError = 1.0;           // Максимальная допустимая ошибка цены.
input double InpM05VerifyMinScenarioCoverage = 0.80;    // Минимальная доля целевых точек, покрытых сценариями.
input uint InpM05VerifyMinDistinctScenarios = 2;        // Минимум различных сценариев из набора модели.

const string DPCC_M05_TASK_DIRECTORY = "DPCC/StudyForecast_M05_20261017";
const bool   DPCC_COMMON_FILES = true;

CDPCCStudyForecastCore ExtTrainer;
string                  ExtRoot;
ulong                   ExtLastProgressMilestone = 0;
SDPCCExecutionControl   ExtExecutionControl;
SDPCCM05OMPBProfile      EXTM05DonorProfile;
SDPCCM05VerifyProfile    EXTM05VerifyProfile;
float                   EXTM05PriceLambda = 0, EXTM05HuberDelta = 0, EXTM05RhoTolerance = 0;
string                  EXTM05ProfileSHA = "", EXTM05DonorSHA = "";
CDPCCStopButton         ExtStopButton;
bool                    ExtM05StopSaveFailed = false;
//+------------------------------------------------------------------+
//| M05Enabled selects the direct-input M05 Market training path     |
//+------------------------------------------------------------------+
bool M05Enabled(void)
  {
   return(InpUseM05);
  }
//+------------------------------------------------------------------+
//| Persist one whole phase boundary without a second OnDeinit save  |
//+------------------------------------------------------------------+
bool SaveM05Checkpoint(const string cause)
  {
   string path = "", reason = "", publication_reason = "";
   const bool saved = ExtTrainer.SaveM05Boundary(ExtExecutionControl, cause, path, reason);
   const bool consumer_ready = ExtTrainer.MarketConsumerReady(publication_reason);
   Print("DPCC_MARKET_CHECKPOINT cause=", cause, " stage=", ExtTrainer.PriceStudyStage(),
         " updates=", ExtTrainer.Updates(), " model_save=", (saved ? 1 : 0), " path=", path,
         " ompb_epochs=", ExtTrainer.OMPBTrainingEpochsCompleted(), "/", ExtTrainer.OMPBTrainingEpochLimit(),
         " ompb_budget=", (ExtTrainer.OMPBTrainingBudgetComplete() ? "COMPLETE" : "PARTIAL"),
         " ompb_eval_lanes=", ExtTrainer.OMPBStage02EvaluationLanes(), "/6",
         " ompb_quality=", (!ExtTrainer.OMPBStage02QualityEvaluated() ? "PENDING" :
                            (ExtTrainer.OMPBStage02QualityPassed() ? "PASS" : "FAIL")),
         " quality_gate=ADVISORY",
         " consumer_ready=", consumer_ready, " technical_acceptance=NOT_RUN",
         " reason=", reason, " consumer_reason=", publication_reason);
   return(saved);
  }
//+------------------------------------------------------------------+
//| Function OnInit                                                  |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   string control_reason = "";
   if(!DPCCConfigureExecutionControl(DPCC_M05_TASK_DIRECTORY, InpTimeframe, DPCC_STAGE_STUDYFORECAST, "",
                                     InpStopAfterUpdates, InpResumeMode, InpMode, DPCC_STAGE_STUDYFORECAST,
                                     ExtExecutionControl, control_reason))
     {
      Print("DPCC_STUDYFORECAST_BLOCKED phase=profile_control reason=", control_reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   const bool use_m05 = M05Enabled();
   if(InpM05TrainingPhase != "AUTO" && InpM05TrainingPhase != "LATENT_PREPARE" &&
      InpM05TrainingPhase != "PRICE_JOINT" && InpM05TrainingPhase != "OMPB_CALIBRATE" &&
      InpM05TrainingPhase != "PRICE_ADAPT")
     {
      Print("DPCC_M05_ABORT phase=parameters reason=unknown_training_phase value=", InpM05TrainingPhase);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpM05TrainingPhase != "AUTO" && !use_m05)
     {
      Print("DPCC_M05_ABORT phase=parameters reason=explicit_phase_requires_M05_inputs");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if((InpPriceJoint && InpPriceForwardOnly) || (use_m05 && (InpPriceJoint || InpPriceForwardOnly)))
      return(INIT_PARAMETERS_INCORRECT);
   if(InpPriceJoint &&
      (!MathIsValidNumber(InpPriceLambda) || InpPriceLambda <= 0 ||
       !MathIsValidNumber(InpPriceHuberDelta) || InpPriceHuberDelta <= 0 ||
       !MathIsValidNumber(InpPriceRhoTolerance) || InpPriceRhoTolerance <= 0 ||
       InpPriceRhoTolerance >= 1))
     {
      Print("DPCC_PRICE_JOINT_ABORT phase=parameters reason=positive_loss_profile_required");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(use_m05)
     {
      string profile_reason = "";
      SDPCCM05DirectInputs direct;
      ZeroMemory(direct);
      direct.task_id = InpM05TaskId;
      direct.noise_seed = InpM05NoiseSeed;
      direct.timeframe = ExtExecutionControl.timeframe;
      direct.rsi_period = (uint)RSIPeriod;
      direct.cci_period = (uint)CCIPeriod;
      direct.price_atr_period = (uint)ATRPeriod;
      direct.fast_period = (uint)FastPeriod;
      direct.slow_period = (uint)SlowPeriod;
      direct.signal_period = (uint)SignalPeriod;
      direct.latent_start = InpM05LatentStart;
      direct.latent_end = InpM05LatentEnd;
      direct.reference_start = InpM05ReferenceStart;
      direct.reference_end = InpM05ReferenceEnd;
      direct.calibration_start = InpM05CalibrationStart;
      direct.calibration_end = InpM05CalibrationEnd;
      direct.latent_epochs = InpM05LatentEpochs;
      direct.ompb_epochs = InpM05OMPBTrainingEpochs;
      direct.source_anchor_period = InpM05SourceAnchorPeriod;
      direct.rsi_price = RSIPrice;
      direct.cci_price = CCIPrice;
      direct.macd_price = MACDPrice;
      direct.tau = InpM05Tau;
      direct.lambda_dis = InpM05LambdaDis;
      direct.lambda_kl = InpM05LambdaKL;
      direct.lambda_alpha = InpM05LambdaAlpha;
      direct.alpha_prior = InpM05AlphaPrior;
      direct.max_kl = InpM05MaxKL;
      direct.price_joint_epochs = InpM05PriceJointEpochs;
      direct.price_joint_max_updates = InpM05PriceJointMaxUpdates;
      direct.price_adapt_epochs = InpM05PriceAdaptEpochs;
      direct.price_adapt_max_updates = InpM05PriceAdaptMaxUpdates;
      direct.max_file_bytes = InpM05MaxFileBytes;
      direct.lambda_price = InpM05PriceLambda;
      direct.huber_delta = InpM05PriceHuberDelta;
      direct.rho_tolerance = InpM05PriceRhoTolerance;
      direct.verify_atol = InpM05VerifyAtol;
      direct.verify_rtol = InpM05VerifyRtol;
      direct.verify_max_price_error = InpM05VerifyMaxPriceError;
      direct.verify_min_scenario_coverage = InpM05VerifyMinScenarioCoverage;
      direct.verify_min_distinct_scenarios = InpM05VerifyMinDistinctScenarios;
      direct.task_directory = DPCC_M05_TASK_DIRECTORY;
      if(!DPCCBuildM05ExecutionProfileFromInputs(ExtExecutionControl, direct, DPCC_COMMON_FILES,
            EXTM05DonorProfile, EXTM05VerifyProfile, EXTM05PriceLambda, EXTM05HuberDelta,
            EXTM05RhoTolerance, profile_reason))
        {
         Print("DPCC_M05_ABORT phase=input_profile reason=", profile_reason);
         return(INIT_PARAMETERS_INCORRECT);
        }
      EXTM05ProfileSHA = EXTM05DonorProfile.assigned_profile_sha256;
      EXTM05DonorSHA = EXTM05DonorProfile.calibration_preset_sha256;
      if(EXTM05DonorProfile.price_joint_epochs != 1 ||
         EXTM05DonorProfile.price_joint_max_updates != 0 ||
         EXTM05DonorProfile.price_adapt_epochs != 1 ||
         EXTM05DonorProfile.price_adapt_max_updates != 0)
        {
         Print("DPCC_M05_ABORT phase=donor_profile reason=one_full_2025_fit_pass_required");
         return(INIT_PARAMETERS_INCORRECT);
        }
      if(!ExtTrainer.ConfigureM05OMPB(EXTM05DonorProfile, profile_reason))
        {
         Print("DPCC_M05_ABORT phase=donor_profile reason=", profile_reason);
         return(INIT_PARAMETERS_INCORRECT);
        }
     }
   ExtRoot = "DPCC_StudyEncoder_" + (string)GetMicrosecondCount();
   ExtLastProgressMilestone = 0;
   MathSrand(20261007);
   rnd_x = (uint)MathRand();
   rnd_y = (uint)MathRand();
   rnd_z = (uint)MathRand();
   rnd_w = (uint)MathRand();
   rnd_t = 0;
   if((rnd_x | rnd_y | rnd_z | rnd_w) == 0)
      return(INIT_FAILED);
   Print("DPCC_STUDYFORECAST_INITIAL_RNG seed=20261007 x=", rnd_x, " y=", rnd_y,
         " z=", rnd_z, " w=", rnd_w, " t=", rnd_t);
   datetime capture_from = StringToTime("2024.12.01 00:00");
   datetime capture_to = StringToTime("2026.02.15 00:00");
   if(use_m05)
     {
      const long context_seconds = 7 * 24 * 60 * 60;
      if(EXTM05DonorProfile.latent_start < (datetime)((long)capture_from + context_seconds))
         capture_from = (datetime)((long)EXTM05DonorProfile.latent_start - context_seconds);
      if(EXTM05DonorProfile.latent_end > (datetime)((long)capture_to - context_seconds))
         capture_to = (datetime)((long)EXTM05DonorProfile.latent_end + context_seconds);
      if(EXTM05DonorProfile.reference_start < (datetime)((long)capture_from + context_seconds))
         capture_from = (datetime)((long)EXTM05DonorProfile.reference_start - context_seconds);
      if(EXTM05DonorProfile.calibration_end > (datetime)((long)capture_to - context_seconds))
         capture_to = (datetime)((long)EXTM05DonorProfile.calibration_end + context_seconds);
     }
   string raw_path = "", raw_sha = "", reason = "";
   uint raw_rows = 0;
   const ENUM_TIMEFRAMES timeframe = ExtExecutionControl.timeframe;
   const bool capture_price_ticks = ((InpPriceForwardOnly || InpPriceJoint) && !use_m05);
   if(!DPCCCaptureStudyForecastRaw(_Symbol, timeframe, capture_from, capture_to, raw_path, raw_sha, raw_rows, reason,
                                   capture_price_ticks,
                                   (use_m05 ? EXTM05DonorProfile.rsi_price : RSIPrice),
                                   (use_m05 ? EXTM05DonorProfile.cci_price : CCIPrice),
                                   (use_m05 ? EXTM05DonorProfile.macd_price : MACDPrice), use_m05, use_m05))
     {
      Print("DPCC_STUDYFORECAST_ABORT phase=capture reason=", reason);
      return(INIT_FAILED);
     }
   if(InpPriceForwardOnly)
     {
      if(!ExtTrainer.InitializePriceForward(raw_path, raw_sha, timeframe, reason))
        {
         Print("DPCC_PRICE_FORWARD_ABORT phase=initialize reason=", reason);
         return(INIT_FAILED);
        }
      int completed = 0, normalized_count = 0, native_count = 0;
      ulong first_decision_id = 0, last_decision_id = 0;
      long first_quote_time = 0, last_quote_time = 0, first_quote_msc = 0, last_quote_msc = 0;
      double first_anchor = 0, first_atr14 = 0;
      string quote_basis = "UNRESOLVED", quote_time_resolution = "";
      float first_open = 0, first_high = 0, first_low = 0, first_close = 0;
      for(int i = 0; i < 2; i++)
        {
         float normalized[], native[];
         SDPCCHistoricalPriceSnapshot snapshot;
         if(!ExtTrainer.AdvancePriceForward(normalized, native, snapshot, reason))
           {
            Print("DPCC_PRICE_FORWARD_ABORT phase=forward reason=", reason);
            return(INIT_FAILED);
           }
         if(i == 0)
           {
            first_decision_id = snapshot.decision_id;
            first_quote_time = snapshot.quote_time_server;
            first_quote_msc = snapshot.quote_time_msc;
            first_anchor = snapshot.anchor;
            first_atr14 = snapshot.scale;
            quote_basis = snapshot.quote_basis;
            quote_time_resolution = snapshot.quote_time_resolution;
            if(ArraySize(native) >= 4)
              {
               first_open = native[0];
               first_high = native[1];
               first_low = native[2];
               first_close = native[3];
              }
           }
         last_decision_id = snapshot.decision_id;
         last_quote_time = snapshot.quote_time_server;
         last_quote_msc = snapshot.quote_time_msc;
         normalized_count = ArraySize(normalized);
         native_count = ArraySize(native);
         completed++;
        }
      Print("DPCC_PRICE_FORWARD_COMPLETE decisions=", completed, " updates=0 model_save=0 orders=0",
            " raw_sha256=", ExtTrainer.RawSHA(), " symbol=", _Symbol, " timeframe=", EnumToString(timeframe),
            " quote_basis=", quote_basis, " quote_time_resolution=", quote_time_resolution,
            " tick_quote_time_resolved=true first_decision_id=", first_decision_id,
            " last_decision_id=", last_decision_id, " first_quote_time_server=", first_quote_time,
            " last_quote_time_server=", last_quote_time, " first_quote_time_msc=", first_quote_msc,
            " last_quote_time_msc=", last_quote_msc, " first_anchor=", DoubleToString(first_anchor, 16),
            " first_ATR14=", DoubleToString(first_atr14, 16),
            " first_scenario_ohlc=", first_open, ",", first_high, ",", first_low, ",", first_close,
            " normalized_count=", normalized_count, " native_count=", native_count,
            " trained_market=false scope=historical_H1_first_tick readiness=NOT_CLAIMED");
      return(EventSetTimer(1) ? INIT_SUCCEEDED : INIT_FAILED);
     }
   if(InpPriceJoint || use_m05)
     {
      const bool initialized = (use_m05 ?
                                ExtTrainer.InitializeM05(raw_path, raw_sha, EXTM05PriceLambda,
                                      EXTM05HuberDelta, EXTM05RhoTolerance, reason, true, timeframe) :
                                ExtTrainer.InitializePriceJoint(raw_path, raw_sha, InpMaxUpdates, false, InpPriceLambda,
                                      InpPriceHuberDelta, InpPriceRhoTolerance, reason));
      if(!initialized)
        {
         Print("DPCC_PRICE_JOINT_ABORT phase=initialize reason=", reason);
         return(INIT_FAILED);
        }
      bool m05_restored = false;
      if(use_m05 && !ExtTrainer.RestoreM05Boundary(ExtExecutionControl, m05_restored, reason, InpM05TrainingPhase))
        {
         Print("DPCC_M05_ABORT phase=restore reason=", reason);
         return(INIT_FAILED);
        }
      if(use_m05 && m05_restored)
         Print("DPCC_M05_RESUME_READY stage=", ExtTrainer.PriceStudyStage(),
               " position=", ExtTrainer.PriceStudyPosition(), " / ", ExtTrainer.PriceStudyTotal(),
               " updates=", ExtTrainer.Updates(), " mode=",
               (InpM05TrainingPhase == "AUTO" ? "exact_checkpoint_resume" : "selected_phase_model_warm_start"),
               " weights=latest checkpoint Adam=preserved recurrence=",
               (InpM05TrainingPhase == "AUTO" ? "warmup" : "reset"));
      if(use_m05 && InpM05TrainingPhase != "AUTO")
        {
         if(!SaveM05Checkpoint("USER_SELECTED_PHASE"))
            return(INIT_FAILED);
         Print("DPCC_M05_PHASE_SELECTED stage=", ExtTrainer.PriceStudyStage(),
               " position=", ExtTrainer.PriceStudyPosition(), " / ", ExtTrainer.PriceStudyTotal(),
               " updates=", ExtTrainer.Updates(), " prior_quality=ADVISORY checkpoint=latest_weights_and_adam");
        }
      if(use_m05 && ExtTrainer.Finished())
        {
         if(!SaveM05Checkpoint("PHASE_COMPLETED"))
            return(INIT_FAILED);
         string publication_reason = "";
         const bool consumer_ready = ExtTrainer.MarketConsumerReady(publication_reason);
         Print("DPCC_M05_PHASE_COMPLETE stage=", ExtTrainer.PriceStudyStage(),
               " status=ALREADY_COMPLETED next_stage=COMPLETE updates=", ExtTrainer.Updates(), " model_save=1");
         Print("DPCC_M05_COMPLETE status=", ExtTrainer.PricePhaseTerminalStatus(),
               " stage=", ExtTrainer.PriceStudyStage(), " updates=", ExtTrainer.Updates(),
               " market_consumer_ready=", consumer_ready, " consumer_reason=", publication_reason,
               " price_verify=REMOVED_BY_USER action=remove_advisor");
         ExtStopButton.RemoveAdvisor();
         return(INIT_SUCCEEDED);
        }
      if(use_m05 && !ExtStopButton.Create("M05", "Stop and save"))
        {
         Print("DPCC_M05_ABORT phase=stop_button reason=chart_object_create_failed error=", GetLastError());
         return(INIT_FAILED);
        }
      if(!EventSetTimer(1))
         return(INIT_FAILED);
      if(use_m05)
         Print("DPCC_M05_PHASE_START requested=", InpM05TrainingPhase,
               " stage=", ExtTrainer.PriceStudyStage(), " epochs_completed=", ExtTrainer.PriceStudyEpochsCompleted(),
               " position=", ExtTrainer.PriceStudyPosition(), " / ", ExtTrainer.PriceStudyTotal(),
               " updates=", ExtTrainer.Updates(), " restored=", m05_restored);
      Print("DPCC_PRICE_JOINT_READY raw_path=", raw_path, " raw_sha256=", ExtTrainer.RawSHA(),
            " raw_common_files=", (use_m05 ? "true" : "false"), " fit_eligible=", ExtTrainer.FitCount(),
            " max_updates=", (use_m05 ? EXTM05DonorProfile.price_joint_max_updates : InpMaxUpdates),
            " lambda_price=", (use_m05 ? EXTM05PriceLambda : InpPriceLambda),
            " huber_delta=", (use_m05 ? EXTM05HuberDelta : InpPriceHuberDelta),
            " rho_tolerance=", (use_m05 ? EXTM05RhoTolerance : InpPriceRhoTolerance),
            " phase=", ExtTrainer.PriceStudyStage(), " quote_time_resolution=",
            (use_m05 ? "SERVER_BAR_OPEN_MSC" : "SERVER_FIRST_AVAILABLE_TICK_MSC"));
      if(use_m05)
         Print("DPCC_M05_ACSRM_PROFILE_ADAPTED profile_sha256=", ExtTrainer.OMPBProfileSHA256(),
               " raw_sha256=", ExtTrainer.RawSHA(),
               " direct_input_profile_sha256=", EXTM05ProfileSHA,
               " ompb_config_sha256=", EXTM05DonorSHA, " stage=2 smoke=false epochs=", EXTM05DonorProfile.epochs,
               " reference=", TimeToString(EXTM05DonorProfile.reference_start, TIME_DATE), "..",
               TimeToString(EXTM05DonorProfile.reference_end, TIME_DATE), " calibration=",
               TimeToString(EXTM05DonorProfile.calibration_start, TIME_DATE), "..",
               TimeToString(EXTM05DonorProfile.calibration_end, TIME_DATE),
               " source_period=", EXTM05DonorProfile.source_anchor_period,
               " RSIPrice=", EnumToString(EXTM05DonorProfile.rsi_price),
               " CCIPrice=", EnumToString(EXTM05DonorProfile.cci_price),
               " MACDPrice=", EnumToString(EXTM05DonorProfile.macd_price),
               " tau=", EXTM05DonorProfile.tau, " lambda_dis=", EXTM05DonorProfile.lambda_dis,
               " lambda_kl=", EXTM05DonorProfile.lambda_kl, " lambda_alpha=", EXTM05DonorProfile.lambda_alpha,
               " alpha_prior=", EXTM05DonorProfile.alpha_prior, " max_kl=", EXTM05DonorProfile.max_kl,
               " price_joint_epochs=", EXTM05DonorProfile.price_joint_epochs,
               " price_joint_max_updates=", EXTM05DonorProfile.price_joint_max_updates,
               " price_adapt_epochs=", EXTM05DonorProfile.price_adapt_epochs,
               " price_adapt_max_updates=", EXTM05DonorProfile.price_adapt_max_updates,
               " posterior_layer=CNeuronOMPBOCL",
               " forecast_layer=CNeuronScenarioForecastPriceOCL", " donor_checkpoint_loaded=false");
      return(INIT_SUCCEEDED);
     }
   if(!ExtTrainer.Initialize(raw_path, raw_sha, ExtRoot, InpMaxUpdates, reason))
     {
      Print("DPCC_STUDYFORECAST_ABORT phase=initialize reason=", reason);
      return(INIT_FAILED);
     }
   if(!EventSetTimer(1))
     {
      Print("DPCC_STUDYFORECAST_ABORT phase=timer reason=", GetLastError());
      return(INIT_FAILED);
     }
   Print("DPCC_STUDYFORECAST_READY root=", ExtRoot, " raw_sha256=", ExtTrainer.RawSHA(),
         " raw_rows=", raw_rows, " fit_eligible=", ExtTrainer.FitCount(),
         " holdout_eligible=", ExtTrainer.HoldoutCount(), " max_updates=", InpMaxUpdates,
         " stages=encoder_fit,frozen_latent_holdout",
         " server_time=2024.12.01..2026.02.15 timezone_resolution=NOT_CLAIMED");
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnTimer                                                 |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   if(InpPriceForwardOnly)
     {
      EventKillTimer();
      ExtStopButton.RemoveAdvisor();
      return;
     }
   if(InpPriceJoint || M05Enabled())
     {
      double price_loss = 0, rho = 0;
      uint nonzero = 0, generator = 0, router = 0, confidence = 0;
      string reason = "";
      const string phase_before = ExtTrainer.PriceStudyStage();
      const uint epochs_before = (M05Enabled() ? ExtTrainer.PriceStudyEpochsCompleted() : 0);
      if(!ExtTrainer.AdvancePriceJoint(price_loss, rho, nonzero, generator, router, confidence, reason))
        {
         EventKillTimer();
         Print("DPCC_PRICE_JOINT_ABORT phase=advance reason=", reason, " updates=", ExtTrainer.Updates());
         return;
        }
      if(phase_before != ExtTrainer.PriceStudyStage())
         Print("DPCC_MARKET_PHASE_TRANSITION from=", phase_before, " to=", ExtTrainer.PriceStudyStage(),
               " graph_recreated=false parameters_retained=true updates=", ExtTrainer.Updates());
      const bool phase_completed = (phase_before != ExtTrainer.PriceStudyStage() ||
                                    (ExtTrainer.Finished() &&
                                     ExtTrainer.PricePhaseTerminalStatus() != "STOPPED_COMPLETE_UPDATE"));
      const bool selected_phase_completed = (M05Enabled() && InpM05TrainingPhase != "AUTO" && phase_completed);
      const bool epoch_completed = (M05Enabled() && phase_before == ExtTrainer.PriceStudyStage() &&
                                    ExtTrainer.PriceStudyEpochsCompleted() > epochs_before);
      const bool controlled_stop = (M05Enabled() && ExtExecutionControl.stop_after_updates > 0 &&
                                    ExtTrainer.Updates() >= ExtExecutionControl.stop_after_updates);
      if(M05Enabled() && (phase_before != ExtTrainer.PriceStudyStage() || ExtTrainer.Finished() ||
                          controlled_stop || epoch_completed))
        {
         string cause = "PHASE_COMPLETED";
         if(controlled_stop || ExtTrainer.PricePhaseTerminalStatus() == "STOPPED_COMPLETE_UPDATE")
            cause = "CONTROLLED_STOP";
         else
            if(epoch_completed && !phase_completed)
               cause = "EPOCH_COMPLETED";
         if(!SaveM05Checkpoint(cause))
           {
            EventKillTimer();
            ExtM05StopSaveFailed = true;
            ExtStopButton.SetText("Save failed; click to retry");
            return;
           }
         if(phase_completed)
            Print("DPCC_M05_PHASE_COMPLETE stage=", phase_before, " status=PHASE_COMPLETED",
                  " next_stage=", (ExtTrainer.Finished() ? "COMPLETE" : ExtTrainer.PriceStudyStage()),
                  " ompb_epochs=", ExtTrainer.OMPBTrainingEpochsCompleted(), "/", ExtTrainer.OMPBTrainingEpochLimit(),
                  " ompb_budget=", (ExtTrainer.OMPBTrainingBudgetComplete() ? "COMPLETE" : "PARTIAL"),
                  " ompb_eval_lanes=", ExtTrainer.OMPBStage02EvaluationLanes(), "/6",
                  " ompb_quality=", (!ExtTrainer.OMPBStage02QualityEvaluated() ? "PENDING" :
                                     (ExtTrainer.OMPBStage02QualityPassed() ? "PASS" : "FAIL")),
                  " quality_gate=ADVISORY",
                  " updates=", ExtTrainer.Updates(), " model_save=1");
         if(selected_phase_completed && !ExtTrainer.Finished())
           {
            EventKillTimer();
            Print("DPCC_M05_SELECTED_PHASE_COMPLETE requested=", InpM05TrainingPhase,
                  " saved_next_stage=", ExtTrainer.PriceStudyStage(), " action=remove_advisor");
            ExtStopButton.RemoveAdvisor();
            return;
           }
         if(controlled_stop && !ExtTrainer.Finished())
           {
            EventKillTimer();
            Print("DPCC_M05_CONTROLLED_STOP saved=true updates=", ExtTrainer.Updates(),
                  " stage=", ExtTrainer.PriceStudyStage(), " action=remove_advisor");
            ExtStopButton.RemoveAdvisor();
            return;
           }
         if(phase_completed && !ExtTrainer.Finished())
            Print("DPCC_M05_PHASE_START requested=AUTO stage=", ExtTrainer.PriceStudyStage(),
                  " epochs_completed=", ExtTrainer.PriceStudyEpochsCompleted(),
                  " position=", ExtTrainer.PriceStudyPosition(), " / ", ExtTrainer.PriceStudyTotal(),
                  " updates=", ExtTrainer.Updates(), " restored=false");
        }
      const string display_stage = ExtTrainer.PriceStudyStage();
      const int display_position = ExtTrainer.PriceStudyPosition();
      const int display_total = ExtTrainer.PriceStudyTotal();
      string chart_status = StringFormat("DPCC %s %d/%d", display_stage, display_position, display_total);
      const bool same_phase = (display_stage == phase_before);
      if(M05Enabled())
        {
         uint epoch_limit = 0;
         if(display_stage == "LATENT_PREPARE")
            epoch_limit = EXTM05DonorProfile.latent_epochs;
         else
            if(display_stage == "PRICE_JOINT")
               epoch_limit = EXTM05DonorProfile.price_joint_epochs;
            else
               if(display_stage == "OMPB_CALIBRATE")
                  epoch_limit = EXTM05DonorProfile.epochs;
               else
                  if(display_stage == "PRICE_ADAPT")
                     epoch_limit = EXTM05DonorProfile.price_adapt_epochs;
         uint epoch = ExtTrainer.PriceStudyEpochsCompleted();
         if(same_phase && display_position > 0 && display_position < display_total && epoch < epoch_limit)
            epoch++;
         chart_status = StringFormat("DPCC %s epoch %u/%u | %d/%d",
                                     display_stage, epoch, epoch_limit, display_position, display_total);
        }
      if(!same_phase)
         chart_status += "\nphase transition; next update will report new-phase metrics";
      else
         if(display_stage == "LATENT_PREPARE")
            chart_status += StringFormat("\nlatent: rho_mass=%.5g active_scenarios=%u optimizer_steps=%u/%u/%u",
                                         rho, nonzero, generator, router, confidence);
         else
            if(display_stage == "PRICE_JOINT")
               chart_status += StringFormat(
                                  "\nprice: loss=%.6g mean=%.6g rho_mass=%.5g active_scenarios=%u optimizer_steps=%u/%u/%u",
                                  price_loss, ExtTrainer.PriceJointMeanLoss(), rho, nonzero,
                                  generator, router, confidence);
            else
               if(display_stage == "PRICE_ADAPT")
                  chart_status += StringFormat(
                                     "\nOMPB %s epochs=%u/%u eval=%u/6 quality=%s (advisory) | adaptation loss=%.6g rho_mass=%.5g active_scenarios=%u optimizer_steps=%u/%u/%u",
                                     (ExtTrainer.OMPBTrainingBudgetComplete() ?
                                      "budget complete" : "operator accepted"),
                                     ExtTrainer.OMPBTrainingEpochsCompleted(),
                                     ExtTrainer.OMPBTrainingEpochLimit(), ExtTrainer.OMPBStage02EvaluationLanes(),
                                               (!ExtTrainer.OMPBStage02QualityEvaluated() ? "PENDING" :
                                                (ExtTrainer.OMPBStage02QualityPassed() ? "PASS" : "FAIL")),
                                               price_loss, rho, nonzero, generator, router, confidence);
               else
                  if(display_stage == "OMPB_CALIBRATE")
                    {
                     SDPCCM05ChartMetrics metrics;
                     if(ExtTrainer.M05ChartStatus(metrics))
                       {
                        if(metrics.subphase == "CALIBRATION" || metrics.subphase == "SOURCE_ANCHOR")
                          {
                           const uint active_epoch = (metrics.epoch < metrics.epoch_limit ?
                                                      metrics.epoch + 1 : metrics.epoch);
                           chart_status = StringFormat("DPCC OMPB_CALIBRATE/%s epoch %u/%u | %d/%d",
                                                       metrics.subphase, active_epoch, metrics.epoch_limit,
                                                       display_position, display_total);
                          }
                        else
                           if(StringFind(metrics.subphase, "EVALUATION_") == 0)
                              chart_status = StringFormat("DPCC OMPB_CALIBRATE/%s lane %d/6 | %d/%d",
                                                          metrics.subphase, metrics.evaluation_stage + 1,
                                                          display_position, display_total);
                           else
                              chart_status = StringFormat("DPCC OMPB_CALIBRATE/%s | %d/%d",
                                                          metrics.subphase, display_position, display_total);
                        if(metrics.diagnostics_ready)
                           chart_status += StringFormat(
                                                  "\nlast disagreement=%.5g KL=%.5g alpha_prior=%.5g fallback=%u KL_reject=%u quality=%s",
                                                  metrics.disagreement, metrics.kl, metrics.alpha_prior,
                                                        metrics.invalid_fallbacks, metrics.kl_rejects,
                                                        (!ExtTrainer.OMPBStage02QualityEvaluated() ? "pending" :
                                                         (ExtTrainer.OMPBStage02QualityPassed() ?
                                                          "PASS" : "FAIL (advisory)")));
                        else
                           chart_status += "\nOMPB metrics: waiting for first calibration update";
                        string lane_names[6] = {"bypass-source", "bypass-target", "pre-source", "pre-target",
                                                "post-source", "post-target"
                                               };
                        for(int lane = 0; lane < metrics.evaluation_stage && lane < 6; lane++)
                          {
                           double evaluation_loss = 0, evaluation_valid = 0, evaluation_invalid = 0;
                           if(ExtTrainer.OMPBStage02Evaluation(lane, evaluation_loss,
                                                               evaluation_valid, evaluation_invalid))
                              chart_status += StringFormat(
                                                 "\n%s loss=%.5g valid=%.0f invalid=%.0f",
                                                 lane_names[lane], evaluation_loss,
                                                 evaluation_valid, evaluation_invalid);
                          }
                       }
                     else
                        chart_status += "\nOMPB status unavailable";
                    }
                  else
                     if(display_stage == "VERIFY")
                        chart_status += StringFormat(
                                           "\nverify=%s snapshots=%u max_error=%.5g coverage=%.5g distinct=%u",
                                           ExtTrainer.PriceVerifyStatus(), ExtTrainer.PriceVerifySnapshots(),
                                                     ExtTrainer.PriceVerifyMaxError(), ExtTrainer.PriceVerifyCoverage(),
                                                     ExtTrainer.PriceVerifyMinDistinct());
                     else
                        if(!M05Enabled())
                           chart_status += StringFormat("\nprice: loss=%.6g rho_mass=%.5g active_scenarios=%u",
                                                        price_loss, rho, nonzero);
      Comment(chart_status);
      if(ExtTrainer.Finished())
        {
         EventKillTimer();
         const string terminal_status = ExtTrainer.PricePhaseTerminalStatus();
         if(M05Enabled())
           {
            string publication_reason = "";
            const bool consumer_ready = ExtTrainer.MarketConsumerReady(publication_reason);
            Print("DPCC_M05_COMPLETE status=", terminal_status, " stage=", ExtTrainer.PriceStudyStage(),
                  " updates=", ExtTrainer.Updates(), " market_consumer_ready=", consumer_ready,
                  " consumer_reason=", publication_reason, " price_verify=REMOVED_BY_USER",
                  " action=remove_advisor");
           }
         else
            if(terminal_status == "STOPPED_COMPLETE_UPDATE")
               Print("DPCC_PRICE_PHASE_STOPPED stage=", phase_before, " status=", terminal_status,
                     " updates=", ExtTrainer.Updates(), " phase_completed=false acceptance=NOT_CLAIMED");
            else
               Print("DPCC_PRICE_JOINT_COMPLETE updates=", ExtTrainer.Updates(),
                     " mean_weighted_price_huber=", ExtTrainer.PriceJointMeanLoss(),
                     " parameters=frozen next_stage=", ExtTrainer.PriceStudyNextStage(),
                     " model_save=0 M05_complete=false acceptance=NOT_CLAIMED");
         if(M05Enabled() || terminal_status != "STOPPED_COMPLETE_UPDATE")
            ExtStopButton.RemoveAdvisor();
        }
      return;
     }
   if(ExtTrainer.PartialFitComplete())
     {
      EventKillTimer();
      Print("DPCC_STUDYFORECAST_PARTIAL root=", ExtRoot, " updates=", ExtTrainer.Updates(),
            " fit_eligible=", ExtTrainer.FitCount(), " holdout=NOT_RUN checkpoint=NOT_WRITTEN");
      Comment("DPCC StudyForecast partial fit: ", ExtTrainer.Updates(), " / ", ExtTrainer.FitCount());
      return;
     }
   double rho = 0;
   uint nonzero = 0, generator = 0, router = 0, confidence = 0;
   bool holdout_step = false;
   string reason = "";
   if(!ExtTrainer.Advance(rho, nonzero, generator, router, confidence, holdout_step, reason))
     {
      EventKillTimer();
      Print("DPCC_STUDYFORECAST_ABORT phase=advance reason=", reason, " updates=", ExtTrainer.Updates(),
            " holdout_examples=", ExtTrainer.HoldoutCompleted());
      return;
     }
   if(holdout_step)
     {
      Comment("DPCC StudyForecast frozen holdout: ", ExtTrainer.HoldoutCompleted(), " / ", ExtTrainer.HoldoutCount());
      if(ExtTrainer.Finished())
        {
         if(!ExtTrainer.SaveCompleted(reason))
            Print("DPCC_STUDYFORECAST_ABORT phase=save reason=", reason);
         else
            Print("DPCC_STUDYFORECAST_COMPLETE root=", ExtRoot, " fit_updates=", ExtTrainer.Updates(),
                  " holdout_examples=", ExtTrainer.HoldoutCount(),
                  " valid_examples=", ExtTrainer.HoldoutValidExamples(),
                  " invalid_examples=", ExtTrainer.HoldoutInvalidExamples(),
                  " latent_coordinate_mae=", DoubleToString(ExtTrainer.HoldoutLatentError(), 12),
                  " teacher=fresh_frozen_untrained metric_scope=latent_diagnostics weights_changed_during_holdout=false");
         ExtStopButton.RemoveAdvisor();
         EventKillTimer();
        }
      return;
     }
   const ulong updates = ExtTrainer.Updates();
   const ulong progress_milestone = updates / 1000;
   if(progress_milestone > ExtLastProgressMilestone)
     {
      ExtLastProgressMilestone = progress_milestone;
      Print("DPCC_STUDYFORECAST_UPDATE updates=", updates, " / ", ExtTrainer.FitCount(),
            " rho_sum=", rho, " rho_nonzero=", nonzero,
            " generator_step=", generator, " router_step=", router, " confidence_step=", confidence);
     }
   Comment("DPCC StudyForecast fit ", ExtTrainer.Updates(), " / ", ExtTrainer.FitCount(),
           " rho=", rho, " heads=", generator, "/", router, "/", confidence);
  }
//+------------------------------------------------------------------+
//| Function OnChartEvent                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(!M05Enabled() || !ExtStopButton.Clicked(id, sparam))
      return;
   EventKillTimer();
   ExtStopButton.SetText(ExtM05StopSaveFailed ? "Retrying save..." : "Saving and stopping...");
   if(!SaveM05Checkpoint("CONTROLLED_STOP"))
     {
      ExtM05StopSaveFailed = true;
      ExtStopButton.SetText("Retry save and stop");
      Comment("DPCC M05: save failed; click button to retry. Training is paused.");
      return;
     }
   ExtM05StopSaveFailed = false;
   Print("DPCC_M05_CONTROLLED_STOP saved=true action=remove_advisor");
   ExtStopButton.RemoveAdvisor();
  }
//+------------------------------------------------------------------+
//| Function OnDeinit                                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ExtStopButton.Delete();
   Comment("");
  }
//+------------------------------------------------------------------+