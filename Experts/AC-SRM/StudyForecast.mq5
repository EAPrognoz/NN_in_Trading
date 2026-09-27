//+------------------------------------------------------------------+
//|                                                StudyForecast.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "AC-SRM Stages 01/02: базовое обучение Forecast-энкодера и OMPB-калибровка цепочки."
#property description "Режимы: InpOMPBStage=1 — MarketEncoder, InpOMPBStage=2 — калибровка (эталон + смоук)."
#property description "Источник — исторические бары EURUSD (TimeFrame); результат — чекпоинты энкодера и манифест."
#property strict
//---
#include "Trajectory.mqh"
//---
input group                       "---- Forecast training ----"
input datetime                    InpStart                 = D'2024.01.01'; //Training period start
input datetime                    InpEnd                   = D'2026.01.01'; //Training period end
input int                         InpEpochs                = 100; //Training epochs
//---
input group                       "---- OMPB stage ----"
input ENUM_OMPB_STAGE             InpOMPBStage              = OMPB_STAGE_MARKET_ENCODER; //Chain stage: 1=forecast (encoder), 2=calibration
input bool                        InpOMPBSmoke              = false; //Smoke preflight: quick pass, no production checkpoint
//--- Stage 02 uses these values for strict checkpoint preflight, causal
//--- partitioning and frozen BYPASS baselines. The calibration/update
//--- transaction remains in the following block.
input group                       "---- OMPB calibration ----"
input uint                        InpOMPBEpochs             = 3; //Calibration epochs
input datetime                    InpOMPBReferenceStart     = D'2024.01.01'; //Reference window start (frozen baseline)
input datetime                    InpOMPBReferenceEnd       = D'2024.07.01'; //Reference window end
input datetime                    InpOMPBCalibrationStart   = D'2024.07.01'; //Calibration window start
input datetime                    InpOMPBCalibrationEnd     = D'2026.01.01'; //Calibration window end
input uint                        InpOMPBSourcePeriod       = 0; //Source period (bars) for feature sampling (0=auto)
input float                       InpOMPBTau                = 1.0f; //Calibration temperature
input float                       InpOMPBLambdaDis          = 0.0f; //Distribution-coupling regularization weight
input float                       InpOMPBLambdaKL           = 0.0f; //KL-divergence regularization weight
input float                       InpOMPBLambdaAlpha        = 0.0f; //Alpha-head regularization weight
input float                       InpOMPBAlphaPrior         = -6.0f; //Logit prior for the alpha head
input float                       InpOMPBMaxKL              = 0.0f; //KL cap during calibration (0=off)
//---
input group                       "---- Forecast recovery smoke ----"
input bool                        InpRecoverySmoke         = false; //Run recovery smoke
input uint                        InpRecoverySmokeBatches  = 2000; //Recovery smoke batches
input uint                        InpRecoverySmokeAge      = 512; //Recovery inactive age
//---
datetime Start;
datetime End;
int      Epochs;
bool     TesterTriggered = false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   Start = InpStart;
   End = InpEnd;
   Epochs = InpEpochs;
//--- Strategy Tester has no custom-event controller; a millisecond timer
//--- auto-triggers the same stage dispatch once per run.
   if(!EventSetMillisecondTimer(1))
      return(INIT_PARAMETERS_INCORRECT);
   const bool smoke = (InpRecoverySmoke || InpOMPBSmoke);
   if(InpOMPBStage == OMPB_STAGE_CALIBRATION)
     {
      if(InpOMPBEpochs == 0)
         return(INIT_PARAMETERS_INCORRECT);
      if(!SkillConfigureForecastRun(smoke, InpRecoverySmokeBatches,
                                    InpRecoverySmokeAge) ||
         !CreateACSRMStage02ReferenceStudy(InpOMPBReferenceStart, InpOMPBReferenceEnd,
                                           InpOMPBCalibrationStart, InpOMPBCalibrationEnd,
                                           InpOMPBSourcePeriod, InpOMPBTau, InpOMPBLambdaDis,
                                           InpOMPBLambdaKL, InpOMPBLambdaAlpha, InpOMPBAlphaPrior,
                                           InpOMPBMaxKL, smoke, InpRecoverySmokeBatches, InpOMPBEpochs))
         return(INIT_PARAMETERS_INCORRECT);
      return(INIT_SUCCEEDED);
     }
   if(InpOMPBStage != OMPB_STAGE_MARKET_ENCODER)
     {
      PrintFormat("OMPB_STAGE_UNSUPPORTED stage=%d", InpOMPBStage);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(Start >= End || Epochs <= 0 ||
      !SkillConfigureForecastRun(smoke, InpRecoverySmokeBatches,
                                 InpRecoverySmokeAge))
      return(INIT_PARAMETERS_INCORRECT);
   return(CreateSkillForecastStudy() ? INIT_SUCCEEDED : INIT_FAILED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ReleaseSkillForecastStudy(reason);
  }
//+------------------------------------------------------------------+
//| Chart event callback                                             |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!TesterTriggered)
     {
      TesterTriggered = true;
      EventKillTimer();
      if(InpOMPBStage == OMPB_STAGE_CALIBRATION)
         RunACSRMStage02ReferenceStudy();
      else
         TrainSkillForecast();
     }
  }
//+------------------------------------------------------------------+
//| Chart event callback (OnChartEvent).                             |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam,
                  const string &sparam)
  {
   if(id == 1001 || id == 1)   // self-posted Init event (tester has no 1001 controller)
     {
      if(InpOMPBStage == OMPB_STAGE_CALIBRATION)
         RunACSRMStage02ReferenceStudy();
      else
         TrainSkillForecast();
     }
  }
//+---+