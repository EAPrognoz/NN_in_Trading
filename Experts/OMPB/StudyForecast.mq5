//+------------------------------------------------------------------+
//| StudyForecast.mq5                                                |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
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
input ENUM_OMPB_STAGE             InpOMPBStage             = OMPB_STAGE_MARKET_ENCODER;
input bool                        InpOMPBSmoke             = false; //Stage smoke; never writes a production checkpoint
//--- Stage 02 uses these values for strict checkpoint preflight, causal
//--- partitioning, Reference collection and frozen BYPASS baselines.  The
//--- calibration/update transaction remains in the following block.
input group                       "---- OMPB calibration ----"
input datetime                    InpOMPBReferenceStart    = D'2024.01.01';
input datetime                    InpOMPBReferenceEnd      = D'2025.06.01';
input datetime                    InpOMPBCalibrationStart  = D'2025.06.01';
input datetime                    InpOMPBCalibrationEnd    = D'2026.01.01';
input uint                        InpOMPBSourcePeriod      = 0;
input float                       InpOMPBTau               = 1.0f;
input float                       InpOMPBLambdaDis         = 0.0f;
input float                       InpOMPBLambdaKL          = 0.0f;
input float                       InpOMPBLambdaAlpha       = 0.0f;
input float                       InpOMPBAlphaPrior        = 0.5f;
input float                       InpOMPBMaxKL             = 0.0f;
//---
input group                       "---- Forecast recovery smoke ----"
input bool                        InpRecoverySmoke         = false; //Run recovery smoke
input uint                        InpRecoverySmokeBatches  = 2000; //Recovery smoke batches
input uint                        InpRecoverySmokeAge      = 512; //Recovery inactive age
//---
datetime Start;
datetime End;
int      Epochs;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   Start = InpStart;
   End = InpEnd;
   Epochs = InpEpochs;
   const bool smoke = (InpRecoverySmoke || InpOMPBSmoke);
   if(InpOMPBStage == OMPB_STAGE_CALIBRATION)
     {
      if(!D2SkillConfigureForecastRun(smoke, InpRecoverySmokeBatches,
                                      InpRecoverySmokeAge) ||
         !CreateOMPBStage02ReferenceStudy(InpOMPBReferenceStart, InpOMPBReferenceEnd,
                                          InpOMPBCalibrationStart, InpOMPBCalibrationEnd,
                                          InpOMPBSourcePeriod, InpOMPBTau, InpOMPBLambdaDis,
                                          InpOMPBLambdaKL, InpOMPBLambdaAlpha, InpOMPBAlphaPrior,
                                          InpOMPBMaxKL, smoke, InpRecoverySmokeBatches))
         return(INIT_PARAMETERS_INCORRECT);
      return(INIT_SUCCEEDED);
     }
   if(InpOMPBStage != OMPB_STAGE_MARKET_ENCODER)
     {
      PrintFormat("OMPB_STAGE_UNSUPPORTED stage=%d", InpOMPBStage);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(Start >= End || Epochs <= 0 ||
      !D2SkillConfigureForecastRun(smoke, InpRecoverySmokeBatches,
                                   InpRecoverySmokeAge))
      return(INIT_PARAMETERS_INCORRECT);
   return(CreateD2SkillForecastStudy() ? INIT_SUCCEEDED : INIT_FAILED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ReleaseD2SkillForecastStudy(reason);
  }
//+------------------------------------------------------------------+
//| Chart event callback                                             |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam,
                  const string &sparam)
  {
   if(id == 1001)
     {
      if(InpOMPBStage == OMPB_STAGE_CALIBRATION)
         RunOMPBStage02ReferenceStudy();
      else
         TrainD2SkillForecast();
     }
  }
//+------------------------------------------------------------------+
