//+------------------------------------------------------------------+
//|                                                StudyForecast.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property strict
#property description "Харнесс обучения прогнозного блока D2."
#property description "Обучает модель на заданном историческом интервале и проверяет восстановление чекпоинта."
#property description "Только для тестов; ордера не выставляет."
//---
#include "Trajectory.mqh"
//---
input group                       "---- Forecast training ----"
input datetime                    InpStart                 = D'2024.01.01'; //Training period start
input datetime                    InpEnd                   = D'2026.01.01'; //Training period end
input int                         InpEpochs                = 100; //Training epochs
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
   if(Start >= End || Epochs <= 0 ||
      !D2SkillConfigureForecastRun(InpRecoverySmoke, InpRecoverySmokeBatches,
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
      TrainD2SkillForecast();
  }
//+---+