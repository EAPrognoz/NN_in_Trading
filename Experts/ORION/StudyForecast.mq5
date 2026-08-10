//+------------------------------------------------------------------+
//|                                               StudyForecast.mq5 |
//|                                      ORION joint forecast study |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define ORION_FORECAST_STUDY
#define Study
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input datetime Start  = D'2024.01.01';
input datetime End    = D'2026.01.01';
input int      Epochs = 100;
input bool     ORIONRecoverySmoke = false;
input uint     ORIONRecoverySmokeBatches = 2000;
input uint     ORIONRecoverySmokeAge = 512;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   return (CreateORIONForecastStudy() ? INIT_SUCCEEDED : INIT_FAILED);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ReleaseORIONForecastStudy(reason);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id == 1001)
      TrainORIONForecast();
  }
//+------------------------------------------------------------------+
