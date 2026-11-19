//+------------------------------------------------------------------+
//|                                                   NetCreator.mq5 |
//|                                                       NetCreator |
//|                                                   Copyright DNG® |
//|                           https://www.mql5.com/ru/articles/11273 |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "NetCreatorPanel.mqh"
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CNetCreatorPanel Panel;
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool CommonDirectory=false;   //Use Common Directory
//+------------------------------------------------------------------+
//| OnInit - expert initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!Panel.Create(0,"NetCreator",0,50,50,CommonDirectory))
      return(INIT_FAILED);
   if(!Panel.Run())
      return(INIT_FAILED);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnDeinit - expert deinitialization function                      |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Panel.Destroy(reason);
  }
//+------------------------------------------------------------------+
//| OnTick - expert tick function                                    |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  }
//+------------------------------------------------------------------+
//| OnChartEvent - chartEvent function                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   Panel.ChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+

