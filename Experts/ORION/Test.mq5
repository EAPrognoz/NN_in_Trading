//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                                            ORION |
//|                                              Copyright 2026, DNG |
//|                           https://www.mql5.com/ru/articles/23517 |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, DNG"
#property link        "https://www.mql5.com/ru/users/dng"
#property version     "1.00"
#property description "Харнесс вневыборочного (OOS) теста ORION."
#property description "Прогоняет инференс на отложенном интервале и оценивает качество политик."
#property description "Только для тестов; ордера не выставляет."
#define StudyOnline
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CNet Actor;
CBufferFloat State,TimeState,Account,Action;
double PreviousBalance=0,PreviousEquity=0;
bool   ORIONTestInitialized=false;
//+------------------------------------------------------------------+
//| OnInit - function OnInit.                                        |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//--- exact Forecast and six-artifact AC manifest/hash validation happens
//--- before loading/binding the Actor and before any possible policy forward.
   ORIONTestInitialized=false;
   if(!ORIONInitIndicators())
     { Print("ORION test init: indicators=FAIL"); return(INIT_FAILED); }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
     { Print("ORION test init: trade filling=FAIL"); return(INIT_FAILED); }
   if(!ORIONLoadForecastInference())
     { Print("ORION test init: forecast=FAIL"); return(INIT_FAILED); }
   if(!ORIONLoadInferenceActor(Actor))
     { Print("ORION test init: actor checkpoint=FAIL"); return(INIT_FAILED); }
   if(!ORIONVerifyFrozenForecastExact())
     { Print("ORION test init: frozen forecast=FAIL"); return(INIT_FAILED); }
   PreviousBalance=AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity=AccountInfoDouble(ACCOUNT_EQUITY);
   ORIONTestInitialized=true;

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnDeinit - function OnDeinit.                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- inference is pure: neither policy nor Forecast artifacts are saved.
   if(ORIONTestInitialized&&ORIONForecast!=NULL&&!ORIONVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation",__FUNCTION__,__LINE__);
   ORIONForecast=NULL;
  }
//+------------------------------------------------------------------+
//| OnTick - function OnTick.                                        |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;

   if(!ORIONRefreshLiveMarket(GetPointer(State),GetPointer(TimeState))||
      !ORIONMarket.Clear()||
      !ORIONForwardForecastState(GetPointer(State)))
     { PrintFormat("%s -> %d market/forecast",__FUNCTION__,__LINE__); return; }
   double buy_value=0,sell_value=0;
   if(!ORIONBuildLiveAccount(PreviousBalance,PreviousEquity,Rates[0].time,GetPointer(Account),buy_value,sell_value)||
      !Actor.feedForward(GetPointer(Account),1,false,GetPointer(ORIONMarket),-1)||
      !ReadAction(Actor,GetPointer(Action))||
      !ORIONExecuteAction(GetPointer(Action),buy_value,sell_value))
     { PrintFormat("%s -> %d actor/execute",__FUNCTION__,__LINE__); return; }
   PreviousBalance=AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity=AccountInfoDouble(ACCOUNT_EQUITY);
  }
//+------------------------------------------------------------------+
