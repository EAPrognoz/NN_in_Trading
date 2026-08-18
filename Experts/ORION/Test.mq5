//+------------------------------------------------------------------+
//| Test.mq5                                                         |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#define StudyOnline
#include "Trajectory.mqh"
CNet Actor;
CBufferFloat State, TimeState, Account, Action;
double PreviousBalance = 0, PreviousEquity = 0;
bool ORIONTestInitialized = false;
ulong OOSTransitions = 0, OOSExecutableActions = 0;
double OOSInitialBalance = 0;
//+------------------------------------------------------------------+
//| Function OnInit.                                                 |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//--- Exact Forecast and six-artifact AC manifest/hash validation happens
//--- before loading/binding the Actor and before any possible policy forward.
   ORIONTestInitialized = false;
   if(!ORIONInitIndicators())
     { Print("ORION test init: indicators=FAIL"); return INIT_FAILED; }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
     { Print("ORION test init: trade filling=FAIL"); return INIT_FAILED; }
   if(!ORIONLoadForecastInference())
     { Print("ORION test init: forecast=FAIL"); return INIT_FAILED; }
   if(!ORIONLoadInferenceActor(Actor))
     { Print("ORION test init: actor checkpoint=FAIL"); return INIT_FAILED; }
   if(!ORIONVerifyFrozenForecastExact())
     { Print("ORION test init: frozen forecast=FAIL"); return INIT_FAILED; }
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   OOSInitialBalance = PreviousBalance;
   OOSTransitions = 0;
   OOSExecutableActions = 0;
   ORIONTestInitialized = true;
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Inference is pure: neither policy nor Forecast artifacts are saved.
   if(ORIONTestInitialized && ORIONForecast != NULL && !ORIONVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   if(ORIONTestInitialized)
      PrintFormat("ORION OOS summary: transitions=%I64u executable_actions=%I64u initial_balance=%.2f final_balance=%.2f",
                  OOSTransitions, OOSExecutableActions, OOSInitialBalance,
                  AccountInfoDouble(ACCOUNT_BALANCE));
   ORIONForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Function OnTick.                                                 |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
   if(!ORIONRefreshLiveMarket(GetPointer(State), GetPointer(TimeState)) ||
      !ORIONMarket.Clear() ||
      !ORIONForwardForecastState(GetPointer(State)))
     { PrintFormat("%s -> %d market/forecast", __FUNCTION__, __LINE__); return; }
   double buy_value = 0, sell_value = 0;
   if(!ORIONBuildLiveAccount(PreviousBalance, PreviousEquity, Rates[0].time, GetPointer(Account), buy_value, sell_value) ||
      !Actor.feedForward(GetPointer(Account), 1, false, GetPointer(ORIONMarket), -1) ||
      !ReadAction(Actor, GetPointer(Action)) ||
      !ORIONExecuteAction(GetPointer(Action), buy_value, sell_value))
     { PrintFormat("%s -> %d actor/execute", __FUNCTION__, __LINE__); return; }
   OOSTransitions++;
   if(IsExecutableOrder(double(Action[0]), double(Action[1]), double(Action[2])) ||
      IsExecutableOrder(double(Action[3]), double(Action[4]), double(Action[5])))
      OOSExecutableActions++;
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
  }
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
