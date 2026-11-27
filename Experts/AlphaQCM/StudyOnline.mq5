//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|                                                         AlphaQCM |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#define ALPHAGEN_TEST_DIAGNOSTICS
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool OpenCLWriteTrace=false;
#define ALPHAGEN_TRACE_WRITES OpenCLWriteTrace
input bool PPOProfile=false;
#define ALPHAGEN_PPO_PROFILE PPOProfile
input int TradeBackwardBars=64;   // Trading LSTM backpropagation bars (0 = full context)
#define ALPHAGEN_TRADE_BACKWARD_BARS TradeBackwardBars
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CAlphaQCMApplication App;
//+------------------------------------------------------------------+
//| OnInit - initialize the EA and its event resources.              |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(TradeBackwardBars<0)
      return(INIT_PARAMETERS_INCORRECT);

   if(!App.Init(true,false)||!EventSetTimer(1))
     {
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnTick - Process quotes through the execution driver             |
//+------------------------------------------------------------------+
void OnTick()
  {
   App.Tick();
  }
//+------------------------------------------------------------------+
//| OnTimer - continue bounded search/PPO work between market ticks. |
//+------------------------------------------------------------------+
void OnTimer()
  {
   App.Tick();
  }
//+------------------------------------------------------------------+
//| OnTradeTransaction - Forward trade notifications                 |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   App.Transaction(trans,request,result);
  }
//+------------------------------------------------------------------+
//| OnTester - Finalize confirmed test results                       |
//+------------------------------------------------------------------+
double OnTester()
  {
   if(!App.Finalized&&!App.failed&&!App.Finish(App.money.stopout,true,true))
     {
      Print("AlphaQCM Tester finalization incomplete");
     }
   PrintFormat("AlphaQCM Strategy Tester actual profit %.2f profit factor %.6f",
               TesterStatistics(STAT_PROFIT),TesterStatistics(STAT_PROFIT_FACTOR));
   PrintFormat("AlphaQCM Tester deals %d equity drawdown %.2f (%.2f%%)",
               (int)TesterStatistics(STAT_DEALS),TesterStatistics(STAT_EQUITY_DD),
               TesterStatistics(STAT_EQUITYDD_PERCENT));
   PrintFormat("AlphaQCM Tester closed trades %d",(int)TesterStatistics(STAT_TRADES));

   return(App.money.equity-App.money.reference);
  }
//+------------------------------------------------------------------+
//| OnDeinit - release event resources without trading or saving.    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
