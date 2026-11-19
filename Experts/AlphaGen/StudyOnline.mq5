//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|                                                         AlphaGen |
//|                           https://www.mql5.com/ru/articles/25997 |
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
CAlphaGenApplication App;
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
//| Process current quotes through the shared execution driver.      |
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
//| Forward trade notifications without issuing a new action.        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   App.Transaction(trans,request,result);
  }
//+------------------------------------------------------------------+
//| Finalize confirmed test results before terminal deinitialization.|
//+------------------------------------------------------------------+
double OnTester()
  {
   if(!App.Finalized&&!App.failed&&!App.Finish(App.money.stopout,true,true))
     {
      Print("AlphaGen Tester finalization incomplete");
     }
   PrintFormat("AlphaGen Strategy Tester actual profit %.2f profit factor %.6f",
               TesterStatistics(STAT_PROFIT),TesterStatistics(STAT_PROFIT_FACTOR));

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
