//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                                         AlphaGen |
//|                           https://www.mql5.com/ru/articles/25997 |
//+------------------------------------------------------------------+
#define ALPHAGEN_TEST_DIAGNOSTICS
#define ALPHAGEN_DECISION_TRACE
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool OpenCLWriteTrace=false;
#define ALPHAGEN_TRACE_WRITES OpenCLWriteTrace
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
   return(App.Init(false,false)?INIT_SUCCEEDED:INIT_FAILED);
  }
//+------------------------------------------------------------------+
//| Process current quotes through the shared execution driver.      |
//+------------------------------------------------------------------+
void OnTick()
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
   if(!App.Finalized&&!App.failed&&!App.Finish(false,false))
     {
      Print("AlphaGen Test financial finalization incomplete");
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
   Print("AlphaGen Test unloaded, finalized=",App.Finalized);
  }
//+------------------------------------------------------------------+
