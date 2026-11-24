//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                                         AlphaQCM |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#define ALPHAGEN_TEST_DIAGNOSTICS
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool OpenCLWriteTrace=false;
#define ALPHAGEN_TRACE_WRITES OpenCLWriteTrace
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
   if(!App.Finalized&&!App.failed&&!App.FinishTestReport())
     {
      Print("AlphaQCM Test financial finalization incomplete");
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
   Print("AlphaQCM Test unloaded, finalized=",App.Finalized);
  }
//+------------------------------------------------------------------+
