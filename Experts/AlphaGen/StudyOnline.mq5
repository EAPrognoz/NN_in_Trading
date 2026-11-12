#property strict
#define ALPHAGEN_TEST_DIAGNOSTICS
#include "Trajectory.mqh"
CAlphaGenApplication App;
//+------------------------------------------------------------------+
//| Initialize the EA and its event resources.                       |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!App.Init(true,false) || !EventSetTimer(1))
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
//| Continue bounded search/PPO work between market ticks.           |
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
   if(!App.Finalized && !App.failed && !App.Finish(App.money.stopout,true,true))
     {
      Print("AlphaGen Tester finalization incomplete");
     }
   PrintFormat("AlphaGen Strategy Tester actual profit %.2f profit factor %.6f",
   TesterStatistics(STAT_PROFIT),TesterStatistics(STAT_PROFIT_FACTOR));
   return(App.money.equity-App.money.reference);
  }
//+------------------------------------------------------------------+
//| Release event resources without trading or saving.               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Print("AlphaGen StudyOnline unloaded, finalized=",App.Finalized," failed=",App.failed);
  }
