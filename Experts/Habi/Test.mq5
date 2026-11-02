//+------------------------------------------------------------------+
//|                                                        Test.mq5  |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, MetaQuotes Ltd."
#property link        "https://www.mql5.com"
#property version     "1.00"
#property description "Online smoke test for the Habi pipeline"
#property description "Runs one decision per closed bar"
#property description "Sends orders through the execution module"
#property strict
#include "Trajectory.mqh"

//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CHabiModels     ExtModels;         // model container
CHabiHistory    ExtHistory;        // history container
CHabiExecution  ExtExecution;      // execution module
datetime        ExtLastBar=0;      // last processed bar
bool            ExtFailed=false;   // fatal error flag

//+------------------------------------------------------------------+
//| OnInit - initialises modules for the single slot test            |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!HabiStart(ExtModels,ExtHistory,2) || !ExtExecution.Init(1,true))
      return(INIT_FAILED);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnTick - builds one decision per bar and sends it                |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(ExtFailed)
      return;

   datetime bar=iTime(_Symbol,TimeFrame,0);

   if(bar<=0 || bar==ExtLastBar)
      return;

   ExtLastBar=bar;

   if(!ExtExecution.PrepareBar(bar))
     {
      HabiFatal("bar reconciliation failed");

      return;
     }

   ulong start=GetMicrosecondCount();
   float observation[];
   float scores[];

   if(!ExtExecution.Observation(ExtHistory,bar,observation) || !ExtModels.Predict(observation,false) || !ExtModels.Scores(ExtModels.critic,ExtModels.last_z,ExtModels.last_actions,++ExtModels.serial,35,false,scores))
     {
      ExtFailed=true;
      HabiFatal("Test observation/forward failed");

      return;
     }

   ExtExecution.CommitAccountBoundary();

   int best=0;

   for(uint n=1;n<ExtModels.g.N;n++)
      if(scores[n]>scores[best])
         best=(int)n;

   ulong elapsed=GetMicrosecondCount()-start;

   PrintFormat("Habi Test decision_us=%I64u N=%u selected=%d Q=%.9g",elapsed,ExtModels.g.N,best,scores[best]);

   float raw[];

   ArrayResize(raw,6);

   ArrayCopy(raw,ExtModels.last_actions,0,best*6,6);

   ExtExecution.generation++;

   if(!ExtExecution.Execute(0,raw,bar))
     {
      ExtFailed=true;
      HabiFatal("Test execution data/profile failure");
     }
  }
//+------------------------------------------------------------------+
//| OnTradeTransaction - forwards deals to the execution module      |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   ExtExecution.DealEvent(trans.type,trans.deal);

   if(ArraySize(ExtExecution.slots)>0)
      ExtExecution.ReconcileSlot(0);
  }

//+------------------------------------------------------------------+
