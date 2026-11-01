#property strict
#property version "0.26"
#include "Trajectory.mqh"
CHabiModels Models;
CHabiHistory History;
CHabiExecution Execution;
datetime LastBar=0;
bool Failed=false;

//+------------------------------------------------------------------+
//| OnInit.
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!HabiStart(Models,History,2) || !Execution.Init(1,true))
      return INIT_FAILED;
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| OnTick.
//+------------------------------------------------------------------+
void OnTick()
  {
   if(Failed)
      return;
   datetime bar=iTime(_Symbol,DataTimeframe,0);
   if(bar<=0 || bar==LastBar)
      return;
   LastBar=bar;
   if(!Execution.PrepareBar(bar))
     {
      HabiFatal("bar reconciliation failed");
      return;
     }
   ulong start=GetMicrosecondCount();
   float observation[],scores[];
   if(!Execution.Observation(History,bar,observation) || !Models.Predict(observation,false) || !Models.Scores(Models.critic,Models.last_z,Models.last_actions,++Models.serial,35,false,scores))
     {
      Failed=true;
      HabiFatal("Test observation/forward failed");
      return;
     }
   Execution.CommitAccountBoundary();
   int best=0;
   for(uint n=1;n<Models.g.N;n++)
      if(scores[n]>scores[best])
         best=(int)n;
   ulong elapsed=GetMicrosecondCount()-start;
   PrintFormat("Habi Test decision_us=%I64u N=%u selected=%d Q=%.9g",elapsed,Models.g.N,best,scores[best]);
   float raw[];
   ArrayResize(raw,6);
   ArrayCopy(raw,Models.last_actions,0,best*6,6);
   Execution.generation++;
   if(!Execution.Execute(0,raw,bar))
     {
      Failed=true;
      HabiFatal("Test execution data/profile failure");
     }
  }

//+------------------------------------------------------------------+
//| OnTradeTransaction.
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   Execution.DealEvent(trans.type,trans.deal);
   if(ArraySize(Execution.slots)>0)
      Execution.ReconcileSlot(0);
  }
