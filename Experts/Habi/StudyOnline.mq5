#property strict
#property version "0.26"
#include "Trajectory.mqh"
CHabiModels Models;
CHabiHistory History;
CHabiExecution Execution;
datetime LastBar=0;
uint PolicyTick=0,TDTick=0;
bool Failed=false;

//+------------------------------------------------------------------+
//| OnInit.
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!HabiStart(Models,History,1) || !Execution.Init(Models.g.N,false))
      return INIT_FAILED;
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Stop.
//+------------------------------------------------------------------+
void Stop(string operation)
  {
   Failed=true;
   HabiFatal(operation);
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
   bool terminal=OnlineEpisodeEnd>0 && bar>=OnlineEpisodeEnd;
   float observation[];
   if(!terminal && !Execution.Observation(History,bar,observation))
     {
      Stop("invalid online observation");
      return;
     }
   if(!Execution.Boundary(bar,observation))
     {
      Stop("ledger boundary failed");
      return;
     }
   if(terminal)
      for(int k=0;k<ArraySize(Execution.records);k++)
         if(Execution.records[k].boundary && Execution.records[k].finish==bar && !Execution.records[k].used)
            ArrayInitialize(Execution.records[k].done,1);
   bool target_due=false;
   if(enable_critic_td)
     {
      int k=Execution.OldestReady();
      if(k>=0)
        {
         if(!Models.TD(Execution.records[k].z,Execution.records[k].actions,Execution.records[k].reward,Execution.records[k].done,Execution.records[k].next))
           {
            Stop("Critic TD update failed");
            return;
           }
         Execution.records[k].used=true;
         TDTick++;
         target_due=TDTick%(uint)target_period==0;
        }
     }
   Execution.Prune(enable_critic_td);
   PolicyTick++;
   if(enable_policy && PolicyTick%(uint)policy_period==0)
     {
      if(!Models.Policy())
        {
         Stop("previous-bar POLICY update failed");
         return;
        }
     }
   else
     {
      HabiRelease(Models.prior);
      HabiRelease(Models.decoder);
      Models.policy_live=false;
     }
   if(target_due && !Models.Sync())
     {
      Stop("target synchronization failed");
      return;
     }
   if(terminal)
     {
      if(!Models.Save())
        {
         Stop("Save failed");
         return;
        }
      Print("Habi known episode end: unresolved records are not fabricated as READY");
      ExpertRemove();
      return;
     }
   if(!Execution.Observation(History,bar,observation))
     {
      Stop("fresh online observation failed");
      return;
     }
   if(!Models.Predict(observation,enable_policy && (train_prior || train_decoder)))
     {
      Stop("new online generation failed");
      return;
     }
   Execution.CommitAccountBoundary();
   if(!Execution.Begin(observation,Models.last_z,Models.last_actions,bar))
     {
      Stop("new command packet failed");
      return;
     }
  }

//+------------------------------------------------------------------+
//| OnTradeTransaction.
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
// Reconciliation only: no intra-bar continuation or new prediction.
   Execution.DealEvent(trans.type,trans.deal);
   for(int n=0;n<ArraySize(Execution.slots);n++)
      Execution.ReconcileSlot(n);
  }

//+------------------------------------------------------------------+
//| OnDeinit.
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Stopping is not evidence of an episode terminal boundary.
   if(!Failed && Models.prior && !Models.Save())
      Print("Habi: Save on stop failed");
  }
