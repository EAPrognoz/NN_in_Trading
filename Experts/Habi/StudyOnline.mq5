//+------------------------------------------------------------------+
//|                                                 StudyOnline.mq5  |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, MetaQuotes Ltd."
#property link        "https://www.mql5.com"
#property version     "1.00"
#property description "Online study and trading loop for Habi"
#property description "Updates critic and policy on closed bars"
#property description "Keeps ledger records across bar boundaries"
#property strict
#include "Trajectory.mqh"

//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CHabiModels      ExtModels;          // model container
CHabiHistory     ExtHistory;         // history container
CHabiExecution   ExtExecution;       // execution module
datetime         ExtLastBar=0;       // last processed bar
uint             ExtPolicyTick=0;    // policy update counter
uint             ExtTDTick=0;        // critic TD update counter
bool             ExtFailed=false;    // fatal error flag

//+------------------------------------------------------------------+
//| OnInit - initialises modules for the online stage                |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!HabiStart(ExtModels,ExtHistory,1) || !ExtExecution.Init(ExtModels.g.N,false))
      return(INIT_FAILED);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Stop - marks a fatal state and removes the program               |
//+------------------------------------------------------------------+
void Stop(string operation)
  {
   ExtFailed=true;
   HabiFatal(operation);
  }
//+------------------------------------------------------------------+
//| OnTick - updates the online modules once per closed bar          |
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

   bool terminal=OnlineEpisodeEnd>0 && bar>=OnlineEpisodeEnd;
   float observation[];

   if(!terminal && !ExtExecution.Observation(ExtHistory,bar,observation))
     {
      Stop("invalid online observation");

      return;
     }

   if(!ExtExecution.Boundary(bar,observation))
     {
      Stop("ledger boundary failed");

      return;
     }

   if(terminal)
      for(int k=0;k<ArraySize(ExtExecution.records);k++)
         if(ExtExecution.records[k].boundary && ExtExecution.records[k].finish==bar && !ExtExecution.records[k].used)
            ArrayInitialize(ExtExecution.records[k].done,1);

   bool target_due=false;

   if(enable_critic_td)
     {
      int k=ExtExecution.OldestReady();

      if(k>=0)
        {
         if(!ExtModels.TD(ExtExecution.records[k].distribution,ExtExecution.records[k].actions,ExtExecution.records[k].reward,ExtExecution.records[k].done,ExtExecution.records[k].next))
           {
            Stop("Critic TD update failed");

            return;
           }

         ExtExecution.records[k].used=true;
         ExtTDTick++;
         target_due=ExtTDTick%(uint)target_period==0;
        }
     }

   ExtExecution.Prune(enable_critic_td);
   ExtPolicyTick++;

   if(enable_policy && ExtPolicyTick%(uint)policy_period==0)
     {
      if(!ExtModels.Policy())
        {
         Stop("previous-bar POLICY update failed");

         return;
        }
     }
   else
     {
      HabiRelease(ExtModels.prior);
      HabiRelease(ExtModels.decoder);
      ExtModels.policy_live=false;
     }

   if(target_due && !ExtModels.Sync())
     {
      Stop("target synchronization failed");

      return;
     }

   if(terminal)
     {
      if(!ExtModels.Save())
        {
         Stop("Save failed");

         return;
        }

      Print("Habi known episode end: unresolved records are not fabricated as READY");
      ExpertRemove();

      return;
     }

   if(!ExtExecution.Observation(ExtHistory,bar,observation))
     {
      Stop("fresh online observation failed");

      return;
     }

   if(!ExtModels.Predict(observation,enable_policy && (train_prior || train_decoder)))
     {
      Stop("new online generation failed");

      return;
     }

   ExtExecution.CommitAccountBoundary();

   if(!ExtExecution.Begin(observation,ExtModels.last_distribution,ExtModels.last_actions,bar))
     {
      Stop("new command packet failed");

      return;
     }
  }
//+------------------------------------------------------------------+
//| OnTradeTransaction - forwards broker deals to the ledger         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
//--- reconciliation only: no intra-bar continuation or new prediction
   ExtExecution.DealEvent(trans.type,trans.deal);

   for(int n=0;n<ArraySize(ExtExecution.slots);n++)
      ExtExecution.ReconcileSlot(n);
  }
//+------------------------------------------------------------------+
//| OnDeinit - saves the model when the program stops                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- stopping is not evidence of an episode terminal boundary
   if(!ExtFailed && ExtModels.prior!=NULL && !ExtModels.Save())
      Print("Habi: Save on stop failed");
  }

//+------------------------------------------------------------------+
