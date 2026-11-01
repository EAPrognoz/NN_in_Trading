//+------------------------------------------------------------------+
//|                                                       Study.mq5  |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, MetaQuotes Ltd."
#property link        "https://www.mql5.com"
#property version     "1.00"
#property description "Offline study pass for the Habi policy"
#property description "Fits prior, decoder and critic from history"
#property description "Runs bounded work chunks from a timer"
#property strict
#include "Trajectory.mqh"

//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
input uint StudyWorkBudgetMs=200;    // timer work chunk, ms

//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CHabiModels   ExtModels;          // model container
CHabiHistory  ExtHistory;         // history and sampling
bool          ExtRunning=false;   // timer reentry guard
bool          ExtFinished=false;  // study finished flag
bool          ExtFailed=false;    // fatal error flag
int           ExtPass=0;          // fitted pass counter
int           ExtDecision=-1;     // current sample index
ulong         ExtExamples=0;      // trained examples counter
ulong         ExtSkipped=0;       // skipped examples counter

//+------------------------------------------------------------------+
//| OnInit - prepares modules and starts the study timer             |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(StudyWorkBudgetMs<1 || !HabiStart(ExtModels,ExtHistory,0) || !ExtHistory.Load())
      return(INIT_FAILED);

   ExtDecision=ArraySize(ExtHistory.rates)-MathMax(HistoryBars,5)-1;

   if(!EventSetMillisecondTimer(20))
      return(INIT_FAILED);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| StudyStep - performs one training step on the current sample     |
//+------------------------------------------------------------------+
bool StudyStep()
  {
   if(ExtDecision<MathMax(PlanningHorizonBars,NForecast)-1)
     {
      ExtPass++;

      if(ExtPass>=FitPasses)
        {
         ExtFinished=true;
         EventKillTimer();

         if(!ExtModels.Save())
           {
            ExtFailed=true;
            HabiFatal("Study Save failed");
           }
         else
            PrintFormat("Habi Study complete: examples=%I64u skipped=%I64u",ExtExamples,ExtSkipped);

         return(false);
        }

      ExtDecision=ArraySize(ExtHistory.rates)-MathMax(HistoryBars,5)-1;
     }

   int e=ExtDecision--;
   SState record;
   SHabiSnapshot snapshot;
   float observation[];
   float teacher[];
   float z[];
   float actions[];
   SHabiEvaluation evaluation;
   uint id=0;
   double teacher_return=0;

   if(!ExtHistory.Sample(e,snapshot,record,observation) || !HabiTeacher(ExtHistory,e,snapshot,observation,teacher,teacher_return))
     {
      ExtSkipped++;

      return(true);
     }

   if(!ExtModels.BC(observation,teacher))
     {
      ExtFailed=true;
      EventKillTimer();
      HabiFatal("Study BC update failed");

      return(false);
     }

   for(int k=0;k<6;k++)
      record.action[k]=teacher[k];

   record.rewards[0]=(float)teacher_return;

   if(!ExtModels.MC(observation,teacher,z,actions,id))
     {
      ExtFailed=true;
      EventKillTimer();
      HabiFatal("Study posterior generation failed");

      return(false);
     }

   if(!HabiHistoricalEvaluate(ExtHistory,e,snapshot,actions,ExtModels.g.N,evaluation))
     {
      ExtSkipped++;

      return(true);
     }

   if(!ExtModels.LearnMC(z,actions,evaluation.target,id))
     {
      ExtFailed=true;
      EventKillTimer();
      HabiFatal("Study Critic MC update failed");

      return(false);
     }

   ExtExamples++;

   return(true);
  }
//+------------------------------------------------------------------+
//| OnTimer - runs study steps within the time budget                |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(ExtRunning || ExtFinished || ExtFailed || IsStopped())
      return;

   ExtRunning=true;
   ulong started=GetMicrosecondCount();
   ulong budget=(ulong)StudyWorkBudgetMs*1000;

   do
     {
      if(IsStopped() || !StudyStep())
         break;
     }
   while(!ExtFinished && !ExtFailed && !IsStopped() && GetMicrosecondCount()-started<budget);

   ExtRunning=false;
  }
//+------------------------------------------------------------------+
//| OnDeinit - stops the study timer                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+
