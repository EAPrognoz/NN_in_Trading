#property strict
#property version "0.26"
#include "Trajectory.mqh"
input uint StudyWorkBudgetMs=200;
// Scheduling budget, independent of model/batch geometry.
CHabiModels Models;
CHabiHistory History;
bool Running=false,Finished=false,Failed=false;
int Pass=0,Decision=-1;
ulong Examples=0,Skipped=0;

//+------------------------------------------------------------------+
//| OnInit.
//+------------------------------------------------------------------+
int OnInit()
  {
   if(StudyWorkBudgetMs<1 || !HabiStart(Models,History,0) || !History.Load())
      return INIT_FAILED;
   Decision=ArraySize(History.rates)-MathMax(HistoryBars,5)-1;
   if(!EventSetMillisecondTimer(20))
      return INIT_FAILED;
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| StudyStep.
//+------------------------------------------------------------------+
bool StudyStep()
  {
   if(Decision<MathMax(PlanningHorizonBars,NForecast)-1)
     {
      Pass++;
      if(Pass>=FitPasses)
        {
         Finished=true;
         EventKillTimer();
         if(!Models.Save())
           {
            Failed=true;
            HabiFatal("Study Save failed");
           }
         else
            PrintFormat("Habi Study complete: examples=%I64u skipped=%I64u",Examples,Skipped);
         return false;
        }
      Decision=ArraySize(History.rates)-MathMax(HistoryBars,5)-1;
     }
   int e=Decision--;
   SState record;
   SHabiSnapshot snapshot;
   float observation[],teacher[],z[],actions[];
   SHabiEvaluation evaluation;
   uint id=0;
   double teacher_return=0;
   if(!History.Sample(e,snapshot,record,observation) || !HabiTeacher(History,e,snapshot,observation,teacher,teacher_return))
     {
      Skipped++;
      return true;
     }
   if(!Models.BC(observation,teacher))
     {
      Failed=true;
      EventKillTimer();
      HabiFatal("Study BC update failed");
      return false;
     }
   for(int k=0;k<6;k++)
      record.action[k]=teacher[k];
   record.rewards[0]=(float)teacher_return;
   if(!Models.MC(observation,teacher,z,actions,id))
     {
      Failed=true;
      EventKillTimer();
      HabiFatal("Study posterior generation failed");
      return false;
     }
   if(!HabiHistoricalEvaluate(History,e,snapshot,actions,Models.g.N,evaluation))
     {
      Skipped++;
      return true;
     }
   if(!Models.LearnMC(z,actions,evaluation.target,id))
     {
      Failed=true;
      EventKillTimer();
      HabiFatal("Study Critic MC update failed");
      return false;
     }
   Examples++;
   return true;
  }

//+------------------------------------------------------------------+
//| OnTimer.
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(Running || Finished || Failed || IsStopped())
      return;
   Running=true;
   ulong started=GetMicrosecondCount(),budget=(ulong)StudyWorkBudgetMs*1000;
   do
     {
      if(IsStopped() || !StudyStep())
         break;
     }
   while(!Finished && !Failed && !IsStopped() && GetMicrosecondCount()-started<budget);
   Running=false;
  }

//+------------------------------------------------------------------+
//| OnDeinit.
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
