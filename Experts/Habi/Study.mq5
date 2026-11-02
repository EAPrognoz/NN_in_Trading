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
//--- Offline settings belong only to this program.
#define HABI_OFFLINE_STUDY
input group    "---- Study ----"
input datetime Start=D'2024.01.01';
input datetime End=D'2026.01.01';
input int      Epochs=1;
input double   MinBalance=50;
input double   MaxBalance=150;
#include "Trajectory.mqh"

//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
const uint StudyWorkBudgetMs=200;    // timer work chunk, ms

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
ulong         ExtStartedMs=0;     // elapsed wall time, including initialization
ulong         ExtShownMs=0;       // last chart update
ulong         ExtTotal=0;         // planned sample decisions across all passes
string        ExtStage="Инициализация моделей / OpenCL";

//+------------------------------------------------------------------+
//| StudyProgress - refreshes the chart without slowing each sample |
//+------------------------------------------------------------------+
void StudyProgress(bool force=false)
  {
   ulong now=GetTickCount64();
   if(!force && now-ExtShownMs<500)
      return;
   ExtShownMs=now;
   ulong processed=ExtExamples+ExtSkipped;
   double percent=ExtTotal>0?MathMin(100.0,100.0*(double)processed/(double)ExtTotal):0;
   ulong seconds=(now-ExtStartedMs)/1000;
   double speed=seconds>0?(double)ExtExamples/(double)seconds:0;
   Comment(StringFormat("HABI Study | %s %s\nЭтап: %s\nFIT: %s — %s\n"+
                        "Проход: %d/%d | Прогресс: %.2f%% (%I64u/%I64u)\n"+
                        "Обучено: %I64u | Пропущено: %I64u\n"+
                        "Время: %I64u:%02u:%02u | Скорость: %.2f прим./с",
                        _Symbol,EnumToString(TimeFrame==PERIOD_CURRENT?(ENUM_TIMEFRAMES)_Period:TimeFrame),
                        ExtStage,TimeToString(Start,TIME_DATE),TimeToString(End,TIME_DATE),
                        MathMin(ExtPass+1,MathMax(Epochs,1)),Epochs,percent,processed,ExtTotal,
                        ExtExamples,ExtSkipped,seconds/3600,(uint)((seconds/60)%60),(uint)(seconds%60),speed));
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| OnInit - prepares modules and starts the study timer             |
//+------------------------------------------------------------------+
int OnInit()
  {
   ExtStartedMs=GetTickCount64();
   StudyProgress(true);
   if(StudyWorkBudgetMs<1 || !HabiStart(ExtModels,ExtHistory,0))
     {
      ExtFailed=true;
      ExtStage="Ошибка инициализации моделей / OpenCL";
      StudyProgress(true);
      return(INIT_FAILED);
     }

   ExtStage="Загрузка истории";
   StudyProgress(true);
   if(!ExtHistory.Load())
     {
      ExtFailed=true;
      ExtStage="Ошибка загрузки истории";
      StudyProgress(true);
      return(INIT_FAILED);
     }

   ExtDecision=ArraySize(ExtHistory.rates)-MathMax(HistoryBars,5)-1;
   ExtTotal=(ulong)MathMax(0,ExtDecision-MathMax(PlanningHorizonBars,NForecast)+2)*(ulong)Epochs;

   if(!EventSetMillisecondTimer(20))
     {
      ExtFailed=true;
      ExtStage="Ошибка запуска таймера";
      StudyProgress(true);
      return(INIT_FAILED);
     }

   ExtStage="Обучение BC / MC";
   StudyProgress(true);
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

      if(ExtPass>=Epochs)
        {
         ExtFinished=true;
         EventKillTimer();
         ExtStage="Сохранение моделей";
         StudyProgress(true);

         if(!ExtModels.Save())
           {
            ExtFailed=true;
            ExtStage="Ошибка сохранения моделей";
            StudyProgress(true);
            HabiFatal("Study Save failed");
           }
         else
           {
            ExtStage="Завершено: модели сохранены";
            StudyProgress(true);
            PrintFormat("Habi Study complete: examples=%I64u skipped=%I64u",ExtExamples,ExtSkipped);
           }

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
      ExtStage="Ошибка обучения BC";
      StudyProgress(true);
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
      ExtStage="Ошибка генерации Posterior";
      StudyProgress(true);
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
      ExtStage="Ошибка обучения Critic MC";
      StudyProgress(true);
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
   StudyProgress();
  }
//+------------------------------------------------------------------+
//| OnDeinit - stops the study timer                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(!ExtFinished && !ExtFailed)
     {
      ExtStage="Остановлено";
      StudyProgress(true);
     }
  }

//+------------------------------------------------------------------+
