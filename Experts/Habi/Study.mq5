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
input int      Epochs=100;
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
const string  StudyStopButton="HABI_Study_Stop";
ulong         ExtBCUpdates=0;
double        ExtBCSquaredError=0;
double        ExtKLSum=0;
double        ExtBeta=0;
double        ExtMCSquaredError=0;
string        ExtStage="Инициализация моделей / OpenCL";

//+------------------------------------------------------------------+
//| StudyProgress - refreshes the chart without slowing each sample |
//+------------------------------------------------------------------+
string StudyMetrics()
  {
   string bc=ExtBCUpdates>0 ?
             StringFormat("BC Action6 RMSE=%.6g | KL(q||p)=%.6g | beta=%.6g | BC=%I64u",
                          MathSqrt(ExtBCSquaredError/ExtBCUpdates),ExtKLSum/ExtBCUpdates,
                          ExtBeta,ExtBCUpdates) : "BC Action6 RMSE / KL / beta: —";
   string mc=ExtExamples>0 ?
             StringFormat("Critic MC RMSE=%.6g | MC=%I64u",
                          MathSqrt(ExtMCSquaredError/ExtExamples),ExtExamples) : "Critic MC RMSE: —";
   return(bc+" | "+mc);
  }

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
                        "Время: %I64u:%02u:%02u | Скорость: %.2f прим./с\n%s",
                        _Symbol,EnumToString(TimeFrame==PERIOD_CURRENT?(ENUM_TIMEFRAMES)_Period:TimeFrame),
                        ExtStage,TimeToString(Start,TIME_DATE),TimeToString(End,TIME_DATE),
                        MathMin(ExtPass+1,MathMax(Epochs,1)),Epochs,percent,processed,ExtTotal,
                        ExtExamples,ExtSkipped,seconds/3600,(uint)((seconds/60)%60),(uint)(seconds%60),speed,StudyMetrics()));
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| OnInit - prepares modules and starts the study timer             |
//+------------------------------------------------------------------+
int OnInit()
  {
   ExtStartedMs=GetTickCount64();
   if(!ObjectCreate(0,StudyStopButton,OBJ_BUTTON,0,0,0) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_CORNER,CORNER_RIGHT_UPPER) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_XDISTANCE,200) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_YDISTANCE,5) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_XSIZE,180) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_YSIZE,28) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_COLOR,clrWhite) ||
      !ObjectSetInteger(0,StudyStopButton,OBJPROP_BGCOLOR,clrFireBrick) ||
      !ObjectSetString(0,StudyStopButton,OBJPROP_TEXT,"Остановить и сохранить"))
     {
      ExtFailed=true;
      ExtStage="Ошибка создания кнопки остановки";
      StudyProgress(true);
      return(INIT_FAILED);
     }

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
void StudyFinish(bool interrupted)
  {
   ExtFinished=true;
   EventKillTimer();
   ExtStage="Сохранение моделей";
   StudyProgress(true);

   if(!ExtModels.Save())
     {
      ExtStage="Ошибка сохранения моделей";
      ObjectSetInteger(0,StudyStopButton,OBJPROP_STATE,false);
      ObjectSetString(0,StudyStopButton,OBJPROP_TEXT,"Повторить сохранение");
      StudyProgress(true);
      Print("Habi Study: save failed, click the stop button to retry");
      return;
     }

   PrintFormat("Habi Study %s: %s %s, FIT %s - %s, epochs=%d/%d, examples=%I64u skipped=%I64u elapsed_ms=%I64u | %s",
               interrupted?"stopped":"complete",
               _Symbol,EnumToString(TimeFrame==PERIOD_CURRENT?(ENUM_TIMEFRAMES)_Period:TimeFrame),
               TimeToString(Start,TIME_DATE),TimeToString(End,TIME_DATE),
               ExtPass,Epochs,ExtExamples,ExtSkipped,GetTickCount64()-ExtStartedMs,StudyMetrics());
   ObjectDelete(0,StudyStopButton);
   Comment("");
   ChartRedraw();
   ExpertRemove();
  }

void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK && sparam==StudyStopButton && !ExtFailed && !IsStopped())
      StudyFinish(ExtPass<Epochs);
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
         StudyFinish(false);
         return(false);
        }

      ExtDecision=ArraySize(ExtHistory.rates)-MathMax(HistoryBars,5)-1;
     }

   int e=ExtDecision--;
   SState record;
   SHabiSnapshot snapshot;
   float observation[];
   float teacher[];
   float distribution[];
   float actions[];
   SHabiEvaluation evaluation;
   uint id=0;
   double teacher_return=0;

   if(!ExtHistory.Sample(e,snapshot,record,observation) || !HabiTeacher(ExtHistory,e,snapshot,observation,teacher,teacher_return))
     {
      ExtSkipped++;

      return(true);
     }

   double bc_mse=0,kl=0,beta=0;
   if(!ExtModels.BC(observation,teacher,bc_mse,kl,beta))
     {
      ExtFailed=true;
      ExtStage="Ошибка обучения BC";
      StudyProgress(true);
      EventKillTimer();
      HabiFatal("Study BC update failed");

      return(false);
     }

   ExtBCUpdates++;
   ExtBCSquaredError+=bc_mse;
   ExtKLSum+=kl;
   ExtBeta=beta;

   for(int k=0;k<6;k++)
      record.action[k]=teacher[k];

   record.rewards[0]=(float)teacher_return;

   if(!ExtModels.MC(observation,teacher,distribution,actions,id))
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

   double mc_mse=0;
   if(!ExtModels.LearnMC(distribution,actions,evaluation.target,id,mc_mse))
     {
      ExtFailed=true;
      ExtStage="Ошибка обучения Critic MC";
      StudyProgress(true);
      EventKillTimer();
      HabiFatal("Study Critic MC update failed");

      return(false);
     }

   ExtMCSquaredError+=mc_mse;
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
   if(!ExtFinished)
      StudyProgress();
  }
//+------------------------------------------------------------------+
//| OnDeinit - stops the study timer                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectDelete(0,StudyStopButton);
   ChartRedraw();
   if(!ExtFinished && !ExtFailed)
     {
      ExtStage="Остановлено";
      StudyProgress(true);
     }
  }

//+------------------------------------------------------------------+
