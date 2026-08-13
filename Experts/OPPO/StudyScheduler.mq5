//+------------------------------------------------------------------+
//| Study.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define Study
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 1e3;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Scheduler;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         *Result;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
//--- load models
   float temp;
   if(!Scheduler.Load(FileName + "Sch.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *scheduler = new CArrayObj();
      if(!CreateSchedulerDescriptions(scheduler))
        {
         delete scheduler;
         return INIT_FAILED;
        }
      if(!Scheduler.Create(scheduler))
        {
         delete scheduler;
         return INIT_FAILED;
        }
      delete scheduler;
      //---
     }
//---
   Scheduler.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Scheduler-model does not match the Embedding count (%d <> %d)", EmbeddingSize, Result.Total());
      return INIT_FAILED;
     }
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Scheduler.Save(FileName + "Sch.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Result;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//| Train function                                                 |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
   vector<double> probability = GetProbTrajectories(Buffer, 0.1f);
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      int tr_p = SampleTrajectory(probability);
      int tr_m = SampleTrajectory(probability);
      while(tr_p == tr_m)
         tr_m = SampleTrajectory(probability);
      if(Buffer[tr_p].Profit < Buffer[tr_m].Profit)
        {
         int t = tr_p;
         tr_p = tr_m;
         tr_m = t;
        }
      //--- Positive
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr_p].Total - 2 * HistoryBars - NBarInPattern, MathMin(Buffer[tr_p].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Scheduler.Clear();
      for(int state = i; state < MathMin(Buffer[tr_p].Total - 1 - NBarInPattern, i + HistoryBars * 2); state++)
        {
         //--- History data
         State.AssignArray(Buffer[tr_p].States[state].state);
         //--- Account description
         float PrevBalance = (state == 0 ? Buffer[tr_p].States[state].account[0] : Buffer[tr_p].States[state - 1].account[0]);
         float PrevEquity = (state == 0 ? Buffer[tr_p].States[state].account[1] : Buffer[tr_p].States[state - 1].account[1]);
         State.Add((Buffer[tr_p].States[state].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr_p].States[state].account[1] / PrevBalance);
         State.Add((Buffer[tr_p].States[state].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr_p].States[state].account[2]);
         State.Add(Buffer[tr_p].States[state].account[3]);
         State.Add(Buffer[tr_p].States[state].account[4] / PrevBalance);
         State.Add(Buffer[tr_p].States[state].account[5] / PrevBalance);
         State.Add(Buffer[tr_p].States[state].account[6] / PrevBalance);
         //--- Time label
         double x = (double)Buffer[tr_p].States[state].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr_p].States[state].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(2.0 * M_PI * x));
         x = (double)Buffer[tr_p].States[state].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr_p].States[state].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         //--- Prev action
         if(state > 0)
            State.AddArray(Buffer[tr_p].States[state - 1].action);
         else
            State.AddArray(vector<float>::Zeros(NActions));
         //--- Feed Forward
         if(!Scheduler.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Study
         Result.AssignArray(Buffer[tr_p].States[state].scheduler);
         if(!Scheduler.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Scheeduler", iter * 100.0 / (double)(Iterations), Scheduler.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
      //--- Negotive
      i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr_m].Total - 2 * HistoryBars - NBarInPattern, MathMin(Buffer[tr_m].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Scheduler.Clear();
      for(int state = i; state < MathMin(Buffer[tr_m].Total - 1 - NBarInPattern, i + HistoryBars * 2); state++)
        {
         //--- History data
         State.AssignArray(Buffer[tr_m].States[state].state);
         //--- Account description
         float PrevBalance = (state == 0 ? Buffer[tr_m].States[state].account[0] : Buffer[tr_m].States[state - 1].account[0]);
         float PrevEquity = (state == 0 ? Buffer[tr_m].States[state].account[1] : Buffer[tr_m].States[state - 1].account[1]);
         State.Add((Buffer[tr_m].States[state].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr_m].States[state].account[1] / PrevBalance);
         State.Add((Buffer[tr_m].States[state].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr_m].States[state].account[2]);
         State.Add(Buffer[tr_m].States[state].account[3]);
         State.Add(Buffer[tr_m].States[state].account[4] / PrevBalance);
         State.Add(Buffer[tr_m].States[state].account[5] / PrevBalance);
         State.Add(Buffer[tr_m].States[state].account[6] / PrevBalance);
         //--- Time label
         double x = (double)Buffer[tr_m].States[state].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr_m].States[state].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(2.0 * M_PI * x));
         x = (double)Buffer[tr_m].States[state].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr_m].States[state].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         //--- Prev action
         if(state > 0)
            State.AddArray(Buffer[tr_m].States[state - 1].action);
         else
            State.AddArray(vector<float>::Zeros(NActions));
         //--- Feed Forward
         if(!Scheduler.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Study
         if(Buffer[tr_p].Profit == Buffer[tr_m].Profit)
            Result.AssignArray(Buffer[tr_m].States[state].scheduler);
         else
           {
            vector<float> target, forecast;
            target.Assign(Buffer[tr_m].States[state].scheduler);
            Scheduler.getResults(forecast);
            target = forecast - (target - forecast) / 2;
            Result.AssignArray(target);
           }
         if(!Scheduler.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Scheeduler", (iter + 0.5) * 100.0 / (double)(Iterations), Scheduler.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Scheduler", Scheduler.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

