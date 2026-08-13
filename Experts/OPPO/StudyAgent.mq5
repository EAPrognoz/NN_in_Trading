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
CNet                 Agent;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         *Result;
vector<float>        Actions;
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
   if(!Agent.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *agent = new CArrayObj();
      if(!CreateAgentDescriptions(agent))
        {
         delete agent;
         return INIT_FAILED;
        }
      if(!Agent.Create(agent))
        {
         delete agent;
         return INIT_FAILED;
        }
      delete agent;
     }
//---
   Agent.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the Agent does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Agent.GetLayerOutput(0, Result);
   if(Result.Total() != (BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions + EmbeddingSize))
     {
      PrintFormat("Input size of Agent doesn't match state description (%d <> %d)", Result.Total(), (BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions + EmbeddingSize));
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
   Agent.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
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
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   vector<double> probability = GetProbTrajectories(Buffer, 0.1f);
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr].Total - 2 * HistoryBars - NBarInPattern, MathMin(Buffer[tr].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Agent.Clear();
      for(int state = i; state < MathMin(Buffer[tr].Total - 1 - NBarInPattern, i + HistoryBars * 2); state++)
        {
         //--- History data
         State.AssignArray(Buffer[tr].States[state].state);
         //--- Account description
         float PrevBalance = (state == 0 ? Buffer[tr].States[state].account[0] : Buffer[tr].States[state - 1].account[0]);
         float PrevEquity = (state == 0 ? Buffer[tr].States[state].account[1] : Buffer[tr].States[state - 1].account[1]);
         State.Add((Buffer[tr].States[state].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr].States[state].account[1] / PrevBalance);
         State.Add((Buffer[tr].States[state].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr].States[state].account[2]);
         State.Add(Buffer[tr].States[state].account[3]);
         State.Add(Buffer[tr].States[state].account[4] / PrevBalance);
         State.Add(Buffer[tr].States[state].account[5] / PrevBalance);
         State.Add(Buffer[tr].States[state].account[6] / PrevBalance);
         //--- Time label
         double x = (double)Buffer[tr].States[state].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(2.0 * M_PI * x));
         //--- Prev action
         if(state > 0)
            State.AddArray(Buffer[tr].States[state - 1].action);
         else
            State.AddArray(vector<float>::Zeros(NActions));
         //--- Scheduler
         State.AddArray(Buffer[tr].States[state].scheduler);
         //--- Feed Forward
         if(!Agent.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy study
         Result.AssignArray(Buffer[tr].States[state].action);
         if(!Agent.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Agent", iter * 100.0 / (double)(Iterations), Agent.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Agent", Agent.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

