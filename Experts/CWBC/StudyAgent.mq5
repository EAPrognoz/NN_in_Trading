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
input int                  Iterations     = 1000;
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
      if(!CreateDescriptions(agent))
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
   if(Result.Total() != (NRewards + BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions))
     {
      PrintFormat("Input size of Agent doesn't match state description (%d <> %d)", Result.Total(), (NRewards + BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions));
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
   float max_reward = 0, quanitle = 0;
   vector<float> std;
   vector<float> probability = GetProbTrajectories(Buffer, max_reward, quanitle, std, 0.95, 0.1f);
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr].Total - 2 * HistoryBars - ValueBars, MathMin(Buffer[tr].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Actions = vector<float>::Zeros(NActions);
      Agent.Clear();
      for(int state = i; state < MathMin(Buffer[tr].Total - 1 - ValueBars, i + HistoryBars * 3); state++)
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
         //--- Return to go
         vector<float> target, result;
         vector<float> noise = vector<float>::Zeros(NRewards);
         target.Assign(Buffer[tr].States[0].rewards);
         if(target.Sum() >= quanitle)
            noise = Noise(std, 100);
         target.Assign(Buffer[tr].States[state + 1].rewards);
         result.Assign(Buffer[tr].States[state + ValueBars].rewards);
         target = target - result * MathPow(DiscFactor, ValueBars) + noise;
         State.AddArray(target);
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets ProbTrajectories.                                         |
//+------------------------------------------------------------------+
vector<float> GetProbTrajectories(STrajectory &buffer[], float &max_reward, float &quanitle, vector<float> &std, double quant, float lanbda)
  {
   ulong total = buffer.Size();
   matrix<float> rewards = matrix<float>::Zeros(total, NRewards);
   vector<float> result;
   for(ulong i = 0; i < total; i++)
     {
      result.Assign(buffer[i].States[0].rewards);
      rewards.Row(result, i);
     }
   std = rewards.Std(0);
   result = rewards.Sum(1);
   max_reward = result.Max();
//---
   vector<float> sorted = result;
   bool sort = true;
   int iter = 0;
   while(sort)
     {
      sort = false;
      for(ulong i = 0; i < sorted.Size() - 1; i++)
         if(sorted[i] > sorted[i + 1])
           {
            float temp = sorted[i];
            sorted[i] = sorted[i + 1];
            sorted[i + 1] = temp;
            sort = true;
           }
      iter++;
     }
   quanitle = sorted.Quantile(quant);
//---
   float min = result.Min() - 0.1f * std.Sum();
   if(max_reward > min)
     {
      float k=result.Percentile(90) - max_reward;
      vector<float> multipl = MathAbs(result - max_reward)/ (k==0 ? -std.Sum() : k);
      multipl=exp(multipl);
      result = (result - min) / (max_reward - min);
      result = result / (result + lanbda) * multipl;
      result.ReplaceNan(0);
     }
   else
      result.Fill(1);
   result = result / result.Sum();
   result = result.CumSum();
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements SampleTrajectory.                                   |
//+------------------------------------------------------------------+
int SampleTrajectory(vector<float> &probability)
  {
//--- check
   ulong total = probability.Size();
   if(total <= 0)
      return -1;
//--- randomize
   float rnd = float(MathRand() / 32767.0);
//--- search
   if(rnd <= probability[0] || total == 1)
      return 0;
   if(rnd > probability[total - 2])
      return int(total - 1);
   int result = int(rnd * total);
   if(probability[result] < rnd)
      while(probability[result] < rnd)
         result++;
   else
      while(probability[result - 1] >= rnd)
         result--;
//--- return result
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements Noise.                                              |
//+------------------------------------------------------------------+
vector<float> Noise(vector<float> &std, float multiplyer)
  {
//--- check
   ulong total = std.Size();
   if(total <= 0)
      return vector<float>::Zeros(0);
//---
   vector<float> result = vector<float>::Zeros(total);
   for(ulong i = 0; i < total; i++)
     {
      float rnd = float(MathRand() / 32767.0);
      result[i] = std[i] * rnd * multiplyer;
     }
//--- return result
   return result;
  }
//+------------------------------------------------------------------+

