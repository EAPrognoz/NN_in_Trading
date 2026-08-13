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
input int                  Iterations     = 1e7;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 RTG;
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
   if(!RTG.Load(FileName + "RTG.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *rtg = new CArrayObj();
      if(!CreateRTGDescriptions(rtg))
        {
         delete rtg;
         return INIT_FAILED;
        }
      if(!RTG.Create(rtg))
        {
         delete rtg;
         return INIT_FAILED;
        }
      delete rtg;
      //---
     }
//---
   RTG.getResults(Result);
   if(Result.Total() != NRewards)
     {
      PrintFormat("The scope of the RTG-model does not match the rewards count (%d <> %d)", NRewards, Result.Total());
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
   RTG.Save(FileName + "RTG.nnw", 0, 0, 0, TimeCurrent(), true);
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
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 * ValueBars));
      if(i < 0)
        {
         iter--;
         continue;
        }
      //--- History data
      State.AssignArray(Buffer[tr].States[i].state);
      for(int state = 1; state < ValueBars; state++)
         State.AddArray(Buffer[tr].States[i + state].state);
      //--- Study
      if(!RTG.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         StopFlag = true;
         break;
        }
      vector<float> target, result;
      target.Assign(Buffer[tr].States[i + ValueBars].rewards);
      result.Assign(Buffer[tr].States[i + 2 * ValueBars - 1].rewards);
      target = target - result*MathPow(DiscFactor,ValueBars);
      Result.AssignArray(target);
      if(!RTG.backProp(Result, (CBufferFloat *)NULL, (CBufferFloat *)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         StopFlag = true;
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "RTG-model", iter * 100.0 / (double)(Iterations), RTG.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Value", RTG.getRecentAverageError());
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
      vector<float> multipl=exp(MathAbs(result - max_reward) / (result.Percentile(90)-max_reward));
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

