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
input int                  Iterations     = 1e4;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Agent;
CNet                 Planner;
CNet                 FutureEmbedding;
CNet                 RTG;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
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
   if(!Agent.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Planner.Load(FileName + "Pln.nnw", temp, temp, temp, dtStudied, true) ||
      !FutureEmbedding.Load(FileName + "FEm.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *agent = new CArrayObj();
      CArrayObj *planner = new CArrayObj();
      CArrayObj *future_embedding = new CArrayObj();
      if(!CreateDescriptions(agent, planner, future_embedding))
        {
         delete agent;
         delete planner;
         delete future_embedding;
         return INIT_FAILED;
        }
      if(!Agent.Create(agent) || !Planner.Create(planner) ||
         !FutureEmbedding.Create(future_embedding))
        {
         delete agent;
         delete planner;
         delete future_embedding;
         return INIT_FAILED;
        }
      delete agent;
      delete planner;
      delete future_embedding;
      //---
     }
//---
   if(!RTG.Load(FileName + "RTG.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *rtg = new CArrayObj();
      if(!CreateValueDescriptions(rtg))
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
   COpenCL *opcl = Agent.GetOpenCL();
   Planner.SetOpenCL(opcl);
   FutureEmbedding.SetOpenCL(opcl);
   RTG.SetOpenCL(opcl);
//---
   Agent.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the Agent does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   RTG.getResults(Result);
   if(Result.Total() != NRewards)
     {
      PrintFormat("The scope of the RTG does not match the rewards count (%d <> %d)", NRewards, Result.Total());
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
   Planner.Save(FileName + "Pln.nnw", 0, 0, 0, TimeCurrent(), true);
   FutureEmbedding.Save(FileName + "FEm.nnw", 0, 0, 0, TimeCurrent(), true);
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
//| Train function                                                 |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   float err = 0;
   int err_count = 0;
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr].Total - 2 * HistoryBars - ValueBars, MathMin(Buffer[tr].Total, 20 + ValueBars)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Actions = vector<float>::Zeros(NActions);
      for(int state = i; state < MathMin(Buffer[tr].Total - 2 - ValueBars, i + HistoryBars * 3); state++)
        {
         //--- History data
         State.AssignArray(Buffer[tr].States[state].state);
         if(!Planner.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
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
         State.AddArray(Actions);
         //--- Target
         Result.AssignArray(Buffer[tr].States[state + 1].state);
         for(int s = 1; s < ValueBars; s++)
            Result.AddArray(Buffer[tr].States[state + 1].state);
         if(!FutureEmbedding.feedForward(Result, 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         FutureEmbedding.getResults(Result);
         //--- Policy Feed Forward
         if(!Agent.feedForward(GetPointer(State), 1, false, Result))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Return-To-Go
         Account.AssignArray(Buffer[tr].States[state + 1].account);
         if(!RTG.feedForward(GetPointer(Account), 1, false, Result))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Planner Study
         if(!Planner.backProp(Result, NULL, NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy study
         Actions.Assign(Buffer[tr].States[state].action);
         vector<float> result;
         Agent.getResults(result);
         Result.AssignArray(CAGrad(Actions - result) + result);
         if(!Agent.backProp(Result, GetPointer(FutureEmbedding)) ||
            !FutureEmbedding.backPropGradient((CBufferFloat *)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Return To Go study
         vector<float> target;
         target.Assign(Buffer[tr].States[state + 1].rewards);
         result.Assign(Buffer[tr].States[state + ValueBars].rewards);
         target = target - result * MathPow(DiscFactor, ValueBars);
         Result.AssignArray(target);
         if(!RTG.backProp(Result, GetPointer(FutureEmbedding)) ||
            !FutureEmbedding.backPropGradient((CBufferFloat *)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Agent", iter * 100.0 / (double)(Iterations), Agent.getRecentAverageError());
            str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Planner", iter * 100.0 / (double)(Iterations), Planner.getRecentAverageError());
            str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "RTG", iter * 100.0 / (double)(Iterations), RTG.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Agent", Agent.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Planner", Planner.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "RTG", RTG.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> CAGrad(vector<float> &grad)
  {
   matrix<float> GG = grad.Outer(grad);
   GG.ReplaceNan(0);
   if(MathAbs(GG).Sum() == 0)
      return grad;
   float scale = MathSqrt(GG.Diag() + 1.0e-4f).Mean();
   GG = GG / MathPow(scale, 2);
   vector<float> Gg = GG.Mean(1);
   float gg = Gg.Mean();
   vector<float> w = vector<float>::Zeros(grad.Size());
   float c = MathSqrt(gg + 1.0e-4f) * fCAGrad_C;
   vector<float> w_best = w;
   float obj_best = FLT_MAX;
   vector<float> moment = vector<float>::Zeros(w.Size());
   for(int i = 0; i < iCAGrad_Iters; i++)
     {
      vector<float> ww;
      w.Activation(ww, AF_SOFTMAX);
      float obj = ww.Dot(Gg) + c * MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f);
      if(MathAbs(obj) < obj_best)
        {
         obj_best = MathAbs(obj);
         w_best = w;
        }
      if(i < (iCAGrad_Iters - 1))
        {
         float loss = -obj;
         vector<float> derev = Gg + GG.MatMul(ww) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2) + ww.MatMul(GG) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2);
         vector<float> delta = derev * loss;
         ulong size = delta.Size();
         matrix<float> ident = matrix<float>::Identity(size, size);
         vector<float> ones = vector<float>::Ones(size);
         matrix<float> sm_der = ones.Outer(ww);
         sm_der = sm_der.Transpose() * (ident - sm_der);
         delta = sm_der.MatMul(delta);
         if(delta.Ptp() != 0)
            delta = delta / delta.Ptp();
         moment = delta * 0.8f + moment * 0.5f;
         w += moment;
         if(w.Ptp() != 0)
            w = w / w.Ptp();
        }
     }
   w_best.Activation(w, AF_SOFTMAX);
   float gw_norm = MathSqrt(w.MatMul(GG).Dot(w) + 1.0e-4f);
   float lmbda = c / (gw_norm + 1.0e-4f);
   vector<float> result = ((w * lmbda + 1.0f / (float)grad.Size()) * grad) / (1 + MathPow(fCAGrad_C, 2));
//---
   return result;
  }
//+------------------------------------------------------------------+

