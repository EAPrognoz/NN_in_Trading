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
CNet                 Latent;
CNet                 Agent;
CNet                 World;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         *Result;
CBufferFloat         *Result2;
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
      !World.Load(FileName + "Wld.nnw", temp, temp, temp, dtStudied, true) ||
      !Latent.Load(FileName + "Lat.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *agent = new CArrayObj();
      CArrayObj *latent = new CArrayObj();
      CArrayObj *world = new CArrayObj();
      if(!CreateDescriptions(agent, latent, world))
        {
         delete agent;
         delete latent;
         delete world;
         return INIT_FAILED;
        }
      if(!Agent.Create(agent) ||
         !World.Create(world) ||
         !Latent.Create(latent))
        {
         delete agent;
         delete latent;
         delete world;
         return INIT_FAILED;
        }
      delete agent;
      delete latent;
      delete world;
      //---
     }
//---
   COpenCL *opcl = Agent.GetOpenCL();
   Latent.SetOpenCL(opcl);
   World.SetOpenCL(opcl);
//---
   Agent.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the Agent does not match the actions count (%d <> %d)", 6, Result.Total());
      return INIT_FAILED;
     }
//---
   Latent.GetLayerOutput(0, Result);
   if(Result.Total() != (BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions))
     {
      PrintFormat("Input size of Latent model doesn't match state description (%d <> %d)", Result.Total(), (BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions));
      return INIT_FAILED;
     }
   Latent.Clear();
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
   World.Save(FileName + "Wld.nnw", 0, 0, 0, TimeCurrent(), true);
   Latent.Save(FileName + "Lat.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Result;
   delete Result2;
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
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr].Total - 2 * HistoryBars, MathMin(Buffer[tr].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      Actions = vector<float>::Zeros(NActions);
      Latent.Clear();
      for(int state = i; state < MathMin(Buffer[tr].Total - 2, i + HistoryBars * 3); state++)
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
         State.AddArray(Actions);
         //--- Latent and Wordl
         if(!Latent.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL) ||
            !World.feedForward(GetPointer(Latent), -1, GetPointer(Latent), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         Actions.Assign(Buffer[tr].States[state].rewards);
         vector<float> result;
         World.getResults(result);
         Result.AssignArray(CAGrad(Actions - result) + result);
         if(!World.backProp(Result, GetPointer(Latent), LatentLayer) ||
            !Latent.backPropGradient((CBufferFloat *)NULL, (CBufferFloat *)NULL, LatentLayer) ||
            !Latent.backPropGradient((CBufferFloat *)NULL, (CBufferFloat *)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy Feed Forward
         Result.AssignArray(Buffer[tr].States[state + 1].latent);
         Latent.GetLayerOutput(LatentLayer, Result2);
         if(Result2.GetIndex() >= 0)
            Result2.BufferWrite();
         if(!World.feedForward(Result, 1, false, Result2) ||
            !Agent.feedForward(GetPointer(World), 2, (CBufferFloat *)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy study
         Actions.Assign(Buffer[tr].States[state].action);
         Agent.getResults(result);
         Result.AssignArray(CAGrad(Actions - result) + result);
         if(!Agent.backProp(Result, NULL, NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Agent", iter * 100.0 / (double)(Iterations), Agent.getRecentAverageError());
            str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "World", iter * 100.0 / (double)(Iterations), World.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Agent", Agent.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "World", World.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for CAGrad.               |
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

