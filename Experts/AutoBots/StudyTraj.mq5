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
CNet                 Autobot;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat        *Result;
CBufferFloat         Ones;
CBufferFloat         Gradient;
vector<float>        STE;
vector<float>        STE_Noise;
//---
COpenCLMy           *OpenCL;
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
   if(!Autobot.Load(FileName + "Traj.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *autobot = new CArrayObj();
      if(!CreateTrajNetDescriptions(autobot))
        {
         delete autobot;
         return INIT_FAILED;
        }
      if(!Autobot.Create(autobot))
        {
         delete autobot;
         return INIT_FAILED;
        }
      delete autobot;
      //---
     }
//---
   Autobot.getResults(Result);
   if(Result.Total() != PrecoderBars * 3)
     {
      PrintFormat("The scope of the Autobot does not match the precoder bars (%d <> %d)", PrecoderBars * 3, Result.Total());
      return INIT_FAILED;
     }
//---
   Autobot.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Autobot doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   OpenCL = Autobot.GetOpenCL();
   if(!Ones.BufferInit(EmbeddingSize, 1) ||
      !Gradient.BufferInit(EmbeddingSize, 0) ||
      !Ones.BufferCreate(OpenCL) ||
      !Gradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
   State.BufferInit(HistoryBars * BarDescr, 0);
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
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
      Autobot.Save(FileName + "Traj.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Result;
   delete OpenCL;
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
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target, inp;
   matrix<float> targets;
   matrix<float> delta;
   STE = vector<float>::Zeros(PrecoderBars * 3);
   int std_count = 0;
   int batch = GPTBars + 50;
   bool Stop = false;
   uint ticks = GetTickCount();
   ulong size = HistoryBars * BarDescr;
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3 - PrecoderBars - batch));
      if(state < 0)
        {
         iter--;
         continue;
        }
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      Autobot.Clear();
      delta = matrix<float>::Zeros(end - state - 1, Buffer[tr].States[state].state.Size());
      for(int i = state; i < end; i++)
        {
         inp.Assign(Buffer[tr].States[i].state);
         State.AssignArray(inp);
         int row = i - state;
         if(i < (end - 1))
            delta.Row(inp, row);
         if(row > 0)
            delta.Row(delta.Row(row - 1) - inp, row - 1);
         //---
         if(!Autobot.feedForward((CBufferFloat*)GetPointer(State), 1, false, (CBufferFloat*)GetPointer(Ones)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         targets = matrix<float>::Zeros(PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + 1 + t].state);
            if(size > BarDescr)
              {
               matrix<float> temp(1, size);
               temp.Row(target, 0);
               temp.Reshape(size / BarDescr, BarDescr);
               temp.Resize(size / BarDescr, 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, t);
           }
         targets.Reshape(1, targets.Rows()*targets.Cols());
         target = targets.Row(0);
         Autobot.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STE = MathSqrt((MathPow(STE, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         vector<float> check = MathAbs(error) - STE * STE_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(target);
            if(!Autobot.backProp(Result, GetPointer(Ones), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / (2 * (end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Autobot", percent, Autobot.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
      //--- With noise
      vector<float> std_delta = delta.Std(0) * STD_Delta_Multiplier;
      vector<float> mean_delta = delta.Mean(0);
      ulong inp_total = std_delta.Size();
      vector<float> noise = vector<float>::Zeros(inp_total);
      double ar_noise[];
      //---
      tr = SampleTrajectory(probability);
      state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3 - PrecoderBars - batch));
      if(state < 0)
        {
         iter--;
         continue;
        }
      end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      Autobot.Clear();
      for(int i = state; i < end; i++)
        {
         if(!MathRandomNormal(0, 1, (int)inp_total, ar_noise))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         noise.Assign(ar_noise);
         noise = mean_delta + std_delta * noise;
         inp.Assign(Buffer[tr].States[i].state);
         inp = inp + noise;
         State.AssignArray(inp);
         //---
         if(!Autobot.feedForward((CBufferFloat*)GetPointer(State), 1, false, (CBufferFloat*)GetPointer(Ones)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         targets = matrix<float>::Zeros(PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + 1 + t].state);
            if(size > BarDescr)
              {
               matrix<float> temp(1, size);
               temp.Row(target, 0);
               temp.Reshape(size / BarDescr, BarDescr);
               temp.Resize(size / BarDescr, 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, t);
           }
         targets.Reshape(1, targets.Rows()*targets.Cols());
         target = targets.Row(0);
         Autobot.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STE = MathSqrt((MathPow(STE, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         vector<float> check = MathAbs(error) - STE * STE_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(target);
            if(!Autobot.backProp(Result, GetPointer(Ones), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / (2 * (end - state)) + iter + 0.5) * 100.0 / (Iterations);
            string str = StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Autobot", percent, Autobot.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Autobot", Autobot.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

