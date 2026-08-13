//+------------------------------------------------------------------+
//| StudyÈncoder.mq5 |
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
input int                  Iterations     = 10000;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 Decoder;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bGradient;
CBufferFloat         bProbs;
CBufferFloat         *Result;
vector<float>        check;
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
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Decoder.Load(FileName + "Dec.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateEncoderDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) ||
         !Decoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
     }
//---
   OpenCL = Encoder.GetOpenCL();
   Decoder.SetOpenCL(OpenCL);
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(LatentLayer, Result);
   int latent=Result.Total();
   Decoder.getResults(Result);
   if(Result.Total() != latent)
     {
      PrintFormat("Ouput size of Decoder doesn't match latent state Encoder (%d <> %d)", Result.Total(), latent);
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
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      Encoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      Decoder.Save(FileName + "Dec.nnw", 0, 0, 0, TimeCurrent(), true);
     }
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
//| Train function                                                 |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 48;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
      Encoder.Clear();
      int end = MathMin(state + batch, Buffer[tr].Total);
      for(int i = state; i < end; i++)
        {
         bState.AssignArray(Buffer[tr].States[i].state);
         //--- Trajectory
         if(!Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!Decoder.feedForward((CNet*)GetPointer(Encoder),-1,(CBufferFloat *)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         Encoder.GetLayerOutput(LatentLayer,Result);
         if(!Decoder.backProp(Result,(CBufferFloat*)NULL) ||
            !Encoder.backPropGradient((CBufferFloat*)NULL)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Decoder", percent, Decoder.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Decoder", Decoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

