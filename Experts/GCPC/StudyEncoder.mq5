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
CNet                 Encoder;
CNet                 Decoder;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat        *Result;
CBufferFloat         LastEncoder;
CBufferFloat         Gradient;
vector<float>        STD;
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
      !Decoder.Load(FileName + "Dec.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) || !Decoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
      //---
     }
//---
   OpenCL = Encoder.GetOpenCL();
   Decoder.SetOpenCL(OpenCL);
//---
   Encoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Encoder does not match the embedding size count (%d <> %d)", EmbeddingSize, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Decoder.GetLayerOutput(0, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("Input size of Decoder doesn't match Encoder output (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   if(!LastEncoder.BufferInit(EmbeddingSize, 0) ||
      !Gradient.BufferInit(EmbeddingSize, 0) ||
      !LastEncoder.BufferCreate(OpenCL) ||
      !Gradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
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
      Decoder.Save(FileName + "Dec.nnw", Decoder.getRecentAverageError(), 0, 0, TimeCurrent(), true);
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
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target;
   matrix<float> targets;
   STD = vector<float>::Zeros((HistoryBars + PrecoderBars) * 3);
   int std_count = 0;
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 50;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
      Encoder.Clear();
      Decoder.Clear();
      LastEncoder.BufferInit(EmbeddingSize, 0);
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      for(int i = state; i < end; i++)
        {
         State.AssignArray(Buffer[tr].States[i].state);
         //---
         if(!LastEncoder.BufferWrite() || !Encoder.feedForward((CBufferFloat*)GetPointer(State), 1, false, (CBufferFloat*)GetPointer(LastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //---
         if(!Decoder.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //---
         target.Assign(Buffer[tr].States[i].state);
         ulong size = target.Size();
         targets = matrix<float>::Zeros(1, size);
         targets.Row(target, 0);
         if(size > BarDescr)
            targets.Reshape(size / BarDescr, BarDescr);
         ulong shift = targets.Rows();
         targets.Resize(shift + PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + t].state);
            if(size > BarDescr)
              {
               matrix<float> temp(1, size);
               temp.Row(target, 0);
               temp.Reshape(size / BarDescr, BarDescr);
               temp.Resize(size / BarDescr, 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, shift + t);
           }
         targets.Reshape(1, targets.Rows()*targets.Cols());
         target = targets.Row(0);
         Decoder.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STD = MathSqrt((MathPow(STD, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         vector<float> check = MathAbs(error) - STD * STD_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Decoder.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               break;
              }
           }
         //---
         Encoder.getResults(result);
         LastEncoder.AssignArray(result);
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

