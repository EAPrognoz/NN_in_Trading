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
CNet                 Noise;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat        *Result;
CBufferFloat         LastEncoder;
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
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Decoder.Load(FileName + "Dec.nnw", temp, temp, temp, dtStudied, true) ||
      !Noise.Load(FileName + "NP.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      CArrayObj *noise = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, decoder, noise))
        {
         delete encoder;
         delete decoder;
         delete noise;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) || !Decoder.Create(decoder) ||
         !Noise.Create(noise))
        {
         delete encoder;
         delete decoder;
         delete noise;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
      delete noise;
      //---
     }
//---
   OpenCL = Encoder.GetOpenCL();
   Decoder.SetOpenCL(OpenCL);
   Noise.SetOpenCL(OpenCL);
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
   Noise.GetLayerOutput(0, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("Input size of Noise Prediction model doesn't match Encoder output (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   Noise.getResults(Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Output size of Noise Prediction model doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
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
      Noise.Save(FileName + "NP.nnw", Noise.getRecentAverageError(), 0, 0, TimeCurrent(), true);
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
   vector<float> result, target, inp;
   matrix<float> targets;
   matrix<float> delta;
   STE = vector<float>::Zeros((HistoryBars + PrecoderBars) * 3);
   STE_Noise = vector<float>::Zeros(HistoryBars * BarDescr);
   int std_count = 0;
   int batch = GPTBars + 50;
   bool Stop = false;
   uint ticks = GetTickCount();
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
      Encoder.Clear();
      Decoder.Clear();
      Noise.Clear();
      LastEncoder.BufferInit(EmbeddingSize, 0);
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
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
         if(!LastEncoder.BufferWrite() || !Encoder.feedForward((CBufferFloat*)GetPointer(State), 1, false, (CBufferFloat*)GetPointer(LastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(!Decoder.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(!Noise.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
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
         STE = MathSqrt((MathPow(STE, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         vector<float> check = MathAbs(error) - STE * STE_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Decoder.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         target = vector<float>::Zeros(delta.Cols());
         Noise.getResults(result);
         error = (target - result) * STE_Noise_Multiplier;
         STE_Noise = MathSqrt((MathPow(STE_Noise, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         check = MathAbs(error) - STE_Noise;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Noise.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         Encoder.getResults(result);
         LastEncoder.AssignArray(result);
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / (2 * (end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Decoder", percent, Decoder.getRecentAverageError());
            str += StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Noise Prediction", percent, Noise.getRecentAverageError());
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
      tr = SampleTrajectory(probability);
      state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3 - PrecoderBars - batch));
      if(state < 0)
        {
         iter--;
         continue;
        }
      Encoder.Clear();
      Decoder.Clear();
      Noise.Clear();
      LastEncoder.BufferInit(EmbeddingSize, 0);
      end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
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
         if(!LastEncoder.BufferWrite() || !Encoder.feedForward((CBufferFloat*)GetPointer(State), 1, false, (CBufferFloat*)GetPointer(LastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(!Decoder.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(!Noise.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
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
         STE = MathSqrt((MathPow(STE, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         vector<float> check = MathAbs(error) - STE * STE_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Decoder.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         target = noise;
         Noise.getResults(result);
         error = (target - result) * STE_Noise_Multiplier;
         STE_Noise = MathSqrt((MathPow(STE_Noise, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         check = MathAbs(error) - STE_Noise;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Noise.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         Encoder.getResults(result);
         LastEncoder.AssignArray(result);
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / (2 * (end - state)) + iter + 0.5) * 100.0 / (Iterations);
            string str = StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Decoder", percent, Decoder.getRecentAverageError());
            str += StringFormat("%-20s %6.2f%% -> Error %15.8f\n", "Noise Prediction", percent, Noise.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Decoder", Decoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Noise Prediction", Noise.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

