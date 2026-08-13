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
input int                  Iterations     = 100000;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 RelateEncoder;
CNet                 RelateDecoder;
CNet                 ShortEncoder;
CNet                 ShortDecoder;
CNet                 LongEncoder;
CNet                 LongDecoder;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bShort;
CBufferFloat         bLong;
CBufferFloat         *Result;
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
   float temp = 0;
   if(!RelateEncoder.Load(FileName + "RelE.nnw", temp, temp, temp, dtStudied, true) ||
      !RelateDecoder.Load(FileName + "RelD.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Create new Relate models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateRelationDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!RelateEncoder.Create(encoder) ||
         !RelateDecoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
     }
//---
   if(!ShortEncoder.Load(FileName + "ShE.nnw", temp, temp, temp, dtStudied, true) ||
      !ShortDecoder.Load(FileName + "ShD.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Create new Short predict models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreatePredictionDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!ShortEncoder.Create(encoder) ||
         !ShortDecoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
     }
//---
   if(!LongEncoder.Load(FileName + "LnE.nnw", temp, temp, temp, dtStudied, true) ||
      !LongDecoder.Load(FileName + "LnD.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Create new Long predict models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreatePredictionDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!LongEncoder.Create(encoder) ||
         !LongDecoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
     }
//---
   RelateEncoder.TrainMode(true);
   RelateDecoder.TrainMode(true);
   ShortEncoder.TrainMode(true);
   ShortDecoder.TrainMode(true);
   LongEncoder.TrainMode(true);
   LongDecoder.TrainMode(true);
//---
   OpenCL = RelateEncoder.GetOpenCL();
   RelateDecoder.SetOpenCL(OpenCL);
   ShortEncoder.SetOpenCL(OpenCL);
   ShortDecoder.SetOpenCL(OpenCL);
   LongEncoder.SetOpenCL(OpenCL);
   LongDecoder.SetOpenCL(OpenCL);
//---
   RelateDecoder.getResults(Result);
   if(Result.Total() != MathMin(HistoryBars, 100)*BarDescr)
     {
      PrintFormat("The scope of the Decoder does not match the actions count (%d <> %d)", MathMin(HistoryBars, 100)*BarDescr, Result.Total());
      return INIT_FAILED;
     }
//---
   RelateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
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
      RelateEncoder.Save(FileName + "RelE.nnw", 0, 0, 0, TimeCurrent(), true);
      RelateDecoder.Save(FileName + "RelD.nnw", 0, 0, 0, TimeCurrent(), true);
      ShortEncoder.Save(FileName + "ShE.nnw", 0, 0, 0, TimeCurrent(), true);
      ShortDecoder.Save(FileName + "ShD.nnw", 0, 0, 0, TimeCurrent(), true);
      LongEncoder.Save(FileName + "LnE.nnw", 0, 0, 0, TimeCurrent(), true);
      LongDecoder.Save(FileName + "LnD.nnw", 0, 0, 0, TimeCurrent(), true);
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
   vector<float> result, target, state;
   matrix<float> predict;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - NForecast));
      if(i <= 0)
        {
         iter --;
         continue;
        }
      if(!state.Assign(Buffer[tr].States[i].state) ||
         MathAbs(state).Sum() == 0 ||
         !bState.AssignArray(state))
        {
         iter --;
         continue;
        }
      if(!state.Assign(Buffer[tr].States[i + NForecast].state) ||
         !state.Resize((NForecast + 1)*BarDescr) ||
         // Implements MathAbs.
         MathAbs(state).Sum() == 0)
        {
         iter --;
         continue;
        }
      //--- Feed Forward
      if(!RelateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !RelateDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(RelateEncoder)) ||
         !ShortEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(ShortEncoder)) ||
         !LongEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !LongDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(LongEncoder)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Relation
      if(!RelateDecoder.backProp(GetPointer(bState), (CNet *)GetPointer(RelateEncoder)) ||
         !RelateEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Prediction
      if(!predict.Resize(1, state.Size()) ||
         !predict.Row(state, 0) ||
         !predict.Reshape(NForecast + 1, BarDescr)
        )
        {
         iter --;
         continue;
        }
      result = MathAbs(predict).Max(0);
      target = (predict.Row(NForecast - 1) - predict.Row(NForecast)) / result;
      if(!bShort.AssignArray(target))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      for(int i = 0; i < NForecast - 1; i++)
         target += (predict.Row(i) - predict.Row(i + 1)) / result *
                              MathPow(DiscFactor, NForecast - i - 1);
      if(!bLong.AssignArray(target))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Short prediction
      if(!ShortDecoder.backProp(GetPointer(bShort), (CNet *)GetPointer(ShortEncoder)) ||
         !ShortEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Long prediction
      if(!LongDecoder.backProp(GetPointer(bLong), (CNet *)GetPointer(LongEncoder)) ||
         !LongEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Relate", percent, RelateDecoder.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Short", percent, ShortDecoder.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Long", percent, LongDecoder.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Relate", RelateDecoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Short", ShortDecoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Long", LongDecoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

