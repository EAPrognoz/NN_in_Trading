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
CNet                 Encoder;
CNet                 Decoder;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bStateE;
CBufferFloat         bStateD;
CBufferFloat         *Result;
vector<float>        check;
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
      Print("Create new model");
      CArrayObj *encoder = NULL, *decoder = NULL;
      if(!CreateStateDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) ||
         !Decoder.Create(decoder)
        )
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
     }
//---
   Encoder.SetOpenCL(Decoder.GetOpenCL());
//---
   Encoder.getResults(Result);
   if(Result.Total() != HistoryBars * BarDescr)
     {
      PrintFormat("The scope of the Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("The inputs of the Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Decoder.getResults(Result);
   if(Result.Total() != ForecastBarDescr * NForecast)
     {
      PrintFormat("The scope of the Decoder does not match the forecast state count (%d <> %d)", NForecast * ForecastBarDescr, Result.Total());
      return INIT_FAILED;
     }
//---
   Decoder.GetLayerOutput(0, Result);
   if(Result.Total() != ForecastBarDescr * NForecast)
     {
      PrintFormat("The inputs of the Decoder does not match the forecast state count (%d <> %d)", NForecast * ForecastBarDescr, Result.Total());
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
   matrix<float> mstate = matrix<float>::Zeros(1, NForecast * ForecastBarDescr);
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
         iter--;
         continue;
        }
      state.Assign(Buffer[tr].States[i].state);
      if(MathAbs(state).Sum() == 0)
        {
         iter--;
         continue;
        }
      bStateE.AssignArray(state);
      mstate.Assign(state);
      mstate.Reshape(HistoryBars, BarDescr);
      mstate.Resize(NForecast, ForecastBarDescr);
      bStateD.AssignArray(mstate);
      //--- State Encoder
      if(!Encoder.feedForward((CBufferFloat*)GetPointer(bStateE), 1, false, (CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!Decoder.feedForward((CBufferFloat*)GetPointer(bStateD), 1, false, GetPointer(Encoder)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Collect target data
      mstate.Assign(Buffer[tr].States[i + NForecast].state);
      mstate.Reshape(HistoryBars, BarDescr);
      mstate.Resize(NForecast, ForecastBarDescr);
      if(!Result.AssignArray(mstate))
         continue;
      if(!Decoder.backProp(Result, GetPointer(Encoder)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!Encoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Decoder", percent, Decoder.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Decoder", Decoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

