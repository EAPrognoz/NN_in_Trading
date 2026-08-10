//+------------------------------------------------------------------+
//|                                                 StudyStateEncoder.mq5 |
//|                                                   Copyright DNGР вЂ™Р’В® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNGР вЂ™Р’В®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define Study
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input datetime             Start          = D'2024.01.01';
input datetime             End            = D'2026.01.01';
input int                  Epochs         = 100;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNet                 StateEncoder;
CNet                 ForecastDecoder;
//---

datetime             dtStudied;
//---
CBufferFloat         bStateE;
CBufferFloat         bTarget;
CBufferFloat         bTime;
CBufferFloat         *Result;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
//--- load models
   float temp;
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !ForecastDecoder.Load(FileName + "FDec.nnw", temp, temp, temp, dtStudied, true)
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
      if(!StateEncoder.Create(encoder) ||
         !ForecastDecoder.Create(decoder)
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
   StateEncoder.SetOpenCL(ForecastDecoder.GetOpenCL());
//---
   StateEncoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the StateEncoder doesn't match scenario embedding (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   StateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("The inputs of the StateEncoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   StateEncoder.GetLayerOutput(StateScenarioLayer, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scenario layer does not match scenario embedding (%d <> %d)", EmbeddingSize, Result.Total());
      return INIT_FAILED;
     }
//---
   ForecastDecoder.getResults(Result);
   if(Result.Total() != BarDescr * NForecast)
     {
      PrintFormat("The scope of the ForecastDecoder does not match the forecast state count (%d <> %d)", NForecast * BarDescr, Result.Total());
      return INIT_FAILED;
     }
//---
   if(!Symb.Name(_Symbol) || !Symb.Refresh())
      return INIT_FAILED;
//---
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
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
      StateEncoder.Save(FileName + "StEnc.nnw", 0, 0, 0, TimeCurrent(), true);
      ForecastDecoder.Save(FileName + "FDec.nnw", 0, 0, 0, TimeCurrent(), true);
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
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars < 0)
     {
      PrintFormat("%s -> %d CopyRates failed (%d)", __FUNCTION__, __LINE__, GetLastError());
      ExpertRemove();
      return;
     }
//---
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   int count = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars &&
                    CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars &&
                    MACD.BarsCalculated() >= bars
                   );
      Sleep(100);
      count++;
     }
   while(!calculated && count < 100);
   if(!calculated)
     {
      PrintFormat("%s -> %d The training data has not been loaded", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int epoch = 0; (epoch < Epochs && !IsStopped() && !Stop); epoch ++)
     {
      if(!StateEncoder.Clear() ||
         !ForecastDecoder.Clear())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      for(int posit = start - HistoryBars - NForecast - 1; posit >= end; posit--)
        {
         if(!CreateBuffers(posit, GetPointer(bStateE), GetPointer(bTime), GetPointer(bTarget)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- State Encoder
         if(!StateEncoder.feedForward((CBufferFloat*)GetPointer(bStateE), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!ForecastDecoder.feedForward((CNet*)GetPointer(StateEncoder), StateScenarioLayer, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!ForecastDecoder.backProp(GetPointer(bTarget), (CBufferFloat*)NULL, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!StateEncoder.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, StateScenarioLayer, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(epoch) + 1.0 - double(posit - end) / (start - end - HistoryBars - NForecast)) * 100.0 / Epochs;
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "ForecastDecoder", percent, ForecastDecoder.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "ForecastDecoder", ForecastDecoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
  
