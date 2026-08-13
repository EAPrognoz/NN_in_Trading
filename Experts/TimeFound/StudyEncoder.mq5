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
#define StudyOnline
#include "Trajectory.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input datetime             Start          = D'2020.01.01';
input datetime             End            = D'2025.01.01';
input int                  Iterations     = 100000;
input int                  Repeats        = 5;
input group                "---- Indicators ----"
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_M1;
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet                 cEncoder;
//---
CSymbolInfo          Symb;
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         bGradient;
CBufferFloat         *Result;
vector<float>        check;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
//--- load models
   float temp;
   CArrayObj *encoder = new CArrayObj();
   CArrayObj *actor = new CArrayObj();
   CArrayObj *director = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(encoder, actor, director, critic))
     {
      delete encoder;
      delete actor;
      delete director;
      delete critic;
      return INIT_FAILED;
     }
   if(!cEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new State Encoder");
      if(!cEncoder.Create(encoder))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   delete encoder;
   delete actor;
   delete director;
   delete critic;
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
   cEncoder.TrainMode(true);
//---
   cEncoder.GetLayerOutput(0, Result);
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
      if(!cEncoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Encoder");
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
   switch(id)
     {
      case 1001:
         Train();
         break;
      case 1007:
         PrintFormat("Wrong event: %d", id);
         ExpertRemove();
         break;
     }
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
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
   bool load = false;
   do
     {
      RSI.Refresh();
      CCI.Refresh();
      ATR.Refresh();
      MACD.Refresh();
      count++;
      load = (RSI.BarsCalculated() >= bars &&
              CCI.BarsCalculated() >= bars &&
              ATR.BarsCalculated() >= bars &&
              MACD.BarsCalculated() >= bars
             );
      Sleep(100);
      count++;
     }
   while(!load && count < 100);
   if(!load)
     {
      PrintFormat("%s -> %d The training data has not been loaded", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
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
   vector<float> result, target, neg_target;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int posit = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * bars);
      if(!CreateBuffers(posit + end, GetPointer(bState), GetPointer(bTime), Result))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      for(int r = 0; r < Repeats; r++)
        {
         //--- Feed Forward
         if(!cEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Study
         if(!cEncoder.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = double(iter) * 100.0 / (Iterations);
            string str = StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Encoder", percent, cEncoder.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Encoder", cEncoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Buffers.                                |
//+------------------------------------------------------------------+
bool CreateBuffers(const int start_bar, CBufferFloat* state, CBufferFloat *time, CBufferFloat* forecast)
  {
   if(!state || !time || start_bar < 0 ||
      (start_bar + HistoryBars + NForecast) >= int(Rates.Size()))
      return false;
//---
   vector<float> vState = vector<float>::Zeros(HistoryBars * BarDescr);
   vector<float> vForecast = vector<float>::Zeros(NForecast * BarDescr);
   time.Clear();
   time.Reserve(HistoryBars);
   int bar = start_bar + NForecast;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b + bar].open;
      float rsi = (float)RSI.Main(b + bar);
      float cci = (float)CCI.Main(b + bar);
      float atr = (float)ATR.Main(b + bar);
      float macd = (float)MACD.Main(b + bar);
      float sign = (float)MACD.Signal(b + bar);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         return false;
      //---
      int shift = b * BarDescr;
      vState[shift] = (float)(Rates[b + bar].close - open);
      vState[shift + 1] = (float)(Rates[b + bar].high - open);
      vState[shift + 2] = (float)(Rates[b + bar].low - open);
      vState[shift + 3] = (float)(Rates[b + bar].tick_volume / 1000.0f);
      vState[shift + 4] = rsi;
      vState[shift + 5] = cci;
      vState[shift + 6] = atr;
      vState[shift + 7] = macd;
      vState[shift + 8] = sign;
      if(!time.Add(float(Rates[b + bar].time)))
         return false;
     }
//---
   bar--;
   for(int b = 0; b < (int)NForecast; b++)
     {
      float open = (float)Rates[bar - b].open;
      float rsi = (float)RSI.Main(bar - b);
      float cci = (float)CCI.Main(bar - b);
      float atr = (float)ATR.Main(bar - b);
      float macd = (float)MACD.Main(bar - b);
      float sign = (float)MACD.Signal(bar - b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         return false;
      //---
      int shift = (NForecast - b - 1) * BarDescr;
      vForecast[shift] = (float)(Rates[bar - b].close - open);
      vForecast[shift + 1] = (float)(Rates[bar - b].high - open);
      vForecast[shift + 2] = (float)(Rates[bar - b].low - open);
      vForecast[shift + 3] = (float)(Rates[bar - b].tick_volume / 1000.0f);
      vForecast[shift + 4] = rsi;
      vForecast[shift + 5] = cci;
      vForecast[shift + 6] = atr;
      vForecast[shift + 7] = macd;
      vForecast[shift + 8] = sign;
     }
//---
   if(!state.AssignArray(vState))
      return false;
   if(!forecast.AssignArray(vForecast))
      return false;
   if(time.GetIndex() >= 0)
      if(!time.BufferWrite())
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+

