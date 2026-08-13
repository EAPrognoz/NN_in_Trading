//+------------------------------------------------------------------+
//| Genetic.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "NetEvolution.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define MODEL           Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_Evolution"
#define NeuronsToBar    12
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  2;             //Study period, years
      uint                 HistoryBars =  20;            //Depth of history
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_H1;
input int                  PopulationSize =  50;
input int                  Generations =  1000;
input float                Mutation    =  0.01f;
int                        Actions     =  3;
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
CSymbolInfo          Symb;
MqlRates             Rates[];
CNetEvolution        Models;
CBufferFloat        *TempData;
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
MqlDateTime          sTime;
float                BestLoss;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
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
   if(!Models.Load(MODEL + ".nnw", PopulationSize, false))
      return INIT_FAILED;
//---
   if(!Models.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / NeuronsToBar;
   Models.getResults(TempData);
   if(TempData.Total() != Actions)
      return INIT_PARAMETERS_INCORRECT;
//---
   bEventStudy = EventChartCustom(ChartID(), 1, 0, 0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(!!TempData)
      delete TempData;
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
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
//| |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year <= 0)
      start_time.year = 1900;
   datetime st_time = StructToTime(start_time);
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, st_time, TimeCurrent(), Rates);
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) || !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
     {
      ExpertRemove();
      return;
     }
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   CBufferFloat* State = new CBufferFloat();
   float loss = 0;
   uint count = 0;
   uint total = bars - HistoryBars - 2;
   ulong ticks = GetTickCount64();
   uint test_size = 22 * 24;
   for(int gen = 0; (gen < Generations && !IsStopped()); gen ++)
     {
      for(uint i = total; i > test_size; i--)
        {
         uint r = i + HistoryBars;
         if(r > (uint)bars)
           {
            continue;
           }
         State.Clear();
         for(uint b = 0; b < HistoryBars; b++)
           {
            uint bar_t = r - b;
            float open = (float)Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time, sTime);
            float rsi = (float)RSI.Main(bar_t);
            float cci = (float)CCI.Main(bar_t);
            float atr = (float)ATR.Main(bar_t);
            float macd = (float)MACD.Main(bar_t);
            float sign = (float)MACD.Signal(bar_t);
            if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
               continue;
            //---
            if(!State.Add((float)Rates[bar_t].close - open) || !State.Add((float)Rates[bar_t].high - open) ||
               !State.Add((float)Rates[bar_t].low - open) || !State.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !State.Add(sTime.hour) || !State.Add(sTime.day_of_week) || !State.Add(sTime.mon) ||
               !State.Add(rsi) || !State.Add(cci) || !State.Add(atr) || !State.Add(macd) || !State.Add(sign))
              {
               break;
              }
           }
         if(IsStopped())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         if(State.Total() < (int)HistoryBars * NeuronsToBar)
           {
            continue;
           }
         if(!Models.feedForward(State, NeuronsToBar, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
         double reward = Rates[i - 1].close - Rates[i - 1].open;
         TempData.Clear();
         if(!TempData.Add((float)(reward < 0 ? 20 * reward : reward)) ||
            !TempData.Add((float)(reward > 0 ? -reward * 20 : -reward)) ||
            !TempData.Add((float) - fabs(reward * 10)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         if(!Models.Rewards(TempData))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         if(GetTickCount64() - ticks > 250)
           {
            uint x = total - i;
            double perc = x * 100.0 / (total - test_size);
            Comment(StringFormat("%d from %d -> %.2f%% from %.2f%%", x, total - test_size, perc, 100));
            ticks = GetTickCount64();
           }
        }
      //---
      if(!IsStopped())
         Models.Save(MODEL + ".nnw", false);
      //---
      float average, maximum;
      if(!Models.NextGeneration(Mutation, average, maximum))
        {
         PrintFormat("Error of create next generation: %d", GetLastError());
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      PrintFormat("Generation %d, Average Cummulative reward %.5f, Max Reward %.5f", gen, average, maximum);
     }
   delete State;
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+

