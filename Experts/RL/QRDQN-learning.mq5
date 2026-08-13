//+------------------------------------------------------------------+
//| QR-DQN-Learning.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define FileName        Symb.Name()+"_"+EnumToString(TimeFrame)+"_"+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0))
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "QRDQN.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  2;             //Study period, years
uint                 HistoryBars =  20;            //Depth of history
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input int                  Batch =  100;
input int                  UpdateTarget = 1000;
input int                  Iterations = 1000;
input float                DiscountFactor =   0.3f;
int                  Actions     =  3;
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
CQRDQN               StudyNet;
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
//---
CBufferFloat         State1;
CBufferFloat         State2;
CBufferFloat         Rewards;
float                min_loss = FLT_MAX;
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
   if(!StudyNet.Load(FileName + ".nnw", dtStudied, false))
      return INIT_FAILED;
   if(!StudyNet.TrainMode(true))
      return INIT_FAILED;
//---
   if(!StudyNet.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / 12;
   if(!StudyNet.SetActions(Actions))
      return INIT_PARAMETERS_INCORRECT;
   StudyNet.SetUpdateTarget(1000000);
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
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   int total = bars - (int)HistoryBars - 240;
   bool use_target = false;
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int i = 0;
      uint ticks = GetTickCount();
      int count = 0;
      int total_max = 0;
      for(int batch = 0; batch < (Batch * UpdateTarget); batch++)
        {
         i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * total + 240);
         State1.Clear();
         State2.Clear();
         int r = i + (int)HistoryBars;
         if(r > bars)
            continue;
         for(int b = 0; b < (int)HistoryBars; b++)
           {
            int bar_t = r - b;
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
            if(!State1.Add((float)Rates[bar_t].close - open) || !State1.Add((float)Rates[bar_t].high - open) || !State1.Add((float)Rates[bar_t].low - open) || !State1.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !State1.Add(sTime.hour) || !State1.Add(sTime.day_of_week) || !State1.Add(sTime.mon) ||
               !State1.Add(rsi) || !State1.Add(cci) || !State1.Add(atr) || !State1.Add(macd) || !State1.Add(sign))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               break;
              }
            if(!use_target)
               continue;
            //---
            bar_t--;
            open = (float)Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time, sTime);
            rsi = (float)RSI.Main(bar_t);
            cci = (float)CCI.Main(bar_t);
            atr = (float)ATR.Main(bar_t);
            macd = (float)MACD.Main(bar_t);
            sign = (float)MACD.Signal(bar_t);
            if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
               continue;
            //---
            if(!State2.Add((float)Rates[bar_t].close - open) || !State2.Add((float)Rates[bar_t].high - open) || !State2.Add((float)Rates[bar_t].low - open) || !State2.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !State2.Add(sTime.hour) || !State2.Add(sTime.day_of_week) || !State2.Add(sTime.mon) ||
               !State2.Add(rsi) || !State2.Add(cci) || !State2.Add(atr) || !State2.Add(macd) || !State2.Add(sign))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               break;
              }
           }
         if(IsStopped())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         if(State1.Total() < (int)HistoryBars * 12 ||
            (use_target && State2.Total() < (int)HistoryBars * 12))
            continue;
         if(!StudyNet.feedForward(GetPointer(State1), 12, true))
            return;
         Rewards.BufferInit(Actions, 0);
         double reward = Rates[i].close - Rates[i].open;
         if(reward >= 0)
           {
            if(!Rewards.Update(0, (float)(2 * reward)))
               return;
            if(!Rewards.Update(1, (float)(-5 * reward)))
               return;
            if(!Rewards.Update(2, (float) - reward))
               return;
           }
         else
           {
            if(!Rewards.Update(0, (float)(5 * reward)))
               return;
            if(!Rewards.Update(1, (float)(-2 * reward)))
               return;
            if(!Rewards.Update(2, (float)reward))
               return;
           }
         //---
         if(!StudyNet.backProp(GetPointer(Rewards), DiscountFactor, (use_target ? GetPointer(State2) : NULL), 12, true))
            return;
         if(GetTickCount() - ticks > 500)
           {
            Comment(StringFormat("%.2f%%", batch * 100.0 / (double)(Batch * UpdateTarget)));
            ticks = GetTickCount();
           }
        }
      if(StudyNet.getRecentAverageError() <= min_loss)
        {
         if(!StudyNet.UpdateTarget(FileName + ".nnw"))
            continue;
         use_target = true;
         min_loss = StudyNet.getRecentAverageError();
        }
      PrintFormat("Iteration %d, loss %.8f", iter, StudyNet.getRecentAverageError());
     }
   Comment("");
//---
   PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
   ExpertRemove();
  }
//+------------------------------------------------------------------+

