//+------------------------------------------------------------------+
//| QR-DQN-Learning.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "FQF.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define FileName        Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_"+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0))
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
uint                 HistoryBars =  20;            //Depth of history
input ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_H1;
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
CSymbolInfo         Symb;
MqlRates            Rates[];
CFQF                StudyNet;
CiRSI               RSI;
CiCCI               CCI;
CiATR               ATR;
CiMACD              MACD;
CTrade              Trade;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
MqlDateTime          sTime;
//---
CBufferFloat         State1;
datetime             lastBar;
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
   if(!RSI.BufferResize(HistoryBars + 1) || !CCI.BufferResize(HistoryBars + 1) ||
      !ATR.BufferResize(HistoryBars + 1) || !MACD.BufferResize(HistoryBars + 1))
      return INIT_FAILED;
//---
   if(!StudyNet.Load(FileName + ".nnw", dtStudied, true))
      return INIT_FAILED;
//---
   CBufferFloat *TempData;
   if(!StudyNet.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / 12;
   StudyNet.getResults(TempData);
   if(!TempData)
      return INIT_FAILED;
   Actions = TempData.Total();
//---
   lastBar = 0;
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(lastBar >= iTime(Symb.Name(), TimeFrame, 0))
      return;
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, HistoryBars + 1, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   State1.Clear();
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      int bar_t = (int)HistoryBars - b;
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
         break;
     }
//---
   if(State1.Total() < (int)(HistoryBars * 12))
      return;
   if(!StudyNet.feedForward(GetPointer(State1), 12, true))
      return;
   int action = StudyNet.getSample();
//---
   bool Buy = false;
   bool Sell = false;
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      switch((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            Buy = true;
            break;
         case POSITION_TYPE_SELL:
            Sell = true;
            break;
        }
     }
   switch(action)
     {
      case 0:
         if(!Buy)
           {
            if((Sell && !Trade.PositionClose(Symb.Name())) ||
               !Trade.Buy(Symb.LotsMin(), Symb.Name()))
              {
               lastBar = 0;
               return;
              }
           }
         break;
      case 1:
         if(!Sell)
           {
            if((Buy && !Trade.PositionClose(Symb.Name())) ||
               !Trade.Sell(Symb.LotsMin(), Symb.Name()))
              {
               lastBar = 0;
               return;
              }
           }
         break;
      case 2:
         if(Buy || Sell)
            if(!Trade.PositionClose(Symb.Name()))
              {
               lastBar = 0;
               return;
              }
         break;
     }
//---
   lastBar = iTime(Symb.Name(), TimeFrame, 0);
  }
//+------------------------------------------------------------------+

