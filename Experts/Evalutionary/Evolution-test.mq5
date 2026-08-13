//+------------------------------------------------------------------+
//| Q-Learning.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define FileName           Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_Evolution"
#define NeuronsToBar       12
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
uint                 HistoryBars =  20;            //Depth of history
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_H1;
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
CNet                StudyNet;
CBufferFloat       *TempData;
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
CBufferFloat         State;
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
   float temp1, temp2;
   if(!StudyNet.Load(FileName + ".nnw", dError, temp1, temp2, dtStudied, true))
      return INIT_FAILED;
//---
   if(!StudyNet.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / NeuronsToBar;
   StudyNet.getResults(TempData);
   if(TempData.Total() != Actions)
      return INIT_PARAMETERS_INCORRECT;
//---
   if(!RSI.BufferResize(HistoryBars + 2) || !CCI.BufferResize(HistoryBars + 2) ||
      !ATR.BufferResize(HistoryBars + 2) || !MACD.BufferResize(HistoryBars + 2))
      return INIT_FAILED;
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
   if(!!TempData)
      delete TempData;
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
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, HistoryBars + 2, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   State.Clear();
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      uint bar_t = (int)HistoryBars - b;
      float open = (float)Rates[bar_t].open;
      TimeToStruct(Rates[bar_t].time, sTime);
      float rsi = (float)RSI.Main(bar_t);
      float rsi_p = (float)RSI.Main(bar_t + 1);
      float cci = (float)CCI.Main(bar_t);
      float cci_p = (float)CCI.Main(bar_t + 1);
      float atr = (float)ATR.Main(bar_t);
      float atr_p = (float)ATR.Main(bar_t + 1);
      float macd = (float)MACD.Main(bar_t);
      float macd_p = (float)MACD.Main(bar_t + 1);
      float sign = (float)MACD.Signal(bar_t);
      float sign_p = (float)MACD.Signal(bar_t + 1);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      if(!State.Add((float)Rates[bar_t].close - open) || !State.Add((float)Rates[bar_t].high - open) ||
         !State.Add((float)Rates[bar_t].low - open) || !State.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
         !State.Add(sTime.hour) || !State.Add(sTime.day_of_week) || !State.Add(sTime.mon) ||
         !State.Add(rsi) || !State.Add(cci) || !State.Add(atr) || !State.Add(macd) || !State.Add(sign))
         break;
     }
//---
   if(State.Total() < (int)(HistoryBars * NeuronsToBar))
      return;
   if(!StudyNet.feedForward(GetPointer(State), NeuronsToBar, true,(CBufferFloat*)NULL))
      return;
   StudyNet.getResults(TempData);
   if(!TempData)
      return;
   lastBar = Rates[0].time;
   int action = GetAction(TempData);
   delete TempData;
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
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets Action.                                                   |
//+------------------------------------------------------------------+
int GetAction(CBufferFloat* probability)
  {
   vectorf prob;
   if(!probability.GetData(prob))
      return -1;
   prob = prob.CumSum();
   prob = prob / prob.Max();
   int err_code;
   float random = (float)MathRandomNormal(0.5, 0.5, err_code);
   if(random >= 1)
      return (int)prob.Size() - 1;
   for(int i = 0; i < (int)prob.Size(); i++)
      if(random <= prob[i] && prob[i] > 0)
         return i;
//---
   return -1;
  }
//+------------------------------------------------------------------+

