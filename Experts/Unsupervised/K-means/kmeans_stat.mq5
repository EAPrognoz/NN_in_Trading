//+------------------------------------------------------------------+
//| kmeans_stat.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "kmeans.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  15;            //Study period, years
input uint                 HistoryBars =  20;            //Depth of history
input int                  Clusters    =  500;           //Clusters
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_CURRENT;
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
CSymbolInfo         *Symb;
MqlRates            Rates[];
CKmeans             *Kmeans;
CArrayDouble        *TempData;
CiRSI               *RSI;
CiCCI               *CCI;
CiATR               *ATR;
CiMACD              *MACD;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   Symb = new CSymbolInfo();
   if(CheckPointer(Symb) == POINTER_INVALID || !Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
//---
   RSI = new CiRSI();
   if(CheckPointer(RSI) == POINTER_INVALID || !RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   CCI = new CiCCI();
   if(CheckPointer(CCI) == POINTER_INVALID || !CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   ATR = new CiATR();
   if(CheckPointer(ATR) == POINTER_INVALID || !ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   MACD = new CiMACD();
   if(CheckPointer(MACD) == POINTER_INVALID || !MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
//---
   Kmeans = new CKmeans();
   if(CheckPointer(Kmeans) == POINTER_INVALID)
      return INIT_FAILED;
//---
   bool    bEventStudy = EventChartCustom(ChartID(), 1, 0, 0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(CheckPointer(Symb) != POINTER_INVALID)
      delete Symb;
//---
   if(CheckPointer(RSI) != POINTER_INVALID)
      delete RSI;
//---
   if(CheckPointer(CCI) != POINTER_INVALID)
      delete CCI;
//---
   if(CheckPointer(ATR) != POINTER_INVALID)
      delete ATR;
//---
   if(CheckPointer(MACD) != POINTER_INVALID)
      delete MACD;
//---
   if(CheckPointer(Kmeans) != POINTER_INVALID)
      delete Kmeans;
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
      Stat();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S ta t implementation.                                         |
//+------------------------------------------------------------------+
void Stat(void)
  {
   COpenCLMy *opencl = OpenCLCreate(cl_unsupervised);
   if(CheckPointer(opencl) == POINTER_INVALID)
     {
      ExpertRemove();
      return;
     }
   if(!Kmeans.SetOpenCL(opencl))
     {
      delete opencl;
      ExpertRemove();
      return;
     }
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
   int handl = FileOpen(StringFormat("kmeans_%d.net", Clusters), FILE_READ | FILE_BIN);
   if(handl == INVALID_HANDLE)
     {
      ExpertRemove();
      return;
     }
   if(FileReadInteger(handl) != Kmeans.Type())
     {
      ExpertRemove();
      return;
     }
   bool result = Kmeans.Load(handl);
   FileClose(handl);
   if(!result)
     {
      ExpertRemove();
      return;
     }
//---
   int total = bars - (int)HistoryBars - 480;
   double data[], fractals[];
   if(ArrayResize(data, total * 8 * HistoryBars) <= 0 ||
//| ArrayResizeimplementation.                                   |
      ArrayResize(fractals, total * 3) <= 0)
     {
      ExpertRemove();
      return;
     }
//---
   for(int i = 0; (i < total && !IsStopped()); i++)
     {
      Comment(StringFormat("Create data: %d of %d", i, total));
      for(int b = 0; b < (int)HistoryBars; b++)
        {
         int bar = i + b + 480;
         int shift = (i * (int)HistoryBars + b) * 8;
         double open = Rates[bar]
                       .open;
         data[shift] = open - Rates[bar].low;
         data[shift + 1] = Rates[bar].high - open;
         data[shift + 2] = Rates[bar].close - open;
         data[shift + 3] = RSI.GetData(MAIN_LINE, bar);
         data[shift + 4] = CCI.GetData(MAIN_LINE, bar);
         data[shift + 5] = ATR.GetData(MAIN_LINE, bar);
         data[shift + 6] = MACD.GetData(MAIN_LINE, bar);
         data[shift + 7] = MACD.GetData(SIGNAL_LINE, bar);
        }
      int shift = i * 3;
      int bar = i + 480;
      fractals[shift] = (int)(Rates[bar - 1].high <= Rates[bar].high && Rates[bar + 1].high < Rates[bar].high);
      fractals[shift + 1] = (int)(Rates[bar - 1].low >= Rates[bar].low && Rates[bar + 1].low > Rates[bar].low);
      fractals[shift + 2] = (int)((fractals[shift] + fractals[shift]) == 0);
     }
   if(IsStopped())
     {
      ExpertRemove();
      return;
     }
   CBufferFloat *Data = new CBufferFloat();
   if(CheckPointer(Data) == POINTER_INVALID ||
      !Data.AssignArray(data))
      return;
   CBufferFloat *Fractals = new CBufferFloat();
   if(CheckPointer(Fractals) == POINTER_INVALID ||
      !Fractals.AssignArray(fractals))
      return;
//---
   ResetLastError();
   if(!Kmeans.Statistic(Data, Fractals))
     {
      printf("Ошибка выполнения %d", GetLastError());
      ExpertRemove();
     }
//---
   CBufferFloat *prob = Kmeans.GetProbability(Data);
   if(CheckPointer(prob) != POINTER_INVALID)
     {
      double buy_min = 1;
      double buy_max = 0;
      double sell_min = 1;
      double sell_max = 0;
      double skip_min = 1;
      double skip_max = 0;
      total = prob.Total() / 3;
      for(int i = 0; i < total; i++)
        {
         int shift = i * 3;
         double buy = prob.At(shift);
         double sell = prob.At(shift + 1);
         double skip = prob.At(shift + 2);
         if(buy == sell && sell == skip && skip == 0)
            continue;
         //---
         buy_min = fmin(buy_min, buy);
         buy_max = fmax(buy_max, buy);
         sell_min = fmin(sell_min, sell);
         sell_max = fmax(sell_max, sell);
         skip_min = fmin(skip_min, skip);
         skip_max = fmax(skip_max, skip);
        }
      //---
      PrintFormat("Statictic for %d clusters", Clusters);
      PrintFormat("Buy probability min %.2f%% max %.2f%%", buy_min * 100, buy_max * 100);
      PrintFormat("Sell probability min %.2f%% max %.2f%%", sell_min * 100, sell_max * 100);
      PrintFormat("Skip probability min %.2f%% max %.2f%%", skip_min * 100, skip_max * 100);
     }
   if(CheckPointer(Data) == POINTER_DYNAMIC)
      delete Data;
   if(CheckPointer(Fractals) == POINTER_DYNAMIC)
      delete Fractals;
   if(CheckPointer(prob) == POINTER_DYNAMIC)
      delete prob;
   if(CheckPointer(opencl) == POINTER_DYNAMIC)
      delete opencl;
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+

