//+------------------------------------------------------------------+
//| crossentropy.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\Unsupervised\K-means\kmeans.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  15;            //Study period, years
input uint                 HistoryBars =  20;            //Depth of history
input int                  Clusters    =  500;           //Clusters
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_CURRENT;
input int                  Samples     =  100;
input int                  Percentile  =  70;
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
      Train();
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void Train(void)
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
      // Implements ArrayResize.
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
   Data = Kmeans.SoftMax(Data);
//---
   vector   env = vector::Zeros(Data.Total() / Clusters);
   vector   target = vector::Zeros(env.Size());
   matrix   states = matrix::Zeros(Samples, env.Size());
   matrix   actions = matrix::Zeros(Samples, env.Size());
   vector   CumRewards = vector::Zeros(Samples);
   double   average = 1.0 / Actions;
   matrix   policy = matrix::Full(Clusters, Actions, average);
   for(ulong state = 0; state < env.Size(); state++)
     {
      ulong shift = state * Clusters;
      env[state] = (double)(Data.Maximum((int)shift, Clusters) - shift);
      shift = state * Actions;
      target[state] = (double)(Fractals.Maximum((int)shift, Actions) - shift);
     }
//---
   for(int iter = 0; iter < 200; iter++)
     {
      CumRewards.Fill(0);
      //--- Samples
      for(int sampl = 0; sampl < Samples; sampl++)
        {
         for(ulong state = 0; state < env.Size(); state++)
           {
            ulong a = policy.Row((int)env[state]).ArgMax();
            if(policy[(int)env[state], a] <= average)
               a = (int)(MathRand() / 32768.0 * Actions);
            if(a == target[state])
               CumRewards[sampl] += 1;
            else
               CumRewards[sampl] -= 1;
            actions[sampl, state] = (double)a;
            states[sampl,state]=env[state];
           }
        }
      //--- update policy
      double percentile = CumRewards.Percentile(Percentile);
      policy.Fill(0);
      for(int sampl = 0; sampl < Samples; sampl++)
        {
         if(CumRewards[sampl] < percentile)
            continue;
         for(int state = 0; state < (int)env.Size(); state++)
            policy[(int)states[sampl, state], (int)actions[sampl, state]] += 1;
        }
      for(int row = 0; row < Clusters; row++)
        {
         vector temp = policy.Row(row);
         double sum = temp.Sum();
         if(sum > 0)
            temp = temp / sum;
         else
            temp.Fill(average);
         if(!policy.Row(temp, row))
            break;
        }
      PrintFormat("Iteration %d, Max reward %.0f", iter, CumRewards.Max());
     }
   if(CheckPointer(Data) == POINTER_DYNAMIC)
      delete Data;
   if(CheckPointer(Fractals) == POINTER_DYNAMIC)
      delete Fractals;
   if(CheckPointer(opencl) == POINTER_DYNAMIC)
      delete opencl;
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+

