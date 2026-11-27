//+------------------------------------------------------------------+
//|                                                         vae2.mq5 |
//|                                                               AE |
//|                                              Copyright 2022, DNG |
//|                           https://www.mql5.com/ru/articles/11206 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//---
#define FileName        Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_"+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0)-1)
#define CSV             __FILE__+".csv"
//---
enum ENUM_SIGNAL
  {
   Sell=-1,
   Undefine=0,
   Buy=1
  };
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int             StudyPeriod=15;             //Study period, years
input uint            HistoryBars=40;             //Depth of history
input ENUM_TIMEFRAMES TimeFrame=PERIOD_CURRENT;
//---
input group "---- RSI ----"
input int                RSIPeriod=14;           //Period
input ENUM_APPLIED_PRICE RSIPrice=PRICE_CLOSE;   //Applied price
//---
input group "---- CCI ----"
input int                CCIPeriod=14;             //Period
input ENUM_APPLIED_PRICE CCIPrice=PRICE_TYPICAL;   //Applied price
//---
input group "---- ATR ----"
input int ATRPeriod=14;   //Period
//---
input group "---- MACD ----"
input int                FastPeriod=12;           //Fast
input int                SlowPeriod=26;           //Slow
input int                SignalPeriod=9;          //Signal
input ENUM_APPLIED_PRICE MACDPrice=PRICE_CLOSE;   //Applied price
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CSymbolInfo  *Symb;
MqlRates     Rates[];
CBufferFloat *TempData;
CiRSI        *RSI;
CiCCI        *CCI;
CiATR        *ATR;
CiMACD       *MACD;
CNet         *Net;
//---
float    dError;
datetime dtStudied;
bool     bEventStudy;
//+------------------------------------------------------------------+
//| OnInit - expert initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   Symb=new CSymbolInfo();
   if(CheckPointer(Symb)==POINTER_INVALID||!Symb.Name(_Symbol))
      return(INIT_FAILED);
   Symb.Refresh();
//---
   RSI=new CiRSI();
   if(CheckPointer(RSI)==POINTER_INVALID||!RSI.Create(Symb.Name(),TimeFrame,RSIPeriod,RSIPrice))
      return(INIT_FAILED);
//---
   CCI=new CiCCI();
   if(CheckPointer(CCI)==POINTER_INVALID||!CCI.Create(Symb.Name(),TimeFrame,CCIPeriod,CCIPrice))
      return(INIT_FAILED);
//---
   ATR=new CiATR();
   if(CheckPointer(ATR)==POINTER_INVALID||!ATR.Create(Symb.Name(),TimeFrame,ATRPeriod))
      return(INIT_FAILED);
//---
   MACD=new CiMACD();
   if(CheckPointer(MACD)==POINTER_INVALID||!MACD.Create(Symb.Name(),TimeFrame,FastPeriod,SlowPeriod,SignalPeriod,MACDPrice))
      return(INIT_FAILED);
//---
   Net=new CNet(NULL);
   ResetLastError();
   float temp1;
   float temp2;
   if(CheckPointer(Net)==POINTER_INVALID||!Net.Load(FileName+".nnw",dError,temp1,temp2,dtStudied,false))
     {
      printf("%s - %d -> Error of read %s prev Net %d",__FUNCTION__,__LINE__,FileName+".nnw",GetLastError());

      return(INIT_FAILED);
     }
//---
   TempData=new CBufferFloat();
   if(CheckPointer(TempData)==POINTER_INVALID)
      return(INIT_FAILED);
//---
   bEventStudy=EventChartCustom(ChartID(),1,(long)MathMax(0,MathMin(iTime(Symb.Name(),PERIOD_CURRENT,(int)(100 * Net.recentAverageSmoothingFactor * 10)),dtStudied)),0,"Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnDeinit - expert deinitialization function                      |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(CheckPointer(Symb)!=POINTER_INVALID)
      delete Symb;
//---
   if(CheckPointer(RSI)!=POINTER_INVALID)
      delete RSI;
//---
   if(CheckPointer(CCI)!=POINTER_INVALID)
      delete CCI;
//---
   if(CheckPointer(ATR)!=POINTER_INVALID)
      delete ATR;
//---
   if(CheckPointer(MACD)!=POINTER_INVALID)
      delete MACD;
//---
   if(CheckPointer(Net)!=POINTER_INVALID)
      delete Net;
//---
   if(CheckPointer(TempData)!=POINTER_INVALID)
      delete TempData;
  }
//+------------------------------------------------------------------+
//| OnTick - expert tick function                                    |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!bEventStudy&&(dtStudied<SeriesInfoInteger(Symb.Name(),TimeFrame,SERIES_LASTBAR_DATE)))
      bEventStudy=EventChartCustom(ChartID(),1,(long)0,0,"New Bar");
//---
  }
//+------------------------------------------------------------------+
//| OnTrade - trade function                                         |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---
  }
//+------------------------------------------------------------------+
//| OnTradeTransaction - tradeTransaction function                   |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
//---
  }
//+------------------------------------------------------------------+
//| OnChartEvent - chartEvent function                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id==1001)
     {
      Train(lparam);
      bEventStudy=false;
      OnTick();
     }
  }
//+------------------------------------------------------------------+
//| Train - Train the neural networks                                |
//+------------------------------------------------------------------+
void Train(datetime StartTrainBar=0)
  {
   int count=0;
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year<=0)
      start_time.year=1900;
   datetime st_time=StructToTime(start_time);
   dtStudied=MathMax(StartTrainBar,st_time);
   ulong last_tick=0;
//---
   double   prev_er=DBL_MAX;
   datetime bar_time=0;
   bool     stop=IsStopped();
   int      handle=FileOpen("VAE_latent.csv",FILE_WRITE | FILE_CSV);
//---
   int bars=CopyRates(Symb.Name(),TimeFrame,st_time,TimeCurrent(),Rates);
   prev_er=dError;
//---
   if(!RSI.BufferResize(bars)||!CCI.BufferResize(bars)||!ATR.BufferResize(bars)||!MACD.BufferResize(bars))
     {
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates,true))
     {
      ExpertRemove();
      return;
     }
   RSI.Refresh(OBJ_ALL_PERIODS);
   CCI.Refresh(OBJ_ALL_PERIODS);
   ATR.Refresh(OBJ_ALL_PERIODS);
   MACD.Refresh(OBJ_ALL_PERIODS);
//---
   int    total=bars-(int)HistoryBars;
   matrix data;
   if(!data.Init(total,12 * HistoryBars))
     {
      ExpertRemove();
      return;
     }
//---
   MqlDateTime sTime;
   for(int i=0; i<total; i++)
     {
      Comment(StringFormat("Create data: %d of %d",i,total));
      for(int b=0; b<(int)HistoryBars; b++)
        {
         int    bar=i+b;
         int    shift=b * 12;
         double open=Rates[bar].open;
         TimeToStruct(Rates[bar].time,sTime);
         data[i,shift+0]=open-Rates[bar].low;
         data[i,shift+1]=Rates[bar].high-open;
         data[i,shift+2]=Rates[bar].close-open;
         data[i,shift+3]=RSI.GetData(MAIN_LINE,bar);
         data[i,shift+4]=CCI.GetData(MAIN_LINE,bar);
         data[i,shift+5]=ATR.GetData(MAIN_LINE,bar);
         data[i,shift+6]=MACD.GetData(MAIN_LINE,bar);
         data[i,shift+7]=MACD.GetData(SIGNAL_LINE,bar);
         data[i,shift+8]=(double)Rates[bar].tick_volume / 1000.0;
         data[i,shift+9]=sTime.hour;
         data[i,shift+10]=sTime.day_of_week;
         data[i,shift+11]=sTime.mon;
        }
     }
//---
     {
      //---
      stop=IsStopped();
      bool add_loop=false;
      for(int it=0; it<1000&&!stop; it++)
        {
         if((GetTickCount64()-last_tick)>=250)
           {
            string com=StringFormat("Calculation -> %d of %d -> %.2f%%",it+1,1000,(double)(it+1.0) / 1000 * 100);
            Comment(com);
            last_tick=GetTickCount64();
           }
         int i=(int)((MathRand() * MathRand() / MathPow(32767,2)) * (total)+HistoryBars / 2);
         TempData.Clear();
         int r=i+(int)HistoryBars;
         if(r>bars)
            continue;
         //---
         for(int b=0; b<(int)HistoryBars; b++)
           {
            int    bar_t=r-b;
            double open=Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time,sTime);
            float rsi=(float)RSI.Main(bar_t);
            float cci=(float)CCI.Main(bar_t);
            float atr=(float)ATR.Main(bar_t);
            float macd=(float)MACD.Main(bar_t);
            float sign=(float)MACD.Signal(bar_t);
            if(rsi==EMPTY_VALUE||cci==EMPTY_VALUE||atr==EMPTY_VALUE||macd==EMPTY_VALUE||sign==EMPTY_VALUE)
               continue;
            //---
            if(!TempData.Add((float)(Rates[bar_t].close-open))||!TempData.Add((float)(Rates[bar_t].high-open))||!TempData.Add((float)(Rates[bar_t].low-open))||!TempData.Add((float)(Rates[bar_t].tick_volume / 1000.0))||
               !TempData.Add(sTime.hour)||!TempData.Add(sTime.day_of_week)||!TempData.Add(sTime.mon)||
               !TempData.Add(rsi)||!TempData.Add(cci)||!TempData.Add(atr)||!TempData.Add(macd)||!TempData.Add(sign))
               break;
           }
         if(TempData.Total()<(int)HistoryBars * 12)
            continue;
         Net.feedForward(TempData,12,true,(CBufferFloat*)NULL);
         TempData.Clear();
         if(!Net.GetLayerOutput(6,TempData))
            break;
         //---
         bool sell=(Rates[i-1].high<=Rates[i].high&&Rates[i+1].high<Rates[i].high);
         bool buy=(Rates[i-1].low>=Rates[i].low&&Rates[i+1].low>Rates[i].low);
         if(buy&&sell)
            buy=sell=false;
         //---
         FileWrite(handle,(buy ? DoubleToString(TempData.At(0)) : " "),(buy ? DoubleToString(TempData.At(1)) : " "),
                   (sell ? DoubleToString(TempData.At(0)) : " "),(sell ? DoubleToString(TempData.At(1)) : " "),
                   (!(buy||sell) ? DoubleToString(TempData.At(0)) : " "),
                   (!(buy||sell) ? DoubleToString(TempData.At(1)) : " "));
         stop=IsStopped();
        }
     }
//---
   FileClose(handle);
   Comment("");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
