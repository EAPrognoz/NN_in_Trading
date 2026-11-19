//+------------------------------------------------------------------+
//|                                                          vae.mq5 |
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
#define FileName        Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_"+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0))
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
   float temp1,temp2;
   if(!Net||!Net.Load(FileName+".nnw",dError,temp1,temp2,dtStudied,false))
     {
      printf("%s - %d -> Error of read %s prev Net %d",__FUNCTION__,__LINE__,FileName+".nnw",GetLastError());
      CArrayObj *Topology=new CArrayObj();
      if(CheckPointer(Topology)==POINTER_INVALID)
         return(INIT_FAILED);

   //--- 0
      CLayerDescription *desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      uint prev=desc.count=(int)HistoryBars * 12;
      desc.type=defNeuronBaseOCL;
      desc.optimization=ADAM;
      desc.activation=None;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 1
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=prev;
      desc.batch=1000;
      desc.type=defNeuronBatchNormOCL;
      desc.activation=None;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 2
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      prev=desc.count=(int)HistoryBars;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 3
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      prev=desc.count=prev / 2;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 4
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      prev=desc.count=prev / 2;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 5
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=4;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 6
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=2;
      desc.type=defNeuronVAEOCL;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 7
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=(int) HistoryBars;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 8
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=(int) HistoryBars * 2;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 9
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=(int) HistoryBars * 4;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 10
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=(int) HistoryBars * 12;
      desc.type=defNeuronBaseOCL;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

      delete Net;
      Net=new CNet(Topology);
      delete Topology;
      if(CheckPointer(Net)==POINTER_INVALID)
         return(INIT_FAILED);
      dError=FLT_MAX;
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
//| Train                                                            |
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
   double       prev_er=DBL_MAX;
   datetime     bar_time=0;
   bool         stop=IsStopped();
   CArrayDouble *loss=new CArrayDouble();
   MqlDateTime  sTime;
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
   int total=(int)(bars-MathMax(HistoryBars,0));
   do
     {
      //---
      stop=IsStopped();
      prev_er=dError;
      for(int it=total-1; it>=0&&!stop; it--)
        {
         int i=(int)((MathRand() * MathRand() / MathPow(32767,2)) * (total));
         if((GetTickCount64()-last_tick)>=250)
           {
            string com=StringFormat("Study -> Era %d -> %.6f\n %d of %d -> %.2f%% \nError %.5f",count,prev_er,bars-it+1,bars,(double)(bars-it+1.0) / bars * 100,Net.getRecentAverageError());
            Comment(com);
            last_tick=GetTickCount64();
           }
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
         if(!Net.GetLayerOutput(1,TempData))
            break;
         Net.backProp(TempData,NULL,NULL);
         stop=IsStopped();
        }
      if(!stop)
        {
         dError=Net.getRecentAverageError();
         Net.Save(FileName+".nnw",dError,0,0,dtStudied,false);
         printf("Era %d -> error %.5f %%",count,dError);
         loss.Add(dError);
         count++;
        }
     }
   while(!(dError<0.01&&(prev_er-dError)<0.01)&&!stop);
//---
   Comment("Write dinamic of error");
   int handle=FileOpen("ae_loss.csv",FILE_WRITE | FILE_CSV | FILE_ANSI,",",CP_UTF8);
   if(handle==INVALID_HANDLE)
     {
      PrintFormat("Error of open loss file: %d",GetLastError());
      delete loss;
      return;
     }
   for(int i=0; i<loss.Total(); i++)
      if(FileWrite(handle,loss.At(i))<=0)
         break;
   FileClose(handle);
   PrintFormat("The dynamics of the error change is saved to a file %s\\%s",
               TerminalInfoString(TERMINAL_DATA_PATH),"ae_loss.csv");
   delete loss;
   Comment("");
   ExpertRemove();
  }
//+------------------------------------------------------------------+
