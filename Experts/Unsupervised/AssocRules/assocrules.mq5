//+------------------------------------------------------------------+
//|                                                   assocrules.mq5 |
//|                                                       AssocRules |
//|                                              Copyright 2022, DNG |
//|                           https://www.mql5.com/ru/articles/11141 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "AssocRules.mqh"
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
input int StudyPeriod=15;   //Study period, years
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
          uint HistoryBars=20;   //Depth of history
ENUM_TIMEFRAMES TimeFrame=PERIOD_CURRENT;
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
CSymbolInfo  *Symb;
MqlRates     Rates[];
CArrayDouble *TempData;
CiRSI        *RSI;
CiCCI        *CCI;
CiATR        *ATR;
CiMACD       *MACD;
CAssocRules  *Rules;
//---
double   dError;
double   dUndefine;
double   dForecast;
double   dPrevSignal;
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
   Rules=new CAssocRules();
   if(!Rules)
      return(INIT_FAILED);
//---
   bEventStudy=EventChartCustom(ChartID(),1,0,0,"Init");
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
   if(CheckPointer(Rules)!=POINTER_INVALID)
      delete Rules;
  }
//+------------------------------------------------------------------+
//| OnTick - expert tick function                                    |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!bEventStudy&&(dPrevSignal==-2||dtStudied<SeriesInfoInteger(Symb.Name(),TimeFrame,SERIES_LASTBAR_DATE)))
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
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year<=0)
      start_time.year=1900;
   datetime st_time=StructToTime(start_time);
//---
   int bars=CopyRates(Symb.Name(),TimeFrame,st_time,TimeCurrent(),Rates);
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
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   int     total=bars-(int)HistoryBars-1;
   matrixf fractals;
   matrixf data;
   if(!data.Init(total,8 * HistoryBars)||
      !fractals.Init(total,3))
     {
      ExpertRemove();
      return;
     }
//---
   for(int i=0; (i<total&&!IsStopped()); i++)
     {
      Comment(StringFormat("Create data: %d of %d",i,total));
      for(int b=0; b<(int)HistoryBars; b++)
        {
         int    bar=i+b;
         int    shift=b * 8;
         double open=Rates[bar].open;
         data[i,shift]=float(open-Rates[bar].low);
         data[i,shift+1]=float(Rates[bar].high-open);
         data[i,shift+2]=float(Rates[bar].close-open);
         data[i,shift+3]=float(RSI.GetData(MAIN_LINE,bar));
         data[i,shift+4]=float(CCI.GetData(MAIN_LINE,bar));
         data[i,shift+5]=float(ATR.GetData(MAIN_LINE,bar));
         data[i,shift+6]=float(MACD.GetData(MAIN_LINE,bar));
         data[i,shift+7]=float(MACD.GetData(SIGNAL_LINE,bar));
        }
      int bar=i+1;
      fractals[i,0]=(float)(Rates[bar-1].high<=Rates[bar].high&&Rates[bar+1].high<Rates[bar].high);
      fractals[i,1]=(float)(Rates[bar-1].low>=Rates[bar].low&&Rates[bar+1].low>Rates[bar].low);
      fractals[i,2]=(float)((fractals[i,0]+fractals[i,1])==0);
     }
   if(IsStopped())
     {
      ExpertRemove();
      return;
     }
//---
   ulong   array[]={300};
   matrixf ar_data[],ar_fractals[];
   data.Hsplit(array,ar_data);
   fractals.Hsplit(array,ar_fractals);
//---
   Rules.CreateRules(ar_data[1],ar_fractals[1].Col(0),ar_fractals[1].Col(1),3,0.05f,0.05f);
   float buy_prob,sell_prob;
   for(ulong i=0; i<ar_data[0].Rows(); i++)
     {
      if(!Rules.Probability(ar_data[0].Row(i),buy_prob,sell_prob))
         return;

      if(DoubleToSignal(buy_prob-sell_prob)==Undefine)
         DeleteObject(Rates[i].time);
      else
         DrawObject(Rates[i].time,(buy_prob-sell_prob),Rates[i].high,Rates[i].low);
     }
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//| DoubleToSignal                                                   |
//+------------------------------------------------------------------+
ENUM_SIGNAL DoubleToSignal(double value)
  {
   value=NormalizeDouble(value,1);
   if(MathAbs(value)>1||MathAbs(value)<=0)
      return(Undefine);

   if(value>0)
      return(Buy);
   else
      return(Sell);
//---
   return(Undefine);
  }
//+------------------------------------------------------------------+
//| DrawObject                                                       |
//+------------------------------------------------------------------+
void DrawObject(datetime time,double signal,double high,double low)
  {
   double            price=0;
   int               arrow=0;
   color             clr=0;
   ENUM_ARROW_ANCHOR anch=ANCHOR_BOTTOM;
   switch(DoubleToSignal(signal))
     {
      case Buy:
         price=low;
         arrow=217;
         clr=clrBlue;
         anch=ANCHOR_TOP;
         break;
      case Sell:
         price=high;
         arrow=218;
         clr=clrRed;
         anch=ANCHOR_BOTTOM;
         break;
     }
   if(price==0||arrow==0)
      return;
//---
   string name=TimeToString(time);
   if(ObjectFind(0,name)<0)
     {
      ResetLastError();
      if(!ObjectCreate(0,name,OBJ_ARROW,0,time,0))
        {
         printf("Error of creating object %d",GetLastError());
         return;
        }
     }
//printf("%s - %d -> %s",__FUNCTION__,__LINE__,name);
   ObjectSetDouble(0,name,OBJPROP_PRICE,price);
   ObjectSetInteger(0,name,OBJPROP_ARROWCODE,arrow);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(0,name,OBJPROP_ANCHOR,anch);
   ObjectSetString(0,name,OBJPROP_TOOLTIP,EnumToString(DoubleToSignal(signal))+" "+DoubleToString(signal,5));
  }
//+------------------------------------------------------------------+
//| DeleteObject                                                     |
//+------------------------------------------------------------------+
void DeleteObject(datetime time)
  {
   string name=TimeToString(time);
   if(ObjectFind(0,name)>=0)
      ObjectDelete(0,name);
  }
//+------------------------------------------------------------------+
