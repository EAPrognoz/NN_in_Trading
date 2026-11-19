//+------------------------------------------------------------------+
//|                                                       ae_net.mq5 |
//|                                                               AE |
//|                                              Copyright 2022, DNG |
//|                           https://www.mql5.com/ru/articles/11172 |
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
#define FileName        Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_"+IntegerToString(HistoryBars,3)+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0))
#define FileName_AE     Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_ae"
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
input int StudyPeriod=2;   //Study period, years
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
          uint HistoryBars=40;   //Depth of history
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
CNet         *Net;
CBufferFloat *TempData;
CiRSI        *RSI;
CiCCI        *CCI;
CiATR        *ATR;
CiMACD       *MACD;
CNet         *AE;
CBufferFloat Fractals;

//---
float    dError;
float    dUndefine;
float    dForecast;
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
   AE=new CNet(NULL);
   if(CheckPointer(AE)==POINTER_INVALID)
      return(INIT_FAILED);
   if(!AE.Load(FileName_AE+".nnw",dError,dUndefine,dForecast,dtStudied,false))
      return(INVALID_HANDLE);
//---
   if(!AE.GetLayerOutput(0,TempData))
      return(INIT_FAILED);
   HistoryBars=TempData.Total() / 12;
//---
   Net=new CNet(NULL);
   ResetLastError();
   if(CheckPointer(Net)==POINTER_INVALID||!Net.Load(FileName+".nnw",dError,dUndefine,dForecast,dtStudied,false))
     {
      printf("%s - %d -> Error of read %s prev Net %d",__FUNCTION__,__LINE__,FileName+".nnw",GetLastError());
      CArrayObj *Topology=new CArrayObj();
      if(CheckPointer(Topology)==POINTER_INVALID)
         return(INIT_FAILED);
      if(!AE.GetLayerOutput(5,TempData))
         return(INIT_FAILED);

   //--- 0
      CLayerDescription *desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=TempData.Total();
      desc.type=defNeuronBaseOCL;
      desc.optimization=ADAM;
      desc.activation=None;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 1
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=300;
      desc.type=defNeuron;
      desc.activation=LReLU;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 2
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=300;
      desc.type=defNeuron;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 3
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=300;
      desc.type=defNeuron;
      desc.activation=TANH;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

   //--- 4
      desc=new CLayerDescription();
      if(CheckPointer(desc)==POINTER_INVALID)
         return(INIT_FAILED);

      desc.count=3;
      desc.type=defNeuron;
      desc.activation=SIGMOID;
      desc.optimization=ADAM;
      if(!Topology.Add(desc))
         return(INIT_FAILED);

      delete Net;
      Net=new CNet(Topology);
      delete Topology;
      if(CheckPointer(Net)==POINTER_INVALID)
         return(INIT_FAILED);

      dError=-1;
      dUndefine=0;
      dForecast=0;
      dtStudied=0;
     }
//---
   bEventStudy=EventChartCustom(ChartID(),1,(long)MathMax(0,MathMin(iTime(Symb.Name(),PERIOD_CURRENT,(int)(100 * Net.recentAverageSmoothingFactor * (dForecast>=70 ? 1 : 10))),dtStudied)),0,"Init");
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
   if(CheckPointer(AE)!=POINTER_INVALID)
      delete AE;
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
   if(CheckPointer(TempData)!=POINTER_INVALID)
      delete TempData;
  }
//+------------------------------------------------------------------+
//| OnTick - expert tick function                                    |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!bEventStudy&&(dPrevSignal==-2||dtStudied<SeriesInfoInteger(Symb.Name(),TimeFrame,SERIES_LASTBAR_DATE)))
      bEventStudy=EventChartCustom(ChartID(),1,(long)MathMax(0,MathMin(iTime(Symb.Name(),PERIOD_CURRENT,(int)(100 * Net.recentAverageSmoothingFactor * (dForecast>=70 ? 1 : 10))),dtStudied)),0,"New Bar");
//---
   Comment(StringFormat("Вызов события %s; PrevSignal %.5f; Модель обучена %s -> %s",(string)bEventStudy,dPrevSignal,TimeToString(dtStudied),TimeToString(SeriesInfoInteger(Symb.Name(),TimeFrame,SERIES_LASTBAR_DATE))));
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
      Net.TrainMode(true);
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
   RSI.Refresh(OBJ_ALL_PERIODS);
   CCI.Refresh(OBJ_ALL_PERIODS);
   ATR.Refresh(OBJ_ALL_PERIODS);
   MACD.Refresh(OBJ_ALL_PERIODS);
//---
   MqlDateTime sTime;
   int         total=(int)(bars-MathMax(HistoryBars,0)-300);
   do
     {
      prev_er=dError;
      stop=IsStopped();
      bool add_loop=false;
      //---
      for(int it=total; it>=0&&!stop; it--)
        {
         int i=(int)((MathRand() * MathRand() / MathPow(32767,2)) * (total))+300;
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
            double rsi=RSI.Main(bar_t);
            double cci=CCI.Main(bar_t);
            double atr=ATR.Main(bar_t);
            double macd=MACD.Main(bar_t);
            double sign=MACD.Signal(bar_t);
            if(rsi==EMPTY_VALUE||cci==EMPTY_VALUE||atr==EMPTY_VALUE||macd==EMPTY_VALUE||sign==EMPTY_VALUE)
               continue;
            //---
            if(!TempData.Add((float)(Rates[bar_t].close-open))||!TempData.Add((float)(Rates[bar_t].high-open))||!TempData.Add((float)(Rates[bar_t].low-open))||!TempData.Add((float)Rates[bar_t].tick_volume / 1000.0f)||
               !TempData.Add(sTime.hour)||!TempData.Add(sTime.day_of_week)||!TempData.Add(sTime.mon)||
               !TempData.Add((float)rsi)||!TempData.Add((float)cci)||!TempData.Add((float)atr)||!TempData.Add((float)macd)||!TempData.Add((float)sign))
               break;
           }
         if(TempData.Total()<(int)HistoryBars * 12)
            continue;
         AE.feedForward(TempData,12,true,(CBufferFloat*)NULL);
         if(!AE.GetLayerOutput(5,TempData))
            break;
         Net.feedForward(TempData,2,true,(CBufferFloat*)NULL);
         Net.getResults(TempData);
         //---
         float sum=0;
         for(int res=0; res<3; res++)
           {
            float temp=exp(TempData.At(res));
            sum += temp;
            TempData.Update(res,temp);
           }
         for(int res=0; (res<3&&sum>0); res++)
            TempData.Update(res,TempData.At(res) / sum);
         //---
         switch(TempData.Argmax())
           {
            case 1:
               dPrevSignal=(TempData[1]!=TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal=-TempData[2];
               break;
            default:
               dPrevSignal=0;
               break;
           }
         if((GetTickCount64()-last_tick)>=250)
           {
            string s=StringFormat("Study -> Era %d -> %.2f -> Undefine %.2f%% foracast %.2f%%\n %d of %d -> %.2f%% \nError %.2f\n%s -> %.2f ->> Buy %.5f - Sell %.5f - Undef %.5f",count,dError,dUndefine,dForecast,it+1,total,(double)(it+1.0) / (total) * 100,Net.getRecentAverageError(),EnumToString(DoubleToSignal(dPrevSignal)),dPrevSignal,TempData[1],TempData[2],TempData[0]);
            Comment(s);
            last_tick=GetTickCount64();
           }
         stop=IsStopped();
         if(!stop)
           {
            TempData.Clear();
            bool sell=(Rates[i-1].high<=Rates[i].high&&Rates[i+1].high<Rates[i].high);
            bool buy=(Rates[i-1].low>=Rates[i].low&&Rates[i+1].low>Rates[i].low);
            TempData.Add(!(buy||sell));
            TempData.Add(buy);
            TempData.Add(sell);
            Net.backProp(TempData,NULL,NULL);
            ENUM_SIGNAL signal=DoubleToSignal(dPrevSignal);
            if(signal!=Undefine)
              {
               if((signal==Sell&&sell)||(signal==Buy&&buy))
                  dForecast += (100-dForecast) / Net.recentAverageSmoothingFactor;
               else
                  dForecast -= dForecast / Net.recentAverageSmoothingFactor;
               dUndefine -= dUndefine / Net.recentAverageSmoothingFactor;
              }
            else
              {
               if(!(buy||sell))
                  dUndefine += (100-dUndefine) / Net.recentAverageSmoothingFactor;
              }
           }
        }
      count++;
      for(int i=0; i<300; i++)
        {
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
            double rsi=RSI.Main(bar_t);
            double cci=CCI.Main(bar_t);
            double atr=ATR.Main(bar_t);
            double macd=MACD.Main(bar_t);
            double sign=MACD.Signal(bar_t);
            if(rsi==EMPTY_VALUE||cci==EMPTY_VALUE||atr==EMPTY_VALUE||macd==EMPTY_VALUE||sign==EMPTY_VALUE)
               continue;
            //---
            if(!TempData.Add((float)(Rates[bar_t].close-open))||!TempData.Add((float)(Rates[bar_t].high-open))||!TempData.Add((float)(Rates[bar_t].low-open))||!TempData.Add((float)Rates[bar_t].tick_volume / 1000.0f)||
               !TempData.Add(sTime.hour)||!TempData.Add(sTime.day_of_week)||!TempData.Add(sTime.mon)||
               !TempData.Add((float)rsi)||!TempData.Add((float)cci)||!TempData.Add((float)atr)||!TempData.Add((float)macd)||!TempData.Add((float)sign))
               break;
           }
         if(TempData.Total()<(int)HistoryBars * 12)
            continue;
         AE.feedForward(TempData,12,true,(CBufferFloat*)NULL);
         if(!AE.GetLayerOutput(5,TempData))
            break;
         Net.feedForward(TempData,2,true,(CBufferFloat*)NULL);
         Net.getResults(TempData);
         //---
         float sum=0;
         for(int res=0; res<3; res++)
           {
            float temp=exp(TempData.At(res));
            sum += temp;
            TempData.Update(res,temp);
           }
         for(int res=0; (res<3&&sum>0); res++)
            TempData.Update(res,TempData.At(res) / sum);
         //---
         switch(TempData.Argmax())
           {
            case 1:
               dPrevSignal=(TempData[1]!=TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal=(TempData[1]!=TempData[2] ?-TempData[2] : 0);
               break;
            default:
               dPrevSignal=0;
               break;
           }
         if(DoubleToSignal(dPrevSignal)==Undefine)
            DeleteObject(Rates[i].time);
         else
            DrawObject(Rates[i].time,dPrevSignal,Rates[i].high,Rates[i].low);
        }
      if(!stop)
        {
         dError=Net.getRecentAverageError();
         Net.Save(FileName+".nnw",dError,dUndefine,dForecast,Rates[0].time,false);
         printf("Era %d -> error %.2f %% forecast %.2f",count,dError,dForecast);
         ChartScreenShot(0,FileName+IntegerToString(count)+".png",750,400);
         int h=FileOpen(FileName+".csv",FILE_READ | FILE_WRITE | FILE_CSV);
         if(h!=INVALID_HANDLE)
           {
            FileSeek(h,0,SEEK_END);
            FileWrite(h,lr,count,dError,dUndefine,dForecast);
            FileFlush(h);
            FileClose(h);
           }
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
