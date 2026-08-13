//+------------------------------------------------------------------+
//| Research.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
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
input int                  Agent = 1;
bool                 TrainMode = true;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
SState               sState;
STrajectory          Base;
STrajectory          Buffer[];
STrajectory          Frame[1];
CNet                 Actor;
CNet                 Schedule;
int                  Models = 1;
vector<float>        ModelsCount;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
//---
CBufferFloat         State1;
CBufferFloat         *Result;
//---
CSymbolInfo          Symb;
CTrade               Trade;
//---
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
vector<float>        prev_scheduler;
vector<float>        prev_actor;
vector<float>        prev_state;
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
   if(!RSI.BufferResize(HistoryBars) || !CCI.BufferResize(HistoryBars) ||
      !ATR.BufferResize(HistoryBars) || !MACD.BufferResize(HistoryBars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
   float temp;
//--- load models
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Schedule.Load(FileName + "Sch.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *schedule = new CArrayObj();
      if(!CreateDescriptions(actor, critic, schedule))
        {
         delete actor;
         delete critic;
         delete schedule;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !Schedule.Create(schedule))
        {
         delete actor;
         delete critic;
         delete schedule;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete schedule;
     }
   Schedule.getResults(Result);
   Models = Result.Total();
   Actor.getResults(Result);
   if(Result.Total() % Models != 0)
     {
      PrintFormat("The scope of the scheduler does not match the scope of the Agent (%d <> %d)", Models, Result.Total());
      return INIT_FAILED;
     }
//---
   ModelsCount = vector<float>::Zeros(Models);
   prev_scheduler.Init(Models);
   prev_actor.Init(Result.Total());
   prev_state.Init(9 + Result.Total());
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Print(ModelsCount);
   delete Result;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!IsNewBar())
      return;
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), HistoryBars, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   MqlDateTime sTime;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b].open;
      TimeToStruct(Rates[b].time, sTime);
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      float atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      sState.state[b * 12] = (float)Rates[b].close - open;
      sState.state[b * 12 + 1] = (float)Rates[b].high - open;
      sState.state[b * 12 + 2] = (float)Rates[b].low - open;
      sState.state[b * 12 + 3] = (float)Rates[b].tick_volume / 1000.0f;
      sState.state[b * 12 + 4] = (float)sTime.hour;
      sState.state[b * 12 + 5] = (float)sTime.day_of_week;
      sState.state[b * 12 + 6] = (float)sTime.mon;
      sState.state[b * 12 + 7] = rsi;
      sState.state[b * 12 + 8] = cci;
      sState.state[b * 12 + 9] = atr;
      sState.state[b * 12 + 10] = macd;
      sState.state[b * 12 + 11] = sign;
     }
//---
   sState.account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   sState.account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
   sState.account[2] = (float)AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   sState.account[3] = (float)AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   sState.account[4] = (float)AccountInfoDouble(ACCOUNT_PROFIT);
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += PositionGetDouble(POSITION_PROFIT);
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += PositionGetDouble(POSITION_PROFIT);
            break;
        }
     }
   sState.account[5] = (float)buy_value;
   sState.account[6] = (float)sell_value;
   sState.account[7] = (float)buy_profit;
   sState.account[8] = (float)sell_profit;
//---
   State1.AssignArray(sState.state);
   if(!Actor.feedForward(GetPointer(State1), 12, false, (CBufferFloat*)NULL))
      return;
   Actor.getResults(Result);
   State1.AssignArray(sState.account);
   State1.AddArray(Result);
   if(!Schedule.feedForward(GetPointer(State1), 12, false, (CBufferFloat*)NULL))
      return;
   vector<float> temp;
   State1.GetData(temp);
   float delta = (prev_state - temp).Sum();
   prev_state = temp;
   Schedule.getResults(Result);
   Result.GetData(temp);
   delta = MathAbs(prev_scheduler - temp).Sum();
   int model = GetAction(Result, 0, 1);
   prev_scheduler = temp;
   Actor.getResults(Result);
   Result.GetData(temp);
   delta = MathAbs(prev_actor - temp).Sum();
   prev_actor = temp;
   ModelsCount[model]++;
   int act = GetAction(Result, model, Models);
   switch(act)
     {
      case 0:
         if(!Trade.Buy(Symb.LotsMin(), Symb.Name()))
            act = 3;
         break;
      case 1:
         if(!Trade.Sell(Symb.LotsMin(), Symb.Name()))
            act = 3;
         break;
      case 2:
         for(int i = PositionsTotal() - 1; i >= 0; i--)
            if(PositionGetSymbol(i) == Symb.Name())
               if(!Trade.PositionClose(PositionGetInteger(POSITION_IDENTIFIER)))
                 {
                  act = 3;
                  break;
                 }
         break;
     }
//---
   float reward = 0;
   if(Base.Total > 0)
      reward = (sState.account[0] - Base.States[Base.Total - 1].account[0]) / 10;
   if(!Base.Add(sState, act, reward))
      ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//| Tester function                                                |
//+------------------------------------------------------------------+
//| | Strategy tester callback.                                      |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret = 0.0;
//---
   //LoadTotalBase();
   //double profit = TesterStatistics(STAT_PROFIT);
   //int total = ArraySize(Buffer);
   //if(ArrayResize(Buffer, total + 1, 0) < 0)
   //   return(ret);
   //Buffer[total] = Base;
   //SaveTotalBase();
//---
   return(ret);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                            |
//+------------------------------------------------------------------+
//| | Tester initialization handler.                                 |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
//---
  }
//+------------------------------------------------------------------+
//| TesterPass function                                            |
//+------------------------------------------------------------------+
//| | Tester pass callback.                                          |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
//---
   ulong pass;
   string name;
   long id;
   double value;
   STrajectory array[];
   while(FrameNext(pass, name, id, value, array))
     {
      int total = ArraySize(Buffer);
      if(name != MQLInfoString(MQL_PROGRAM_NAME))
         continue;
      if(id <= 0)
         continue;
     }
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//---
   int total = ArraySize(Buffer);
   printf("total %d", total);
   Print("Saving...");
   SaveTotalBase();
   Print("Saved");
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets Action.                                                   |
//+------------------------------------------------------------------+
int GetAction(CBufferFloat *results, int model, int total_models)
  {
   if(!results)
      return -1;
   int actions = results.Total() / total_models;
   vectorf temp;
   temp.Init(actions);
   int start = model * actions;
   for(int i = 0; i < actions; i++)
      temp[i] = results.At(start + i);
//---
   return (int)temp.ArgMax();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Checks NewBar.                                                 |
//+------------------------------------------------------------------+
bool IsNewBar(void)
  {
   static datetime last_bar = 0;
   if(last_bar >= iTime(Symb.Name(), TimeFrame, 0))
      return false;
//---
   last_bar = iTime(Symb.Name(), TimeFrame, 0);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves TotalBase.                                               |
//+------------------------------------------------------------------+
bool SaveTotalBase(void)
  {
   int total = ArraySize(Buffer);
   if(total < 0)
      return true;
   int handle = FileOpen(FileName + ".bd", FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle < 0)
      return false;
   if(FileWriteInteger(handle, total) < INT_VALUE)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Save(handle))
        {
         FileClose(handle);
         return false;
        }
   FileFlush(handle);
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Loads TotalBase.                                               |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+

