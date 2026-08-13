//+------------------------------------------------------------------+
//| Test.mq5 |
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
/*input*/ string               ActorFile      =  "Act";
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
CNet                 Scheduler;
CFQF                 Actor;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
//---
CBufferFloat         State;
CBufferFloat         Account;
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
vector<float>        prev_actor;
float                prev_balance;
float                prev_equity;
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
//--- load models
   ResetLastError();
   if(!Actor.Load(FileName + ActorFile + ".nnw", dtStudied, true))
     {
      PrintFormat("Error of load Actor: %5d", GetLastError());
      return INIT_FAILED;
     }
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   prev_actor.Init(NActions);
   prev_balance = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   prev_equity = (float)AccountInfoDouble(ACCOUNT_EQUITY);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
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
//---
   MqlDateTime sTime;
   float atr = 0;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b].open;
      TimeToStruct(Rates[b].time, sTime);
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
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
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   double position_discount = 0;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
   int total = PositionsTotal();
   datetime current = TimeCurrent();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      double profit = PositionGetDouble(POSITION_PROFIT);
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += profit;
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += profit;
            break;
        }
      position_discount += profit - (current - PositionGetInteger(POSITION_TIME)) * multiplyer * MathAbs(profit);
     }
   sState.account[2] = (float)buy_value;
   sState.account[3] = (float)sell_value;
   sState.account[4] = (float)buy_profit;
   sState.account[5] = (float)sell_profit;
   sState.account[6] = (float)position_discount;
//---
   State.AssignArray(sState.state);
   Account.Clear();
   float PrevBalance = (Base.Total <= 0 ? sState.account[0] : Base.States[Base.Total - 1].account[0]);
   float PrevEquity = (Base.Total <= 0 ? sState.account[1] : Base.States[Base.Total - 1].account[1]);
   Account.Add((sState.account[0] - PrevBalance) / PrevBalance);
   Account.Add(sState.account[1] / PrevBalance);
   Account.Add((sState.account[1] - PrevEquity) / PrevEquity);
   Account.Add(sState.account[2]);
   Account.Add(sState.account[3]);
   Account.Add(sState.account[4] / PrevBalance);
   Account.Add(sState.account[5] / PrevBalance);
   Account.Add(sState.account[6] / PrevBalance);
//---
   if(Account.GetIndex() >= 0)
      if(!Account.BufferWrite())
         return;
   if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
      return;
//---
   Actor.getResults(Result);
   vector<float> temp;
   Result.GetData(temp);
   float delta = MathAbs(prev_actor - temp).Sum();
   prev_actor = temp;
   int act = Actor.getAction();
   float reward = Account[0]+MathMin(Account[5]+Account[6],0);
   switch(act)
     {
      case 0:
         if((buy_value+sell_value)==0)
              reward += atr;
         else
            if(buy_value>0.1)
              reward -= atr;
//---
         if(!Trade.Buy(Symb.LotsMin(), Symb.Name()))
            act = 3;
         break;
      case 1:
         if((buy_value+sell_value)==0)
              reward += atr;
         else
            if(sell_value>0.1)
              reward -= atr;
//---
         if(!Trade.Sell(Symb.LotsMin(), Symb.Name()))
            act = 3;
         break;
      case 2:
         if((buy_value+sell_value)>0)
              reward += atr;
         for(int i = PositionsTotal() - 1; i >= 0; i--)
            if(PositionGetSymbol(i) == Symb.Name())
               if(!Trade.PositionClose(PositionGetInteger(POSITION_IDENTIFIER)))
                 {
                  act = 3;
                  break;
                 }
         break;
      case 3:
         if((buy_value+sell_value)==0)
              reward -= atr;
     }
//---
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
   LoadTotalBase();
   double profit = TesterStatistics(STAT_PROFIT);
   int total = ArraySize(Buffer);
   if(ArrayResize(Buffer, total + 1, 0) < 0)
      return(ret);
   Buffer[total] = Base;
   SaveTotalBase();
//---
   return(ret);
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
   for(int i = (total > 100 ? total - 100 : 0); i < total; i++)
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

