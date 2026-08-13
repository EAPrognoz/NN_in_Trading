//+------------------------------------------------------------------+
//| Faza1.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#include "Cell.mqh"
#include <Math\Stat\Normal.mqh>
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input int                  Start =  100;
input int                  MinStartSteps = 96;
input int                  MaxSteps = 120;
input double               MinProfit = 10;
input double               MaxPosition = 0.1;
input int                  MaxLifeTime = 48;
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
bool                 TrainMode = true;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
Cell                 Base[Buffer_Size];
Cell                 Total[];
CSymbolInfo          Symb;
CTrade               Trade;
//---
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
int action_count = 0;
int actions[Buffer_Size];
Cell StartCell;
int bar = -1;
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
//---
   if(LoadTotalBase())
     {
      int total = ArraySize(Total);
      Cell temp[];
      ArrayResize(temp, total);
      int count = 0;
      for(int i = 0; i < total; i++)
         if(Total[i].total_actions >= MinStartSteps)
           {
            temp[count] = Total[i];
            count++;
           }
      if(count > 0)
        {
         count = (int)(((double)(MathRand() * MathRand()) / MathPow(32768.0, 2.0)) * (count - 1));
         StartCell = temp[count];
        }
      else
        {
         count = (int)(((double)(MathRand() * MathRand()) / MathPow(32768.0, 2.0)) * (total - 1));
         StartCell = Total[count];
        }
      PrintFormat("Start step %d, balance %10.2f, equity %10.2f", StartCell.total_actions, StartCell.state[240], StartCell.state[241]);
     }
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
//---
   if(!IsNewBar())
      return;
   bar++;
   if(bar < StartCell.total_actions)
     {
      switch(StartCell.actions[bar])
        {
         case 0:
            Trade.Buy(Symb.LotsMin(), Symb.Name());
            break;
         case 1:
            Trade.Sell(Symb.LotsMin(), Symb.Name());
            break;
         case 2:
            for(int i = PositionsTotal() - 1; i >= 0; i--)
               if(PositionGetSymbol(i) == Symb.Name())
                  Trade.PositionClose(PositionGetInteger(POSITION_IDENTIFIER));
            break;
        }
      return;
     }
//---
   if(bar == StartCell.total_actions)
     {
      ArrayCopy(actions, StartCell.actions, 0, 0, StartCell.total_actions);
      PrintFormat("Go-go balance %10.2f, equity %10.2f", AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
     }
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
   float state[249];
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
      state[b * 12] = (float)Rates[b].close - open;
      state[b * 12 + 1] = (float)Rates[b].high - open;
      state[b * 12 + 2] = (float)Rates[b].low - open;
      state[b * 12 + 3] = (float)Rates[b].tick_volume / 1000.0f;
      state[b * 12 + 4] = (float)sTime.hour;
      state[b * 12 + 5] = (float)sTime.day_of_week;
      state[b * 12 + 6] = (float)sTime.mon;
      state[b * 12 + 7] = rsi;
      state[b * 12 + 8] = cci;
      state[b * 12 + 9] = atr;
      state[b * 12 + 10] = macd;
      state[b * 12 + 11] = sign;
     }
//---
   state[240] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   state[240 + 1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
   state[240 + 2] = (float)AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   state[240 + 3] = (float)AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   state[240 + 4] = (float)AccountInfoDouble(ACCOUNT_PROFIT);
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   int total = PositionsTotal();
   datetime time_current = TimeCurrent();
   int first_order = 0;
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
      first_order = MathMax((int)(PositionGetInteger(POSITION_TIME) - time_current) / PeriodSeconds(TimeFrame), first_order);
     }
   state[240 + 5] = (float)buy_value;
   state[240 + 6] = (float)sell_value;
   state[240 + 7] = (float)buy_profit;
   state[240 + 8] = (float)sell_profit;
//---
   int act = (action_count < MaxSteps || first_order >= MaxLifeTime ? SampleAction(4) : 2);
   switch(act)
     {
      case 0:
         if(buy_value >= MaxPosition || !Trade.Buy(Symb.LotsMin(), Symb.Name()))
            act = 3;
         break;
      case 1:
         if(sell_value >= MaxPosition || !Trade.Sell(Symb.LotsMin(), Symb.Name()))
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
//--- copy cell
   Base[action_count].total_actions = action_count + StartCell.total_actions;
   actions[Base[action_count].total_actions] = act;
   if(action_count > 0)
     {
      ArrayCopy(Base[action_count].actions, actions, 0, 0, Base[action_count].total_actions + 1);
      Base[action_count - 1].value = (Base[action_count - 1].state[241] - state[241] + Base[action_count - 1].state[240] - state[240]) / 2.0f;
     }
   ArrayCopy(Base[action_count].state, state, 0, 0);
//---
   if(action_count > MaxSteps)
      ExpertRemove();
   action_count++;
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
   double profit = TesterStatistics(STAT_PROFIT);
   action_count--;
   if(profit >= MinProfit)
      FrameAdd(MQLInfoString(MQL_PROGRAM_NAME), action_count, profit, Base);
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
   LoadTotalBase();
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
   Cell array[];
   while(FrameNext(pass, name, id, value, array))
     {
      int total = ArraySize(Total);
      if(name != MQLInfoString(MQL_PROGRAM_NAME))
         continue;
      if(id <= 0)
         continue;
      if(ArrayResize(Total, total + (int)id, 10000) < 0)
         return;
      ArrayCopy(Total, array, total, 0, (int)id);
     }
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//---
   int total = ArraySize(Total);
   printf("total %d", total);
   Print("Saving...");
   SaveTotalBase();
   Print("Saved");
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements SampleAction.                                       |
//+------------------------------------------------------------------+
int SampleAction(int actions_space)
  {
   vectorf temp = vector<float>::Ones(actions_space);
   temp = (temp.CumSum())  / (float)actions_space;
   int err_code;
   float random = (float)MathRandomNormal(0.5, 0.4, err_code);
   if(random >= 0 && random <= 1)
      for(int i = 0; i < actions_space; i++)
         if(random <= temp[i])
            return i;
//---
   return (actions_space - 1);
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
   int total = ArraySize(Total);
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
      if(!Total[i].Save(handle))
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
   if(ArrayResize(Total, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Total[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+

