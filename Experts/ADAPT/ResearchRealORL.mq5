//+------------------------------------------------------------------+
//|                                              ResearchRealORL.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "Харнесс исследований ADAPT с реальным онлайн-обучением."
#property description "Прогоняет исследование на реальном потоке баров и калибрует целевые сети."
#property description "Только для тестов; ордера не выставляет."
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input double               MinProfit   =  -10000;
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
input int                  Agent = 1;
STrajectory          Base;
STrajectory          Buffer[];
STrajectory          Frame[1];
CDeals               Deals;
float                dError;
datetime             dtStudied;
CSymbolInfo          Symb;
CTrade               Trade;
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
double               PrevBalance = 0;
double               PrevEquity = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
  {
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
   if(!RSI.BufferResize(HistoryBars) || !CCI.BufferResize(HistoryBars) ||
      !ATR.BufferResize(HistoryBars) || !MACD.BufferResize(HistoryBars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//--- Load History.
   if(!Deals.LoadDeals(SignalFile(Agent), "EURUSD", SymbolInfoDouble(_Symbol, SYMBOL_POINT)))
      return INIT_FAILED;
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
  {
   if(!IsNewBar())
      return;
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), HistoryBars, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
   float atr = 0;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b].open;
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      int shift = b * BarDescr;
      sState.state[shift] = (float)(Rates[b].close - open);
      sState.state[shift + 1] = (float)(Rates[b].high - open);
      sState.state[shift + 2] = (float)(Rates[b].low - open);
      sState.state[shift + 3] = (float)(Rates[b].tick_volume / 1000.0f);
      sState.state[shift + 4] = rsi;
      sState.state[shift + 5] = cci;
      sState.state[shift + 6] = atr;
      sState.state[shift + 7] = macd;
      sState.state[shift + 8] = sign;
     }
   sState.account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   sState.account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
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
   sState.account[7] = (float)Rates[0].time;
   sState.rewards[0] = float((sState.account[0] - PrevBalance) / PrevBalance);
   sState.rewards[1] = float(1.0 - sState.account[1] / PrevBalance);
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
   vector<float> temp = Deals.Action(TimeCurrent(), SymbolInfoDouble(_Symbol, SYMBOL_ASK), SymbolInfoDouble(_Symbol, SYMBOL_BID), PeriodSeconds(TimeFrame));
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = MathMax(Symb.StopsLevel(), 1) * Symb.Point();
   if(temp[0] >= temp[3])
     {
      temp[0] -= temp[3];
      temp[3] = 0;
     }
   else
     {
      temp[3] -= temp[0];
      temp[0] = 0;
     }
//--- Buy Control.
   if(temp[0] < min_lot || (temp[1] > 0 && (temp[1] * MaxTP * Symb.Point()) <= stops) || (temp[2] > 0 && (temp[2] * MaxSL * Symb.Point()) <= stops))
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(temp[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = (temp[1] > 0 ? NormalizeDouble(Symb.Ask() + temp[1] * MaxTP * Symb.Point(), Symb.Digits()) : 0);
      double buy_sl = (temp[2] > 0 ? NormalizeDouble(Symb.Ask() - temp[2] * MaxSL * Symb.Point(), Symb.Digits()) : 0);
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, buy_sl, buy_tp);
      if(buy_value != buy_lot)
        {
         if(buy_value > buy_lot)
            ClosePartial(POSITION_TYPE_BUY, buy_value - buy_lot);
         else
            Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp);
        }
     }
//--- Sell Control.
   if(temp[3] < min_lot || (temp[4] > 0 && (temp[4] * MaxTP * Symb.Point()) <= stops) || (temp[5] > 0 && (temp[5] * MaxSL * Symb.Point()) <= stops))
     {
      if(sell_value > 0)
         CloseByDirection(POSITION_TYPE_SELL);
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(temp[3] - min_lot) / step_lot) * step_lot;;
      double sell_tp = (temp[4] > 0 ? NormalizeDouble(Symb.Bid() - temp[4] * MaxTP * Symb.Point(), Symb.Digits()) : 0);
      double sell_sl = (temp[5] > 0 ? NormalizeDouble(Symb.Bid() + temp[5] * MaxSL * Symb.Point(), Symb.Digits()) : 0);
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sell_sl, sell_tp);
      if(sell_value != sell_lot)
        {
         if(sell_value > sell_lot)
            ClosePartial(POSITION_TYPE_SELL, sell_value - sell_lot);
         else
            Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp);
        }
     }
   if((buy_value + sell_value) == 0)
      sState.rewards[2] -= (float)(atr / PrevBalance);
   else
      sState.rewards[2] = 0;
   for(ulong i = 0; i < NActions; i++)
      sState.action[i] = temp[i];
   if(!Base.Add(sState))
      ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
   double ret = 0.0;
   double profit = TesterStatistics(STAT_PROFIT);
   Frame[0] = Base;
   if(profit >= MinProfit && profit != 0)
      FrameAdd(MQLInfoString(MQL_PROGRAM_NAME), 1, profit, Frame);
   return(ret);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
   LoadTotalBase();
  }
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
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
      if(total >= MaxReplayBuffer)
        {
         for(int a = 0; a < id; a++)
           {
            float min = FLT_MAX;
            int min_tr = 0;
            for(int i = 0; i < total; i++)
              {
               if(Buffer[i].Total<=0)
                {
                 min = FLT_MIN;
                 min_tr = i;
                 break;
                }
               float prof = Buffer[i].States[Buffer[i].Total - 1].account[1];
               if(prof < min)
                 {
                  min = MathMin(prof, min);
                  min_tr = i;
                 }
              }
            float prof = array[a].States[array[a].Total - 1].account[1];
            if(min <= prof)
              {
               Buffer[min_tr] = array[a];
               PrintFormat("Replace %.2f to %.2f -> bars %d", min, prof, array[a].Total);
              }
           }
        }
      else
        {
         if(ArrayResize(Buffer, total + (int)id, 10) < 0)
            return;
         ArrayCopy(Buffer, array, total, 0, (int)id);
        }
     }
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
  {
   int total = ArraySize(Buffer);
   printf("total %d", MathMin(total, MaxReplayBuffer));
   Print("Saving...");
   SaveTotalBase();
   Print("Saved");
  }
//+------------------------------------------------------------------+