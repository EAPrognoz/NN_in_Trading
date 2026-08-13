//+------------------------------------------------------------------+
//| Test.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                    |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                            |
//+------------------------------------------------------------------+
input int                  Agent = 1;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         *bAction;
CBufferFloat         *Result;
double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 bFillStack = true;
//+------------------------------------------------------------------+
//| Expert initialization function                              |
//+------------------------------------------------------------------+
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
   if(!RSI.BufferResize(StackSize + HistoryBars) || !CCI.BufferResize(StackSize + HistoryBars) ||
      !ATR.BufferResize(StackSize + HistoryBars) || !MACD.BufferResize(StackSize + HistoryBars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//--- load models
   float temp;
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Error load pretrained Actor");
      return INIT_FAILED;
     }
//---
   cActor.TrainMode(false);
//---
   cActor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   cActor.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr + AccountDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr + AccountDescr));
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   bFillStack = true;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                            |
//+------------------------------------------------------------------+
  {
//---
   DeleteObj(Result);
   DeleteObj(bAction);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                        |
//+------------------------------------------------------------------+
  {
   if(bFillStack)
     {
      if(!FillStack())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      bFillStack = false;
     }
//---
   if(!IsNewBar())
      return;
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), HistoryBars, Rates);
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
   bTime.Clear();
   bTime.Reserve(HistoryBars);
//---
   if(!CreateBuffers(0, GetPointer(bState), GetPointer(bTime), (CBufferFloat*)NULL))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
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
      position_discount += (current - PositionGetInteger(POSITION_TIME)) * multiplyer * MathAbs(profit);
     }
//---
   vector<float> account = vector<float>::Zeros(AccountDescr);
   account[0] = float(AccountInfoDouble(ACCOUNT_BALANCE) / EtalonBalance);
   account[1] = float((AccountInfoDouble(ACCOUNT_BALANCE) - PrevBalance) / PrevBalance);
   account[2] = float(AccountInfoDouble(ACCOUNT_EQUITY) / PrevBalance);
   account[3] = float((AccountInfoDouble(ACCOUNT_EQUITY) - PrevEquity) / PrevEquity);
   account[4] = (float)buy_value;
   account[5] = (float)sell_value;
   account[6] = float(buy_profit / PrevBalance);
   account[7] = float(sell_profit / PrevBalance);
   account[8] = float(position_discount / PrevBalance);
   double time = (double)Rates[0].time;
   double x = time / (double)(D'2024.01.01' - D'2023.01.01');
   account[9] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_MN1);
   account[10] = (float)MathCos(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_W1);
   account[11] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
   x = time / (double)PeriodSeconds(PERIOD_D1);
   account[12] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
//---
   if(!bState.AddArray(account))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   if(!cActor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)(CBufferFloat*)GetPointer(bTime)))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
//---
   cActor.getResults(bAction);
//---
   if(bAction.Total() < NActions)
      bAction.BufferInit(NActions, 0);
//---
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
   if(bAction[0] >= bAction[3])
     {
      bAction.Update(0, (bAction[0] - bAction[3])*10);
      bAction.Update(3, 0);
     }
   else
     {
      bAction.Update(3, (bAction[3] - bAction[0])*10);
      bAction.Update(0, 0);
     }
//--- buy control
   if(bAction[0] < min_lot ||
      (bAction[1] * MaxTP * Symb.Point()) <= 2 * stops ||
      (bAction[2] * MaxSL * Symb.Point()) <= stops
     )
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(bAction[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = NormalizeDouble(Symb.Ask() + bAction[1] * MaxTP * Symb.Point(), Symb.Digits());
      double buy_sl = NormalizeDouble(Symb.Ask() - bAction[2] * MaxSL * Symb.Point(), Symb.Digits());
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, buy_sl, buy_tp);
      if(buy_value != buy_lot)
        {
         if((buy_value - buy_lot) >= min_lot)
            ClosePartial(POSITION_TYPE_BUY, buy_value - buy_lot);
         else
            if((buy_lot - buy_value) >= min_lot)
               Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp);
        }
     }
//--- sell control
   if(bAction[3] < min_lot ||
      (bAction[4] * MaxTP * Symb.Point()) <= 2 * stops ||
      (bAction[5] * MaxSL * Symb.Point()) <= stops
     )
     {
      if(sell_value > 0)
         CloseByDirection(POSITION_TYPE_SELL);
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(bAction[3] - min_lot) / step_lot) * step_lot;;
      double sell_tp = NormalizeDouble(Symb.Bid() - bAction[4] * MaxTP * Symb.Point(), Symb.Digits());
      double sell_sl = NormalizeDouble(Symb.Bid() + bAction[5] * MaxSL * Symb.Point(), Symb.Digits());
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sell_sl, sell_tp);
      if(sell_value != sell_lot)
        {
         if((sell_value - sell_lot) >= min_lot)
            ClosePartial(POSITION_TYPE_SELL, sell_value - sell_lot);
         else
            if((sell_lot - sell_value) >= min_lot)
               Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp);
        }
     }
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
bool FillStack(void)
  {
   int start = StackSize + HistoryBars;
   int end = 0;
   int bars = CopyRates(Symb.Name(), TimeFrame, 2, start, Rates);
//---
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
      ReturnFalse;
//---
   if(RSI.BarsCalculated() < bars ||
      CCI.BarsCalculated() < bars ||
      ATR.BarsCalculated() < bars ||
      MACD.BarsCalculated() < bars)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   if(!ArraySetAsSeries(Rates, true))
      ReturnFalse;
   bars -= end + HistoryBars;
   if(bars < 0)
      ReturnFalse;
//---
   vector<float> result, target, neg_target;
//---
   uint ticks = GetTickCount();
   uint tester_print = 0;
   if(!cActor.Clear())
      ReturnFalse;
   for(int posit = start - HistoryBars - 1; posit >= end; posit--)
     {
      if(!CreateBuffers(posit, GetPointer(bState), GetPointer(bTime), NULL))
         ReturnFalse;
      vector<float> account = vector<float>::Zeros(AccountDescr);
      account[0] = float(AccountInfoDouble(ACCOUNT_BALANCE) / EtalonBalance);
      account[2] = 1;
      double time = (double)bTime[0];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      account[9] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      account[10] = (float)MathCos(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_W1);
      account[11] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      x = time / (double)PeriodSeconds(PERIOD_D1);
      account[12] = (float)MathSin(x != 0 ? 2.0 * M_PI * x : 0);
      //---
      if(!bState.AddArray(account))
         ReturnFalse;
      //--- Feed Forward
      if(!cActor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
         ReturnFalse;
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = (1.0 - double(posit - end) / (start - end - HistoryBars - NForecast))  * 100.0;
         string str = StringFormat("%-12s %6.2f%%", "Fill stack", percent);
         Comment(str);
         ticks = GetTickCount();
         if(MQLInfoInteger(MQL_TESTER) && percent >= tester_print)
           {
            Print(str);
            tester_print += 10;
           }
        }
     }
   Comment("");
   if(MQLInfoInteger(MQL_TESTER))
      Print("Fill stack Done");
//---
   return true;
  }
//+------------------------------------------------------------------+

