//+------------------------------------------------------------------+
//| Research.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
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
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
SState               sState;
STrajectory          Base;
STrajectory          Buffer[];
STrajectory          Frame[1];
CNet                 Encoder;
CNet                 Actor;
//---
float                dError;
datetime             dtStudied;
//---
CSymbolInfo          Symb;
CTrade               Trade;
//---
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bOnes;
CBufferFloat         *Result;
vector<float>        check;
double               PrevBalance = 0;
double               PrevEquity = 0;
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
   float temp;
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Cann't load pretrained models");
      return INIT_FAILED;
     }
//---
   COpenCLMy *OpenCL = Encoder.GetOpenCL();
   Actor.SetOpenCL(OpenCL);
   Encoder.TrainMode(false);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
//if(Base.Total > 0)
//  {
//   LoadTotalBase();
//   int total = ArraySize(Buffer);
//   if(ArrayResize(Buffer, total + 1, 0) < 0)
//      return;
//   Buffer[total] = Base;
//   printf("total %d", MathMin(total + 1, MaxReplayBuffer));
//   SaveTotalBase();
//  }
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
   Symb.Refresh();
   Symb.RefreshRates();
//---
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
      //---
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
   bState.AssignArray(sState.state);
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
   sState.account[7] = (float)Rates[0].time;
//---
   bAccount.Clear();
   bAccount.Add((float)((sState.account[0] - PrevBalance) / PrevBalance));
   bAccount.Add((float)(sState.account[1] / PrevBalance));
   bAccount.Add((float)((sState.account[1] - PrevEquity) / PrevEquity));
   bAccount.Add(sState.account[2]);
   bAccount.Add(sState.account[3]);
   bAccount.Add((float)(sState.account[4] / PrevBalance));
   bAccount.Add((float)(sState.account[5] / PrevBalance));
   bAccount.Add((float)(sState.account[6] / PrevBalance));
   double x = (double)Rates[0].time / (double)(D'2024.01.01' - D'2023.01.01');
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_MN1);
   bAccount.Add((float)MathCos(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_W1);
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_D1);
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
//---
   if(bAccount.GetIndex() >= 0)
      if(!bAccount.BufferWrite())
         return;
//---
   if(!Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//--- Actor
   if(!Actor.feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, GetPointer(Encoder),LatentLayer))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//---
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
//---
   vector<float> temp;
   Actor.getResults(temp);
   if(temp.Size() < NActions)
      temp = vector<float>::Zeros(NActions);
//---
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
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
//--- buy control
   if(temp[0] < min_lot || (temp[1] * MaxTP * Symb.Point()) <= stops || (temp[2] * MaxSL * Symb.Point()) <= stops)
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(temp[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = NormalizeDouble(Symb.Ask() + temp[1] * MaxTP * Symb.Point(), Symb.Digits());
      double buy_sl = NormalizeDouble(Symb.Ask() - temp[2] * MaxSL * Symb.Point(), Symb.Digits());
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
//--- sell control
   if(temp[3] < min_lot || (temp[4] * MaxTP * Symb.Point()) <= stops || (temp[5] * MaxSL * Symb.Point()) <= stops)
     {
      if(sell_value > 0)
         CloseByDirection(POSITION_TYPE_SELL);
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(temp[3] - min_lot) / step_lot) * step_lot;;
      double sell_tp = NormalizeDouble(Symb.Bid() - temp[4] * MaxTP * Symb.Point(), Symb.Digits());
      double sell_sl = NormalizeDouble(Symb.Bid() + temp[5] * MaxSL * Symb.Point(), Symb.Digits());
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
  }
//+------------------------------------------------------------------+

