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
#define StudyOnline
#define Critic
#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_CURRENT;
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
CNet                 cEncoder[2];
CNet                 cActor[2];
CNet                 cCritic[6];
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
CBufferFloat         bTime;
CBufferFloat         bAccount;
CBufferFloat         *Result;
vector<float>        check;
double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 bFirstRun = true;
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
   CArrayObj *encoder = new CArrayObj();
   CArrayObj *actor = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(encoder, actor, critic))
     {
      delete encoder;
      delete actor;
      delete critic;
      return INIT_FAILED;
     }
   if(!cEncoder[0].Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !cEncoder[1].Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new State Encoder");
      for(uint i = 0; i < cEncoder.Size(); i++)
         if(!cEncoder[i].Create(encoder))
           {
            delete encoder;
            delete actor;
            delete critic;
            return INIT_FAILED;
           }
     }
   if(!cActor[0].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !cActor[1].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor[0].Create(actor) ||
         !cActor[1].Create(actor))
        {
         delete encoder;
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      cActor[1].SetOpenCL(cActor[0].GetOpenCL());
      cActor[1].WeightsUpdate(GetPointer(cActor[0]), 1);
     }
//---
   bool result = true;
   for(uint i = 0; (i < cCritic.Size() && result); i++)
      if(!cCritic[i].Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true))
         result = false;
   if(!result)
     {
      Print("Create new Critic model");
      for(uint i = 0; i < cCritic.Size(); i++)
         if(!cCritic[i].Create(critic))
           {
            delete encoder;
            delete actor;
            delete critic;
            return INIT_FAILED;
           }
      for(int i = 1; i < 3; i++)
        {
         cCritic[2 * i].SetOpenCL(cCritic[0].GetOpenCL());
         cCritic[2 * i + 1].SetOpenCL(cCritic[1].GetOpenCL());
         if(!cCritic[2 * i].WeightsUpdate(GetPointer(cCritic[0]), 1) ||
            !cCritic[2 * i + 1].WeightsUpdate(GetPointer(cCritic[1]), 1)
           )
           {
            delete encoder;
            delete actor;
            delete critic;
            return INIT_FAILED;
           }
        }
     }
   delete encoder;
   delete actor;
   delete critic;
//---
   for(int i = 0; i < 2; i++)
     {
      cEncoder[i].TrainMode(true);
      cEncoder[i].Clear();
      cActor[i].TrainMode(true);
      cCritic[i].TrainMode(true);
      cEncoder[i].SetOpenCL(cActor[0].GetOpenCL());
      cCritic[i].SetOpenCL(cActor[0].GetOpenCL());
      cEncoder[i].Clear();
      cActor[i].Clear();
      cCritic[i].Clear();
     }
//---
   cActor[1].SetOpenCL(cActor[0].GetOpenCL());
   for(uint i = 2; i < cCritic.Size(); i++)
     {
      cCritic[i].TrainMode(true);
      cCritic[i].SetOpenCL(cActor[0].GetOpenCL());
      cCritic[i].Clear();
     }
//---
   cActor[0].getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   cEncoder[0].GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   bFirstRun = true;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(!(reason == REASON_RECOMPILE))
     {
      cActor[0].Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      cEncoder[0].Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      if(cCritic[0].getRecentAverageError() <= cCritic[1].getRecentAverageError())
         cCritic[0].Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
      else
         cCritic[1].Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
     }
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
   bTime.Clear();
   bTime.Reserve(HistoryBars);
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
      if(!bTime.Add(float(Rates[b].time)))
         return;
     }
//---
   if(bTime.GetIndex() >= 0)
      bTime.BufferWrite();
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
      position_discount += (current - PositionGetInteger(POSITION_TIME)) * multiplyer * MathAbs(profit);
     }
   sState.account[2] = (float)buy_value;
   sState.account[3] = (float)sell_value;
   sState.account[4] = (float)buy_profit;
   sState.account[5] = (float)sell_profit;
   sState.account[6] = (float)position_discount;
   sState.account[7] = (float)Rates[0].time;
//---
   bAccount.Clear();
   bAccount.Add((float)(sState.account[0] / EtalonBalance));
   bAccount.Add((float)((sState.account[0] - PrevBalance) / PrevBalance));
   bAccount.Add((float)(sState.account[1] / PrevBalance));
   bAccount.Add((float)((sState.account[1] - PrevEquity) / PrevEquity));
   bAccount.Add(sState.account[2]);
   bAccount.Add(sState.account[3]);
   bAccount.Add((float)(sState.account[4] / PrevBalance));
   bAccount.Add((float)(sState.account[5] / PrevBalance));
   bAccount.Add((float)(sState.account[6] / PrevBalance));
   double time = (double)Rates[0].time;
   double x = time / (double)(D'2024.01.01' - D'2023.01.01');
   bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_MN1);
   bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_W1);
   bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_D1);
   bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
//---
   if(bAccount.GetIndex() >= 0)
      if(!bAccount.BufferWrite())
         return;
//---
   bState.AssignArray(sState.state);
   if(!bFirstRun)
     {
      //--- Target Nets
      if(!cEncoder[1].feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)(CBufferFloat*)GetPointer(bTime))
         || !cActor[1].feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, GetPointer(cEncoder[1]), LatentLayer)
         || !cCritic[2].feedForward(GetPointer(cActor[1]), -1, GetPointer(cEncoder[1]), LatentLayer)
         || !cCritic[3].feedForward(GetPointer(cActor[1]), -1, GetPointer(cEncoder[1]), LatentLayer)
         || !cCritic[4].feedForward(GetPointer(cActor[1]), -1, GetPointer(cEncoder[1]), LatentLayer)
         || !cCritic[5].feedForward(GetPointer(cActor[1]), -1, GetPointer(cEncoder[1]), LatentLayer)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      //--- Critic 1
      cCritic[2].getResults(Result);
      float reward = Result[0];
      cCritic[4].getResults(Result);
      reward = (reward + Result[0]) / 2 * DiscFactor + float(sState.account[1] - PrevEquity + sState.account[0] - PrevBalance);
      if(PrevBalance < 50)
         reward -= 1000;
      if(reward < 0)
         reward *= 2;
      else
         if(reward == 0)
            reward = -10;
      Result.Clear();
      if(!Result.Add(reward)
         || !cCritic[0].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      if(cCritic[0].getRecentAverageError() <= cCritic[1].getRecentAverageError() &&
         (MathRand() % ActorUpdate) == 0)
         if(!cActor[0].backPropGradient((CNet*)GetPointer(cEncoder[0]), LatentLayer, -1, false) ||
            !cEncoder[0].backPropGradient((CBufferFloat*)NULL, NULL, LatentLayer, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
      //--- Critic 2
      cCritic[3].getResults(Result);
      reward = Result[0];
      cCritic[5].getResults(Result);
      // Function implementation for float.
      reward = float((reward + Result[0]) / 2 * DiscFactor +
                     (sState.account[1] - PrevEquity + sState.account[0] - PrevBalance) / EtalonBalance);
      if(PrevBalance < 50)
         reward -= 1000;
      if(reward < 0)
         reward *= 2;
      else
         if(reward == 0)
            reward = -10;
      Result.Clear();
      if(!Result.Add(reward)
         || !cCritic[1].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      if(cCritic[0].getRecentAverageError() > cCritic[1].getRecentAverageError() &&
         (MathRand() % ActorUpdate) == 0)
         if(!cActor[0].backPropGradient((CNet*)GetPointer(cEncoder[0]), LatentLayer, -1, false) ||
            !cEncoder[0].backPropGradient((CBufferFloat*)NULL, NULL, LatentLayer, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
      if(PrevBalance < 50)
         ExpertRemove();
     }
//--- New state
   if(!cEncoder[0].feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)(CBufferFloat*)GetPointer(bTime))
      || !cActor[0].feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, (CNet*)GetPointer(cEncoder[0]), LatentLayer)
      || !cCritic[0].feedForward((CNet*)GetPointer(cActor[0]), -1, (CNet*)GetPointer(cEncoder[0]), LatentLayer)
      || !cCritic[1].feedForward((CNet*)GetPointer(cActor[0]), -1, (CNet*)GetPointer(cEncoder[0]), LatentLayer)
     )
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//---
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
//---
   vector<float> temp;
   cActor[0].getResults(temp);
//---
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
   if(temp[0] < min_lot || (temp[1] * MaxTP * Symb.Point()) <= 2 * stops || (temp[2] * MaxSL * Symb.Point()) <= stops)
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
         if((buy_value - buy_lot) >= min_lot)
            ClosePartial(POSITION_TYPE_BUY, buy_value - buy_lot);
         else
            if((buy_lot - buy_value) >= min_lot)
               if(!Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp))
                  if(Trade.CheckResultRetcode() == 10019)
                    {
                     Result.Clear();
                     Result.Add(float(-100 * (buy_lot - buy_value)));
                     if(!cCritic[0].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer) ||
                        !cCritic[1].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer)
                       )
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        return;
                       }
                    }
        }
     }
//--- sell control
   if(temp[3] < min_lot || (temp[4] * MaxTP * Symb.Point()) <= 2 * stops || (temp[5] * MaxSL * Symb.Point()) <= stops)
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
         if((sell_value - sell_lot) >= min_lot)
            ClosePartial(POSITION_TYPE_SELL, sell_value - sell_lot);
         else
            if((sell_lot - sell_value) >= min_lot)
               if(!Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp))
                  if(Trade.CheckResultRetcode() == 10019)
                    {
                     Result.Clear();
                     Result.Add(float(-100 * (sell_lot - sell_value)));
                     if(!cCritic[0].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer) ||
                        !cCritic[1].backProp(Result, (CNet*)GetPointer(cEncoder[0]), LatentLayer)
                       )
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        return;
                       }
                    }
        }
     }
//---
   bFirstRun = false;
//---
   if((int(Rates[0].time / PeriodSeconds(TimeFrame)) % TragetUpdate) == 0)
     {
      if(MathRand() / 32767.0 > 0.5)
         cCritic[2].WeightsUpdate(GetPointer(cCritic[0]), tau);
      else
         cCritic[4].WeightsUpdate(GetPointer(cCritic[0]), tau);
      if(MathRand() / 32767.0 > 0.5)
         cCritic[3].WeightsUpdate(GetPointer(cCritic[1]), tau);
      else
         cCritic[5].WeightsUpdate(GetPointer(cCritic[1]), tau);
      cActor[1].WeightsUpdate(GetPointer(cActor[0]), tau);
     }
  }
//+------------------------------------------------------------------+

