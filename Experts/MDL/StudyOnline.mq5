//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "Харнесс онлайн-обучения и калибровки политик MDL."
#property description "Выполняет обучающие переходы на закрытых барах в режиме онлайн-калибровки целевых сетей."
#property description "Только для тестов; ордера не выставляет."
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
input group                "---- Other ----"
input int                  Agent = 1;
CNet                 cActor[2];
CNet                 cCritic[3];
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         *bAction;
CBufferFloat         *Result;
vector<float>        check;
double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 bFirstRun = true;
bool                 bFillStack = true;
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
   CArrayObj *actor = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(actor, critic))
     {
      delete actor;
      delete critic;
      return INIT_FAILED;
     }
   if(!cActor[0].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !cActor[1].Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor[0].Create(actor) ||
         !cActor[1].Create(actor))
        {
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
            DeleteObj(actor);
            DeleteObj(critic);
            return INIT_FAILED;
           }
      for(uint i = 1; i < cCritic.Size(); i++)
        {
         cCritic[i].SetOpenCL(cCritic[0].GetOpenCL());
         if(!cCritic[i].WeightsUpdate(GetPointer(cCritic[0]), 1))
           {
            DeleteObj(actor);
            DeleteObj(critic);
            return INIT_FAILED;
           }
        }
     }
   DeleteObj(actor);
   DeleteObj(critic);
//---
   for(int i = 0; i < 2; i++)
     {
      cActor[i].TrainMode(i == 0);
      cCritic[i].TrainMode(i == 0);
      cCritic[i].SetOpenCL(cActor[0].GetOpenCL());
      cActor[i].Clear();
      cCritic[i].Clear();
     }
//---
   cActor[1].SetOpenCL(cActor[0].GetOpenCL());
   for(uint i = 2; i < cCritic.Size(); i++)
     {
      cCritic[i].TrainMode(false);
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
   cActor[0].GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr + AccountDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr + AccountDescr));
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
   if(!(reason == REASON_RECOMPILE || reason == REASON_INITFAILED))
     {
      cActor[0].Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      cCritic[0].Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
     }
   DeleteObj(Result);
   DeleteObj(bAction);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
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
//---
   if(!bFirstRun)
     {
      //--- Target Nets
      if(!cActor[1].feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime))
         || !cCritic[1].feedForward(GetPointer(cActor[1]), -1, GetPointer(cActor[1]), LatentLayer)
         || !cCritic[2].feedForward(GetPointer(cActor[1]), -1, GetPointer(cActor[1]), LatentLayer)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         bFirstRun = true;
         return;
        }
      //--- Critic
      cCritic[1].getResults(Result);
      float forward = Result[0];
      cCritic[2].getResults(Result);
      forward = MathMin(forward, Result[0]);
      float reward = float(forward * DiscFactor +
                           (PrevEquity - AccountInfoDouble(ACCOUNT_EQUITY) +
                            PrevBalance - AccountInfoDouble(ACCOUNT_BALANCE)) * PrevBalance / EtalonBalance);
      if(MathMax(buy_value, sell_value) < Symb.LotsMin())
        {
         double marg = 0;
         if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
            marg = 200;
         double point_cost = Symb.TickValue() / Symb.TickSize();
         double loss = MathAbs(bState[0]) *
                       point_cost * PrevBalance / (10 * marg);
         reward -= float(loss * PrevBalance / EtalonBalance);
        }
      //---
      Result.Clear();
      if(!Result.Add(reward)
         || !cCritic[0].TrainMode(true)
         || !cCritic[0].backProp(Result, (CNet*)GetPointer(cActor[0]), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         return;
        }
      if((PrevEquity - AccountInfoDouble(ACCOUNT_EQUITY) + PrevBalance - AccountInfoDouble(ACCOUNT_BALANCE)) <= 0 &&
         //--- Implements MathAbs.
         MathAbs(bState[0]) > Symb.StopsLevel()*Symb.Point())
        {
         double point_cost = Symb.TickValue() / Symb.TickSize();
         if(point_cost > 0)
           {
            vector<float> oracul = vector<float>::Zeros(NActions);
            double marg = 0;
            if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
               marg = 200;
            double lot = MathMax(PrevBalance / (10 * marg), Symb.LotsMin());
            if(bState[0] > 0)
              {
               lot = MathMin(lot, PrevBalance / ((-100 * bState[2*HistoryBars] + Symb.Spread() * Symb.Point()) * point_cost));
               oracul[0] = (float)MathMin(lot, 1);
               oracul[1] = (float)MathMin(3 * bState[6] / (Symb.Point() * MaxTP), 1);
               oracul[2] = (float)MathMin(bState[6] / (Symb.Point() * MaxSL), 1);
              }
            else
              {
               lot = MathMin(lot, PrevBalance / ((100 * bState[HistoryBars] + Symb.Spread() * Symb.Point()) * point_cost));
               oracul[3] = (float)MathMin(lot, 1);
               oracul[4] = (float)MathMin(3 * bState[6] / (Symb.Point() * MaxTP), 1);
               oracul[5] = (float)MathMin(bState[6] / (Symb.Point() * MaxSL), 1);
              }
            reward = float(MathAbs(bState[0]) * point_cost * lot);
            reward = float(forward * DiscFactor + reward * PrevBalance / EtalonBalance);
            if(!Result.AssignArray(oracul) ||
               !cCritic[0].feedForward(Result, 1, false, (CNet*)GetPointer(cActor[0]), LatentLayer))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               bFirstRun = true;
               return;
              }
            if(!Result.Update(0, reward)
               || !cCritic[0].backProp(Result, (CNet*)GetPointer(cActor[0]), LatentLayer))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               bFirstRun = true;
               return;
              }
           }
        }
      //---
      if((MathRand() % ActorUpdate) == 0)
        {
         if(!cCritic[0].feedForward(GetPointer(bAction), -1, false, (CNet*)GetPointer(cActor[0]), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
         cCritic[0].getResults(Result);
         if(!Result.Update(0, float(MathMax(Result[0], 0) + PrevBalance * 0.1 * PrevBalance / EtalonBalance)) ||
            !cCritic[0].TrainMode(false) ||
            !cCritic[0].backProp(Result, (CNet*)GetPointer(cActor[0]), LatentLayer) ||
            !cActor[0].backPropGradient((CBufferFloat*)GetPointer(bTime), NULL, -1, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            return;
           }
        }
      if(PrevBalance < 20)
         ExpertRemove();
     }
//--- New state
   if(!cActor[0].feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)(CBufferFloat*)GetPointer(bTime))
      || !cCritic[0].feedForward((CNet*)GetPointer(cActor[0]), -1, (CNet*)GetPointer(cActor[0]), LatentLayer)
     )
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
//---
   cActor[0].getResults(bAction);
//---
   if(bAction.Total() < NActions)
      bAction.BufferInit(NActions, 0);
//---
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
   if(bAction[0] >= bAction[3])
     {
      bAction.Update(0, bAction[0] - bAction[3]);
      bAction.Update(3, 0);
     }
   else
     {
      bAction.Update(3, bAction[3] - bAction[0]);
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
               if(!Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp))
                  if(Trade.CheckResultRetcode() == 10019)
                    {
                     Result.Clear();
                     Result.Add(float(-100 * (buy_lot - buy_value)));
                     if(!cCritic[0].backProp(Result, (CNet*)GetPointer(cActor[0]), LatentLayer))
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        return;
                       }
                    }
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
               if(!Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp))
                  if(Trade.CheckResultRetcode() == 10019)
                    {
                     Result.Clear();
                     Result.Add(float(-100 * (sell_lot - sell_value)));
                     if(!cCritic[0].backProp(Result, (CNet*)GetPointer(cActor[0]), LatentLayer))
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
   if((int(Rates[0].time / PeriodSeconds(TimeFrame)) % TargetUpdate) == 0)
     {
      if(MathRand() / 32767.0 > 0.5)
         cCritic[1].WeightsUpdate(GetPointer(cCritic[0]), tau);
      else
         cCritic[2].WeightsUpdate(GetPointer(cCritic[0]), tau);
      cActor[1].WeightsUpdate(GetPointer(cActor[0]), tau);
     }
  }
//+------------------------------------------------------------------+
//| Implements FillStack.                                            |
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
//---
   for(uint i = 0; i < cActor.Size(); i++)
      if(!cActor[i].Clear())
         ReturnFalse;
   for(uint i = 0; i < cCritic.Size(); i++)
      if(!cCritic[i].Clear())
         ReturnFalse;
   for(int posit = start - HistoryBars - 1; (posit >= end && !IsStopped()); posit--)
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
      for(uint i = 0; (i < cActor.Size() && !IsStopped()); i++)
         if(!cActor[i].feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
            ReturnFalse;
      for(uint i = 0; (i < cCritic.Size() && !IsStopped()); i++)
         if(!cCritic[i].feedForward(GetPointer(cActor[int(i > 0)]), -1, GetPointer(cActor[int(i > 0)]), LatentLayer))
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