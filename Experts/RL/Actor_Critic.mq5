//+------------------------------------------------------------------+
//| Actor_Critic.mq5 |
//| Copyright 2022, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define ACTOR           Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_REINFORCE"
#define CRITIC          Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_Q-learning"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  2;             //Study period, years
uint                 HistoryBars =  20;            //Depth of history
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_H1;
input int                  SessionSize =  24 * 22;
input int                  Iterations = 1000;
input float                DiscountFactor =   0.999f;
int                        Actions     =  3;
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
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CSymbolInfo         Symb;
MqlRates            Rates[];
CNet                Actor;
CNet                Critic;
CArrayObj           States;
vectorf             vActions;
vectorf             vRewards;
vectorf             vProbs;
CBufferFloat       *TempData;
CiRSI               RSI;
CiCCI               CCI;
CiATR               ATR;
CiMACD              MACD;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
MqlDateTime          sTime;
float                BestLoss;
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
//---
   float temp1, temp2;
   if(!Actor.Load(ACTOR + ".nnw", dError, temp1, temp2, dtStudied, false) ||
      !Critic.Load(CRITIC + ".nnw", dError, temp1, temp2, dtStudied, false))
      return INIT_FAILED;
//---
   if(!Actor.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / 12;
   Actor.getResults(TempData);
   if(TempData.Total() != Actions)
      return INIT_PARAMETERS_INCORRECT;
   if(!vActions.Resize(SessionSize) ||
      !vRewards.Resize(SessionSize) ||
      !vProbs.Resize(SessionSize))
      return INIT_FAILED;
//---
   if(!Critic.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   if(HistoryBars != TempData.Total() / 12)
      return INIT_PARAMETERS_INCORRECT;
   Critic.getResults(TempData);
   if(TempData.Total() != Actions)
      return INIT_PARAMETERS_INCORRECT;
//---
   Actor.TrainMode(true);
   Critic.TrainMode(true);
   BestLoss = 1e37f;
//---
   bEventStudy = EventChartCustom(ChartID(), 1, 0, 0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(!!TempData)
      delete TempData;
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year <= 0)
      start_time.year = 1900;
   datetime st_time = StructToTime(start_time);
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, st_time, TimeCurrent(), Rates);
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) || !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
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
   int total = bars - (int)(HistoryBars + SessionSize + 2);
//---
   CBufferFloat* State;
   float loss = 0;
   uint count = 0;
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int error_code;
      int shift = (int)(fmin(fabs(Math::MathRandomNormal(0, 1, error_code)), 1) * (total) + SessionSize);
      States.Clear();
      for(int batch = 0; batch < SessionSize; batch++)
        {
         int i = shift - batch;
         State = new CBufferFloat();
         if(!State)
           {
            ExpertRemove();
            return;
           }
         int r = i + (int)HistoryBars;
         if(r > bars)
           {
            delete State;
            continue;
           }
         for(int b = 0; b < (int)HistoryBars; b++)
           {
            int bar_t = r - b;
            float open = (float)Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time, sTime);
            float rsi = (float)RSI.Main(bar_t);
            float cci = (float)CCI.Main(bar_t);
            float atr = (float)ATR.Main(bar_t);
            float macd = (float)MACD.Main(bar_t);
            float sign = (float)MACD.Signal(bar_t);
            if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
              {
               delete State;
               continue;
              }
            //---
            if(!State.Add((float)Rates[bar_t].close - open) || !State.Add((float)Rates[bar_t].high - open) || !State.Add((float)Rates[bar_t].low - open) || !State.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !State.Add(sTime.hour) || !State.Add(sTime.day_of_week) || !State.Add(sTime.mon) ||
               !State.Add(rsi) || !State.Add(cci) || !State.Add(atr) || !State.Add(macd) || !State.Add(sign))
              {
               delete State;
               break;
              }
           }
         if(IsStopped())
           {
            delete State;
            ExpertRemove();
            return;
           }
         if(State.Total() < (int)HistoryBars * 12)
           {
            delete State;
            continue;
           }
         if(!Actor.feedForward(GetPointer(State), 12, true))
           {
            delete State;
            ExpertRemove();
            return;
           }
         Actor.getResults(TempData);
         int action = GetAction(TempData);
         if(action < 0)
           {
            delete State;
            ExpertRemove();
            return;
           }
         double reward = Rates[i - 1].close - Rates[i - 1].open;
         switch(action)
           {
            case 0:
               if(reward < 0)
                  reward *= -20;
               else
                  reward *= 1;
               break;
            case 1:
               if(reward > 0)
                  reward *= -20;
               else
                  reward *= -1;
               break;
            default:
               if(batch == 0)
                  reward = -fabs(reward);
               else
                 {
                  switch((int)vActions[batch - 1])
                    {
                     case 0:
                        reward *= -1;
                        break;
                     case 1:
                        break;
                     default:
                        reward = -fabs(reward);
                        break;
                    }
                 }
               break;
           }
         if(!States.Add(State))
           {
            delete State;
            ExpertRemove();
            return;
           }
         vActions[batch] = (float)action;
         vRewards[SessionSize - batch - 1] = (float)reward;
         vProbs[SessionSize - batch - 1] = TempData.At(action);
         //---
        }
      //---
      vectorf rewards = vectorf::Full(SessionSize, 1);
      rewards = MathAbs(rewards.CumSum() - SessionSize);
      rewards = (vRewards * MathPow(vectorf::Full(SessionSize, DiscountFactor), rewards)).CumSum();
      rewards = rewards / fmax(rewards.Max(), fabs(rewards.Min()));
      loss = (fmin(count, 9) * loss + (rewards * MathLog(vProbs) * (-1)).Sum() / SessionSize) / fmin(count + 1, 10);
      count++;
      float total_reward = vRewards.Sum();
      //if(BestLoss >= loss)
        {
         if(!Actor.Save(ACTOR + ".nnw", loss, 0, 0, Rates[shift - SessionSize].time, false) ||
            !Critic.Save(CRITIC + ".nnw", Critic.getRecentAverageError(), 0, 0, Rates[shift - SessionSize].time, false))
            return;
         BestLoss = loss;
        }
      //---
      for(int batch = SessionSize - 1; batch >= 0; batch--)
        {
         State = States.At(batch);
         if(!Actor.feedForward(State) ||
            !Critic.feedForward(State))
           {
            ExpertRemove();
            return;
           }
         Critic.getResults(TempData);
         float value = TempData.At(TempData. Maximum(0, 3));
         if(!TempData.Update((int)vActions[batch], rewards[SessionSize - batch - 1]))
           {
            ExpertRemove();
            return;
           }
         if(!Critic.backProp(TempData))
           {
            ExpertRemove();
            return;
           }
         if(!TempData.BufferInit(Actions, 0) ||
            !TempData.Update((int)vActions[batch], rewards[SessionSize - batch - 1] - value))
           {
            ExpertRemove();
            return;
           }
         if(!Actor.backProp(TempData))
           {
            ExpertRemove();
            return;
           }
        }
      PrintFormat("Iteration %d, Cummulative reward %.5f, loss %.5f", iter, total_reward, loss);
     }
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | G et Ac ti on implementation.                                  |
//+------------------------------------------------------------------+
int GetAction(CBufferFloat* probability)
  {
   vectorf prob;
   if(!probability.GetData(prob))
      return -1;
   prob = prob.CumSum();
   prob = prob / prob.Max();
   int err_code;
   float random = (float)Math::MathRandomNormal(0.5, 0.5, err_code);
   if(random >= 1)
      return (int)prob.Size() - 1;
   for(int i = 0; i < (int)prob.Size(); i++)
      if(random <= prob[i])
         return i;
//---
   return -1;
  }
//+------------------------------------------------------------------+

