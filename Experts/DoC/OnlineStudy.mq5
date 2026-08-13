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
//---
input int                  StudyIters  =  5;             //Iterations to Study
input int                  StudyPeriod =  120;           //Bars between Studies
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
SState               sState;
STrajectory          Base;
STrajectory          Buffer[];
CNet                 Agent;
CFQF                 RTG;
CNet                 AgentStudy;
CFQF                 RTGStudy;
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
CBufferFloat         *Result;
vector<float>        AgentResult;
double               PrevBalance = 0;
double               PrevEquity = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   LoadTotalBase();
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
   if(!RSI.BufferResize(NBarInPattern) || !CCI.BufferResize(NBarInPattern) ||
      !ATR.BufferResize(NBarInPattern) || !MACD.BufferResize(NBarInPattern))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//--- load models
   float temp;
   if(!Agent.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !RTG.Load(FileName + "RTG.nnw", dtStudied, true) ||
      !AgentStudy.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !RTGStudy.Load(FileName + "RTG.nnw", dtStudied, true))
     {
      PrintFormat("Can't load pretrained models");
      CArrayObj *agent = new CArrayObj();
      CArrayObj *rtg = new CArrayObj();
      if(!CreateDescriptions(agent, rtg))
        {
         delete agent;
         delete rtg;
         PrintFormat("Can't create description of models");
         return INIT_FAILED;
        }
      if(!Agent.Create(agent) ||
         !RTG.Create(rtg) ||
         !AgentStudy.Create(agent) ||
         !RTGStudy.Create(rtg))
        {
         delete agent;
         delete rtg;
         PrintFormat("Can't create models");
         return INIT_FAILED;
        }
      delete agent;
      delete rtg;
      //---
     }
//---
   Agent.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   AgentResult = vector<float>::Zeros(NActions);
//---
   Agent.GetLayerOutput(0, Result);
   if(Result.Total() != (NRewards + BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (NRewards + BarDescr * NBarInPattern + AccountDescr + TimeDescription + NActions));
      return INIT_FAILED;
     }
   Agent.Clear();
   RTG.Clear();
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
   AgentStudy.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
   RTGStudy.Save(FileName + "RTG.nnw", TimeCurrent(), true);
   delete Result;
   int total = ArraySize(Buffer);
   printf("Saving %d", MathMin(total + 1, MaxReplayBuffer));
   SaveTotalBase();
   Print("Saved");
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
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), NBarInPattern, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
//--- History data
   float atr = 0;
   for(int b = 0; b < (int)NBarInPattern; b++)
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
//--- Account description
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
   bState.Add((float)((sState.account[0] - PrevBalance) / PrevBalance));
   bState.Add((float)(sState.account[1] / PrevBalance));
   bState.Add((float)((sState.account[1] - PrevEquity) / PrevEquity));
   bState.Add(sState.account[2]);
   bState.Add(sState.account[3]);
   bState.Add((float)(sState.account[4] / PrevBalance));
   bState.Add((float)(sState.account[5] / PrevBalance));
   bState.Add((float)(sState.account[6] / PrevBalance));
//--- Time label
   double x = (double)Rates[0].time / (double)(D'2024.01.01' - D'2023.01.01');
   bState.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_MN1);
   bState.Add((float)MathCos(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_W1);
   bState.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_D1);
   bState.Add((float)MathSin(2.0 * M_PI * x));
//--- Prev action
   bState.AddArray(AgentResult);
//--- Return to go
   if(!RTG.feedForward(GetPointer(bState)))
      return;
   RTG.getResults(Result);
   bState.AddArray(Result);
//---
   if(!Agent.feedForward(GetPointer(bState), 1, false, (CBufferFloat*)NULL))
      return;
//---
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
//---
   vector<float> temp;
   Agent.getResults(temp);
//---
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
   float delta = MathAbs(AgentResult - temp).Sum();
   AgentResult = temp;
//--- buy control
   if(temp[0] < min_lot || (temp[1] * MaxTP * Symb.Point()) <= stops || (temp[2] * MaxSL * Symb.Point()) <= stops)
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(temp[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = Symb.NormalizePrice(Symb.Ask() + temp[1] * MaxTP * Symb.Point());
      double buy_sl = Symb.NormalizePrice(Symb.Ask() - temp[2] * MaxSL * Symb.Point());
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
      double sell_tp = Symb.NormalizePrice(Symb.Bid() - temp[4] * MaxTP * Symb.Point());
      double sell_sl = Symb.NormalizePrice(Symb.Bid() + temp[5] * MaxSL * Symb.Point());
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
//---
   int shift = BarDescr * (NBarInPattern - 1);
   sState.rewards[0] = bState[shift];
   sState.rewards[1] = bState[shift + 1] - 1.0f;
   if((buy_value + sell_value) == 0)
      sState.rewards[2] -= (float)(atr / PrevBalance);
   else
      sState.rewards[2] = 0;
   for(ulong i = 0; i < NActions; i++)
      sState.action[i] = AgentResult[i];
   if(!Base.Add(sState))
      ExpertRemove();
//---
   if((Bars(_Symbol, TimeFrame) % StudyPeriod) == 0)
      Train();
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   if(Base.Total >= StudyPeriod)
      if(ArrayResize(Buffer, total_tr + 1) == (total_tr + 1))
        {
         Buffer[total_tr] = Base;
         Buffer[total_tr].CumRevards();
         total_tr++;
        }
   int clear = Base.Total + StudyPeriod - Buffer_Size;
   if(clear > 0)
      Base.ClearFirstN(clear);
//---
   int count = 0;
   for(int i = 0; i < (total_tr + count); i++)
     {
      if(Buffer[i + count].Total < StudyPeriod)
        {
         count++;
         i--;
         continue;
        }
      if(count > 0)
         Buffer[i] = Buffer[i + count];
     }
   if(count > 0)
     {
      ArrayResize(Buffer, total_tr - count);
      total_tr = ArraySize(Buffer);
     }
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < StudyIters && !IsStopped() && !StopFlag); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * MathMax(Buffer[tr].Total - 2 * HistoryBars, MathMin(Buffer[tr].Total, 20)));
      if(i < 0)
        {
         iter--;
         continue;
        }
      vector<float> Actions = vector<float>::Zeros(NActions);
      AgentStudy.Clear();
      RTGStudy.Clear();
      for(int state = i; state < MathMin(Buffer[tr].Total - 2, int(i + HistoryBars * 1.5)); state++)
        {
         //--- History data
         bState.AssignArray(Buffer[tr].States[state].state);
         //--- Account description
         float prevBalance = (state == 0 ? Buffer[tr].States[state].account[0] : Buffer[tr].States[state - 1].account[0]);
         float prevEquity = (state == 0 ? Buffer[tr].States[state].account[1] : Buffer[tr].States[state - 1].account[1]);
         bState.Add((Buffer[tr].States[state].account[0] - prevBalance) / prevBalance);
         bState.Add(Buffer[tr].States[state].account[1] / prevBalance);
         bState.Add((Buffer[tr].States[state].account[1] - prevEquity) / prevEquity);
         bState.Add(Buffer[tr].States[state].account[2]);
         bState.Add(Buffer[tr].States[state].account[3]);
         bState.Add(Buffer[tr].States[state].account[4] / prevBalance);
         bState.Add(Buffer[tr].States[state].account[5] / prevBalance);
         bState.Add(Buffer[tr].States[state].account[6] / prevBalance);
         //--- Time label
         double x = (double)Buffer[tr].States[state].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         bState.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         bState.Add((float)MathCos(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_W1);
         bState.Add((float)MathSin(2.0 * M_PI * x));
         x = (double)Buffer[tr].States[state].account[7] / (double)PeriodSeconds(PERIOD_D1);
         bState.Add((float)MathSin(2.0 * M_PI * x));
         //--- Prev action
         bState.AddArray(Actions);
         //--- Return to go
         if(!RTGStudy.feedForward(GetPointer(bState)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         Result.AssignArray(Buffer[tr].States[state + 1].rewards);
         if(!RTGStudy.backProp(Result))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy Feed Forward
         bState.AddArray(Buffer[tr].States[state + 1].rewards);
         if(!AgentStudy.feedForward(GetPointer(bState), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //--- Policy study
         Actions.Assign(Buffer[tr].States[state].action);
         vector<float> result;
         AgentStudy.getResults(result);
         Result.AssignArray(CAGrad(Actions - result) + result);
         if(!AgentStudy.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            StopFlag = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Agent", iter * 100.0 / (double)(StudyIters), AgentStudy.getRecentAverageError());
            str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "RTG", iter * 100.0 / (double)(StudyIters), RTGStudy.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   Agent.WeightsUpdate(GetPointer(AgentStudy), 1.0f);
   RTG.WeightsUpdate(GetPointer(RTGStudy), 1.0f);
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for CAGrad.               |
//+------------------------------------------------------------------+
vector<float> CAGrad(vector<float> &grad)
  {
   matrix<float> GG = grad.Outer(grad);
   GG.ReplaceNan(0);
   if(MathAbs(GG).Sum() == 0)
      return grad;
   float scale = MathSqrt(GG.Diag() + 1.0e-4f).Mean();
   GG = GG / MathPow(scale, 2);
   vector<float> Gg = GG.Mean(1);
   float gg = Gg.Mean();
   vector<float> w = vector<float>::Zeros(grad.Size());
   float c = MathSqrt(gg + 1.0e-4f) * fCAGrad_C;
   vector<float> w_best = w;
   float obj_best = FLT_MAX;
   vector<float> moment = vector<float>::Zeros(w.Size());
   for(int i = 0; i < iCAGrad_Iters; i++)
     {
      vector<float> ww;
      w.Activation(ww, AF_SOFTMAX);
      float obj = ww.Dot(Gg) + c * MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f);
      if(MathAbs(obj) < obj_best)
        {
         obj_best = MathAbs(obj);
         w_best = w;
        }
      if(i < (iCAGrad_Iters - 1))
        {
         float loss = -obj;
         vector<float> derev = Gg + GG.MatMul(ww) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2) + ww.MatMul(GG) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2);
         vector<float> delta = derev * loss;
         ulong size = delta.Size();
         matrix<float> ident = matrix<float>::Identity(size, size);
         vector<float> ones = vector<float>::Ones(size);
         matrix<float> sm_der = ones.Outer(ww);
         sm_der = sm_der.Transpose() * (ident - sm_der);
         delta = sm_der.MatMul(delta);
         if(delta.Ptp() != 0)
            delta = delta / delta.Ptp();
         moment = delta * 0.8f + moment * 0.5f;
         w += moment;
         if(w.Ptp() != 0)
            w = w / w.Ptp();
        }
     }
   w_best.Activation(w, AF_SOFTMAX);
   float gw_norm = MathSqrt(w.MatMul(GG).Dot(w) + 1.0e-4f);
   float lmbda = c / (gw_norm + 1.0e-4f);
   vector<float> result = ((w * lmbda + 1.0f / (float)grad.Size()) * grad) / (1 + MathPow(fCAGrad_C, 2));
//---
   return result;
  }
//+------------------------------------------------------------------+

