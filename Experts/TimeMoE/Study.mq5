//+------------------------------------------------------------------+
//| Study.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define Study
#define StudyOnline
#include "Trajectory.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input datetime             Start          = D'2010.01.01';
input datetime             End            = D'2025.01.01';
input int                  Iterations     = 1000000;
input double               MinBalance     = 50;
input double               MaxBalance     = 300;
input group                "---- Indicators ----"
input ENUM_TIMEFRAMES      TimeFrame      =  PERIOD_M1;
//---
input group                "---- RSI ----"
input int                  RSIPeriod      =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice       =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod      =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice       =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod      =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod     =  12;            //Fast
input int                  SlowPeriod     =  26;            //Slow
input int                  SignalPeriod   =  9;             //Signal
input ENUM_APPLIED_PRICE   MACDPrice      =  PRICE_CLOSE;   //Applied price
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet                 cEncoder;
CNet                 caForecast[3];
CNet                 cActor;
CNet                 cCritic;
CNet                 cDirector;
//---
CSymbolInfo          Symb;
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bTime;
CBufferFloat         bGradient;
CBufferFloat         *Result;
CBufferFloat         *Action;
vector<float>        check;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
//--- load models
   float temp;
   CArrayObj *encoder = new CArrayObj();
   CArrayObj *forecast[3];
   CArrayObj *actor = new CArrayObj();
   CArrayObj *director = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(encoder, forecast[0], forecast[1], forecast[2], actor, director, critic))
     {
      delete encoder;
      delete actor;
      delete director;
      delete critic;
      for(uint i = 0; i < forecast.Size(); i++)
         delete forecast[i];
      return INIT_FAILED;
     }
   if(!cEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new State Encoder");
      if(!cEncoder.Create(encoder))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         for(uint i = 0; i < forecast.Size(); i++)
            delete forecast[i];
         return INIT_FAILED;
        }
     }
   for(uint f = 0; f < forecast.Size(); f++)
     {
      if(!caForecast[f].Load(StringFormat("%sFrc%d.nnw", FileName, f), temp, temp, temp, dtStudied, true))
        {
         Print("Create new Forecast model");
         if(!caForecast[f].Create(forecast[f]))
           {
            delete encoder;
            delete actor;
            delete director;
            delete critic;
            for(uint i = 0; i < forecast.Size(); i++)
               delete forecast[i];
            return INIT_FAILED;
           }
        }
     }
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor.Create(actor))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         for(uint i = 0; i < forecast.Size(); i++)
            delete forecast[i];
         return INIT_FAILED;
        }
     }
   if(!cCritic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Critic");
      if(!cCritic.Create(critic))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         for(uint i = 0; i < forecast.Size(); i++)
            delete forecast[i];
         return INIT_FAILED;
        }
     }
   if(!cDirector.Load(FileName + "Drc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Director");
      if(!cDirector.Create(director))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         for(uint i = 0; i < forecast.Size(); i++)
            delete forecast[i];
         return INIT_FAILED;
        }
     }
   delete encoder;
   delete actor;
   delete director;
   delete critic;
   for(uint i = 0; i < forecast.Size(); i++)
      delete forecast[i];
//---
   cEncoder.TrainMode(true);
   cActor.TrainMode(true);
   cCritic.TrainMode(true);
   cDirector.TrainMode(true);
   for(uint i = 0; i < caForecast.Size(); i++)
      caForecast[i].TrainMode(true);
   COpenCL *opencl=cActor.GetOpenCL();
   cDirector.SetOpenCL(opencl);
   cCritic.SetOpenCL(opencl);
   cEncoder.SetOpenCL(opencl);
   for(uint i = 0; i < caForecast.Size(); i++)
      caForecast[i].SetOpenCL(opencl);

//---
   if(!Symb.Name(_Symbol) || !Symb.Refresh())
      return INIT_FAILED;
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
   cEncoder.TrainMode(true);
//---
   cEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
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
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      if(!cEncoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Encoder");
      for(uint f = 0; f < caForecast.Size(); f++)
        {
         if(!caForecast[f].Save(StringFormat("%sFrc%d.nnw", FileName, f), 0, 0, 0, TimeCurrent(), true))
            PrintFormat("Error of save model: %s %d", "Forecast", f);
        }
      if(!cActor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Encoder");
      if(!cCritic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Encoder");
      if(!cDirector.Save(FileName + "Drc.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Encoder");
     }
   delete Result;
   delete Action;
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
   switch(id)
     {
      case 1001:
         Train();
         break;
      case 1007:
         PrintFormat("Wrong event: %d", id);
         ExpertRemove();
         break;
     }
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
//---
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   int count = -1;
   bool calculated = false;
   do
     {
      count++;
      calculated = (RSI.BarsCalculated() >= bars &&
                    CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars &&
                    MACD.BarsCalculated() >= bars
                   );
      Sleep(100);
      count++;
     }
   while(!calculated && count < 100);
   if(!calculated)
     {
      PrintFormat("%s -> %d The training data has not been loaded", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   vector<float> result, target, neg_target;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int posit = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * bars);
      if(!CreateBuffers(posit + end, GetPointer(bState), GetPointer(bTime), Result))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      const vector<float> account = SampleAccount(GetPointer(bState), datetime(bTime[0]));
      if(!bAccount.AssignArray(account))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      //--- Feed Forward
      if(!cEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      for(uint f = 0; f < caForecast.Size(); f++)
         if(!caForecast[f].feedForward(GetPointer(cEncoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d - Forecast %d", __FUNCTION__, __LINE__, f);
            Stop = true;
            break;
           }
      if(!cActor.feedForward(GetPointer(bAccount), 1, false, GetPointer(cEncoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!cCritic.feedForward(GetPointer(cActor), -1, GetPointer(cEncoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!cDirector.feedForward(GetPointer(cActor), -1, GetPointer(cEncoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Study
      for(uint f = 0; f < caForecast.Size(); f++)
         if(!caForecast[f].backProp(Result, (CBufferFloat*)NULL) ||
            !cEncoder.backPropGradient((CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d - Forecast %d", __FUNCTION__, __LINE__, f);
            Stop = true;
            break;
           }
      //---
      cActor.getResults(Action);
      double equity = bAccount[2] * bAccount[0] * EtalonBalance / (1 + bAccount[1]);
      double reward = CheckAction(Action, Result, equity);
      Result.Clear();
      if(!Result.Add(float(reward)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!cCritic.backProp(Result, GetPointer(cEncoder), LatentLayer) ||
         !cActor.backPropGradient(GetPointer(cEncoder), LatentLayer, -1, true))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!Result.Update(0, float(reward > 0)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(!cDirector.backProp(Result, GetPointer(cEncoder), LatentLayer) ||
         !cActor.backPropGradient(GetPointer(cEncoder), LatentLayer, -1, true))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = "";
         for(uint f = 0; f < caForecast.Size(); f++)
            str += StringFormat("%-12s%d %6.2f%% -> Error %15.8f\n", "Forecast", f, percent, caForecast[f].getRecentAverageError());
         str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Critic", percent, cCritic.getRecentAverageError());
         str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Director", percent, cDirector.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   for(uint f = 0; f < caForecast.Size(); f++)
      PrintFormat("%s -> %d -> %-15s%d %10.7f", __FUNCTION__, __LINE__, "Forecast", f, caForecast[f].getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", cCritic.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Director", cDirector.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Buffers.                                |
//+------------------------------------------------------------------+
bool CreateBuffers(const int start_bar, CBufferFloat* state, CBufferFloat *time, CBufferFloat* forecast)
  {
   if(!state || !time || start_bar < 0 ||
      (start_bar + HistoryBars + NForecast) >= int(Rates.Size()))
      return false;
//---
   vector<float> vState = vector<float>::Zeros(HistoryBars * BarDescr);
   vector<float> vForecast = vector<float>::Zeros(NForecast * BarDescr);
   time.Clear();
   time.Reserve(HistoryBars);
   int bar = start_bar + NForecast;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b + bar].open;
      float rsi = (float)RSI.Main(b + bar);
      float cci = (float)CCI.Main(b + bar);
      float atr = (float)ATR.Main(b + bar);
      float macd = (float)MACD.Main(b + bar);
      float sign = (float)MACD.Signal(b + bar);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         return false;
      //---
      int shift = b * BarDescr;
      vState[shift] = (float)(Rates[b + bar].close - open);
      vState[shift + 1] = (float)(Rates[b + bar].high - open);
      vState[shift + 2] = (float)(Rates[b + bar].low - open);
      vState[shift + 3] = (float)(Rates[b + bar].tick_volume / 1000.0f);
      vState[shift + 4] = rsi;
      vState[shift + 5] = cci;
      vState[shift + 6] = atr;
      vState[shift + 7] = macd;
      vState[shift + 8] = sign;
      if(!time.Add(float(Rates[b + bar].time)))
         return false;
     }
//---
   bar--;
   for(int b = 0; b < (int)NForecast; b++)
     {
      float open = (float)Rates[bar - b].open;
      float rsi = (float)RSI.Main(bar - b);
      float cci = (float)CCI.Main(bar - b);
      float atr = (float)ATR.Main(bar - b);
      float macd = (float)MACD.Main(bar - b);
      float sign = (float)MACD.Signal(bar - b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         return false;
      //---
      int shift = (NForecast - b - 1) * BarDescr;
      vForecast[shift] = (float)(Rates[bar - b].close - open);
      vForecast[shift + 1] = (float)(Rates[bar - b].high - open);
      vForecast[shift + 2] = (float)(Rates[bar - b].low - open);
      vForecast[shift + 3] = (float)(Rates[bar - b].tick_volume / 1000.0f);
      vForecast[shift + 4] = rsi;
      vForecast[shift + 5] = cci;
      vForecast[shift + 6] = atr;
      vForecast[shift + 7] = macd;
      vForecast[shift + 8] = sign;
     }
//---
   if(!state.AssignArray(vState))
      return false;
   if(!forecast.AssignArray(vForecast))
      return false;
   if(time.GetIndex() >= 0)
      if(!time.BufferWrite())
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements SampleAccount.                                      |
//+------------------------------------------------------------------+
const vector<float> SampleAccount(CBufferFloat *state, datetime time)
  {
   vector<float> result = vector<float>::Zeros(AccountDescr);
   if(!state)
      return result;
//---
   double marg = 0;
   if(!Symb.RefreshRates() || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return result;
   double buy_lot = 0, sell_lot = 0, profit = 0;
   double deal = MathRand() / (32767 * 0.5) - 1;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
//---
   double balance = (MaxBalance - MinBalance) * MathRand() / 32767.0 + MinBalance;
   double prev_balance = balance;
   double equity = balance;
   double prev_equity = balance;
   double position_discount = 0;
//---
   if(deal > 0)
     {
      double lot = balance / (2.0 * marg) * deal;
      if(lot >= Symb.LotsMin())
         buy_lot = MathMin(int((lot - Symb.LotsMin()) / Symb.LotsStep()) * Symb.LotsStep() + Symb.LotsMin(), 1.0);
     }
   else
      if(deal < 0)
        {
         double lot = MathAbs(balance / (2.0 * marg) * deal);
         if(lot >= Symb.LotsMin())
            sell_lot = MathMin(int((lot - Symb.LotsMin()) / Symb.LotsStep()) * Symb.LotsStep() + Symb.LotsMin(), 1.0);
        }
      else
         prev_balance += (MathRand() / (2.0 * 32767.0) - 0.25) * balance;
//---
   if(sell_lot > 0 || buy_lot > 0)
     {
      int pos_open = int(MathRand() / 32767.0 * (state.Total()/BarDescr - 1));
      for(int i = 0; i <= pos_open; i++)
         profit += state.At(i * BarDescr) / Symb.TickSize() * Symb.TickValue();
      profit *= buy_lot - sell_lot;
      equity += profit;
      prev_equity = equity - state.At(0) / Symb.TickSize() * Symb.TickValue() * (buy_lot - sell_lot);
      position_discount = pos_open * PeriodSeconds(TimeFrame) * multiplyer * MathAbs(profit);
     }
//---
   result[0] = float(balance / EtalonBalance);
   result[1] = float((balance - prev_balance) / prev_balance);
   result[2] = float(equity / prev_balance);
   result[3] = float((equity - prev_equity) / prev_equity);
   result[4] = float(buy_lot);
   result[5] = float(sell_lot);
   result[6] = float(buy_lot * profit / prev_balance);
   result[7] = float(sell_lot * profit / prev_balance);
   result[8] = float(position_discount / prev_balance);
   double x = time / (double)(D'2024.01.01' - D'2023.01.01');
   result[9] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_MN1);
   result[10] = float(MathCos(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_W1);
   result[11] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_D1);
   result[12] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for CheckAction.          |
//+------------------------------------------------------------------+
double CheckAction(CBufferFloat *action, CBufferFloat *forecast, double equity)
  {
   if(!action || !forecast)
      return 0;
//---
   double buy_lot = MathMax(double(action[0] - action[3]), 0);
   double sell_lot = MathMax(double(action[3] - action[0]), 0);
   if(MathMax(buy_lot, sell_lot) < Symb.LotsMin())
      return -10;
   double marg = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return 0;
   if((marg * MathMax(buy_lot, sell_lot)) >= equity)
      return -10;
//---
   double tp = 0, sl = 0, profit = 0;
   double point_cost = 1 / Symb.TickSize() * Symb.TickValue() * (buy_lot - sell_lot);
   if(buy_lot > 0)
     {
      tp = action[1] * MaxTP - Symb.Spread();
      sl = action[2] * MaxSL + Symb.Spread();
      if(int(tp) < Symb.StopsLevel() || int(sl) < Symb.StopsLevel())
         return -10;
      tp *= Symb.Point();
      sl *= Symb.Point();
      for(int i = 0; i < forecast.Total(); i++)
        {
         int shift = i * BarDescr;
         if(sl <= (-forecast[shift + 2]))
           {
            profit -= sl * point_cost * MathPow(DiscFactor, i);
            break;
           }
         if(tp <= forecast[shift + 1])
           {
            profit += tp * point_cost * MathPow(DiscFactor, i);
            break;
           }
         profit += forecast[shift] * point_cost * MathPow(DiscFactor, i);
         if(-profit >= equity)
            break;
         tp-=forecast[shift];
         sl+=forecast[shift];
        }
     }
//---
   if(sell_lot > 0)
     {
      tp = action[4] * MaxTP - Symb.Spread();
      sl = action[5] * MaxSL + Symb.Spread();
      if(int(tp) < Symb.StopsLevel() || int(sl) < Symb.StopsLevel())
         return -10;
      tp *= Symb.Point();
      sl *= Symb.Point();
      for(int i = 0; i < forecast.Total(); i++)
        {
         int shift = i * BarDescr;
         if(sl <= forecast[shift + 1])
           {
            profit += sl * point_cost * MathPow(DiscFactor, i);
            break;
           }
         if(tp <= (-forecast[shift + 2]))
           {
            profit -= tp * point_cost * MathPow(DiscFactor, i);
            break;
           }
         profit += forecast[shift] * point_cost * MathPow(DiscFactor, i);
         if(-profit >= equity)
            break;
         sl-=forecast[shift];
         tp+=forecast[shift];
        }
     }
//---
   return profit;
  }
//+------------------------------------------------------------------+

