//+------------------------------------------------------------------+
//| GE-lerning.mq5 |
//| Copyright 2023, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\RL\FQF.mqh"
#include "..\RL\ICM.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
#include "Cell.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input int                  Batch =  100;
input float                DiscountFactor =   0.3f;
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
input bool                 TrainMode = true;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CSymbolInfo          Symb;
MqlRates             Rates[];
CFQF                 StudyNet;
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
CReplayBuffer        cReplay;

//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
MqlDateTime          sTime;
//---
CBufferFloat         State1;
CBufferFloat         *pstate1;
CBufferFloat         *pstate2;
CBufferFloat         *Rewards;
float                min_loss = FLT_MAX;
CTrade               Trade;
int                  Action;
float                Equity = -1;
float                Balance = -1;
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
   datetime time;
   if(!StudyNet.Load(FileName + ".nnw", time, true))
      return INIT_FAILED;
   if(!StudyNet.TrainMode(TrainMode))
      return INIT_FAILED;
//---
   CBufferFloat* temp;
   if(!StudyNet.GetLayerOutput(0, temp))
      return INIT_FAILED;
   if(HistoryBars != (temp.Total() - 9) / 12)
      return INIT_FAILED;
   delete temp;
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
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   StudyNet.Save(FileName + ".nnw", true);
   if(!!Rewards)
      delete Rewards;
   if(!!pstate1)
      delete pstate1;
   if(!!pstate2)
      delete pstate2;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar())
      return;
//---
   float current_eq = (float)AccountInfoDouble(ACCOUNT_EQUITY);
   float current_bl = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   if(Equity >= 0 && State1.Total() == (HistoryBars * 12 + 9))
      cReplay.AddState(GetPointer(State1), Action, (double)(current_eq + current_bl - Equity - Balance) / 2.0);
   Equity = current_eq;
   Balance = current_bl;
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
//---
   State1.Clear();
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
      if(!State1.Add((float)Rates[b].close - open) || !State1.Add((float)Rates[b].high - open) || !State1.Add((float)Rates[b].low - open) || !State1.Add((float)Rates[b].tick_volume / 1000.0f) ||
         !State1.Add(sTime.hour) || !State1.Add(sTime.day_of_week) || !State1.Add(sTime.mon) ||
         !State1.Add(rsi) || !State1.Add(cci) || !State1.Add(atr) || !State1.Add(macd) || !State1.Add(sign))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
     }
//---
   if(!State1.Add((float)AccountInfoDouble(ACCOUNT_BALANCE)) || !State1.Add((float)AccountInfoDouble(ACCOUNT_EQUITY)) ||
      !State1.Add((float)AccountInfoDouble(ACCOUNT_MARGIN_FREE)) || !State1.Add((float)AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)) ||
      !State1.Add((float)AccountInfoDouble(ACCOUNT_PROFIT)))
      return;
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   int total = PositionsTotal();
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
            return;
        }
     }
   if(!State1.Add((float)buy_value) || !State1.Add((float)sell_value) || !State1.Add((float)buy_profit) || !State1.Add((float)sell_profit))
      return;
   if(!StudyNet.feedForward(GetPointer(State1), 12, true))
      return;
   Action = StudyNet.getAction();
   switch(Action)
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
   MqlDateTime time;
   TimeCurrent(time);
   if(time.hour == 0)
     {
      int repl_action;
      double repl_reward;
      for(int i = 0; i < 10; i++)
        {
         if(cReplay.GetRendomState(pstate1, repl_action, repl_reward, pstate2))
            return;
         if(!StudyNet.feedForward(pstate1, 12, true))
            return;
         StudyNet.getResults(Rewards);
         if(!Rewards.Update(repl_action, (float)repl_reward))
            return;
         if(!StudyNet.backProp(GetPointer(Rewards), DiscountFactor, pstate2, 12, true))
            return;
        }
     }
//---
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
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *Description, CArrayObj *Forward)
  {
//---
   if(!Description)
     {
      Description = new CArrayObj();
      if(!Description)
         return false;
     }
//---
   if(!Forward)
     {
      Forward = new CArrayObj();
      if(!Forward)
         return false;
     }
//--- Model
   Description.Clear();
   CLayerDescription *descr;
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (int)(HistoryBars * 12 + 9);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 100;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = prev_count / 3 - 1 ;
   descr.window = 6;
   descr.step = 3;
   descr.window_out = 6;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 49;
   descr.window = 4;
   descr.step = 2;
   descr.window_out = 8;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMLMHSparseAttentionOCL;
   descr.count = 20;
   descr.window = 5;
   descr.step = 4;
   descr.window_out = 8;
   descr.layers = 2;
   descr.probability = 0.3f;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFQF;
   descr.count = 4;
   descr.window_out = 32;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Forward
   Forward.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!Forward.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 1000;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!Forward.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 400;
   descr.window = 200;
   descr.step = 5;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!Forward.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+

