//+------------------------------------------------------------------+
//| StudyActor.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Defines                                                          |
//+------------------------------------------------------------------+
#define                    HistoryBars  20            //Depth of history
#define                    BarDescr     12            //Elements for 1 bar description
#define                    AccountDescr 8             //Account description
#define                    FileName     "DDPG"
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
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         PrevAccount;
CBufferFloat         Gradient;
CBufferFloat         *Result;
vector<float>        check;
vector<float>        ActorResult;
float                PrevTarget = 0;
double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 FirstBar = true;
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
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
         return INIT_FAILED;
     }
//---
   COpenCLMy *opencl = Actor.GetOpenCL();
//---
   Actor.getResults(Result);
   if(Result.Total() != 6)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", 6, Result.Total());
      return INIT_FAILED;
     }
   ActorResult = vector<float>::Zeros(6);
//---
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   Gradient.BufferInit(AccountDescr, 0);
   Gradient.BufferCreate(opencl);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
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
   MqlDateTime sTime;
   float atr = 0;
   State.Clear();
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b].open;
      TimeToStruct(Rates[b].time, sTime);
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      State.Add((float)Rates[b].close - open);
      State.Add((float)Rates[b].high - open);
      State.Add((float)Rates[b].low - open);
      State.Add((float)Rates[b].tick_volume / 1000.0f);
      State.Add((float)sTime.hour);
      State.Add((float)sTime.day_of_week);
      State.Add((float)sTime.mon);
      State.Add(rsi);
      State.Add(cci);
      State.Add(atr);
      State.Add(macd);
      State.Add(sign);
     }
//---
   vector<float> account = vector<float>::Zeros(AccountDescr - 1);
   account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
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
   account[2] = (float)buy_value;
   account[3] = (float)sell_value;
   account[4] = (float)buy_profit;
   account[5] = (float)sell_profit;
   account[6] = (float)position_discount;
//---
   Account.Clear();
   Account.Add((float)((account[0] - PrevBalance) / PrevBalance));
   Account.Add((float)(account[1] / PrevBalance));
   Account.Add((float)((account[1] - PrevEquity) / PrevEquity));
   Account.Add(account[2]);
   Account.Add(account[3]);
   Account.Add((float)(account[4] / PrevBalance));
   Account.Add((float)(account[5] / PrevBalance));
   Account.Add((float)(account[6] / PrevBalance));
   PrevBalance=account[0];
   PrevEquity=account[1];
//---
   if(Account.GetIndex() >= 0)
      if(!Account.BufferWrite())
         return;
//---
   if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
      return;
   Actor.getResults(ActorResult);
//---
   double min_lot = Symb.LotsMin();
   double buy_lot = MathRound((double)ActorResult[0] / min_lot) * min_lot;
   double sell_lot = MathRound((double)ActorResult[3] / min_lot) * min_lot;
   double buy_tp = NormalizeDouble(Symb.Ask() + ActorResult[1], Symb.Digits());
   double buy_sl = NormalizeDouble(Symb.Ask() - ActorResult[2], Symb.Digits());
   double sell_tp = NormalizeDouble(Symb.Bid() - ActorResult[4], Symb.Digits());
   double sell_sl = NormalizeDouble(Symb.Bid() + ActorResult[5], Symb.Digits());
//---
   if(ActorResult[0] > 0 && ActorResult[1] > 0 && ActorResult[2] > 0 && buy_sl > 0)
      Trade.Buy(buy_lot, Symb.Name(), Symb.Ask(), buy_sl, buy_tp);
   if(ActorResult[3] > 0 && ActorResult[4] > 0 && ActorResult[5] > 0 && sell_tp > 0)
      Trade.Sell(sell_lot, Symb.Name(), Symb.Bid(), sell_sl, sell_tp);
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *actor, CArrayObj *critic)
  {
//---
   CLayerDescription *descr;
//---
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         return false;
     }
//--- Actor
   actor.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (HistoryBars * BarDescr);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count - 1;
   descr.window = 2;
   descr.step = 1;
   descr.window_out = 8;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count;
   descr.window = 8;
   descr.step = 8;
   descr.window_out = 8;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 128;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = 256;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 6;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = 128;
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = 128;
   descr.window = prev_count;
   descr.step = 6;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 128;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 128;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 1;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
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

