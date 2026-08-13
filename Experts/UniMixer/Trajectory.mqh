//+------------------------------------------------------------------+
//| Trajectory.mqh |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
input group                "---- Indicators ----"
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
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
int            iLatentLayer = -1;
//---
#define        HistoryBars             5
#define        BarDescr                9             //Elements for 1 bar description
#define        AccountDescr            13            //Account description
#define        NActions                6             //Number of possible Actions
#define        NRewards                1             //Number of rewards
#define        NForecast               12            //Number of forecast
#define        EtalonBalance           1e4
#define        BatchSize               1e+5
#define        EmbeddingSize           18
#define        DiscFactor              0.9f
#define        FileName                "UniMixer"
#define        LatentCount             64
#define        LatentLayer             iLatentLayer
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        ActorUpdate             5
#define        TargetUpdate            24*5
#define        tau                     0.9f
#define        NHeads                  4
#define        NExperts                5
#define        NScenarios              21
#define        TopK                    3
#define        StackSize               24*5
#define        Blocks                  24
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CSymbolInfo          Symb;
CTrade               Trade;
MqlRates             Rates[];
//---
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct SState
  {
   float             state[HistoryBars * BarDescr];
   float             account[AccountDescr - 4];
   float             action[NActions];
   float             rewards[NRewards];
   //---
                     SState(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
   //--- overloading
   void              operator=(const SState &obj)
     {
      ArrayCopy(state, obj.state);
      ArrayCopy(account, obj.account);
      ArrayCopy(action, obj.action);
      ArrayCopy(rewards, obj.rewards);
     }
  };
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S St at e implementation.                                      |
//+------------------------------------------------------------------+
SState::SState(void)
  {
   ArrayInitialize(state, 0);
   ArrayInitialize(account, 0);
   ArrayInitialize(action, 0);
   ArrayInitialize(rewards, 0);
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S av e implementation.                                         |
//+------------------------------------------------------------------+
bool SState::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      return false;
//---
   int total = ArraySize(state);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      return false;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, state[i]) < sizeof(float))
         return false;
//---
   total = ArraySize(account);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      return false;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, account[i]) < sizeof(float))
         return false;
//---
   total = ArraySize(action);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      return false;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, action[i]) < sizeof(float))
         return false;
   total = ArraySize(rewards);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      return false;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, rewards[i]) < sizeof(float))
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | L oa d implementation.                                         |
//+------------------------------------------------------------------+
bool SState::Load(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      return false;
   if(FileIsEnding(file_handle))
      return false;
//---
   int total = FileReadInteger(file_handle);
   if(total != ArraySize(state))
      return false;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         return false;
      state[i] = FileReadFloat(file_handle);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(account))
      return false;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         return false;
      account[i] = FileReadFloat(file_handle);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(action))
      return false;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         return false;
      action[i] = MathMin(MathMax(FileReadFloat(file_handle), 0), 1);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(rewards))
      return false;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         return false;
      rewards[i] = FileReadFloat(file_handle);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *&actor,
                        CArrayObj *&critic
                       )
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
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (HistoryBars * BarDescr) + AccountDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer = 0;
//--- layer 1
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronUniMixerBlock;
     {
      if(ArrayResize(descr.windows, BarDescr + 3) < (BarDescr + 3))
         ReturnFalse;
      ArrayFill(descr.windows, 0, BarDescr, HistoryBars);
      descr.windows[BarDescr] = 4;              // Account
      descr.windows[BarDescr + 1] = 5;          // Position
      descr.windows[BarDescr + 2] = 4;          // Time
     }
     {
      uint temp[] = {BarDescr, NScenarios};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.step = Blocks;
   descr.count = StackSize;
   descr.layers = 3;
   descr.window_out = EmbeddingSize;
   descr.variables = NExperts;                     //Candidates
   descr.probability = TopK;
   descr.batch = BatchSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer++;
   uint count = prev_count = descr.step;        //Blocks
   uint prev_out = descr.window_out;
   uint window = prev_out;
//--- layer 2
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = prev_count;
   descr.window = prev_out;
   descr.step = prev_out;
   descr.window_out = (LatentCount + prev_count - 1) / prev_count;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 3
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 4
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 5
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronVAEOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 6
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 3;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 1
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronMHCrossFAT;
     {
      uint temp[] = {3,                      // Inputs window
                     EmbeddingSize,          // Key Dimension
                     window,                 // Cross window
                     EmbeddingSize           // Embedding size
                    };
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
     {
      uint temp[] = {NActions / 3,           // Query units
                     count                   // Cross units
                    };
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         return false;
     }
   descr.step = NHeads;                      // Heads
   descr.batch = 1e4;
   descr.layers = NExperts;                  // Candidates
   descr.variables = TopK;                   // Top-K
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 2
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = EmbeddingSize;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 4
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = NActions / 3;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 5
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 6
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = NRewards;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//---
   return true;
  }
#ifndef Study
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | I sN ew Ba r implementation.                                   |
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
//| | C lo se By Di re ct io n implementation.                       |
//+------------------------------------------------------------------+
bool CloseByDirection(ENUM_POSITION_TYPE type)
  {
   int total = PositionsTotal();
   bool result = true;
   for(int i = total - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
         continue;
      result = (Trade.PositionClose(PositionGetInteger(POSITION_TICKET)) && result);
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | T ra il Po si ti on implementation.                            |
//+------------------------------------------------------------------+
bool TrailPosition(ENUM_POSITION_TYPE type, double sl, double tp)
  {
   int total = PositionsTotal();
   bool result = true;
   datetime time = TimeCurrent() - 5 * PeriodSeconds(TimeFrame);
//---
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
         continue;
      if(PositionGetInteger(POSITION_TIME_UPDATE) > time)
         continue;
      bool modify = false;
      double psl = PositionGetDouble(POSITION_SL);
      double ptp = PositionGetDouble(POSITION_TP);
      switch(type)
        {
         case POSITION_TYPE_BUY:
            if((sl - psl) >= Symb.Point())
              {
               psl = sl;
               modify = true;
              }
            if(MathAbs(tp - ptp) >= Symb.Point())
              {
               ptp = tp;
               modify = true;
              }
            break;
         case POSITION_TYPE_SELL:
            if((psl - sl) >= Symb.Point())
              {
               psl = sl;
               modify = true;
              }
            if(MathAbs(tp - ptp) >= Symb.Point())
              {
               ptp = tp;
               modify = true;
              }
            break;
        }
      if(modify)
         result = (Trade.PositionModify(PositionGetInteger(POSITION_TICKET), psl, ptp) && result);
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | C lo se Pa rt ia l implementation.                             |
//+------------------------------------------------------------------+
bool ClosePartial(ENUM_POSITION_TYPE type, double value)
  {
   if(value <= 0)
      return true;
//---
   for(int i = 0; (i < PositionsTotal() && value > 0); i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
         continue;
      double pvalue = PositionGetDouble(POSITION_VOLUME);
      if(pvalue <= value)
        {
         if(Trade.PositionClose(PositionGetInteger(POSITION_TICKET)))
           {
            value -= pvalue;
            i--;
           }
        }
      else
        {
         if(Trade.PositionClosePartial(PositionGetInteger(POSITION_TICKET), value))
            value = 0;
        }
     }
//---
   return (value <= 0);
  }
#endif
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
class CDeal       :  public CObject
  {
public:
   datetime             OpenTime;
   datetime             CloseTime;
   ENUM_POSITION_TYPE   Type;
   double               Volume;
   double               OpenPrice;
   double               StopLos;
   double               TakeProfit;
   double               point;
   //---
                     CDeal(void);
                    ~CDeal(void) {};
   //---
   vector<float>        Action(datetime current, double ask, double bid, int period_seconds);
  };
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void CDeal::CDeal(void) :  OpenTime(0),
   CloseTime(0),
   Type(POSITION_TYPE_BUY),
   Volume(0),
   OpenPrice(0),
   StopLos(0),
   TakeProfit(0),
//| Implements point                                             |
   point(1e-5)
  {
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> CDeal::Action(datetime current, double ask, double bid, int period_seconds)
  {
   vector<float> result = vector<float>::Zeros(NActions);
   if((OpenTime - period_seconds) > current || CloseTime <= current)
      return result;
//---
   switch(Type)
     {
      case POSITION_TYPE_BUY:
         result[0] = float(Volume);
         if(TakeProfit > 0)
            result[1] = float((TakeProfit - ask) / (MaxTP * point));
         if(StopLos > 0)
            result[2] = float((ask - StopLos) / (MaxSL * point));
         break;
      case POSITION_TYPE_SELL:
         result[3] = float(Volume);
         if(TakeProfit > 0)
            result[4] = float((bid - TakeProfit) / (MaxTP * point));
         if(StopLos > 0)
            result[5] = float((StopLos - bid) / (MaxSL * point));
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
class CDeals
  {
protected:
   CArrayObj         Deals;
public:
                     CDeals(void) { Deals.Clear(); }
                    ~CDeals(void) { Deals.Clear(); }
   //---
   bool              LoadDeals(string file_name, string symbol, double point);
   vector<float>     Action(datetime current, double ask, double bid, int period_seconds);
  };
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | L oa dD ea ls implementation.                                  |
//+------------------------------------------------------------------+
bool CDeals::LoadDeals(string file_name, string symbol, double point)
  {
   if(file_name == NULL || !FileIsExist(file_name, FILE_COMMON))
     {
      PrintFormat("File %s not exist", file_name);
      return false;
     }
   if(symbol == NULL)
     {
      symbol = _Symbol;
      point = _Point;
     }
//---
   ResetLastError();
   int handle = FileOpen(file_name, FILE_READ | FILE_ANSI | FILE_CSV | FILE_COMMON, short(';'), CP_ACP);
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("Error of open file %s: %d", file_name, GetLastError());
      return false;
     }
   FileSeek(handle, 0, SEEK_SET);
   while(!FileIsEnding(handle))
     {
      string s = FileReadString(handle);
      datetime open_time = StringToTime(s);
      string type = FileReadString(handle);
      double volume = StringToDouble(FileReadString(handle));
      string deal_symbol = FileReadString(handle);
      double open_price = StringToDouble(FileReadString(handle));
      volume = MathMin(volume, StringToDouble(FileReadString(handle)));
      datetime close_time = StringToTime(FileReadString(handle));
      double close_price = StringToDouble(FileReadString(handle));
      s = FileReadString(handle);
      s = FileReadString(handle);
      s = FileReadString(handle);
      if(StringFind(deal_symbol, symbol, 0) < 0)
         continue;
      //---
      ResetLastError();
      CDeal *deal = new CDeal();
      if(!deal)
        {
         PrintFormat("Error of create new deal object: %d", GetLastError());
         return false;
        }
      deal.OpenTime = open_time;
      deal.CloseTime = close_time;
      deal.OpenPrice = open_price;
      deal.Volume = volume;
      deal.point = point;
      if(type == "Sell")
        {
         deal.Type = POSITION_TYPE_SELL;
         if(close_price < open_price)
           {
            deal.TakeProfit = close_price;
            deal.StopLos = 0;
           }
         else
           {
            deal.TakeProfit = 0;
            deal.StopLos = close_price;
           }
        }
      else
        {
         deal.Type = POSITION_TYPE_BUY;
         if(close_price > open_price)
           {
            deal.TakeProfit = close_price;
            deal.StopLos = 0;
           }
         else
           {
            deal.TakeProfit = 0;
            deal.StopLos = close_price;
           }
        }
      //---
      ResetLastError();
      if(!Deals.Add(deal))
        {
         PrintFormat("Error of add new deal: %d", GetLastError());
         return false;
        }
     }
//---
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> CDeals::Action(datetime current, double ask, double bid, int period_seconds)
  {
   vector<float> result = vector<float>::Zeros(NActions);
   for(int i = 0; i < Deals.Total(); i++)
     {
      CDeal *deal = Deals.At(i);
      if(!deal)
         continue;
      vector<float> action = deal.Action(current, ask, bid, period_seconds);
      result[0] += action[0];
      result[3] += action[3];
      result[1] = MathMax(result[1], action[1]);
      result[2] = MathMax(result[2], action[2]);
      result[4] = MathMax(result[4], action[4]);
      result[5] = MathMax(result[5], action[5]);
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | C re at eB uf fe rs implementation.                            |
//+------------------------------------------------------------------+
bool CreateBuffers(const int start_bar, CBufferFloat* state, CBufferFloat *time, CBufferFloat* forecast)
  {
   int total_bars = (start_bar + HistoryBars + (!!forecast ? NForecast : 0));
   if(!state || !time || start_bar < 0 ||
      total_bars > int(Rates.Size()))
      return false;
//---
   matrix<float> mState = matrix<float>::Zeros(BarDescr, HistoryBars);
   vector<float> vForecast = vector<float>::Zeros(NForecast * BarDescr);
   time.Clear();
   time.Reserve(HistoryBars);
   int bar = start_bar + (!!forecast ? NForecast : 0);
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
      mState[0, b] = (float)(Rates[b + bar].close - open);
      mState[1, b] = (float)(Rates[b + bar].high - open);
      mState[2, b] = (float)(Rates[b + bar].low - open);
      mState[3, b] = (float)(Rates[b + bar].tick_volume / 1000.0f);
      mState[4, b] = rsi;
      mState[5, b] = cci;
      mState[6, b] = atr;
      mState[7, b] = macd;
      mState[8, b] = sign;
      if(!time.Add(float(Rates[b + bar].time)))
         return false;
     }
   if(!state.AssignArray(mState))
      return false;
   if(time.GetIndex() >= 0)
      if(!time.BufferWrite())
         return false;
   if(!forecast)
      return true;
//---
   for(int b = 1; b <= (int)NForecast; b++)
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
      int shift = (NForecast - b) * BarDescr;
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
   if(!forecast.AssignArray(vForecast))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
const vector<float> SampleAccount(CBufferFloat *state, datetime time, double max_balance, double min_balance = 0)
  {
   vector<float> result = vector<float>::Zeros(AccountDescr);
   if(!state)
      return result;
//---
   double marg = 0;
   if(!Symb.RefreshRates() || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return result;
   double buy_lot = 0, sell_lot = 0, profit = 0;
   double deal = 0;//MathRand() / (32767 * 0.5) - 1;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
//---
   double balance = (max_balance - min_balance) * MathRand() / 32767.0 + min_balance;
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
      int pos_open = int(MathRand() / 32767.0 * (state.Total() / BarDescr - 1));
      for(int i = 0; i <= pos_open; i++)
        {
         profit += state.At(i * BarDescr) / Symb.TickSize() * Symb.TickValue();
         if(((buy_lot > 0 && profit < 0) ||
             (sell_lot > 0 && profit > 0))
            // Function implementation for MathAbs.
            && MathAbs(profit * (buy_lot - sell_lot)) > balance / 2)
           {
            pos_open = i;
            break;
           }
        }
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
//| | C he ck Ac ti on implementation.                               |
//+------------------------------------------------------------------+
double CheckAction(CBufferFloat *action, double balance, uint start_position)
  {
   if(!action || start_position >= Rates.Size())
      return 0;
//---
   double buy_lot = MathMax(double(action[0] - action[3]), 0);
   double sell_lot = MathMax(double(action[3] - action[0]), 0);
   double marg = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return 0;
   double point_cost = Symb.TickValue() / Symb.TickSize();
   if(MathMax(buy_lot, sell_lot) < Symb.LotsMin())
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * balance / (2 * marg);
      return loss;
     }
   if((marg * MathMax(buy_lot, sell_lot)) >= balance)
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * MathMax(buy_lot, sell_lot);
      return loss;
     }
   point_cost *= MathAbs(buy_lot - sell_lot);
//---
   double tp = 0, sl = 0, profit = 0, reward = 0;
   int stops = MathMax(Symb.StopsLevel(), 10);
   int spread = Symb.Spread();
   if(buy_lot > 0)
     {
      tp = action[1] * MaxTP;
      sl = action[2] * MaxSL;
      if(int(tp) < stops || int(sl) < (stops + spread))
        {
         double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                                Rates[start_position].open - Rates[start_position].low) *
                       point_cost * buy_lot;
         return loss;
        }
      tp = (tp + spread) * Symb.Point() + Rates[start_position].open;
      sl = Rates[start_position].open - (sl + spread) * Symb.Point();
      reward = profit = -spread * Symb.Point() * point_cost;
      for(uint i = start_position; i > 0; i--)
        {
         if(sl >= Rates[i].low)
           {
            double p = (Rates[i].open - sl) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         if(tp <= Rates[i].high)
           {
            double p = (tp - Rates[i].open) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit += p;
         reward += p * MathPow(DiscFactor, float(i - start_position));
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
//---
   if(sell_lot > 0)
     {
      tp = action[4] * MaxTP;
      sl = action[5] * MaxSL;
      if(int(tp) < stops || int(sl) < (stops + spread))
        {
         double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                                Rates[start_position].open - Rates[start_position].low) *
                       point_cost * sell_lot;
         return loss;
        }
      tp = Rates[start_position].open - (tp + spread) * Symb.Point();
      sl = Rates[start_position].open + (sl - spread) * Symb.Point();
      for(uint i = start_position; i > 0; i--)
        {
         if(sl <= Rates[i].high)
           {
            double p = (sl - Rates[i].open) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         if(tp >= Rates[i].low)
           {
            double p = (Rates[i].open - tp) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit -= p;
         reward -= p  * MathPow(DiscFactor, float(i - start_position));
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
//---
   return reward;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> OraculAction(const vector<float> &account, CBufferFloat *forecat)
  {
//--- Look for target
   vector<float> result = vector<float>::Zeros(NActions);
   matrix<float> fstate = matrix<float>::Zeros(NForecast, BarDescr);
   if(!forecat.GetData(fstate))
      return result;
//---
   vector<float> target = fstate.Col(0).CumSum();
   if(account[4] > account[5])
     {
      float tp = 0;
      float sl = 0;
      float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * Point());
      int pos = 0;
      for(int j = 0; j < NForecast; j++)
        {
         tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
         pos = j;
         if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
            break;
         sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
        }
      if(pos > 0 && tp > 0)
        {
         sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01));
         tp = float(MathMax(MathMin(tp / (MaxTP * Point()), 1), 0.01));
         result[0] = MathMax(result[0] - result[3], 0.011f);
         result[5] = result[1] = tp;
         result[4] = result[2] = sl;
         result[3] = 0;
        }
     }
   else
     {
      if(account[4] < account[5])
        {
         float tp = 0;
         float sl = 0;
         float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * Point());
         int pos = 0;
         for(int j = 0; j < NForecast; j++)
           {
            tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
            pos = j;
            if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
               break;
            sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
           }
         if(pos > 0 && tp < 0)
           {
            sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01));
            tp = float(MathMax(MathMin(-tp / (MaxTP * Point()), 1), 0.01));
            result[3] = MathMax(result[3] - result[0], 0.011f);
            result[2] = result[4] = tp;
            result[1] = result[5] = sl;
            result[0] = 0;
           }
        }
      else
        {
         ulong argmin = target.ArgMin();
         ulong argmax = target.ArgMax();
         float max_sl = float(MaxSL * Point());
         double equity = account[2] * account[0] * EtalonBalance / (1 + account[1]);
         while(argmax > 0 && argmin > 0)
           {
            if(argmax < argmin && target[argmax] / 2 > MathAbs(target[argmin]) && MathAbs(target[argmin]) < max_sl)
               break;
            if(argmax > argmin && target[argmax] < MathAbs(target[argmin] / 2) && target[argmax] < max_sl)
               break;
            target.Resize(MathMin(argmax, argmin));
            argmin = target.ArgMin();
            argmax = target.ArgMax();
           }
         if(argmin == 0 || (argmax < argmin && argmax > 0))
           {
            float tp = 0;
            float sl = 0;
            float cur_sl = - float(MaxSL * Point());
            ulong pos = 0;
            for(ulong j = 0; j < argmax; j++)
              {
               tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
               pos = j;
               if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
                  break;
               sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
              }
            if(pos > 0 && tp > 0)
              {
               sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
               tp = (float)MathMin(tp / (MaxTP * Point()), 1);
               result[0] = float(MathMax(equity / 100 * 0.01, 0.011));
               result[5] = result[1] = tp;
               result[4] = result[2] = sl;
               result[3] = 0;
              }
           }
         else
           {
            if(argmax == 0 || argmax > argmin)
              {
               float tp = 0;
               float sl = 0;
               float cur_sl = float(MaxSL * Point());
               ulong pos = 0;
               for(ulong j = 0; j < argmin; j++)
                 {
                  tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
                  pos = j;
                  if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
                     break;
                  sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
                 }
               if(pos > 0 && tp < 0)
                 {
                  sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                  tp = (float)MathMin(-tp / (MaxTP * Point()), 1);
                  result[3] = float(MathMax(equity / 100 * 0.01, 0.011));
                  result[2] = result[4] = tp;
                  result[1] = result[5] = sl;
                  result[0] = 0;
                 }
              }
           }
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+

