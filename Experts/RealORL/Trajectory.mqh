//+------------------------------------------------------------------+
//| Trajectory.mqh |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Rewards structure                                                |
//| 0     -  Delta Balance                                         |
//| 1     -  Delta Equity ( "-" Drawdown / "+" Profit)             |
//| 2     -  Penalty for no open positions                         |
//| 3     -  NNM                                                   |
//| 4     -  Latent NNM                                            |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define        HistoryBars             20            //Depth of history
#define        BarDescr                9             //Elements for 1 bar description
#define        AccountDescr            12            //Account description
#define        NActions                6             //Number of possible Actions
#define        NRewards                5             //Number of rewards
#define        EmbeddingSize           16
#define        Buffer_Size             6500
#define        DiscFactor              0.99f
#define        FileName                "RealORL"
#define        SignalFile(agent)       StringFormat("Signals\\Signal%d.csv",agent)
#define        LatentLayer             9
#define        LatentCount             1024
#define        SamplLatentStates       32
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        MaxReplayBuffer         500
#define        StartTargetIteration    50000
#define        fCAGrad_C               0.5f
#define        iCAGrad_Iters           15
#define        Quant                   5e-4
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct STarget
  {
   vector<float>     rewards;
   vector<float>     actions;
   //---
//| Implements STarget                                           |
                     STarget(void)
     {
      rewards = vector<float>::Zeros(NRewards);
      actions = vector<float>::Zeros(NActions);
     }
  };
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
      action[i] = FileReadFloat(file_handle);
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
struct STrajectory
  {
   SState            States[Buffer_Size];
   int               Total;
   float             DiscountFactor;
   bool              CumCounted;
   //---
                     STrajectory(void);
   //---
   bool              Add(SState &state);
   void              CumRevards(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
  };
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   DiscountFactor(DiscFactor),
//| Implements CumCounted                                        |
   CumCounted(false)
  {
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S av e implementation.                                         |
//+------------------------------------------------------------------+
bool STrajectory::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      return false;
//---
   if(!CumCounted)
      CumRevards();
   if(FileWriteInteger(file_handle, Total) < sizeof(int))
      return false;
   if(FileWriteFloat(file_handle, DiscountFactor) < sizeof(float))
      return false;
   for(int i = 0; i < Total; i++)
      if(!States[i].Save(file_handle))
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | L oa d implementation.                                         |
//+------------------------------------------------------------------+
bool STrajectory::Load(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      return false;
//---
   Total = FileReadInteger(file_handle);
   if(FileIsEnding(file_handle) || Total >= ArraySize(States))
      return false;
   DiscountFactor = FileReadFloat(file_handle);
   CumCounted = true;
//---
   for(int i = 0; i < Total; i++)
      if(!States[i].Load(file_handle))
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | C um Re va rd s implementation.                                |
//+------------------------------------------------------------------+
void STrajectory::CumRevards(void)
  {
   if(CumCounted)
      return;
//---
   for(int i = Total - 2; i >= 0; i--)
      for(int r = 0; r < NRewards; r++)
         States[i].rewards[r] += States[i + 1].rewards[r] * DiscountFactor;
   CumCounted = true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | A dd implementation.                                           |
//+------------------------------------------------------------------+
bool STrajectory::Add(SState &state)
  {
   if(Total + 1 >= ArraySize(States))
      return false;
   States[Total] = state;
   Total++;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S av eT ot al Ba se implementation.                            |
//+------------------------------------------------------------------+
bool SaveTotalBase(void)
  {
   int total = ArraySize(Buffer);
   if(total < 0)
      return true;
   int handle = FileOpen(FileName + ".bd", FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle < 0)
      return false;
   int start = MathMax(total - MaxReplayBuffer, 0);
   if(FileWriteInteger(handle, total - start) < INT_VALUE)
     {
      FileClose(handle);
      return false;
     }
   for(int i = start; i < total; i++)
      if(!Buffer[i].Save(handle))
        {
         FileClose(handle);
         return false;
        }
   FileFlush(handle);
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | L oa dT ot al Ba se implementation.                            |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | C re at eD es cr ip ti on s implementation.                    |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *actor, CArrayObj *critic, CArrayObj *convolution)
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
   if(!convolution)
     {
      convolution = new CArrayObj();
      if(!convolution)
         return false;
     }
//--- Actor
   actor.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (HistoryBars * BarDescr);
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
   prev_count = descr.count = HistoryBars;
   descr.window = BarDescr;
   descr.step = BarDescr;
   uint prev_wout = descr.window_out = BarDescr / 2;
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
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count;
   descr.step = prev_wout;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   prev_wout = descr.window_out = 8;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count;
   descr.step = prev_wout;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
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
   prev_count = descr.count = LatentCount;
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
   descr.type = defNeuronConcatenate;
   descr.count = 2 * LatentCount;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = LatentCount;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 10
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 11
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 12
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 13
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
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
   prev_count = descr.count = LatentCount;
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
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = NActions;
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
   descr.count = LatentCount;
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
   descr.count = LatentCount;
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
   descr.count = NRewards;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Convolution
   convolution.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (HistoryBars * BarDescr) + AccountDescr + NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = HistoryBars * BarDescr;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = HistoryBars;
   descr.step = BarDescr;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = HistoryBars;
   descr.window = BarDescr;
   descr.step = BarDescr;
   prev_wout = descr.window_out = BarDescr / 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   prev_wout = descr.window_out = prev_wout / 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   prev_wout = descr.window_out = 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count * prev_wout;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = EmbeddingSize;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!convolution.Add(descr))
     {
      delete descr;
      return false;
     }
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
//---
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
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
vector<float> ForecastAccount(float &prev_account[], vector<float> &actions, double prof_1l, float time_label)
  {
   vector<float> account;
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double stops = MathMax(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL), 1) * Point();
   double margin_buy, margin_sell;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 1.0, SymbolInfoDouble(_Symbol, SYMBOL_ASK), margin_buy) ||
      !OrderCalcMargin(ORDER_TYPE_SELL, _Symbol, 1.0, SymbolInfoDouble(_Symbol, SYMBOL_BID), margin_sell))
      return vector<float>::Zeros(prev_account.Size());
//---
   account.Assign(prev_account);
//---
   if(actions[0] >= actions[3])
     {
      actions[0] -= actions[3];
      actions[3] = 0;
      if(actions[0]*margin_buy >= MathMin(account[0], account[1]))
         actions[0] = 0;
     }
   else
     {
      actions[3] -= actions[0];
      actions[0] = 0;
      if(actions[3]*margin_sell >= MathMin(account[0], account[1]))
         actions[3] = 0;
     }
//--- buy control
   if(actions[0] < min_lot || (actions[1] * MaxTP * Point()) <= stops || (actions[2] * MaxSL * Point()) <= stops)
     {
      account[0] += account[4];
      account[2] = 0;
      account[4] = 0;
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(actions[0] - min_lot) / step_lot) * step_lot;
      if(account[2] > buy_lot)
        {
         float koef = (float)buy_lot / account[2];
         account[0] += account[4] * (1 - koef);
         account[4] *= koef;
        }
      account[2] = (float)buy_lot;
      account[4] += float(buy_lot * prof_1l);
     }
//--- sell control
   if(actions[3] < min_lot || (actions[4] * MaxTP * Point()) <= stops || (actions[5] * MaxSL * Point()) <= stops)
     {
      account[0] += account[5];
      account[3] = 0;
      account[5] = 0;
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(actions[3] - min_lot) / step_lot) * step_lot;
      if(account[3] > sell_lot)
        {
         float koef = float(sell_lot / account[3]);
         account[0] += account[5] * (1 - koef);
         account[5] *= koef;
        }
      account[3] = float(sell_lot);
      account[5] -= float(sell_lot * prof_1l);
     }
   account[6] = account[4] + account[5];
   account[1] = account[0] + account[6];
//---
   vector<float> result = vector<float>::Zeros(AccountDescr);
   result[0] = (account[0] - prev_account[0]) / prev_account[0];
   result[1] = account[1] / prev_account[0];
   result[2] = (account[1] - prev_account[1]) / prev_account[1];
   result[3] = account[2];
   result[4] = account[3];
   result[5] = account[4] / prev_account[0];
   result[6] = account[5] / prev_account[0];
   result[7] = account[6] / prev_account[0];
   double x = (double)time_label / (double)(D'2024.01.01' - D'2023.01.01');
   result[8] = (float)MathSin(2.0 * M_PI * x);
   x = (double)time_label / (double)PeriodSeconds(PERIOD_MN1);
   result[9] = (float)MathCos(2.0 * M_PI * x);
   x = (double)time_label / (double)PeriodSeconds(PERIOD_W1);
   result[10] = (float)MathSin(2.0 * M_PI * x);
   x = (double)time_label / (double)PeriodSeconds(PERIOD_D1);
   result[11] = (float)MathSin(2.0 * M_PI * x);
//--- return result
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | E nt ro py La te nt St at e implementation.                    |
//+------------------------------------------------------------------+
float EntropyLatentState(CNet &net)
  {
//--- random values
   double random[];
   MathRandomNormal(0, 1, LatentCount * SamplLatentStates, random);
   matrix<float> states;
   states.Assign(random);
   states.Reshape(SamplLatentStates, LatentCount);
//--- get means and std
   vector<float> temp;
   matrix<float> stats = matrix<float>::Zeros(SamplLatentStates, 2 * LatentCount);
   net.GetLayerOutput(LatentLayer - 1, temp);
   stats.Row(temp, 0);
   stats = stats.CumSum(0);
   matrix<float> split[];
   stats.Vsplit(2, split);
//--- calculate latent values
   states = states * split[1] + split[0];
//--- add current latent value
   net.GetLayerOutput(LatentLayer, temp);
   states.Resize(SamplLatentStates + 1, LatentCount);
   states.Row(temp, SamplLatentStates);
//--- calculate entropy
   states.SVD(split[0], split[1], temp);
   float result = temp.Sum() / (MathSqrt(MathPow(states, 2.0f).Sum() * MathMax(SamplLatentStates + 1, LatentCount)));
//---
   return result;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> GetProbTrajectories(STrajectory &buffer[], double lambda)
  {
   ulong total = buffer.Size();
   vector<float> result = vector<float>::Zeros(total);
   vector<float> temp;
   for(ulong i = 0; i < total; i++)
     {
      temp.Assign(buffer[i].States[0].rewards);
      result[i] = temp.Sum();
      if(!MathIsValidNumber(result[i]))
         result[i] = -FLT_MAX;
     }
   float max_reward = result.Max();
//---
   vector<float> sorted = result;
   bool sort = true;
   int iter = 0;
   while(sort)
     {
      sort = false;
      for(ulong i = 0; i < sorted.Size() - 1; i++)
         if(sorted[i] > sorted[i + 1])
           {
            float temp = sorted[i];
            sorted[i] = sorted[i + 1];
            sorted[i + 1] = temp;
            sort = true;
           }
      iter++;
     }
//---
   float min = result.Min() - 0.1f * MathAbs(max_reward);
   if(max_reward > min)
     {
      float k = sorted.Percentile(80) - max_reward;
      vector<float> multipl = MathExp(MathAbs(result - max_reward) / (k == 0 ? -1 : k));
      result = (result - min) / (max_reward - min);
      result = result / (result + lambda) * multipl;
      result.ReplaceNan(0);
     }
   else
      result.Fill(1);
   result = result / result.Sum();
   result = result.CumSum();
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | S am pl eT ra je ct or y implementation.                       |
//+------------------------------------------------------------------+
int SampleTrajectory(vector<float> &probability)
  {
//--- check
   ulong total = probability.Size();
   if(total <= 0)
      return -1;
//--- randomize
   float rnd = float(MathRand() / 32767.0);
//--- search
   if(rnd <= probability[0] || total == 1)
      return 0;
   if(rnd > probability[total - 2])
      return int(total - 1);
   int result = int(rnd * total);
   if(probability[result] < rnd)
      while(probability[result] < rnd)
         result++;
   else
     {
      if(result <= 0)
         Sleep(0);
      while(probability[result - 1] >= rnd)
         result--;
     }
//--- return result
   return result;
  }
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

