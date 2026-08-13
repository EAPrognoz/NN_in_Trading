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
//| 2     -  Penalty for no opened positions                         |
//+------------------------------------------------------------------+
#include "..\RL\FQF.mqh"
//---
#define        HistoryBars             100            //Depth of history
#define        BarDescr                9              //Elements for 1 bar description
#define        NBarInPattern           1              //Bars for 1 pattern description
#define        AccountDescr            8              //Account description
#define        NActions                6              //Number of possible Actions
#define        NRewards                3              //Number of rewards
#define        TimeDescription         4
#define        LatentCount             1024
#define        EmbeddingSize           32
#define        Sparse                  0.4f
#define        Buffer_Size             6500
#define        DiscFactor              0.99f
#define        FileName                "DoC"
#define        MaxReplayBuffer         500
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        fCAGrad_C               0.3f
#define        iCAGrad_Iters           15
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct STrajectory;
extern STrajectory          Buffer[];
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct SState
  {
   float             state[BarDescr*NBarInPattern];
   float             account[AccountDescr];
   float             action[NActions];
   float             rewards[NRewards];
   //---
                     SState(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
   //---
   // Creates and manages object lifecycle for Clear.
   void              Clear(void)
     {
      ArrayInitialize(state, 0);
      ArrayInitialize(account, 0);
      ArrayInitialize(action, 0);
      ArrayInitialize(rewards, 0);
     }
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
//| | Implements SState.                                             |
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
//| | Saves .                                                        |
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
//| | Loads .                                                        |
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
   void              ClearFirstN(const int n);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
  };
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements STrajectory.                                        |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   // Implements DiscountFactor.
   DiscountFactor(DiscFactor),
   // Creates and manages object lifecycle for CumCounted.
   CumCounted(false)
  {
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves .                                                        |
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
//| | Loads .                                                        |
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
//| | Creates and manages object lifecycle for CumRevards.           |
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
//| | Implements Add.                                                |
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
//| | Creates and manages object lifecycle for ClearFirstN.          |
//+------------------------------------------------------------------+
void STrajectory::ClearFirstN(const int n)
  {
   for(int i=0; i<Buffer_Size-n; i++)
      States[i]=States[i+n];
   Total=MathMax(0,Buffer_Size-n);
   for(int i=Total; i<Buffer_Size; i++)
      States[i].Clear();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves TotalBase.                                               |
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
//| | Loads TotalBase.                                               |
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
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *agent, CArrayObj *rtg)
  {
//---
   CLayerDescription *descr;
//---
   if(!agent)
     {
      agent = new CArrayObj();
      if(!agent)
         return false;
     }
//--- Agent
   agent.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (NRewards + BarDescr*NBarInPattern + AccountDescr + TimeDescription + NActions);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
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
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronEmbeddingOCL;
   prev_count = descr.count = HistoryBars;
     {
      int temp[] = {BarDescr*NBarInPattern,AccountDescr,TimeDescription,NActions,NRewards};
      ArrayCopy(descr.windows,temp);
     }
   uint prev_wout = descr.window_out = EmbeddingSize;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMLMHSparseAttentionOCL;
   prev_count = descr.count = prev_count*5;
   descr.window = prev_wout;
   descr.step = 4;
   descr.window_out = 32;
   descr.layers = 8;
   descr.probability = Sparse;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   descr.window_out = 4;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!agent.Add(descr))
     {
      delete descr;
      return false;
     }
//--- RTG
   if(!rtg)
     {
      rtg = new CArrayObj();
      if(!rtg)
         return false;
     }
//---
   rtg.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (BarDescr*NBarInPattern + AccountDescr + TimeDescription + NActions);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!rtg.Add(descr))
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
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronEmbeddingOCL;
   prev_count = descr.count = HistoryBars;
     {
      int temp[] = {BarDescr*NBarInPattern,AccountDescr,TimeDescription,NActions};
      ArrayCopy(descr.windows,temp);
     }
   prev_wout = descr.window_out = EmbeddingSize;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMLMHSparseAttentionOCL;
   prev_count=descr.count = prev_count*4;
   descr.window = prev_wout;
   descr.step = 4;
   descr.window_out = 32;
   descr.layers = 8;
   descr.probability = Sparse;
   descr.optimization = ADAM;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   descr.window_out = 4;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!rtg.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFQF;
   descr.count = NRewards;
   descr.window_out = 32;
   descr.optimization = ADAM;
   if(!rtg.Add(descr))
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
//| | Creates and manages object lifecycle for CloseByDirection.     |
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
//| | Implements TrailPosition.                                      |
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
//| | Creates and manages object lifecycle for ClosePartial.         |
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements ForecastAccount.                                    |
//+------------------------------------------------------------------+
vector<float> ForecastAccount(float &prev_account[], vector<float> &actions,double prof_1l,float time_label)
  {
   vector<float> account;
   double min_lot = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double step_lot = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double stops = MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL), 1) * Point();
   double margin_buy,margin_sell;
   if(!OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,1.0,SymbolInfoDouble(_Symbol,SYMBOL_ASK),margin_buy) ||
      !OrderCalcMargin(ORDER_TYPE_SELL,_Symbol,1.0,SymbolInfoDouble(_Symbol,SYMBOL_BID),margin_sell))
      return vector<float>::Zeros(prev_account.Size());
//---
   account.Assign(prev_account);
//---
   if(actions[0] >= actions[3])
     {
      actions[0] -= actions[3];
      actions[3] = 0;
      if(actions[0]*margin_buy >= MathMin(account[0],account[1]))
         actions[0] = 0;
     }
   else
     {
      actions[3] -= actions[0];
      actions[0] = 0;
      if(actions[3]*margin_sell >= MathMin(account[0],account[1]))
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

