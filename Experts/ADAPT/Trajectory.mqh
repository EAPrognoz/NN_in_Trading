//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+-----------------------------------------------------------------------------------------------------------------------+
//| Rewards structure 0 - Delta Balance 1 - Delta Equity ( "-"  Drawdown / "+" Profit) 2 - Penalty for no open positions  |
//+-----------------------------------------------------------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define        HistoryBars             1             //Depth of history
#define        GPTBars                 100           //Analitic Depth of history
#define        PrecoderBars            10            //Forecast Depth
#define        BarDescr                9             //Elements for 1 bar description
#define        AccountDescr            12            //Account description
#define        NActions                6             //Number of possible Actions
#define        NRewards                3             //Number of rewards
#define        NForecast               5             //Number of forecast
#define        Lenc                    3             //Number ADAPT Encoder blocks
#define        EmbeddingSize           32
#define        Buffer_Size             6500
#define        DiscFactor              0.99f
#define        FileName                "ADAPT"
#define        SignalFile(agent)       StringFormat("Signals\\Signal%d.csv",agent)
#define        LatentCount             512
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        MaxReplayBuffer         500
#define        StartTargetIteration    200000
#define        STE_Multiplier          1.0f/2
#define        STE_Noise_Multiplier    1.0f/10        // λ определяет влияние ошибки прогнозирования шума
#define        STD_Delta_Multiplier    1.0f/20        // коэффициент шума ω
//+------------------------------------------------------------------+
//|                                                                  |
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
   //--- Overloading.
   void              operator=(const SState &obj)
     {
      ArrayCopy(state, obj.state);
      ArrayCopy(account, obj.account);
      ArrayCopy(action, obj.action);
      ArrayCopy(rewards, obj.rewards);
     }
  };
//+------------------------------------------------------------------+
//| Implements SState::SState.                                       |
//+------------------------------------------------------------------+
SState::SState(void)
  {
   ArrayInitialize(state, 0);
   ArrayInitialize(account, 0);
   ArrayInitialize(action, 0);
   ArrayInitialize(rewards, 0);
  }
//+------------------------------------------------------------------+
//| Implements SState::Save.                                         |
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
//| Implements SState::Load.                                         |
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
//|                                                                  |
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
//| Implements STrajectory::STrajectory.                             |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   //--- Implements DiscountFactor.
   DiscountFactor(DiscFactor),
   //--- Creates and manages object lifecycle for CumCounted.
   CumCounted(false)
  {
  }
//+------------------------------------------------------------------+
//| Implements STrajectory::Save.                                    |
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
//| Implements STrajectory::Load.                                    |
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
//| Implements STrajectory::CumRevards.                              |
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
//| Implements STrajectory::Add.                                     |
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
//| Implements SaveTotalBase.                                        |
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
//| Implements LoadTotalBase.                                        |
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
//| Implements CreateDescriptions.                                   |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *actor, CArrayObj *end_encoder, CArrayObj *state_encoder)
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
   if(!end_encoder)
     {
      end_encoder = new CArrayObj();
      if(!end_encoder)
         return false;
     }
   if(!state_encoder)
     {
      state_encoder = new CArrayObj();
      if(!state_encoder)
         return false;
     }
//--- Endpoints Encoder.
   end_encoder.Clear();
//--- Input layer.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = 3 * NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!end_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = NForecast;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!end_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- State Encoder.
   state_encoder.Clear();
//--- Input layer.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = GPTBars * EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!state_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!state_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Actor.
   actor.Clear();
//--- Input layer.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = LatentCount;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4.
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
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Implements CreateTrajNetDescriptions.                            |
//+------------------------------------------------------------------+
bool CreateTrajNetDescriptions(CArrayObj *encoder, CArrayObj *endpoints, CArrayObj *probability)
  {
//---
   CLayerDescription *descr;
//---
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         return false;
     }
   if(!endpoints)
     {
      endpoints = new CArrayObj();
      if(!endpoints)
         return false;
     }
   if(!probability)
     {
      probability = new CArrayObj();
      if(!probability)
         return false;
     }
//--- Encoder.
   encoder.Clear();
//--- Input layer.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (HistoryBars * BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = MathMax(1000, GPTBars);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronEmbeddingOCL;
     {
      uint temp[] = {prev_count};
      ArrayCopy(descr.windows, temp);
     }
   prev_count = descr.count = GPTBars;
   uint prev_wout = descr.window_out = EmbeddingSize;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronPEOCL;
   descr.count = prev_count;
   descr.window = prev_wout;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
   for(int l = 0; l < Lenc; l++)
     {
      //--- layer 4.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronTransposeOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
      //--- layer 5.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMH2AttentionOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      descr.step = 8;
      descr.window_out = 16;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
      //--- layer 6.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMLMHAttentionOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
      //--- layer 7.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronTransposeOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
      //--- layer 8.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMH2AttentionOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
      //--- layer 9.
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMLMHAttentionOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
     }
//--- Endpoints.
   endpoints.Clear();
//--- Input layer.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (prev_count * prev_wout);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = LatentCount;
   descr.step = 1;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 3 * NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Probability.
   probability.Clear();
//--- Input layer.
   if(!probability.Add(endpoints.At(0)))
      return false;
//--- layer 1.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = 3 * NForecast;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4.
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = NForecast;
   descr.step = 1;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
#ifndef Study
//+------------------------------------------------------------------+
//| Implements IsNewBar.                                             |
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
//| Implements CloseByDirection.                                     |
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
//| Implements TrailPosition.                                        |
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
//| Implements ClosePartial.                                         |
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
//| Implements GetProbTrajectories.                                  |
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
//| Implements SampleTrajectory.                                     |
//+------------------------------------------------------------------+
int SampleTrajectory(vector<float> &probability)
  {
//--- check.
   ulong total = probability.Size();
   if(total <= 0)
      return -1;
//--- randomize.
   float rnd = float(MathRand() / 32767.0);
//--- search.
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
//--- return result.
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//| Implements CDeal::CDeal.                                         |
//+------------------------------------------------------------------+
void CDeal::CDeal(void) :  OpenTime(0),
   //--- Creates and manages object lifecycle for CloseTime.
   CloseTime(0),
   //--- Implements Type.
   Type(POSITION_TYPE_BUY),
   //--- Implements Volume.
   Volume(0),
   //--- Implements OpenPrice.
   OpenPrice(0),
   //--- Implements StopLos.
   StopLos(0),
   //--- Implements TakeProfit.
   TakeProfit(0),
   //--- Implements point.
   point(1e-5)
  {
  }
//+------------------------------------------------------------------+
//| Implements CDeal::Action.                                        |
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
//|                                                                  |
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
//| Implements CDeals::LoadDeals.                                    |
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
//| Implements CDeals::Action.                                       |
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
