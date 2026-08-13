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
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define        HistoryBars             120           //Depth of history
#define        Segments                10            //Segments number (must be HistoryBars % Segments == 0)
#define        BarDescr                9             //Elements for 1 bar description
#define        AccountDescr            12            //Account description
#define        NActions                6             //Number of possible Actions
#define        NRewards                3             //Number of rewards
#define        NForecast               24            //Number of forecast
#define        EmbeddingSize           8
#define        PatchSize               3
#define        Buffer_Size             6500
#define        DiscFactor              0.99f
#define        FileName                "MASAAT"
#define        SignalFile(agent)       StringFormat("Signals\\Signal%d.csv",agent)
#define        LatentCount             256
#define        LatentLayer             -1
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        MaxReplayBuffer         500
#define        StartTargetIteration    200000
#define        STE_Multiplier          1.0f/2
#define        Rho                     0.2f
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
   if(Total <= 0)
      return true;
//---
   if(!CumCounted)
      CumRevards();
   Total = MathMin((int)States.Size(), Total);
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
         total = i;
         i--;
         continue;
        }
   FileClose(handle);
//---
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
//---
   return total > 0;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *&actor, CArrayObj *&critic)
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
//---
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
   descr.batch = 1e4;
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
   descr.type = defNeuronMASAAT;
//--- Windows
     {
      int temp[] = {BarDescr, LatentCount};
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.window_out = 32;
   descr.count = HistoryBars;
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
//--- Min Distance
     {
      vector<float> ones = vector<float>::Ones(3);
      vector<float> cs = ones.CumSum() - 1;
      descr.radius = pow(ones * 2, cs) * 0.01f;
     }
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = LatentCount;
   descr.step = AccountDescr;
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
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
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
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 3;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
////--- layer 6
//   if(!(descr = new CLayerDescription()))
//      return false;
//   descr.type = defNeuronFreDFOCL;
//   descr.window = NActions;
//   descr.count =  1;
//   descr.step = int(false);
//   descr.probability = 0.7f;
//   descr.activation = None;
//   descr.optimization = ADAM;
//   if(!actor.Add(descr))
//     {
//      delete descr;
//      return false;
//     }
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (HistoryBars * BarDescr);
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
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2 - 4
   for(int i = 0; i < 3; i++)
     {
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronPSformer;
      descr.window = BarDescr;
      descr.count = HistoryBars;
      descr.window_out = Segments;
      descr.probability = Rho;
      descr.batch = 1e4;
      descr.activation = None;
      descr.optimization = ADAM;
      if(!critic.Add(descr))
        {
         delete descr;
         return false;
        }
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvSAMOCL;
   descr.count = HistoryBars;
   descr.window = BarDescr;
   descr.step = BarDescr;
   descr.window_out = int(LatentCount / descr.count);
   descr.probability = Rho;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.probability = Rho;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = LatentCount;
   descr.step = NActions;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 10
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 11
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = NRewards;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
////--- layer 12
//   if(!(descr = new CLayerDescription()))
//      return false;
//   descr.type = defNeuronFreDFOCL;
//   descr.window = NRewards;
//   descr.count =  1;
//   descr.step = int(false);
//   descr.probability = 0.7f;
//   descr.activation = None;
//   descr.optimization = ADAM;
//   if(!critic.Add(descr))
//     {
//      delete descr;
//      return false;
//     }
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
//| | Gets ProbTrajectories.                                         |
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
//| | Implements SampleTrajectory.                                   |
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for CDeal.                |
//+------------------------------------------------------------------+
void CDeal::CDeal(void) :  OpenTime(0),
   // Creates and manages object lifecycle for CloseTime.
   CloseTime(0),
   // Implements Type.
   Type(POSITION_TYPE_BUY),
   // Implements Volume.
   Volume(0),
   // Implements OpenPrice.
   OpenPrice(0),
   // Implements StopLos.
   StopLos(0),
   // Implements TakeProfit.
   TakeProfit(0),
   // Implements point.
   point(1e-5)
  {
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements Action.                                             |
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
//| | Loads Deals.                                                   |
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements Action.                                             |
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

