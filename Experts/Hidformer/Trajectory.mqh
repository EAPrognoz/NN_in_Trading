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
#define        EmbeddingSize           32
#define        PatchSize               3
#define        Buffer_Size             31000
#define        DiscFactor              0.99f
#define        FileName                "Hidformer"
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
//| | Function implementation for STrajectory.                       |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   // Function implementation for DiscountFactor.
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
//| |
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
//| |
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
//| |
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
   int load_false = 0;
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         total = i;
         break;
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
//| |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *&actor)
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
   descr.type = defNeuronHidformer;
//--- Windows
     {
      int temp[] = {BarDescr, 120, NActions}; //Window, Stack Size, N Actions
      if(ArrayCopy(descr.windows, temp) < int(temp.Size()))
         return false;
     }
   descr.count = HistoryBars;
   descr.window_out = 32;
   descr.step = 4;                              // Heads
   descr.layers =3;
   descr.batch = 1e4;
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
   descr.type = defNeuronMacroHFTvsRiskManager;
//--- Windows
     {
      int temp[] = {3, 15, NActions,AccountDescr}; //Window, Stack Size, N Actions, Account Description
      if(ArrayCopy(descr.windows, temp) < int(temp.Size()))
         return false;
     }
   descr.count = 10;
   descr.window_out = 16;
   descr.step = 4;                              // Heads
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvSAMOCL;
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
//| |
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
//| |
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
//| |
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
//| | Function implementation for GetProbTrajectories.               |
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
//| |
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
//| | Function implementation for CDeal.                             |
//+------------------------------------------------------------------+
void CDeal::CDeal(void) :  OpenTime(0),
   // Function implementation for CloseTime.
   CloseTime(0),
   // Function implementation for Type.
   Type(POSITION_TYPE_BUY),
   // Function implementation for Volume.
   Volume(0),
   // Function implementation for OpenPrice.
   OpenPrice(0),
   // Function implementation for StopLos.
   StopLos(0),
   // Function implementation for TakeProfit.
   TakeProfit(0),
//| Implements point                                             |
   point(1e-5)
  {
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Function implementation for Action.                            |
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Function implementation for Action.                            |
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

