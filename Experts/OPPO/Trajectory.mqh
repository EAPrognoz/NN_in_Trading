//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                                             OPPO |
//|                                                   Copyright DNG® |
//|                           https://www.mql5.com/ru/articles/13912 |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Rewards structure                                                |
//|   0     -  Delta Balance                                         |
//|   1     -  Delta Equity ( "-" Drawdown / "+" Profit)             |
//|   2     -  Penalty for no opened positions                       |
//+------------------------------------------------------------------+
#include "..\RL\FQF.mqh"
//---
#define        HistoryBars             48             //Depth of history
#define        BarDescr                9              //Elements for 1 bar description
#define        NBarInPattern           1              //Bars for 1 pattern description
#define        AccountDescr            8              //Account description
#define        NActions                6              //Number of possible Actions
#define        TimeDescription         4
#define        LatentCount             1024
#define        LatentLayer             2
#define        EmbeddingSize           32
#define        Buffer_Size             6500
#define        DiscFactor              0.99f
#define        FileName                "OPPO"
#define        MaxReplayBuffer         500
#define        MaxSL                   1000
#define        MaxTP                   1000
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
struct STrajectory;
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
extern STrajectory Buffer[];
//+------------------------------------------------------------------+
//| SState - Store one trajectory state                              |
//+------------------------------------------------------------------+
struct SState
  {
   float state[BarDescr * NBarInPattern];
   float account[AccountDescr];
   float action[NActions];
   float scheduler[EmbeddingSize];
   //---
                     SState(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
   //---
//+------------------------------------------------------------------+
//| Clear                                                            |
//+------------------------------------------------------------------+
   void              Clear(void);

//--- overloading
   void              operator=(const SState &obj);
  };
//+------------------------------------------------------------------+
//| Clear - Clears state data.                                       |
//+------------------------------------------------------------------+
void SState::Clear(void)
  {
   ArrayInitialize(state,0);
   ArrayInitialize(account,0);
   ArrayInitialize(action,0);
   ArrayInitialize(scheduler,0);
  }
//+------------------------------------------------------------------+
//| operator= - Copies state data.                                   |
//+------------------------------------------------------------------+
void SState::operator=(const SState &obj)
  {
   ArrayCopy(state,obj.state);
   ArrayCopy(account,obj.account);
   ArrayCopy(action,obj.action);
   ArrayCopy(scheduler,obj.scheduler);
  }
//+------------------------------------------------------------------+
//| SState - Initialize the trajectory state                         |
//+------------------------------------------------------------------+
SState::SState(void)
  {
   ArrayInitialize(state,0);
   ArrayInitialize(account,0);
   ArrayInitialize(action,0);
   ArrayInitialize(scheduler,0);
  }
//+------------------------------------------------------------------+
//| Save - Write the trajectory state to a file                      |
//+------------------------------------------------------------------+
bool SState::Save(int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalse
     }
//---
   int total=ArraySize(state);
   dFileWriteUInt(file_handle,total)

   for(int i=0; i<total; i++)
     {
      dFileWriteFloat(file_handle,state[i])
     }
//---
   total=ArraySize(account);
   dFileWriteUInt(file_handle,total)

   for(int i=0; i<total; i++)
     {
      dFileWriteFloat(file_handle,account[i])
     }
//---
   total=ArraySize(action);
   dFileWriteUInt(file_handle,total)

   for(int i=0; i<total; i++)
     {
      dFileWriteFloat(file_handle,action[i])
     }
//---
   total=ArraySize(scheduler);
   dFileWriteUInt(file_handle,total)

   for(int i=0; i<total; i++)
     {
      dFileWriteFloat(file_handle,scheduler[i])
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Load - Read the trajectory state from a file                     |
//+------------------------------------------------------------------+
bool SState::Load(int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalseEx("Invalid file handle")
     }
   if(FileIsEnding(file_handle))
      return(false);
//---
   int total=FileReadInteger(file_handle);
   if(total!=ArraySize(state))
      return(false);
//---
   for(int i=0; i<total; i++)
     {
      dFileReadUFloat(file_handle,state[i])
     }
//---
   total=FileReadInteger(file_handle);
   if(total!=ArraySize(account))
      return(false);
//---
   for(int i=0; i<total; i++)
     {
      dFileReadUFloat(file_handle,account[i])
     }
//---
   total=FileReadInteger(file_handle);
   if(total!=ArraySize(action))
      return(false);
//---
   for(int i=0; i<total; i++)
     {
      dFileReadUFloat(file_handle,action[i])
     }
//---
   total=FileReadInteger(file_handle);
   if(total!=ArraySize(scheduler))
      return(false);
//---
   for(int i=0; i<total; i++)
     {
      dFileReadUFloat(file_handle,scheduler[i])
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| STrajectory - Store a simulated trading trajectory               |
//+------------------------------------------------------------------+
struct STrajectory
  {
   SState States[Buffer_Size];
   int    Total;
   double Profit;
   //---
                     STrajectory(void);
   //---
   bool              Add(SState &state);
   void              ClearFirstN(const int n);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);

//--- overloading
   void              operator=(const STrajectory &obj);
  };
//+------------------------------------------------------------------+
//| operator= - Copies trajectory data.                              |
//+------------------------------------------------------------------+
void STrajectory::operator=(const STrajectory &obj)
  {
   Total=obj.Total;
   Profit=obj.Profit;
   for(int i=0; i<Buffer_Size; i++)
      States[i]=obj.States[i];
  }
//+------------------------------------------------------------------+
//| STrajectory - Initialize an empty trajectory                     |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   Profit(0)
  {
  }
//+------------------------------------------------------------------+
//| Save - Write the trajectory to a file                            |
//+------------------------------------------------------------------+
bool STrajectory::Save(int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalseEx("Invalid file handle")
     }
//---
   if(FileWriteInteger(file_handle,Total)<sizeof(int))
      return(false);

   if(FileWriteDouble(file_handle,Profit)<sizeof(float))
      return(false);

   for(int i=0; i<Total; i++)
      if(!States[i].Save(file_handle))
         return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Load - Read the trajectory from a file                           |
//+------------------------------------------------------------------+
bool STrajectory::Load(int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalseEx("Invalid file handle")
     }
//---
   Total=FileReadInteger(file_handle);
   if(FileIsEnding(file_handle)||Total>=ArraySize(States))
      return(false);
   Profit=FileReadDouble(file_handle);
//---
   for(int i=0; i<Total; i++)
      if(!States[i].Load(file_handle))
         return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Add - Append a state to the trajectory                           |
//+------------------------------------------------------------------+
bool STrajectory::Add(SState &state)
  {
   if(Total+1>=ArraySize(States))
      return(false);
   States[Total]=state;
   Total++;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| ClearFirstN - Remove the oldest trajectory states                |
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
//| SaveTotalBase - Write the trajectory buffer to a file            |
//+------------------------------------------------------------------+
int SaveTotalBase(int min_bars)
  {
   int total=ArraySize(Buffer);
   if(total<0)
      return(0);

   int handle=FileOpen(FileName+".bd",FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle<0)
      return(0);

   int indexes[MaxReplayBuffer];
   int count=0;
   for(int i=total-1; i>=0; i--)
     {
      if(Buffer[i].Total<min_bars)
         continue;
      indexes[count]=i;
      count++;
     }
   if(FileWriteInteger(handle,count)<INT_VALUE)
     {
      FileClose(handle);

      return(0);
     }
   for(int i=count-1; i>=0; i--)
      if(!Buffer[indexes[i]].Save(handle))
        {
         FileClose(handle);

         return(count-(i+1));
        }
   FileFlush(handle);
   FileClose(handle);
//---
   return(count);
  }
//+------------------------------------------------------------------+
//| LoadTotalBase - Read the trajectory buffer from a file           |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle=FileOpen(FileName+".bd",FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle<0)
      return(false);

   int total=FileReadInteger(handle);
   if(total<=0)
     {
      FileClose(handle);

      return(false);
     }
   if(ArrayResize(Buffer,total)<total)
     {
      FileClose(handle);

      return(false);
     }
   for(int i=0; i<total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);

         return(false);
        }
   FileClose(handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| CreateAgentDescriptions - Create agent descriptions              |
//+------------------------------------------------------------------+
bool CreateAgentDescriptions(CArrayObj *agent)
  {
//---
   CLayerDescription *descr;
//---
   if(!agent)
     {
      agent=new CArrayObj();
      if(!agent)
         return(false);
     }

//--- agent
   agent.Clear();

//--- input layer
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   uint prev_count=descr.count=(BarDescr * NBarInPattern+AccountDescr+TimeDescription+NActions+EmbeddingSize);
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 1
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBatchNormOCL;
   descr.count=prev_count;
   descr.batch=1000;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 2
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronEmbeddingOCL;
   prev_count=descr.count=HistoryBars;
     {
      int temp[]={BarDescr * NBarInPattern,AccountDescr,TimeDescription,NActions,EmbeddingSize};
      ArrayCopy(descr.windows,temp);
     }
   uint prev_wout=descr.window_out=EmbeddingSize;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 3
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=EmbeddingSize;
   descr.step=prev_count * 5;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 4
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronMLMHAttentionOCL;
   prev_count=descr.count=prev_count * 5;
   descr.window=EmbeddingSize;
   descr.step=8;
   descr.window_out=32;
   descr.layers=4;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 5
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=EmbeddingSize;
   descr.step=prev_count;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 6
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronConvOCL;
   prev_count=descr.count=prev_count;
   descr.window=EmbeddingSize;
   descr.step=EmbeddingSize;
   prev_wout=descr.window_out=EmbeddingSize;
   descr.optimization=ADAM;
   descr.activation=LReLU;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 7
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=prev_count;
   descr.step=prev_wout;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 8
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronConvOCL;
   prev_count=descr.count=prev_count;
   descr.window=prev_wout;
   descr.step=prev_wout;
   prev_wout=descr.window_out=prev_wout / 2;
   descr.optimization=ADAM;
   descr.activation=LReLU;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 9
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=prev_count;
   descr.step=prev_wout;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 10
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   descr.count=LatentCount;
   descr.optimization=ADAM;
   descr.activation=LReLU;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 11
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   prev_count=descr.count=LatentCount;
   descr.activation=LReLU;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 12
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   descr.count=LatentCount;
   descr.activation=LReLU;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 13
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   descr.count=NActions;
   descr.activation=SIGMOID;
   descr.optimization=ADAM;
   if(!agent.Add(descr))
     {
      delete descr;

      return(false);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| CreateSchedulerDescriptions - Create scheduler descriptions      |
//+------------------------------------------------------------------+
bool CreateSchedulerDescriptions(CArrayObj *scheduler)
  {
//---
   CLayerDescription *descr;
//---
   if(!scheduler)
     {
      scheduler=new CArrayObj();
      if(!scheduler)
         return(false);
     }

//--- scheduler
   scheduler.Clear();

//--- input layer
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   uint prev_count=descr.count=(BarDescr * NBarInPattern+AccountDescr+TimeDescription+NActions);
   descr.activation=None;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 1
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBatchNormOCL;
   descr.count=prev_count;
   descr.batch=1000;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 2
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronEmbeddingOCL;
   prev_count=descr.count=HistoryBars;
     {
      uint temp[]={BarDescr * NBarInPattern,AccountDescr,TimeDescription,NActions};
      ArrayCopy(descr.windows,temp);
     }
   uint prev_wout=descr.window_out=EmbeddingSize;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 3
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=EmbeddingSize;
   descr.step=prev_count * 4;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 4
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronMLMHAttentionOCL;
   prev_count=descr.count=prev_count * 4;
   descr.window=EmbeddingSize;
   descr.step=8;
   descr.window_out=32;
   descr.layers=4;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 5
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronSoftMaxOCL;
   descr.count=EmbeddingSize;
   descr.step=prev_count;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 6
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   descr.count=LatentCount;
   descr.optimization=ADAM;
   descr.activation=LReLU;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }

//--- layer 7
   if(!(descr=new CLayerDescription()))
      return(false);

   descr.type=defNeuronBaseOCL;
   prev_count=descr.count=EmbeddingSize;
   descr.activation=None;
   descr.optimization=ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;

      return(false);
     }
//---
   return(true);
  }
#ifndef Study
//+------------------------------------------------------------------+
//| IsNewBar - Detect a new bar                                      |
//+------------------------------------------------------------------+
bool IsNewBar(void)
  {
   static datetime last_bar=0;
   if(last_bar>=iTime(Symb.Name(),TimeFrame,0))
      return(false);
//---
   last_bar=iTime(Symb.Name(),TimeFrame,0);

   return(true);
  }
//+------------------------------------------------------------------+
//| CloseByDirection - Close positions of the specified type         |
//+------------------------------------------------------------------+
bool CloseByDirection(ENUM_POSITION_TYPE type)
  {
   int  total=PositionsTotal();
   bool result=true;
   for(int i=total-1; i>=0; i--)
     {
      if(PositionGetSymbol(i)!=Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE)!=type)
         continue;
      result=(Trade.PositionClose(PositionGetInteger(POSITION_TICKET))&&result);
     }
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| TrailPosition - Adjust trailing stops on open positions          |
//+------------------------------------------------------------------+
bool TrailPosition(ENUM_POSITION_TYPE type,double sl,double tp)
  {
   int  total=PositionsTotal();
   bool result=true;
//---
   for(int i=0; i<total; i++)
     {
      if(PositionGetSymbol(i)!=Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE)!=type)
         continue;
      bool   modify=false;
      double psl=PositionGetDouble(POSITION_SL);
      double ptp=PositionGetDouble(POSITION_TP);
      switch(type)
        {
         case POSITION_TYPE_BUY:
            if((sl-psl)>=Symb.Point())
              {
               psl=sl;
               modify=true;
              }
            if(MathAbs(tp-ptp)>=Symb.Point())
              {
               ptp=tp;
               modify=true;
              }
            break;
         case POSITION_TYPE_SELL:
            if((psl-sl)>=Symb.Point())
              {
               psl=sl;
               modify=true;
              }
            if(MathAbs(tp-ptp)>=Symb.Point())
              {
               ptp=tp;
               modify=true;
              }
            break;
        }
      if(modify)
         result=(Trade.PositionModify(PositionGetInteger(POSITION_TICKET),psl,ptp)&&result);
     }
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| ClosePartial - Close part of the position volume                 |
//+------------------------------------------------------------------+
bool ClosePartial(ENUM_POSITION_TYPE type,double value)
  {
   if(value<=0)
      return(true);
//---
   for(int i=0; (i<PositionsTotal()&&value>0); i++)
     {
      if(PositionGetSymbol(i)!=Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE)!=type)
         continue;
      double pvalue=PositionGetDouble(POSITION_VOLUME);
      if(pvalue<=value)
        {
         if(Trade.PositionClose(PositionGetInteger(POSITION_TICKET)))
           {
            value -= pvalue;
            i--;
           }
        }
      else
        {
         if(Trade.PositionClosePartial(PositionGetInteger(POSITION_TICKET),value))
            value=0;
        }
     }
//---
   return(value<=0);
  }
#endif
//+------------------------------------------------------------------+
//| GetProbTrajectories                                              |
//+------------------------------------------------------------------+
vector<double>GetProbTrajectories(STrajectory &buffer[],float lanbda)
  {
   ulong total=buffer.Size();
   vector<double>rewards=vector<double>::Zeros(total);
   for(ulong i=0; i<total; i++)
      rewards[i]=Buffer[i].Profit;
   double std=rewards.Std();
   double max_profit=rewards.Max();
//---
   vector<double>sorted=rewards;
   bool sort=true;
   while(sort)
     {
      sort=false;
      for(ulong i=0; i<sorted.Size()-1; i++)
         if(sorted[i]>sorted[i+1])
           {
            double temp=sorted[i];
            sorted[i]=sorted[i+1];
            sorted[i+1]=temp;
            sort=true;
           }
     }
//---
   double min=rewards.Min()-0.1 * std;
   if(max_profit>min)
     {
      double k=sorted.Percentile(90)-max_profit;
      vector<double>multipl=MathAbs(rewards-max_profit) / (k==0 ?-std : k);
      multipl=exp(multipl);
      rewards=(rewards-min) / (max_profit-min);
      rewards=rewards / (rewards+lanbda) * multipl;
      rewards.ReplaceNan(0);
     }
   else
      rewards.Fill(1);
   rewards=rewards / rewards.Sum();
   rewards=rewards.CumSum();
//---
   return(rewards);
  }
//+------------------------------------------------------------------+
//| SampleTrajectory - Sample a trajectory from the replay buffer    |
//+------------------------------------------------------------------+
int SampleTrajectory(vector<double>&probability)
  {
//--- check
   ulong total=probability.Size();
   if(total<=0)
      return(-1);

//--- randomize
   double rnd=double(MathRand()) / 32767.0;

//--- search
   if(rnd<=probability[0]||total==1)
      return(0);

   if(rnd>probability[total-2])
      return(int(total-1));

   int result=int(rnd * total);
   if(probability[result]<rnd)
      while(probability[result]<rnd)
         result++;
   else
      while(probability[result-1]>=rnd)
         result--;

//--- return result
   return(result);
  }
//+------------------------------------------------------------------+

