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
#include "..\RL\FQF.mqh"
//---
#define                    HistoryBars  20            //Depth of history
#define                    Buffer_Size  2112
#define                    FileName     "SAC"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct SState
  {
   float             state[HistoryBars * 12];
   float             account[9];
   //---
                     SState(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
   //--- overloading
   void              operator=(const SState &obj)   { ArrayCopy(state, obj.state); ArrayCopy(account, obj.account); }
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
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct STrajectory
  {
   SState            States[Buffer_Size];
   int               Actions[Buffer_Size];
   float             Revards[Buffer_Size];
   int               Total;
   float             DiscountFactor;
   bool              CumCounted;
   //---
                     STrajectory(void);
   //---
   bool              Add(SState &state, int action, float reward);
   void              CumRevards(void);
   //---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
  };
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory::STrajectory(void)  :   Total(0),
   DiscountFactor(0.9f),
   // Creates and manages object lifecycle for CumCounted.
   CumCounted(false)
  {
   ArrayInitialize(Actions, -1);
   ArrayInitialize(Revards, 0);
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
   for(int i = 0; i < Total; i++)
     {
      if(!States[i].Save(file_handle))
         return false;
      if(FileWriteInteger(file_handle, Actions[i]) < sizeof(int))
         return false;
      if(FileWriteFloat(file_handle, Revards[i]) < sizeof(float))
         return false;
     }
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
   if(Total >= ArraySize(Actions))
      return false;
//---
   for(int i = 0; i < Total; i++)
     {
      if(!States[i].Load(file_handle))
         return false;
      Actions[i] = FileReadInteger(file_handle);
      Revards[i] = FileReadFloat(file_handle);
     }
   CumCounted = true;
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
   for(int i = Buffer_Size - 2; i >= 0; i--)
      Revards[i] += Revards[i + 1] * DiscountFactor;
   CumCounted = true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements Add.                                                |
//+------------------------------------------------------------------+
bool STrajectory::Add(SState &state, int action, float reward)
  {
   if(Total + 1 >= ArraySize(Actions))
      return false;
   States[Total] = state;
   Actions[Total] = action;
   if(Total > 0)
      Revards[Total - 1] = reward;
   Total++;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *actor, CArrayObj *critic, CArrayObj *scheduler)
  {
//---
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
//---
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         return false;
     }
//---
   if(!scheduler)
     {
      scheduler = new CArrayObj();
      if(!scheduler)
         return false;
     }
//--- Actor
   actor.Clear();
   CLayerDescription *descr;
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = (int)(HistoryBars * 12);
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
   prev_count=descr.count = prev_count-2;
   descr.window = 3;
   descr.step = 1;
   descr.window_out = 2;
   prev_count*=descr.window_out;
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
   descr.count = (prev_count+1)/2;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
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
   descr.count = 1000;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 200;
   descr.window = 100;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 50;
   descr.window = 200;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 4;
   descr.window = 50;
   descr.step = 10;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = 4;
   descr.step = 10;
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
   prev_count = descr.count = (int)(HistoryBars * 12 + 9);
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
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count=descr.count = prev_count-2;
   descr.window = 3;
   descr.step = 1;
   descr.window_out = 2;
   prev_count*=descr.window_out;
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
   descr.type = defNeuronConvOCL;
   descr.count = (prev_count+1)/2;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 150;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 500;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 500;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
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
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Scheduler
   scheduler.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (9 + 40);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
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
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 10;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = 10;
   descr.step = 1;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+

