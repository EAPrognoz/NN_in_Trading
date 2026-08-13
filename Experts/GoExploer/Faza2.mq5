//+------------------------------------------------------------------+
//| Faza2.mq5 |
//| Copyright 2023, DNG |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "Cell.mqh"
#include "..\RL\FQF.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 100000;
input int                  UpdateTarget   = 10000;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CFQF                 StudyNet;
//---
float                dError;
datetime             dtStudied;
bool                 bEventStudy;
//---
CBufferFloat         State1;
CBufferFloat         *Rewards;

Cell                 Base[];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!LoadTotalBase())
      return(INIT_FAILED);
//---
   if(!StudyNet.Load(FileName + ".nnw", dtStudied, true))
     {
      CArrayObj *model = new CArrayObj();
      if(!CreateDescriptions(model))
        {
         delete model;
         return INIT_FAILED;
        }
      if(!StudyNet.Create(model))
        {
         delete model;
         return INIT_FAILED;
        }
      delete model;
     }
   if(!StudyNet.TrainMode(true))
      return INIT_FAILED;
   StudyNet.SetUpdateTarget(UpdateTarget);
//---
   bEventStudy = EventChartCustom(ChartID(), 1, 0, 0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(!!Rewards)
      delete Rewards;
//---
   StudyNet.Save(FileName + ".nnw", 0, true);
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total = ArraySize(Base);
   uint ticks = GetTickCount();
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int i = 0;
      int count = 0;
      int total_max = 0;
      i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (total - 1));
      State1.AssignArray(Base[i].state);
      if(IsStopped())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      int action = Base[i].total_actions;
      if(action < 0)
        {
         iter--;
         continue;
        }
      if(!StudyNet.feedForward(GetPointer(State1), 12, true))
         return;
      action = Base[i].actions[action];
      if(action < 0 || action > 3)
         action = 3;
      StudyNet.getResults(Rewards);
      Rewards.BufferInit(4, 0);
      if(!Rewards.Update(action, -Base[i].value))
         return;
      //---
      if(!StudyNet.backProp(GetPointer(Rewards)))
         return;
      if(GetTickCount() - ticks > 500)
        {
         Comment(StringFormat("%.2f%% -> Error %.8f", iter * 100.0 / (double)(Iterations), StudyNet.getRecentAverageError()));
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %10.7f", __FUNCTION__, __LINE__, StudyNet.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes Descriptions.                           |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *Description)
  {
//---
   if(!Description)
     {
      Description = new CArrayObj();
      if(!Description)
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
   descr.batch = 200;
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
   descr.type = defNeuronFQF;
   descr.count = 4;
   descr.window_out = 32;
   descr.optimization = ADAM;
   if(!Description.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Loads TotalBase.                                               |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Base, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Base[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+

