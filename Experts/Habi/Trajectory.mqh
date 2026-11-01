//+------------------------------------------------------------------+
//|                                                  Trajectory.mqh  |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#ifndef HABI_TRAJECTORY_MQH
#define HABI_TRAJECTORY_MQH
#include "Habi_Models.mqh"
#include "Habi_History.mqh"
#include "Habi_Execution.mqh"

//+------------------------------------------------------------------+
//| HabiStart - validates configuration and initialises modules      |
//+------------------------------------------------------------------+
bool HabiStart(CHabiModels &models,CHabiHistory &history,int stage)
  {
   if(!HabiCommonConfig(stage!=2,stage==1))
     {
      Print("Habi: invalid configuration; set required phase parameters and positive PenaltyInvalidCurrency for training");

      return(false);
     }

   uint seed=RandomSeed!=0?RandomSeed:(uint)(GetMicrosecondCount()^(ulong)TimeLocal());

   MathSrand(seed);
   PrintFormat("Habi r0.26 base=c0b9b0e seed=%u stage=%d",seed,stage);

   if(!history.Init())
     {
      Print("Habi: unavailable indicators or incompatible symbol profile");

      return(false);
     }

   if(!models.Init(stage))
     {
      Print("Habi: model creation/load failed; check native layer geometry, files, BN and OpenCL");

      return(false);
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| HabiFatal - prints a critical error and removes the program      |
//+------------------------------------------------------------------+
void HabiFatal(string operation)
  {
   Print("Habi CRITICAL: ",operation,"; stopping before new predictions/orders. Existing broker positions remain under broker control.");
   ExpertRemove();
  }
#endif

//+------------------------------------------------------------------+
