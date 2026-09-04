//+------------------------------------------------------------------+
//|                                                      NeuroNet.mqh|
//|                   Facade include for the logical NeuroNet blocks.|
//|                                               Copyright 2024, DNG|
//|                                 https://www.mql5.com/ru/users/dng|
//+------------------------------------------------------------------+

#ifndef NEURONET_MQH
#define NEURONET_MQH
#define NEURONET_BUILDING_FACADE
/// \file
/// \brief NeuroNet.mqh
/// Facade include for the logical NeuroNet blocks.
//+------------------------------------------------------------------+
#include "NeuroNet_Definitions.mqh"
#include "NeuroNet_Declarations.mqh"
#include "NeuroNet_Core.mqh"
#include "NeuroNet_RecurrentState.mqh"
#include "NeuroNet_SpikeFlow.mqh"
#include "NeuroNet_Attention.mqh"
#include "NeuroNet_RAGMemory.mqh"
#include "NeuroNet_GraphSparse.mqh"
#include "NeuroNet_AgentsRL.mqh"
#include "NeuroNet_TimeSeries.mqh"
#include "NeuroNet_D2Skill.mqh"
#include "NeuroNet_Misc.mqh"
#undef NEURONET_BUILDING_FACADE
#endif // NEURONET_MQH


//+------------------------------------------------------------------+
