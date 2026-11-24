//+------------------------------------------------------------------+
//|                                                     NeuroNet.mqh |
//|                                                     NeuroNet_DNG |
//|                                              Copyright 2026, DNG |
//|                            https://www.mql5.com/ru/articles/7447 |
//+------------------------------------------------------------------+

#ifndef NEURONET_MQH
#define NEURONET_MQH
#define NEURONET_BUILDING_FACADE
//--- \file
//--- \brief NeuroNet.mqh
//--- facade include for the logical NeuroNet blocks.
#include "NeuroNet_Definitions.mqh"
#include "NeuroNet_Declarations.mqh"
#include "NeuroNet_AlphaGen.mqh"
#include "NeuroNet_AlphaGenPool.mqh"
#include "NeuroNet_AlphaGenPolicy.mqh"
#include "NeuroNet_AlphaGenComposite.mqh"
#ifdef NN_ALPHAQCM
#include "NeuroNet_AlphaQCMPool.mqh"
#include "NeuroNet_AlphaQCMReplay.mqh"
#include "NeuroNet_AlphaQCMOpenCL.mqh"
#include "NeuroNet_AlphaQCMGenerator.mqh"
#include "NeuroNet_AlphaQCMEncoder.mqh"
#include "NeuroNet_AlphaQCMQuantile.mqh"
#include "NeuroNet_AlphaQCMComposite.mqh"
#endif
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
#include "NeuroNet_InferenceSnapshot.mqh"
#undef NEURONET_BUILDING_FACADE
#endif // NEURONET_MQH
//+------------------------------------------------------------------+
