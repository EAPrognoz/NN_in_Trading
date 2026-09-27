//+------------------------------------------------------------------+
//|                                           NeuroNet_RAGMemory.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#ifndef NEURONET_RAG_MEMORY_MQH
#define NEURONET_RAG_MEMORY_MQH
//--- \file
//--- \brief NeuroNet_RAGMemory.mqh
//--- Immutable RAG memory layer implementation: GPU TopK retrieval
//--- and copy-on-write snapshot publication.
//--- \author <A HREF="https://www.mql5.com/en/users/dng"> DNG </A>
//+------------------------------------------------------------------+
//| RAG memory local constants                                       |
//+------------------------------------------------------------------+
#define RAG_MAX_FLOAT          3.402823466e+38f
#define RAG_TERMINAL_EPSILON   1.0e-6f
#define RAG_NORM_MIN           0.999
#define RAG_NORM_MAX           1.001
#define RAG_SNAPSHOT_MAGIC     0x52414731
#define RAG_FNV_PRIME          1099511628211
#define RAG_FNV_OFFSET         1469598103934665603
#define RAG_MAX_CANDIDATES     65536
#define RAG_FORMAT_SAVE        2
#define RAG_FORMAT_PUBLISHED   3
//+------------------------------------------------------------------+
//| Constructor: initializes the CRAGInferenceSet layer defaults.    |
//+------------------------------------------------------------------+
CRAGInferenceSet::CRAGInferenceSet(void) : opencl(NULL), centroids(NULL), used(NULL),
   scores(NULL), partial(NULL), merge(NULL), top(NULL), actions(NULL), actionUsed(NULL),
   actionMeanReward(NULL), actionHitCount(NULL), absoluteRelevance(NULL), tokens(NULL),
   fingerprint(0)
  {
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CRAGInferenceSet GPU resources.         |
//+------------------------------------------------------------------+
CRAGInferenceSet::~CRAGInferenceSet(void)
  {
   if(centroids != NULL)
      delete centroids;
   if(used != NULL)
      delete used;
   if(scores != NULL)
      delete scores;
   if(partial != NULL)
      delete partial;
   if(merge != NULL)
      delete merge;
   if(top != NULL)
      delete top;
   if(actions != NULL)
      delete actions;
   if(actionUsed != NULL)
      delete actionUsed;
   if(actionMeanReward != NULL)
      delete actionMeanReward;
   if(actionHitCount != NULL)
      delete actionHitCount;
   if(absoluteRelevance != NULL)
      delete absoluteRelevance;
   if(tokens != NULL)
      delete tokens;
   opencl = NULL;
  }
//+------------------------------------------------------------------+
//--- CLayerDescription mapping for defNeuronRAGMemory:
//--- window=scenario embedding, window_out=action dimension, count=token width,
//--- layers=TopK, units[0]=scenario capacity, heads[0]=action capacity.
//--- This is structural model metadata. OnlineMemorySize is runtime-only.
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronRAGMemory layer defaults.    |
//+------------------------------------------------------------------+
CNeuronRAGMemory::CNeuronRAGMemory(void) : iScenarioEmbedding(0),
   iActionDimension(0), iTokenWidth(0), iScenarioCapacity(0),
   iActionCapacity(0), iTopK(0), iOnlineMemorySize(0), iScenarioCount(0),
   iFormatVersion(RAG_FORMAT_SAVE), m_inference(NULL), iPublicationGeneration(0), iNextPendingId(1)
  {
  }
//+-----------------------------------------------------------------------------------+
//| Constructor: initializes the CNeuronRAGMemory layer defaults (CNeuronRAGMemory).  |
//+-----------------------------------------------------------------------------------+
CNeuronRAGMemory::CNeuronRAGMemory(CLayerDescription &description) : iScenarioEmbedding(0),
   iActionDimension(0), iTokenWidth(0), iScenarioCapacity(0),
   iActionCapacity(0), iTopK(0), iOnlineMemorySize(0), iScenarioCount(0),
   iFormatVersion(RAG_FORMAT_SAVE), m_inference(NULL), iPublicationGeneration(0), iNextPendingId(1)
  {
   Configure(description);
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronRAGMemory GPU resources.         |
//+------------------------------------------------------------------+
CNeuronRAGMemory::~CNeuronRAGMemory(void)
  {
   if(m_inference != NULL)
      delete m_inference;
   m_inference = NULL;
   ArrayResize(pending_records, 0);
   ArrayResize(pending_facts, 0);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory configure operation.                |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Configure(CLayerDescription &description)
  {
   if(description.type != defNeuronRAGMemory || description.window == 0 ||
      description.window_out == 0 || description.count == 0 || description.layers == 0 ||
      ArraySize(description.units) < 1 || ArraySize(description.heads) < 1 ||
      description.units[0] == 0 || description.heads[0] == 0 ||
      description.count != description.window_out + 2 ||
      description.layers > RAG_TOPK_MAX || description.units[0] > RAG_FLOAT_INDEX_LIMIT ||
      description.units[0] >= (uint)INT_MAX || description.heads[0] >= (uint)INT_MAX)
      ReturnFalse;
   iScenarioEmbedding = description.window;
   iActionDimension = description.window_out;
   iTokenWidth = description.count;
   iTopK = description.layers;
   iScenarioCapacity = description.units[0];
   iActionCapacity = description.heads[0];
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory clear terminal operation.           |
//+------------------------------------------------------------------+
void CNeuronRAGMemory::ClearTerminal(SRAGTerminalOutcome &terminal) const
  {
   terminal.relative_return = 0.0f;
   terminal.maxDrawdown = 0.0f;
   terminal.costs = 0.0f;
   terminal.riskUsage = 0.0f;
   terminal.duration = 0.0f;
   terminal.reward = 0.0f;
  }
//+-----------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory normalize action descriptor  operation.  |
//+-----------------------------------------------------------------------+
bool CNeuronRAGMemory::NormalizeActionDescriptor(const float &source[], const double equity,
      const double riskBudget, const double pointCost,
      const double point, const double volatility,
      const double volatilityRange,
      const double maximumTakeProfit,
      const double maximumStopLoss,
      float &destination[]) const
  {
   ArrayInitialize(destination, 0.0f);
//--- The generic RAG layer owns the numerical canonical descriptor, but its
//--- caller owns the interpretation and supplies every scale explicitly.
   if(iActionDimension != 6 || ArraySize(source) != (int)iActionDimension ||
      ArraySize(destination) != (int)iActionDimension || equity <= 0 || riskBudget <= 0 ||
      pointCost <= 0 || point <= 0 || volatility <= 0 || volatilityRange <= 0 ||
      maximumTakeProfit < 0 || maximumStopLoss < 0 ||
      !MathIsValidNumber(equity) || !MathIsValidNumber(riskBudget) ||
      !MathIsValidNumber(pointCost) || !MathIsValidNumber(point) ||
      !MathIsValidNumber(volatility) || !MathIsValidNumber(volatilityRange) ||
      !MathIsValidNumber(maximumTakeProfit) || !MathIsValidNumber(maximumStopLoss))
      ReturnFalse;
   for(uint element = 0; element < iActionDimension; element++)
      if(!MathIsValidNumber(source[element]))
         ReturnFalse;
   const double scale = MathMax(volatility * volatilityRange, point);
   const double buySL = MathMax(double(source[2]), 0.0) * maximumStopLoss * point;
   const double sellSL = MathMax(double(source[5]), 0.0) * maximumStopLoss * point;
   const double buyTP = MathMax(double(source[1]), 0.0) * maximumTakeProfit * point;
   const double sellTP = MathMax(double(source[4]), 0.0) * maximumTakeProfit * point;
   destination[0] = float(MathMax(MathMin(double(source[0]) * buySL * pointCost / riskBudget, 1.0), 0.0));
   destination[1] = float(MathMax(MathMin(buyTP / scale, 1.0), 0.0));
   destination[2] = float(MathMax(MathMin(buySL / scale, 1.0), 0.0));
   destination[3] = float(MathMax(MathMin(double(source[3]) * sellSL * pointCost / riskBudget, 1.0), 0.0));
   destination[4] = float(MathMax(MathMin(sellTP / scale, 1.0), 0.0));
   destination[5] = float(MathMax(MathMin(sellSL / scale, 1.0), 0.0));
   return true;
  }
//+----------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory normalize terminal outcome  operation.  |
//+----------------------------------------------------------------------+
void CNeuronRAGMemory::NormalizeTerminalOutcome(const double reward, const double maxDrawdown,
      const double costs, const double riskUsage,
      const double duration, const double startingEquity,
      const double riskBudget, const double durationScale,
      SRAGTerminalOutcome &result) const
  {
   ClearTerminal(result);
   const bool equityValid = (MathIsValidNumber(startingEquity) && startingEquity > 0);
   const bool riskBudgetValid = (MathIsValidNumber(riskBudget) && riskBudget > 0);
   const bool durationScaleValid = (MathIsValidNumber(durationScale) && durationScale > 0);
   if(equityValid)
     {
      if(MathIsValidNumber(reward))
        {
         result.relative_return = float(reward / startingEquity);
         result.reward = result.relative_return;
        }
      if(MathIsValidNumber(maxDrawdown))
         result.maxDrawdown = float(MathMax(maxDrawdown, 0.0) / startingEquity);
      if(MathIsValidNumber(costs))
         result.costs = float(MathMax(costs, 0.0) / startingEquity);
     }
   if(riskBudgetValid && MathIsValidNumber(riskUsage))
      result.riskUsage = float(MathMax(MathMin(MathMax(riskUsage, 0.0) / riskBudget, 1.0), 0.0));
   if(durationScaleValid && MathIsValidNumber(duration))
      result.duration = float(MathMax(MathMin(MathMax(duration, 0.0) / durationScale, 1.0), 0.0));
  }
//+-------------------------------------------------------------------------+
//| Initializes the CNeuronRAGMemory layer and allocates its GPU  buffers.  |
//+-------------------------------------------------------------------------+
bool CNeuronRAGMemory::InitializeAction(SRAGActionCentroid &action, const uint scenarioId) const
  {
   action.used = false;
   action.scenarioId = scenarioId;
   action.hit_count = 0;
   action.mean_reward = 0.0f;
   ClearTerminal(action.mean_terminal);
   if(ArrayResize(action.centroid, (int)iActionDimension) != (int)iActionDimension)
      ReturnFalse;
   ArrayInitialize(action.centroid, 0.0f);
   return true;
  }
//+--------------------------------------------------------------------------+
//| Initializes the CNeuronRAGMemory layer and allocates its GPU   buffers.  |
//+--------------------------------------------------------------------------+
bool CNeuronRAGMemory::InitializeScenario(SRAGScenarioCentroid &scenario, const uint scenarioId) const
  {
   scenario.used = false;
   scenario.scenarioId = scenarioId;
   scenario.hit_count = 0;
   if(ArrayResize(scenario.centroid, (int)iScenarioEmbedding) != (int)iScenarioEmbedding)
      ReturnFalse;
   ArrayInitialize(scenario.centroid, 0.0f);
   if(ArrayResize(scenario.actions, (int)iActionCapacity) != (int)iActionCapacity)
      ReturnFalse;
   for(uint action = 0; action < iActionCapacity; action++)
      if(!InitializeAction(scenario.actions[action], scenarioId))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy action operation.              |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyAction(const SRAGActionCentroid &source,
                                  SRAGActionCentroid &destination) const
  {
   if(!ValidAction(source.centroid) ||
      ArrayResize(destination.centroid, (int)iActionDimension) != (int)iActionDimension)
      ReturnFalse;
   destination.used = source.used;
   destination.scenarioId = source.scenarioId;
   destination.hit_count = source.hit_count;
   destination.mean_reward = source.mean_reward;
   destination.mean_terminal = source.mean_terminal;
   ArrayCopy(destination.centroid, source.centroid);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy scenario operation.            |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyScenario(const SRAGScenarioCentroid &source,
                                    SRAGScenarioCentroid &destination) const
  {
   if(ArraySize(source.centroid) != (int)iScenarioEmbedding ||
      ArraySize(source.actions) != (int)iActionCapacity ||
      ArrayResize(destination.centroid, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(destination.actions, (int)iActionCapacity) != (int)iActionCapacity)
      ReturnFalse;
   destination.used = source.used;
   destination.scenarioId = source.scenarioId;
   destination.hit_count = source.hit_count;
   ArrayCopy(destination.centroid, source.centroid);
   for(uint action = 0; action < iActionCapacity; action++)
      if(!CopyAction(source.actions[action], destination.actions[action]))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy scenario flexible operation.   |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyScenarioFlexible(const SRAGScenarioCentroid &source,
      SRAGScenarioCentroid &destination) const
  {
   const int actionTotal = ArraySize(source.actions);
   if(ArraySize(source.centroid) != (int)iScenarioEmbedding ||
      actionTotal < (int)iActionCapacity ||
      ArrayResize(destination.centroid, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(destination.actions, actionTotal) != actionTotal)
      ReturnFalse;
   destination.used = source.used;
   destination.scenarioId = source.scenarioId;
   destination.hit_count = source.hit_count;
   ArrayCopy(destination.centroid, source.centroid);
   for(int action = 0; action < actionTotal; action++)
      if(!CopyAction(source.actions[action], destination.actions[action]))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory append action candidate operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::AppendActionCandidate(const uint scenarioId,
      SRAGScenarioCentroid &scenario,
      const SRAGActionCentroid &incoming) const
  {
   if(scenarioId >= iScenarioCapacity || !scenario.used || !incoming.used ||
      ArraySize(incoming.centroid) != (int)iActionDimension)
      ReturnFalse;
   const int current = ArraySize(scenario.actions);
   for(int action = 0; action < current; action++)
      if(!scenario.actions[action].used)
        {
         if(!CopyAction(incoming, scenario.actions[action]))
            ReturnFalse;
         scenario.actions[action].scenarioId = scenarioId;
         return true;
        }
   if(current < 0 || current >= RAG_MAX_CANDIDATES || ArrayResize(scenario.actions, current + 1) != current + 1 ||
      !InitializeAction(scenario.actions[current], scenarioId) ||
      !CopyAction(incoming, scenario.actions[current]))
      ReturnFalse;
   scenario.actions[current].scenarioId = scenarioId;
   return true;
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory compact action candidates  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronRAGMemory::CompactActionCandidates(SRAGScenarioCentroid &scenario) const
  {
   SRAGActionCentroid compacted[];
   if(ArrayResize(compacted, (int)iActionCapacity) != (int)iActionCapacity)
      ReturnFalse;
   for(uint action = 0; action < iActionCapacity; action++)
      if(!InitializeAction(compacted[action], scenario.scenarioId))
         ReturnFalse;
   uint destination = 0;
   for(int action = 0; action < ArraySize(scenario.actions); action++)
      if(scenario.actions[action].used)
        {
         if(destination >= iActionCapacity ||
            !CopyAction(scenario.actions[action], compacted[destination]))
            ReturnFalse;
         compacted[destination].scenarioId = scenario.scenarioId;
         destination++;
        }
   ArraySwap(scenario.actions, compacted);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory normalize l2 operation.             |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::NormalizeL2(const float &source[], float &destination[]) const
  {
   const int total = ArraySize(source);
   if(total <= 0 || ArrayResize(destination, total) != total)
      ReturnFalse;
   double squareSum = 0.0;
   for(int i = 0; i < total; i++)
     {
      if(!MathIsValidNumber(source[i]))
         ReturnFalse;
      squareSum += (double)source[i] * source[i];
     }
   if(!MathIsValidNumber(squareSum) || squareSum <= DBL_EPSILON)
      ReturnFalse;
   const double length = MathSqrt(squareSum);
   for(int i = 0; i < total; i++)
      destination[i] = (float)(source[i] / length);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory squared distance operation.         |
//+------------------------------------------------------------------+
float CNeuronRAGMemory::SquaredDistance(const float &left[], const float &right[]) const
  {
   const float maxFloat = RAG_MAX_FLOAT;
   if(ArraySize(left) != ArraySize(right) || ArraySize(left) <= 0)
      return maxFloat;
   float distance = 0.0f;
   for(int i = 0; i < ArraySize(left); i++)
     {
      if(!MathIsValidNumber(left[i]) || !MathIsValidNumber(right[i]))
         return maxFloat;
      const float delta = (float)(left[i] - right[i]);
      const float squared = (float)(delta * delta);
      distance = (float)(distance + squared);
     }
   return (MathIsValidNumber(distance) ? distance : maxFloat);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory valid action operation.             |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ValidAction(const float &action[]) const
  {
   if(ArraySize(action) != (int)iActionDimension)
      ReturnFalse;
   for(uint element = 0; element < iActionDimension; element++)
      if(!MathIsValidNumber(action[element]) || action[element] < 0.0f || action[element] > 1.0f)
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory valid terminal operation.           |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ValidTerminal(const SRAGTerminalOutcome &terminal) const
  {
   return (MathIsValidNumber(terminal.relative_return) &&
           MathIsValidNumber(terminal.maxDrawdown) &&
           MathIsValidNumber(terminal.costs) &&
           MathIsValidNumber(terminal.riskUsage) &&
           MathIsValidNumber(terminal.duration) &&
           MathIsValidNumber(terminal.reward) &&
           terminal.maxDrawdown >= 0.0f && terminal.costs >= 0.0f &&
           terminal.riskUsage >= 0.0f && terminal.riskUsage <= 1.0f &&
           terminal.duration >= 0.0f && terminal.duration <= 1.0f &&
           MathAbs(terminal.reward - terminal.relative_return) <= RAG_TERMINAL_EPSILON);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory merge terminal operation.           |
//+------------------------------------------------------------------+
void CNeuronRAGMemory::MergeTerminal(SRAGTerminalOutcome &destination,
                                     const ulong destinationHits,
                                     const SRAGTerminalOutcome &source,
                                     const ulong sourceHits) const
  {
   const double total = (double)destinationHits + (double)sourceHits;
   destination.relative_return = (float)((destination.relative_return * destinationHits +
                                          source.relative_return * sourceHits) / total);
   destination.maxDrawdown = (float)((destination.maxDrawdown * destinationHits +
                                      source.maxDrawdown * sourceHits) / total);
   destination.costs = (float)((destination.costs * destinationHits +
                                source.costs * sourceHits) / total);
   destination.riskUsage = (float)((destination.riskUsage * destinationHits +
                                    source.riskUsage * sourceHits) / total);
   destination.duration = (float)((destination.duration * destinationHits +
                                   source.duration * sourceHits) / total);
   destination.reward = (float)((destination.reward * destinationHits +
                                 source.reward * sourceHits) / total);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory first free scenario operation.      |
//+------------------------------------------------------------------+
int CNeuronRAGMemory::FirstFreeScenario() const
  {
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!scenarios[scenario].used)
         return (int)scenario;
   return -1;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory first free action operation.        |
//+------------------------------------------------------------------+
int CNeuronRAGMemory::FirstFreeAction(const uint scenarioId) const
  {
   if(scenarioId >= iScenarioCapacity || !scenarios[scenarioId].used)
      return -1;
   for(uint action = 0; action < iActionCapacity; action++)
      if(!scenarios[scenarioId].actions[action].used)
         return (int)action;
   return -1;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory rebind scenario actions operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::RebindScenarioActions(SRAGScenarioCentroid &scenario,
      const uint destinationScenarioId) const
  {
   scenario.scenarioId = destinationScenarioId;
   for(int action = 0; action < ArraySize(scenario.actions); action++)
      scenario.actions[action].scenarioId = destinationScenarioId;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory merge action candidates operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::MergeActionCandidates(const uint scenarioId,
      SRAGActionCentroid &destination,
      const SRAGActionCentroid &source) const
  {
   if(!destination.used || !source.used || destination.hit_count == 0 || source.hit_count == 0 ||
      !ValidAction(destination.centroid) || !ValidAction(source.centroid) ||
      !ValidTerminal(destination.mean_terminal) || !ValidTerminal(source.mean_terminal) ||
      !MathIsValidNumber(destination.mean_reward) || !MathIsValidNumber(source.mean_reward) ||
      destination.hit_count > ulong(-1) - source.hit_count)
      ReturnFalse;
   const ulong destinationHits = destination.hit_count;
   const ulong sourceHits = source.hit_count;
   const double total = (double)destinationHits + (double)sourceHits;
   for(uint element = 0; element < iActionDimension; element++)
      destination.centroid[element] = (float)((destination.centroid[element] * destinationHits +
                                              source.centroid[element] * sourceHits) / total);
   destination.mean_reward = (float)((destination.mean_reward * destinationHits +
                                      source.mean_reward * sourceHits) / total);
   MergeTerminal(destination.mean_terminal, destinationHits, source.mean_terminal, sourceHits);
   destination.hit_count = destinationHits + sourceHits;
   destination.used = true;
   destination.scenarioId = scenarioId;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory pair count safe operation.          |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::PairCountSafe(const uint candidateCount, int &pairCount) const
  {
   pairCount = 0;
   if(candidateCount < 2 || candidateCount > RAG_MAX_CANDIDATES)
      ReturnFalse;
   const ulong pairs = ((candidateCount & 1) == 0 ?
                        (ulong)(candidateCount / 2) * (ulong)(candidateCount - 1) :
                        (ulong)candidateCount * (ulong)((candidateCount - 1) / 2));
   if(pairs > (ulong)INT_MAX)
      ReturnFalse;
   pairCount = (int)pairs;
   return true;
  }
//--- A launch reduces every independent candidate row in parallel.  The device
//--- returns only one float4 per row; the immutable snapshot itself stays on the
//--- host until the count-weighted COW transaction has succeeded.
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory find nearest pairs gpu operation.   |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::FindNearestPairsGPU(const float &candidateCentroids[],
      const float &candidateUsed[],
      const uint candidateCount, const uint dimension,
      const uint batchCount, int &destinations[], int &sources[])
  {
   ArrayResize(destinations, 0);
   ArrayResize(sources, 0);
   int pairCount = 0;
   if(OpenCL == NULL || candidateCount < 2 || dimension == 0 || batchCount == 0 ||
      candidateCount > RAG_MAX_CANDIDATES || dimension > (uint)INT_MAX || batchCount > (uint)INT_MAX ||
      batchCount > UINT_MAX / 64 || batchCount > (uint)INT_MAX / 4 ||
      !PairCountSafe(candidateCount, pairCount) || pairCount <= 0 ||
      (ulong)candidateCount * (ulong)batchCount > ULONG_MAX / (ulong)dimension)
      ReturnFalse;
   const ulong rows = (ulong)candidateCount * (ulong)batchCount;
   const ulong values = rows * (ulong)dimension;
   if(values > (ulong)INT_MAX || rows > (ulong)INT_MAX)
      ReturnFalse;
   if(ArraySize(candidateCentroids) != (int)values || ArraySize(candidateUsed) != (int)rows)
      ReturnFalse;
   if(ArrayResize(destinations, (int)batchCount) != (int)batchCount ||
      ArrayResize(sources, (int)batchCount) != (int)batchCount)
      ReturnFalse;
   ArrayInitialize(destinations, -1);
   ArrayInitialize(sources, -1);
//--- BufferCreate reuses the allocation and performs BufferWrite when OpenCL
//--- and the shape are unchanged.  BufferInit frees only on a shape change.
   if(!cNearestCentroids.BufferInit((uint)values, 0.0f) ||
      !cNearestCentroids.AssignArray(candidateCentroids) ||
      !cNearestCentroids.BufferCreate(OpenCL) ||
      !cNearestUsed.BufferInit((uint)rows, 0.0f) ||
      !cNearestUsed.AssignArray(candidateUsed) || !cNearestUsed.BufferCreate(OpenCL) ||
      !cNearestResult.BufferInit(batchCount * 4, 0.0f) || !cNearestResult.BufferCreate(OpenCL))
      ReturnFalse;
   uint offset[] = {0};
   uint global[] = {batchCount * RAG_TOPK_LOCAL_WIDTH};
   uint local[] = {RAG_TOPK_LOCAL_WIDTH};
   if(!OpenCL.SetArgumentBuffer(def_k_RAGNearestPair, def_k_ragn_centroids, cNearestCentroids.GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_RAGNearestPair, def_k_ragn_used, cNearestUsed.GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_RAGNearestPair, def_k_ragn_result, cNearestResult.GetIndex()) ||
      !OpenCL.SetArgument(def_k_RAGNearestPair, def_k_ragn_candidate_count, candidateCount) ||
      !OpenCL.SetArgument(def_k_RAGNearestPair, def_k_ragn_dimension, dimension) ||
      !OpenCL.SetArgument(def_k_RAGNearestPair, def_k_ragn_batch_count, batchCount) ||
      !OpenCL.Execute(def_k_RAGNearestPair, 1, offset, global, local))
      ReturnFalse;
   float valuesOut[];
   if(ArrayResize(valuesOut, (int)(batchCount * 4)) != (int)(batchCount * 4) ||
      cNearestResult.GetData(valuesOut) != (int)(batchCount * 4))
      ReturnFalse;
   for(uint batch = 0; batch < batchCount; batch++)
     {
      const uint rowOffset = batch * 4;
      if(valuesOut[rowOffset + 3] < 0.5f || !MathIsValidNumber(valuesOut[rowOffset]))
         ReturnFalse;
      const int left = (int)(valuesOut[rowOffset + 1] + 0.5f);
      const int right = (int)(valuesOut[rowOffset + 2] + 0.5f);
      if(left < 0 || right <= left || right >= (int)candidateCount)
         ReturnFalse;
      destinations[batch] = left;
      sources[batch] = right;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory find nearest pair gpu operation.    |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::FindNearestPairGPU(const float &candidateCentroids[],
      const float &candidateUsed[],
      const uint candidateCount, const uint dimension,
      const uint batchCount, int &destination, int &source)
  {
   destination = -1;
   source = -1;
   int destinations[], sources[];
   if(!FindNearestPairsGPU(candidateCentroids, candidateUsed, candidateCount, dimension,
                           batchCount, destinations, sources) || ArraySize(destinations) < 1 ||
      ArraySize(sources) < 1)
      ReturnFalse;
   destination = destinations[0];
   source = sources[0];
   return true;
  }
//+-------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory find nearest action pair operation.  |
//+-------------------------------------------------------------------+
bool CNeuronRAGMemory::FindNearestActionPair(const uint scenarioId,
      const SRAGActionCentroid &actionCandidates[],
      int &destination, int &source)
  {
   destination = -1;
   source = -1;
   if(OpenCL != NULL)
     {
      const int candidates = ArraySize(actionCandidates);
      float flat[], used[];
      if(candidates < 2 || ArrayResize(flat, candidates * (int)iActionDimension) != candidates * (int)iActionDimension ||
         ArrayResize(used, candidates) != candidates)
         ReturnFalse;
      for(int action = 0; action < candidates; action++)
        {
         used[action] = (actionCandidates[action].used ? 1.0f : 0.0f);
         ArrayCopy(flat, actionCandidates[action].centroid, action * (int)iActionDimension, 0, (int)iActionDimension);
        }
      return (scenarioId < iScenarioCapacity && FindNearestPairGPU(flat, used, (uint)candidates,
              iActionDimension, 1, destination, source));
     }
   const float maxFloat = RAG_MAX_FLOAT;
   float nearest = maxFloat;
   bool found = false;
   for(int left = 0; left < ArraySize(actionCandidates); left++)
     {
      if(!actionCandidates[left].used)
         continue;
      for(int right = left + 1; right < ArraySize(actionCandidates); right++)
        {
         if(!actionCandidates[right].used)
            continue;
         const float distance = SquaredDistance(actionCandidates[left].centroid,
                                                actionCandidates[right].centroid);
         if(!found || distance < nearest)
           {
            nearest = distance;
            destination = left;
            source = right;
            found = true;
           }
        }
     }
   return (scenarioId < iScenarioCapacity && destination >= 0 && source >= 0);
  }
//+-------------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory consolidate action candidates  operation.  |
//+-------------------------------------------------------------------------+
bool CNeuronRAGMemory::ConsolidateActionCandidates(SRAGScenarioCentroid &candidateSnapshot[])
  {
   if(ArraySize(candidateSnapshot) != (int)iScenarioCapacity)
      ReturnFalse;
   while(true)
     {
      int activeScenarios[];
      int maxCandidates = 0;
      for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
        {
         if(!candidateSnapshot[scenario].used)
            continue;
         int usedActions = 0;
         const int candidates = ArraySize(candidateSnapshot[scenario].actions);
         if(candidates < 0 || candidates > 65536)
            ReturnFalse;
         for(int action = 0; action < candidates; action++)
            if(candidateSnapshot[scenario].actions[action].used)
               usedActions++;
         if(usedActions > (int)iActionCapacity)
           {
            const int active = ArraySize(activeScenarios);
            if(ArrayResize(activeScenarios, active + 1) != active + 1)
               ReturnFalse;
            activeScenarios[active] = (int)scenario;
            if(candidates > maxCandidates)
               maxCandidates = candidates;
           }
        }
      const uint batchCount = (uint)ArraySize(activeScenarios);
      if(batchCount == 0)
         break;
      if(OpenCL == NULL)
        {
         for(uint batch = 0; batch < batchCount; batch++)
           {
            const int scenario = activeScenarios[batch];
            int destination = -1, source = -1;
            if(!FindNearestActionPair((uint)scenario, candidateSnapshot[scenario].actions,
                                      destination, source) ||
               !MergeActionCandidates((uint)scenario,
                                      candidateSnapshot[scenario].actions[destination],
                                      candidateSnapshot[scenario].actions[source]) ||
               !InitializeAction(candidateSnapshot[scenario].actions[source], (uint)scenario))
               ReturnFalse;
           }
         continue;
        }
      const ulong rows = (ulong)batchCount * (ulong)maxCandidates;
      if(maxCandidates < 2 || rows > (ulong)INT_MAX ||
         rows > ULONG_MAX / (ulong)iActionDimension ||
         rows * (ulong)iActionDimension > (ulong)INT_MAX)
         ReturnFalse;
      const int values = (int)(rows * (ulong)iActionDimension);
      float flat[], used[];
      if(ArrayResize(flat, values) != values || ArrayResize(used, (int)rows) != (int)rows)
         ReturnFalse;
      ArrayInitialize(flat, 0.0f);
      ArrayInitialize(used, 0.0f);
      for(uint batch = 0; batch < batchCount; batch++)
        {
         const int scenario = activeScenarios[batch];
         for(int action = 0; action < ArraySize(candidateSnapshot[scenario].actions); action++)
           {
            if(!candidateSnapshot[scenario].actions[action].used)
               continue;
            const int slot = (int)batch * maxCandidates + action;
            used[slot] = 1.0f;
            ArrayCopy(flat, candidateSnapshot[scenario].actions[action].centroid,
                      slot * (int)iActionDimension, 0, (int)iActionDimension);
           }
        }
      int destinations[], sources[];
      if(!FindNearestPairsGPU(flat, used, (uint)maxCandidates, iActionDimension, batchCount,
                              destinations, sources) ||
         ArraySize(destinations) != (int)batchCount || ArraySize(sources) != (int)batchCount)
         ReturnFalse;
      for(uint batch = 0; batch < batchCount; batch++)
        {
         const int scenario = activeScenarios[batch];
         const int destination = destinations[batch];
         const int source = sources[batch];
         if(destination < 0 || source <= destination ||
            source >= ArraySize(candidateSnapshot[scenario].actions) ||
            !candidateSnapshot[scenario].actions[destination].used ||
            !candidateSnapshot[scenario].actions[source].used ||
            !MergeActionCandidates((uint)scenario,
                                   candidateSnapshot[scenario].actions[destination],
                                   candidateSnapshot[scenario].actions[source]) ||
            !InitializeAction(candidateSnapshot[scenario].actions[source], (uint)scenario))
            ReturnFalse;
        }
     }
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!CompactActionCandidates(candidateSnapshot[scenario]) ||
         !RebindScenarioActions(candidateSnapshot[scenario], scenario))
         ReturnFalse;
   return true;
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory merge scenario candidates  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronRAGMemory::MergeScenarioCandidates(SRAGScenarioCentroid &destination,
      const SRAGScenarioCentroid &source)
  {
   if(!destination.used || !source.used || destination.hit_count == 0 || source.hit_count == 0 ||
      ArraySize(destination.centroid) != (int)iScenarioEmbedding ||
      ArraySize(source.centroid) != (int)iScenarioEmbedding ||
      destination.hit_count > ulong(-1) - source.hit_count)
      ReturnFalse;
   const ulong destinationHits = destination.hit_count;
   const ulong sourceHits = source.hit_count;
   float previousCentroid[];
   if(ArrayResize(previousCentroid, (int)iScenarioEmbedding) != (int)iScenarioEmbedding)
      ReturnFalse;
   ArrayCopy(previousCentroid, destination.centroid);
   float weighted[];
   if(ArrayResize(weighted, (int)iScenarioEmbedding) != (int)iScenarioEmbedding)
      ReturnFalse;
   for(uint element = 0; element < iScenarioEmbedding; element++)
      weighted[element] = (float)(destination.centroid[element] * destinationHits +
                                  source.centroid[element] * sourceHits);
//--- Equal and opposite normalized observations have no unique normalized
//--- mean. Retain the prior valid centroid while still merging their facts.
   if(!NormalizeL2(weighted, destination.centroid))
      ArrayCopy(destination.centroid, previousCentroid);
   destination.hit_count = destinationHits + sourceHits;
   for(int action = 0; action < ArraySize(source.actions); action++)
      if(source.actions[action].used &&
         !AppendActionCandidate(destination.scenarioId, destination, source.actions[action]))
         ReturnFalse;
   return RebindScenarioActions(destination, destination.scenarioId);
  }
//+----------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory find nearest scenario pair  operation.  |
//+----------------------------------------------------------------------+
bool CNeuronRAGMemory::FindNearestScenarioPair(const SRAGScenarioCentroid &scenarioCandidates[],
      int &destination, int &source)
  {
   destination = -1;
   source = -1;
   if(OpenCL != NULL)
     {
      const int candidates = ArraySize(scenarioCandidates);
      float flat[], used[];
      if(candidates < 2 || ArrayResize(flat, candidates * (int)iScenarioEmbedding) != candidates * (int)iScenarioEmbedding ||
         ArrayResize(used, candidates) != candidates)
         ReturnFalse;
      for(int scenario = 0; scenario < candidates; scenario++)
        {
         used[scenario] = (scenarioCandidates[scenario].used ? 1.0f : 0.0f);
         ArrayCopy(flat, scenarioCandidates[scenario].centroid, scenario * (int)iScenarioEmbedding, 0,
                   (int)iScenarioEmbedding);
        }
      return FindNearestPairGPU(flat, used, (uint)candidates, iScenarioEmbedding, 1, destination, source);
     }
   const float maxFloat = RAG_MAX_FLOAT;
   float nearest = maxFloat;
   bool found = false;
   for(int left = 0; left < ArraySize(scenarioCandidates); left++)
     {
      if(!scenarioCandidates[left].used)
         continue;
      for(int right = left + 1; right < ArraySize(scenarioCandidates); right++)
        {
         if(!scenarioCandidates[right].used)
            continue;
         const float distance = SquaredDistance(scenarioCandidates[left].centroid,
                                                scenarioCandidates[right].centroid);
         if(!found || distance < nearest)
           {
            nearest = distance;
            destination = left;
            source = right;
            found = true;
           }
        }
     }
   return (destination >= 0 && source >= 0);
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory stage scenario candidates  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronRAGMemory::StageScenarioCandidates(const uint sourceScenarioId,
      const float &scenarioEmbedding[],
      const float &action[],
      const SRAGTerminalOutcome &terminal,
      SRAGScenarioCentroid &scenarioCandidates[])
  {
   if(ArraySize(scenarioEmbedding) != (int)iScenarioEmbedding ||
      !ValidAction(action) || !ValidTerminal(terminal) ||
      ArrayResize(scenarioCandidates, iScenarioCapacity + 1) != (int)(iScenarioCapacity + 1))
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!CopyScenario(scenarios[scenario], scenarioCandidates[scenario]))
         ReturnFalse;
   if(!InitializeScenario(scenarioCandidates[iScenarioCapacity], sourceScenarioId) ||
      !NormalizeL2(scenarioEmbedding, scenarioCandidates[iScenarioCapacity].centroid))
      ReturnFalse;
   scenarioCandidates[iScenarioCapacity].used = true;
   scenarioCandidates[iScenarioCapacity].hit_count = 1;
   scenarioCandidates[iScenarioCapacity].actions[0].used = true;
   scenarioCandidates[iScenarioCapacity].actions[0].scenarioId = sourceScenarioId;
   scenarioCandidates[iScenarioCapacity].actions[0].hit_count = 1;
   scenarioCandidates[iScenarioCapacity].actions[0].mean_reward = terminal.reward;
   scenarioCandidates[iScenarioCapacity].actions[0].mean_terminal = terminal;
   ArrayCopy(scenarioCandidates[iScenarioCapacity].actions[0].centroid, action);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory add completed scenario operation.   |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::AddCompletedScenario(const uint sourceScenarioId,
      const float &scenarioEmbedding[],
      const float &action[],
      const SRAGTerminalOutcome &terminal)
  {
   SRAGScenarioCentroid candidateSnapshot[];
   uint candidateCount = iScenarioCount;
   if(!CopySnapshot(scenarios, candidateSnapshot) ||
      !ApplyCompletedRecordToSnapshot(candidateSnapshot, candidateCount, sourceScenarioId,
                                      scenarioEmbedding, action, terminal) ||
      !ConsolidateActionCandidates(candidateSnapshot))
      ReturnFalse;
   ArraySwap(scenarios, candidateSnapshot);
   iScenarioCount = candidateCount;
   return true;
  }
//+------------------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory apply completed record to snapshot  operation.  |
//+------------------------------------------------------------------------------+
bool CNeuronRAGMemory::ApplyCompletedRecordToSnapshot(SRAGScenarioCentroid &candidateSnapshot[],
      uint &candidateCount,
      const uint sourceScenarioId,
      const float &scenarioEmbedding[],
      const float &action[],
      const SRAGTerminalOutcome &terminal)
  {
   if(ArraySize(candidateSnapshot) != (int)iScenarioCapacity || candidateCount > iScenarioCapacity ||
      ArraySize(scenarioEmbedding) != (int)iScenarioEmbedding ||
      !ValidAction(action) || !ValidTerminal(terminal))
      ReturnFalse;
   float normalized[];
   if(!NormalizeL2(scenarioEmbedding, normalized))
      ReturnFalse;
   int freeScenario = -1;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!candidateSnapshot[scenario].used)
        {
         freeScenario = (int)scenario;
         break;
        }
   if(freeScenario >= 0)
     {
      SRAGScenarioCentroid incoming;
      if(!InitializeScenario(incoming, sourceScenarioId))
         ReturnFalse;
      incoming.used = true;
      incoming.hit_count = 1;
      ArrayCopy(incoming.centroid, normalized);
      incoming.actions[0].used = true;
      incoming.actions[0].scenarioId = sourceScenarioId;
      incoming.actions[0].hit_count = 1;
      incoming.actions[0].mean_reward = terminal.reward;
      incoming.actions[0].mean_terminal = terminal;
      ArrayCopy(incoming.actions[0].centroid, action);
      RebindScenarioActions(incoming, (uint)freeScenario);
      if(!CopyScenario(incoming, candidateSnapshot[freeScenario]))
         ReturnFalse;
      candidateCount++;
      return true;
     }
   SRAGScenarioCentroid scenarioCandidates[];
   if(ArrayResize(scenarioCandidates, (int)(iScenarioCapacity + 1)) != (int)(iScenarioCapacity + 1))
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!CopyScenarioFlexible(candidateSnapshot[scenario], scenarioCandidates[scenario]))
         ReturnFalse;
   if(!InitializeScenario(scenarioCandidates[iScenarioCapacity], sourceScenarioId))
      ReturnFalse;
   scenarioCandidates[iScenarioCapacity].used = true;
   scenarioCandidates[iScenarioCapacity].hit_count = 1;
   ArrayCopy(scenarioCandidates[iScenarioCapacity].centroid, normalized);
   scenarioCandidates[iScenarioCapacity].actions[0].used = true;
   scenarioCandidates[iScenarioCapacity].actions[0].scenarioId = sourceScenarioId;
   scenarioCandidates[iScenarioCapacity].actions[0].hit_count = 1;
   scenarioCandidates[iScenarioCapacity].actions[0].mean_reward = terminal.reward;
   scenarioCandidates[iScenarioCapacity].actions[0].mean_terminal = terminal;
   ArrayCopy(scenarioCandidates[iScenarioCapacity].actions[0].centroid, action);
   int destination = -1;
   int source = -1;
   if(!FindNearestScenarioPair(scenarioCandidates, destination, source) ||
      !MergeScenarioCandidates(scenarioCandidates[destination], scenarioCandidates[source]))
      ReturnFalse;
   RebindScenarioActions(scenarioCandidates[destination], (uint)destination);
   if(!CopyScenarioFlexible(scenarioCandidates[destination], candidateSnapshot[destination]))
      ReturnFalse;
   if(source == (int)iScenarioCapacity)
      return true;
   RebindScenarioActions(scenarioCandidates[iScenarioCapacity], (uint)source);
   if(!CopyScenarioFlexible(scenarioCandidates[iScenarioCapacity], candidateSnapshot[source]))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory validate snapshot operation.        |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ValidateSnapshot(const SRAGScenarioCentroid &snapshot[],
                                        const uint scenarioCount,
                                        const uint scenarioEmbedding,
                                        const uint actionDimension,
                                        const uint scenarioCapacity,
                                        const uint actionCapacity) const
  {
   if(scenarioEmbedding == 0 || actionDimension == 0 || scenarioCapacity == 0 || actionCapacity == 0 ||
      scenarioCapacity > (uint)INT_MAX || actionCapacity > (uint)INT_MAX ||
      ArraySize(snapshot) != (int)scenarioCapacity || scenarioCount > scenarioCapacity)
      ReturnFalse;
   uint usedScenarios = 0;
   for(uint scenario = 0; scenario < scenarioCapacity; scenario++)
     {
      if(snapshot[scenario].scenarioId != scenario ||
         ArraySize(snapshot[scenario].centroid) != (int)scenarioEmbedding ||
         ArraySize(snapshot[scenario].actions) != (int)actionCapacity)
         ReturnFalse;
      double norm = 0.0;
      for(uint element = 0; element < scenarioEmbedding; element++)
        {
         if(!MathIsValidNumber(snapshot[scenario].centroid[element]))
            ReturnFalse;
         norm += (double)snapshot[scenario].centroid[element] * snapshot[scenario].centroid[element];
        }
      if(snapshot[scenario].used)
        {
         if(snapshot[scenario].hit_count == 0 || norm < RAG_NORM_MIN || norm > RAG_NORM_MAX)
            ReturnFalse;
         usedScenarios++;
        }
      else
         if(snapshot[scenario].hit_count != 0 || norm > DBL_EPSILON)
            ReturnFalse;
      for(uint action = 0; action < actionCapacity; action++)
        {
         if(snapshot[scenario].actions[action].scenarioId != scenario ||
            ArraySize(snapshot[scenario].actions[action].centroid) != (int)actionDimension ||
            !MathIsValidNumber(snapshot[scenario].actions[action].mean_reward) ||
            !ValidTerminal(snapshot[scenario].actions[action].mean_terminal))
            ReturnFalse;
         for(uint element = 0; element < actionDimension; element++)
            if(!MathIsValidNumber(snapshot[scenario].actions[action].centroid[element]) ||
               snapshot[scenario].actions[action].centroid[element] < 0.0f ||
               snapshot[scenario].actions[action].centroid[element] > 1.0f)
               ReturnFalse;
         if(snapshot[scenario].actions[action].used &&
            (!snapshot[scenario].used || snapshot[scenario].actions[action].hit_count == 0))
            ReturnFalse;
         if(!snapshot[scenario].actions[action].used &&
            snapshot[scenario].actions[action].hit_count != 0)
            ReturnFalse;
        }
     }
   return (usedScenarios == scenarioCount);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory commit snapshot operation.          |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CommitSnapshot(const SRAGScenarioCentroid &snapshot[],
                                      const uint scenarioCount)
  {
   SRAGScenarioCentroid committed[];
   if(ArrayResize(committed, (int)iScenarioCapacity) != (int)iScenarioCapacity)
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!CopyScenario(snapshot[scenario], committed[scenario]))
         ReturnFalse;
   ArraySwap(scenarios, committed);
   iScenarioCount = scenarioCount;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy completed record operation.    |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyCompletedRecord(const SRAGCompletedRecord &source,
      SRAGCompletedRecord &destination) const
  {
   if(ArraySize(source.scenarioEmbedding) != (int)iScenarioEmbedding ||
      !ValidAction(source.action) || !ValidTerminal(source.terminal) ||
      ArrayResize(destination.scenarioEmbedding, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(destination.action, (int)iActionDimension) != (int)iActionDimension)
      ReturnFalse;
   ArrayCopy(destination.scenarioEmbedding, source.scenarioEmbedding);
   ArrayCopy(destination.action, source.action);
   destination.terminal = source.terminal;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy pending event operation.       |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyPendingEvent(const SRAGPendingEvent &source,
                                        SRAGPendingEvent &destination) const
  {
   if(!ValidAction(source.action) ||
      ArrayResize(destination.action, (int)iActionDimension) != (int)iActionDimension)
      ReturnFalse;
   for(uint element = 0; element < iActionDimension; element++)
      if(!MathIsValidNumber(source.action[element]))
         ReturnFalse;
   ArrayCopy(destination.action, source.action);
   destination.confirmed = source.confirmed;
   destination.deal = source.deal;
   destination.position = source.position;
   return (!destination.confirmed || (destination.deal >= 0 && destination.position > 0));
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy pending fact operation.        |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopyPendingFact(const SRAGPendingFact &source,
                                       SRAGPendingFact &destination) const
  {
   const int eventCount = ArraySize(source.events);
   if(source.id == 0 || eventCount <= 0 || eventCount >= INT_MAX ||
      ArraySize(source.scenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(destination.scenarioEmbedding, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(destination.events, eventCount) != eventCount)
      ReturnFalse;
   for(uint element = 0; element < iScenarioEmbedding; element++)
      if(!MathIsValidNumber(source.scenarioEmbedding[element]))
         ReturnFalse;
   ArrayCopy(destination.scenarioEmbedding, source.scenarioEmbedding);
   destination.id = source.id;
   for(int event = 0; event < eventCount; event++)
      if(!CopyPendingEvent(source.events[event], destination.events[event]))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory find pending fact operation.        |
//+------------------------------------------------------------------+
int CNeuronRAGMemory::FindPendingFact(const ulong pendingId) const
  {
   if(pendingId == 0)
      return -1;
   for(int fact = 0; fact < ArraySize(pending_facts); fact++)
      if(pending_facts[fact].id == pendingId)
         return fact;
   return -1;
  }
//+-------------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory build remaining pending facts  operation.  |
//+-------------------------------------------------------------------------+
bool CNeuronRAGMemory::BuildRemainingPendingFacts(const int excluded,
      SRAGPendingFact &remaining[]) const
  {
   const int total = ArraySize(pending_facts);
   if(excluded < 0 || excluded >= total || ArrayResize(remaining, total - 1) != total - 1)
      ReturnFalse;
   int destination = 0;
   for(int source = 0; source < total; source++)
     {
      if(source == excluded)
         continue;
      if(!CopyPendingFact(pending_facts[source], remaining[destination]))
         ReturnFalse;
      destination++;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory copy snapshot operation.            |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CopySnapshot(const SRAGScenarioCentroid &source[],
                                    SRAGScenarioCentroid &destination[]) const
  {
   if(ArraySize(source) != (int)iScenarioCapacity ||
      ArrayResize(destination, (int)iScenarioCapacity) != (int)iScenarioCapacity)
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!CopyScenario(source[scenario], destination[scenario]))
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Returns whether hash value CNeuronRAGMemory.                     |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::HashValue(const ulong hash, const ulong value) const
  {
   return (hash ^ value) * RAG_FNV_PRIME;
  }
//+------------------------------------------------------------------+
//| Returns whether hash float bits CNeuronRAGMemory.                |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::HashFloatBits(const ulong hash, const float value) const
  {
   URAGFloatBits raw;
   raw.value = value;
   return HashValue(hash, raw.bits);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory snapshot fingerprint operation.     |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::SnapshotFingerprint(const SRAGScenarioCentroid &snapshot[],
      const uint scenarioCount) const
  {
   ulong hash = RAG_FNV_OFFSET;
   hash = HashValue(hash, iFormatVersion);
   hash = HashValue(hash, iScenarioEmbedding);
   hash = HashValue(hash, iActionDimension);
   hash = HashValue(hash, iTokenWidth);
   hash = HashValue(hash, iScenarioCapacity);
   hash = HashValue(hash, iActionCapacity);
   hash = HashValue(hash, iTopK);
   hash = HashValue(hash, iOnlineMemorySize);
   hash = HashValue(hash, scenarioCount);
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
     {
      hash = HashValue(hash, (snapshot[scenario].used ? 1 : 0));
      hash = HashValue(hash, snapshot[scenario].scenarioId);
      hash = HashValue(hash, snapshot[scenario].hit_count);
      for(uint element = 0; element < iScenarioEmbedding; element++)
         hash = HashFloatBits(hash, snapshot[scenario].centroid[element]);
      for(uint action = 0; action < iActionCapacity; action++)
        {
         hash = HashValue(hash, (snapshot[scenario].actions[action].used ? 1 : 0));
         hash = HashValue(hash, snapshot[scenario].actions[action].scenarioId);
         hash = HashValue(hash, snapshot[scenario].actions[action].hit_count);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_reward);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.relative_return);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.maxDrawdown);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.costs);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.riskUsage);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.duration);
         hash = HashFloatBits(hash, snapshot[scenario].actions[action].mean_terminal.reward);
         for(uint element = 0; element < iActionDimension; element++)
            hash = HashFloatBits(hash, snapshot[scenario].actions[action].centroid[element]);
        }
     }
   return (hash == 0 ? 1 : hash);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory build inference set operation.      |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::BuildInferenceSet(const SRAGScenarioCentroid &snapshot[],
      const ulong fingerprint,
      CRAGInferenceSet *&result) const
  {
   result = NULL;
   if(OpenCL == NULL || ArraySize(snapshot) != (int)iScenarioCapacity)
      ReturnFalse;
   const ulong centroidTotal = (ulong)iScenarioCapacity * iScenarioEmbedding;
   const ulong actionSlots = (ulong)iScenarioCapacity * iActionCapacity;
   if(actionSlots > ULONG_MAX / iActionDimension)
      ReturnFalse;
   const ulong actionTotal = actionSlots * iActionDimension;
   const uint partialCount = (iScenarioCapacity + RAG_TOPK_LOCAL_WIDTH - 1) / RAG_TOPK_LOCAL_WIDTH;
   const ulong partialTotal = (ulong)partialCount * iTopK * 2;
   const ulong topTotal = (ulong)iTopK * 2;
   const ulong topActions = (ulong)iTopK * iActionCapacity;
   if(topActions > ULONG_MAX / iTokenWidth)
      ReturnFalse;
   const ulong tokenTotal = topActions * iTokenWidth;
   if(centroidTotal > (ulong)INT_MAX || actionSlots > (ulong)INT_MAX ||
      actionTotal > (ulong)INT_MAX || partialTotal > (ulong)INT_MAX ||
      topTotal > (ulong)INT_MAX || tokenTotal > (ulong)INT_MAX)
      ReturnFalse;
   float flatCentroids[], flatUsed[], flatActions[], flatActionUsed[];
   float flatActionMeanReward[], flatActionHitCount[];
   if(ArrayResize(flatCentroids, (int)centroidTotal) != (int)centroidTotal ||
      ArrayResize(flatUsed, (int)iScenarioCapacity) != (int)iScenarioCapacity ||
      ArrayResize(flatActions, (int)actionTotal) != (int)actionTotal ||
      ArrayResize(flatActionUsed, (int)actionSlots) != (int)actionSlots ||
      ArrayResize(flatActionMeanReward, (int)actionSlots) != (int)actionSlots ||
      ArrayResize(flatActionHitCount, (int)actionSlots) != (int)actionSlots)
      ReturnFalse;
   ArrayInitialize(flatCentroids, 0.0f);
   ArrayInitialize(flatUsed, 0.0f);
   ArrayInitialize(flatActions, 0.0f);
   ArrayInitialize(flatActionUsed, 0.0f);
   ArrayInitialize(flatActionMeanReward, 0.0f);
   ArrayInitialize(flatActionHitCount, 0.0f);
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
     {
      if(!snapshot[scenario].used)
         continue;
      flatUsed[scenario] = 1.0f;
      for(uint element = 0; element < iScenarioEmbedding; element++)
         flatCentroids[scenario * iScenarioEmbedding + element] = snapshot[scenario].centroid[element];
      for(uint action = 0; action < iActionCapacity; action++)
        {
         if(!snapshot[scenario].actions[action].used)
            continue;
         const uint slot = scenario * iActionCapacity + action;
         flatActionUsed[slot] = 1.0f;
         flatActionMeanReward[slot] = snapshot[scenario].actions[action].mean_reward;
         flatActionHitCount[slot] = float(snapshot[scenario].actions[action].hit_count);
         for(uint element = 0; element < iActionDimension; element++)
            flatActions[slot * iActionDimension + element] = snapshot[scenario].actions[action].centroid[element];
        }
     }
   CRAGInferenceSet *candidate = new CRAGInferenceSet();
   if(candidate == NULL)
      ReturnFalse;
   candidate.opencl = OpenCL;
   candidate.fingerprint = fingerprint;
   candidate.centroids = new CBufferFloat();
   candidate.used = new CBufferFloat();
   candidate.scores = new CBufferFloat();
   candidate.partial = new CBufferFloat();
   candidate.merge = new CBufferFloat();
   candidate.top = new CBufferFloat();
   candidate.actions = new CBufferFloat();
   candidate.actionUsed = new CBufferFloat();
   candidate.actionMeanReward = new CBufferFloat();
   candidate.actionHitCount = new CBufferFloat();
   candidate.absoluteRelevance = new CBufferFloat();
   candidate.tokens = new CBufferFloat();
   if(candidate.centroids == NULL || candidate.used == NULL || candidate.scores == NULL ||
      candidate.partial == NULL || candidate.merge == NULL || candidate.top == NULL || candidate.actions == NULL ||
      candidate.actionUsed == NULL || candidate.actionMeanReward == NULL ||
      candidate.actionHitCount == NULL || candidate.absoluteRelevance == NULL || candidate.tokens == NULL ||
      !candidate.centroids.AssignArray(flatCentroids) || !candidate.centroids.BufferCreate(OpenCL) ||
      !candidate.used.AssignArray(flatUsed) || !candidate.used.BufferCreate(OpenCL) ||
      !candidate.actions.AssignArray(flatActions) || !candidate.actions.BufferCreate(OpenCL) ||
      !candidate.actionUsed.AssignArray(flatActionUsed) || !candidate.actionUsed.BufferCreate(OpenCL) ||
      !candidate.actionMeanReward.AssignArray(flatActionMeanReward) ||
      !candidate.actionMeanReward.BufferCreate(OpenCL) ||
      !candidate.actionHitCount.AssignArray(flatActionHitCount) ||
      !candidate.actionHitCount.BufferCreate(OpenCL) ||
      !candidate.absoluteRelevance.BufferInit(1, 0.0f) ||
      !candidate.absoluteRelevance.BufferCreate(OpenCL) ||
      !candidate.scores.BufferInit(iScenarioCapacity, 0.0f) || !candidate.scores.BufferCreate(OpenCL) ||
      !candidate.partial.BufferInit((uint)partialTotal, 0.0f) || !candidate.partial.BufferCreate(OpenCL) ||
      !candidate.merge.BufferInit((uint)partialTotal, 0.0f) || !candidate.merge.BufferCreate(OpenCL) ||
      !candidate.top.BufferInit((uint)topTotal, 0.0f) || !candidate.top.BufferCreate(OpenCL) ||
      !candidate.tokens.BufferInit((uint)tokenTotal, 0.0f) || !candidate.tokens.BufferCreate(OpenCL))
     {
      delete candidate;
      ReturnFalse;
     }
   result = candidate;
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Initializes the CNeuronRAGMemory layer and allocates its GPU  buffers (Init).  |
//+--------------------------------------------------------------------------------+
bool CNeuronRAGMemory::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            CLayerDescription &description)
  {
   if(!Configure(description))
      ReturnFalse;
   if(iTopK > UINT_MAX / iActionCapacity)
      ReturnFalse;
   const uint tokenCount = iTopK * iActionCapacity;
   if(tokenCount == 0 || iTokenWidth > UINT_MAX / tokenCount)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, tokenCount * iTokenWidth,
                            description.optimization, description.batch))
      ReturnFalse;
   if(ArrayResize(scenarios, (int)iScenarioCapacity) != (int)iScenarioCapacity)
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!InitializeScenario(scenarios[scenario], scenario))
         ReturnFalse;
   iScenarioCount = 0;
   return Clear();
  }
//+------------------------------------------------------------------+
//| Sets the online memory size of the CNeuronRAGMemory.             |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::SetOnlineMemorySize(const uint value)
  {
   if(value >= (uint)INT_MAX || (value > 0 && value < (uint)ArraySize(pending_records)))
      ReturnFalse;
   iOnlineMemorySize = value;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory bind null inference operation.      |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::BindNullInference(void)
  {
   SRAGScenarioCentroid empty[];
   if(ArrayResize(empty, (int)iScenarioCapacity) != (int)iScenarioCapacity)
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
      if(!InitializeScenario(empty[scenario], scenario))
         ReturnFalse;
   CRAGInferenceSet *candidate = NULL;
   if(!BuildInferenceSet(empty, 0, candidate))
      ReturnFalse;
   CRAGInferenceSet *previous = m_inference;
   m_inference = candidate;
   if(previous != NULL)
      delete previous;
   return true;
  }
//+------------------------------------------------------------------+
//| Read and sanitize a model-owned scenario embedding from OpenCL.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ReadScenarioEmbedding(CBufferFloat *source,
      float &destination[], uint &replaced) const
  {
   replaced = 0;
   if(source == NULL)
      ReturnFalse;
   if(OpenCL == NULL)
      ReturnFalse;
   if(source.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(source.GetIndex() == INVALID_HANDLE)
      ReturnFalse;
   if(source.Total() != (int)iScenarioEmbedding)
      ReturnFalse;
   if(ArrayResize(destination, (int)iScenarioEmbedding) != (int)iScenarioEmbedding)
      ReturnFalse;
   if(source.GetData(destination) != (int)iScenarioEmbedding)
      ReturnFalse;
   double squareSum = 0.0;
   for(uint element = 0; element < iScenarioEmbedding; element++)
     {
      if(!MathIsValidNumber(destination[element]))
        {
         destination[element] = 0.0f;
         replaced++;
        }
      squareSum += (double)destination[element] * destination[element];
     }
   if(!MathIsValidNumber(squareSum))
      ReturnFalse;
   if(squareSum <= DBL_EPSILON)
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory add completed record operation.     |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::AddCompletedRecord(const float &scenarioEmbedding[],
      const float &action[],
      const SRAGTerminalOutcome &terminal)
  {
   const uint limit = (iOnlineMemorySize > 0 ? iOnlineMemorySize : iScenarioCapacity);
   const int total = ArraySize(pending_records);
   if(limit == 0 || total >= (int)limit || ArraySize(scenarioEmbedding) != (int)iScenarioEmbedding ||
      !ValidAction(action) || !ValidTerminal(terminal) ||
      ArrayResize(pending_records, total + 1) != total + 1)
      ReturnFalse;
   SRAGCompletedRecord source;
   if(ArrayResize(source.scenarioEmbedding, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(source.action, (int)iActionDimension) != (int)iActionDimension)
     {
      ArrayResize(pending_records, total);
      ReturnFalse;
     }
   ArrayCopy(source.scenarioEmbedding, scenarioEmbedding);
   ArrayCopy(source.action, action);
   source.terminal = terminal;
   if(!CopyCompletedRecord(source, pending_records[total]))
     {
      ArrayResize(pending_records, total);
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory open pending operation.             |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::OpenPending(const float &scenarioEmbedding[],
                                    const float &action[])
  {
   const int total = ArraySize(pending_facts);
   if(iNextPendingId == 0 || iNextPendingId == ULONG_MAX || total >= INT_MAX - 1 ||
      ArraySize(scenarioEmbedding) != (int)iScenarioEmbedding || !ValidAction(action))
      return 0;
   const ulong pendingId = iNextPendingId;
   SRAGPendingFact source;
   source.id = pendingId;
   if(ArrayResize(source.scenarioEmbedding, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
      ArrayResize(source.events, 1) != 1 ||
      ArrayResize(source.events[0].action, (int)iActionDimension) != (int)iActionDimension)
      return 0;
   ArrayCopy(source.scenarioEmbedding, scenarioEmbedding);
   ArrayCopy(source.events[0].action, action);
   source.events[0].confirmed = false;
   source.events[0].deal = 0;
   source.events[0].position = 0;
   SRAGPendingFact candidateFacts[];
   if(ArrayResize(candidateFacts, total + 1) != total + 1)
      return 0;
   for(int fact = 0; fact < total; fact++)
      if(!CopyPendingFact(pending_facts[fact], candidateFacts[fact]))
         return 0;
   if(!CopyPendingFact(source, candidateFacts[total]))
      return 0;
   ArraySwap(pending_facts, candidateFacts);
   iNextPendingId++;
   return pendingId;
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory confirm pending execution  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronRAGMemory::ConfirmPendingExecution(const ulong pendingId,
      const ulong deal,
      const ulong position)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0 || deal == 0 || position == 0)
      ReturnFalse;
   const int event = ArraySize(pending_facts[fact].events) - 1;
   if(event < 0 || pending_facts[fact].events[event].confirmed)
      ReturnFalse;
   pending_facts[fact].events[event].deal = deal;
   pending_facts[fact].events[event].position = position;
   pending_facts[fact].events[event].confirmed = true;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory correct pending operation.          |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CorrectPending(const ulong pendingId, const float &action[])
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0 || !ValidAction(action))
      ReturnFalse;
   const int eventCount = ArraySize(pending_facts[fact].events);
   if(eventCount <= 0 || eventCount >= INT_MAX ||
      !pending_facts[fact].events[eventCount - 1].confirmed)
      ReturnFalse;
   SRAGPendingFact candidateFacts[];
   const int total = ArraySize(pending_facts);
   if(ArrayResize(candidateFacts, total) != total)
      ReturnFalse;
   for(int source = 0; source < total; source++)
      if(!CopyPendingFact(pending_facts[source], candidateFacts[source]))
         ReturnFalse;
   if(ArrayResize(candidateFacts[fact].events, eventCount + 1) != eventCount + 1 ||
      ArrayResize(candidateFacts[fact].events[eventCount].action,
                  (int)iActionDimension) != (int)iActionDimension)
      ReturnFalse;
   ArrayCopy(candidateFacts[fact].events[eventCount].action, action);
   candidateFacts[fact].events[eventCount].confirmed = false;
   candidateFacts[fact].events[eventCount].deal = 0;
   candidateFacts[fact].events[eventCount].position = 0;
   ArraySwap(pending_facts, candidateFacts);
   return true;
  }
//+----------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory confirm pending correction  operation.  |
//+----------------------------------------------------------------------+
bool CNeuronRAGMemory::ConfirmPendingCorrection(const ulong pendingId, const ulong position)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0 || position == 0)
      ReturnFalse;
   const int eventCount = ArraySize(pending_facts[fact].events);
   const int event = eventCount - 1;
//--- A correction has no terminal deal. It is valid only after the opening
//--- event is confirmed for the same live position.
   if(eventCount < 2 || event < 0 || pending_facts[fact].events[event].confirmed ||
      !pending_facts[fact].events[0].confirmed ||
      pending_facts[fact].events[0].position != position)
      ReturnFalse;
   pending_facts[fact].events[event].deal = 0;
   pending_facts[fact].events[event].position = position;
   pending_facts[fact].events[event].confirmed = true;
   return true;
  }
//+------------------------------------------------------------------+
//| Confirm an executed correction for a position in the same plan.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ConfirmPendingCorrection(const ulong pendingId,
                                                 const ulong deal,
                                                 const ulong position)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0 || deal == 0 || position == 0)
      ReturnFalse;
   const int eventCount = ArraySize(pending_facts[fact].events);
   const int event = eventCount - 1;
   if(eventCount < 2 || event < 0 || pending_facts[fact].events[event].confirmed ||
      !pending_facts[fact].events[0].confirmed)
      ReturnFalse;
   pending_facts[fact].events[event].deal = deal;
   pending_facts[fact].events[event].position = position;
   pending_facts[fact].events[event].confirmed = true;
   return true;
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory cancel pending correction  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronRAGMemory::CancelPendingCorrection(const ulong pendingId)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0)
      ReturnFalse;
   const int eventCount = ArraySize(pending_facts[fact].events);
   if(eventCount < 2 || pending_facts[fact].events[eventCount - 1].confirmed)
      ReturnFalse;
   SRAGPendingFact candidateFacts[];
   const int total = ArraySize(pending_facts);
   if(ArrayResize(candidateFacts, total) != total)
      ReturnFalse;
   for(int source = 0; source < total; source++)
      if(!CopyPendingFact(pending_facts[source], candidateFacts[source]))
         ReturnFalse;
   if(ArrayResize(candidateFacts[fact].events, eventCount - 1) != eventCount - 1)
      ReturnFalse;
   ArraySwap(pending_facts, candidateFacts);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory close pending operation.            |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ClosePending(const ulong pendingId,
                                    const SRAGTerminalOutcome &terminal)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0 || !ValidTerminal(terminal))
      ReturnFalse;
   int confirmedCount = 0;
   for(int event = 0; event < ArraySize(pending_facts[fact].events); event++)
      if(pending_facts[fact].events[event].confirmed)
         confirmedCount++;
   const int current = ArraySize(pending_records);
   if(confirmedCount <= 0 || current < 0 ||
      (ulong)current + (ulong)confirmedCount > (ulong)INT_MAX)
      ReturnFalse;
   SRAGPendingFact remainingFacts[];
   if(!BuildRemainingPendingFacts(fact, remainingFacts))
      ReturnFalse;
   SRAGCompletedRecord candidateRecords[];
   const int candidateCount = current + confirmedCount;
   if(ArrayResize(candidateRecords, candidateCount) != candidateCount)
      ReturnFalse;
   for(int record = 0; record < current; record++)
      if(!CopyCompletedRecord(pending_records[record], candidateRecords[record]))
         ReturnFalse;
   int destination = current;
   for(int event = 0; event < ArraySize(pending_facts[fact].events); event++)
     {
      if(!pending_facts[fact].events[event].confirmed)
         continue;
      SRAGCompletedRecord source;
      if(ArrayResize(source.scenarioEmbedding, (int)iScenarioEmbedding) != (int)iScenarioEmbedding ||
         ArrayResize(source.action, (int)iActionDimension) != (int)iActionDimension)
         ReturnFalse;
      ArrayCopy(source.scenarioEmbedding, pending_facts[fact].scenarioEmbedding);
      ArrayCopy(source.action, pending_facts[fact].events[event].action);
      source.terminal = terminal;
      if(!CopyCompletedRecord(source, candidateRecords[destination]))
         ReturnFalse;
      destination++;
     }
   ArraySwap(pending_records, candidateRecords);
   ArraySwap(pending_facts, remainingFacts);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory cancel pending operation.           |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::CancelPending(const ulong pendingId)
  {
   const int fact = FindPendingFact(pendingId);
   if(fact < 0)
      ReturnFalse;
   for(int event = 0; event < ArraySize(pending_facts[fact].events); event++)
      if(pending_facts[fact].events[event].confirmed)
         ReturnFalse;
   SRAGPendingFact remainingFacts[];
   if(!BuildRemainingPendingFacts(fact, remainingFacts))
      ReturnFalse;
   ArraySwap(pending_facts, remainingFacts);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory publish operation.                  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Publish(const bool episodeClosed)
  {
   const uint pending = PendingRecordCount();
   if(pending == 0)
      return true;
   if(!episodeClosed && (iOnlineMemorySize == 0 || pending < iOnlineMemorySize))
      return true;
   SRAGScenarioCentroid candidateSnapshot[];
   uint candidateCount = iScenarioCount;
   if(!CopySnapshot(scenarios, candidateSnapshot))
      ReturnFalse;
   for(uint record = 0; record < pending; record++)
      if(!ApplyCompletedRecordToSnapshot(candidateSnapshot, candidateCount,
                                         iScenarioCapacity + record,
                                         pending_records[record].scenarioEmbedding,
                                         pending_records[record].action,
                                         pending_records[record].terminal))
         ReturnFalse;
   if(!ConsolidateActionCandidates(candidateSnapshot))
      ReturnFalse;
   const ulong fingerprint = SnapshotFingerprint(candidateSnapshot, candidateCount);
   CRAGInferenceSet *candidate = NULL;
   if(!BuildInferenceSet(candidateSnapshot, fingerprint, candidate))
      ReturnFalse;
   ArraySwap(scenarios, candidateSnapshot);
   iScenarioCount = candidateCount;
   CRAGInferenceSet *previous = m_inference;
   m_inference = candidate;
   if(previous != NULL)
      delete previous;
   ArrayResize(pending_records, 0);
   iPublicationGeneration++;
   return true;
  }
//+----------------------------------------------------------------------------+
//| Runs the CNeuronRAGMemory feed-forward pass and writes the layer  output.  |
//+----------------------------------------------------------------------------+
bool CNeuronRAGMemory::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   return Retrieve(NeuronOCL == NULL ? NULL : NeuronOCL.getOutput());
  }
//+-------------------------------------------------------------------+
//| Clears the CNeuronRAGMemory execution state and its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronRAGMemory::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(m_inference == NULL || m_inference.tokens == NULL)
      return true;
   if(!m_inference.tokens.BufferInit(iTopK * iActionCapacity * iTokenWidth, 0.0f))
      ReturnFalse;
   return m_inference.tokens.BufferCreate(OpenCL);
  }
//+-------------------------------------------------------------------------+
//| Rebinds the CNeuronRAGMemory GPU buffers to the given OpenCL  context.  |
//+-------------------------------------------------------------------------+
void CNeuronRAGMemory::SetOpenCL(COpenCLMy *obj)
  {
   const bool wasBound = (m_inference != NULL);
   const ulong fingerprint = (wasBound ? m_inference.fingerprint : 0);
   if(m_inference != NULL)
      delete m_inference;
   m_inference = NULL;
   cNearestCentroids.BufferFree();
   cNearestUsed.BufferFree();
   cNearestResult.BufferFree();
   CNeuronBaseOCL::SetOpenCL(obj);
   if(obj == NULL || !wasBound)
      return;
   if(fingerprint == 0)
      BindNullInference();
   else
     {
      CRAGInferenceSet *candidate = NULL;
      if(BuildInferenceSet(scenarios, fingerprint, candidate))
         m_inference = candidate;
     }
  }
//+------------------------------------------------------------------+
//| Returns whether has bytes CNeuronRAGMemory.                      |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::HasBytes(const int file_handle, const ulong bytes) const
  {
   const ulong position = (ulong)FileTell(file_handle);
   const ulong size = (ulong)FileSize(file_handle);
   return (position <= size && bytes <= size - position);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory write u int operation.              |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::WriteUInt(const int file_handle, const uint value) const
  {
   return (FileWriteInteger(file_handle, (int)value, INT_VALUE) == sizeof(int));
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory write u long operation.             |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::WriteULong(const int file_handle, const ulong value) const
  {
   return (FileWriteLong(file_handle, (long)value) == sizeof(long));
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory write float operation.              |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::WriteFloat(const int file_handle, const float value) const
  {
   return (MathIsValidNumber(value) && FileWriteFloat(file_handle, value) == sizeof(float));
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory read u int operation.               |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ReadUInt(const int file_handle, uint &value) const
  {
   if(!HasBytes(file_handle, sizeof(int)) || FileIsEnding(file_handle))
      ReturnFalse;
   value = (uint)FileReadInteger(file_handle, INT_VALUE);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory read u long operation.              |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ReadULong(const int file_handle, ulong &value) const
  {
   if(!HasBytes(file_handle, sizeof(long)) || FileIsEnding(file_handle))
      ReturnFalse;
   value = (ulong)FileReadLong(file_handle);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory read float operation.               |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ReadFloat(const int file_handle, float &value) const
  {
   if(!HasBytes(file_handle, sizeof(float)) || FileIsEnding(file_handle))
      ReturnFalse;
   value = FileReadFloat(file_handle);
   return MathIsValidNumber(value);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory snapshot payload bytes operation.   |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::SnapshotPayloadBytes(const uint scenarioEmbedding,
      const uint actionDimension,
      const uint scenarioCapacity,
      const uint actionCapacity,
      ulong &bytes) const
  {
   const ulong maximum = ulong(-1);
   if(scenarioEmbedding > (maximum - 16) / sizeof(float) ||
      actionDimension > (maximum - 44) / sizeof(float))
      ReturnFalse;
   const ulong actionBytes = 44 + (ulong)actionDimension * sizeof(float);
   if(actionCapacity > maximum / actionBytes)
      ReturnFalse;
   const ulong allActions = (ulong)actionCapacity * actionBytes;
   const ulong scenarioPrefix = 16 + (ulong)scenarioEmbedding * sizeof(float);
   if(allActions > maximum - scenarioPrefix)
      ReturnFalse;
   const ulong scenarioBytes = scenarioPrefix + allActions;
   if(scenarioCapacity > maximum / scenarioBytes)
      ReturnFalse;
   bytes = (ulong)scenarioCapacity * scenarioBytes;
   return true;
  }
//+-------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory write published snapshot operation.  |
//+-------------------------------------------------------------------+
bool CNeuronRAGMemory::WritePublishedSnapshot(const int file_handle, const ulong fingerprint) const
  {
   const uint magic = RAG_SNAPSHOT_MAGIC;
   const uint format = RAG_FORMAT_PUBLISHED;
   ulong payloadBytes = 0;
   if(!ValidateSnapshot(scenarios, iScenarioCount, iScenarioEmbedding, iActionDimension,
                        iScenarioCapacity, iActionCapacity) ||
      !SnapshotPayloadBytes(iScenarioEmbedding, iActionDimension, iScenarioCapacity,
                            iActionCapacity, payloadBytes) ||
      !WriteUInt(file_handle, magic) || !WriteUInt(file_handle, format) ||
      !WriteUInt(file_handle, iScenarioEmbedding) || !WriteUInt(file_handle, iActionDimension) ||
      !WriteUInt(file_handle, iTokenWidth) || !WriteUInt(file_handle, iScenarioCapacity) ||
      !WriteUInt(file_handle, iActionCapacity) || !WriteUInt(file_handle, iTopK) ||
      !WriteUInt(file_handle, iOnlineMemorySize) ||
      !WriteUInt(file_handle, iScenarioCount) || !WriteULong(file_handle, fingerprint) ||
      !WriteULong(file_handle, payloadBytes))
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
     {
      if(!WriteUInt(file_handle, (scenarios[scenario].used ? 1 : 0)) ||
         !WriteUInt(file_handle, scenarios[scenario].scenarioId) ||
         !WriteULong(file_handle, scenarios[scenario].hit_count))
         ReturnFalse;
      for(uint element = 0; element < iScenarioEmbedding; element++)
         if(!WriteFloat(file_handle, scenarios[scenario].centroid[element]))
            ReturnFalse;
      for(uint action = 0; action < iActionCapacity; action++)
        {
         if(!WriteUInt(file_handle, (scenarios[scenario].actions[action].used ? 1 : 0)) ||
            !WriteUInt(file_handle, scenarios[scenario].actions[action].scenarioId) ||
            !WriteULong(file_handle, scenarios[scenario].actions[action].hit_count) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_reward) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.relative_return) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.maxDrawdown) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.costs) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.riskUsage) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.duration) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.reward))
            ReturnFalse;
         for(uint element = 0; element < iActionDimension; element++)
            if(!WriteFloat(file_handle, scenarios[scenario].actions[action].centroid[element]))
               ReturnFalse;
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory read published snapshot operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ReadPublishedSnapshot(const int file_handle,
      SRAGScenarioCentroid &snapshot[],
      uint &scenarioCount, ulong &fingerprint) const
  {
   uint magic = 0, format = 0, embedding = 0, actionDimension = 0, tokenWidth = 0, scenarioCapacity = 0,
        actionCapacity = 0, topK = 0, onlineMemorySize = 0;
   ulong payloadBytes = 0;
   if(!ReadUInt(file_handle, magic) || !ReadUInt(file_handle, format) ||
      !ReadUInt(file_handle, embedding) || !ReadUInt(file_handle, actionDimension) ||
      !ReadUInt(file_handle, tokenWidth) || !ReadUInt(file_handle, scenarioCapacity) ||
      !ReadUInt(file_handle, actionCapacity) || !ReadUInt(file_handle, topK) ||
      !ReadUInt(file_handle, onlineMemorySize) ||
      !ReadUInt(file_handle, scenarioCount) || !ReadULong(file_handle, fingerprint) ||
      !ReadULong(file_handle, payloadBytes) || magic != RAG_SNAPSHOT_MAGIC || format != RAG_FORMAT_PUBLISHED ||
      embedding != iScenarioEmbedding || actionDimension != iActionDimension ||
      tokenWidth != iTokenWidth || scenarioCapacity != iScenarioCapacity ||
      actionCapacity != iActionCapacity || topK != iTopK || onlineMemorySize != iOnlineMemorySize ||
      topK > RAG_TOPK_MAX || scenarioCapacity > RAG_FLOAT_INDEX_LIMIT ||
      scenarioCount > scenarioCapacity || fingerprint == 0)
      ReturnFalse;
   ulong expected = 0;
   if(!SnapshotPayloadBytes(embedding, actionDimension, scenarioCapacity, actionCapacity, expected) ||
      expected != payloadBytes || !HasBytes(file_handle, payloadBytes) ||
      ArrayResize(snapshot, (int)scenarioCapacity) != (int)scenarioCapacity)
      ReturnFalse;
   const ulong snapshotEnd = (ulong)FileTell(file_handle) + payloadBytes;
   for(uint scenario = 0; scenario < scenarioCapacity; scenario++)
     {
      uint used = 0;
      if(!ReadUInt(file_handle, used) || used > 1 ||
         !ReadUInt(file_handle, snapshot[scenario].scenarioId) ||
         !ReadULong(file_handle, snapshot[scenario].hit_count) ||
         ArrayResize(snapshot[scenario].centroid, (int)embedding) != (int)embedding ||
         ArrayResize(snapshot[scenario].actions, (int)actionCapacity) != (int)actionCapacity)
         ReturnFalse;
      snapshot[scenario].used = (used == 1);
      for(uint element = 0; element < embedding; element++)
         if(!ReadFloat(file_handle, snapshot[scenario].centroid[element]))
            ReturnFalse;
      for(uint action = 0; action < actionCapacity; action++)
        {
         uint actionUsed = 0;
         if(!ReadUInt(file_handle, actionUsed) || actionUsed > 1 ||
            !ReadUInt(file_handle, snapshot[scenario].actions[action].scenarioId) ||
            !ReadULong(file_handle, snapshot[scenario].actions[action].hit_count) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_reward) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.relative_return) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.maxDrawdown) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.costs) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.riskUsage) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.duration) ||
            !ReadFloat(file_handle, snapshot[scenario].actions[action].mean_terminal.reward) ||
            ArrayResize(snapshot[scenario].actions[action].centroid, (int)actionDimension) != (int)actionDimension)
            ReturnFalse;
         snapshot[scenario].actions[action].used = (actionUsed == 1);
         for(uint element = 0; element < actionDimension; element++)
            if(!ReadFloat(file_handle, snapshot[scenario].actions[action].centroid[element]))
               ReturnFalse;
        }
     }
   return ((ulong)FileTell(file_handle) == snapshotEnd &&
           snapshotEnd == (ulong)FileSize(file_handle) &&
           ValidateSnapshot(snapshot, scenarioCount, embedding, actionDimension,
                            scenarioCapacity, actionCapacity) &&
           SnapshotFingerprint(snapshot, scenarioCount) == fingerprint);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory save published snapshot operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::SavePublishedSnapshot(const string path) const
  {
   if(StringLen(path) == 0)
      ReturnFalse;
   const ulong fingerprint = SnapshotFingerprint(scenarios, iScenarioCount);
   const int file_handle = FileOpen(path, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   const bool result = WritePublishedSnapshot(file_handle, fingerprint);
   if(result)
      FileFlush(file_handle);
   FileClose(file_handle);
   return result;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory load published snapshot operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::LoadPublishedSnapshot(const string path)
  {
   if(StringLen(path) == 0 || OpenCL == NULL)
      ReturnFalse;
   const int file_handle = FileOpen(path, FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   SRAGScenarioCentroid loaded[];
   uint scenarioCount = 0;
   ulong fingerprint = 0;
   const bool read = ReadPublishedSnapshot(file_handle, loaded, scenarioCount, fingerprint);
   const bool exactEnd = (FileTell(file_handle) == FileSize(file_handle));
   FileClose(file_handle);
   if(!read || !exactEnd)
      ReturnFalse;
   CRAGInferenceSet *candidate = NULL;
   if(!BuildInferenceSet(loaded, fingerprint, candidate))
      ReturnFalse;
   if(!CommitSnapshot(loaded, scenarioCount))
     {
      delete candidate;
      ReturnFalse;
     }
   CRAGInferenceSet *previous = m_inference;
   m_inference = candidate;
   if(previous != NULL)
      delete previous;
   ArrayResize(pending_records, 0);
   iPublicationGeneration++;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory bind inference snapshot operation.  |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::BindInferenceSnapshot(const string path)
  {
   return LoadPublishedSnapshot(path);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronRAGMemory state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Save(const int file_handle)
  {
   if(!ValidateSnapshot(scenarios, iScenarioCount, iScenarioEmbedding, iActionDimension,
                        iScenarioCapacity, iActionCapacity) ||
      !CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   ulong payloadBytes = 0;
   if(!SnapshotPayloadBytes(iScenarioEmbedding, iActionDimension, iScenarioCapacity,
                            iActionCapacity, payloadBytes) ||
      !WriteUInt(file_handle, iFormatVersion) ||
      !WriteUInt(file_handle, iScenarioEmbedding) ||
      !WriteUInt(file_handle, iActionDimension) ||
      !WriteUInt(file_handle, iTokenWidth) ||
      !WriteUInt(file_handle, iScenarioCapacity) ||
      !WriteUInt(file_handle, iActionCapacity) ||
      !WriteUInt(file_handle, iTopK) ||
      !WriteUInt(file_handle, iOnlineMemorySize) ||
      !WriteUInt(file_handle, iScenarioCount) ||
      !WriteULong(file_handle, payloadBytes))
      ReturnFalse;
   for(uint scenario = 0; scenario < iScenarioCapacity; scenario++)
     {
      if(!WriteUInt(file_handle, (scenarios[scenario].used ? 1 : 0)) ||
         !WriteUInt(file_handle, scenarios[scenario].scenarioId) ||
         !WriteULong(file_handle, scenarios[scenario].hit_count))
         ReturnFalse;
      for(uint element = 0; element < iScenarioEmbedding; element++)
         if(!WriteFloat(file_handle, scenarios[scenario].centroid[element]))
            ReturnFalse;
      for(uint action = 0; action < iActionCapacity; action++)
        {
         if(!WriteUInt(file_handle, (scenarios[scenario].actions[action].used ? 1 : 0)) ||
            !WriteUInt(file_handle, scenarios[scenario].actions[action].scenarioId) ||
            !WriteULong(file_handle, scenarios[scenario].actions[action].hit_count) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_reward) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.relative_return) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.maxDrawdown) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.costs) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.riskUsage) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.duration) ||
            !WriteFloat(file_handle, scenarios[scenario].actions[action].mean_terminal.reward))
            ReturnFalse;
         for(uint element = 0; element < iActionDimension; element++)
            if(!WriteFloat(file_handle, scenarios[scenario].actions[action].centroid[element]))
               ReturnFalse;
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronRAGMemory state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   uint formatVersion = 0;
   uint scenarioEmbedding = 0;
   uint actionDimension = 0;
   uint tokenWidth = 0;
   uint scenarioCapacity = 0;
   uint actionCapacity = 0;
   uint topK = 0;
   uint onlineMemorySize = 0;
   uint scenarioCount = 0;
   ulong payloadBytes = 0;
   if(!ReadUInt(file_handle, formatVersion) || !ReadUInt(file_handle, scenarioEmbedding) ||
      !ReadUInt(file_handle, actionDimension) || !ReadUInt(file_handle, tokenWidth) ||
      !ReadUInt(file_handle, scenarioCapacity) || !ReadUInt(file_handle, actionCapacity) ||
      !ReadUInt(file_handle, topK) || !ReadUInt(file_handle, onlineMemorySize) ||
      !ReadUInt(file_handle, scenarioCount) || !ReadULong(file_handle, payloadBytes) ||
      formatVersion != RAG_FORMAT_SAVE || scenarioEmbedding == 0 || actionDimension == 0 || tokenWidth == 0 ||
      scenarioCapacity == 0 || actionCapacity == 0 || topK == 0 ||
      topK > RAG_TOPK_MAX || scenarioCapacity > RAG_FLOAT_INDEX_LIMIT ||
      scenarioCapacity >= (uint)INT_MAX || actionCapacity >= (uint)INT_MAX ||
      scenarioCount > scenarioCapacity)
      ReturnFalse;
   ulong expectedPayload = 0;
   if(!SnapshotPayloadBytes(scenarioEmbedding, actionDimension, scenarioCapacity,
                            actionCapacity, expectedPayload) || payloadBytes != expectedPayload ||
      !HasBytes(file_handle, payloadBytes))
      ReturnFalse;
   const ulong snapshotEnd = (ulong)FileTell(file_handle) + payloadBytes;
   SRAGScenarioCentroid loaded[];
   if(ArrayResize(loaded, (int)scenarioCapacity) != (int)scenarioCapacity)
      ReturnFalse;
   for(uint scenario = 0; scenario < scenarioCapacity; scenario++)
     {
      uint used = 0;
      if(!ReadUInt(file_handle, used) || used > 1 ||
         !ReadUInt(file_handle, loaded[scenario].scenarioId) ||
         !ReadULong(file_handle, loaded[scenario].hit_count) ||
         ArrayResize(loaded[scenario].centroid, (int)scenarioEmbedding) != (int)scenarioEmbedding ||
         ArrayResize(loaded[scenario].actions, (int)actionCapacity) != (int)actionCapacity)
         ReturnFalse;
      loaded[scenario].used = (used == 1);
      for(uint element = 0; element < scenarioEmbedding; element++)
         if(!ReadFloat(file_handle, loaded[scenario].centroid[element]))
            ReturnFalse;
      for(uint action = 0; action < actionCapacity; action++)
        {
         uint actionUsed = 0;
         if(!ReadUInt(file_handle, actionUsed) || actionUsed > 1 ||
            !ReadUInt(file_handle, loaded[scenario].actions[action].scenarioId) ||
            !ReadULong(file_handle, loaded[scenario].actions[action].hit_count) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_reward) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.relative_return) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.maxDrawdown) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.costs) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.riskUsage) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.duration) ||
            !ReadFloat(file_handle, loaded[scenario].actions[action].mean_terminal.reward) ||
            ArrayResize(loaded[scenario].actions[action].centroid, (int)actionDimension) != (int)actionDimension)
            ReturnFalse;
         loaded[scenario].actions[action].used = (actionUsed == 1);
         for(uint element = 0; element < actionDimension; element++)
            if(!ReadFloat(file_handle, loaded[scenario].actions[action].centroid[element]))
               ReturnFalse;
        }
     }
   if((ulong)FileTell(file_handle) != snapshotEnd ||
      !ValidateSnapshot(loaded, scenarioCount, scenarioEmbedding, actionDimension,
                        scenarioCapacity, actionCapacity))
      ReturnFalse;
   const uint oldFormatVersion = iFormatVersion;
   const uint oldScenarioEmbedding = iScenarioEmbedding;
   const uint oldActionDimension = iActionDimension;
   const uint oldTokenWidth = iTokenWidth;
   const uint oldScenarioCapacity = iScenarioCapacity;
   const uint oldActionCapacity = iActionCapacity;
   const uint oldTopK = iTopK;
   const uint oldOnlineMemorySize = iOnlineMemorySize;
   iFormatVersion = formatVersion;
   iScenarioEmbedding = scenarioEmbedding;
   iActionDimension = actionDimension;
   iTokenWidth = tokenWidth;
   iScenarioCapacity = scenarioCapacity;
   iActionCapacity = actionCapacity;
   iTopK = topK;
   iOnlineMemorySize = onlineMemorySize;
   if(CommitSnapshot(loaded, scenarioCount))
      return true;
   iFormatVersion = oldFormatVersion;
   iScenarioEmbedding = oldScenarioEmbedding;
   iActionDimension = oldActionDimension;
   iTokenWidth = oldTokenWidth;
   iScenarioCapacity = oldScenarioCapacity;
   iActionCapacity = oldActionCapacity;
   iTopK = oldTopK;
   iOnlineMemorySize = oldOnlineMemorySize;
   ReturnFalse;
  }
//--- Generic device gateway.  Callers provide buffers sized from their own
//--- CLayerDescription; no Expert constant participates in launch geometry.
//+------------------------------------------------------------------+
//| Запускает RAG-конвейер с подготовленными буферами слоёв.         |
//+------------------------------------------------------------------+
bool DispatchRAGPipeline(COpenCLMy *opencl, CBufferFloat *embedding,
                         CBufferFloat *centroids, CBufferFloat *used,
                         CBufferFloat *scores, CBufferFloat *partial, CBufferFloat *merge,
                         CBufferFloat *top, CBufferFloat *actions,
                         CBufferFloat *actionUsed, CBufferFloat *actionMeanReward,
                         CBufferFloat *actionHitCount, CBufferFloat *absoluteRelevance,
                         CBufferFloat *tokens, const uint scenarioCount,
                         const uint embeddingDimension, const uint actionCount,
                         const uint actionDimension, const uint tokenWidth, const uint topK)
  {
   if(opencl == NULL || embedding == NULL || centroids == NULL || used == NULL ||
      scores == NULL || partial == NULL || merge == NULL || top == NULL || actions == NULL || actionUsed == NULL || tokens == NULL ||
      actionMeanReward == NULL || actionHitCount == NULL ||
      absoluteRelevance == NULL ||
      scenarioCount == 0 || embeddingDimension == 0 || actionCount == 0 || actionDimension == 0 || topK == 0 ||
      actionDimension > INT_MAX - 2 || tokenWidth != actionDimension + 2 ||
      scenarioCount > INT_MAX || embeddingDimension > INT_MAX || actionCount > INT_MAX ||
      tokenWidth > INT_MAX || topK > INT_MAX || topK > RAG_TOPK_MAX ||
      scenarioCount > RAG_FLOAT_INDEX_LIMIT)
      ReturnFalse;
   const uint blockSize = RAG_TOPK_LOCAL_WIDTH;
//--- The staged ping-pong compaction keeps every intermediate buffer bounded
//--- by the initial partial image.  TopK must therefore reduce a full block.
   if(topK > RAG_TOPK_MAX)
      ReturnFalse;
   const uint partialCount = (scenarioCount + blockSize - 1) / blockSize;
   if((ulong)scenarioCount > ULONG_MAX / (ulong)embeddingDimension ||
      (ulong)scenarioCount > ULONG_MAX / (ulong)actionCount)
      ReturnFalse;
   const ulong centroidFloats = (ulong)scenarioCount * (ulong)embeddingDimension;
   const ulong actionSlots = (ulong)scenarioCount * (ulong)actionCount;
   if(actionSlots > ULONG_MAX / (ulong)actionDimension || (ulong)topK > ULONG_MAX / (ulong)actionCount)
      ReturnFalse;
   const ulong actionFloats = actionSlots * (ulong)actionDimension;
   const ulong topActions = (ulong)topK * (ulong)actionCount;
   if(topActions > ULONG_MAX / (ulong)tokenWidth)
      ReturnFalse;
   const ulong tokenFloats = topActions * (ulong)tokenWidth;
   const int embeddingTotal = embedding.Total();
   const int centroidsTotal = centroids.Total();
   const int usedTotal = used.Total();
   const int scoresTotal = scores.Total();
   const int partialTotal = partial.Total();
   const int mergeTotal = merge.Total();
   const int topTotal = top.Total();
   const int actionsTotal = actions.Total();
   const int actionUsedTotal = actionUsed.Total();
   const int actionMeanRewardTotal = actionMeanReward.Total();
   const int actionHitCountTotal = actionHitCount.Total();
   const int absoluteRelevanceTotal = absoluteRelevance.Total();
   const int tokensTotal = tokens.Total();
   if(embeddingTotal < 0 || centroidsTotal < 0 || usedTotal < 0 || scoresTotal < 0 || partialTotal < 0 || mergeTotal < 0 ||
      topTotal < 0 || actionsTotal < 0 || actionUsedTotal < 0 || actionMeanRewardTotal < 0 ||
      actionHitCountTotal < 0 || absoluteRelevanceTotal < 0 || tokensTotal < 0)
      ReturnFalse;
   if(centroidFloats > UINT_MAX || actionSlots > UINT_MAX || actionFloats > UINT_MAX || tokenFloats > UINT_MAX ||
      partialCount > UINT_MAX / topK || partialCount * topK > UINT_MAX / 2 || (ulong)embeddingTotal < embeddingDimension ||
      (ulong)centroidsTotal < centroidFloats || (ulong)usedTotal < scenarioCount || (ulong)scoresTotal < scenarioCount ||
      (ulong)partialTotal < partialCount * topK * 2 || (ulong)mergeTotal < partialCount * topK * 2 ||
      (ulong)topTotal < topK * 2 || (ulong)actionsTotal < actionFloats ||
      (ulong)actionUsedTotal < actionSlots || (ulong)actionMeanRewardTotal < actionSlots ||
      (ulong)actionHitCountTotal < actionSlots || absoluteRelevanceTotal < 1 ||
      (ulong)tokensTotal < tokenFloats)
      ReturnFalse;
   uint offset[] = {0};
   uint scoreGlobal[] = {scenarioCount};
   uint partialGlobal[] = {partialCount * blockSize};
   uint partialLocal[] = {blockSize};
   uint gatherGlobal[] = {topK * actionCount};
   if(!opencl.SetArgumentBuffer(def_k_RAGScore, def_k_rags_embedding, embedding.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGScore, def_k_rags_centroids, centroids.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGScore, def_k_rags_used, used.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGScore, def_k_rags_scores, scores.GetIndex()) ||
      !opencl.SetArgument(def_k_RAGScore, def_k_rags_scenario_count, scenarioCount) ||
      !opencl.SetArgument(def_k_RAGScore, def_k_rags_embedding_dimension, embeddingDimension) ||
      !opencl.Execute(def_k_RAGScore, 1, offset, scoreGlobal))
      ReturnFalse;
   if(!opencl.SetArgumentBuffer(def_k_RAGPartial, def_k_ragp_scores, scores.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGPartial, def_k_ragp_partial, partial.GetIndex()) ||
      !opencl.SetArgument(def_k_RAGPartial, def_k_ragp_scenario_count, scenarioCount) ||
      !opencl.SetArgument(def_k_RAGPartial, def_k_ragp_block_size, blockSize) ||
      !opencl.SetArgument(def_k_RAGPartial, def_k_ragp_top_k, topK) ||
      !opencl.Execute(def_k_RAGPartial, 1, offset, partialGlobal, partialLocal))
      ReturnFalse;
   uint mergeCount = partialCount * topK;
   CBufferFloat *source = partial;
   CBufferFloat *destination = merge;
   while(mergeCount > blockSize)
     {
      const uint mergeBlocks = (mergeCount + blockSize - 1) / blockSize;
      if(mergeBlocks > UINT_MAX / topK)
         ReturnFalse;
      const uint nextCount = mergeBlocks * topK;
      uint mergeGlobal[] = {mergeBlocks * blockSize};
      uint mergeLocal[] = {blockSize};
      if(!opencl.SetArgumentBuffer(def_k_RAGMerge, def_k_ragm_source, source.GetIndex()) ||
         !opencl.SetArgumentBuffer(def_k_RAGMerge, def_k_ragm_destination, destination.GetIndex()) ||
         !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_candidate_count, mergeCount) ||
         !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_block_size, blockSize) ||
         !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_top_k, topK) ||
         !opencl.Execute(def_k_RAGMerge, 1, offset, mergeGlobal, mergeLocal))
         ReturnFalse;
      mergeCount = nextCount;
      CBufferFloat *swap = source;
      source = destination;
      destination = swap;
     }
   uint mergeGlobal[] = {blockSize};
   uint mergeLocal[] = {blockSize};
   if(!opencl.SetArgumentBuffer(def_k_RAGMerge, def_k_ragm_source, source.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGMerge, def_k_ragm_destination, top.GetIndex()) ||
      !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_candidate_count, mergeCount) ||
      !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_block_size, blockSize) ||
      !opencl.SetArgument(def_k_RAGMerge, def_k_ragm_top_k, topK) ||
      !opencl.Execute(def_k_RAGMerge, 1, offset, mergeGlobal, mergeLocal))
      ReturnFalse;
   if(!opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_top, top.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_actions, actions.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_tokens, tokens.GetIndex()) ||
      !opencl.SetArgument(def_k_RAGGather, def_k_ragg_scenario_count, scenarioCount) ||
      !opencl.SetArgument(def_k_RAGGather, def_k_ragg_action_count, actionCount) ||
      !opencl.SetArgument(def_k_RAGGather, def_k_ragg_action_dimension, actionDimension) ||
      !opencl.SetArgument(def_k_RAGGather, def_k_ragg_token_width, tokenWidth) ||
      !opencl.SetArgument(def_k_RAGGather, def_k_ragg_top_k, topK) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_action_used, actionUsed.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_action_mean_reward, actionMeanReward.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_action_hit_count, actionHitCount.GetIndex()) ||
      !opencl.SetArgumentBuffer(def_k_RAGGather, def_k_ragg_absolute_relevance, absoluteRelevance.GetIndex()) ||
      !opencl.Execute(def_k_RAGGather, 1, offset, gatherGlobal))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory retrieve operation.                 |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Retrieve(CNet *scenarioModel, const uint layer)
  {
   if(!scenarioModel)
      ReturnFalse;
   CBufferFloat *scenarioEmbedding = NULL;
   if(!scenarioModel.GetLayerOutputDevice(layer, scenarioEmbedding))
      ReturnFalse;
   return Retrieve(scenarioEmbedding);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory retrieve operation (Retrieve).      |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::Retrieve(CBufferFloat *scenarioEmbedding)
  {
   if(m_inference == NULL || m_inference.opencl == NULL || m_inference.opencl != OpenCL ||
      scenarioEmbedding == NULL || scenarioEmbedding.GetOpenCL() != OpenCL ||
      m_inference.centroids == NULL || m_inference.used == NULL || m_inference.scores == NULL ||
      m_inference.partial == NULL || m_inference.merge == NULL || m_inference.top == NULL || m_inference.actions == NULL ||
      m_inference.actionUsed == NULL || m_inference.actionMeanReward == NULL ||
      m_inference.actionHitCount == NULL || m_inference.absoluteRelevance == NULL ||
      m_inference.tokens == NULL)
      ReturnFalse;
   return DispatchRAGPipeline(OpenCL, scenarioEmbedding, m_inference.centroids, m_inference.used,
                              m_inference.scores, m_inference.partial, m_inference.merge, m_inference.top,
                              m_inference.actions, m_inference.actionUsed, m_inference.actionMeanReward,
                              m_inference.actionHitCount, m_inference.absoluteRelevance, m_inference.tokens,
                              iScenarioCapacity, iScenarioEmbedding, iActionCapacity,
                              iActionDimension, iTokenWidth, iTopK);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory scenario centroid operation.        |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ScenarioCentroid(const uint scenarioId, float &destination[]) const
  {
   if(scenarioId >= iScenarioCapacity || !scenarios[scenarioId].used ||
      ArraySize(destination) < (int)iScenarioEmbedding)
      ReturnFalse;
   ArrayCopy(destination, scenarios[scenarioId].centroid);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory action centroid operation.          |
//+------------------------------------------------------------------+
bool CNeuronRAGMemory::ActionCentroid(const uint scenarioId, const uint actionId,
                                      float &destination[]) const
  {
   if(scenarioId >= iScenarioCapacity || actionId >= iActionCapacity ||
      !scenarios[scenarioId].actions[actionId].used ||
      ArraySize(destination) < (int)iActionDimension)
      ReturnFalse;
   ArrayCopy(destination, scenarios[scenarioId].actions[actionId].centroid);
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory scenario hit count operation.       |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::ScenarioHitCount(const uint scenarioId) const
  {
   return (scenarioId < iScenarioCapacity && scenarios[scenarioId].used ?
           scenarios[scenarioId].hit_count : 0);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronRAGMemory action hit count operation.         |
//+------------------------------------------------------------------+
ulong CNeuronRAGMemory::ActionHitCount(const uint scenarioId, const uint actionId) const
  {
   return (scenarioId < iScenarioCapacity && actionId < iActionCapacity &&
           scenarios[scenarioId].actions[actionId].used ?
           scenarios[scenarioId].actions[actionId].hit_count : 0);
  }
//+------------------------------------------------------------------+
//| Returns the layer info of the CNeuronRAGMemory.                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronRAGMemory::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronBaseOCL::GetLayerInfo();
   if(result == NULL)
      return NULL;
   result.type = defNeuronRAGMemory;
   result.count = iTokenWidth;
   result.window = iScenarioEmbedding;
   result.window_out = iActionDimension;
   result.layers = iTopK;
   ArrayResize(result.units, 1);
   ArrayResize(result.heads, 1);
   result.units[0] = iScenarioCapacity;
   result.heads[0] = iActionCapacity;
   return result;
  }

#endif // NEURONET_RAG_MEMORY_MQH
//+---+