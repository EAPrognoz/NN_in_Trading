//+------------------------------------------------------------------+
//|                          Trajectory.mqh                          |
//|                       Copyright 2026, DNG                        |
//|                       https://www.mql5.com                       |
//+------------------------------------------------------------------+
//|                                                                  |
//| Trajectory state, network topology and RAG memory helpers        |
//| for the VLADriver-RAG Expert Advisor.                            |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
input group                "---- Indicators ----"
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
input group                "---- RAG runtime ----"
// Completed records staged before the library publishes a new immutable
// snapshot. It is collection policy, never part of layer topology.
input uint                 OnlineMemorySize = 256;
// Domain facts supplied with completed trading records for normalization.
input double               MemoryRiskBudgetFraction = 0.01;
input double               MemoryVolatilityRange = 10.0;
int            iLatentLayer = -1;
#define        HistoryBars             5
#define        BarDescr                9               //Elements for 1 bar description
#define        AccountDescr            13              //Account description
#define        NActions                6               //Number of possible Actions
#define        NRewards                1               //Number of rewards
#define        NForecast               12              //Number of forecast
#define        EtalonBalance           1e4
#define        BatchSize               1e+5
#define        EmbeddingSize           16
#define        DiscFactor              0.5f
#define        FileName                "VLADriverRAG"
#define        ActorGraphVersion       3
#define        CriticCheckpointFile    "Crt.nnw"
#define        LatentCount             64
#define        LatentLayer             iLatentLayer
#define        StateScenarioLayer      7
// The Critic attends to the full causal RankTCM output [BarDescr, EmbeddingSize].
// The Actor/RAG path uses the separately pooled causal scenario embedding.
#define        StateTokenLayer         4
#define        ForecastTokenDim        (EmbeddingSize + 1)
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        ActorUpdate             5
#define        TargetUpdate            24*5
#define        tau                     0.9f
#define        NHeads                  4
#define        NExperts                5
#define        RAGScenarioCentroids    512             //CLayerDescription capacity
#define        RAGActionCentroids      24              //per-scenario CLayerDescription capacity
#define        NScenarios              3               //Account, position and time primary candidates
#define        ActorMPITopK            3               //Must not exceed the three Actor primary candidates
#define        TopK                    5               //Critic candidate selection remains independent from Actor MPI
#define        Quantiles               8
#define        StackSize               24*NScenarios
#define        Blocks                  24
CSymbolInfo          Symb;
CTrade               Trade;
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
struct SState
  {
   float             state[HistoryBars * BarDescr];
   float             account[AccountDescr - 4];
   float             action[NActions];
   float             rewards[NRewards];
                     SState(void);
   bool              Save(int file_handle);
   bool              Load(int file_handle);
   // Overloading.
   void              operator=(const SState &obj)
     {
      ArrayCopy(state, obj.state);
      ArrayCopy(account, obj.account);
      ArrayCopy(action, obj.action);
      ArrayCopy(rewards, obj.rewards);
     }
  };
//| SState implementation.                                           |
//+------------------------------------------------------------------+
SState::SState(void)
  {
   ArrayInitialize(state, 0);
   ArrayInitialize(account, 0);
   ArrayInitialize(action, 0);
   ArrayInitialize(rewards, 0);
  }
//| Save implementation.                                             |
//+------------------------------------------------------------------+
bool SState::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   int total = ArraySize(state);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, state[i]) < sizeof(float))
         ReturnFalse;
   total = ArraySize(account);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, account[i]) < sizeof(float))
         ReturnFalse;
   total = ArraySize(action);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, action[i]) < sizeof(float))
         ReturnFalse;
   total = ArraySize(rewards);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, rewards[i]) < sizeof(float))
         ReturnFalse;
   return true;
  }
//| Load implementation.                                             |
//+------------------------------------------------------------------+
bool SState::Load(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   int total = FileReadInteger(file_handle);
   if(total != ArraySize(state))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      state[i] = FileReadFloat(file_handle);
     }
   total = FileReadInteger(file_handle);
   if(total != ArraySize(account))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      account[i] = FileReadFloat(file_handle);
     }
   total = FileReadInteger(file_handle);
   if(total != ArraySize(action))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      action[i] = MathMin(MathMax(FileReadFloat(file_handle), 0), 1);
     }
   total = FileReadInteger(file_handle);
   if(total != ArraySize(rewards))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      rewards[i] = FileReadFloat(file_handle);
     }
   return true;
  }
//+------------------------------------------------------------------+
//| CreateStateDescriptions.                                         |
//+------------------------------------------------------------------+
bool CreateStateDescriptions(CArrayObj *&encoder,
                             CArrayObj *&decoder
                            )
  {
   CLayerDescription *descr;
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         ReturnFalse;
     }
   if(!decoder)
     {
      decoder = new CArrayObj();
      if(!decoder)
         ReturnFalse;
     }
// State Encoder.
   encoder.Clear();
// Input layer.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = HistoryBars * BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 1.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 2.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronDropoutOCL;
   descr.count = prev_count;
   descr.probability = 0.1f;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 3.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverData;
   descr.window = HistoryBars;
   descr.count = BarDescr;
     {
      uint temp[] = {StackSize, StackSize, Quantiles};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.probability = 1.0f;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 4.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverRankTCM;
   descr.window = HistoryBars * (2 * Quantiles + 1);
   descr.count = EmbeddingSize;
   descr.variables = BarDescr;
     {
      uint temp[] = {StackSize, NHeads};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 5.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConv;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = EmbeddingSize / 2;
   descr.count = 1;
   descr.variables = BarDescr;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 6.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = EmbeddingSize;
   descr.batch = BatchSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 7.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeActivation;
   descr.count = EmbeddingSize;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
// Forecast Decoder.
   decoder.Clear();
// Input layer.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 1.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = 1;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = NForecast * EmbeddingSize;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 2.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = 2 * EmbeddingSize;
   descr.activation = GELU;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 3.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = 2 * EmbeddingSize;
   descr.step = 2 * EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.activation = GELU;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 4.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = BarDescr;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
// layer 5.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBatchNormOCL;
   descr.count = NForecast * BarDescr;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
   return true;
  }
//+------------------------------------------------------------------+
//| CreateDescriptions.                                              |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *&actor,
                        CArrayObj *&critic
                       )
  {
   CLayerDescription *descr;
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         ReturnFalse;
     }
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         ReturnFalse;
     }
// Actor.
   actor.Clear();
// Input layer.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = AccountDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer = 0;
// layer 1: account/position/time candidate projection.
//    This learned FC projection turns the compact AccountDescr into the three
//    primary MPI candidates; market and RAG remain auxiliary contexts.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NScenarios * EmbeddingSize;
   descr.activation = GELU;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
// layer 2: MPI primary candidates, market second input and bound RAG third input.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronVLADriverRAGMPI;
   descr.window = EmbeddingSize;
   descr.count = StackSize;
     {
      // CNeuronVLADriverRAGMPI consumes a single frozen ScenarioEmbedding
      // through its second input and learns NHeads market K/V tokens locally.
      uint temp[] = {NScenarios, EmbeddingSize, NHeads};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.probability = ActorMPITopK;
   descr.step = NHeads;
   descr.window_out = EmbeddingSize / NHeads;
// One local action-centroid set is gathered for each selected scenario.
// The RAG-memory descriptor below uses the same Top-K, so binding is exact.
   descr.layers = ActorMPITopK * RAGActionCentroids;
   descr.variables = (NActions + 2);
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer = 2;
// layer 3.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = 1;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
// layer 4.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
// layer 5.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronConvOCL;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 3;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
// Critic.
   critic.Clear();
// Input layer.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
// layer 1.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronMHCrossFAT;
     {
      uint temp[] = {3,                      // Inputs window
                     EmbeddingSize,          // Key Dimension
                     EmbeddingSize,          // Cross window
                     EmbeddingSize           // Embedding size
                    };
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         ReturnFalse;
     }
     {
      uint temp[] = {NActions / 3,           // Query units
                     BarDescr                // Cross units
                    };
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         ReturnFalse;
     }
   descr.step = NHeads;                      // Heads
   descr.batch = 1e4;
   descr.layers = NExperts;                  // Candidates
   descr.variables = TopK;                   // Top-K
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
// layer 2.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = EmbeddingSize;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
// layer 4.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronSpikeConvBlock;
   descr.count = NActions / 3;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.variables = 1;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
// layer 5.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
// layer 6.
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = NRewards;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
   return true;
  }
//+------------------------------------------------------------------+
//| Shared RAG-memory layer shape. The memory remains library-owned; |
//+------------------------------------------------------------------+
//| Implements CreateRAGMemoryDescription                            |
bool CreateRAGMemoryDescription(CLayerDescription &descr)
  {
   descr.type = defNeuronRAGMemory;
   descr.window = EmbeddingSize;
   descr.window_out = NActions;
   descr.count = NActions + 2;
   descr.layers = ActorMPITopK;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   uint scenario_capacity[] = {RAGScenarioCentroids};
   uint action_capacity[] = {RAGActionCentroids};
   if(ArrayCopy(descr.units, scenario_capacity, 0, 0, scenario_capacity.Size()) < int(scenario_capacity.Size()) ||
      ArrayCopy(descr.heads, action_capacity, 0, 0, action_capacity.Size()) < int(action_capacity.Size()))
      ReturnFalse;
   return true;
  }
#ifndef Study
//| IsNewBar implementation.                                         |
//+------------------------------------------------------------------+
bool IsNewBar(void)
  {
   static datetime last_bar = 0;
   if(last_bar >= iTime(Symb.Name(), TimeFrame, 0))
      return false;
   last_bar = iTime(Symb.Name(), TimeFrame, 0);
   return true;
  }
//| CloseByDirection implementation.                                 |
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
   return result;
  }
//| TrailPosition implementation.                                    |
//+------------------------------------------------------------------+
bool TrailPosition(ENUM_POSITION_TYPE type, double sl, double tp)
  {
   int total = PositionsTotal();
   bool result = true;
   datetime time = TimeCurrent() - 5 * PeriodSeconds(TimeFrame);
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
         continue;
      if(PositionGetInteger(POSITION_TIME_UPDATE) > time)
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
   return result;
  }
//| ClosePartial implementation.                                     |
//+------------------------------------------------------------------+
bool ClosePartial(ENUM_POSITION_TYPE type, double value)
  {
   if(value <= 0)
      return true;
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
   return (value <= 0);
  }
#endif
//| Implements CreateBuffers                                         |
bool CreateBuffers(const int start_bar, CBufferFloat* state, CBufferFloat *time, CBufferFloat* forecast)
  {
   int total_bars = (start_bar + HistoryBars + (!!forecast ? NForecast : 0));
   if(!state || !time || start_bar < 0 ||
      total_bars > int(Rates.Size()))
      ReturnFalse;
   matrix<float> mState = matrix<float>::Zeros(BarDescr, HistoryBars);
   vector<float> vForecast = vector<float>::Zeros(NForecast * BarDescr);
   time.Clear();
   time.Reserve(HistoryBars);
   int bar = start_bar + (!!forecast ? NForecast : 0);
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b + bar].open;
      float rsi = (float)RSI.Main(b + bar);
      float cci = (float)CCI.Main(b + bar);
      float atr = (float)ATR.Main(b + bar);
      float macd = (float)MACD.Main(b + bar);
      float sign = (float)MACD.Signal(b + bar);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         ReturnFalse;
      mState[0, b] = (float)(Rates[b + bar].close - open);
      mState[1, b] = (float)(Rates[b + bar].high - open);
      mState[2, b] = (float)(Rates[b + bar].low - open);
      mState[3, b] = (float)(Rates[b + bar].tick_volume / 1000.0f);
      mState[4, b] = rsi;
      mState[5, b] = cci;
      mState[6, b] = atr;
      mState[7, b] = macd;
      mState[8, b] = sign;
      if(!time.Add(float(Rates[b + bar].time)))
         ReturnFalse;
     }
   if(!state.AssignArray(mState))
      ReturnFalse;
   if(time.GetIndex() >= 0)
      if(!time.BufferWrite())
         ReturnFalse;
   if(!forecast)
      return true;
   for(int b = 1; b <= (int)NForecast; b++)
     {
      float open = (float)Rates[bar - b].open;
      float rsi = (float)RSI.Main(bar - b);
      float cci = (float)CCI.Main(bar - b);
      float atr = (float)ATR.Main(bar - b);
      float macd = (float)MACD.Main(bar - b);
      float sign = (float)MACD.Signal(bar - b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         ReturnFalse;
      int shift = (NForecast - b) * BarDescr;
      vForecast[shift] = (float)(Rates[bar - b].close - open);
      vForecast[shift + 1] = (float)(Rates[bar - b].high - open);
      vForecast[shift + 2] = (float)(Rates[bar - b].low - open);
      vForecast[shift + 3] = (float)(Rates[bar - b].tick_volume / 1000.0f);
      vForecast[shift + 4] = rsi;
      vForecast[shift + 5] = cci;
      vForecast[shift + 6] = atr;
      vForecast[shift + 7] = macd;
      vForecast[shift + 8] = sign;
     }
   if(!forecast.AssignArray(vForecast))
      ReturnFalse;
   return true;
  }
enum ENUM_MEMORY_ACTION_NORMALIZATION_STAGE
  {
   MEMORY_ACTION_NORMALIZATION_OK = 0,
   MEMORY_ACTION_NORMALIZATION_INPUT,
   MEMORY_ACTION_NORMALIZATION_MARKET,
   MEMORY_ACTION_NORMALIZATION_VOLATILITY,
   MEMORY_ACTION_NORMALIZATION_DESCRIPTOR,
   MEMORY_ACTION_NORMALIZATION_ZERO
  };
//| MemoryActionNormalizationStageName implementation.               |
//| Returns a human-readable name for each normalization stage.      |
//+------------------------------------------------------------------+
string MemoryActionNormalizationStageName(const ENUM_MEMORY_ACTION_NORMALIZATION_STAGE stage)
  {
   switch(stage)
     {
      case MEMORY_ACTION_NORMALIZATION_INPUT:
         return "input";
      case MEMORY_ACTION_NORMALIZATION_MARKET:
         return "market";
      case MEMORY_ACTION_NORMALIZATION_VOLATILITY:
         return "volatility";
      case MEMORY_ACTION_NORMALIZATION_DESCRIPTOR:
         return "descriptor";
      case MEMORY_ACTION_NORMALIZATION_ZERO:
         return "zero";
      default:
         return "ok";
     }
  }
//+------------------------------------------------------------------+
//| NormalizeMemoryAction (with diagnostics).                        |
//+------------------------------------------------------------------+
bool NormalizeMemoryAction(CNeuronRAGMemory &memory, CBufferFloat *action, const uint start_position,
                           const double equity, float &destination[],
                           ENUM_MEMORY_ACTION_NORMALIZATION_STAGE &stage)
  {
   stage = MEMORY_ACTION_NORMALIZATION_INPUT;
   ArrayInitialize(destination, 0.0f);
   if(action == NULL || action.Total() < NActions || ArraySize(destination) < NActions ||
      equity <= 0 || start_position >= Rates.Size())
      ReturnFalse;
   stage = MEMORY_ACTION_NORMALIZATION_MARKET;
   if(!Symb.RefreshRates())
      ReturnFalse;
   const double volatility = ATR.Main(start_position);
   const double point_cost = Symb.TickValue() / Symb.TickSize();
   if(point_cost <= 0 || Symb.Point() <= 0)
      ReturnFalse;
   stage = MEMORY_ACTION_NORMALIZATION_VOLATILITY;
   if(volatility == EMPTY_VALUE || volatility <= 0)
      ReturnFalse;
   float raw[NActions];
   for(int i = 0; i < NActions; i++)
      raw[i] = action[i];
   const double risk_budget = equity * MemoryRiskBudgetFraction;
   stage = MEMORY_ACTION_NORMALIZATION_DESCRIPTOR;
   if(!memory.NormalizeActionDescriptor(raw, equity, risk_budget, point_cost, Symb.Point(),
                                        volatility, MemoryVolatilityRange, MaxTP, MaxSL, destination))
      ReturnFalse;
   double action_square = 0.0;
   for(int i = 0; i < NActions; i++)
      action_square += destination[i] * destination[i];
   stage = MEMORY_ACTION_NORMALIZATION_ZERO;
   if(action_square <= DBL_EPSILON)
      ReturnFalse;
   stage = MEMORY_ACTION_NORMALIZATION_OK;
   return true;
  }
//+------------------------------------------------------------------+
//| Preserve the existing callers that do not need diagnostics.      |
//+------------------------------------------------------------------+
bool NormalizeMemoryAction(CNeuronRAGMemory &memory, CBufferFloat *action, const uint start_position,
                           const double equity, float &destination[])
  {
   ENUM_MEMORY_ACTION_NORMALIZATION_STAGE stage;
   return NormalizeMemoryAction(memory, action, start_position, equity, destination, stage);
  }
const vector<float> SampleAccount(CBufferFloat *state, datetime time, double max_balance, double min_balance = 0)
  {
   vector<float> result = vector<float>::Zeros(AccountDescr);
   if(!state)
      return result;
   double marg = 0;
   if(!Symb.RefreshRates() || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return result;
   double buy_lot = 0, sell_lot = 0, profit = 0;
   double deal = 0;//MathRand() / (32767 * 0.5) - 1;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
   double balance = (max_balance - min_balance) * MathRand() / 32767.0 + min_balance;
   double prev_balance = balance;
   double equity = balance;
   double prev_equity = balance;
   double position_discount = 0;
   if(deal > 0)
     {
      double lot = balance / (2.0 * marg) * deal;
      if(lot >= Symb.LotsMin())
         buy_lot = MathMin(int((lot - Symb.LotsMin()) / Symb.LotsStep()) * Symb.LotsStep() + Symb.LotsMin(), 1.0);
     }
   else
      if(deal < 0)
        {
         double lot = MathAbs(balance / (2.0 * marg) * deal);
         if(lot >= Symb.LotsMin())
            sell_lot = MathMin(int((lot - Symb.LotsMin()) / Symb.LotsStep()) * Symb.LotsStep() + Symb.LotsMin(), 1.0);
        }
      else
         prev_balance += (MathRand() / (2.0 * 32767.0) - 0.25) * balance;
   if(sell_lot > 0 || buy_lot > 0)
     {
      int pos_open = int(MathRand() / 32767.0 * (state.Total() / BarDescr - 1));
      for(int i = 0; i <= pos_open; i++)
        {
         profit += state.At(i * BarDescr) / Symb.TickSize() * Symb.TickValue();
         if(((buy_lot > 0 && profit < 0) ||
             (sell_lot > 0 && profit > 0))
            && MathAbs(profit * (buy_lot - sell_lot)) > balance / 2)
           {
            pos_open = i;
            break;
           }
        }
      profit *= buy_lot - sell_lot;
      equity += profit;
      prev_equity = equity - state.At(0) / Symb.TickSize() * Symb.TickValue() * (buy_lot - sell_lot);
      position_discount = pos_open * PeriodSeconds(TimeFrame) * multiplyer * MathAbs(profit);
     }
   result[0] = float(balance / EtalonBalance);
   result[1] = float((balance - prev_balance) / prev_balance);
   result[2] = float(equity / prev_balance);
   result[3] = float((equity - prev_equity) / prev_equity);
   result[4] = float(buy_lot);
   result[5] = float(sell_lot);
   result[6] = float(buy_lot * profit / prev_balance);
   result[7] = float(sell_lot * profit / prev_balance);
   result[8] = float(position_discount / prev_balance);
   double x = time / (double)(D'2024.01.01' - D'2023.01.01');
   result[9] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_MN1);
   result[10] = float(MathCos(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_W1);
   result[11] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   x = time / (double)PeriodSeconds(PERIOD_D1);
   result[12] = float(MathSin(x != 0 ? 2.0 * M_PI * x : 0));
   return result;
  }
//+------------------------------------------------------------------+
//| UpdateTerminalPath.                                              |
//+------------------------------------------------------------------+
void UpdateTerminalPath(const double balance, const double profit, const uint bars,
                        double &peak_equity, double &max_drawdown, uint &duration)
  {
   const double equity = balance + profit;
   peak_equity = MathMax(peak_equity, equity);
   max_drawdown = MathMax(max_drawdown, peak_equity - equity);
   duration = MathMax(duration, bars);
  }
// The historical simulator is evaluated over the complete position path, not a
// one-bar balance delta. It reports normalized return, drawdown, costs, risk
// usage and duration; CheckAction below remains its scalar compatibility view.
bool EvaluateTerminalOutcome(CNeuronRAGMemory &memory, CBufferFloat *action, double balance,
                             uint start_position, SRAGTerminalOutcome &outcome)
  {
   const double risk_budget = balance * MemoryRiskBudgetFraction;
   memory.NormalizeTerminalOutcome(0, 0, 0, 0, 0, balance, risk_budget, NForecast, outcome);
   if(!action || start_position >= Rates.Size())
      ReturnFalse;
   double buy_lot = MathMax(double(action[0] - action[3]), 0);
   double sell_lot = MathMax(double(action[3] - action[0]), 0);
   double marg = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      ReturnFalse;
   double point_cost = Symb.TickValue() / Symb.TickSize();
   if(MathMax(buy_lot, sell_lot) < Symb.LotsMin())
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * balance / (2 * marg);
      memory.NormalizeTerminalOutcome(loss, 0, 0, 0, 0, balance, risk_budget, NForecast, outcome);
      return true;
     }
   if((marg * MathMax(buy_lot, sell_lot)) >= balance)
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * MathMax(buy_lot, sell_lot);
      memory.NormalizeTerminalOutcome(loss, 0, 0, 0, 0, balance, risk_budget, NForecast, outcome);
      return true;
     }
   point_cost *= MathAbs(buy_lot - sell_lot);
   double tp = 0, sl = 0, profit = 0, reward = 0;
   double peak_equity = balance;
   double max_drawdown = 0;
   double costs = 0;
   double risk_usage = 0;
   uint duration = 0;
   int stops = MathMax(Symb.StopsLevel(), 10);
   int spread = Symb.Spread();
   if(buy_lot > 0)
     {
      tp = action[1] * MaxTP;
      sl = action[2] * MaxSL;
      if(int(tp) < stops || int(sl) < (stops + spread))
        {
         double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                                Rates[start_position].open - Rates[start_position].low) *
                       point_cost * buy_lot;
         memory.NormalizeTerminalOutcome(loss, 0, 0, 0, 0, balance, risk_budget, NForecast, outcome);
         return true;
        }
      tp = (tp + spread) * Symb.Point() + Rates[start_position].open;
      sl = Rates[start_position].open - (sl + spread) * Symb.Point();
      reward = profit = -spread * Symb.Point() * point_cost;
      costs = spread * Symb.Point() * point_cost;
      risk_usage = (Rates[start_position].open - sl) * point_cost;
      for(uint i = start_position; i > 0; i--)
        {
         if(sl >= Rates[i].low)
           {
            double p = (Rates[i].open - sl) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
            break;
           }
         if(tp <= Rates[i].high)
           {
            double p = (tp - Rates[i].open) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit += p;
         reward += p * MathPow(DiscFactor, float(i - start_position));
         UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
   if(sell_lot > 0)
     {
      tp = action[4] * MaxTP;
      sl = action[5] * MaxSL;
      if(int(tp) < stops || int(sl) < (stops + spread))
        {
         double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                                Rates[start_position].open - Rates[start_position].low) *
                       point_cost * sell_lot;
         memory.NormalizeTerminalOutcome(loss, 0, 0, 0, 0, balance, risk_budget, NForecast, outcome);
         return true;
        }
      tp = Rates[start_position].open - (tp + spread) * Symb.Point();
      sl = Rates[start_position].open + (sl - spread) * Symb.Point();
      reward = profit = -spread * Symb.Point() * point_cost;
      costs = spread * Symb.Point() * point_cost;
      risk_usage = (sl - Rates[start_position].open) * point_cost;
      for(uint i = start_position; i > 0; i--)
        {
         if(sl <= Rates[i].high)
           {
            double p = (sl - Rates[i].open) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
            break;
           }
         if(tp >= Rates[i].low)
           {
            double p = (Rates[i].open - tp) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit -= p;
         reward -= p * MathPow(DiscFactor, float(i - start_position));
         UpdateTerminalPath(balance, profit, start_position - i + 1, peak_equity, max_drawdown, duration);
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
   memory.NormalizeTerminalOutcome(profit, max_drawdown, costs, risk_usage, duration,
                                   balance, risk_budget, NForecast, outcome);
   return true;
  }
//+------------------------------------------------------------------+
//| CheckAction.                                                     |
//+------------------------------------------------------------------+
double CheckAction(CNeuronRAGMemory &memory, CBufferFloat *action, double balance,
                   uint start_position)
  {
   SRAGTerminalOutcome outcome;
   EvaluateTerminalOutcome(memory, action, balance, start_position, outcome);
   return outcome.reward;
  }
vector<float> OraculAction(const vector<float> &account, CBufferFloat *forecat)
  {
// Look for target.
   vector<float> result = vector<float>::Zeros(NActions);
   matrix<float> fstate = matrix<float>::Zeros(NForecast, BarDescr);
   if(!forecat.GetData(fstate))
      return result;
   vector<float> target = fstate.Col(0).CumSum();
   if(account[4] > account[5])
     {
      float tp = 0;
      float sl = 0;
      float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * Point());
      int pos = 0;
      for(int j = 0; j < NForecast; j++)
        {
         tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
         pos = j;
         if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
            break;
         sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
        }
      if(pos > 0 && tp > 0)
        {
         sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01));
         tp = float(MathMax(MathMin(tp / (MaxTP * Point()), 1), 0.01));
         result[0] = MathMax(result[0] - result[3], 0.011f);
         result[5] = result[1] = tp;
         result[4] = result[2] = sl;
         result[3] = 0;
        }
     }
   else
     {
      if(account[4] < account[5])
        {
         float tp = 0;
         float sl = 0;
         float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * Point());
         int pos = 0;
         for(int j = 0; j < NForecast; j++)
           {
            tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
            pos = j;
            if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
               break;
            sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
           }
         if(pos > 0 && tp < 0)
           {
            sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01));
            tp = float(MathMax(MathMin(-tp / (MaxTP * Point()), 1), 0.01));
            result[3] = MathMax(result[3] - result[0], 0.011f);
            result[2] = result[4] = tp;
            result[1] = result[5] = sl;
            result[0] = 0;
           }
        }
      else
        {
         ulong argmin = target.ArgMin();
         ulong argmax = target.ArgMax();
         float max_sl = float(MaxSL * Point());
         double equity = account[2] * account[0] * EtalonBalance / (1 + account[1]);
         while(argmax > 0 && argmin > 0)
           {
            if(argmax < argmin && target[argmax] / 2 > MathAbs(target[argmin]) && MathAbs(target[argmin]) < max_sl)
               break;
            if(argmax > argmin && target[argmax] < MathAbs(target[argmin] / 2) && target[argmax] < max_sl)
               break;
            target.Resize(MathMin(argmax, argmin));
            argmin = target.ArgMin();
            argmax = target.ArgMax();
           }
         if(argmin == 0 || (argmax < argmin && argmax > 0))
           {
            float tp = 0;
            float sl = 0;
            float cur_sl = - float(MaxSL * Point());
            ulong pos = 0;
            for(ulong j = 0; j < argmax; j++)
              {
               tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
               pos = j;
               if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
                  break;
               sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
              }
            if(pos > 0 && tp > 0)
              {
               sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
               tp = (float)MathMin(tp / (MaxTP * Point()), 1);
               result[0] = float(MathMax(equity / 100 * 0.01, 0.011));
               result[5] = result[1] = tp;
               result[4] = result[2] = sl;
               result[3] = 0;
              }
           }
         else
           {
            if(argmax == 0 || argmax > argmin)
              {
               float tp = 0;
               float sl = 0;
               float cur_sl = float(MaxSL * Point());
               ulong pos = 0;
               for(ulong j = 0; j < argmin; j++)
                 {
                  tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
                  pos = j;
                  if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
                     break;
                  sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
                 }
               if(pos > 0 && tp < 0)
                 {
                  sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                  tp = (float)MathMin(-tp / (MaxTP * Point()), 1);
                  result[3] = float(MathMax(equity / 100 * 0.01, 0.011));
                  result[2] = result[4] = tp;
                  result[1] = result[5] = sl;
                  result[0] = 0;
                 }
              }
           }
        }
     }
   return result;
  }


//| End of Trajectory.mqh                                            |
