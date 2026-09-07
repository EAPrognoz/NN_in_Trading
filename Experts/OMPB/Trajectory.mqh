//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| D2Skill learning lifecycle stage.                                |
//+------------------------------------------------------------------+
enum ENUM_D2SKILL_STAGE
  {
   D2Skill_D2_STAGE_BASE = 0,      //Base model
   D2Skill_D2_STAGE_FORMATION,     //Skill-bank formation
   D2Skill_D2_STAGE_HINDSIGHT,     //Paired hindsight utility
   D2Skill_D2_STAGE_AUGMENTED,     //Utility-aware skill augmentation
   D2Skill_D2_STAGE_ONLINE         //Online calibration
  };
//+------------------------------------------------------------------+
//| D2Skill bank selection.                                          |
//+------------------------------------------------------------------+
enum ENUM_D2SKILL_BANK_MODE
  {
   D2Skill_D2_MODE_BASE = 0,       //Without D2 banks
   D2Skill_D2_MODE_TASK,           //Task skill bank
   D2Skill_D2_MODE_STEP,           //Step skill bank
   D2Skill_D2_MODE_FULL            //D2 full mode
  };
//+------------------------------------------------------------------+
//| OMPB lifecycle stage.  Stage 02 is intentionally a scaffold in |
//| this block; only the MarketEncoder stage is executable here.    |
//+------------------------------------------------------------------+
enum ENUM_OMPB_STAGE
  {
   OMPB_STAGE_MARKET_ENCODER = 1,
   OMPB_STAGE_CALIBRATION,
   OMPB_STAGE_BASE_POLICY,
   OMPB_STAGE_D2SKILL,
   OMPB_STAGE_OPTIMIZATION,
   OMPB_STAGE_INFERENCE
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input group                "---- Indicators ----"
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;    //Working timeframe
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
//---
input group                "---- D2Skill ----"
input ENUM_D2SKILL_BANK_MODE InpD2SkillMode = D2Skill_D2_MODE_FULL; //Skill bank mode
input bool                 InpD2SkillUtilityAware = true; //Utility-aware retrieval
input double               InpD2SkillMinUtility = -1.0; //Minimum utility
input double               InpD2SkillUtilityScale = 1.0; //Utility scale
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int            iLatentLayer = -1;
int            iStateRawForecastLayer = 5;
int            iStateTokenLayer = 6;
//---
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
#define        FileName                "OMPB"
#define        OMPB_LOG_PREFIX         "OMPB"
#define        LatentCount             64
#define        LatentLayer             iLatentLayer
#define        StateRawForecastLayer   iStateRawForecastLayer
#define        StateTokenLayer         iStateTokenLayer
#define        ForecastTokenDim        (EmbeddingSize + 1)
#define        MaxSL                   1000
#define        MaxTP                   1000
#define        ActorUpdate             5
#define        TargetUpdate            24*5
#define        tau                     0.9f
#define        NHeads                  4
#define        NExperts                5
#define        NScenarios              21
#define        TopK                    5
#define        OMPBSamples              5
#define        OMPBReferenceSize        256
#define        OMPBCurrentWindow        64
#define        Quantiles               8
#define        StackSize               24*21
#define        Blocks                  24
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolInfo          Symb;
CTrade               Trade;
MqlRates             Rates[];
//---
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//+------------------------------------------------------------------+
//|                                                                  |
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
//| Complete outcome of one independently simulated D2Skill episode.|
//+------------------------------------------------------------------+
struct SD2SkillEpisodeOutcome
  {
private:
   double            dOutcome;
   double            dBalance;
   double            dEquity;
   double            dDrawdown;
   double            dCost;
   double            dRisk;
   double            dPeakEquity;
   uint              uDuration;

public:
   bool              Reset(void)
     {
      dOutcome = 0.0;
      dBalance = 0.0;
      dEquity = 0.0;
      dDrawdown = 0.0;
      dCost = 0.0;
      dRisk = 0.0;
      dPeakEquity = 0.0;
      uDuration = 0;
      return(true);
     }
   bool              Accumulate(const double reward, const double balance,
                                const double equity, const double drawdown,
                                const double cost, const uint duration,
                                const double risk = 0.0)
     {
      if(!MathIsValidNumber(reward) || !MathIsValidNumber(balance) ||
         !MathIsValidNumber(equity) || !MathIsValidNumber(drawdown) ||
         !MathIsValidNumber(cost) || !MathIsValidNumber(risk))
         ReturnFalse;
      dOutcome += reward;
      dBalance = balance;
      dEquity = equity;
      dPeakEquity = MathMax(dPeakEquity, equity);
      const double peak_drawdown = (dPeakEquity > 0.0 ?
                                    (dPeakEquity - equity) / dPeakEquity : 0.0);
      dDrawdown = MathMax(dDrawdown, MathMax(MathAbs(drawdown), peak_drawdown));
      dCost += MathAbs(cost);
      dRisk = MathMax(dRisk, MathAbs(risk));
      uDuration += duration;
      return(true);
     }
   double            Outcome(void) const { return(dOutcome); }
   double            Balance(void) const { return(dBalance); }
   double            Equity(void) const { return(dEquity); }
   double            Drawdown(void) const { return(dDrawdown); }
   double            Cost(void) const { return(dCost); }
   double            Risk(void) const { return(dRisk); }
   uint              Duration(void) const { return(uDuration); }
  };
//+------------------------------------------------------------------+
//| Calculates terminal paired utility from complete episode state.  |
//+------------------------------------------------------------------+
bool D2SkillComputePairedEpisodeDelta(const SD2SkillEpisodeOutcome &base,
                                      const SD2SkillEpisodeOutcome &skill,
                                      double &delta_j)
  {
   const double base_outcome = base.Outcome();
   const double skill_outcome = skill.Outcome();
   if(!MathIsValidNumber(base_outcome) || !MathIsValidNumber(skill_outcome))
      ReturnFalse;
   delta_j = skill_outcome - base_outcome;
   return(MathIsValidNumber(delta_j));
  }
//+------------------------------------------------------------------+
//| Validates a complete, synchronous paired terminal outcome.       |
//+------------------------------------------------------------------+
bool D2SkillValidatePairedEpisode(const SD2SkillEpisodeOutcome &base,
                                  const SD2SkillEpisodeOutcome &skill)
  {
   if(!MathIsValidNumber(base.Outcome()) || !MathIsValidNumber(skill.Outcome()) ||
      base.Duration() == 0 || skill.Duration() == 0 ||
      base.Duration() != skill.Duration())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns whether a checkpoint requests a later boundary save.     |
//+------------------------------------------------------------------+
bool D2SkillCheckpointDue(const ulong transitions, const int checkpoint_interval)
  {
   return(checkpoint_interval > 0 && transitions > 0 &&
          transitions % (ulong)checkpoint_interval == 0);
  }
//+------------------------------------------------------------------+
//| DeltaJ may close only at a terminal or configured pair horizon.  |
//+------------------------------------------------------------------+
bool D2SkillOnlinePairBoundary(const bool real_terminal, const bool virtual_terminal,
                               const ulong pair_transitions, const int pair_horizon)
  {
   if(real_terminal || virtual_terminal)
      return(true);
   return(pair_horizon > 0 && pair_transitions >= (ulong)pair_horizon);
  }
//+------------------------------------------------------------------+
//| An independent pair ends as soon as either branch is terminal.  |
//+------------------------------------------------------------------+
bool D2SkillPairReachedTerminal(const bool base_terminal, const bool skill_terminal)
  {
   return(base_terminal || skill_terminal);
  }
//+------------------------------------------------------------------+
//| Implements SState.                                               |
//+------------------------------------------------------------------+
SState::SState(void)
  {
   ArrayInitialize(state, 0);
   ArrayInitialize(account, 0);
   ArrayInitialize(action, 0);
   ArrayInitialize(rewards, 0);
  }
//+------------------------------------------------------------------+
//| Saves                                                            |
//+------------------------------------------------------------------+
bool SState::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
//---
   int total = ArraySize(state);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, state[i]) < sizeof(float))
         ReturnFalse;
//---
   total = ArraySize(account);
   if(FileWriteInteger(file_handle, total) < sizeof(int))
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(FileWriteFloat(file_handle, account[i]) < sizeof(float))
         ReturnFalse;
//---
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
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Loads.                                                           |
//+------------------------------------------------------------------+
bool SState::Load(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
//---
   int total = FileReadInteger(file_handle);
   if(total != ArraySize(state))
      ReturnFalse;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      state[i] = FileReadFloat(file_handle);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(account))
      ReturnFalse;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      account[i] = FileReadFloat(file_handle);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(action))
      ReturnFalse;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      action[i] = MathMin(MathMax(FileReadFloat(file_handle), 0), 1);
     }
//---
   total = FileReadInteger(file_handle);
   if(total != ArraySize(rewards))
      ReturnFalse;
//---
   for(int i = 0; i < total; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      rewards[i] = FileReadFloat(file_handle);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CreateStateDescriptions(CArrayObj *&encoder,
                             CArrayObj *&decoder
                            )
  {
//---
   CLayerDescription *descr;
//---
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
//--- State Encoder
   encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = HistoryBars * BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 1
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 2
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronDropoutOCL;
   descr.count = prev_count;
   descr.probability = 0.1f;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 3
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverData;
   descr.window = BarDescr;
   descr.count = HistoryBars;
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
//--- layer 4
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverRankTCM;
   descr.window = BarDescr * (2 * Quantiles + 1);
   descr.count = EmbeddingSize;
   descr.variables = HistoryBars;
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
//--- layer 5
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverForecastHead;
   descr.window = EmbeddingSize;
   descr.count = NForecast;
   descr.variables = HistoryBars;
     {
      uint temp[] = {Blocks, NHeads};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
   iStateRawForecastLayer = 5;
//--- layer 6
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronCogDriverForecastToken;
   descr.window = EmbeddingSize;
   descr.count = NForecast;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
      DeleteObjAndFalse(descr);
   iStateTokenLayer = 6;
//--- Forecast Decoder
   decoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = NForecast * EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 1
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
//--- layer 2
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
//--- layer 3
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
//--- layer 4
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBatchNormOCL;
   descr.count = NForecast * BarDescr;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
      DeleteObjAndFalse(descr);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *&actor,
                        CArrayObj *&critic
                       )
  {
//---
   CLayerDescription *descr;
//---
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
//--- Actor
   actor.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   uint prev_count = descr.count = AccountDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer = 0;
//--- layer 1
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NScenarios * EmbeddingSize;
   descr.activation = GELU;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 2
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronMomADMPI;
   descr.window = EmbeddingSize;
   descr.count = StackSize;
     {
      uint temp[] = {NScenarios, ForecastTokenDim, NForecast};
      if(ArrayCopy(descr.units, temp, 0, 0, temp.Size()) < int(temp.Size()))
         ReturnFalse;
     }
   descr.probability = TopK;
   descr.step = NHeads;
   descr.window_out = EmbeddingSize / NHeads;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
   iLatentLayer = 2;
//--- layer 3
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
//--- layer 4
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 5
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
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   descr.count = NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 1
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronMHCrossFAT;
     {
      uint temp[] = {3,                      // Inputs window
                     EmbeddingSize,          // Key Dimension
                     ForecastTokenDim,       // Cross window
                     EmbeddingSize           // Embedding size
                    };
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         ReturnFalse;
     }
     {
      uint temp[] = {NActions / 3,           // Query units
                     NForecast               // Cross units
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
//--- layer 2
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
//--- layer 4
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
//--- layer 5
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//--- layer 6
   if(!(descr = new CLayerDescription()))
      DeleteObjAndFalse(descr);
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = NRewards;
   descr.activation = None;
   descr.batch = BatchSize;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
      DeleteObjAndFalse(descr);
//---
   return(true);
  }
#ifndef Study
//+------------------------------------------------------------------+
//| Checks NewBar.                                                   |
//+------------------------------------------------------------------+
bool IsNewBar(void)
  {
   static datetime last_bar = 0;
   if(last_bar >= iTime(Symb.Name(), TimeFrame, 0))
      return(false);
//---
   last_bar = iTime(Symb.Name(), TimeFrame, 0);
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for CloseByDirection.       |
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
   return(result);
  }
//+------------------------------------------------------------------+
//| Implements TrailPosition.                                        |
//+------------------------------------------------------------------+
bool TrailPosition(ENUM_POSITION_TYPE type, double sl, double tp)
  {
   int total = PositionsTotal();
   bool result = true;
   datetime time = TimeCurrent() - 5 * PeriodSeconds(TimeFrame);
//---
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
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for ClosePartial.           |
//+------------------------------------------------------------------+
bool ClosePartial(ENUM_POSITION_TYPE type, double value)
  {
   if(value <= 0)
      return(true);
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
   return(value <= 0);
  }
#endif
//+------------------------------------------------------------------+
//|                                                                  |
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
//| Creates and manages object lifecycle for CDeal.                  |
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
//| Implements Action.                                               |
//+------------------------------------------------------------------+
vector<float> CDeal::Action(datetime current, double ask, double bid, int period_seconds)
  {
   vector<float> result = vector<float>::Zeros(NActions);
   if((OpenTime - period_seconds) > current || CloseTime <= current)
      return(result);
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
   return(result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//| Loads Deals.                                                     |
//+------------------------------------------------------------------+
bool CDeals::LoadDeals(string file_name, string symbol, double point)
  {
   if(file_name == NULL || !FileIsExist(file_name, FILE_COMMON))
     {
      PrintFormat("File %s not exist", file_name);
      ReturnFalse;
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
      ReturnFalse;
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
         ReturnFalse;
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
         ReturnFalse;
        }
     }
//---
   FileClose(handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements Action.                                               |
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
   return(result);
  }
//+------------------------------------------------------------------+
//| Creates or initializes Buffers.                                  |
//+------------------------------------------------------------------+
bool CreateBuffers(const int start_bar, CBufferFloat* state, CBufferFloat *time, CBufferFloat* forecast)
  {
   int total_bars = (start_bar + HistoryBars + (!!forecast ? NForecast : 0));
   if(!state || !time || start_bar < 0 ||
      total_bars > int(Rates.Size()))
      ReturnFalse;
//---
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
      //---
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
      return(true);
//---
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
      //---
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
//---
   if(!forecast.AssignArray(vForecast))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SampleAccount.                                        |
//+------------------------------------------------------------------+
const vector<float> SampleAccount(CBufferFloat *state, datetime time, double max_balance, double min_balance = 0)
  {
   vector<float> result = vector<float>::Zeros(AccountDescr);
   if(!state)
      return(result);
//---
   double marg = 0;
   if(!Symb.RefreshRates() || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return(result);
   double buy_lot = 0, sell_lot = 0, profit = 0;
   double deal = 0;//MathRand() / (32767 * 0.5) - 1;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
//---
   double balance = (max_balance - min_balance) * MathRand() / 32767.0 + min_balance;
   double prev_balance = balance;
   double equity = balance;
   double prev_equity = balance;
   double position_discount = 0;
//---
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
//---
   if(sell_lot > 0 || buy_lot > 0)
     {
      int pos_open = int(MathRand() / 32767.0 * (state.Total() / BarDescr - 1));
      for(int i = 0; i <= pos_open; i++)
        {
         profit += state.At(i * BarDescr) / Symb.TickSize() * Symb.TickValue();
         if(((buy_lot > 0 && profit < 0) ||
             (sell_lot > 0 && profit > 0))
            // Implements MathAbs.
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
//---
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
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for CheckAction.            |
//+------------------------------------------------------------------+
double CheckAction(CBufferFloat *action, double balance, uint start_position)
  {
   if(!action || start_position >= Rates.Size())
      return(0);
//---
   double buy_lot = MathMax(double(action[0] - action[3]), 0);
   double sell_lot = MathMax(double(action[3] - action[0]), 0);
   double marg = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return(0);
   double point_cost = Symb.TickValue() / Symb.TickSize();
   if(MathMax(buy_lot, sell_lot) < Symb.LotsMin())
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * balance / (2 * marg);
      return(loss);
     }
   if((marg * MathMax(buy_lot, sell_lot)) >= balance)
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * MathMax(buy_lot, sell_lot);
      return(loss);
     }
   point_cost *= MathAbs(buy_lot - sell_lot);
//---
   double tp = 0, sl = 0, profit = 0, reward = 0;
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
         return(loss);
        }
      tp = (tp + spread) * Symb.Point() + Rates[start_position].open;
      sl = Rates[start_position].open - (sl + spread) * Symb.Point();
      reward = profit = -spread * Symb.Point() * point_cost;
      for(uint i = start_position; i > 0; i--)
        {
         if(sl >= Rates[i].low)
           {
            double p = (Rates[i].open - sl) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         if(tp <= Rates[i].high)
           {
            double p = (tp - Rates[i].open) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit += p;
         reward += p * MathPow(DiscFactor, float(i - start_position));
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
//---
   if(sell_lot > 0)
     {
      tp = action[4] * MaxTP;
      sl = action[5] * MaxSL;
      if(int(tp) < stops || int(sl) < (stops + spread))
        {
         double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                                Rates[start_position].open - Rates[start_position].low) *
                       point_cost * sell_lot;
         return(loss);
        }
      tp = Rates[start_position].open - (tp + spread) * Symb.Point();
      sl = Rates[start_position].open + (sl - spread) * Symb.Point();
      for(uint i = start_position; i > 0; i--)
        {
         if(sl <= Rates[i].high)
           {
            double p = (sl - Rates[i].open) * point_cost;
            profit -= p;
            reward -= p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         if(tp >= Rates[i].low)
           {
            double p = (Rates[i].open - tp) * point_cost;
            profit += p;
            reward += p * MathPow(DiscFactor, float(i - start_position));
            break;
           }
         double p = (Rates[i - 1].open - Rates[i].open) * point_cost;
         profit -= p;
         reward -= p * MathPow(DiscFactor, float(i - start_position));
         if(-profit >= balance)
           {
            reward -= 1000;
            break;
           }
        }
     }
//---
   return(reward);
  }
//+------------------------------------------------------------------+
//| Implements OraculAction.                                         |
//+------------------------------------------------------------------+
vector<float> OraculAction(const vector<float> &account, CBufferFloat *forecat)
  {
//--- Look for target
   vector<float> result = vector<float>::Zeros(NActions);
   matrix<float> fstate = matrix<float>::Zeros(NForecast, BarDescr);
   if(!forecat.GetData(fstate))
      return(result);
//---
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
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| D2Skill ScenarioForecast and Actor/Critic runtime                |
//+------------------------------------------------------------------+
bool                 D2SkillRecoverySmoke = false;
uint                 D2SkillRecoverySmokeBatches = 2000;
uint                 D2SkillRecoverySmokeAge = 512;
const uint           D2Skill_FORMAT_VERSION = 7;
const string         D2Skill_MARKET_FILE = "OMPBMarket.nnw";
const string         D2Skill_TARGET_FILE = "OMPBTarget.nnw";
const string         D2Skill_MANIFEST_FILE = "OMPBForecast.manifest";
const string         D2Skill_ACTOR_FILE = "OMPBActor.nnw";
const string         D2Skill_Q1_FILE = "OMPBQ1.nnw";
const string         D2Skill_Q2_FILE = "OMPBQ2.nnw";
const string         D2Skill_AC_MANIFEST_FILE = "OMPBActorCritic.manifest";
const uint           D2Skill_AC_FORMAT_VERSION = 13;
const string         D2Skill_OOS_FILE = "OMPBOOS.csv";
const string         D2Skill_OOS_MANIFEST_FILE = "OMPBOOS.manifest";
//--- Stage 02 publishes only the canonical production tuple.  A complete
//--- next tuple is validated first; previous is retained only until reload
//--- proof accepts the canonical manifest as the commit point.
const string         OMPB_STAGE02_MARKET_NEXT_FILE = "OMPBMarket.next.nnw";
const string         OMPB_STAGE02_TARGET_NEXT_FILE = "OMPBTarget.next.nnw";
const string         OMPB_STAGE02_MANIFEST_NEXT_FILE = "OMPBForecast.next.manifest";
const string         OMPB_STAGE02_MARKET_PREVIOUS_FILE = "OMPBMarket.previous.nnw";
const string         OMPB_STAGE02_TARGET_PREVIOUS_FILE = "OMPBTarget.previous.nnw";
const string         OMPB_STAGE02_MANIFEST_PREVIOUS_FILE = "OMPBForecast.previous.manifest";
//--- Stage 01 uses an independent transaction namespace.  The short names
//--- remain the only normal load contract; manifest is its commit record.
const string         OMPB_STAGE01_MARKET_NEXT_FILE = "OMPBMarket.stage01.next.nnw";
const string         OMPB_STAGE01_TARGET_NEXT_FILE = "OMPBTarget.stage01.next.nnw";
const string         OMPB_STAGE01_MANIFEST_NEXT_FILE = "OMPBForecast.stage01.next.manifest";
const string         OMPB_STAGE01_MARKET_PREVIOUS_FILE = "OMPBMarket.stage01.previous.nnw";
const string         OMPB_STAGE01_TARGET_PREVIOUS_FILE = "OMPBTarget.stage01.previous.nnw";
const string         OMPB_STAGE01_MANIFEST_PREVIOUS_FILE = "OMPBForecast.stage01.previous.manifest";
const string         OMPB_STAGE01_TRANSACTION_FILE = "OMPBForecast.stage01.transaction";
const uint           OMPB_STAGE01_TRANSACTION_VERSION = 1;
//--- Legacy generation selectors are migration markers only.  They are never
//--- resolved by a normal loader, which fails closed until an explicit migration.
const string         OMPB_STAGE02_ACTIVE_SELECTOR = "OMPBForecast.stage02.active";
const string         OMPB_STAGE02_SELECTOR_NEXT = "OMPBForecast.stage02.active.next";
const string         OMPB_STAGE02_SELECTOR_PREVIOUS = "OMPBForecast.stage02.active.previous";
const string         OMPB_STAGE02_SELECTOR_RESTORE = "OMPBForecast.stage02.active.restore";
const string         OMPB_STAGE02_GENERATION_PREFIX = ".stage02.";
const string         OMPB_STAGE02_LEGACY_TRANSACTION_FILE = "OMPBForecast.stage02.transaction";
const string         OMPB_STAGE02_LEGACY_TRANSACTION_NEXT_FILE = "OMPBForecast.stage02.transaction.next";
const uint           OMPB_STAGE02_SELECTOR_VERSION = 1;
//--- Normal loading always uses these canonical physical production names.
string               D2SkillActiveMarketFile = D2Skill_MARKET_FILE;
string               D2SkillActiveTargetFile = D2Skill_TARGET_FILE;
string               D2SkillActiveManifestFile = D2Skill_MANIFEST_FILE;
//--- Utility is defined only by the counterfactual paired terminal outcome.
const int            D2Skill_D2_UTILITY_PAIRED_HINDSIGHT = 0;
//+------------------------------------------------------------------+
//| D2Skill builds transitions directly from historical windows      |
//| Base CNet already provides Layer(int) and                        |
//| FeedForwardLayer(CNeuronBaseOCL*).                               |
//| A transient, non-owning layer view lets the library transpose    |
//+------------------------------------------------------------------+
class CD2SkillBufferView : public CNeuronBaseOCL
  {
public:
   //+------------------------------------------------------------------+
   //| Implements Bind.                                                 |
   //+------------------------------------------------------------------+
   bool              Bind(CBufferFloat *source)
     {
      if(!source || source.GetIndex() < 0)
         ReturnFalse;
      if(Output != source)
         DeleteObj(Output);
      Output = source;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Implements Unbind.                                               |
   //+------------------------------------------------------------------+
   void              Unbind(void)
     {
      Output = NULL;
     }
  };
//+------------------------------------------------------------------+
//| Device-only composition adapter, all operations reuse existing   |
//+------------------------------------------------------------------+
class CD2SkillDeviceOps : public CNeuronBaseOCL
  {
public:
   //+------------------------------------------------------------------+
   //| Implements Bind.                                                 |
   //+------------------------------------------------------------------+
   bool              Bind(COpenCLMy *open_cl)
     {
      OpenCL = open_cl;
      return(CheckPointer(OpenCL) != POINTER_INVALID);
     }
   //+------------------------------------------------------------------+
   //| Creates and manages object lifecycle for Copy.                   |
   //+------------------------------------------------------------------+
   bool              Copy(CBufferFloat *source, CBufferFloat *destination, const uint total)
     {
      if(!source || !destination || source.Total() != int(total) ||
         destination.Total() != int(total) || source.GetIndex() < 0 ||
         destination.GetIndex() < 0 || source.GetOpenCL() != OpenCL ||
         destination.GetOpenCL() != OpenCL)
         ReturnFalse;
      //---
      return(CopyBufferRaw(source, destination, total));
     }
   bool              Join2(CBufferFloat *first, const uint first_total, CBufferFloat *second,
                           const uint second_total, CBufferFloat *destination)
     {
      if(!first || !second || !destination || first.Total() != int(first_total) ||
         second.Total() != int(second_total) ||
         destination.Total() != int(first_total + second_total) || first.GetIndex() < 0 ||
         second.GetIndex() < 0 || destination.GetIndex() < 0)
         ReturnFalse;
      return(Concat(first, second, destination, first_total, second_total, 1));
     }
   bool              Join4(CBufferFloat *first, CBufferFloat *second, CBufferFloat *third,
                           CBufferFloat *fourth, CBufferFloat *destination, const uint block)
     {
      if(!first || !second || !third || !fourth || !destination ||
         first.Total() != int(block) || second.Total() != int(block) ||
         third.Total() != int(block) || fourth.Total() != int(block) ||
         destination.Total() != int(4 * block) || first.GetIndex() < 0 || second.GetIndex() < 0 ||
         third.GetIndex() < 0 || fourth.GetIndex() < 0 || destination.GetIndex() < 0)
         ReturnFalse;
      return(Concat(first, second, third, fourth, destination, block, block, block, block, 1));
     }
   bool              Subtract(CBufferFloat *first, CBufferFloat *second, CBufferFloat *destination,
                              const uint dimension)
     {
      return(Different(first, second, destination, dimension));
     }
   bool              BroadcastSum(CBufferFloat *vector_in, CBufferFloat *matrix_in,
                                  CBufferFloat *destination, const uint dimension, const uint variables)
     {
      return(SumVecMatrix(vector_in, matrix_in, destination, dimension, variables));
     }
   bool              Add(CBufferFloat *first, CBufferFloat *second, CBufferFloat *destination,
                         const uint dimension)
     {
      return(SumAndNormalize(first, second, destination, dimension, false, 0, 0, 0, 1.0f));
     }
   bool              Split2(CBufferFloat *first, CBufferFloat *second, CBufferFloat *source,
                            const uint first_total, const uint second_total)
     {
      return(DeConcat(first, second, source, first_total, second_total, 1));
     }
  };
//---
CNet                 D2SkillMarket;
CNet                 D2SkillTarget;
CNeuronScenarioForecast *D2SkillForecast = NULL;
//+------------------------------------------------------------------+
//| Resolves RankTCM at its fixed OMPB graph slot.                  |
//+------------------------------------------------------------------+
CNeuronBaseOCL *GetRankTCM(void)
  {
   //--- Resolve RankTCM only when the fixed OMPB graph slot has its declared type.
   CNeuronBaseOCL *layer = D2SkillMarket.Layer(4);
   //--- Return the typed predecessor or NULL for a stale or incompatible graph.
   return(layer && layer.Type() == defNeuronCogDriverRankTCM ? layer : NULL);
  }
//+------------------------------------------------------------------+
//| Resolves OMPB at its fixed MarketEncoder graph slot.            |
//+------------------------------------------------------------------+
CNeuronOMPBOCL *GetOMPB(void)
  {
   //--- Resolve OMPB only when the fixed MarketEncoder graph slot has its declared type.
   CNeuronBaseOCL *layer = D2SkillMarket.Layer(5);
   //--- Return the typed layer or NULL for a stale or incompatible graph.
   return(layer && layer.Type() == defNeuronOMPBOCL ? (CNeuronOMPBOCL *)layer : NULL);
  }
//+------------------------------------------------------------------+
//| Sets an OMPB runtime mode after clearing source-anchor state.    |
//+------------------------------------------------------------------+
bool SetOMPBMode(const ENUM_OMPB_MODE mode)
  {
   //--- Clear source-anchor state before selecting the requested OMPB runtime mode.
   CNeuronOMPBOCL *layer = GetOMPB();
   //--- Finalize the mode change only for a validated OMPB layer.
   return(layer && layer.SetSourceAnchor(false) && layer.SetMode(mode));
  }
//+------------------------------------------------------------------+
//| Configures OMPB mode and BYPASS training safeguards.            |
//+------------------------------------------------------------------+
bool ConfigureOMPB(const ENUM_OMPB_MODE mode)
  {
   //--- Resolve the layer and install the requested OMPB runtime mode.
   CNeuronOMPBOCL *layer = GetOMPB();
   if(!layer || !SetOMPBMode(mode))
      ReturnFalse;
   //--- BYPASS must not retain a trainable OMPB layer during MarketEncoder training.
   if(mode == OMPB_BYPASS)
      layer.TrainMode(false);
   //--- Finalize the OMPB configuration after the runtime safeguards are applied.
   return(true);
  }
//+------------------------------------------------------------------+
//| Resets only the OMPB immutable Reference history.               |
//+------------------------------------------------------------------+
bool ResetOMPBReference(void)
  {
   //--- Resolve OMPB before discarding its immutable Reference history explicitly.
   CNeuronOMPBOCL *layer = GetOMPB();
   //--- Finalize only when the validated layer reset succeeds.
   return(layer && layer.ResetReference());
  }
//+------------------------------------------------------------------+
//| Reads OMPB runtime diagnostic counters and regularizer values.  |
//+------------------------------------------------------------------+
bool ReadOMPBDiagnostics(uint &reference_count, uint &current_count,
                         float &disagreement, float &kl, float &alpha_prior,
                         uint &invalid_fallbacks, uint &kl_rejects)
  {
   //--- Resolve OMPB before exporting its runtime-only diagnostic counters.
   CNeuronOMPBOCL *layer = GetOMPB();
   if(!layer)
      ReturnFalse;
   //--- Read the immutable Reference and rolling Current occupancy counters.
   reference_count = layer.ReferenceCount();
   current_count = layer.CurrentCount();
   //--- Read the latest regularizer values and guarded-inference counters.
   disagreement = layer.LastDisagreement();
   kl = layer.LastKL();
   alpha_prior = layer.LastAlphaPrior();
   invalid_fallbacks = layer.InvalidFallbacks();
   kl_rejects = layer.KLRejects();
   //--- Finalize the diagnostics snapshot after all values have been read.
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks that BYPASS leaves OMPB diagnostics unchanged.           |
//+------------------------------------------------------------------+
bool D2SkillCheckOMPBBypassInvariant(const uint reference_count_before,
                                     const uint current_count_before,
                                     const float disagreement_before,
                                     const float kl_before,
                                     const float alpha_prior_before,
                                     const uint invalid_fallbacks_before,
                                     const uint kl_rejects_before,
                                     const string scope)
  {
   uint reference_count_after, current_count_after, invalid_fallbacks_after, kl_rejects_after;
   float disagreement_after, kl_after, alpha_prior_after;
   if(!ReadOMPBDiagnostics(reference_count_after, current_count_after, disagreement_after,
                           kl_after, alpha_prior_after, invalid_fallbacks_after, kl_rejects_after))
     {
      PrintFormat("OMPB_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=%s reason=read_after", scope);
      return(false);
     }
   const bool unchanged = (reference_count_after == reference_count_before &&
                           current_count_after == current_count_before &&
                           disagreement_after == disagreement_before &&
                           kl_after == kl_before &&
                           alpha_prior_after == alpha_prior_before &&
                           invalid_fallbacks_after == invalid_fallbacks_before &&
                           kl_rejects_after == kl_rejects_before);
   if(!unchanged)
     {
      PrintFormat("OMPB_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=%s " +
                  "before=(%u,%u,%.9g,%.9g,%.9g,%u,%u) " +
                  "after=(%u,%u,%.9g,%.9g,%.9g,%u,%u)",
                  scope, reference_count_before, current_count_before, disagreement_before,
                  kl_before, alpha_prior_before, invalid_fallbacks_before, kl_rejects_before,
                  reference_count_after, current_count_after, disagreement_after, kl_after,
                  alpha_prior_after, invalid_fallbacks_after, kl_rejects_after);
      return(false);
     }
   PrintFormat("OMPB_STAGE01_BYPASS_DIAGNOSTIC_PASS scope=%s state_unchanged=true", scope);
   return(true);
  }
//---
CBufferFloat         D2SkillState;
CBufferFloat         D2SkillTime;
CBufferFloat         D2SkillFuture;
CBufferFloat         D2SkillLatentTarget;
CBufferFloat         D2SkillLatentDelta;
CBufferFloat         D2SkillProbeState;
CBufferFloat         D2SkillProbeTime;
CD2SkillBufferView     D2SkillFutureView;
CNeuronTransposeRCDOCL D2SkillFutureTranspose;
CBufferFloat         D2SkillLatentZero;
CBufferFloat         D2SkillLatentNegativeMarket;
//---
ulong                D2SkillInvalidBatches = 0;
ulong                D2SkillBatches = 0;
ulong                D2SkillResponsibilityMicroseconds = 0;
uint                 D2SkillCompletedEpochs = 0;
ulong                D2SkillLastSignature = 0;
ulong                D2SkillProductionBaseFingerprint = 0;
ulong                D2SkillProductionOMPBFingerprint = 0;
bool                 D2SkillProductionSignatureReady = false;
bool                 D2SkillReady = false;
CBufferFloat         D2SkillFrozenGeneratorWeights;
CBufferFloat         D2SkillFrozenRouterWeights;
CBufferFloat         D2SkillFrozenConfidenceWeights;
CBufferFloat         D2SkillFrozenPrototypes;
CBufferFloat         D2SkillFrozenEMASums;
CBufferFloat         D2SkillFrozenEMACounts;
CBufferFloat         D2SkillFrozenUsage;
CBufferFloat         D2SkillFrozenInactive;
CBufferFloat         D2SkillFrozenInactivityAge;
bool                 D2SkillFrozenBaselineReady = false;
CD2SkillDeviceOps      D2SkillDevice;
ENUM_D2SKILL_STAGE   D2SkillD2Stage = D2Skill_D2_STAGE_BASE;
ENUM_D2SKILL_BANK_MODE D2SkillD2Mode = D2Skill_D2_MODE_BASE;
ENUM_D2SKILL_MODE    D2SkillD2ExecutionMode = D2_DISABLED;
bool                 D2SkillD2UtilityAware = false;
double               D2SkillD2UtilityScale = 1.0;
double               D2SkillD2MinUtility = -1.0;
bool                 D2SkillD2OnlineDirectionUpdate = false;
bool                 D2SkillD2OnlineCriticUpdate = false;
ED2SkillRepresentation D2SkillD2Representation = D2SkillDirectionMagnitude;
bool                 D2SkillD2ResetBanksOnRepresentationMismatch = false;
bool                 D2SkillD2RecreateIncompatibleCheckpoint = false;
int                  D2SkillD2UtilityMode = D2Skill_D2_UTILITY_PAIRED_HINDSIGHT;
ulong                D2SkillD2PairedUtilityUpdates = 0;
//+------------------------------------------------------------------+
//| Reports completed terminal utility updates for online calibration. |
//+------------------------------------------------------------------+
bool D2SkillOnlineCalibrationHasClosedUtilityUpdates(void)
  {
   return(D2SkillD2PairedUtilityUpdates > 0);
  }
//+------------------------------------------------------------------+
bool D2SkillRequiresBankGradientBackward(void)
  {
   if(D2SkillD2Mode == D2Skill_D2_MODE_BASE)
      return(false);
   return(D2SkillD2ExecutionMode == D2_COLLECT ||
          D2SkillD2ExecutionMode == D2_EVALUATE ||
          D2SkillD2ExecutionMode == D2_ONLINE_CALIBRATION);
  }
//+------------------------------------------------------------------+
//| Configures D2Skill execution without preprocessor injection.     |
//+------------------------------------------------------------------+
bool D2SkillConfigureRuntime(const ENUM_D2SKILL_STAGE stage, const ENUM_D2SKILL_BANK_MODE mode,
                             const ENUM_D2SKILL_MODE execution_mode,
                             const bool utility_aware, const double min_utility,
                             const double utility_scale,
                             const bool online_direction_update = false,
                             const bool online_critic_update = false,
                             const ED2SkillRepresentation representation = D2SkillDirectionMagnitude,
                             const bool reset_banks_on_representation_mismatch = false,
                             const bool recreate_incompatible_checkpoint = false)
  {
   if(stage < D2Skill_D2_STAGE_BASE || stage > D2Skill_D2_STAGE_ONLINE ||
      mode < D2Skill_D2_MODE_BASE || mode > D2Skill_D2_MODE_FULL ||
      execution_mode < D2_DISABLED || execution_mode > D2_ONLINE_CALIBRATION ||
      representation < D2SkillFullResidual || representation > D2SkillDirectionMagnitude ||
      !MathIsValidNumber(min_utility) || !MathIsValidNumber(utility_scale) ||
      min_utility < -1.0 || min_utility > 1.0 || utility_scale < 0.0)
      ReturnFalse;
   D2SkillD2Stage = stage;
   D2SkillD2Mode = mode;
   D2SkillD2ExecutionMode = execution_mode;
   D2SkillD2UtilityAware = utility_aware;
   D2SkillD2MinUtility = min_utility;
   D2SkillD2UtilityScale = utility_scale;
   D2SkillD2OnlineDirectionUpdate = online_direction_update;
   D2SkillD2OnlineCriticUpdate = online_critic_update;
   D2SkillD2Representation = representation;
   D2SkillD2ResetBanksOnRepresentationMismatch = reset_banks_on_representation_mismatch;
   D2SkillD2RecreateIncompatibleCheckpoint = recreate_incompatible_checkpoint;
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures optional forecast recovery diagnostics.               |
//+------------------------------------------------------------------+
bool D2SkillConfigureForecastRun(const bool recovery_smoke,
                                 const uint recovery_batches, const uint recovery_age)
  {
   if(recovery_smoke && (recovery_batches == 0 || recovery_age == 0))
      ReturnFalse;
   D2SkillRecoverySmoke = recovery_smoke;
   D2SkillRecoverySmokeBatches = recovery_batches;
   D2SkillRecoverySmokeAge = recovery_age;
   return(true);
  }
//+------------------------------------------------------------------+
//| Device-resident snapshot data for one Actor trajectory branch.  |
//+------------------------------------------------------------------+
struct SD2SkillActorForwardState
  {
   CNeuronScenarioCrossAttention *cross;
   CD2SkillDeviceOps               device;
   CBufferFloat                    cross_output;
   CBufferFloat                    history;
   CBufferFloat                    batch_first;
   CBufferFloat                    batch_second;
   CBufferFloat                    spike_first_output;
   CBufferFloat                    spike_first_previous;
   CBufferFloat                    spike_second_output;
   CBufferFloat                    spike_second_previous;
   int                             batch_first_count;
   int                             batch_second_count;
   bool                            first_has_options;
   bool                            second_has_options;
   bool                            ready;
  };
//+------------------------------------------------------------------+
//| Creates a device buffer owned by local trajectory state.         |
//+------------------------------------------------------------------+
bool D2SkillPrepareForwardStateBuffer(CBufferFloat *source, CBufferFloat &snapshot,
                                      COpenCLMy *open_cl)
  {
   if(!source || !open_cl || source.Total() <= 0 || source.GetIndex() < 0 ||
      source.GetOpenCL() != open_cl)
      ReturnFalse;
   snapshot.BufferFree();
   if(!snapshot.BufferInit((uint)source.Total(), 0.0f) || !snapshot.BufferCreate(open_cl))
      ReturnFalse;
   return(snapshot.Total() == source.Total() && snapshot.GetIndex() >= 0 &&
          snapshot.GetOpenCL() == open_cl);
  }
//+------------------------------------------------------------------+
//| Copies an optional forward buffer without host readback.         |
//+------------------------------------------------------------------+
bool D2SkillCopyOptionalForwardState(CD2SkillDeviceOps &device, CBufferFloat *source,
                                     CBufferFloat &snapshot)
  {
   if(!source)
      return(true);
   return(device.Copy(source, GetPointer(snapshot), (uint)source.Total()));
  }
//+------------------------------------------------------------------+
//| Initializes the local device state for one Actor trajectory.     |
//+------------------------------------------------------------------+
bool D2SkillInitActorForwardState(SD2SkillActorForwardState &state,
                                  CNeuronScenarioCrossAttention *cross,
                                  COpenCLMy *open_cl)
  {
   state.ready = false;
   state.cross = cross;
   if(!state.cross || !open_cl || state.cross.IsCritic() || !state.cross.HasHistory() ||
      state.cross.ForwardStateBatchBlocks() != 2 || !state.device.Bind(open_cl))
      ReturnFalse;
   CNeuronAddToStack *history = state.cross.GetHistoryStack();
   CBufferFloat *first = state.cross.GetForwardStateBatchOptions(0);
   CBufferFloat *second = state.cross.GetForwardStateBatchOptions(1);
   CBufferFloat *spike_first = state.cross.GetForwardStateSpikeOutput(0);
   CBufferFloat *spike_first_previous = state.cross.GetForwardStateSpikePrevOutput(0);
   CBufferFloat *spike_second = state.cross.GetForwardStateSpikeOutput(1);
   CBufferFloat *spike_second_previous = state.cross.GetForwardStateSpikePrevOutput(1);
   if(!history || !D2SkillPrepareForwardStateBuffer(state.cross.getOutput(), state.cross_output, open_cl) ||
      !D2SkillPrepareForwardStateBuffer(history.getOutput(), state.history, open_cl) ||
      !D2SkillPrepareForwardStateBuffer(spike_first, state.spike_first_output, open_cl) ||
      !D2SkillPrepareForwardStateBuffer(spike_first_previous, state.spike_first_previous, open_cl) ||
      !D2SkillPrepareForwardStateBuffer(spike_second, state.spike_second_output, open_cl) ||
      !D2SkillPrepareForwardStateBuffer(spike_second_previous, state.spike_second_previous, open_cl))
      ReturnFalse;
   state.first_has_options = (first != NULL);
   state.second_has_options = (second != NULL);
   if((state.first_has_options &&
       !D2SkillPrepareForwardStateBuffer(first, state.batch_first, open_cl)) ||
      (state.second_has_options &&
       !D2SkillPrepareForwardStateBuffer(second, state.batch_second, open_cl)))
      ReturnFalse;
   state.batch_first_count = state.cross.GetForwardStateBatchCount(0);
   state.batch_second_count = state.cross.GetForwardStateBatchCount(1);
   if(state.batch_first_count < 1 || state.batch_second_count < 1)
      ReturnFalse;
   state.ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes local trajectory state from the D2 Actor topology.   |
//+------------------------------------------------------------------+
bool D2SkillInitActorForwardState(SD2SkillActorForwardState &state, CNet &actor)
  {
   CNeuronBaseOCL *layer = actor.Layer(1);
   if(!layer || (layer.Type() != defNeuronD2Skill &&
                 layer.Type() != defNeuronScenarioCrossAttention))
      ReturnFalse;
   return(D2SkillInitActorForwardState(state, (CNeuronScenarioCrossAttention*)layer,
                                       actor.GetOpenCL()));
  }
//+------------------------------------------------------------------+
//| Captures forward state into a local trajectory branch.           |
//+------------------------------------------------------------------+
bool D2SkillCaptureActorForwardState(SD2SkillActorForwardState &state)
  {
   if(!state.ready || !state.cross)
      ReturnFalse;
   CNeuronAddToStack *history = state.cross.GetHistoryStack();
   CBufferFloat *first = state.cross.GetForwardStateBatchOptions(0);
   CBufferFloat *second = state.cross.GetForwardStateBatchOptions(1);
   CBufferFloat *spike_first = state.cross.GetForwardStateSpikeOutput(0);
   CBufferFloat *spike_first_previous = state.cross.GetForwardStateSpikePrevOutput(0);
   CBufferFloat *spike_second = state.cross.GetForwardStateSpikeOutput(1);
   CBufferFloat *spike_second_previous = state.cross.GetForwardStateSpikePrevOutput(1);
   if(!history || (state.first_has_options != (first != NULL)) ||
      (state.second_has_options != (second != NULL)) || !spike_first ||
      !spike_first_previous || !spike_second || !spike_second_previous)
      ReturnFalse;
   state.batch_first_count = state.cross.GetForwardStateBatchCount(0);
   state.batch_second_count = state.cross.GetForwardStateBatchCount(1);
   if(state.batch_first_count < 1 || state.batch_second_count < 1 ||
      !state.device.Copy(state.cross.getOutput(), GetPointer(state.cross_output),
                         (uint)state.cross_output.Total()) ||
      !state.device.Copy(history.getOutput(), GetPointer(state.history),
                         (uint)state.history.Total()) ||
      !state.device.Copy(spike_first, GetPointer(state.spike_first_output),
                         (uint)state.spike_first_output.Total()) ||
      !state.device.Copy(spike_first_previous, GetPointer(state.spike_first_previous),
                         (uint)state.spike_first_previous.Total()) ||
      !state.device.Copy(spike_second, GetPointer(state.spike_second_output),
                         (uint)state.spike_second_output.Total()) ||
      !state.device.Copy(spike_second_previous, GetPointer(state.spike_second_previous),
                         (uint)state.spike_second_previous.Total()) ||
      !D2SkillCopyOptionalForwardState(state.device, first, state.batch_first) ||
      !D2SkillCopyOptionalForwardState(state.device, second, state.batch_second))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores one local trajectory branch without host readback.      |
//+------------------------------------------------------------------+
bool D2SkillRestoreActorForwardState(SD2SkillActorForwardState &state)
  {
   if(!state.ready || !state.cross)
      ReturnFalse;
   CNeuronAddToStack *history = state.cross.GetHistoryStack();
   CBufferFloat *first = state.cross.GetForwardStateBatchOptions(0);
   CBufferFloat *second = state.cross.GetForwardStateBatchOptions(1);
   CBufferFloat *spike_first = state.cross.GetForwardStateSpikeOutput(0);
   CBufferFloat *spike_first_previous = state.cross.GetForwardStateSpikePrevOutput(0);
   CBufferFloat *spike_second = state.cross.GetForwardStateSpikeOutput(1);
   CBufferFloat *spike_second_previous = state.cross.GetForwardStateSpikePrevOutput(1);
   if(!history || (state.first_has_options != (first != NULL)) ||
      (state.second_has_options != (second != NULL)) || !spike_first ||
      !spike_first_previous || !spike_second || !spike_second_previous ||
      !state.device.Copy(GetPointer(state.cross_output), state.cross.getOutput(),
                         (uint)state.cross_output.Total()) ||
      !state.device.Copy(GetPointer(state.history), history.getOutput(),
                         (uint)state.history.Total()) ||
      !state.device.Copy(GetPointer(state.spike_first_output), spike_first,
                         (uint)state.spike_first_output.Total()) ||
      !state.device.Copy(GetPointer(state.spike_first_previous), spike_first_previous,
                         (uint)state.spike_first_previous.Total()) ||
      !state.device.Copy(GetPointer(state.spike_second_output), spike_second,
                         (uint)state.spike_second_output.Total()) ||
      !state.device.Copy(GetPointer(state.spike_second_previous), spike_second_previous,
                         (uint)state.spike_second_previous.Total()) ||
      (first && !state.device.Copy(GetPointer(state.batch_first), first,
                                   (uint)state.batch_first.Total())) ||
      (second && !state.device.Copy(GetPointer(state.batch_second), second,
                                    (uint)state.batch_second.Total())) ||
      !state.cross.SetForwardStateBatchCount(0, state.batch_first_count) ||
      !state.cross.SetForwardStateBatchCount(1, state.batch_second_count))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Confirms that a local trajectory branch was initialized.         |
//+------------------------------------------------------------------+
bool D2SkillActorForwardStateReady(const SD2SkillActorForwardState &state)
  {
   return(state.ready);
  }
//+------------------------------------------------------------------+
//| Returns whether this exact Actor state is eligible for pairing.  |
//+------------------------------------------------------------------+
bool D2SkillUsePairedHindsight(CD2Skill *skill,
                               SD2SkillActorForwardState &state)
  {
   return((D2SkillD2ExecutionMode == D2_EVALUATE ||
           D2SkillD2ExecutionMode == D2_ONLINE_CALIBRATION) &&
          D2SkillD2Mode != D2Skill_D2_MODE_BASE && skill != NULL &&
          skill.Ready() && D2SkillActorForwardStateReady(state));
  }
//+------------------------------------------------------------------+
//| Restores an Actor pair capture and its exact configured D2 mode. |
//+------------------------------------------------------------------+
bool D2SkillRestorePairedActorState(SD2SkillActorForwardState &state,
                                    CD2Skill *skill,
                                    const bool task_enabled,
                                    const bool step_enabled,
                                    const string stage)
  {
   const bool state_restored = D2SkillRestoreActorForwardState(state);
   const bool mode_restored = (skill != NULL &&
                               skill.Enable(task_enabled, step_enabled));
   if(!state_restored || !mode_restored)
      PrintFormat("D2Skill paired cleanup stage=%s state=%s mode=%s",
                  stage, (state_restored ? "restored" : "failed"),
                  (mode_restored ? "restored" : "failed"));
   return(state_restored && mode_restored);
  }
//+------------------------------------------------------------------+
//| Implements ForecastRecoveryAge.                                  |
//+------------------------------------------------------------------+
uint ForecastRecoveryAge(void)
  {
   if(D2SkillRecoverySmoke)
      return(MathMax(1, int(D2SkillRecoverySmokeAge)));
   const int seconds = PeriodSeconds(TimeFrame);
   if(seconds <= 0)
      return(6240);
//--- 52 five-day trading weeks: one calendar-equivalent year on H1.
   return(uint)MathMax(1.0, MathRound(6240.0 * PeriodSeconds(PERIOD_H1) / seconds));
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for ConfigureForecastRec... |
//+------------------------------------------------------------------+
bool ConfigureForecastRecoveryAge(void)
  {
   if(!D2SkillForecast || !D2SkillForecast.SetRecoveryAge(ForecastRecoveryAge()))
      ReturnFalse;
   PrintFormat("%s %s recovery_age=%u bars", OMPB_LOG_PREFIX, (D2SkillRecoverySmoke ? "smoke" : "forecast"),
               D2SkillForecast.RecoveryAge());
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillInitTrainingBuffers.                             |
//+------------------------------------------------------------------+
bool D2SkillInitTrainingBuffers(void)
  {
   COpenCLMy *open_cl = D2SkillMarket.GetOpenCL();
   if(!open_cl || !D2SkillDevice.Bind(open_cl))
      ReturnFalse;
   if(!D2SkillFutureTranspose.Init(0, 0, open_cl, NForecast, BarDescr, 1, ADAM, BatchSize) ||
      !D2SkillFuture.BufferInit((NForecast * BarDescr), 0) ||
      !D2SkillFuture.BufferCreate(open_cl) ||
      !D2SkillLatentZero.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !D2SkillLatentZero.BufferCreate(open_cl) ||
      !D2SkillLatentNegativeMarket.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !D2SkillLatentNegativeMarket.BufferCreate(open_cl) ||
      !D2SkillLatentTarget.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !D2SkillLatentTarget.BufferCreate(open_cl) ||
      !D2SkillLatentDelta.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !D2SkillLatentDelta.BufferCreate(open_cl))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillAddBase.                                         |
//+------------------------------------------------------------------+
bool D2SkillAddBase(CArrayObj *description, const uint count)
  {
   if(!description)
      ReturnFalse;
   CLayerDescription *descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronBaseOCL;
   descr.count = count;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return(true);
   DeleteObjAndFalse(descr);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillNormalizeDescriptionBatch.                       |
//+------------------------------------------------------------------+
bool D2SkillNormalizeDescriptionBatch(CArrayObj *description)
  {
   if(!description)
      ReturnFalse;
   for(int i = 0; i < description.Total(); i++)
     {
      CLayerDescription *layer = (CLayerDescription*)description.At(i);
      if(!layer)
         ReturnFalse;
      layer.batch = BatchSize;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillCreateDescriptions.                              |
//+------------------------------------------------------------------+
bool D2SkillCreateDescriptions(CArrayObj *&market, CArrayObj *&target)
  {
   CArrayObj *legacy_decoder = NULL;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!CreateStateDescriptions(market, legacy_decoder))
     {
      DeleteObj(legacy_decoder);
      ReturnFalse;
     }
   DeleteObj(legacy_decoder);
//--- Keep exactly CogDriver layers 0..4 (RankTCM is the boundary).
   int layer = market.Total() - 1;
//+------------------------------------------------------------------+
//| Function while.                                                  |
//+------------------------------------------------------------------+
   while(layer >= 0)
     {
      CLayerDescription* descr = market.At(layer);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!descr || descr.type != defNeuronCogDriverRankTCM)
        {
         if(!market.Delete(layer))
            ReturnFalse;
         layer--;
        }
      else
         break;
     }
   if(market.Total() <= 0)
      ReturnFalse;
   if(!D2SkillNormalizeDescriptionBatch(market))
      ReturnFalse;
//--- CreateBuffers already stores state feature-major as
//--- [BarDescr,HistoryBars], which is CogDriverData's required input layout.
   if(!market.Delete(layer) || !market.Delete(layer - 1))
      ReturnFalse;
   CLayerDescription *descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronCogDriverData;
   descr.window = HistoryBars;
   descr.count = BarDescr;
     {
      uint units[] = {StackSize, StackSize, Quantiles};
      if(ArrayCopy(descr.units, units, 0, 0, units.Size()) < int(units.Size()))
         DeleteObjAndFalse(descr);
     }
   descr.probability = 1.0f;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronCogDriverRankTCM;
   descr.window = HistoryBars * (2 * Quantiles + 1);
   descr.count = EmbeddingSize;
   descr.variables = BarDescr;
     {
      uint units[] = {StackSize, NHeads};
      if(ArrayCopy(descr.units, units, 0, 0, units.Size()) < int(units.Size()))
         DeleteObjAndFalse(descr);
     }
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronOMPBOCL;
   descr.window = EmbeddingSize;
   descr.count = BarDescr;
   descr.layers = OMPBSamples;
     {
      uint units[] = {OMPBReferenceSize, OMPBCurrentWindow};
      if(ArrayCopy(descr.units, units, 0, 0, units.Size()) < int(units.Size()))
         DeleteObjAndFalse(descr);
     }
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronScenarioForecast;
   descr.count = NScenarios;
   descr.variables = TopK;
   descr.window_out = NForecast;
   descr.window = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   if(!target)
      target = new CArrayObj();
   else
      target.Clear();
   if(!target)
      ReturnFalse;
   target.FreeMode(true);
//--- Detached Target Encoder: one normalized future trajectory per BarDescr
//--- variable.  ConvOCL zero-fills the unavailable edge of each window, so
//--- window=3 preserves NForecast positions without an auxiliary pad buffer.
   if(!D2SkillAddBase(target, (BarDescr * NForecast)))
      ReturnFalse;
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronPeriodNorm;
   descr.count = 1;
   descr.window = NForecast;
   descr.variables = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = 3;
   descr.step = 1;
   descr.window_out = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = GELU;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = 2 * EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = GELU;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = 2 * EmbeddingSize;
   descr.step = 2 * EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronPeriodNorm;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.variables = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillCreateNetworks.                                  |
//+------------------------------------------------------------------+
bool D2SkillCreateNetworks(void)
  {
   CArrayObj *market = NULL, *target = NULL;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillCreateDescriptions(market, target))
     {
      DeleteObj(market);
      DeleteObj(target);
      ReturnFalse;
     }
   const bool market_created = D2SkillMarket.Create(market);
   PrintFormat("%s Create market=%s batch=%u", OMPB_LOG_PREFIX, (market_created ? "OK" : "FAIL"), uint(BatchSize));
   const bool target_created = (market_created && D2SkillTarget.Create(target));
   PrintFormat("%s Create target=%s", OMPB_LOG_PREFIX, (target_created ? "OK" : "FAIL"));
   const bool created = (market_created && target_created);
   DeleteObj(market);
   DeleteObj(target);
   if(!created)
      ReturnFalse;
//--- CNet::Create initializes its OpenCL program.  Build each network first,
//--- then move its device buffers to the Market context; otherwise a later
//--- Create invalidates buffers belonging to an earlier network.
   if(!D2SkillTarget.SetOpenCLChecked(D2SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillInitTrainingBuffers())
     {
      PrintFormat("%s init: training buffers=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   D2SkillForecast = (CNeuronScenarioForecast*)D2SkillMarket.Layer(-1);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillForecast || D2SkillForecast.Type() != defNeuronScenarioForecast)
     {
      PrintFormat("%s init: forecast layer=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
   D2SkillTarget.TrainMode(false);
   D2SkillMarket.TrainMode(true);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !ConfigureOMPB(OMPB_BYPASS))
      ReturnFalse;
   ompb.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillValidateShapes.                                  |
//+------------------------------------------------------------------+
bool D2SkillValidateShapes(const bool training = true)
  {
   CNeuronBaseOCL *layer = D2SkillMarket.Layer(0);
   CBufferFloat *buffer = (layer ? layer.getOutput() : NULL);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!buffer || buffer.Total() != HistoryBars * BarDescr)
     {
      PrintFormat("%s shape: market input expected=%d actual=%d", OMPB_LOG_PREFIX, HistoryBars * BarDescr,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   layer = D2SkillMarket.Layer(4);
   buffer = (layer ? layer.getOutput() : NULL);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!buffer || buffer.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("%s shape: market RankTCM expected=%d actual=%d", OMPB_LOG_PREFIX, BarDescr * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   layer = GetOMPB();
   buffer = (layer ? layer.getOutput() : NULL);
   if(!buffer || buffer.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("OMPB shape: bridge expected=%d actual=%d", BarDescr * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(training)
     {
      layer = D2SkillTarget.Layer(0);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: target input expected=%d actual=%d", OMPB_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = D2SkillTarget.Layer(1);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: target PeriodNorm expected=%d actual=%d", OMPB_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = D2SkillTarget.Layer(2);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target Conv3 expected=%d actual=%d", OMPB_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = D2SkillTarget.Layer(3);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast * 2 * EmbeddingSize))
        {
         PrintFormat("%s shape: target Conv2 expected=%d actual=%d", OMPB_LOG_PREFIX,
                     BarDescr * NForecast * 2 * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = D2SkillTarget.Layer(4);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target final Conv expected=%d actual=%d", OMPB_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = D2SkillTarget.Layer(-1);
      buffer = (layer ? layer.getOutput() : NULL);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target output PeriodNorm expected=%d actual=%d", OMPB_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      buffer = D2SkillFutureTranspose.getOutput();
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: future transpose expected=%d actual=%d", OMPB_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
     }
   layer = D2SkillMarket.Layer(-1);
   buffer = (layer ? layer.getOutput() : NULL);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!buffer || buffer.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize)
     {
      PrintFormat("%s shape: Scenario Z expected=%d actual=%d", OMPB_LOG_PREFIX,
                  NScenarios * BarDescr * NForecast * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = D2SkillForecast.GetU();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!buffer || buffer.Total() != (NScenarios * BarDescr * NForecast))
     {
      PrintFormat("%s shape: Scenario U expected=%d actual=%d", OMPB_LOG_PREFIX, NScenarios * BarDescr * NForecast,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = D2SkillForecast.GetPi();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!buffer || buffer.Total() != NScenarios)
     {
      PrintFormat("%s shape: Scenario Pi expected=%d actual=%d", OMPB_LOG_PREFIX, NScenarios,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   if(D2SkillForecast.Variables() != BarDescr || D2SkillForecast.Scenarios() != NScenarios ||
      D2SkillForecast.Horizon() != NForecast || D2SkillForecast.Dimension() != EmbeddingSize ||
//+------------------------------------------------------------------+
//| Function ActiveTrajectories.                                     |
//+------------------------------------------------------------------+
      D2SkillForecast.ActiveTrajectories() > TopK)
     {
      PrintFormat("%s shape: Forecast V=%d/%d K=%d/%d H=%d/%d D=%d/%d active=%d/%d", OMPB_LOG_PREFIX,
                  D2SkillForecast.Variables(), BarDescr, D2SkillForecast.Scenarios(), NScenarios,
                  D2SkillForecast.Horizon(), NForecast, D2SkillForecast.Dimension(), EmbeddingSize,
                  D2SkillForecast.ActiveTrajectories(), TopK);
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillHashUInt.                                        |
//+------------------------------------------------------------------+
ulong D2SkillHashUInt(ulong hash, const ulong value)
  {
   hash ^= value;
   return(hash * ulong(1099511628211));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillHashText.                                        |
//+------------------------------------------------------------------+
ulong D2SkillHashText(ulong hash, const string text)
  {
   for(int i = 0; i < StringLen(text); i++)
      hash = D2SkillHashUInt(hash, (ulong)StringGetCharacter(text, i));
   return(hash);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillHashFile.                                        |
//+------------------------------------------------------------------+
ulong D2SkillHashFile(ulong hash, const string file_name)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return(0);
   const ulong length = FileSize(handle);
   if(length == 0 || length > ulong(INT_MAX))
     { FileClose(handle); return 0; }
   uchar bytes[];
   if(ArrayResize(bytes, (int)length) != (int)length ||
// Read file payload as bytes after successful allocation.
      FileReadArray(handle, bytes, 0, (int)length) != (int)length)
     { FileClose(handle); return 0; }
   FileClose(handle);
   for(int i = 0; i < (int)length; i++)
      hash = D2SkillHashUInt(hash, (ulong)bytes[i]);
   return(hash);
  }
//+------------------------------------------------------------------+
//| Exact signature of the unified Market/Scenario inference grap... |
//+------------------------------------------------------------------+
ulong D2SkillForecastSignature(const string market_file = "")
  {
   if(!D2SkillForecast)
      return(0);
   ulong hash = ulong(1469598103934665603);
   const string checkpoint = (market_file == "" ? D2SkillActiveMarketFile : market_file);
   hash = D2SkillHashFile(hash, checkpoint);
   if(hash == 0)
      return(0);
   hash = D2SkillHashUInt(hash, D2Skill_FORMAT_VERSION);
   hash = D2SkillHashUInt(hash, BarDescr);
   hash = D2SkillHashUInt(hash, NScenarios);
   hash = D2SkillHashUInt(hash, TopK);
   hash = D2SkillHashUInt(hash, NForecast);
   hash = D2SkillHashUInt(hash, EmbeddingSize);
   hash = D2SkillHashUInt(hash, D2SkillForecast.ContractSignature());
   hash = D2SkillHashText(hash, "z_layout=K,V,H,D;u_layout=K,V,H;pi_layout=K;codebook_layout=K,V,H,D");
   hash = D2SkillHashText(hash,
                           "market_layout=RankTCM_then_OMPB_then_ScenarioForecast;" +
                           "variable_order=BarDescr_feature_series_0_to_8");
   hash = D2SkillHashText(hash, "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   return(hash);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillForecastTrainingSignature.                       |
//+------------------------------------------------------------------+
ulong D2SkillForecastTrainingSignature(const ulong forecast_signature, const string target_file = "")
  {
   if(forecast_signature == 0)
      return(0);
   const string checkpoint = (target_file == "" ? D2SkillActiveTargetFile : target_file);
   return(D2SkillHashFile(forecast_signature, checkpoint));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillWriteManifest.                                   |
//+------------------------------------------------------------------+
bool D2SkillWriteManifest(const uint completed_epochs)
  {
   if(!D2SkillForecast)
      ReturnFalse;
   const ulong signature = D2SkillForecastSignature(D2Skill_MARKET_FILE);
   const ulong target_hash = D2SkillHashFile(ulong(1469598103934665603), D2Skill_TARGET_FILE);
   const ulong training_signature = D2SkillForecastTrainingSignature(signature, D2Skill_TARGET_FILE);
   if(signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
   int handle = FileOpen(D2Skill_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=OMPB_FORECAST");
   FileWrite(handle, StringFormat("version=%u", D2Skill_FORMAT_VERSION));
   FileWrite(handle, StringFormat("forecast_type=%d", defNeuronScenarioForecast));
   FileWrite(handle, StringFormat("variables=%u", BarDescr));
   FileWrite(handle, StringFormat("scenarios=%u", NScenarios));
   FileWrite(handle, StringFormat("top_k=%u", TopK));
   FileWrite(handle, StringFormat("horizon=%u", NForecast));
   FileWrite(handle, StringFormat("latent=%u", EmbeddingSize));
   FileWrite(handle, "z_layout=K,V,H,D");
   FileWrite(handle, "u_layout=K,V,H");
   FileWrite(handle, "pi_layout=K");
   FileWrite(handle, "codebook_layout=K,V,H,D");
   FileWrite(handle, "variable_order=BarDescr_feature_series_0_to_8");
   FileWrite(handle, StringFormat("contract_signature=%I64u", D2SkillForecast.ContractSignature()));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", signature));
   FileWrite(handle, StringFormat("target_hash=%I64u", target_hash));
   FileWrite(handle, StringFormat("training_signature=%I64u", training_signature));
   FileWrite(handle, StringFormat("completed_epochs=%u", completed_epochs));
   FileWrite(handle, StringFormat("training_batches=%I64u", D2SkillBatches));
   FileWrite(handle, StringFormat("invalid_batches=%I64u", D2SkillInvalidBatches));
   FileWrite(handle, "normalization=OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   FileClose(handle);
   D2SkillLastSignature = signature;
   return(true);
  }
bool D2SkillStage01SaveCheckpoint(const uint completed_epochs);
//+------------------------------------------------------------------+
//| Implements D2SkillSaveCheckpoint.                               |
//+------------------------------------------------------------------+
bool D2SkillSaveCheckpoint(const uint completed_epochs)
  {
   return(D2SkillStage01SaveCheckpoint(completed_epochs));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillInitIndicators.                                  |
//+------------------------------------------------------------------+
bool D2SkillInitIndicators(void)
  {
   return (Symb.Name(_Symbol) && Symb.Refresh() &&
           RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice) &&
           CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice) &&
           ATR.Create(Symb.Name(), TimeFrame, ATRPeriod) &&
           MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillLoadForecastTraining(void);
bool D2SkillCaptureFrozenForecastBaseline(CNeuronScenarioForecast *forecast);
bool D2SkillVerifyFrozenForecastExact(CNeuronScenarioForecast *forecast);
bool D2SkillForwardForecast(const int position, CBufferFloat *state, CBufferFloat *time);
string D2SkillManifestValue(const string file_name, const string key);
bool OMPBStage02ExplicitCheckpointFilesValid(const string market_file, const string target_file,
      const string manifest_file);
bool OMPBStage02LoadExplicitCheckpoint(const string market_file, const string target_file,
                                       const string manifest_file);
bool OMPBStage02ReloadProofDeterministic(const string market_file, const string target_file,
      const string manifest_file, const int position,
      CBufferFloat *state, CBufferFloat *time);
bool OMPBStage02ReloadCheckpointProof(const string market_file, const string target_file,
                                      const string manifest_file, const ulong expected_market,
                                      const ulong expected_target, const ulong expected_posterior);
bool OMPBStage02FinalizeSteadySelector(void);
//+------------------------------------------------------------------+
//| Stage 02 preflight and Reference collection only.  This path is  |
//| intentionally separate from CreateD2SkillForecastStudy(): Stage  |
//| 02 must reject a missing/incompatible Stage 01 checkpoint rather |
//| than creating a new random Market graph.                         |
//+------------------------------------------------------------------+
datetime             OMPBStage02ReferenceStart = 0;
datetime             OMPBStage02ReferenceSplit = 0;
datetime             OMPBStage02ReferenceEnd = 0;
datetime             OMPBStage02CalibrationStart = 0;
datetime             OMPBStage02CalibrationSplit = 0;
datetime             OMPBStage02CalibrationEnd = 0;
int                  OMPBStage02ReferenceFirst = -1;
int                  OMPBStage02ReferenceLast = -1;
int                  OMPBStage02SourceEvalFirst = -1;
int                  OMPBStage02SourceEvalLast = -1;
int                  OMPBStage02TargetCalibrationFirst = -1;
int                  OMPBStage02TargetCalibrationLast = -1;
int                  OMPBStage02TargetEvalFirst = -1;
int                  OMPBStage02TargetEvalLast = -1;
bool                 OMPBStage02Smoke = false;
uint                 OMPBStage02SmokeLimit = 0;
uint                 OMPBStage02SourcePeriod = 0;
ulong                OMPBStage02BaseFingerprint = 0;
ulong                OMPBStage02ForecastFingerprint = 0;
ulong                OMPBStage02TargetFingerprint = 0;
ulong                OMPBStage02PosteriorFingerprint = 0;
//--- The Stage 01 state remains the only production checkpoint until an
//--- accepted Stage 02 transaction has completed its reload smoke.
bool                 OMPBStage02CheckpointPublished = false;
//+------------------------------------------------------------------+
bool OMPBStage02Finite(const double value)
  {
   return(MathIsValidNumber(value));
  }
//+------------------------------------------------------------------+
bool OMPBStage02Configure(const datetime reference_start, const datetime reference_end,
                          const datetime calibration_start, const datetime calibration_end,
                          const uint anchor_period, const float tau_value, const float lambda_dis,
                          const float lambda_kl, const float lambda_alpha, const float alpha_prior,
                          const float max_kl, const bool smoke, const uint smoke_limit)
  {
   if(reference_start >= reference_end || reference_end > calibration_start ||
      calibration_start >= calibration_end || tau_value < 0.0f || tau_value > 1.0f ||
      lambda_dis < 0.0f || lambda_kl < 0.0f || lambda_alpha < 0.0f ||
      alpha_prior <= 0.0f || alpha_prior >= 1.0f || max_kl < 0.0f ||
      !OMPBStage02Finite(tau_value) || !OMPBStage02Finite(lambda_dis) ||
      !OMPBStage02Finite(lambda_kl) || !OMPBStage02Finite(lambda_alpha) ||
      !OMPBStage02Finite(alpha_prior) || !OMPBStage02Finite(max_kl))
     {
      Print("OMPB_STAGE02_PREFLIGHT_FAIL reason=input_contract");
      ReturnFalse;
     }
   const long reference_span = long(reference_end) - long(reference_start);
   const long calibration_span = long(calibration_end) - long(calibration_start);
   const datetime reference_split = reference_start + int(reference_span * 4 / 5);
   const datetime calibration_split = calibration_start + int(calibration_span * 4 / 5);
   if(reference_span <= 0 || calibration_span <= 0 || reference_split <= reference_start ||
      reference_split >= reference_end || calibration_split <= calibration_start ||
      calibration_split >= calibration_end)
     {
      Print("OMPB_STAGE02_PREFLIGHT_FAIL reason=empty_80_20_partition");
      ReturnFalse;
     }
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !ompb.SetTau(tau_value) || !ompb.SetDisagreementMult(lambda_dis) ||
      !ompb.SetKLDMult(lambda_kl) || !ompb.SetAlphaMult(lambda_alpha) ||
      !ompb.SetAlphaPrior(alpha_prior) || !ompb.SetMaxKL(max_kl) ||
      !ConfigureOMPB(OMPB_BYPASS))
     {
      Print("OMPB_STAGE02_PREFLIGHT_FAIL reason=ompb_configuration");
      ReturnFalse;
     }
   ompb.TrainMode(false);
   OMPBStage02ReferenceStart = reference_start;
   OMPBStage02ReferenceSplit = reference_split;
   OMPBStage02ReferenceEnd = reference_end;
   OMPBStage02CalibrationStart = calibration_start;
   OMPBStage02CalibrationSplit = calibration_split;
   OMPBStage02CalibrationEnd = calibration_end;
   OMPBStage02Smoke = smoke;
   OMPBStage02SmokeLimit = (smoke ? MathMax(1, int(smoke_limit)) : 0);
   OMPBStage02SourcePeriod = anchor_period;
   PrintFormat("OMPB_STAGE02_PREFLIGHT_CONFIG reference=%s..%s split=%s calibration=%s..%s",
               TimeToString(reference_start, TIME_DATE), TimeToString(reference_end, TIME_DATE),
               TimeToString(reference_split, TIME_DATE), TimeToString(calibration_start, TIME_DATE),
               TimeToString(calibration_end, TIME_DATE));
   PrintFormat("OMPB_STAGE02_PREFLIGHT_CONFIG split=%s source_period=%u smoke=%s",
               TimeToString(calibration_split, TIME_DATE), anchor_period, (smoke ? "true" : "false"));
   return(true);
  }
//+------------------------------------------------------------------+
bool OMPBStage02PrepareData(void)
  {
   const int reference_start = iBarShift(Symb.Name(), TimeFrame, OMPBStage02ReferenceStart);
   const int reference_split = iBarShift(Symb.Name(), TimeFrame, OMPBStage02ReferenceSplit);
   const int reference_end = iBarShift(Symb.Name(), TimeFrame, OMPBStage02ReferenceEnd);
   const int calibration_start = iBarShift(Symb.Name(), TimeFrame, OMPBStage02CalibrationStart);
   const int calibration_split = iBarShift(Symb.Name(), TimeFrame, OMPBStage02CalibrationSplit);
   const int calibration_end = iBarShift(Symb.Name(), TimeFrame, OMPBStage02CalibrationEnd);
   if(reference_start <= 0 || reference_split <= 0 || reference_end <= 0 ||
      calibration_start <= 0 || calibration_split <= 0 || calibration_end <= 0 ||
      reference_start <= reference_split || reference_split <= reference_end ||
      reference_end < calibration_start || calibration_start <= calibration_split ||
      calibration_split <= calibration_end)
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=bar_order shifts=(%d,%d,%d,%d,%d,%d)",
                  reference_start, reference_split, reference_end, calibration_start,
                  calibration_split, calibration_end);
      ReturnFalse;
     }
   const int bars = CopyRates(Symb.Name(), TimeFrame, 0, reference_start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=rates bars=%d error=%d", bars, GetLastError());
      ReturnFalse;
     }
   int wait = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars && CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars && MACD.BarsCalculated() >= bars);
      Sleep(100);
      wait++;
     }
   while(!calculated && wait < 100);
   if(!calculated)
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=indicators bars=%d error=%d", bars, GetLastError());
      ReturnFalse;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=rates_series error=%d", GetLastError());
      ReturnFalse;
     }
   OMPBStage02ReferenceFirst = reference_start - HistoryBars - NForecast - 1;
   OMPBStage02ReferenceLast = reference_split;
   OMPBStage02SourceEvalFirst = reference_split - 1;
   OMPBStage02SourceEvalLast = reference_end;
   OMPBStage02TargetCalibrationFirst = calibration_start - HistoryBars - NForecast - 1;
   OMPBStage02TargetCalibrationLast = calibration_split;
   OMPBStage02TargetEvalFirst = calibration_split - 1;
   OMPBStage02TargetEvalLast = calibration_end;
   const int reference_rows = OMPBStage02ReferenceFirst - OMPBStage02ReferenceLast + 1;
   const int source_eval_rows = OMPBStage02SourceEvalFirst - OMPBStage02SourceEvalLast + 1;
   const int target_calibration_rows = OMPBStage02TargetCalibrationFirst - OMPBStage02TargetCalibrationLast + 1;
   const int target_eval_rows = OMPBStage02TargetEvalFirst - OMPBStage02TargetEvalLast + 1;
   if(OMPBStage02ReferenceFirst < OMPBStage02ReferenceLast ||
      OMPBStage02SourceEvalFirst < OMPBStage02SourceEvalLast ||
      OMPBStage02TargetCalibrationFirst < OMPBStage02TargetCalibrationLast ||
      OMPBStage02TargetEvalFirst < OMPBStage02TargetEvalLast ||
      reference_rows < int(OMPBReferenceSize) || source_eval_rows <= 0 ||
      target_calibration_rows < int(OMPBCurrentWindow) || target_eval_rows <= 0)
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=bars ref=%d source=%d calibration=%d target=%d",
                  reference_rows, source_eval_rows, target_calibration_rows, target_eval_rows);
      PrintFormat("OMPB_STAGE02_PREFLIGHT_REQUIRED reference=%u target_calibration=%u",
                  OMPBReferenceSize, OMPBCurrentWindow);
      ReturnFalse;
     }
   PrintFormat("OMPB_STAGE02_PREFLIGHT_PASS bars=%d reference=%d source_eval=%d target_calibration=%d target_eval=%d",
               bars, reference_rows, source_eval_rows, target_calibration_rows, target_eval_rows);
   return(true);
  }
//+------------------------------------------------------------------+
bool OMPBStage02CollectReference(void)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !ResetOMPBReference() || !ConfigureOMPB(OMPB_REFERENCE))
     {
      Print("OMPB_STAGE02_REFERENCE_FAIL reason=setup");
      ReturnFalse;
     }
   ompb.TrainMode(false);
   uint collected = 0;
   for(int position = OMPBStage02ReferenceFirst;
       position >= OMPBStage02ReferenceLast && !IsStopped(); position--)
     {
      if(!CreateBuffers(position + NForecast, GetPointer(D2SkillState),
                        GetPointer(D2SkillTime), NULL) ||
         !D2SkillMarket.feedForward(GetPointer(D2SkillState), 1, false, (CBufferFloat *)NULL))
        {
         PrintFormat("OMPB_STAGE02_REFERENCE_FAIL reason=forward position=%d line=%d", position, __LINE__);
         ReturnFalse;
        }
      collected++;
      if(OMPBStage02Smoke && collected >= OMPBStage02SmokeLimit)
         break;
      if(ompb.ReferenceCount() >= OMPBReferenceSize)
         break;
     }
   if(collected == 0 || (!OMPBStage02Smoke && ompb.ReferenceCount() != OMPBReferenceSize))
     {
      PrintFormat("OMPB_STAGE02_REFERENCE_FAIL reason=count collected=%u reference=%u required=%u",
                  collected, ompb.ReferenceCount(), OMPBReferenceSize);
      ReturnFalse;
     }
   PrintFormat("OMPB_STAGE02_REFERENCE_PASS collected=%u reference=%u capacity=%u smoke=%s",
               collected, ompb.ReferenceCount(), OMPBReferenceSize, (OMPBStage02Smoke ? "true" : "false"));
   return(ConfigureOMPB(OMPB_BYPASS));
  }
//+------------------------------------------------------------------+
bool OMPBStage02Baseline(const int first, const int last, const string scope,
                         double &forecast_loss, uint &valid_batches, uint &invalid_batches)
  {
   forecast_loss = 0.0;
   valid_batches = 0;
   invalid_batches = 0;
   if(first < last || !D2SkillForecast || !ConfigureOMPB(OMPB_BYPASS) ||
      !D2SkillMarket.Clear() || !D2SkillTarget.Clear() ||
      !D2SkillForecast.ResetEpochDiagnostics())
     {
      PrintFormat("OMPB_STAGE02_BASELINE_FAIL scope=%s reason=setup", scope);
      ReturnFalse;
     }
   for(int position = first; position >= last && !IsStopped(); position--)
     {
      if(!D2SkillTrainBatch(position))
        {
         invalid_batches++;
         continue;
        }
      valid_batches++;
      if(OMPBStage02Smoke && valid_batches >= OMPBStage02SmokeLimit)
         break;
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(valid_batches == 0 || !D2SkillForecast.BuildEpochCodebookDiagnostics() ||
      !D2SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence, latent,
            observation, valid, invalid, entropy, distance,
            inactive, recovered) ||
      valid < double(valid_batches) || !OMPBStage02Finite(lmix) ||
      !OMPBStage02Finite(valid) || !OMPBStage02Finite(invalid))
     {
      PrintFormat("OMPB_STAGE02_BASELINE_FAIL scope=%s reason=metrics valid=%u invalid=%u",
                  scope, valid_batches, invalid_batches);
      ReturnFalse;
     }
   forecast_loss = lmix / valid;
   if(!OMPBStage02Finite(forecast_loss))
     {
      PrintFormat("OMPB_STAGE02_BASELINE_FAIL scope=%s reason=nonfinite_loss", scope);
      ReturnFalse;
     }
   PrintFormat("OMPB_STAGE02_BASELINE_PASS scope=%s loss=%.9f valid=%u invalid=%u",
               scope, forecast_loss, valid_batches, invalid_batches);
   return(true);
  }
//+------------------------------------------------------------------+
//| Fingerprints only the specified market layers.  The seed and the |
//| layer index make base, OMPB and ScenarioForecast checks separate. |
//+------------------------------------------------------------------+
bool OMPBStage02MarketFingerprint(const int first, const int last, ulong &fingerprint)
  {
   if(first < 0 || first > last)
      ReturnFalse;
   fingerprint = ulong(1469598103934665603);
   for(int index = first; index <= last; index++)
     {
      CNeuronBaseOCL *layer = D2SkillMarket.Layer(index);
      if(!layer)
         ReturnFalse;
      fingerprint = (fingerprint ^ ulong(index + 1)) * ulong(1099511628211);
      if(!layer.AppendParameterFingerprint(fingerprint))
         ReturnFalse;
     }
   return(fingerprint != 0);
  }
//+------------------------------------------------------------------+
//| Captures every state that must remain frozen while OMPB learns.   |
//+------------------------------------------------------------------+
bool OMPBStage02CaptureSignatures(void)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !OMPBStage02MarketFingerprint(0, 4, OMPBStage02BaseFingerprint) ||
      !OMPBStage02MarketFingerprint(6, 6, OMPBStage02ForecastFingerprint) ||
      !D2SkillTarget.ParameterFingerprint(OMPBStage02TargetFingerprint))
      ReturnFalse;
   OMPBStage02PosteriorFingerprint = ulong(1469598103934665603);
   if(!ompb.AppendParameterFingerprint(OMPBStage02PosteriorFingerprint) ||
      OMPBStage02PosteriorFingerprint == 0)
      ReturnFalse;
   PrintFormat("OMPB_STAGE02_SIGNATURES_BEFORE base=%I64u forecast=%I64u target=%I64u posterior=%I64u",
               OMPBStage02BaseFingerprint, OMPBStage02ForecastFingerprint,
               OMPBStage02TargetFingerprint, OMPBStage02PosteriorFingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Ensures Stage 02 changed only the posterior/alpha parameter set. |
//+------------------------------------------------------------------+
bool OMPBStage02VerifySignatures(void)
  {
   ulong base = 0, forecast = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !OMPBStage02MarketFingerprint(0, 4, base) ||
      !OMPBStage02MarketFingerprint(6, 6, forecast) ||
      !D2SkillTarget.ParameterFingerprint(target) ||
      !ompb.AppendParameterFingerprint(posterior))
      ReturnFalse;
   const bool outer_unchanged = (base == OMPBStage02BaseFingerprint &&
                                 forecast == OMPBStage02ForecastFingerprint &&
                                 target == OMPBStage02TargetFingerprint);
   const bool posterior_changed = (posterior != OMPBStage02PosteriorFingerprint);
   PrintFormat("OMPB_STAGE02_SIGNATURES_AFTER base=%I64u forecast=%I64u target=%I64u posterior=%I64u",
               base, forecast, target, posterior);
   if(!outer_unchanged || !posterior_changed)
     {
      PrintFormat("OMPB_STAGE02_SIGNATURES_FAIL outer_unchanged=%s posterior_changed=%s",
                  (outer_unchanged ? "true" : "false"),
                  (posterior_changed ? "true" : "false"));
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Computes an evaluation loss through posterior-mean inference.    |
//+------------------------------------------------------------------+
bool OMPBStage02InferenceEvaluation(const int first, const int last, const string scope,
                                    double &forecast_loss, uint &valid_batches,
                                    uint &invalid_batches)
  {
   forecast_loss = 0.0;
   valid_batches = 0;
   invalid_batches = 0;
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(first < last || !ompb || !D2SkillForecast || !ConfigureOMPB(OMPB_INFERENCE) ||
      !D2SkillMarket.Clear() || !D2SkillTarget.Clear() ||
      !D2SkillForecast.ResetEpochDiagnostics())
     {
      PrintFormat("OMPB_STAGE02_EVALUATION_FAIL scope=%s reason=setup", scope);
      ReturnFalse;
     }
   D2SkillMarket.TrainMode(true);
   D2SkillTarget.TrainMode(false);
   ompb.TrainMode(false);
   for(int position = first; position >= last && !IsStopped(); position--)
     {
      if(!D2SkillTrainBatch(position))
        {
         invalid_batches++;
         continue;
        }
      valid_batches++;
      if(OMPBStage02Smoke && valid_batches >= OMPBStage02SmokeLimit)
         break;
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(valid_batches == 0 || !D2SkillForecast.BuildEpochCodebookDiagnostics() ||
      !D2SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence, latent,
            observation, valid, invalid, entropy, distance,
            inactive, recovered) ||
      valid < double(valid_batches) || !OMPBStage02Finite(lmix) ||
      !OMPBStage02Finite(valid) || !OMPBStage02Finite(invalid))
     {
      PrintFormat("OMPB_STAGE02_EVALUATION_FAIL scope=%s reason=metrics valid=%u invalid=%u",
                  scope, valid_batches, invalid_batches);
      ReturnFalse;
     }
   forecast_loss = lmix / valid;
   if(!OMPBStage02Finite(forecast_loss) || !D2SkillVerifyFrozenForecastExact())
     {
      PrintFormat("OMPB_STAGE02_EVALUATION_FAIL scope=%s reason=finite_or_frozen", scope);
      ReturnFalse;
     }
   PrintFormat("OMPB_STAGE02_EVALUATION_PASS scope=%s loss=%.9f valid=%u invalid=%u",
               scope, forecast_loss, valid_batches, invalid_batches);
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks the frozen acceptance gate without replacing any file.    |
//+------------------------------------------------------------------+
bool OMPBStage02Accept(const double source_before, const double target_before,
                       const double source_after, const double target_after,
                       const uint source_valid, const uint target_valid)
  {
   const double denominator_source = MathMax(MathAbs(source_before), 1.0e-12);
   const double denominator_target = MathMax(MathAbs(target_before), 1.0e-12);
   const double target_improvement = (target_before - target_after) / denominator_target;
   const double source_degradation = (source_after - source_before) / denominator_source;
   const bool finite = (OMPBStage02Finite(source_before) && OMPBStage02Finite(target_before) &&
                        OMPBStage02Finite(source_after) && OMPBStage02Finite(target_after));
   const bool accepted = (finite && source_valid > 0 && target_valid > 0 &&
                          target_improvement >= 0.01 && source_degradation <= 0.01);
   PrintFormat("OMPB_STAGE02_ACCEPTANCE target_improvement=%.8f source_degradation=%.8f " +
               "source_valid=%u target_valid=%u accepted=%s", target_improvement,
               source_degradation, source_valid, target_valid,
               (accepted ? "true" : "false"));
   return(accepted);
  }
//+------------------------------------------------------------------+
//| Candidate manifest signatures are intentionally filename-aware.  |
//| The temporary graph must never validate against a production file.|
//+------------------------------------------------------------------+
ulong OMPBStage02ForecastSignature(const string market_file,
                                   CNeuronScenarioForecast *forecast)
  {
   if(!forecast)
      return(0);
   ulong hash = ulong(1469598103934665603);
   hash = D2SkillHashFile(hash, market_file);
   if(hash == 0)
      return(0);
   hash = D2SkillHashUInt(hash, D2Skill_FORMAT_VERSION);
   hash = D2SkillHashUInt(hash, BarDescr);
   hash = D2SkillHashUInt(hash, NScenarios);
   hash = D2SkillHashUInt(hash, TopK);
   hash = D2SkillHashUInt(hash, NForecast);
   hash = D2SkillHashUInt(hash, EmbeddingSize);
   hash = D2SkillHashUInt(hash, forecast.ContractSignature());
   hash = D2SkillHashText(hash, "z_layout=K,V,H,D;u_layout=K,V,H;pi_layout=K;codebook_layout=K,V,H,D");
   hash = D2SkillHashText(hash,
                          "market_layout=RankTCM_then_OMPB_then_ScenarioForecast;" +
                          "variable_order=BarDescr_feature_series_0_to_8");
   return(D2SkillHashText(hash, "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw"));
  }
//+------------------------------------------------------------------+
bool OMPBStage02WriteCandidateManifest(const string market_file, const string target_file,
                                       const string manifest_file, const uint completed_epochs)
  {
   const ulong forecast_signature = OMPBStage02ForecastSignature(market_file, D2SkillForecast);
   const ulong target_hash = D2SkillHashFile(ulong(1469598103934665603), target_file);
   const ulong training_signature = D2SkillHashFile(forecast_signature, target_file);
   if(!D2SkillForecast || forecast_signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
   int handle = FileOpen(manifest_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   const bool written = (FileWrite(handle, "format=OMPB_FORECAST") > 0 &&
                         FileWrite(handle, StringFormat("version=%u", D2Skill_FORMAT_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("forecast_type=%d", defNeuronScenarioForecast)) > 0 &&
                         FileWrite(handle, StringFormat("variables=%u", BarDescr)) > 0 &&
                         FileWrite(handle, StringFormat("scenarios=%u", NScenarios)) > 0 &&
                         FileWrite(handle, StringFormat("top_k=%u", TopK)) > 0 &&
                         FileWrite(handle, StringFormat("horizon=%u", NForecast)) > 0 &&
                         FileWrite(handle, StringFormat("latent=%u", EmbeddingSize)) > 0 &&
                         FileWrite(handle, "z_layout=K,V,H,D") > 0 &&
                         FileWrite(handle, "u_layout=K,V,H") > 0 &&
                         FileWrite(handle, "pi_layout=K") > 0 &&
                         FileWrite(handle, "codebook_layout=K,V,H,D") > 0 &&
                         FileWrite(handle, "variable_order=BarDescr_feature_series_0_to_8") > 0 &&
                         FileWrite(handle, StringFormat("contract_signature=%I64u",
                                   D2SkillForecast.ContractSignature())) > 0 &&
                         FileWrite(handle, StringFormat("forecast_signature=%I64u", forecast_signature)) > 0 &&
                         FileWrite(handle, StringFormat("target_hash=%I64u", target_hash)) > 0 &&
                         FileWrite(handle, StringFormat("training_signature=%I64u", training_signature)) > 0 &&
                         FileWrite(handle, StringFormat("completed_epochs=%u", completed_epochs)) > 0 &&
                         FileWrite(handle, StringFormat("training_batches=%I64u", D2SkillBatches)) > 0 &&
                         FileWrite(handle, StringFormat("invalid_batches=%I64u", D2SkillInvalidBatches)) > 0 &&
                         FileWrite(handle, "normalization=OHLC_deltas_from_open;" +
                                   "tick_volume_div_1000;RSI_CCI_ATR_MACD_raw") > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   return(written);
  }
//+------------------------------------------------------------------+
bool OMPBStage02ValidateCandidateManifest(const string market_file, const string target_file,
      const string manifest_file,
      CNeuronScenarioForecast *forecast)
  {
   const ulong signature = OMPBStage02ForecastSignature(market_file, forecast);
   const ulong target_hash = D2SkillHashFile(ulong(1469598103934665603), target_file);
   const ulong training_signature = D2SkillHashFile(signature, target_file);
   if(!forecast || signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
#define OMPB_STAGE02_MANIFEST_EQ(KEY,VALUE) if(D2SkillManifestValue(manifest_file, KEY) != (VALUE)) ReturnFalse
   OMPB_STAGE02_MANIFEST_EQ("format", "OMPB_FORECAST");
   OMPB_STAGE02_MANIFEST_EQ("version", IntegerToString(D2Skill_FORMAT_VERSION));
   OMPB_STAGE02_MANIFEST_EQ("forecast_type", IntegerToString(defNeuronScenarioForecast));
   OMPB_STAGE02_MANIFEST_EQ("variables", IntegerToString(BarDescr));
   OMPB_STAGE02_MANIFEST_EQ("scenarios", IntegerToString(NScenarios));
   OMPB_STAGE02_MANIFEST_EQ("top_k", IntegerToString(TopK));
   OMPB_STAGE02_MANIFEST_EQ("horizon", IntegerToString(NForecast));
   OMPB_STAGE02_MANIFEST_EQ("latent", IntegerToString(EmbeddingSize));
   OMPB_STAGE02_MANIFEST_EQ("z_layout", "K,V,H,D");
   OMPB_STAGE02_MANIFEST_EQ("u_layout", "K,V,H");
   OMPB_STAGE02_MANIFEST_EQ("pi_layout", "K");
   OMPB_STAGE02_MANIFEST_EQ("codebook_layout", "K,V,H,D");
   OMPB_STAGE02_MANIFEST_EQ("variable_order", "BarDescr_feature_series_0_to_8");
   OMPB_STAGE02_MANIFEST_EQ("normalization", "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   OMPB_STAGE02_MANIFEST_EQ("contract_signature", StringFormat("%I64u", forecast.ContractSignature()));
   OMPB_STAGE02_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", signature));
   OMPB_STAGE02_MANIFEST_EQ("target_hash", StringFormat("%I64u", target_hash));
   OMPB_STAGE02_MANIFEST_EQ("training_signature", StringFormat("%I64u", training_signature));
#undef OMPB_STAGE02_MANIFEST_EQ
   return(D2SkillManifestValue(manifest_file, "completed_epochs") != "" &&
          D2SkillManifestValue(manifest_file, "training_batches") != "" &&
          D2SkillManifestValue(manifest_file, "invalid_batches") != "");
  }
//+------------------------------------------------------------------+
//| Reload-probes one immutable candidate before selector activation. |
//+------------------------------------------------------------------+
bool OMPBStage02ValidateCandidate(const string market_file, const string target_file,
                                  const string manifest_file)
  {
   CNet market;
   CNet target;
   float error = 0.0f, undefine = 0.0f, forecast = 0.0f;
   datetime studied = 0;
   if(!market.Load(market_file, error, undefine, forecast, studied, true) ||
      !target.Load(target_file, error, undefine, forecast, studied, true))
      ReturnFalse;
   if(!target.SetOpenCLChecked(market.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   CNeuronBaseOCL *rank_tcm = market.Layer(4);
   CNeuronBaseOCL *ompb_layer = market.Layer(5);
   CNeuronBaseOCL *forecast_layer = market.Layer(6);
   CNeuronScenarioForecast *staged_forecast =
      (forecast_layer && forecast_layer.Type() == defNeuronScenarioForecast ?
       (CNeuronScenarioForecast *)forecast_layer : NULL);
   CNeuronOMPBOCL *staged_ompb =
      (ompb_layer && ompb_layer.Type() == defNeuronOMPBOCL ? (CNeuronOMPBOCL *)ompb_layer : NULL);
   ulong staged_market = 0, staged_target = 0, staged_posterior = ulong(1469598103934665603);
   ulong live_market = 0, live_target = 0, live_posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *live_ompb = GetOMPB();
   if(!rank_tcm || rank_tcm.Type() != defNeuronCogDriverRankTCM || !staged_ompb ||
      !staged_forecast ||
      !market.ParameterFingerprint(staged_market) || !target.ParameterFingerprint(staged_target) ||
      !staged_ompb.AppendParameterFingerprint(staged_posterior) || !D2SkillMarket.ParameterFingerprint(live_market) ||
      !D2SkillTarget.ParameterFingerprint(live_target) || !live_ompb ||
      !live_ompb.AppendParameterFingerprint(live_posterior) ||
      !OMPBStage02ValidateCandidateManifest(market_file, target_file, manifest_file,
            staged_forecast))
      ReturnFalse;
   return(staged_market == live_market && staged_target == live_target &&
          staged_posterior == live_posterior);
  }
//+------------------------------------------------------------------+
//--- A malformed active selector is fail-closed.  Its absence deliberately
//--- retains the original Stage 01 fixed-name loading contract.
bool                 OMPBStage02RecoveryFailed = false;
bool                 OMPBStage02SelectorActivated = false;
bool                 OMPBStage02ReloadProofPassed = false;
bool                 OMPBStage02PreviousProofReady = false;
string               OMPBStage02StagedGeneration = "";
ulong                OMPBStage02PreviousMarketFingerprint = 0;
ulong                OMPBStage02PreviousTargetFingerprint = 0;
ulong                OMPBStage02PreviousPosteriorFingerprint = 0;
//+------------------------------------------------------------------+
//| Validates a complete checkpoint set without touching live state. |
//+------------------------------------------------------------------+
bool OMPBStage02ValidateCheckpointSet(const string market_file, const string target_file,
                                      const string manifest_file)
  {
   if(!FileIsExist(market_file, FILE_COMMON) || !FileIsExist(target_file, FILE_COMMON) ||
      !FileIsExist(manifest_file, FILE_COMMON))
      ReturnFalse;
   CNet market;
   CNet target;
   float error = 0.0f, undefine = 0.0f, forecast = 0.0f;
   datetime studied = 0;
   if(!market.Load(market_file, error, undefine, forecast, studied, true) ||
      !target.Load(target_file, error, undefine, forecast, studied, true))
      ReturnFalse;
   if(!target.SetOpenCLChecked(market.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   CNeuronBaseOCL *rank_tcm = market.Layer(4);
   CNeuronBaseOCL *ompb_layer = market.Layer(5);
   CNeuronBaseOCL *forecast_layer = market.Layer(6);
   CNeuronScenarioForecast *checkpoint_forecast =
      (forecast_layer && forecast_layer.Type() == defNeuronScenarioForecast ?
       (CNeuronScenarioForecast *)forecast_layer : NULL);
   return(rank_tcm != NULL && rank_tcm.Type() == defNeuronCogDriverRankTCM &&
          ompb_layer != NULL && ompb_layer.Type() == defNeuronOMPBOCL &&
          checkpoint_forecast != NULL &&
          OMPBStage02ValidateCandidateManifest(market_file, target_file, manifest_file,
                checkpoint_forecast));
  }
//+------------------------------------------------------------------+
//| Identifies the clean initial state before the first Stage 01 save.|
//+------------------------------------------------------------------+
bool OMPBStage02CanonicalTupleMissing(const string market_file, const string target_file,
                                      const string manifest_file)
  {
//--- A Stage 01 restore miss is safe only when no tuple member exists.
   return(!FileIsExist(market_file, FILE_COMMON) && !FileIsExist(target_file, FILE_COMMON) &&
          !FileIsExist(manifest_file, FILE_COMMON));
  }
//+------------------------------------------------------------------+
//| Captures the immutable checkpoint fingerprints for rollback proof.|
//+------------------------------------------------------------------+
bool OMPBStage02CheckpointFingerprints(const string market_file, const string target_file,
                                       ulong &market_fingerprint, ulong &target_fingerprint,
                                       ulong &posterior_fingerprint)
  {
   market_fingerprint = 0;
   target_fingerprint = 0;
   posterior_fingerprint = ulong(1469598103934665603);
   CNet checkpoint_market;
   CNet checkpoint_target;
   float error = 0.0f, undefine = 0.0f, forecast = 0.0f;
   datetime studied = 0;
   if(!checkpoint_market.Load(market_file, error, undefine, forecast, studied, true) ||
      !checkpoint_target.Load(target_file, error, undefine, forecast, studied, true))
      ReturnFalse;
   if(!checkpoint_target.SetOpenCLChecked(checkpoint_market.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   CNeuronBaseOCL *ompb_layer = checkpoint_market.Layer(5);
   CNeuronOMPBOCL *checkpoint_ompb =
      (ompb_layer && ompb_layer.Type() == defNeuronOMPBOCL ? (CNeuronOMPBOCL *)ompb_layer : NULL);
   return(checkpoint_ompb != NULL && checkpoint_market.ParameterFingerprint(market_fingerprint) &&
          checkpoint_target.ParameterFingerprint(target_fingerprint) &&
          checkpoint_ompb.AppendParameterFingerprint(posterior_fingerprint));
  }
//+------------------------------------------------------------------+
//| Restores the Stage 01 fixed-name checkpoint as the default view. |
//+------------------------------------------------------------------+
void OMPBStage02ResetActiveCheckpointFiles(void)
  {
   D2SkillActiveMarketFile = D2Skill_MARKET_FILE;
   D2SkillActiveTargetFile = D2Skill_TARGET_FILE;
   D2SkillActiveManifestFile = D2Skill_MANIFEST_FILE;
  }
//+------------------------------------------------------------------+
//| Accepts only generated identifiers, never arbitrary filenames.  |
//+------------------------------------------------------------------+
bool OMPBStage02GenerationIdValid(const string generation)
  {
   const int length = StringLen(generation);
   int separator = -1;
   if(length < 3)
      return(false);
   for(int index = 0; index < length; index++)
     {
      const ushort character = (ushort)StringGetCharacter(generation, index);
      if(character == (ushort)'_')
        {
         if(separator >= 0)
            return(false);
         separator = index;
         continue;
        }
      if(character < (ushort)'0' || character > (ushort)'9')
         return(false);
     }
   return(separator > 0 && separator < length - 1);
  }
//+------------------------------------------------------------------+
//| Maps a selector generation to its immutable checkpoint trio.    |
//+------------------------------------------------------------------+
bool OMPBStage02GenerationFiles(const string generation, string &market_file,
                                string &target_file, string &manifest_file)
  {
   if(generation == "stage01")
     {
      market_file = D2Skill_MARKET_FILE;
      target_file = D2Skill_TARGET_FILE;
      manifest_file = D2Skill_MANIFEST_FILE;
      return(true);
     }
   if(!OMPBStage02GenerationIdValid(generation))
      return(false);
   market_file = "OMPBMarket" + OMPB_STAGE02_GENERATION_PREFIX + generation + ".nnw";
   target_file = "OMPBTarget" + OMPB_STAGE02_GENERATION_PREFIX + generation + ".nnw";
   manifest_file = "OMPBForecast" + OMPB_STAGE02_GENERATION_PREFIX + generation + ".manifest";
   return(true);
  }
//+------------------------------------------------------------------+
//| Accepts only the fixed Stage 01 trio or one exact generation.   |
//+------------------------------------------------------------------+
bool OMPBStage02ExplicitCheckpointFilesValid(const string market_file, const string target_file,
      const string manifest_file)
  {
   string expected_market = "", expected_target = "", expected_manifest = "";
   if(!OMPBStage02GenerationFiles("stage01", expected_market, expected_target, expected_manifest))
      return(false);
   if(market_file == expected_market && target_file == expected_target &&
      manifest_file == expected_manifest)
      return(true);
   const string market_prefix = "OMPBMarket" + OMPB_STAGE02_GENERATION_PREFIX;
   const string market_suffix = ".nnw";
   const int prefix_length = StringLen(market_prefix);
   const int suffix_length = StringLen(market_suffix);
   const int market_length = StringLen(market_file);
   if(market_length <= prefix_length + suffix_length ||
      StringFind(market_file, market_prefix) != 0 ||
      StringSubstr(market_file, market_length - suffix_length, suffix_length) != market_suffix)
      return(false);
   const string generation = StringSubstr(market_file, prefix_length,
                                          market_length - prefix_length - suffix_length);
   if(!OMPBStage02GenerationFiles(generation, expected_market, expected_target, expected_manifest))
      return(false);
   return(market_file == expected_market && target_file == expected_target &&
          manifest_file == expected_manifest);
  }
//+------------------------------------------------------------------+
//| Rejects interrupted publish sidecars instead of guessing recovery.|
//+------------------------------------------------------------------+
bool OMPBStage02RecoverySidecarPresent(string &sidecar_file)
  {
   sidecar_file = "";
   if(FileIsExist(OMPB_STAGE02_LEGACY_TRANSACTION_FILE, FILE_COMMON))
     {
      sidecar_file = OMPB_STAGE02_LEGACY_TRANSACTION_FILE;
      return(true);
     }
   if(FileIsExist(OMPB_STAGE02_LEGACY_TRANSACTION_NEXT_FILE, FILE_COMMON))
     {
      sidecar_file = OMPB_STAGE02_LEGACY_TRANSACTION_NEXT_FILE;
      return(true);
     }
   if(FileIsExist(OMPB_STAGE02_SELECTOR_NEXT, FILE_COMMON))
     {
      sidecar_file = OMPB_STAGE02_SELECTOR_NEXT;
      return(true);
     }
   if(FileIsExist(OMPB_STAGE02_SELECTOR_RESTORE, FILE_COMMON))
     {
      sidecar_file = OMPB_STAGE02_SELECTOR_RESTORE;
      return(true);
     }
//--- Previous is valid only in the process that created it and will remove it
//--- after deterministic reload proof.  On restart it is an interrupted
//--- transaction marker and must fail closed rather than activate unproven data.
   if(FileIsExist(OMPB_STAGE02_SELECTOR_PREVIOUS, FILE_COMMON) &&
      !OMPBStage02PreviousProofReady)
     {
      sidecar_file = OMPB_STAGE02_SELECTOR_PREVIOUS;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Reads exactly one strict selector without allowing path injection.|
//+------------------------------------------------------------------+
bool OMPBStage02ReadSelector(const string selector_file, string &generation)
  {
   generation = "";
   int handle = FileOpen(selector_file, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return(false);
   bool format_seen = false, version_seen = false, generation_seen = false, valid = true;
   string parsed_generation = "";
   while(!FileIsEnding(handle))
     {
      const string line = FileReadString(handle);
      const int separator = StringFind(line, "=");
      if(separator <= 0 || StringFind(line, "=", separator + 1) >= 0)
        {
         valid = false;
         break;
        }
      const string key = StringSubstr(line, 0, separator);
      const string value = StringSubstr(line, separator + 1);
      if(key == "format")
        {
         if(format_seen || value != "OMPB_STAGE02_SELECTOR")
            valid = false;
         format_seen = true;
        }
      else
         if(key == "version")
           {
            if(version_seen || value != IntegerToString(OMPB_STAGE02_SELECTOR_VERSION))
               valid = false;
            version_seen = true;
           }
         else
            if(key == "generation")
              {
               if(generation_seen)
                  valid = false;
               generation_seen = true;
               parsed_generation = value;
              }
            else
               valid = false;
      if(!valid)
         break;
     }
   FileClose(handle);
   if(!valid || !format_seen || !version_seen || !generation_seen ||
      (parsed_generation != "stage01" && !OMPBStage02GenerationIdValid(parsed_generation)))
      return(false);
   generation = parsed_generation;
   return(true);
  }
//+------------------------------------------------------------------+
//| Binds one selector to the exact expected checkpoint generation. |
//+------------------------------------------------------------------+
bool OMPBStage02SelectorMatchesExpected(const string selector_file,
                                        const string expected_generation,
                                        const ulong expected_market,
                                        const ulong expected_target,
                                        const ulong expected_posterior)
  {
   string generation = "", market_file = "", target_file = "", manifest_file = "";
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   if(!FileIsExist(selector_file, FILE_COMMON) ||
      !OMPBStage02ReadSelector(selector_file, generation) || generation != expected_generation ||
      !OMPBStage02GenerationFiles(generation, market_file, target_file, manifest_file) ||
      !OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !OMPBStage02CheckpointFingerprints(market_file, target_file, market, target, posterior) ||
      market != expected_market || target != expected_target || posterior != expected_posterior)
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Writes and rereads one non-active selector file before a switch.|
//+------------------------------------------------------------------+
bool OMPBStage02WriteSelectorFile(const string selector_file, const string generation)
  {
   if(generation != "stage01" && !OMPBStage02GenerationIdValid(generation))
      return(false);
   int handle = FileOpen(selector_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return(false);
   const bool written = (FileWrite(handle, "format=OMPB_STAGE02_SELECTOR") > 0 &&
                         FileWrite(handle, StringFormat("version=%u",
                                   OMPB_STAGE02_SELECTOR_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("generation=%s", generation)) > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   string reread = "";
   return(written && OMPBStage02ReadSelector(selector_file, reread) && reread == generation);
  }
//+------------------------------------------------------------------+
//| Ensures three tuple names describe three distinct physical files.|
//+------------------------------------------------------------------+
bool OMPBStage02TupleNamesValid(const string market_file, const string target_file,
                                const string manifest_file)
  {
   return(market_file != "" && target_file != "" && manifest_file != "" &&
          market_file != target_file && market_file != manifest_file &&
          target_file != manifest_file);
  }
//+------------------------------------------------------------------+
//| Copies a complete checkpoint tuple with manifest last.           |
//+------------------------------------------------------------------+
bool OMPBStage02CopyTuple(const string source_market, const string source_target,
                          const string source_manifest, const string destination_market,
                          const string destination_target, const string destination_manifest,
                          const bool rewrite)
  {
   if(!OMPBStage02TupleNamesValid(source_market, source_target, source_manifest) ||
      !OMPBStage02TupleNamesValid(destination_market, destination_target,
                                  destination_manifest))
      return(false);
   const uint flags = (rewrite ? FILE_COMMON | FILE_REWRITE : FILE_COMMON);
   return(FileCopy(source_market, FILE_COMMON, destination_market, flags) &&
          FileCopy(source_target, FILE_COMMON, destination_target, flags) &&
          FileCopy(source_manifest, FILE_COMMON, destination_manifest, flags));
  }
//+------------------------------------------------------------------+
//| Publishes a staged tuple through three canonical active filenames.|
//| The staged and previous tuples remain until reload proof completes.|
//+------------------------------------------------------------------+
bool OMPBStage02PublishCanonicalTuple(const string market_file, const string target_file,
                                      const string manifest_file, const string next_market,
                                      const string next_target, const string next_manifest,
                                      const string previous_market,
                                      const string previous_target,
                                      const string previous_manifest)
  {
   if(!OMPBStage02TupleNamesValid(market_file, target_file, manifest_file) ||
      !OMPBStage02TupleNamesValid(next_market, next_target, next_manifest) ||
      !OMPBStage02TupleNamesValid(previous_market, previous_target, previous_manifest) ||
      !OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !OMPBStage02ValidateCheckpointSet(next_market, next_target, next_manifest) ||
      FileIsExist(previous_market, FILE_COMMON) || FileIsExist(previous_target, FILE_COMMON) ||
      FileIsExist(previous_manifest, FILE_COMMON))
      return(false);
   if(!OMPBStage02CopyTuple(market_file, target_file, manifest_file, previous_market,
                            previous_target, previous_manifest, false) ||
      !OMPBStage02ValidateCheckpointSet(previous_market, previous_target, previous_manifest))
      return(false);
   return(OMPBStage02CopyTuple(next_market, next_target, next_manifest, market_file,
                               target_file, manifest_file, true) &&
          OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file));
  }
//+------------------------------------------------------------------+
//| Restores the retained tuple using the manifest as the final write.|
//+------------------------------------------------------------------+
bool OMPBStage02RestoreCanonicalTuple(const string market_file, const string target_file,
                                      const string manifest_file, const string previous_market,
                                      const string previous_target,
                                      const string previous_manifest)
  {
   if(!OMPBStage02TupleNamesValid(market_file, target_file, manifest_file) ||
      !OMPBStage02TupleNamesValid(previous_market, previous_target, previous_manifest) ||
      !OMPBStage02ValidateCheckpointSet(previous_market, previous_target, previous_manifest))
      return(false);
   return(OMPBStage02CopyTuple(previous_market, previous_target, previous_manifest, market_file,
                               target_file, manifest_file, true) &&
          OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file));
  }
//+------------------------------------------------------------------+
//| Classifies the retained tuple required by one Stage 01 phase.    |
//+------------------------------------------------------------------+
bool D2SkillStage01TupleStateValid(const bool market_exists,
                                   const bool target_exists,
                                   const bool manifest_exists,
                                   const bool had_previous,
                                   const string phase)
  {
//--- The first save has no prior tuple; every later phase retains all members.
   const bool complete = (market_exists && target_exists && manifest_exists);
   if(phase == "prepared")
      return(had_previous ? complete : !market_exists && !target_exists && !manifest_exists);
   if(phase == "previous_ready")
      return(had_previous && complete);
   return(false);
  }
//+------------------------------------------------------------------+
//| Removes only files owned by a completed Stage 01 transaction.   |
//+------------------------------------------------------------------+
bool D2SkillStage01DeleteFile(const string file_name)
  {
   return(!FileIsExist(file_name, FILE_COMMON) || FileDelete(file_name, FILE_COMMON));
  }
//+------------------------------------------------------------------+
bool D2SkillStage01Cleanup(void)
  {
//--- Remove the commit marker last; its presence must survive every partial cleanup.
   return(D2SkillStage01DeleteFile(OMPB_STAGE01_MANIFEST_NEXT_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_TARGET_NEXT_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_MARKET_NEXT_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_MANIFEST_PREVIOUS_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_TARGET_PREVIOUS_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_MARKET_PREVIOUS_FILE) &&
          D2SkillStage01DeleteFile(OMPB_STAGE01_TRANSACTION_FILE));
  }
//+------------------------------------------------------------------+
//| Writes one durable marker before canonical Stage 01 publication.|
//+------------------------------------------------------------------+
bool D2SkillStage01WriteTransaction(const bool had_previous, const string phase,
                                    const ulong market, const ulong target,
                                    const ulong posterior)
  {
   if((phase != "prepared" && phase != "previous_ready") ||
      FileIsExist(OMPB_STAGE01_TRANSACTION_FILE, FILE_COMMON))
      return(false);
   int handle = FileOpen(OMPB_STAGE01_TRANSACTION_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return(false);
//--- The candidate fingerprints identify an already-published complete tuple.
   const bool written = (FileWrite(handle, "format=OMPB_STAGE01_TRANSACTION") > 0 &&
                         FileWrite(handle, StringFormat("version=%u", OMPB_STAGE01_TRANSACTION_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("phase=%s", phase)) > 0 &&
                         FileWrite(handle, StringFormat("had_previous=%s",
                                   (had_previous ? "true" : "false"))) > 0 &&
                         FileWrite(handle, StringFormat("candidate_market=%I64u", market)) > 0 &&
                         FileWrite(handle, StringFormat("candidate_target=%I64u", target)) > 0 &&
                         FileWrite(handle, StringFormat("candidate_posterior=%I64u", posterior)) > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   return(written);
  }
//+------------------------------------------------------------------+
//| Rewrites only a complete transaction marker state transition.   |
//+------------------------------------------------------------------+
bool D2SkillStage01AdvanceTransaction(const bool had_previous, const string phase,
                                      const ulong market, const ulong target,
                                      const ulong posterior)
  {
   if(!D2SkillStage01DeleteFile(OMPB_STAGE01_TRANSACTION_FILE))
      return(false);
   return(D2SkillStage01WriteTransaction(had_previous, phase, market, target, posterior));
  }
//+------------------------------------------------------------------+
//| Reads one strict Stage 01 transaction marker.                   |
//+------------------------------------------------------------------+
bool D2SkillStage01ReadTransaction(bool &had_previous, string &phase,
                                   string &market, string &target, string &posterior)
  {
   had_previous = false;
   phase = "";
   market = "";
   target = "";
   posterior = "";
   if(!FileIsExist(OMPB_STAGE01_TRANSACTION_FILE, FILE_COMMON))
      return(false);
   const string format = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "format");
   const string version = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "version");
   const string previous = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "had_previous");
   phase = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "phase");
   market = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "candidate_market");
   target = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "candidate_target");
   posterior = D2SkillManifestValue(OMPB_STAGE01_TRANSACTION_FILE, "candidate_posterior");
   if(format != "OMPB_STAGE01_TRANSACTION" ||
      version != IntegerToString(OMPB_STAGE01_TRANSACTION_VERSION) ||
      (previous != "true" && previous != "false") ||
      (phase != "prepared" && phase != "previous_ready") ||
      market == "" || target == "" || posterior == "")
      return(false);
   had_previous = (previous == "true");
   return(true);
  }
//+------------------------------------------------------------------+
//| Compares a complete tuple against its transaction marker.       |
//+------------------------------------------------------------------+
bool D2SkillStage01TupleMatches(const string market_file, const string target_file,
                                const string manifest_file, const string market,
                                const string target, const string posterior)
  {
   ulong actual_market = 0, actual_target = 0;
   ulong actual_posterior = ulong(1469598103934665603);
   return(OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file) &&
          OMPBStage02CheckpointFingerprints(market_file, target_file, actual_market,
                                            actual_target, actual_posterior) &&
          StringFormat("%I64u", actual_market) == market &&
          StringFormat("%I64u", actual_target) == target &&
          StringFormat("%I64u", actual_posterior) == posterior);
  }
//+------------------------------------------------------------------+
//| Recovers only an interrupted Stage 01-owned publication.        |
//+------------------------------------------------------------------+
bool D2SkillStage01RecoverInterruptedCheckpoint(void)
  {
   const bool marker_exists = FileIsExist(OMPB_STAGE01_TRANSACTION_FILE, FILE_COMMON);
   const bool temporary_exists = (FileIsExist(OMPB_STAGE01_MARKET_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(OMPB_STAGE01_TARGET_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(OMPB_STAGE01_MANIFEST_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(OMPB_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
                                  FileIsExist(OMPB_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
                                  FileIsExist(OMPB_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON));
   if(!marker_exists)
     {
//--- Uncommitted staging never changed canonical names and is safe to discard.
      if(!temporary_exists)
         return(true);
      if(!OMPBStage02ValidateCheckpointSet(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                           D2Skill_MANIFEST_FILE) &&
         !OMPBStage02CanonicalTupleMissing(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                            D2Skill_MANIFEST_FILE))
         return(false);
      return(D2SkillStage01Cleanup());
     }
   bool had_previous = false;
   string phase = "", market = "", target = "", posterior = "";
   if(!D2SkillStage01ReadTransaction(had_previous, phase, market, target, posterior))
      return(false);
//--- A complete candidate published before cleanup is already the committed epoch.
   if(D2SkillStage01TupleMatches(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                 D2Skill_MANIFEST_FILE, market, target, posterior))
      return(D2SkillStage01Cleanup());
//--- Before previous-ready canonical names are still the earlier valid tuple.
   if(phase == "prepared" && OMPBStage02ValidateCheckpointSet(D2Skill_MARKET_FILE,
      D2Skill_TARGET_FILE, D2Skill_MANIFEST_FILE))
      return(D2SkillStage01Cleanup());
   if(had_previous)
     {
      if(!D2SkillStage01TupleStateValid(FileIsExist(OMPB_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON),
         FileIsExist(OMPB_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON),
         FileIsExist(OMPB_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON), true, phase) ||
         !OMPBStage02RestoreCanonicalTuple(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                           D2Skill_MANIFEST_FILE,
                                           OMPB_STAGE01_MARKET_PREVIOUS_FILE,
                                           OMPB_STAGE01_TARGET_PREVIOUS_FILE,
                                           OMPB_STAGE01_MANIFEST_PREVIOUS_FILE))
         return(false);
      return(D2SkillStage01Cleanup());
     }
//--- The first publication has no prior checkpoint; remove only marker-owned partial data.
   if(!D2SkillStage01DeleteFile(D2Skill_MANIFEST_FILE) ||
      !D2SkillStage01DeleteFile(D2Skill_TARGET_FILE) ||
      !D2SkillStage01DeleteFile(D2Skill_MARKET_FILE))
      return(false);
   return(D2SkillStage01Cleanup());
  }
//+------------------------------------------------------------------+
//| Publishes one fully validated Stage 01 checkpoint transaction.  |
//+------------------------------------------------------------------+
bool D2SkillStage01SaveCheckpoint(const uint completed_epochs)
  {
   if(FileIsExist(OMPB_STAGE01_TRANSACTION_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_MARKET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_TARGET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON))
      return(false);
//--- Classify the pre-save canonical state before staging any new model bytes.
   const bool had_previous = OMPBStage02ValidateCheckpointSet(D2Skill_MARKET_FILE,
                                                               D2Skill_TARGET_FILE,
                                                               D2Skill_MANIFEST_FILE);
   if(!had_previous && !OMPBStage02CanonicalTupleMissing(D2Skill_MARKET_FILE,
      D2Skill_TARGET_FILE, D2Skill_MANIFEST_FILE))
      return(false);
//--- The next tuple is self-contained and validated before a commit marker exists.
   const datetime now = TimeCurrent();
   if(!D2SkillMarket.Save(OMPB_STAGE01_MARKET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !D2SkillTarget.Save(OMPB_STAGE01_TARGET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !OMPBStage02WriteCandidateManifest(OMPB_STAGE01_MARKET_NEXT_FILE,
         OMPB_STAGE01_TARGET_NEXT_FILE, OMPB_STAGE01_MANIFEST_NEXT_FILE, completed_epochs) ||
      !OMPBStage02ValidateCheckpointSet(OMPB_STAGE01_MARKET_NEXT_FILE,
                                        OMPB_STAGE01_TARGET_NEXT_FILE,
                                        OMPB_STAGE01_MANIFEST_NEXT_FILE))
      return(false);
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   if(!OMPBStage02CheckpointFingerprints(OMPB_STAGE01_MARKET_NEXT_FILE,
                                         OMPB_STAGE01_TARGET_NEXT_FILE,
                                         market, target, posterior) ||
      !D2SkillStage01WriteTransaction(had_previous, "prepared", market, target, posterior))
      return(false);
//--- Retain the old complete tuple before any short canonical file is replaced.
   if(had_previous &&
      (!OMPBStage02CopyTuple(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE, D2Skill_MANIFEST_FILE,
                              OMPB_STAGE01_MARKET_PREVIOUS_FILE,
                              OMPB_STAGE01_TARGET_PREVIOUS_FILE,
                              OMPB_STAGE01_MANIFEST_PREVIOUS_FILE, false) ||
       !OMPBStage02ValidateCheckpointSet(OMPB_STAGE01_MARKET_PREVIOUS_FILE,
                                         OMPB_STAGE01_TARGET_PREVIOUS_FILE,
                                         OMPB_STAGE01_MANIFEST_PREVIOUS_FILE)))
      return(false);
   if(!D2SkillStage01AdvanceTransaction(had_previous, "previous_ready", market, target, posterior) ||
      !OMPBStage02CopyTuple(OMPB_STAGE01_MARKET_NEXT_FILE, OMPB_STAGE01_TARGET_NEXT_FILE,
                             OMPB_STAGE01_MANIFEST_NEXT_FILE, D2Skill_MARKET_FILE,
                             D2Skill_TARGET_FILE, D2Skill_MANIFEST_FILE, true) ||
      !D2SkillStage01TupleMatches(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                  D2Skill_MANIFEST_FILE, StringFormat("%I64u", market),
                                  StringFormat("%I64u", target), StringFormat("%I64u", posterior)))
      return(false);
   if(!D2SkillStage01Cleanup())
      return(false);
   D2SkillLastSignature = D2SkillForecastSignature(D2Skill_MARKET_FILE);
   PrintFormat("OMPB_STAGE01_CHECKPOINT_PASS epoch=%u", completed_epochs);
   return(true);
  }
//+------------------------------------------------------------------+
//| Proves deterministic serialized parameters without live mutation.|
//+------------------------------------------------------------------+
bool OMPBStage02ReloadProofDeterministicTuple(const string market_file,
      const string target_file,
      const string manifest_file)
  {
   ulong first_market = 0, first_target = 0;
   ulong first_posterior = ulong(1469598103934665603);
   ulong second_market = 0, second_target = 0;
   ulong second_posterior = ulong(1469598103934665603);
   return(OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file) &&
          OMPBStage02CheckpointFingerprints(market_file, target_file, first_market,
                first_target, first_posterior) &&
          OMPBStage02CheckpointFingerprints(market_file, target_file, second_market,
                second_target, second_posterior) &&
          first_market != 0 && first_target != 0 && first_posterior != 0 &&
          first_market == second_market && first_target == second_target &&
          first_posterior == second_posterior);
  }
//+------------------------------------------------------------------+
//| Detects any interrupted canonical or legacy publication state.   |
//+------------------------------------------------------------------+
bool OMPBStage02CanonicalSidecarPresent(string &sidecar_file)
  {
   sidecar_file = "";
   const string sidecars[] =
     {
      OMPB_STAGE02_MARKET_NEXT_FILE,
      OMPB_STAGE02_TARGET_NEXT_FILE,
      OMPB_STAGE02_MANIFEST_NEXT_FILE,
      OMPB_STAGE02_MARKET_PREVIOUS_FILE,
      OMPB_STAGE02_TARGET_PREVIOUS_FILE,
      OMPB_STAGE02_MANIFEST_PREVIOUS_FILE,
      OMPB_STAGE02_ACTIVE_SELECTOR,
      OMPB_STAGE02_SELECTOR_NEXT,
      OMPB_STAGE02_SELECTOR_PREVIOUS,
      OMPB_STAGE02_SELECTOR_RESTORE,
      OMPB_STAGE02_LEGACY_TRANSACTION_FILE,
      OMPB_STAGE02_LEGACY_TRANSACTION_NEXT_FILE
     };
   for(int index = 0; index < ArraySize(sidecars); ++index)
      if(FileIsExist(sidecars[index], FILE_COMMON))
        {
         sidecar_file = sidecars[index];
         return(true);
        }
   return(false);
  }
//+------------------------------------------------------------------+
//| Resolves only the canonical production checkpoint or fails closed.|
//+------------------------------------------------------------------+
bool OMPBStage02ResolveActiveCheckpoint(void)
  {
   OMPBStage02RecoveryFailed = false;
   OMPBStage02ResetActiveCheckpointFiles();
//--- Complete a Stage 01-owned recovery before interpreting canonical files.
   if(!D2SkillStage01RecoverInterruptedCheckpoint())
     {
      OMPBStage02RecoveryFailed = true;
      Print("OMPB_STAGE01_CHECKPOINT_FAIL reason=recovery");
      return(false);
     }
   string sidecar_file = "";
   if(OMPBStage02CanonicalSidecarPresent(sidecar_file))
     {
      OMPBStage02RecoveryFailed = true;
      Print("OMPB_STAGE02_CANONICAL_FAIL reason=incomplete_or_legacy_transaction");
      return(false);
     }
//--- No tuple exists before the first Stage 01 run; its caller creates a new graph.
   if(OMPBStage02CanonicalTupleMissing(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                       D2Skill_MANIFEST_FILE))
     {
      Print("OMPB_STAGE02_CANONICAL_MISS reason=initial_checkpoint_absent");
      return(false);
     }
//--- A partial or unreadable tuple is not a new-model state and remains fail-closed.
   if(!OMPBStage02ValidateCheckpointSet(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                        D2Skill_MANIFEST_FILE))
     {
      OMPBStage02RecoveryFailed = true;
      Print("OMPB_STAGE02_CANONICAL_FAIL reason=invalid_active_tuple");
      return(false);
     }
   Print("OMPB_STAGE02_CANONICAL_PASS checkpoint=active");
   return(true);
  }
//+------------------------------------------------------------------+
//| Retains the previous selection before the one-file activation.  |
//+------------------------------------------------------------------+
bool OMPBStage02PreparePreviousSelector(void)
  {
   string generation = "", market_file = "", target_file = "", manifest_file = "";
   string sidecar_file = "";
   OMPBStage02PreviousProofReady = false;
   if(OMPBStage02RecoverySidecarPresent(sidecar_file) ||
      FileIsExist(OMPB_STAGE02_SELECTOR_PREVIOUS, FILE_COMMON))
      return(false);
   if(!FileIsExist(OMPB_STAGE02_ACTIVE_SELECTOR, FILE_COMMON))
     {
      generation = "stage01";
      market_file = D2Skill_MARKET_FILE;
      target_file = D2Skill_TARGET_FILE;
      manifest_file = D2Skill_MANIFEST_FILE;
     }
   else
      if(!OMPBStage02ReadSelector(OMPB_STAGE02_ACTIVE_SELECTOR, generation) ||
         !OMPBStage02GenerationFiles(generation, market_file, target_file, manifest_file))
         return(false);
   if(!OMPBStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !OMPBStage02CheckpointFingerprints(market_file, target_file,
                                         OMPBStage02PreviousMarketFingerprint,
                                         OMPBStage02PreviousTargetFingerprint,
                                         OMPBStage02PreviousPosteriorFingerprint))
      return(false);
   if(generation == "stage01")
     {
      if(!OMPBStage02WriteSelectorFile(OMPB_STAGE02_SELECTOR_PREVIOUS, generation))
         return(false);
     }
   else
      if(!FileCopy(OMPB_STAGE02_ACTIVE_SELECTOR, FILE_COMMON, OMPB_STAGE02_SELECTOR_PREVIOUS,
                   FILE_COMMON | FILE_REWRITE))
         return(false);
   string copied = "";
   if(!OMPBStage02ReadSelector(OMPB_STAGE02_SELECTOR_PREVIOUS, copied) || copied != generation)
      return(false);
   OMPBStage02PreviousProofReady = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Switches exactly one active file after full generation proof.   |
//+------------------------------------------------------------------+
bool OMPBStage02ActivateGeneration(const string generation)
  {
   if(FileIsExist(OMPB_STAGE02_SELECTOR_NEXT, FILE_COMMON) ||
      !OMPBStage02WriteSelectorFile(OMPB_STAGE02_SELECTOR_NEXT, generation))
      return(false);
   if(!FileMove(OMPB_STAGE02_SELECTOR_NEXT, FILE_COMMON, OMPB_STAGE02_ACTIVE_SELECTOR,
                FILE_COMMON | FILE_REWRITE))
     {
      Print("OMPB_STAGE02_SELECTOR_FAIL reason=activation_move next_retained=true");
      return(false);
     }
   OMPBStage02SelectorActivated = true;
   PrintFormat("OMPB_STAGE02_SELECTOR_SWITCH_PASS generation=%s", generation);
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores a retained canonical tuple after failed reload proof.   |
//+------------------------------------------------------------------+
bool OMPBStage02RestorePreviousSelector(void)
  {
   ulong previous_market = 0, previous_target = 0;
   ulong previous_posterior = ulong(1469598103934665603);
   if(!OMPBStage02PreviousProofReady ||
      !OMPBStage02CheckpointFingerprints(OMPB_STAGE02_MARKET_PREVIOUS_FILE,
                                         OMPB_STAGE02_TARGET_PREVIOUS_FILE,
                                         previous_market, previous_target, previous_posterior) ||
      previous_market != OMPBStage02PreviousMarketFingerprint ||
      previous_target != OMPBStage02PreviousTargetFingerprint ||
      previous_posterior != OMPBStage02PreviousPosteriorFingerprint)
      return(false);
   if(!OMPBStage02RestoreCanonicalTuple(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                        D2Skill_MANIFEST_FILE,
                                        OMPB_STAGE02_MARKET_PREVIOUS_FILE,
                                        OMPB_STAGE02_TARGET_PREVIOUS_FILE,
                                        OMPB_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   OMPBStage02SelectorActivated = false;
   OMPBStage02ReloadProofPassed = false;
   if(!OMPBStage02ReloadCheckpointProof(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                        D2Skill_MANIFEST_FILE,
                                        OMPBStage02PreviousMarketFingerprint,
                                        OMPBStage02PreviousTargetFingerprint,
                                        OMPBStage02PreviousPosteriorFingerprint) ||
      !OMPBStage02FinalizeSteadySelector())
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Stages and reload-probes the complete temporary canonical tuple.|
//+------------------------------------------------------------------+
bool OMPBStage02StageCandidate(void)
  {
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !ompb.AppendParameterFingerprint(posterior) ||
      !OMPBStage02ResolveActiveCheckpoint())
      return(false);
   if(FileIsExist(OMPB_STAGE02_MARKET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE02_TARGET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE02_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE02_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE02_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(OMPB_STAGE02_MANIFEST_PREVIOUS_FILE, FILE_COMMON))
      return(false);
   const datetime now = TimeCurrent();
   if(!D2SkillMarket.Save(OMPB_STAGE02_MARKET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !D2SkillTarget.Save(OMPB_STAGE02_TARGET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !OMPBStage02WriteCandidateManifest(OMPB_STAGE02_MARKET_NEXT_FILE,
                                         OMPB_STAGE02_TARGET_NEXT_FILE,
                                         OMPB_STAGE02_MANIFEST_NEXT_FILE,
                                         D2SkillCompletedEpochs) ||
      !OMPBStage02ValidateCandidate(OMPB_STAGE02_MARKET_NEXT_FILE,
                                    OMPB_STAGE02_TARGET_NEXT_FILE,
                                    OMPB_STAGE02_MANIFEST_NEXT_FILE))
     {
      Print("OMPB_STAGE02_CANONICAL_FAIL reason=stage_or_reload_probe");
      return(false);
     }
   OMPBStage02StagedGeneration = "canonical";
   PrintFormat("OMPB_STAGE02_CANONICAL_STAGE_PASS posterior=%I64u", posterior);
   return(true);
  }
//+------------------------------------------------------------------+
//| Publishes only a fully validated temporary canonical tuple.     |
//+------------------------------------------------------------------+
bool OMPBStage02PublishCandidate(void)
  {
   if(!OMPBStage02CheckpointFingerprints(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                         OMPBStage02PreviousMarketFingerprint,
                                         OMPBStage02PreviousTargetFingerprint,
                                         OMPBStage02PreviousPosteriorFingerprint) ||
      !OMPBStage02PublishCanonicalTuple(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                        D2Skill_MANIFEST_FILE,
                                        OMPB_STAGE02_MARKET_NEXT_FILE,
                                        OMPB_STAGE02_TARGET_NEXT_FILE,
                                        OMPB_STAGE02_MANIFEST_NEXT_FILE,
                                        OMPB_STAGE02_MARKET_PREVIOUS_FILE,
                                        OMPB_STAGE02_TARGET_PREVIOUS_FILE,
                                        OMPB_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   OMPBStage02PreviousProofReady = true;
   OMPBStage02SelectorActivated = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Loads exactly the supplied checkpoint without resolving a selector.|
//+------------------------------------------------------------------+
bool OMPBStage02LoadExplicitCheckpoint(const string market_file, const string target_file,
                                       const string manifest_file)
  {
   if(!OMPBStage02ExplicitCheckpointFilesValid(market_file, target_file, manifest_file))
      return(false);
   D2SkillFrozenBaselineReady = false;
   D2SkillForecast = NULL;
   if(!FileIsExist(market_file, FILE_COMMON) || !FileIsExist(target_file, FILE_COMMON) ||
      !FileIsExist(manifest_file, FILE_COMMON))
      return(false);
//--- The explicit loader retains this validated tuple as its current in-memory
//--- view for its caller's fingerprint/finalize work; it never resolves selectors.
   D2SkillActiveMarketFile = market_file;
   D2SkillActiveTargetFile = target_file;
   D2SkillActiveManifestFile = manifest_file;
   if(!D2SkillValidateForecastManifestHeader())
      return(false);
   float error = 0.0f, undefine = 0.0f, forecast = 0.0f;
   datetime studied = 0;
   if(!D2SkillMarket.Load(market_file, error, undefine, forecast, studied, true) ||
      !D2SkillTarget.Load(target_file, error, undefine, forecast, studied, true))
      return(false);
   if(!D2SkillTarget.SetOpenCLChecked(D2SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   D2SkillForecast = (CNeuronScenarioForecast *)D2SkillMarket.Layer(-1);
   if(!D2SkillForecast || D2SkillForecast.Type() != defNeuronScenarioForecast ||
      !ConfigureForecastRecoveryAge() || !D2SkillInitTrainingBuffers() ||
      !D2SkillValidateShapes())
      return(false);
   const ulong signature = D2SkillForecastSignature();
   const string completed = D2SkillManifestValue(manifest_file, "completed_epochs");
   const string batches = D2SkillManifestValue(manifest_file, "training_batches");
   const string invalid = D2SkillManifestValue(manifest_file, "invalid_batches");
   if(!D2SkillValidateForecastManifest(signature, true) || completed == "" || batches == "" ||
      invalid == "")
      return(false);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !D2SkillMarket.SetWeightsUpdate(false) || !D2SkillTarget.SetWeightsUpdate(false) ||
      !D2SkillForecast.SetCodebookUpdate(false) || !ConfigureOMPB(OMPB_INFERENCE) || !ompb.Clear() ||
      !D2SkillCaptureFrozenForecastBaseline(D2SkillForecast))
      return(false);
   D2SkillCompletedEpochs = (uint)StringToInteger(completed);
   D2SkillBatches = (ulong)StringToInteger(batches);
   D2SkillInvalidBatches = (ulong)StringToInteger(invalid);
   D2SkillMarket.TrainMode(false);
   D2SkillTarget.TrainMode(false);
   ompb.TrainMode(false);
   D2SkillLastSignature = signature;
   return(true);
  }
//+------------------------------------------------------------------+
//| Compares one immutable reload output and names the failed tensor.|
//+------------------------------------------------------------------+
bool OMPBStage02ReloadTensorEqual(const string tensor, CBufferFloat *actual,
                                  CBufferFloat &expected)
  {
   if(!actual || !actual.BufferRead() || actual.Total() != expected.Total())
     {
      PrintFormat("OMPB_STAGE02_RELOAD_TENSOR_FAIL tensor=%s reason=shape actual=%d expected=%d",
                  tensor, (actual ? actual.Total() : -1), expected.Total());
      return(false);
     }
   for(int index = 0; index < actual.Total(); index++)
     {
      if(!MathIsValidNumber(actual[index]) || !MathIsValidNumber(expected[index]) ||
         actual[index] != expected[index])
        {
         PrintFormat("OMPB_STAGE02_RELOAD_TENSOR_FAIL tensor=%s index=%d first=%.9g second=%.9g",
                     tensor, index, expected[index], actual[index]);
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Proves the first inference result after two clean explicit loads.|
//+------------------------------------------------------------------+
bool OMPBStage02ReloadProofDeterministic(const string market_file, const string target_file,
      const string manifest_file, const int position,
      CBufferFloat *state, CBufferFloat *time)
  {
   CBufferFloat z, u, pi;
   if(position < 0 || !state || !time ||
      !OMPBStage02LoadExplicitCheckpoint(market_file, target_file, manifest_file) ||
      !D2SkillForwardForecast(position, state, time) ||
      !D2SkillCaptureFrozenBuffer(D2SkillForecast.GetZ(), z) ||
      !D2SkillCaptureFrozenBuffer(D2SkillForecast.GetU(), u) ||
      !D2SkillCaptureFrozenBuffer(D2SkillForecast.GetPi(), pi) ||
      !OMPBStage02LoadExplicitCheckpoint(market_file, target_file, manifest_file) ||
      !D2SkillForwardForecast(position, state, time) ||
      !OMPBStage02ReloadTensorEqual("Z", D2SkillForecast.GetZ(), z) ||
      !OMPBStage02ReloadTensorEqual("U", D2SkillForecast.GetU(), u) ||
      !OMPBStage02ReloadTensorEqual("Pi", D2SkillForecast.GetPi(), pi) ||
      !D2SkillVerifyFrozenForecastExact(D2SkillForecast))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Reloads one selected checkpoint and proves deterministic mean inference.|
//+------------------------------------------------------------------+
bool OMPBStage02ReloadCheckpointProof(const string market_file, const string target_file,
                                      const string manifest_file, const ulong expected_market,
                                      const ulong expected_target, const ulong expected_posterior)
  {
   OMPBStage02ReloadProofPassed = false;
   if(!OMPBStage02ReloadProofDeterministic(market_file, target_file, manifest_file,
                                           OMPBStage02TargetEvalFirst, GetPointer(D2SkillState),
                                           GetPointer(D2SkillTime)))
      return(false);
   CNeuronOMPBOCL *ompb = GetOMPB();
   ulong market = 0, target = 0, posterior = ulong(1469598103934665603);
   if(!ompb || !D2SkillMarket.ParameterFingerprint(market) ||
      !D2SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      market != expected_market || target != expected_target || posterior != expected_posterior)
      return(false);
   OMPBStage02ReloadProofPassed = true;
   Print("OMPB_STAGE02_RELOAD_SMOKE_PASS mode=inference deterministic=true");
   return(true);
  }
//+------------------------------------------------------------------+
//| Reloads the canonical tuple and proves deterministic mean inference.|
//+------------------------------------------------------------------+
bool OMPBStage02ReloadSmoke(const ulong expected_market, const ulong expected_target,
                            const ulong expected_posterior)
  {
   if(!OMPBStage02SelectorActivated)
      return(false);
   return(OMPBStage02ReloadCheckpointProof(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                           D2Skill_MANIFEST_FILE,
                                           expected_market, expected_target, expected_posterior));
  }
//+------------------------------------------------------------------+
//| Cleans temporary tuples only after deterministic canonical reload.|
//+------------------------------------------------------------------+
bool OMPBStage02FinalizeSteadySelector(void)
  {
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!OMPBStage02ReloadProofPassed ||
      !OMPBStage02ValidateCheckpointSet(D2Skill_MARKET_FILE, D2Skill_TARGET_FILE,
                                        D2Skill_MANIFEST_FILE) ||
      !OMPBStage02ValidateCheckpointSet(OMPB_STAGE02_MARKET_NEXT_FILE,
                                        OMPB_STAGE02_TARGET_NEXT_FILE,
                                        OMPB_STAGE02_MANIFEST_NEXT_FILE) ||
      !OMPBStage02ValidateCheckpointSet(OMPB_STAGE02_MARKET_PREVIOUS_FILE,
                                        OMPB_STAGE02_TARGET_PREVIOUS_FILE,
                                        OMPB_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   if(!ompb || !D2SkillMarket.ParameterFingerprint(market) ||
      !D2SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      market == 0 || target == 0 || posterior == 0)
      return(false);
   if(!FileDelete(OMPB_STAGE02_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(OMPB_STAGE02_TARGET_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(OMPB_STAGE02_MARKET_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(OMPB_STAGE02_MANIFEST_PREVIOUS_FILE, FILE_COMMON) ||
      !FileDelete(OMPB_STAGE02_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      !FileDelete(OMPB_STAGE02_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      !OMPBStage02ResolveActiveCheckpoint())
      return(false);
   OMPBStage02PreviousProofReady = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Publishes only after canonical staging and deterministic reload. |
//+------------------------------------------------------------------+
bool OMPBStage02SaveAcceptedCheckpoint(void)
  {
   ulong market = 0, target = 0, posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetOMPB();
   OMPBStage02ReloadProofPassed = false;
   OMPBStage02PreviousProofReady = false;
   OMPBStage02SelectorActivated = false;
   if(!ompb || !D2SkillMarket.ParameterFingerprint(market) ||
      !D2SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      !OMPBStage02StageCandidate() || !OMPBStage02PublishCandidate())
      return(false);
   if(!OMPBStage02ReloadSmoke(market, target, posterior) || !OMPBStage02FinalizeSteadySelector())
     {
      const bool restored = OMPBStage02RestorePreviousSelector();
      PrintFormat("OMPB_STAGE02_CANONICAL_FAIL reason=reload_or_finalize restored=%s",
                  (restored ? "true" : "false"));
      return(false);
     }
   OMPBStage02CheckpointPublished = true;
   Print("OMPB_STAGE02_CANONICAL_PASS checkpoint=published");
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs one frozen-graph OMPB update.  CNet owns full gradient      |
//| propagation but has parameter writes disabled; the explicit      |
//| OMPB call below is therefore the only update in Stage 02.        |
//+------------------------------------------------------------------+
bool OMPBStage02RunBatch(const int position, const bool source_anchor)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   CNeuronBaseOCL *rank_tcm = GetRankTCM();
   if(!ompb || !rank_tcm || !ConfigureOMPB(OMPB_CALIBRATE) ||
      !ompb.SetSourceAnchor(source_anchor))
      ReturnFalse;
   ompb.TrainMode(true);
   if(!D2SkillTrainBatch(position))
      ReturnFalse;
//--- A source anchor has already created its mean Forecast gradient.  Its
//--- forward/backward phase must retain the anchor flag so Current and both
//--- regularizers remain untouched.  Clear only the update guard afterwards
//--- to apply that already-computed mean posterior gradient.
   if(source_anchor && !ompb.SetSourceAnchor(false))
      ReturnFalse;
   return(ompb.UpdateInputWeights(rank_tcm));
  }
//+------------------------------------------------------------------+
//| Stage 02 online calibration.  The outer network stays frozen;   |
//| only the OMPB posterior and alpha receive explicit updates.      |
//+------------------------------------------------------------------+
bool OMPBStage02Calibrate(void)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !D2SkillForecast ||
      !D2SkillMarket.SetWeightsUpdate(false) || !D2SkillTarget.SetWeightsUpdate(false) ||
      !D2SkillForecast.SetCodebookUpdate(false) || !D2SkillMarket.Clear() ||
      !D2SkillTarget.Clear() || !D2SkillForecast.ResetEpochDiagnostics() ||
      !ConfigureOMPB(OMPB_CALIBRATE))
     {
      Print("OMPB_STAGE02_CALIBRATION_FAIL reason=setup");
      ReturnFalse;
     }
   D2SkillMarket.TrainMode(true);
   D2SkillTarget.TrainMode(false);
   ompb.TrainMode(true);
   uint target_attempts = 0;
   uint target_updates = 0;
   uint target_invalid = 0;
   uint source_attempts = 0;
   uint source_updates = 0;
   uint source_invalid = 0;
   const uint source_rows = uint(OMPBStage02ReferenceFirst - OMPBStage02ReferenceLast + 1);
   if(source_rows == 0)
     {
      Print("OMPB_STAGE02_CALIBRATION_FAIL reason=source_rows");
      ReturnFalse;
     }
   for(int position = OMPBStage02TargetCalibrationFirst;
       position >= OMPBStage02TargetCalibrationLast && !IsStopped(); position--)
     {
      target_attempts++;
      if(!OMPBStage02RunBatch(position, false))
        {
         target_invalid++;
        }
      else
        {
         uint reference_count, current_count, invalid_fallbacks, kl_rejects;
         float disagreement, kl, alpha_prior;
         if(!ReadOMPBDiagnostics(reference_count, current_count, disagreement, kl,
                                 alpha_prior, invalid_fallbacks, kl_rejects) ||
            !OMPBStage02Finite(disagreement) || !OMPBStage02Finite(kl) ||
            !OMPBStage02Finite(alpha_prior))
           {
            PrintFormat("OMPB_STAGE02_CALIBRATION_FAIL reason=target_diagnostics position=%d", position);
            ReturnFalse;
           }
         target_updates++;
         if(OMPBStage02SourcePeriod > 0 && target_updates % OMPBStage02SourcePeriod == 0)
           {
            const int source_position = OMPBStage02ReferenceFirst - int(source_updates % source_rows);
            uint reference_before, current_before, invalid_before, kl_rejects_before;
            float disagreement_before, kl_before, alpha_before;
            if(!ReadOMPBDiagnostics(reference_before, current_before, disagreement_before, kl_before,
                                    alpha_before, invalid_before, kl_rejects_before))
              {
               Print("OMPB_STAGE02_CALIBRATION_FAIL reason=anchor_diagnostics_before");
               ReturnFalse;
              }
            source_attempts++;
            if(!OMPBStage02RunBatch(source_position, true))
              {
               source_invalid++;
              }
            else
              {
               uint reference_after, current_after, invalid_after, kl_rejects_after;
               float disagreement_after, kl_after, alpha_after;
               if(!ReadOMPBDiagnostics(reference_after, current_after, disagreement_after, kl_after,
                                       alpha_after, invalid_after, kl_rejects_after) ||
                  current_after != current_before || reference_after != reference_before ||
                  !OMPBStage02Finite(disagreement_after) || !OMPBStage02Finite(kl_after) ||
                  !OMPBStage02Finite(alpha_after))
                 {
                  PrintFormat("OMPB_STAGE02_CALIBRATION_FAIL reason=anchor_contract position=%d", source_position);
                  ReturnFalse;
                 }
               source_updates++;
              }
           }
        }
      if(OMPBStage02Smoke && target_attempts >= OMPBStage02SmokeLimit)
         break;
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(target_updates == 0 || !D2SkillForecast.BuildEpochCodebookDiagnostics() ||
      !D2SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence, latent,
            observation, valid, invalid, entropy, distance,
            inactive, recovered) ||
      valid < double(target_updates) || !OMPBStage02Finite(lmix) ||
      !OMPBStage02Finite(valid) || !OMPBStage02Finite(invalid) ||
      !D2SkillVerifyFrozenForecastExact())
     {
      PrintFormat("OMPB_STAGE02_CALIBRATION_FAIL reason=final_metrics target=%u source=%u",
                  target_updates, source_updates);
      ReturnFalse;
     }
   PrintFormat("OMPB_STAGE02_CALIBRATION_PASS target_attempts=%u target_updates=%u target_invalid=%u " +
               "source_attempts=%u source_updates=%u source_invalid=%u loss=%.9f reference=%u current=%u smoke=%s",
               target_attempts, target_updates, target_invalid, source_attempts, source_updates,
               source_invalid, lmix / valid, ompb.ReferenceCount(), ompb.CurrentCount(),
               (OMPBStage02Smoke ? "true" : "false"));
   return(true);
  }
//+------------------------------------------------------------------+
bool CreateOMPBStage02ReferenceStudy(const datetime reference_start, const datetime reference_end,
                                     const datetime calibration_start, const datetime calibration_end,
                                     const uint anchor_period, const float tau_value,
                                     const float lambda_dis, const float lambda_kl,
                                     const float lambda_alpha, const float alpha_prior,
                                     const float max_kl, const bool smoke, const uint smoke_limit)
  {
   ResetLastError();
   D2SkillReady = false;
   D2SkillForecast = NULL;
   D2SkillCompletedEpochs = 0;
   D2SkillBatches = 0;
   D2SkillInvalidBatches = 0;
   if(!D2SkillLoadForecastTraining())
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=stage01_checkpoint error=%d", GetLastError());
      ReturnFalse;
     }
   CNeuronBaseOCL *rank_tcm = D2SkillMarket.Layer(4);
   CNeuronBaseOCL *ompb_layer = D2SkillMarket.Layer(5);
   CNeuronBaseOCL *forecast_layer = D2SkillMarket.Layer(6);
   if(!D2SkillForecast || !rank_tcm || !ompb_layer || !forecast_layer ||
      rank_tcm.Type() != defNeuronCogDriverRankTCM || ompb_layer.Type() != defNeuronOMPBOCL ||
      forecast_layer.Type() != defNeuronScenarioForecast || !GetOMPB() ||
      !D2SkillValidateShapes() || !D2SkillInitIndicators() ||
      !D2SkillMarket.SetWeightsUpdate(false) || !D2SkillTarget.SetWeightsUpdate(false))
     {
      Print("OMPB_STAGE02_PREFLIGHT_FAIL reason=loaded_contract");
      ReturnFalse;
     }
   if(!OMPBStage02Configure(reference_start, reference_end, calibration_start, calibration_end,
                            anchor_period, tau_value, lambda_dis, lambda_kl, lambda_alpha,
                            alpha_prior, max_kl, smoke, smoke_limit))
      ReturnFalse;
   D2SkillTarget.TrainMode(false);
   D2SkillMarket.TrainMode(true);
   GetOMPB().TrainMode(false);
   D2SkillReady = true;
   if(!EventChartCustom(ChartID(), 1, 0, 0, "OMPB_STAGE02_INIT"))
     {
      PrintFormat("OMPB_STAGE02_PREFLIGHT_FAIL reason=chart_event error=%d", GetLastError());
      D2SkillReady = false;
      ReturnFalse;
     }
   Print("OMPB_STAGE02_CHECKPOINT_PASS mode=load_only production_save=forbidden");
   return(true);
  }
//+------------------------------------------------------------------+
void RunOMPBStage02ReferenceStudy(void)
  {
   if(!D2SkillReady || !OMPBStage02PrepareData() || !OMPBStage02CollectReference())
     {
      ExpertRemove();
      return;
     }
   if(!D2SkillForecast.SetCodebookUpdate(false))
     {
      Print("OMPB_STAGE02_REFERENCE_FAIL reason=codebook_freeze");
      ExpertRemove();
      return;
     }
   if(!D2SkillCaptureFrozenForecastBaseline())
     {
      Print("OMPB_STAGE02_REFERENCE_FAIL reason=frozen_baseline");
      ExpertRemove();
      return;
     }
   double source_loss, target_loss;
   uint source_valid, source_invalid, target_valid, target_invalid;
   const bool source_ok = OMPBStage02Baseline(OMPBStage02SourceEvalFirst,
                          OMPBStage02SourceEvalLast, "source_eval",
                          source_loss, source_valid, source_invalid);
   const bool target_ok = (source_ok &&
                           OMPBStage02Baseline(OMPBStage02TargetEvalFirst,
                                 OMPBStage02TargetEvalLast, "target_eval",
                                 target_loss, target_valid, target_invalid));
   bool calibration_ok = false;
   bool evaluation_ok = false;
   bool acceptance_ok = false;
   bool checkpoint_saved = false;
   double source_inference_loss = 0.0, target_inference_loss = 0.0;
   uint source_inference_valid = 0, source_inference_invalid = 0;
   uint target_inference_valid = 0, target_inference_invalid = 0;
   if(target_ok && D2SkillVerifyFrozenForecastExact() && OMPBStage02CaptureSignatures())
     {
      PrintFormat("OMPB_STAGE02_REFERENCE_READY source_loss=%.9f target_loss=%.9f source_valid=%u target_valid=%u",
                  source_loss, target_loss, source_valid, target_valid);
      calibration_ok = OMPBStage02Calibrate();
     }
   else
      Print("OMPB_STAGE02_REFERENCE_FAIL reason=baseline_frozen_or_signatures");
//--- Evaluation is intentionally distinct from the frozen BYPASS baselines.
//--- It uses posterior-mean OMPB inference and must be complete before any
//--- candidate checkpoint can be staged.
   if(calibration_ok && ConfigureOMPB(OMPB_INFERENCE))
     {
      const bool source_inference_ok =
         OMPBStage02InferenceEvaluation(OMPBStage02SourceEvalFirst, OMPBStage02SourceEvalLast,
                                        "source_inference", source_inference_loss,
                                        source_inference_valid, source_inference_invalid);
      const bool target_inference_ok =
         (source_inference_ok &&
          OMPBStage02InferenceEvaluation(OMPBStage02TargetEvalFirst, OMPBStage02TargetEvalLast,
                                         "target_inference", target_inference_loss,
                                         target_inference_valid, target_inference_invalid));
      evaluation_ok = (target_inference_ok && OMPBStage02VerifySignatures());
      if(evaluation_ok)
         acceptance_ok = OMPBStage02Accept(source_loss, target_loss, source_inference_loss,
                                           target_inference_loss, source_inference_valid,
                                           target_inference_valid);
      else
         Print("OMPB_STAGE02_ACCEPTANCE_FAIL reason=evaluation_or_signatures");
     }
   else
      Print("OMPB_STAGE02_ACCEPTANCE_FAIL reason=calibration");
//--- Switching to deterministic inference and clearing OMPB drops only Current
//--- and transient buffers.  Clear deliberately preserves posterior parameters.
   CNeuronOMPBOCL *ompb = GetOMPB();
   const bool finalized = (ompb && ConfigureOMPB(OMPB_INFERENCE) && ompb.Clear());
   if(!finalized)
      Print("OMPB_STAGE02_CALIBRATION_FAIL reason=finalize");
   else
     {
      ompb.TrainMode(false);
      if(acceptance_ok && !OMPBStage02Smoke)
         checkpoint_saved = OMPBStage02SaveAcceptedCheckpoint();
      else
         if(OMPBStage02Smoke)
            PrintFormat("OMPB_STAGE02_TRANSACTION_SKIPPED reason=smoke accepted=%s",
                        (acceptance_ok ? "true" : "false"));
         else
            Print("OMPB_STAGE02_TRANSACTION_SKIPPED reason=acceptance");
      CNeuronOMPBOCL *final_ompb = GetOMPB();
      if(!final_ompb)
         Print("OMPB_STAGE02_INFERENCE_READY_FAIL reason=ompb_after_checkpoint");
      else
         PrintFormat("OMPB_STAGE02_INFERENCE_READY calibration=%s evaluation=%s accepted=%s saved=%s " +
                     "reference=%u current=%u", (calibration_ok ? "true" : "false"),
                     (evaluation_ok ? "true" : "false"), (acceptance_ok ? "true" : "false"),
                     (checkpoint_saved ? "true" : "false"), final_ompb.ReferenceCount(),
                     final_ompb.CurrentCount());
     }
   D2SkillMarket.TrainMode(false);
   D2SkillTarget.TrainMode(false);
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Creates or initializes D2SkillForecastStudy.                       |
//+------------------------------------------------------------------+
bool CreateD2SkillForecastStudy(void)
  {
   ResetLastError();
   D2SkillCompletedEpochs = 0;
   D2SkillBatches = 0;
   D2SkillInvalidBatches = 0;
//--- Always prefer an existing compatible checkpoint for continued training.
//--- Only a clean initial checkpoint miss falls back to a new randomized graph.
//--- Partial, incompatible and transaction states remain fail-closed.
   bool resumed = D2SkillLoadForecastTraining();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!resumed)
     {
      if(OMPBStage02RecoveryFailed)
        {
         Print("OMPB checkpoint recovery=FAIL before=training_create fail_closed=true");
         ReturnFalse;
        }
      const int load_error = GetLastError();
      D2SkillForecast = NULL;
      PrintFormat("%s init: forecast restore=FAIL error=%d; creating new random model", OMPB_LOG_PREFIX, load_error);
      ResetLastError();
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(!D2SkillCreateNetworks())
        {
         PrintFormat("%s init: forecast create=FAIL error=%d", OMPB_LOG_PREFIX, GetLastError());
         ReturnFalse;
        }
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillValidateShapes())
     {
      PrintFormat("%s init: shapes=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
//--- Stage 01 keeps OMPB a non-trainable identity bridge.  Its posterior,
//--- alpha and both histories remain untouched while ScenarioForecast trains.
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!ompb || !ConfigureOMPB(OMPB_BYPASS))
      ReturnFalse;
   ompb.TrainMode(false);
   if(!D2SkillInitIndicators())
     {
      PrintFormat("%s init: indicators=FAIL error=%d", OMPB_LOG_PREFIX, GetLastError());
      ReturnFalse;
     }
   PrintFormat("%s init: forecast=%s completed_epochs=%u batches=%I64u invalid=%I64u", OMPB_LOG_PREFIX,
               (resumed ? "RESUMED" : "NEW"), D2SkillCompletedEpochs, D2SkillBatches, D2SkillInvalidBatches);
   D2SkillReady = true;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("%s init: chart event=FAIL error=%d", OMPB_LOG_PREFIX, GetLastError());
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements ReleaseD2SkillForecastStudy.                            |
//+------------------------------------------------------------------+
void ReleaseD2SkillForecastStudy(const int reason)
  {
//--- Complete epochs save transactionally in TrainD2SkillForecast(). Never
//--- overwrite a valid checkpoint with a fresh, partial or failed run here.
   D2SkillForecast = NULL;
   D2SkillReady = false;
  }
//+------------------------------------------------------------------+
//| Implements D2SkillPrepareLatentTarget.                             |
//+------------------------------------------------------------------+
bool D2SkillPrepareLatentTarget(CBufferFloat *market_latent)
  {
   if(!market_latent || market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0 ||
      D2SkillFuture.Total() != (NForecast * BarDescr) || D2SkillFuture.GetIndex() < 0 ||
      D2SkillLatentTarget.Total() != (BarDescr * NForecast * EmbeddingSize) || D2SkillLatentTarget.GetIndex() < 0 ||
      D2SkillLatentDelta.Total() != (BarDescr * NForecast * EmbeddingSize) || D2SkillLatentDelta.GetIndex() < 0)
      ReturnFalse;
//--- D2SkillFuture is decoder-native [H,B].  The existing transpose changes
//--- only this device layout to Target-native [B,H]; Market keeps its original
//--- [BarDescr,HistoryBars] representation and is never transposed.
   if(!D2SkillFutureView.Bind(GetPointer(D2SkillFuture)))
      ReturnFalse;
   const bool transposed = D2SkillFutureTranspose.FeedForward(D2SkillFutureView.AsObject());
   D2SkillFutureView.Unbind();
   if(!transposed)
      ReturnFalse;
   CNeuronBaseOCL *target_input = D2SkillTarget.Layer(0);
   CNeuronBaseOCL *future_latent = D2SkillTarget.Layer(-1);
   if(!target_input || target_input.getOutput().Total() != (BarDescr * NForecast) ||
      target_input.getOutputIndex() < 0 ||
      !D2SkillDevice.Copy(D2SkillFutureTranspose.getOutput(), target_input.getOutput(),
                          BarDescr * NForecast) ||
      !D2SkillTarget.feedForward(GetPointer(D2SkillTarget), 0, (CBufferFloat*)NULL))
      ReturnFalse;
   if(!future_latent || future_latent.getOutput().Total() != (BarDescr * NForecast * EmbeddingSize) ||
      future_latent.getOutputIndex() < 0)
      ReturnFalse;
//--- Forecast loss compares full future Z against full generator Z. Codebook
//--- EMA alone receives the detached delta future-current.
   if(!D2SkillDevice.Copy(future_latent.getOutput(), GetPointer(D2SkillLatentTarget),
                          BarDescr * NForecast * EmbeddingSize) ||
      !D2SkillDevice.Subtract(GetPointer(D2SkillLatentZero), market_latent,
                              GetPointer(D2SkillLatentNegativeMarket), EmbeddingSize))
      ReturnFalse;
   if(!D2SkillDevice.BroadcastSum(GetPointer(D2SkillLatentNegativeMarket),
                                  future_latent.getOutput(), GetPointer(D2SkillLatentDelta),
                                  EmbeddingSize, BarDescr))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillProbeNet.                                        |
//+------------------------------------------------------------------+
bool D2SkillProbeNet(CNet &net, CBufferFloat *probe_state, double &latent[], const bool restore_training)
  {
   if(!probe_state || ArrayResize(latent, (BarDescr * EmbeddingSize)) != (BarDescr * EmbeddingSize))
      ReturnFalse;
//--- Measure both epoch boundaries in inference mode from an empty recurrent
//--- state.  Clear again afterwards so the diagnostic forward cannot seed the
//--- first training batch or inherit the final training batch state.
   if(!net.TrainMode(false))
      ReturnFalse;
   bool result = net.Clear();
   if(result)
      result = net.feedForward(probe_state, 1, false, (CBufferFloat*)NULL);
//--- Get the live OpenCL output owned by RankTCM.  GetLayerOutput() copies
//--- values into a new host CBufferFloat and therefore cannot be BufferRead().
   CNeuronBaseOCL *latent_layer = net.Layer(4);
   CBufferFloat *output = (latent_layer ? latent_layer.getOutput() : NULL);
   CNeuronBaseOCL *bridge_layer = net.Layer(5);
   CNeuronOMPBOCL *bridge = (bridge_layer && bridge_layer.Type() == defNeuronOMPBOCL ?
                             (CNeuronOMPBOCL *)bridge_layer : NULL);
   CBufferFloat *bridge_output = (bridge ? bridge.getOutput() : NULL);
   if(result)
      result = (bridge && bridge.Mode() == OMPB_BYPASS && output != NULL && bridge_output != NULL &&
                output.GetIndex() >= 0 && bridge_output.GetIndex() >= 0 && output.BufferRead() &&
                bridge_output.BufferRead() && output.Total() == (BarDescr * EmbeddingSize) &&
                bridge_output.Total() == (BarDescr * EmbeddingSize));
   for(uint d = 0; result && d < (BarDescr * EmbeddingSize); d++)
     {
      latent[d] = double(output[d]);
      if(!MathIsValidNumber(latent[d]) || output[d] != bridge_output[d])
         result = false;
     }
   const bool cleared = net.Clear();
   const bool restored = net.TrainMode(restore_training);
   return(result && cleared && restored);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillProbeDrift.                                      |
//+------------------------------------------------------------------+
double D2SkillProbeDrift(const double &before[], const double &after[])
  {
   if(ArraySize(before) != (BarDescr * EmbeddingSize) || ArraySize(after) != (BarDescr * EmbeddingSize))
      return(DBL_MAX);
   double square_sum = 0;
   for(uint d = 0; d < (BarDescr * EmbeddingSize); d++)
     {
      const double delta = after[d] - before[d];
      square_sum += delta * delta;
     }
   return(MathSqrt(square_sum / double((BarDescr * EmbeddingSize))));
  }
//+------------------------------------------------------------------+
//| Implements FormatScenarioCounts.                                 |
//+------------------------------------------------------------------+
string FormatScenarioCounts(const uint &counts[])
  {
   string result = "[";
   for(int i = 0; i < ArraySize(counts); i++)
      result += (i > 0 ? "," : "") + IntegerToString((int)counts[i]);
   return(result + "]");
  }
//+------------------------------------------------------------------+
//| Smoke-only responsibility audit; all tensor reads are host-side  |
//+------------------------------------------------------------------+
bool D2SkillCollectRecoverySmoke(ulong &hits[], double &responsibility_sum[], double &scale_sum[],
                                 double &absolute_sum[], double &absolute_over_scale_sum[])
  {
   if(!D2SkillForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios)
      ReturnFalse;
   CBufferFloat *responsibilities = D2SkillForecast.GetResponsibilities();
   CBufferFloat *confidence = D2SkillForecast.GetConfidence();
   CBufferFloat *target = D2SkillForecast.GetTarget();
   CBufferFloat *forecast = D2SkillForecast.getOutput();
   const int target_total = BarDescr * NForecast * EmbeddingSize;
   const int confidence_total = NScenarios * BarDescr * NForecast;
   if(!responsibilities || !confidence || !target || !forecast || responsibilities.Total() != NScenarios ||
      confidence.Total() != confidence_total || target.Total() != target_total ||
      forecast.Total() != NScenarios * target_total || !responsibilities.BufferRead() || !confidence.BufferRead() ||
      !target.BufferRead() || !forecast.BufferRead())
      ReturnFalse;
   for(uint scenario = 0; scenario < NScenarios; scenario++)
     {
      const double responsibility = responsibilities[scenario];
      if(!MathIsValidNumber(responsibility) || responsibility < 0.0)
         ReturnFalse;
      if(responsibility > 0.0)
         hits[scenario]++;
      responsibility_sum[scenario] += responsibility;
      if(responsibility <= 0.0)
         continue;
      for(int coordinate = 0; coordinate < target_total; coordinate++)
        {
         const int token = coordinate / EmbeddingSize;
         const double scale = MathMax(0.001, MathMin(10.0, confidence[int(scenario) * BarDescr * NForecast + token]));
         const double absolute_error = MathAbs(target[coordinate] -
                                               forecast[int(scenario) * target_total + coordinate]);
         if(!MathIsValidNumber(scale) || !MathIsValidNumber(absolute_error))
            ReturnFalse;
         scale_sum[scenario] += responsibility * scale;
         absolute_sum[scenario] += responsibility * absolute_error;
         absolute_over_scale_sum[scenario] += responsibility * absolute_error / scale;
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillLogRecoverySmoke(const ulong &hits[], const double &responsibility_sum[],
                             const double &scale_sum[], const double &absolute_sum[],
                             const double &absolute_over_scale_sum[],
                             const uint &recovered[], const uint &age_zero[])
  {
   if(!D2SkillForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios ||
      ArraySize(recovered) != NScenarios || ArraySize(age_zero) != NScenarios)
      ReturnFalse;
   CScenarioCodebook *codebook = D2SkillForecast.GetCodebook();
   if(!codebook)
      ReturnFalse;
   CBufferFloat *ages = codebook.GetInactivityAge();
   CBufferFloat *counts = codebook.GetEMACounts();
   CBufferFloat *usage = codebook.GetUsage();
   CBufferFloat *inactive = codebook.GetInactive();
   if(!ages || !counts || !usage || !inactive || ages.Total() != NScenarios || counts.Total() != NScenarios ||
      usage.Total() != NScenarios || inactive.Total() != NScenarios || !ages.BufferRead() || !counts.BufferRead() ||
      !usage.BufferRead() || !inactive.BufferRead())
      ReturnFalse;
   for(uint scenario = 0; scenario < NScenarios; scenario++)
     {
      if(!MathIsValidNumber(ages[scenario]) || !MathIsValidNumber(counts[scenario]) ||
         !MathIsValidNumber(usage[scenario]) || !MathIsValidNumber(inactive[scenario]))
         ReturnFalse;
      const double denominator = responsibility_sum[scenario] * double(BarDescr * NForecast * EmbeddingSize);
      const double mean_scale = (denominator > 0.0 ? scale_sum[scenario] / denominator : 0.0);
      const double mean_absolute = (denominator > 0.0 ? absolute_sum[scenario] / denominator : 0.0);
      const double mean_absolute_over_scale = (denominator > 0.0 ?
                                              absolute_over_scale_sum[scenario] / denominator : 0.0);
      PrintFormat("%s smoke state=%d hits=%I64u age_zero=%u responsibility_sum=%.8f mean_U=%.8f " +
                  "mean_abs_error=%.8f mean_abs_over_U=%.8f final_age=%.0f ema_count=%.8f " +
                  "usage=%.8f inactive=%d recovered=%u",
                  OMPB_LOG_PREFIX, scenario, hits[scenario], age_zero[scenario],
                  responsibility_sum[scenario], mean_scale, mean_absolute,
                  mean_absolute_over_scale, ages[scenario], counts[scenario], usage[scenario],
                  int(MathRound(inactive[scenario])), recovered[scenario]);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillTrainBatch.                                      |
//+------------------------------------------------------------------+
bool D2SkillTrainBatch(const int position)
  {
//--- Build exactly the same X<=t state as CreateBuffers(position,...,future),
//--- but do not touch any future sample until Market and Scenario are complete.
   if(!CreateBuffers(position + NForecast, GetPointer(D2SkillState), GetPointer(D2SkillTime), NULL))
      ReturnFalse;
   if(!D2SkillMarket.feedForward(GetPointer(D2SkillState), 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
//--- Target supervision remains bound to raw RankTCM z_t.  OMPB sits between
//--- it and ScenarioForecast, so relative indexing would select the bridge.
   CNeuronBaseOCL *market_layer = GetRankTCM();
   CBufferFloat *market_latent = (market_layer ? market_layer.getOutput() : NULL);
   if(!market_latent ||
      market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0)
      ReturnFalse;
//--- Only now is any future window presented to the detached Target Encoder.
   if(!CreateBuffers(position, GetPointer(D2SkillState), GetPointer(D2SkillTime), GetPointer(D2SkillFuture)))
      ReturnFalse;
   if(!D2SkillPrepareLatentTarget(market_latent))
      ReturnFalse;
   const ulong responsibility_started = GetMicrosecondCount();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillForecast.BuildResponsibilities(GetPointer(D2SkillLatentTarget), GetPointer(D2SkillLatentDelta)))
     {
      //--- Commit only a completed device validation.  Infrastructure failures leave
      //--- the transient control-valid flag false, so stale batch_control is ignored.
      if(D2SkillForecast.LastBatchControlValid() && !D2SkillForecast.CommitBatchDiagnostics())
         PrintFormat("%s -> %d invalid diagnostic commit failed", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   const ulong responsibility_elapsed = GetMicrosecondCount() - responsibility_started;
   if(!D2SkillForecast.CommitBatchDiagnostics())
      ReturnFalse;
//--- The unified CNet updates ScenarioForecast and all preceding Market layers.
   if(!D2SkillMarket.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, -1, true))
      ReturnFalse;
   D2SkillResponsibilityMicroseconds += responsibility_elapsed;
   return(true);
  }
//+------------------------------------------------------------------+
//| Current-epoch progress uses the existing 12-float device summ... |
//+------------------------------------------------------------------+
void D2SkillShowForecastProgress(const double percent, const ulong failed_batches)
  {
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(D2SkillForecast && D2SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence,
         latent, observation, valid, invalid, entropy, distance, inactive, recovered) && valid > 0.0)
     {
      Comment(StringFormat("%s Forecast %6.2f%% L_mix %.8f latent %.8f invalid(current epoch) %I64u",
                           OMPB_LOG_PREFIX, percent, lmix / valid, latent / valid, failed_batches));
      return;
     }
   Comment(StringFormat("%s Forecast %6.2f%% L_mix n/a latent n/a invalid(current epoch) %I64u",
                        OMPB_LOG_PREFIX, percent, failed_batches));
  }
//+------------------------------------------------------------------+
//| Implements TrainD2SkillForecast.                                   |
//+------------------------------------------------------------------+
void TrainD2SkillForecast(void)
  {
   if(!D2SkillReady)
      return;
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     { PrintFormat("%s -> %d", __FUNCTION__, __LINE__); return; }
   int wait = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars && CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars && MACD.BarsCalculated() >= bars);
      Sleep(100);
      wait++;
     }
   while(!calculated && wait < 100);
   if(!calculated)
     { PrintFormat("%s -> %d data unavailable", __FUNCTION__, __LINE__); return; }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
     { PrintFormat("%s -> %d data unavailable", __FUNCTION__, __LINE__); return; }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     { PrintFormat("%s -> %d insufficient history", __FUNCTION__, __LINE__); return; }
//--- With a forecast buffer CreateBuffers(position,...) starts Market state at
//--- position+H.  Reproduce that exact X<=t mapping with NULL, which also keeps
//--- every future target out of the diagnostic input.
   const int probe_position = MathMax(end - 1, 0);
   if(!CreateBuffers(probe_position + NForecast, GetPointer(D2SkillProbeState),
// Gets Pointer.
                     GetPointer(D2SkillProbeTime), NULL))
     { PrintFormat("%s -> %d fixed probe unavailable", __FUNCTION__, __LINE__); return; }
   uint ticks = GetTickCount();
   bool stop = false;
   const uint passes = (D2SkillRecoverySmoke ? 1 : uint(MathMax(Epochs, 0)));
   const ulong smoke_limit = (D2SkillRecoverySmoke ? ulong(MathMax(1, int(D2SkillRecoverySmokeBatches))) : 0);
   for(uint pass = 0; pass < passes && !IsStopped() && !stop; pass++)
     {
      const uint epoch = D2SkillCompletedEpochs;
      ulong epoch_failures = 0;
      ulong smoke_batches = 0;
      ulong smoke_hits[];
      double smoke_responsibility_sum[];
      double smoke_scale_sum[];
      double smoke_absolute_sum[];
      double smoke_absolute_over_scale_sum[];
      if(D2SkillRecoverySmoke &&
         (ArrayResize(smoke_hits, NScenarios) != NScenarios ||
          ArrayResize(smoke_responsibility_sum, NScenarios) != NScenarios ||
          ArrayResize(smoke_scale_sum, NScenarios) != NScenarios ||
          ArrayResize(smoke_absolute_sum, NScenarios) != NScenarios ||
          // All temporary smoke buffers must resize to NScenarios.
          ArrayResize(smoke_absolute_over_scale_sum, NScenarios) != NScenarios))
        { PrintFormat("%s -> %d smoke counter allocation failed", __FUNCTION__, __LINE__); break; }
      ArrayInitialize(smoke_hits, 0);
      ArrayInitialize(smoke_responsibility_sum, 0.0);
      ArrayInitialize(smoke_scale_sum, 0.0);
      ArrayInitialize(smoke_absolute_sum, 0.0);
      ArrayInitialize(smoke_absolute_over_scale_sum, 0.0);
      if(!D2SkillMarket.Clear() || !D2SkillTarget.Clear())
        { PrintFormat("%s -> %d clear failed", __FUNCTION__, __LINE__); break; }
      D2SkillTarget.TrainMode(false);
      double latent_before[], latent_after[];
      if(!D2SkillProbeNet(D2SkillMarket, GetPointer(D2SkillProbeState), latent_before, true) ||
         !ConfigureOMPB(OMPB_BYPASS))
        { PrintFormat("%s -> %d fixed probe forward failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!D2SkillForecast.ResetEpochDiagnostics())
        { PrintFormat("%s -> %d diagnostic reset failed", __FUNCTION__, __LINE__); stop = true; break; }
      uint ompb_reference_count_before, ompb_current_count_before;
      uint ompb_invalid_fallbacks_before, ompb_kl_rejects_before;
      float ompb_disagreement_before, ompb_kl_before, ompb_alpha_prior_before;
      if(!ReadOMPBDiagnostics(ompb_reference_count_before, ompb_current_count_before,
                              ompb_disagreement_before, ompb_kl_before,
                              ompb_alpha_prior_before, ompb_invalid_fallbacks_before,
                              ompb_kl_rejects_before))
        {
         PrintFormat("OMPB_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=epoch-%u reason=read_before", epoch + 1);
         stop = true;
         break;
        }
      D2SkillResponsibilityMicroseconds = 0;
      for(int position = start - HistoryBars - NForecast - 1; position >= end && !IsStopped() && !stop; position--)
        {
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(!D2SkillTrainBatch(position))
           {
            epoch_failures++;
            PrintFormat("%s invalid batch epoch=%d position=%d line=%d", OMPB_LOG_PREFIX, epoch, position, __LINE__);
            //+------------------------------------------------------------------+
            //| Function if.                                                     |
            //+------------------------------------------------------------------+
            if(GetTickCount() - ticks > 500)
              {
               const double percent = (double(pass) + 1.0 -
                                       double(position - end) /
                                       MathMax(start - end - HistoryBars - NForecast, 1)) *
                                      100.0 / Epochs;
               D2SkillShowForecastProgress(percent, epoch_failures);
               ticks = GetTickCount();
              }
            continue;
           }
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(D2SkillRecoverySmoke)
           {
            if(!D2SkillCollectRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
                                            smoke_absolute_sum, smoke_absolute_over_scale_sum))
              { PrintFormat("%s -> %d smoke responsibility read failed", __FUNCTION__, __LINE__); stop = true; break; }
            smoke_batches++;
            if(smoke_batches >= smoke_limit)
               break;
           }
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(GetTickCount() - ticks > 500)
           {
            const double percent = (double(pass) + 1.0 -
                                    double(position - end) /
                                    MathMax(start - end - HistoryBars - NForecast, 1)) *
                                   100.0 / Epochs;
            D2SkillShowForecastProgress(percent, epoch_failures);
            ticks = GetTickCount();
           }
        }
      double lmix, lrouter, ltrajectory, lconfidence, latent, observation;
      double valid_value, invalid_value, entropy, distance, inactive_value, recovered_value;
      if(!D2SkillForecast.BuildEpochCodebookDiagnostics() ||
         !D2SkillForecast.ReadEpochDiagnostics(lmix, lrouter, ltrajectory, lconfidence, latent, observation,
               valid_value, invalid_value, entropy, distance, inactive_value, recovered_value))
        { PrintFormat("%s -> %d diagnostic read failed", __FUNCTION__, __LINE__); stop = true; break; }
      const ulong valid = (ulong)MathRound(valid_value);
      const ulong invalid = (ulong)MathRound(invalid_value);
      const uint inactive = (uint)MathRound(inactive_value);
      const uint recovered = (uint)MathRound(recovered_value);
      uint recovered_by_scenario[];
      uint age_zero_by_scenario[];
      if(!D2SkillForecast.ReadEpochRecoveryEvents(recovered_by_scenario))
        { PrintFormat("%s -> %d recovery-event read failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!D2SkillForecast.ReadEpochAgeZeroEvents(age_zero_by_scenario))
        { PrintFormat("%s -> %d age-zero-event read failed", __FUNCTION__, __LINE__); stop = true; break; }
      ulong recovery_sum = 0;
      for(int i = 0; i < ArraySize(recovered_by_scenario); i++)
         recovery_sum += recovered_by_scenario[i];
      if(recovery_sum != (ulong)recovered)
        {
         PrintFormat("%s -> %d recovery-event mismatch total=%I64u diagnostics=%u",
                     __FUNCTION__, __LINE__, recovery_sum, recovered);
         stop = true;
         break;
        }
      if(D2SkillRecoverySmoke && valid != smoke_batches)
        {
         PrintFormat("%s -> %d smoke valid mismatch valid=%I64u collected=%I64u",
                     __FUNCTION__, __LINE__, valid, smoke_batches);
         stop = true;
         break;
        }
      D2SkillBatches += valid;
      D2SkillInvalidBatches += invalid;
      if(valid == 0)
        { stop = true; break; }
      if(!D2SkillCheckOMPBBypassInvariant(ompb_reference_count_before, ompb_current_count_before,
                                          ompb_disagreement_before, ompb_kl_before, ompb_alpha_prior_before,
                                          ompb_invalid_fallbacks_before, ompb_kl_rejects_before,
                                          StringFormat("epoch-%u", epoch + 1)))
        { stop = true; break; }
      //--- Target is an independently initialized inference-only encoder.  It is
      //--- never copied from Market and is absent from every backward/update path.
      if(!D2SkillProbeNet(D2SkillMarket, GetPointer(D2SkillProbeState), latent_after, true) ||
         !ConfigureOMPB(OMPB_BYPASS))
        { PrintFormat("%s -> %d Market epoch probe failed", __FUNCTION__, __LINE__); stop = true; break; }
      const double latent_drift = D2SkillProbeDrift(latent_before, latent_after);
      PrintFormat("%s epoch=%d batches=%I64u L_mix=%.8f L_router=%.8f L_trajectory=%.8f " +
                  "L_confidence=%.8f NLL_confidence=%.8f usage_entropy=%.8f " +
                  "pairwise_codebook=%.8f invalid=%I64u latent_drift=%.8f inactive=%u " +
                  "recovered=%u recovered_by_scenario=%s responsibility_ms=%.3f",
                  OMPB_LOG_PREFIX, epoch + 1, valid, lmix / valid, lrouter / valid,
                  ltrajectory / valid, latent / valid, lconfidence / valid, entropy,
                  distance, D2SkillInvalidBatches, latent_drift, inactive, recovered,
                  FormatScenarioCounts(recovered_by_scenario),
                  double(D2SkillResponsibilityMicroseconds) / (1000.0 * double(valid)));
      if(D2SkillRecoverySmoke && !D2SkillLogRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
            smoke_absolute_sum, smoke_absolute_over_scale_sum,
            recovered_by_scenario, age_zero_by_scenario))
        { PrintFormat("%s -> %d smoke state log failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!D2SkillRecoverySmoke && !D2SkillSaveCheckpoint(epoch + 1))
        { PrintFormat("%s -> %d checkpoint failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(D2SkillRecoverySmoke)
         PrintFormat("%s smoke complete valid=%I64u recovery_age=%u checkpoint=SKIPPED", OMPB_LOG_PREFIX, valid,
                     D2SkillForecast.RecoveryAge());
      D2SkillCompletedEpochs = epoch + 1;
     }
   Comment("");
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!stop)
     {
      //--- Successful forecast stage is sealed for the later Actor-Critic stage.
      D2SkillMarket.TrainMode(false);
      D2SkillTarget.TrainMode(false);
      D2SkillLastSignature = D2SkillForecastSignature();
      PrintFormat("%s forecast inference-only signature=%I64u batches=%I64u invalid=%I64u", OMPB_LOG_PREFIX,
                  D2SkillLastSignature, D2SkillBatches, D2SkillInvalidBatches);
     }
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Actor-Critic inference and composition helpers Implements ORI... |
//+------------------------------------------------------------------+
string D2SkillManifestValue(const string file_name, const string key)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return("");
   const string prefix = key + "=";
   string value = "";
//+------------------------------------------------------------------+
//| Function while.                                                  |
//+------------------------------------------------------------------+
   while(!FileIsEnding(handle))
     {
      const string line = FileReadString(handle);
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(StringFind(line, prefix) == 0)
        {
         value = StringSubstr(line, StringLen(prefix));
         break;
        }
     }
   FileClose(handle);
   return(value);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillValidateForecastManifestHeader. Static checkp... |
//+------------------------------------------------------------------+
bool D2SkillValidateForecastManifestHeader(void)
  {
#define D2Skill_MANIFEST_HEADER_EQ(KEY,VALUE) if(D2SkillManifestValue(D2SkillActiveManifestFile,KEY)!=(VALUE)) ReturnFalse
   D2Skill_MANIFEST_HEADER_EQ("format", "OMPB_FORECAST");
   D2Skill_MANIFEST_HEADER_EQ("version", IntegerToString(D2Skill_FORMAT_VERSION));
   D2Skill_MANIFEST_HEADER_EQ("forecast_type", IntegerToString(defNeuronScenarioForecast));
   D2Skill_MANIFEST_HEADER_EQ("variables", IntegerToString(BarDescr));
   D2Skill_MANIFEST_HEADER_EQ("scenarios", IntegerToString(NScenarios));
   D2Skill_MANIFEST_HEADER_EQ("top_k", IntegerToString(TopK));
   D2Skill_MANIFEST_HEADER_EQ("horizon", IntegerToString(NForecast));
   D2Skill_MANIFEST_HEADER_EQ("latent", IntegerToString(EmbeddingSize));
   D2Skill_MANIFEST_HEADER_EQ("z_layout", "K,V,H,D");
   D2Skill_MANIFEST_HEADER_EQ("u_layout", "K,V,H");
   D2Skill_MANIFEST_HEADER_EQ("pi_layout", "K");
   D2Skill_MANIFEST_HEADER_EQ("codebook_layout", "K,V,H,D");
   D2Skill_MANIFEST_HEADER_EQ("variable_order", "BarDescr_feature_series_0_to_8");
   D2Skill_MANIFEST_HEADER_EQ("normalization", "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
#undef D2Skill_MANIFEST_HEADER_EQ
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillValidateForecastManifest.                        |
//+------------------------------------------------------------------+
bool D2SkillValidateForecastManifest(const ulong signature, const bool training = false)
  {
   if(signature == 0)
      ReturnFalse;
   if(!D2SkillValidateForecastManifestHeader())
      ReturnFalse;
#define D2Skill_MANIFEST_EQ(KEY,VALUE) if(D2SkillManifestValue(D2SkillActiveManifestFile,KEY)!=(VALUE)) ReturnFalse
   D2Skill_MANIFEST_EQ("contract_signature", StringFormat("%I64u", D2SkillForecast.ContractSignature()));
   D2Skill_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", signature));
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(training)
     {
      const ulong target_hash = D2SkillHashFile(ulong(1469598103934665603), D2SkillActiveTargetFile);
      const ulong training_signature = D2SkillForecastTrainingSignature(signature);
      D2Skill_MANIFEST_EQ("target_hash", StringFormat("%I64u", target_hash));
      D2Skill_MANIFEST_EQ("training_signature", StringFormat("%I64u", training_signature));
      if(target_hash == 0 || training_signature == 0 ||
         D2SkillManifestValue(D2SkillActiveManifestFile, "completed_epochs") == "" ||
         D2SkillManifestValue(D2SkillActiveManifestFile, "training_batches") == "" ||
         D2SkillManifestValue(D2SkillActiveManifestFile, "invalid_batches") == "")
         ReturnFalse;
     }
#undef D2Skill_MANIFEST_EQ
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillCaptureFrozenBuffer.                             |
//+------------------------------------------------------------------+
bool D2SkillCaptureFrozenBuffer(CBufferFloat *source, CBufferFloat &baseline)
  {
   return(source && source.BufferRead() && source.Total() > 0 && baseline.AssignArray(source));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillFrozenBufferEqual.                               |
//+------------------------------------------------------------------+
bool D2SkillFrozenBufferEqual(CBufferFloat *source, CBufferFloat &baseline)
  {
   if(!source || !source.BufferRead() || source.Total() != baseline.Total())
      ReturnFalse;
   for(int i = 0; i < source.Total(); i++)
      if(source[i] != baseline[i])
         ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillCaptureFrozenForecastBaseline.                   |
//+------------------------------------------------------------------+
bool D2SkillCaptureFrozenForecastBaseline(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : D2SkillForecast);
   D2SkillFrozenBaselineReady = false;
   if(!current || !current.GetCodebook())
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   if(trainable <= 0 || trainable != int(current.TrainableWeights()) ||
      !D2SkillCaptureFrozenBuffer(generator, D2SkillFrozenGeneratorWeights) ||
      !D2SkillCaptureFrozenBuffer(router, D2SkillFrozenRouterWeights) ||
      !D2SkillCaptureFrozenBuffer(confidence, D2SkillFrozenConfidenceWeights) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetPrototypes(), D2SkillFrozenPrototypes) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetEMASums(), D2SkillFrozenEMASums) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetEMACounts(), D2SkillFrozenEMACounts) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetUsage(), D2SkillFrozenUsage) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetInactive(), D2SkillFrozenInactive) ||
      !D2SkillCaptureFrozenBuffer(current.GetCodebook().GetInactivityAge(), D2SkillFrozenInactivityAge))
      ReturnFalse;
   D2SkillFrozenBaselineReady = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillVerifyFrozenWeightsExact.                        |
//+------------------------------------------------------------------+
bool D2SkillVerifyFrozenWeightsExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : D2SkillForecast);
   if(!D2SkillFrozenBaselineReady || !current)
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   return (trainable > 0 && trainable == int(current.TrainableWeights()) &&
           D2SkillFrozenBufferEqual(generator, D2SkillFrozenGeneratorWeights) &&
           D2SkillFrozenBufferEqual(router, D2SkillFrozenRouterWeights) &&
           D2SkillFrozenBufferEqual(confidence, D2SkillFrozenConfidenceWeights));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillVerifyFrozenCodebookExact.                       |
//+------------------------------------------------------------------+
bool D2SkillVerifyFrozenCodebookExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : D2SkillForecast);
   if(!D2SkillFrozenBaselineReady || !current || !current.GetCodebook())
      ReturnFalse;
   return (D2SkillFrozenBufferEqual(current.GetCodebook().GetPrototypes(), D2SkillFrozenPrototypes) &&
           D2SkillFrozenBufferEqual(current.GetCodebook().GetEMASums(), D2SkillFrozenEMASums) &&
           D2SkillFrozenBufferEqual(current.GetCodebook().GetEMACounts(), D2SkillFrozenEMACounts) &&
           D2SkillFrozenBufferEqual(current.GetCodebook().GetUsage(), D2SkillFrozenUsage) &&
           D2SkillFrozenBufferEqual(current.GetCodebook().GetInactive(), D2SkillFrozenInactive) &&
           D2SkillFrozenBufferEqual(current.GetCodebook().GetInactivityAge(), D2SkillFrozenInactivityAge));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillVerifyFrozenForecastExact.                       |
//+------------------------------------------------------------------+
bool D2SkillVerifyFrozenForecastExact(CNeuronScenarioForecast *forecast = NULL)
  {
   return(D2SkillVerifyFrozenWeightsExact(forecast) && D2SkillVerifyFrozenCodebookExact(forecast));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillLoadForecastInference.                           |
//+------------------------------------------------------------------+
bool D2SkillLoadForecastInference(void)
  {
   D2SkillFrozenBaselineReady = false;
   D2SkillForecast = NULL;
   if(!OMPBStage02ResolveActiveCheckpoint())
     {
      Print("OMPB checkpoint selector=FAIL before=inference_load");
      ReturnFalse;
     }
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillMarket.Load(D2SkillActiveMarketFile, error, undefine, forecast, studied, true))
     {
      PrintFormat("%s inference: model load=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   D2SkillForecast = (CNeuronScenarioForecast*)D2SkillMarket.Layer(-1);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillForecast || D2SkillForecast.Type() != defNeuronScenarioForecast)
     {
      PrintFormat("%s inference: forecast layer=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(D2SkillForecast.GetTopK() != TopK)
     {
      PrintFormat("%s inference: TopK expected=%d actual=%d", OMPB_LOG_PREFIX, TopK, D2SkillForecast.GetTopK());
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| only the Market/Forecast path. Target and its future-window b... |
//+------------------------------------------------------------------+
   if(!D2SkillValidateShapes(false))
     {
      PrintFormat("%s inference: shape audit=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   D2SkillMarket.TrainMode(false);
   const ulong signature = D2SkillForecastSignature();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillValidateForecastManifest(signature))
     {
      PrintFormat("%s inference: manifest=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!D2SkillCaptureFrozenForecastBaseline())
     {
      PrintFormat("%s inference: frozen baseline=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   D2SkillLastSignature = signature;
   return(true);
  }
//+------------------------------------------------------------------+
//| Full forecast-training restore. Target is required here becau... |
//+------------------------------------------------------------------+
bool D2SkillLoadForecastTraining(void)
  {
   if(!OMPBStage02ResolveActiveCheckpoint())
     {
      Print("OMPB checkpoint selector=FAIL before=training_load");
      ReturnFalse;
     }
   if(!FileIsExist(D2SkillActiveManifestFile, FILE_COMMON) ||
      !FileIsExist(D2SkillActiveMarketFile, FILE_COMMON) ||
      !FileIsExist(D2SkillActiveTargetFile, FILE_COMMON))
      ReturnFalse;
//+------------------------------------------------------------------+
//| Avoid a partial CNet::Load before the fallback random graph i... |
//+------------------------------------------------------------------+
   if(!D2SkillValidateForecastManifestHeader())
     {
      PrintFormat("%s restore: manifest static contract=FAIL", OMPB_LOG_PREFIX);
      ReturnFalse;
     }
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   if(!D2SkillMarket.Load(D2SkillActiveMarketFile, error, undefine, forecast, studied, true) ||
      !D2SkillTarget.Load(D2SkillActiveTargetFile, error, undefine, forecast, studied, true))
      ReturnFalse;
   if(!D2SkillTarget.SetOpenCLChecked(D2SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   D2SkillForecast = (CNeuronScenarioForecast*)D2SkillMarket.Layer(-1);
   if(!D2SkillForecast || D2SkillForecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//--- These are transient training tensors and are intentionally absent from
//--- *.nnw. Recreate them before the common shape audit.
   if(!D2SkillInitTrainingBuffers() || !D2SkillValidateShapes())
      ReturnFalse;
   const ulong signature = D2SkillForecastSignature();
   if(!D2SkillValidateForecastManifest(signature, true))
      ReturnFalse;
   const string completed = D2SkillManifestValue(D2SkillActiveManifestFile, "completed_epochs");
   const string batches = D2SkillManifestValue(D2SkillActiveManifestFile, "training_batches");
   const string invalid = D2SkillManifestValue(D2SkillActiveManifestFile, "invalid_batches");
   if(completed == "" || batches == "" || invalid == "")
      ReturnFalse;
   D2SkillCompletedEpochs = (uint)StringToInteger(completed);
   D2SkillBatches = (ulong)StringToInteger(batches);
   D2SkillInvalidBatches = (ulong)StringToInteger(invalid);
   D2SkillTarget.TrainMode(false);
   D2SkillMarket.TrainMode(true);
   D2SkillLastSignature = signature;
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures the loaded Stage 02 graph for frozen Stage 03 use.    |
//+------------------------------------------------------------------+
bool D2SkillConfigureProductionOMPBCheckpoint(void)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   CNeuronScenarioForecast *forecast = D2SkillForecast;
   if(!ompb || !forecast || !SetOMPBMode(OMPB_INFERENCE) ||
      !D2SkillMarket.SetWeightsUpdate(false) ||
      !forecast.SetCodebookUpdate(false))
      ReturnFalse;
   D2SkillMarket.TrainMode(false);
   ompb.TrainMode(false);
   forecast.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks the strict published Stage 02 contract before Stage 03.   |
//+------------------------------------------------------------------+
bool D2SkillValidateProductionOMPBCheckpoint(void)
  {
   const string checkpoint = "canonical";
   CNeuronBaseOCL *rank_tcm = D2SkillMarket.Layer(4);
   CNeuronOMPBOCL *ompb = GetOMPB();
   CNeuronBaseOCL *forecast_layer = D2SkillMarket.Layer(6);
   CNeuronBaseOCL *market_tail = D2SkillMarket.Layer(-1);
   CNeuronScenarioForecast *forecast = D2SkillForecast;
   CNeuronBaseOCL *ompb_base = (CNeuronBaseOCL *)ompb;
   CNeuronBaseOCL *forecast_base = (CNeuronBaseOCL *)forecast;
   CLayerDescription *ompb_info = (ompb ? ompb.GetLayerInfo() : NULL);
   bool layers_frozen = true;
   for(int index = 0; index <= 6; ++index)
     {
      CNeuronBaseOCL *layer = D2SkillMarket.Layer(index);
      if(!layer || layer.TrainMode())
        {
         layers_frozen = false;
         break;
        }
     }
   const bool descriptor_valid = (ompb_info &&
                                  ompb_info.window == EmbeddingSize &&
                                  ompb_info.count == BarDescr &&
                                  ompb_info.layers == OMPBSamples &&
                                  ompb_info.units.Size() == 2 &&
                                  ompb_info.units[0] == OMPBReferenceSize &&
                                  ompb_info.units[1] == OMPBCurrentWindow &&
                                  ompb_info.batch == BatchSize);
   DeleteObj(ompb_info);
   const bool forecast_tail_valid = (forecast_layer != NULL && market_tail != NULL &&
                                     forecast_layer == market_tail && forecast_base != NULL &&
                                     forecast_layer == forecast_base &&
                                     forecast_layer.Type() == defNeuronScenarioForecast);
   if(!forecast_tail_valid)
     {
      PrintFormat("%s_STAGE03_PRODUCTION_GATE_FORECAST_TAIL_FAIL checkpoint=%s", FileName, checkpoint);
      return(false);
     }
   const ulong signature = D2SkillForecastSignature();
   const bool valid = (OMPBStage02ResolveActiveCheckpoint() &&
                       !OMPBStage02RecoveryFailed &&
                       D2SkillActiveMarketFile == D2Skill_MARKET_FILE &&
                       D2SkillActiveTargetFile == D2Skill_TARGET_FILE &&
                       D2SkillActiveManifestFile == D2Skill_MANIFEST_FILE &&
                       D2SkillValidateForecastManifest(signature) &&
                       rank_tcm != NULL && rank_tcm.Type() == defNeuronCogDriverRankTCM &&
                       ompb != NULL && ompb.Mode() == OMPB_INFERENCE &&
                       ompb_base != NULL && !ompb_base.TrainMode() &&
                       forecast != NULL && forecast_base != NULL && !forecast_base.TrainMode() &&
                       !forecast.CodebookUpdate() &&
                       D2SkillMarket.WeightsUpdateEnabled() == false && layers_frozen &&
                       descriptor_valid);
   if(!valid)
     {
      PrintFormat("%s_STAGE03_PRODUCTION_GATE_FAIL checkpoint=%s", FileName, checkpoint);
      return(false);
     }
   PrintFormat("%s_STAGE03_PRODUCTION_GATE_PASS checkpoint=%s", FileName, checkpoint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Captures Market-base and OMPB parameter fingerprints for Stage 03.|
//+------------------------------------------------------------------+
bool D2SkillCaptureProductionOMPBFingerprints(void)
  {
   CNeuronOMPBOCL *ompb = GetOMPB();
   D2SkillProductionSignatureReady = false;
   D2SkillProductionBaseFingerprint = 0;
   D2SkillProductionOMPBFingerprint = ulong(1469598103934665603);
   if(!ompb || !OMPBStage02MarketFingerprint(0, 4, D2SkillProductionBaseFingerprint) ||
      !ompb.AppendParameterFingerprint(D2SkillProductionOMPBFingerprint) ||
      D2SkillProductionBaseFingerprint == 0 || D2SkillProductionOMPBFingerprint == 0)
      ReturnFalse;
   D2SkillProductionSignatureReady = true;
   PrintFormat("%s_STAGE03_SIGNATURES_BEFORE base=%I64u posterior=%I64u", FileName,
               D2SkillProductionBaseFingerprint, D2SkillProductionOMPBFingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Verifies that Stage 03 preserved the frozen Market and OMPB.     |
//+------------------------------------------------------------------+
bool D2SkillVerifyProductionOMPBFingerprints(void)
  {
   ulong base = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetOMPB();
   if(!D2SkillProductionSignatureReady || !ompb ||
      !OMPBStage02MarketFingerprint(0, 4, base) ||
      !ompb.AppendParameterFingerprint(posterior) ||
      !D2SkillValidateProductionOMPBCheckpoint())
      ReturnFalse;
   const bool unchanged = (base == D2SkillProductionBaseFingerprint &&
                           posterior == D2SkillProductionOMPBFingerprint);
   PrintFormat("%s_STAGE03_SIGNATURES_AFTER base=%I64u posterior=%I64u unchanged=%s", FileName,
               base, posterior, (unchanged ? "true" : "false"));
   return(unchanged);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillAddCross.                                        |
//+------------------------------------------------------------------+
bool D2SkillAddCross(CArrayObj *description, const bool critic, const bool d2 = false)
  {
   CLayerDescription *descr = new CLayerDescription();
   if(!description || !descr)
     { DeleteObj(descr); ReturnFalse; }
   descr.type = (d2 && !critic ? defNeuronD2Skill : defNeuronScenarioCrossAttention);
   descr.count = NScenarios;
   descr.variables = (critic ? 5 : 3);
   descr.window_out = NForecast;
   descr.window = EmbeddingSize;
   descr.layers = BarDescr;
   descr.step = StackSize;
   descr.probability = TopK;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return(true);
   DeleteObj(descr);
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillAddConv(CArrayObj *description, const uint count, const uint window, const uint output,
                    const uint variables, const ENUM_ACTIVATION activation)
  {
   CLayerDescription *descr = new CLayerDescription();
   if(!description || !descr)
     { DeleteObj(descr); ReturnFalse; }
   descr.type = defNeuronConvOCL;
   descr.count = count;
   descr.window = window;
   descr.step = window;
   descr.window_out = output;
   descr.layers = variables;
   descr.activation = activation;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return(true);
   DeleteObj(descr);
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Creates or initializes D2SkillActorCriticDescriptions.             |
//+------------------------------------------------------------------+
bool CreateD2SkillActorCriticDescriptions(CArrayObj *&actor, CArrayObj *&critic)
  {
   actor = new CArrayObj();
   critic = new CArrayObj();
   if(!actor || !critic)
     {
      DeleteObj(actor);
      DeleteObjAndFalse(critic);
     }
   actor.FreeMode(true);
   critic.FreeMode(true);
//--- Full K-major scenario tensors enter one learned trunk. There is no Pi sum.
   if(!D2SkillAddBase(actor, AccountDescr) || !D2SkillAddCross(actor, false, true) ||
      !D2SkillAddConv(actor, 1, (3 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !D2SkillAddConv(actor, 1, EmbeddingSize, 3, 2, SIGMOID))
     {
      DeleteObj(actor);
      DeleteObjAndFalse(critic);
     }
   if(!D2SkillAddBase(critic, AccountDescr + NActions) || !D2SkillAddCross(critic, true) ||
      !D2SkillAddConv(critic, 1, (5 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !D2SkillAddConv(critic, 1, EmbeddingSize, 1, 1, None))
     {
      DeleteObj(actor);
      DeleteObjAndFalse(critic);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillValidatePolicyShape.                             |
//+------------------------------------------------------------------+
bool D2SkillValidatePolicyShape(CNet &net, const bool critic)
  {
   CNeuronScenarioCrossAttention *cross = (CNeuronScenarioCrossAttention*)net.Layer(1);
   const bool type_ok = (cross && (critic ? cross.Type() == defNeuronScenarioCrossAttention
                                   : cross.Type() == defNeuronD2Skill));
   if(!type_ok ||
      cross.Variables() != BarDescr || cross.Scenarios() != NScenarios ||
      cross.Horizon() != NForecast || cross.Latent() != EmbeddingSize ||
      cross.Queries() != (critic ? 5 : 3) || cross.IsCritic() != critic ||
      cross.HistorySize() != StackSize || cross.HistoryTopK() != TopK)
      ReturnFalse;
   CNeuronBaseOCL *output_layer = net.Layer(3);
   CBufferFloat *buffer = (output_layer ? output_layer.getOutput() : NULL);
   if(!buffer || buffer.Total() != (critic ? 1 : NActions))
      ReturnFalse;
   CNeuronBaseOCL *input_layer = net.Layer(0);
   buffer = (input_layer ? input_layer.getOutput() : NULL);
   if(!buffer ||
      buffer.Total() != int(AccountDescr + (critic ? NActions : 0)))
      ReturnFalse;
   CNeuronBaseOCL *cross_layer = net.Layer(1);
   buffer = (cross_layer ? cross_layer.getOutput() : NULL);
   if(!buffer ||
      buffer.Total() != int(critic ? (5 * EmbeddingSize) : (3 * EmbeddingSize)))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillForwardForecast.                                 |
//+------------------------------------------------------------------+
bool D2SkillForwardForecast(const int position, CBufferFloat *state, CBufferFloat *time)
  {
   if(!D2SkillForecast || !CreateBuffers(position, state, time, NULL) ||
      !D2SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = D2SkillForecast.GetZ(), *u = D2SkillForecast.GetU(), *pi = D2SkillForecast.GetPi();
   return (z != NULL && u != NULL && pi != NULL && z.Total() == NScenarios * BarDescr * NForecast * EmbeddingSize &&
           u.Total() == NScenarios * BarDescr * NForecast &&
           pi.Total() == NScenarios && z.GetIndex() >= 0 && u.GetIndex() >= 0 && pi.GetIndex() >= 0);
  }
//+------------------------------------------------------------------+
//| Builds CriticInput.                                              |
//+------------------------------------------------------------------+
bool BuildCriticInput(CBufferFloat *account, CBufferFloat *action, CBufferFloat *combined)
  {
   if(!account || !action || !combined || account.Total() != AccountDescr || action.Total() != NActions ||
      account.GetIndex() < 0 || action.GetIndex() < 0 ||
      (!combined.BufferInit(AccountDescr + NActions, 0)) ||
      (combined.GetIndex() < 0 && !combined.BufferCreate(D2SkillMarket.GetOpenCL())))
      ReturnFalse;
   if(!D2SkillDevice.Bind(D2SkillMarket.GetOpenCL()))
      ReturnFalse;
   return(D2SkillDevice.Join2(account, AccountDescr, action, NActions, combined));
  }
//+------------------------------------------------------------------+
//| Implements EvaluateAction. lot instead of the former balance-... |
//+------------------------------------------------------------------+
double EvaluateAction(CBufferFloat *action, const double balance, const uint position)
  {
   const double reward = CheckAction(action, balance, position);
   if(!MathIsValidNumber(reward) || !action || action.Total() != NActions)
      return(reward);
   const double buy = MathMax(0.0, double(action[0] - action[3]));
   const double sell = MathMax(0.0, double(action[3] - action[0]));
   const double min_lot = Symb.LotsMin();
   if(MathMax(buy, sell) >= min_lot || min_lot <= 0 || balance <= 0)
      return(reward);
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), margin) || margin <= 0)
      return(reward);
   const double virtual_lot = balance / (2.0 * margin);
   if(virtual_lot <= min_lot)
      return(reward);
   return(reward * (min_lot / virtual_lot));
  }
//+------------------------------------------------------------------+
//| Builds TeacherAction. raw future window here; the live Market... |
//+------------------------------------------------------------------+
bool BuildTeacherAction(const int position, CBufferFloat *account, CBufferFloat *action, double &reward)
  {
   reward = 0;
   if(!account || !action || position < 0 || position + HistoryBars + NForecast > int(Rates.Size()) ||
//+------------------------------------------------------------------+
//| Function Total.                                                  |
//+------------------------------------------------------------------+
      account.Total() != AccountDescr)
     {
      PrintFormat("BuildTeacherAction input position=%d rates=%d account=%d action=%d",
                  position, Rates.Size(), (account ? account.Total() : -1), (action ? action.Total() : -1));
      ReturnFalse;
     }
   CBufferFloat teacher_state, teacher_time;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!CreateBuffers(position, GetPointer(teacher_state), GetPointer(teacher_time), GetPointer(D2SkillFuture)))
     {
      PrintFormat("BuildTeacherAction CreateBuffers position=%d future=%d index=%d",
                  position, D2SkillFuture.Total(), D2SkillFuture.GetIndex());
      ReturnFalse;
     }
   vector<float> account_values;
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(account.GetData(account_values) != AccountDescr)
     {
      PrintFormat("BuildTeacherAction account data total=%d index=%d", account.Total(), account.GetIndex());
      ReturnFalse;
     }
   const vector<float> teacher = OraculAction(account_values, GetPointer(D2SkillFuture));
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(teacher.Size() != NActions)
     {
      PrintFormat("BuildTeacherAction oracle size=%d expected=%d", teacher.Size(), NActions);
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!action.AssignArray(teacher))
     {
      Print("BuildTeacherAction action AssignArray");
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(action.GetIndex() < 0 && !action.BufferCreate(D2SkillMarket.GetOpenCL()))
     {
      PrintFormat("BuildTeacherAction action BufferCreate total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(action.GetIndex() >= 0 && !action.BufferWrite())
     {
      PrintFormat("BuildTeacherAction action BufferWrite total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
   reward = EvaluateAction(action, MathMax(0.0, double(account_values[0]) * EtalonBalance), (uint)position);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(!MathIsValidNumber(reward))
     {
      PrintFormat("BuildTeacherAction reward invalid %.8f", reward);
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| A valid random policy action expands Critic coverage only. Bu... |
//+------------------------------------------------------------------+
bool BuildRandomAction(const int position, CBufferFloat *account, CBufferFloat *action, double &reward)
  {
   reward = 0;
   if(!account || !action || position < 0 || position >= int(Rates.Size()) ||
      account.Total() != AccountDescr)
      ReturnFalse;
   vector<float> account_values;
   if(account.GetData(account_values) != AccountDescr)
      ReturnFalse;
   const double balance = MathMax(0.0, double(account_values[0]) * EtalonBalance);
   double margin = 0;
   if(balance <= 0 || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Rates[position].open, margin) || margin <= 0)
      ReturnFalse;
   const double min_lot = Symb.LotsMin();
   const double max_lot = MathMin(Symb.LotsMax(), balance / (2.0 * margin));
   if(min_lot <= 0 || max_lot < min_lot)
      ReturnFalse;
   const double stop_points = MathMax(Symb.StopsLevel(), 10);
   const double min_tp = stop_points / MathMax(MaxTP, 1);
   const double min_sl = (stop_points + Symb.Spread()) / MathMax(MaxSL, 1);
   if(min_tp >= 1.0 || min_sl >= 1.0)
      ReturnFalse;
   const double uniform = MathRand() / 32767.0;
   const double lot = MathMin(max_lot, NormalizeLot(min_lot + (max_lot - min_lot) * uniform));
   const double tp = min_tp + (1.0 - min_tp) * (MathRand() / 32767.0);
   const double sl = min_sl + (1.0 - min_sl) * (MathRand() / 32767.0);
   vector<float> values = vector<float>::Zeros(NActions);
   if((MathRand() & 1) != 0)
      values[0] = float(lot);
   else
      values[3] = float(lot);
   values[1] = values[4] = float(tp);
   values[2] = values[5] = float(sl);
   if(!action.AssignArray(values))
      ReturnFalse;
   if(action.GetIndex() < 0 && !action.BufferCreate(D2SkillMarket.GetOpenCL()))
      ReturnFalse;
   if(action.GetIndex() >= 0 && !action.BufferWrite())
      ReturnFalse;
   reward = EvaluateAction(action, balance, (uint)position);
   return(MathIsValidNumber(reward));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillClampAction.                                     |
//+------------------------------------------------------------------+
bool D2SkillClampAction(CBufferFloat *source, CBufferFloat *target)
  {
   if(!source || !target || !source.BufferRead() || source.Total() != NActions ||
      !target.BufferInit(NActions, 0))
      ReturnFalse;
   for(uint i = 0; i < NActions; i++)
     {
      if(!MathIsValidNumber(source[i]) || !target.Update(i, float(MathMax(0.0, MathMin(1.0, double(source[i]))))))
         ReturnFalse;
     }
   return(target.GetIndex() < 0 || target.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Implements D2SkillWriteACManifest.                                 |
//+------------------------------------------------------------------+
bool D2SkillWriteACManifest(const ulong forecast_signature)
  {
   if(forecast_signature == 0 || forecast_signature != D2SkillLastSignature)
      ReturnFalse;
   const ulong actor_hash = D2SkillHashFile(ulong(1469598103934665603), D2Skill_ACTOR_FILE);
   const ulong q1_hash = D2SkillHashFile(ulong(1469598103934665603), D2Skill_Q1_FILE);
   const ulong q2_hash = D2SkillHashFile(ulong(1469598103934665603), D2Skill_Q2_FILE);
   if(actor_hash == 0 || q1_hash == 0 || q2_hash == 0)
      ReturnFalse;
   int handle = FileOpen(D2Skill_AC_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=OMPB_ACTOR_CRITIC");
   FileWrite(handle, StringFormat("version=%u", D2Skill_AC_FORMAT_VERSION));
   FileWrite(handle, StringFormat("d2_representation=%u", D2SkillD2Representation));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", forecast_signature));
   FileWrite(handle, StringFormat("actor_context=%u", (3 * EmbeddingSize)));
   FileWrite(handle, StringFormat("critic_context=%u", (5 * EmbeddingSize)));
   FileWrite(handle, "action_order=BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL");
   FileWrite(handle, "scenario_policy=preserve_KV_no_probability_aggregation");
   FileWrite(handle, "teacher_policy=realized_future_oracul_positive_actor_all_critic");
   FileWrite(handle, "no_trade_penalty=min_executable_lot");
   FileWrite(handle, "account_execution=target_position_tp_sl_bar_lifecycle");
   FileWrite(handle, "critic_target=direct_episode_return");
   FileWrite(handle, "actor_policy_gradient=executable_action_only");
   FileWrite(handle, "critic_coverage=policy_teacher_all_random_executable");
   FileWrite(handle, "actor_supervision=positive_teacher_and_random");
   FileWrite(handle, StringFormat("actor_hash=%I64u", actor_hash));
   FileWrite(handle, StringFormat("q1_hash=%I64u", q1_hash));
   FileWrite(handle, StringFormat("q2_hash=%I64u", q2_hash));
   FileWrite(handle, StringFormat("generation=%I64u", (ulong)TimeCurrent()));
   FileClose(handle);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillValidateACManifest.                              |
//+------------------------------------------------------------------+
bool D2SkillValidateManifestRepresentation(const string manifest_file,
      const bool allow_bank_reset,
      bool &mismatch)
  {
   mismatch = false;
   const string actual = D2SkillManifestValue(manifest_file, "d2_representation");
   const string expected = IntegerToString(D2SkillD2Representation);
   if(actual != IntegerToString(D2SkillFullResidual) &&
      actual != IntegerToString(D2SkillDirectionMagnitude))
     {
      PrintFormat("D2Skill policy manifest: invalid d2_representation=%s", actual);
      ReturnFalse;
     }
   mismatch = (actual != expected);
   if(mismatch && !allow_bank_reset)
     {
      PrintFormat("D2Skill policy manifest: d2_representation expected=%s actual=%s", expected, actual);
      ReturnFalse;
     }
   if(mismatch)
      PrintFormat("D2Skill policy manifest: d2_representation expected=%s actual=%s; bank reset requested",
                  expected, actual);
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillValidateACManifestFile(const bool required, const string manifest_file,
                                   const bool allow_bank_reset)
  {
   if(D2SkillManifestValue(manifest_file, "format") == "")
      return(!required);
#define D2Skill_AC_MANIFEST_EQ(KEY,VALUE) \
   { const string actual=D2SkillManifestValue(manifest_file,KEY); const string expected=(VALUE); \
     if(actual!=expected) { PrintFormat("D2Skill policy manifest: %s expected=%s actual=%s",KEY,expected,actual); ReturnFalse; } }
   D2Skill_AC_MANIFEST_EQ("format", "OMPB_ACTOR_CRITIC");
   D2Skill_AC_MANIFEST_EQ("version", IntegerToString(D2Skill_AC_FORMAT_VERSION));
   bool representation_mismatch = false;
   if(!D2SkillValidateManifestRepresentation(manifest_file, allow_bank_reset,
         representation_mismatch))
      ReturnFalse;
   D2Skill_AC_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", D2SkillLastSignature));
   D2Skill_AC_MANIFEST_EQ("actor_context", IntegerToString((3 * EmbeddingSize)));
   D2Skill_AC_MANIFEST_EQ("critic_context", IntegerToString((5 * EmbeddingSize)));
   D2Skill_AC_MANIFEST_EQ("action_order", "BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL");
   D2Skill_AC_MANIFEST_EQ("scenario_policy", "preserve_KV_no_probability_aggregation");
   D2Skill_AC_MANIFEST_EQ("teacher_policy", "realized_future_oracul_positive_actor_all_critic");
   D2Skill_AC_MANIFEST_EQ("no_trade_penalty", "min_executable_lot");
   D2Skill_AC_MANIFEST_EQ("account_execution", "target_position_tp_sl_bar_lifecycle");
   D2Skill_AC_MANIFEST_EQ("critic_target", "direct_episode_return");
   D2Skill_AC_MANIFEST_EQ("actor_policy_gradient", "executable_action_only");
   D2Skill_AC_MANIFEST_EQ("critic_coverage", "policy_teacher_all_random_executable");
   D2Skill_AC_MANIFEST_EQ("actor_supervision", "positive_teacher_and_random");
   D2Skill_AC_MANIFEST_EQ("actor_hash", StringFormat("%I64u", D2SkillHashFile(ulong(1469598103934665603), D2Skill_ACTOR_FILE)));
   D2Skill_AC_MANIFEST_EQ("q1_hash", StringFormat("%I64u", D2SkillHashFile(ulong(1469598103934665603), D2Skill_Q1_FILE)));
   D2Skill_AC_MANIFEST_EQ("q2_hash", StringFormat("%I64u", D2SkillHashFile(ulong(1469598103934665603), D2Skill_Q2_FILE)));
#undef D2Skill_AC_MANIFEST_EQ
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(D2SkillManifestValue(manifest_file, "generation") == "")
     {
      Print("D2Skill policy manifest: generation is absent");
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillValidateACManifest(const bool required)
  {
   return(D2SkillValidateACManifestFile(required, D2Skill_AC_MANIFEST_FILE, false));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillLoadPolicyNet. Shared Actor-Critic lifecycle ... |
//+------------------------------------------------------------------+
bool D2SkillConfigureBankRepresentation(CD2SkillBank *task, CD2SkillBank *step)
  {
   if(!task || !step)
      ReturnFalse;
   const bool task_mismatch = (task.Representation() != D2SkillD2Representation);
   const bool step_mismatch = (step.Representation() != D2SkillD2Representation);
   if((task_mismatch || step_mismatch) && !D2SkillD2ResetBanksOnRepresentationMismatch)
     {
      PrintFormat("D2Skill representation mismatch expected=%d task=%d step=%d",
                  D2SkillD2Representation, task.Representation(), step.Representation());
      ReturnFalse;
     }
   if((task_mismatch && !task.ResetDurableState()) ||
      (step_mismatch && !step.ResetDurableState()))
      ReturnFalse;
   task.SetRepresentation(D2SkillD2Representation);
   step.SetRepresentation(D2SkillD2Representation);
   return(true);
  }
//+------------------------------------------------------------------+
//| A newly created policy has no durable checkpoint state to keep. |
//+------------------------------------------------------------------+
bool D2SkillSetNewBankRepresentation(CD2SkillBank *task, CD2SkillBank *step)
  {
   if(!task || !step)
      ReturnFalse;
   task.SetRepresentation(D2SkillD2Representation);
   step.SetRepresentation(D2SkillD2Representation);
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillSetNewPolicyRepresentation(CNet &net)
  {
   CNeuronBaseOCL *layer = net.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill || !skill.Ready() || !skill.TaskBank() || !skill.StepBank())
      ReturnFalse;
   return(D2SkillSetNewBankRepresentation(skill.TaskBank(), skill.StepBank()));
  }
//+------------------------------------------------------------------+
bool D2SkillLoadPolicyNet(CNet &net, const string file_name)
  {
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   return(net.Load(file_name, error, undefine, forecast, studied, true));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillCreatePolicySet.                                 |
//+------------------------------------------------------------------+
bool D2SkillCreatePolicySet(CNet &actor, CNet &q1, CNet &q2)
  {
   CArrayObj *actor_descr = NULL, *critic_descr = NULL;
   if(!CreateD2SkillActorCriticDescriptions(actor_descr, critic_descr))
     { DeleteObj(actor_descr); DeleteObj(critic_descr); ReturnFalse; }
   bool result = actor.Create(actor_descr);
   if(result)
      actor.SetOpenCL(D2SkillMarket.GetOpenCL());
   if(result)
      result = q1.Create(critic_descr);
   if(result)
      q1.SetOpenCL(D2SkillMarket.GetOpenCL());
   if(result)
      result = q2.Create(critic_descr);
   if(result)
      q2.SetOpenCL(D2SkillMarket.GetOpenCL());
   DeleteObj(actor_descr);
   DeleteObj(critic_descr);
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(result)
     {
      //--- A CNet temporary OpenCL context must be released before creating the
      //--- next policy net.  Otherwise nested MSRes buffers can retain stale
      //--- device handles from the preceding Create call.
      actor.SetOpenCL(D2SkillMarket.GetOpenCL());
      q1.SetOpenCL(D2SkillMarket.GetOpenCL());
      q2.SetOpenCL(D2SkillMarket.GetOpenCL());
     }
   return(result && D2SkillSetNewPolicyRepresentation(actor));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void D2SkillD2BankFlags(bool &task_enabled, bool &step_enabled)
  {
   const bool execution_enabled = (D2SkillD2ExecutionMode != D2_DISABLED);
   task_enabled = (execution_enabled &&
                   (D2SkillD2Mode == D2Skill_D2_MODE_TASK ||
                    D2SkillD2Mode == D2Skill_D2_MODE_FULL));
   step_enabled = (execution_enabled &&
                   (D2SkillD2Mode == D2Skill_D2_MODE_STEP ||
                    D2SkillD2Mode == D2Skill_D2_MODE_FULL));
  }
//+------------------------------------------------------------------+
//| Selects the only utility interpretation accepted by this run.   |
//+------------------------------------------------------------------+
bool D2SkillConfigureD2UtilityMode(const int utility_mode)
  {
   if(utility_mode != D2Skill_D2_UTILITY_PAIRED_HINDSIGHT)
      ReturnFalse;
   D2SkillD2UtilityMode = utility_mode;
   D2SkillD2PairedUtilityUpdates = 0;
   Print("D2Skill D2 utility_source=paired_terminal");
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillEnableD2Banks(CNet &actor)
  {
   CNeuronBaseOCL *layer = actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill || !skill.Ready())
      ReturnFalse;
   bool task_enabled = false;
   bool step_enabled = false;
   D2SkillD2BankFlags(task_enabled, step_enabled);
   if(!skill.SetMode(D2SkillD2ExecutionMode) ||
      !skill.SetOnlineDirectionUpdate(D2SkillD2OnlineDirectionUpdate) ||
      !skill.Enable(task_enabled, step_enabled))
      ReturnFalse;
   CD2SkillBank *task = skill.TaskBank();
   CD2SkillBank *step = skill.StepBank();
   if(!task || !step)
      ReturnFalse;
   if(!D2SkillConfigureBankRepresentation(task, step) ||
      !task.SetThresholds(0.65f, 0.0f) || !step.SetThresholds(0.65f, 0.0f) ||
      !task.SetLifecycle(3, 32, 256) || !step.SetLifecycle(3, 32, 256) ||
      !task.SetMaxCorrection(10.0f) || !step.SetMaxCorrection(10.0f) ||
      !task.SetAlpha(1.0f) || !step.SetAlpha(1.0f) ||
      !task.SetUtilityPolicy(D2SkillD2UtilityAware,
                             (float)MathMin(1.0, MathMax(-1.0, D2SkillD2MinUtility))) ||
      !step.SetUtilityPolicy(D2SkillD2UtilityAware,
                             (float)MathMin(1.0, MathMax(-1.0, D2SkillD2MinUtility))))
      ReturnFalse;
   PrintFormat("D2Skill D2 stage=%d mode=%d execution=%d task=%s step=%s utility=%s scale=%.3f direction_ema=%s",
               D2SkillD2Stage, D2SkillD2Mode, D2SkillD2ExecutionMode, (task_enabled ? "on" : "off"),
               (step_enabled ? "on" : "off"),
               (D2SkillD2UtilityAware ? "aware" : "similarity"), D2SkillD2UtilityScale,
               (D2SkillD2OnlineDirectionUpdate ? "on" : "off"));
   return(true);
  }
//+------------------------------------------------------------------+
//| Applies one normalized utility observation to selected banks.    |
//+------------------------------------------------------------------+
bool D2SkillApplyD2Utility(CNet &actor, const double value, bool &applied,
                           bool &task_applied, bool &step_applied)
  {
   applied = false;
   task_applied = false;
   step_applied = false;
   if((D2SkillD2ExecutionMode != D2_EVALUATE &&
       D2SkillD2ExecutionMode != D2_ONLINE_CALIBRATION) ||
      !MathIsValidNumber(value))
      ReturnFalse;
   CNeuronBaseOCL *layer = actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill || !skill.Ready())
      ReturnFalse;
   const double normalized = MathMax(-1.0, MathMin(1.0,
                                     value * D2SkillD2UtilityScale / MathMax(EtalonBalance, 1.0)));
   if(!MathIsValidNumber(normalized))
      ReturnFalse;
   bool task_enabled = false;
   bool step_enabled = false;
   D2SkillD2BankFlags(task_enabled, step_enabled);
   if(task_enabled)
      if(!skill.TaskUtility((float)normalized, task_applied))
         ReturnFalse;
   if(step_enabled)
      if(!skill.StepUtility((float)normalized, step_applied))
         ReturnFalse;
   applied = (task_applied || step_applied);
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillApplyD2Utility(CNet &actor, const double value, bool &applied)
  {
   bool task_applied = false;
   bool step_applied = false;
   return(D2SkillApplyD2Utility(actor, value, applied, task_applied, step_applied));
  }
//+------------------------------------------------------------------+
//| Applies paired hindsight delta JSkill-JBase to selected skills.  |
//| Utility is separate from Actor gradients and pseudo-residuals.  |
//+------------------------------------------------------------------+
bool D2SkillUpdateD2UtilityDelta(CNet &actor, const double delta_j, bool &applied,
                                 bool &task_applied, bool &step_applied)
  {
   applied = false;
   task_applied = false;
   step_applied = false;
   if((D2SkillD2ExecutionMode != D2_EVALUATE &&
       D2SkillD2ExecutionMode != D2_ONLINE_CALIBRATION) ||
      D2SkillD2Mode == D2Skill_D2_MODE_BASE)
      return(true);
   if(D2SkillD2UtilityMode != D2Skill_D2_UTILITY_PAIRED_HINDSIGHT ||
      !MathIsValidNumber(delta_j))
      ReturnFalse;
   if(!D2SkillApplyD2Utility(actor, delta_j, applied, task_applied, step_applied))
      ReturnFalse;
   if(applied)
      D2SkillD2PairedUtilityUpdates++;
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillUpdateD2UtilityDelta(CNet &actor, const double delta_j, bool &applied)
  {
   bool task_applied = false;
   bool step_applied = false;
   return(D2SkillUpdateD2UtilityDelta(actor, delta_j, applied, task_applied, step_applied));
  }
//+------------------------------------------------------------------+
bool D2SkillUpdateD2UtilityDelta(CNet &actor, const double delta_j)
  {
   bool applied = false;
   return(D2SkillUpdateD2UtilityDelta(actor, delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Resets only the selected D2Skill episode influence state.        |
//+------------------------------------------------------------------+
bool D2SkillResetEpisodeInfluence(CNet &actor)
  {
   CNeuronBaseOCL *layer = actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill.Ready() || !skill.ResetEpisodeInfluence())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Closes the offline episode through paired utility then reset.     |
//+------------------------------------------------------------------+
bool D2SkillCloseOfflineEpisode(CNet &actor, const bool paired_active,
                                const bool pair_terminal, const bool episode_limit,
                                const SD2SkillEpisodeOutcome &base,
                                const SD2SkillEpisodeOutcome &skill,
                                double &terminal_delta, bool &utility_applied,
                                bool &influence_reset, bool &episode_closed)
  {
   terminal_delta = 0.0;
   utility_applied = false;
   influence_reset = false;
   episode_closed = (pair_terminal || episode_limit);
   if(!episode_closed)
      return(true);
   if(paired_active)
     {
      if(!D2SkillValidatePairedEpisode(base, skill) ||
         !D2SkillComputePairedEpisodeDelta(base, skill, terminal_delta) ||
         !D2SkillUpdateD2UtilityDelta(actor, terminal_delta, utility_applied))
         ReturnFalse;
     }
   if(!D2SkillResetEpisodeInfluence(actor))
      ReturnFalse;
   influence_reset = true;
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillCanCreatePolicyCheckpoint(const bool manifest_exists, const bool allow_create)
  {
   if(!manifest_exists)
      return(allow_create);
   return(D2SkillD2RecreateIncompatibleCheckpoint);
  }
//+------------------------------------------------------------------+
bool D2SkillLoadOrCreatePolicySetFile(CNet &actor, CNet &q1, CNet &q2,
                                      const bool allow_create, const string manifest_file)
  {
   const bool manifest_exists = FileIsExist(manifest_file, FILE_COMMON);
   bool loaded = false;
   if(manifest_exists)
     {
      //--- This persisted metadata preflight runs before CNet::Load can materialize
      //--- a serialized D2Skill bank.  A mismatch loads only on explicit bank reset.
      loaded = (D2SkillValidateACManifestFile(true, manifest_file,
                                              D2SkillD2ResetBanksOnRepresentationMismatch) &&
                D2SkillLoadPolicyNet(actor, D2Skill_ACTOR_FILE) &&
                D2SkillLoadPolicyNet(q1, D2Skill_Q1_FILE) &&
                D2SkillLoadPolicyNet(q2, D2Skill_Q2_FILE));
      if(!loaded && !D2SkillCanCreatePolicyCheckpoint(true, allow_create))
        {
         Print("D2Skill policy restore=FAIL; incompatible checkpoint recreation is disabled");
         ReturnFalse;
        }
     }
   else
      if(!D2SkillCanCreatePolicyCheckpoint(false, allow_create))
         ReturnFalse;
   if(!loaded)
     {
      if(manifest_exists)
         Print("D2Skill policy restore=FAIL; recreating incompatible policy set by explicit input");
      if(!D2SkillCreatePolicySet(actor, q1, q2))
         ReturnFalse;
     }
   actor.SetOpenCL(D2SkillMarket.GetOpenCL());
   q1.SetOpenCL(D2SkillMarket.GetOpenCL());
   q2.SetOpenCL(D2SkillMarket.GetOpenCL());
   if(!D2SkillValidatePolicyShape(actor, false) || !D2SkillValidatePolicyShape(q1, true) ||
      !D2SkillValidatePolicyShape(q2, true))
      ReturnFalse;
   if(!D2SkillEnableD2Banks(actor))
      ReturnFalse;
   actor.TrainMode(true);
   q1.TrainMode(true);
   q2.TrainMode(true);
   return(true);
  }
//+------------------------------------------------------------------+
bool D2SkillLoadOrCreatePolicySet(CNet &actor, CNet &q1, CNet &q2,
                                  const bool allow_create)
  {
   return(D2SkillLoadOrCreatePolicySetFile(actor, q1, q2, allow_create,
                                           D2Skill_AC_MANIFEST_FILE));
  }
//+------------------------------------------------------------------+
//| Configures the Actor-Critic write permissions for the D2 mode.  |
//+------------------------------------------------------------------+
bool D2SkillConfigureActorCriticUpdates(CNet &actor, CNet &q1, CNet &q2)
  {
//--- D2_DISABLED retains its ordinary Actor-Critic baseline.  Online
//--- calibration freezes Actor by default and permits Critic writes only via
//--- the explicit runtime opt-in.  Direction EMA remains independently opt-in.
   bool actor_weights = false;
   bool critic_weights = false;
   switch(D2SkillD2ExecutionMode)
     {
      case D2_DISABLED:
         actor_weights = true;
         critic_weights = true;
         break;
      case D2_COLLECT:
         critic_weights = true;
         break;
      case D2_EVALUATE:
      case D2_INFERENCE:
         break;
      case D2_ONLINE_CALIBRATION:
         critic_weights = D2SkillD2OnlineCriticUpdate;
         break;
      default:
         ReturnFalse;
     }
//--- CNet preserves TrainMode and full gradient routing.  Only its explicit
//--- bWeightsUpdate gate controls trainable-parameter writes.
   if(!actor.SetWeightsUpdate(actor_weights) || !q1.SetWeightsUpdate(critic_weights) ||
      !q2.SetWeightsUpdate(critic_weights) || !D2SkillMarket.SetWeightsUpdate(false) ||
      !D2SkillMarket.TrainMode(false))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillSavePolicySet.                                   |
//+------------------------------------------------------------------+
bool D2SkillSavePolicySet(CNet &actor, CNet &q1, CNet &q2)
  {
   if(!D2SkillVerifyFrozenWeightsExact() || !D2SkillVerifyFrozenCodebookExact() ||
      D2SkillForecastSignature() != D2SkillLastSignature)
      ReturnFalse;
   const datetime now = TimeCurrent();
   return (actor.Save(D2Skill_ACTOR_FILE, 0, 0, 0, now, true) &&
           q1.Save(D2Skill_Q1_FILE, q1.getRecentAverageError(), 0, 0, now, true) &&
           q2.Save(D2Skill_Q2_FILE, q2.getRecentAverageError(), 0, 0, now, true) &&
           D2SkillWriteACManifest(D2SkillLastSignature));
  }
//+------------------------------------------------------------------+
//| Saves a Stage 03 checkpoint only for the active production OMPB.|
//+------------------------------------------------------------------+
bool D2SkillSaveStage03StopCheckpoint(CNet &actor, CNet &q1, CNet &q2)
  {
   if(!D2SkillVerifyProductionOMPBFingerprints())
      ReturnFalse;
   return(D2SkillSavePolicySet(actor, q1, q2));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillLoadInferenceActor.                              |
//+------------------------------------------------------------------+
bool D2SkillLoadInferenceActor(CNet &actor)
  {
   if(!D2SkillValidateACManifestFile(true, D2Skill_AC_MANIFEST_FILE,
                                     D2SkillD2ResetBanksOnRepresentationMismatch) ||
      !D2SkillLoadPolicyNet(actor, D2Skill_ACTOR_FILE))
      ReturnFalse;
   actor.SetOpenCL(D2SkillMarket.GetOpenCL());
   if(!D2SkillValidatePolicyShape(actor, false))
      ReturnFalse;
   if(!D2SkillEnableD2Banks(actor))
      ReturnFalse;
   actor.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements ReadAction.                                           |
//+------------------------------------------------------------------+
bool ReadAction(CNet &net, CBufferFloat *target)
  {
//--- This is the explicit device-to-CPU boundary for trade execution.
//--- The final actor layer is allowed to be host-only, therefore its live
//--- output must not be read through CBufferFloat::BufferRead().
   if(!target)
      ReturnFalse;
//--- CNet::getResults reuses a valid result object.  Passing target directly
//--- avoids an allocation and the subsequent CPU-to-CPU AssignArray copy.
   CBufferFloat *output = target;
   net.getResults(output);
   return(output == target && target.Total() == NActions);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillAdvanceAccountTime.                              |
//+------------------------------------------------------------------+
bool D2SkillAdvanceAccountTime(CBufferFloat *current, const datetime next_time, CBufferFloat *next)
  {
   if(!current || !next || current.Total() != AccountDescr ||
      (current.GetIndex() >= 0 && !current.BufferRead()) ||
      !next.BufferInit(AccountDescr, 0))
      ReturnFalse;
   for(uint i = 0; i < 9; i++)
      if(!next.Update(i, current[i]))
         ReturnFalse;
   double x = next_time / (double)(D'2024.01.01' - D'2023.01.01');
   if(!next.Update(9, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_MN1);
   if(!next.Update(10, float(MathCos(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_W1);
   if(!next.Update(11, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_D1);
   if(!next.Update(12, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   return(next.GetIndex() < 0 || next.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Implements D2SkillForwardForecastState.                            |
//+------------------------------------------------------------------+
bool D2SkillForwardForecastState(CBufferFloat *state)
  {
//--- CNet accepts a host-only input buffer and uploads it through its first
//--- layer.  Do not require a device allocation from the caller here.
   if(!D2SkillForecast || !state || state.Total() != HistoryBars * BarDescr)
      ReturnFalse;
   if(!D2SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = D2SkillForecast.GetZ(), *u = D2SkillForecast.GetU(), *pi = D2SkillForecast.GetPi();
   if(!z || !u || !pi || z.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize ||
      u.Total() != NScenarios * BarDescr * NForecast || pi.Total() != NScenarios ||
      z.GetIndex() < 0 || u.GetIndex() < 0 || pi.GetIndex() < 0)
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillBuildLiveAccount(const double previous_balance, const double previous_equity,
                             const datetime state_time, CBufferFloat *account,
                             double &buy_value, double &sell_value)
  {
   if(!account || !MathIsValidNumber(previous_balance) ||
      !MathIsValidNumber(previous_equity) || previous_balance <= 0 || previous_equity <= 0)
      ReturnFalse;
   double buy_profit = 0, sell_profit = 0, position_discount = 0;
   buy_value = 0;
   sell_value = 0;
   const datetime current = TimeCurrent();
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      const double profit = PositionGetDouble(POSITION_PROFIT);
      if((int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
        { buy_value += PositionGetDouble(POSITION_VOLUME); buy_profit += profit; }
      else
        { sell_value += PositionGetDouble(POSITION_VOLUME); sell_profit += profit; }
      position_discount += (current - PositionGetInteger(POSITION_TIME)) *
                           (1.0 / (60.0 * 60.0 * 10.0)) * MathAbs(profit);
     }
   vector<float> values = vector<float>::Zeros(AccountDescr);
   const double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   values[0] = float(balance / EtalonBalance);
   values[1] = float((balance - previous_balance) / previous_balance);
   values[2] = float(equity / previous_balance);
   values[3] = float((equity - previous_equity) / previous_equity);
   values[4] = float(buy_value);
   values[5] = float(sell_value);
   values[6] = float(buy_profit / previous_balance);
   values[7] = float(sell_profit / previous_balance);
   values[8] = float(position_discount / previous_balance);
   double x = state_time / (double)(D'2024.01.01' - D'2023.01.01');
   values[9] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_MN1);
   values[10] = float(MathCos(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_W1);
   values[11] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_D1);
   values[12] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   for(uint i = 0; i < AccountDescr; i++)
      if(!MathIsValidNumber(values[i]))
         ReturnFalse;
   return(account.AssignArray(values) && (account.GetIndex() < 0 || account.BufferWrite()));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillRefreshLiveMarket.                               |
//+------------------------------------------------------------------+
bool D2SkillRefreshLiveMarket(CBufferFloat *state, CBufferFloat *time_state)
  {
   const int requested = StackSize + HistoryBars;
//--- A new-bar event is evaluated from the last fully closed bar.  The
//--- forming bar must not leak unfinished OHLC/indicator values into policy.
   const int bars = CopyRates(Symb.Name(), TimeFrame, 1, requested, Rates);
   if(!state || !time_state || bars < HistoryBars || !ArraySetAsSeries(Rates, true) ||
      !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars) ||
      RSI.BarsCalculated() < bars || CCI.BarsCalculated() < bars ||
      ATR.BarsCalculated() < bars || MACD.BarsCalculated() < bars)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
   return(CreateBuffers(0, state, time_state, (CBufferFloat*)NULL));
  }
//+------------------------------------------------------------------+
//| Checks ExecutableOrder. One execution contract for historical... |
//+------------------------------------------------------------------+
bool IsExecutableOrder(const double lot, const double tp_fraction, const double sl_fraction)
  {
   const double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
   return (lot >= Symb.LotsMin() && tp_fraction * MaxTP * Symb.Point() > 2.0 * stops &&
           sl_fraction * MaxSL * Symb.Point() > stops);
  }
//+------------------------------------------------------------------+
//| Implements NormalizeLot.                                         |
//+------------------------------------------------------------------+
double NormalizeLot(const double lot)
  {
   const double min_lot = Symb.LotsMin(), step_lot = Symb.LotsStep();
   return(step_lot > 0 ? min_lot + MathRound((lot - min_lot) / step_lot) * step_lot : lot);
  }
#ifndef Study
//+------------------------------------------------------------------+
//| Implements D2SkillValidateAction.                                  |
//+------------------------------------------------------------------+
bool D2SkillValidateAction(CBufferFloat *action)
  {
   if(!action || action.Total() != NActions ||
      (action.GetIndex() >= 0 && !action.BufferRead()))
      ReturnFalse;
   for(uint i = 0; i < NActions; i++)
      if(!MathIsValidNumber(action[i]) || action[i] < 0 || action[i] > 1)
         ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool D2SkillExecuteAction(CBufferFloat *action, double buy_value, double sell_value, double &margin_penalty,
                          bool &market_closed)
  {
   margin_penalty = 0;
   market_closed = false;
   if(!D2SkillValidateAction(action))
      ReturnFalse;
//--- Canonical CogDriver mutual exclusion and broker constraints.
   if(action[0] >= action[3])
     { action.Update(0, action[0] - action[3]); action.Update(3, 0); }
   else
     { action.Update(3, action[3] - action[0]); action.Update(0, 0); }
   const double min_lot = Symb.LotsMin();
   if(!IsExecutableOrder(action[0], action[1], action[2]))
     { if(buy_value > 0) CloseByDirection(POSITION_TYPE_BUY); }
   else
     {
      const double lot = NormalizeLot(action[0]);
      const double tp = NormalizeDouble(Symb.Ask() + action[1] * MaxTP * Symb.Point(), Symb.Digits());
      const double sl = NormalizeDouble(Symb.Ask() - action[2] * MaxSL * Symb.Point(), Symb.Digits());
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, sl, tp);
      if((buy_value - lot) >= min_lot)
         ClosePartial(POSITION_TYPE_BUY, buy_value - lot);
      else
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if((lot - buy_value) >= min_lot && !Trade.Buy(lot - buy_value, Symb.Name(), Symb.Ask(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
            //+------------------------------------------------------------------+
            //| Function if.                                                     |
            //+------------------------------------------------------------------+
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return(true);
              }
            //+------------------------------------------------------------------+
            //| Function if.                                                     |
            //+------------------------------------------------------------------+
            if(retcode != 10019)
              {
               PrintFormat("D2Skill buy execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
               ReturnFalse;
              }
            //--- Preserve CogDriver's insufficient-margin feedback, but keep the
            //--- transition alive: the order was rejected, not the account.
            margin_penalty -= 100.0 * (lot - buy_value);
           }
     }
   if(!IsExecutableOrder(action[3], action[4], action[5]))
     { if(sell_value > 0) CloseByDirection(POSITION_TYPE_SELL); }
   else
     {
      const double lot = NormalizeLot(action[3]);
      const double tp = NormalizeDouble(Symb.Bid() - action[4] * MaxTP * Symb.Point(), Symb.Digits());
      const double sl = NormalizeDouble(Symb.Bid() + action[5] * MaxSL * Symb.Point(), Symb.Digits());
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sl, tp);
      if((sell_value - lot) >= min_lot)
         ClosePartial(POSITION_TYPE_SELL, sell_value - lot);
      else
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if((lot - sell_value) >= min_lot && !Trade.Sell(lot - sell_value, Symb.Name(), Symb.Bid(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
            //+------------------------------------------------------------------+
            //| Function if.                                                     |
            //+------------------------------------------------------------------+
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return(true);
              }
            //+------------------------------------------------------------------+
            //| Function if.                                                     |
            //+------------------------------------------------------------------+
            if(retcode != 10019)
              {
               PrintFormat("D2Skill sell execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
               ReturnFalse;
              }
            margin_penalty -= 100.0 * (lot - sell_value);
           }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillExecuteAction. Inference callers do not train... |
//+------------------------------------------------------------------+
bool D2SkillExecuteAction(CBufferFloat *action, double buy_value, double sell_value)
  {
   double margin_penalty = 0;
   bool market_closed = false;
   return(D2SkillExecuteAction(action, buy_value, sell_value, margin_penalty, market_closed));
  }
#endif
bool PolicyBackward(CNet &actor, CNet &critic, CBufferFloat *actor_target,
                    CNet *scenario_net, const int scenario_layer)
  {
   CNeuronBaseOCL *critic_output = critic.Layer(3);
   CNeuronBaseOCL *critic_base = critic.Layer(0);
   CNeuronBaseOCL *actor_output = actor.Layer(3);
   if(!critic_output || !critic_base || !actor_output || !actor_target ||
      critic_output.getOutput().Total() != 1 || actor_output.getGradient().Total() != NActions ||
      !actor_target.BufferInit(1, 1.0f) ||
      (actor_target.GetIndex() < 0 && !actor_target.BufferCreate(D2SkillMarket.GetOpenCL())) ||
      !D2SkillDevice.Add(critic_output.getOutput(), actor_target, actor_target, 1))
      ReturnFalse;
   critic.TrainMode(false);
   bool result = critic.backProp(actor_target, scenario_net, scenario_layer);
   CBufferFloat account_gradient;
   if(!(result && critic_base.getGradient().Total() == AccountDescr + NActions &&
        account_gradient.BufferInit(AccountDescr, 0) &&
        (account_gradient.GetIndex() >= 0 || account_gradient.BufferCreate(D2SkillMarket.GetOpenCL())) &&
        D2SkillDevice.Split2(GetPointer(account_gradient), actor_output.getGradient(), critic_base.getGradient(),
                             AccountDescr, NActions) &&
        actor.backPropGradient(scenario_net, scenario_layer, -1, true)))
      result = false;
   critic.TrainMode(true);
   return(result);
  }
//+------------------------------------------------------------------+
//| Finalizes projected balance before writing terminal account state.|
//+------------------------------------------------------------------+
bool D2SkillFinalizeAccountBalance(const double balance, const double realized,
                                   const double min_balance, double &next_balance,
                                   bool &terminal)
  {
   if(!MathIsValidNumber(balance) || !MathIsValidNumber(realized) ||
      !MathIsValidNumber(min_balance) || min_balance < 0.0)
      ReturnFalse;
   const double projected_balance = balance + realized;
   if(!MathIsValidNumber(projected_balance))
      ReturnFalse;
   terminal = (projected_balance <= min_balance);
   next_balance = (terminal ? min_balance : projected_balance);
   return(true);
  }
//+------------------------------------------------------------------+
//| One historical-bar account transition.  CheckAction remains the  |
//+------------------------------------------------------------------+
bool AdvanceAccount(CBufferFloat *current, CBufferFloat *action, const int position,
                    const double min_balance, CBufferFloat *next, double &reward, bool &terminal)
  {
   terminal = true;
   reward = 0;
   if(!current || !action || !next || position <= 0 || position >= int(Rates.Size()) ||
      current.Total() != AccountDescr || action.Total() != NActions ||
      (current.GetIndex() >= 0 && !current.BufferRead()) ||
      (action.GetIndex() >= 0 && !action.BufferRead()))
      ReturnFalse;
   const double balance = MathMax(0.0, double(current[0]) * EtalonBalance);
   reward = EvaluateAction(action, balance, (uint)position);
   if(!MathIsValidNumber(reward))
      ReturnFalse;
   const double buy = MathMax(0.0, double(action[0] - action[3]));
   const double sell = MathMax(0.0, double(action[3] - action[0]));
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Rates[position].open, margin))
      ReturnFalse;
   const double min_lot = Symb.LotsMin();
//+------------------------------------------------------------------+
//| Function if.                                                     |
//+------------------------------------------------------------------+
   if(balance <= min_balance || balance < margin * min_lot)
     {
      terminal = true;
      return(D2SkillAdvanceAccountTime(current, Rates[position - 1].time, next));
     }
//--- Action is the target position for the next bar.  A valid same-direction
//--- action modifies its TP/SL; a valid opposite action closes then reopens;
//--- an invalid action closes both positions.
   const bool open_buy = IsExecutableOrder(buy, action[1], action[2]);
   const bool open_sell = IsExecutableOrder(sell, action[4], action[5]);
   const double target_buy = (open_buy ? NormalizeLot(buy) : 0.0);
   const double target_sell = (open_sell ? NormalizeLot(sell) : 0.0);
   const double point_cost = Symb.TickValue() / Symb.TickSize();
   const double entry = Rates[position].open;
   const double spread = Symb.Spread() * Symb.Point();
   const double current_buy = MathMax(0.0, double(current[4]));
   const double current_sell = MathMax(0.0, double(current[5]));
   double current_buy_profit = double(current[6]) * balance;
   double current_sell_profit = double(current[7]) * balance;
   double next_buy = 0, next_sell = 0;
   double next_buy_profit = 0, next_sell_profit = 0;
   double realized = 0;
//+------------------------------------------------------------------+
//| An invalid target liquidates every open position at the curre... |
//+------------------------------------------------------------------+
   if(!open_buy && !open_sell)
     {
      realized = current_buy_profit + current_sell_profit;
     }
   else
      //+------------------------------------------------------------------+
      //| Function if.                                                     |
      //+------------------------------------------------------------------+
      if(open_buy)
        {
         //--- A reverse target closes the Sell; a same-side reduction realizes only its
         //--- proportional carried P/L.  The retained Buy keeps its marked-to-market P/L.
         realized += current_sell_profit;
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(current_buy > target_buy && current_buy > 0)
           {
            const double closed = current_buy - target_buy;
            realized += current_buy_profit * closed / current_buy;
            current_buy_profit -= current_buy_profit * closed / current_buy;
           }
         const double added = MathMax(0.0, target_buy - current_buy);
         current_buy_profit -= spread * point_cost * added;
         const double tp = entry + (action[1] * MaxTP + Symb.Spread()) * Symb.Point();
         const double sl = entry - (action[2] * MaxSL + Symb.Spread()) * Symb.Point();
         const MqlRates bar = Rates[position - 1];
         if(sl >= bar.low)
            realized += current_buy_profit + (sl - entry) * point_cost * target_buy;
         else
            if(tp <= bar.high)
               realized += current_buy_profit + (tp - entry) * point_cost * target_buy;
            else
              {
               next_buy = target_buy;
               next_buy_profit = current_buy_profit + (bar.open - entry) * point_cost * target_buy;
              }
        }
      else
        {
         //--- Symmetric Sell lifecycle.  The SL-before-TP order matches CheckAction.
         realized += current_buy_profit;
         //+------------------------------------------------------------------+
         //| Function if.                                                     |
         //+------------------------------------------------------------------+
         if(current_sell > target_sell && current_sell > 0)
           {
            const double closed = current_sell - target_sell;
            realized += current_sell_profit * closed / current_sell;
            current_sell_profit -= current_sell_profit * closed / current_sell;
           }
         const double added = MathMax(0.0, target_sell - current_sell);
         current_sell_profit -= spread * point_cost * added;
         const double tp = entry - (action[4] * MaxTP + Symb.Spread()) * Symb.Point();
         const double sl = entry + (action[5] * MaxSL - Symb.Spread()) * Symb.Point();
         const MqlRates bar = Rates[position - 1];
         if(sl <= bar.high)
            realized += current_sell_profit + (entry - sl) * point_cost * target_sell;
         else
            if(tp >= bar.low)
               realized += current_sell_profit + (entry - tp) * point_cost * target_sell;
            else
              {
               next_sell = target_sell;
               next_sell_profit = current_sell_profit + (entry - bar.open) * point_cost * target_sell;
              }
        }
   double next_balance = 0.0;
   bool balance_terminal = false;
   if(!D2SkillFinalizeAccountBalance(balance, realized, min_balance, next_balance, balance_terminal))
      ReturnFalse;
   const double current_equity = balance + double(current[6]) * balance + double(current[7]) * balance;
   const double next_equity = next_balance + next_buy_profit + next_sell_profit;
   if(!D2SkillAdvanceAccountTime(current, Rates[position - 1].time, next) ||
      (next.GetIndex() >= 0 && !next.BufferRead()))
      ReturnFalse;
   if(!next.Update(0, float(next_balance / EtalonBalance)) ||
      !next.Update(1, float((next_balance - balance) / MathMax(balance, 1.0))) ||
      !next.Update(2, float(next_equity / MathMax(balance, 1.0))) ||
      !next.Update(3, float((next_equity - current_equity) / MathMax(MathAbs(current_equity), 1.0))) ||
      !next.Update(4, float(next_buy)) || !next.Update(5, float(next_sell)) ||
      !next.Update(6, float(next_buy_profit / MathMax(next_balance, 1.0))) ||
      !next.Update(7, float(next_sell_profit / MathMax(next_balance, 1.0))) ||
      !next.Update(8, 0.0f))
      ReturnFalse;
   terminal = (balance_terminal || next_equity < margin * min_lot);
   return(next.GetIndex() < 0 || next.BufferWrite());
  }
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
