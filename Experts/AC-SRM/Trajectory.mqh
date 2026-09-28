//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+-------------------------------------------------------------------------------------------------------------------------------+
//| OMPB chain stage selector. Stage 02 was the original scaffold;  the expert now runs the full chain: MarketEncoder..Inference  |
//+-------------------------------------------------------------------------------------------------------------------------------+
enum ENUM_OMPB_STAGE
  {
   OMPB_STAGE_MARKET_ENCODER = 1,     //Chain stage 01 - forecast (market encoder)
   OMPB_STAGE_CALIBRATION,            //Chain stage 02 - OMPB calibration
   OMPB_STAGE_BASE_POLICY,            //Chain stages 03-05 - base policy study / online / test
   OMPB_STAGE_SKILL,                  //Reserved: skill stage
   OMPB_STAGE_OPTIMIZATION,           //Reserved: optimization stage
   OMPB_STAGE_INFERENCE               //Reserved: inference stage
  };
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
#define        FileName                "ACSRM"
#define        ACSRM_LOG_PREFIX         "ACSRM"
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
#define        ACSRMSamples              5
#define        ACSRMReferenceSize        256
#define        ACSRMCurrentWindow        64
#define        Quantiles               8
#define        StackSize               24*21
#define        Blocks                  24
CSymbolInfo          Symb;
CTrade               Trade;
MqlRates             Rates[];
//---
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
//| Complete outcome of one independently simulated Skill episode.   |
//+------------------------------------------------------------------+
struct SSkillEpisodeOutcome
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
   bool              Reset(void);
   bool              Accumulate(const double reward, const double balance,
                                const double equity, const double drawdown,
                                const double cost, const uint duration,
                                const double risk = 0.0);
   double            Outcome(void) const { return(dOutcome); }
   double            Balance(void) const { return(dBalance); }
   double            Equity(void) const { return(dEquity); }
   double            Drawdown(void) const { return(dDrawdown); }
   double            Cost(void) const { return(dCost); }
   double            Risk(void) const { return(dRisk); }
   uint              Duration(void) const { return(uDuration); }
  };
//+------------------------------------------------------------------+
//| Resets the episode outcome block to its neutral state.           |
//+------------------------------------------------------------------+
bool SSkillEpisodeOutcome::Reset(void)
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
//+------------------------------------------------------------------+
//| Accumulates one step into the episode outcome block.             |
//+------------------------------------------------------------------+
bool SSkillEpisodeOutcome::Accumulate(const double reward, const double balance,
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
//+------------------------------------------------------------------+
//| Calculates terminal paired utility from complete episode state.  |
//+------------------------------------------------------------------+
bool SkillComputePairedEpisodeDelta(const SSkillEpisodeOutcome &base,
                                      const SSkillEpisodeOutcome &skill,
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
bool SkillValidatePairedEpisode(const SSkillEpisodeOutcome &base,
                                  const SSkillEpisodeOutcome &skill)
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
bool SkillCheckpointDue(const ulong transitions, const int checkpoint_interval)
  {
   return(checkpoint_interval > 0 && transitions > 0 &&
          transitions % (ulong)checkpoint_interval == 0);
  }
//+------------------------------------------------------------------+
//| DeltaJ may close only at a terminal or configured pair horizon.  |
//+------------------------------------------------------------------+
bool SkillOnlinePairBoundary(const bool real_terminal, const bool virtual_terminal,
                               const ulong pair_transitions, const int pair_horizon)
  {
   if(real_terminal || virtual_terminal)
      return(true);
   return(pair_horizon > 0 && pair_transitions >= (ulong)pair_horizon);
  }
//+------------------------------------------------------------------+
//| An independent pair ends as soon as either branch is terminal.   |
//+------------------------------------------------------------------+
bool SkillPairReachedTerminal(const bool base_terminal, const bool skill_terminal)
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
//| Builds the market state encoder/decoder description stack.       |
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
//+-------------------------------------------------------------------+
//| Builds the Actor/Critic description stacks for the AC-SRM graph.  |
//+-------------------------------------------------------------------+
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
//--- Creates and manages object lifecycle for CloseTime.
   CloseTime(0),
//--- Implements Type.
   Type(POSITION_TYPE_BUY),
//--- Implements Volume.
   Volume(0),
//--- Implements OpenPrice.
   OpenPrice(0),
//--- Implements StopLos.
   StopLos(0),
//--- Implements TakeProfit.
   TakeProfit(0),
//--- Implements point.
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
//| Implements Action (Action).                                      |
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
            //--- Implements MathAbs.
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
double CheckAction(CBufferFloat *action, double balance, uint start_position, const int horizon_bars,
                   int &outcome_tag)
  {
//--- outcome_tag: 0 = no resolvable label (data boundary reached before the
//--- outcome, or unexecutable/penalty value), 1 = TP first, 2 = SL first,
//--- 3 = HORIZON (no level reached before H; includes account blow-out).
//--- Incomplete horizons are NEVER reported as full HORIZON outcomes.
   outcome_tag = 0;
   if(!action || start_position >= Rates.Size() || horizon_bars < 1)
      return(0);
//---
   double buy_lot = MathMax(double(action[0] - action[3]), 0);
   double sell_lot = MathMax(double(action[3] - action[0]), 0);
   double marg = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), marg))
      return(0);
   double point_cost = Symb.TickValue() / Symb.TickSize();
   const double net_lot = MathMax(buy_lot, sell_lot);
   if(net_lot <= 0)
     {
//--- True no-position action: explicit penalty, no resolvable label.
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * balance / (2 * marg);
      return(loss);
     }
//--- The deal result is evaluated at the STATED lot (money P&L is linear in
//--- the volume).  No min-lot floor: the Actor keeps the correct gradient
//--- dSRM/dlot direction even below the broker minimum, and executability
//--- remains an order-level property (IsExecutableOrder), never a label one.
   if((marg * net_lot) >= balance)
     {
      double loss = -MathMax(Rates[start_position].high - Rates[start_position].open,
                             Rates[start_position].open - Rates[start_position].low) *
                    point_cost * net_lot;
      return(loss);
     }
   point_cost *= net_lot;
//--- The deal is single-leg (buy_lot * sell_lot == 0 by construction).
   double tp = 0, sl = 0, profit = 0;
   int stops = MathMax(Symb.StopsLevel(), 10);
   int spread = Symb.Spread();
   const int steps = MathMax(1, horizon_bars);
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
//--- One-time cost at the entry; it is never charged again.
      profit = -spread * Symb.Point() * point_cost;
      for(int n = 0; n < steps; n++)
        {
         const int i = int(start_position) - n;
//--- The label must be resolvable INSIDE the loaded window: a missing bar
//--- means the outcome does not exist yet (tag 0, no label), it is NOT a
//--- HORIZON result.
         if(i < 0)
           {
            outcome_tag = 0;
            return(0.0);
           }
//--- Explicit within-bar rule: SL-before-TP (original methodology order).
//--- The last buy leg is (SL - O_i): a stop closes BELOW the open, so the
//--- realized amount reduces profit.  (O_i - SL) would have ADDED a gain.
         if(sl >= Rates[i].low)
           {
            outcome_tag = 2;
            profit += (sl - Rates[i].open) * point_cost;
            return(MathPow(DiscFactor, float(n)) * profit);
           }
         if(tp <= Rates[i].high)
           {
            outcome_tag = 1;
            profit += (tp - Rates[i].open) * point_cost;
            return(MathPow(DiscFactor, float(n)) * profit);
           }
//--- The next open is required for the drift leg; without it the horizon
//--- cannot be completed and the example has no resolvable label.
         if(i < 1)
           {
            outcome_tag = 0;
            return(0.0);
           }
//--- Mark-to-market drift from the current bar open to the next bar open.
         profit += (Rates[i - 1].open - Rates[i].open) * point_cost;
         if(-profit >= balance)
           {
            outcome_tag = 3;
            return(MathPow(DiscFactor, float(n)) * profit - 1000.0);
           }
        }
//--- Neither level reached before H: discounted Equity change over the period.
      outcome_tag = 3;
      return(MathPow(DiscFactor, float(steps)) * profit);
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
      sl = Rates[start_position].open + (sl + spread) * Symb.Point();
      profit = -spread * Symb.Point() * point_cost;
      for(int n = 0; n < steps; n++)
        {
         const int i = int(start_position) - n;
         if(i < 0)
           {
            outcome_tag = 0;
            return(0.0);
           }
//--- The last sell leg is (O_i - SL): a stop closes ABOVE the open, so the
//--- realized amount reduces profit.  (SL - O_i) had added a gain.
         if(sl <= Rates[i].high)
           {
            outcome_tag = 2;
            profit += (Rates[i].open - sl) * point_cost;
            return(MathPow(DiscFactor, float(n)) * profit);
           }
         if(tp >= Rates[i].low)
           {
            outcome_tag = 1;
            profit += (Rates[i].open - tp) * point_cost;
            return(MathPow(DiscFactor, float(n)) * profit);
           }
         if(i < 1)
           {
            outcome_tag = 0;
            return(0.0);
           }
         profit += (Rates[i].open - Rates[i - 1].open) * point_cost;
         if(-profit >= balance)
           {
            outcome_tag = 3;
            return(MathPow(DiscFactor, float(n)) * profit - 1000.0);
           }
        }
      outcome_tag = 3;
      return(MathPow(DiscFactor, float(steps)) * profit);
     }
//---
   return(0);
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
//| Skill ScenarioForecast and Actor/Critic runtime                  |
//+------------------------------------------------------------------+
bool                 SkillRecoverySmoke = false;
uint                 SkillRecoverySmokeBatches = 2000;
uint                 SkillRecoverySmokeAge = 512;
const uint           Skill_FORMAT_VERSION = 7;
const string         Skill_MARKET_FILE = "ACSRMMarket.nnw";
const string         Skill_TARGET_FILE = "ACSRMTarget.nnw";
const string         Skill_MANIFEST_FILE = "ACSRMForecast.manifest";
const string         Skill_ACTOR_FILE = "ACSRMActor.nnw";
const string         Skill_Q1_FILE = "ACSRMQ1.nnw";
const string         Skill_Q2_FILE = "ACSRMQ2.nnw";
const string         Skill_AC_MANIFEST_FILE = "ACSRMActorCritic.manifest";
const string         Skill_ACTOR_NEXT_FILE = "ACSRMActor.next.nnw";
const string         Skill_Q1_NEXT_FILE = "ACSRMQ1.next.nnw";
const string         Skill_Q2_NEXT_FILE = "ACSRMQ2.next.nnw";
const string         Skill_AC_MANIFEST_NEXT_FILE = "ACSRMActorCritic.next.manifest";
const string         Skill_ACTOR_PREVIOUS_FILE = "ACSRMActor.previous.nnw";
const string         Skill_Q1_PREVIOUS_FILE = "ACSRMQ1.previous.nnw";
const string         Skill_Q2_PREVIOUS_FILE = "ACSRMQ2.previous.nnw";
const string         Skill_AC_MANIFEST_PREVIOUS_FILE = "ACSRMActorCritic.previous.manifest";
const string         Skill_AC_TRANSACTION_FILE = "ACSRMActorCritic.transaction";
const uint           Skill_AC_FORMAT_VERSION = 14;
const uint           Skill_AC_TRANSACTION_VERSION = 1;
const uint           Skill_AC_CONFIG_VERSION = 9;
const string         Skill_OOS_FILE = "ACSRMOOS.csv";
const string         Skill_OOS_MANIFEST_FILE = "ACSRMOOS.manifest";
//--- Stage 02 publishes only the canonical production tuple.  A complete
//--- next tuple is validated first; previous is retained only until reload
//--- proof accepts the canonical manifest as the commit point.
const string         ACSRM_STAGE02_MARKET_NEXT_FILE = "ACSRMMarket.next.nnw";
const string         ACSRM_STAGE02_TARGET_NEXT_FILE = "ACSRMTarget.next.nnw";
const string         ACSRM_STAGE02_MANIFEST_NEXT_FILE = "ACSRMForecast.next.manifest";
const string         ACSRM_STAGE02_MARKET_PREVIOUS_FILE = "ACSRMMarket.previous.nnw";
const string         ACSRM_STAGE02_TARGET_PREVIOUS_FILE = "ACSRMTarget.previous.nnw";
const string         ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE = "ACSRMForecast.previous.manifest";
//--- Stage 01 uses an independent transaction namespace.  The short names
//--- remain the only normal load contract; manifest is its commit record.
const string         ACSRM_STAGE01_MARKET_NEXT_FILE = "ACSRMMarket.stage01.next.nnw";
const string         ACSRM_STAGE01_TARGET_NEXT_FILE = "ACSRMTarget.stage01.next.nnw";
const string         ACSRM_STAGE01_MANIFEST_NEXT_FILE = "ACSRMForecast.stage01.next.manifest";
const string         ACSRM_STAGE01_MARKET_PREVIOUS_FILE = "ACSRMMarket.stage01.previous.nnw";
const string         ACSRM_STAGE01_TARGET_PREVIOUS_FILE = "ACSRMTarget.stage01.previous.nnw";
const string         ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE = "ACSRMForecast.stage01.previous.manifest";
const string         ACSRM_STAGE01_TRANSACTION_FILE = "ACSRMForecast.stage01.transaction";
const uint           ACSRM_STAGE01_TRANSACTION_VERSION = 1;
//--- Legacy generation selectors are migration markers only.  They are never
//--- resolved by a normal loader, which fails closed until an explicit migration.
const string         ACSRM_STAGE02_ACTIVE_SELECTOR = "ACSRMForecast.stage02.active";
const string         ACSRM_STAGE02_SELECTOR_NEXT = "ACSRMForecast.stage02.active.next";
const string         ACSRM_STAGE02_SELECTOR_PREVIOUS = "ACSRMForecast.stage02.active.previous";
const string         ACSRM_STAGE02_SELECTOR_RESTORE = "ACSRMForecast.stage02.active.restore";
const string         ACSRM_STAGE02_GENERATION_PREFIX = ".stage02.";
const string         ACSRM_STAGE02_LEGACY_TRANSACTION_FILE = "ACSRMForecast.stage02.transaction";
const string         ACSRM_STAGE02_LEGACY_TRANSACTION_NEXT_FILE = "ACSRMForecast.stage02.transaction.next";
const uint           ACSRM_STAGE02_SELECTOR_VERSION = 1;
//--- Normal loading always uses these canonical physical production names.
string               SkillActiveMarketFile = Skill_MARKET_FILE;
string               SkillActiveTargetFile = Skill_TARGET_FILE;
string               SkillActiveManifestFile = Skill_MANIFEST_FILE;
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Skill builds transitions directly from historical windows Base  CNet already provides Layer(int) and  FeedForwardLayer(CNeuronBaseOCL*). A transient, non-owning layer  view lets the library transpose  |
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
class CD2SkillBufferView : public CNeuronBaseOCL
  {
public:
   bool              Bind(CBufferFloat *source);
   //--- Implements Unbind.
   void              Unbind(void)
     {
      Output = NULL;
     }
  };
//+------------------------------------------------------------------+
//| Implements Bind.                                                 |
//+------------------------------------------------------------------+
bool CD2SkillBufferView::Bind(CBufferFloat *source)
  {
   if(!source || source.GetIndex() < 0)
      ReturnFalse;
   if(Output != source)
      DeleteObj(Output);
   Output = source;
   return(true);
  }
//+------------------------------------------------------------------+
//| Device-only composition adapter, all operations reuse existing   |
//+------------------------------------------------------------------+
class CD2SkillDeviceOps : public CNeuronBaseOCL
  {
public:
   //--- Implements Bind.
   bool              Bind(COpenCLMy *open_cl)
     {
      OpenCL = open_cl;
      return(CheckPointer(OpenCL) != POINTER_INVALID);
     }
   bool              Copy(CBufferFloat *source, CBufferFloat *destination, const uint total);
   bool              Join2(CBufferFloat *first, const uint first_total, CBufferFloat *second,
                           const uint second_total, CBufferFloat *dest);
   bool              Join4(CBufferFloat *first, CBufferFloat *second, CBufferFloat *third,
                           CBufferFloat *fourth, CBufferFloat *destination, const uint block);
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
   //--- Public device wrappers over the protected base element ops.
   bool              IdentDifferenceOnDevice(CBufferFloat *tensor, CBufferFloat *out,
                                             const int dimension)
     {
      return(IdentDifferent(tensor, out, dimension));
     }
   bool              ElementMultiplyOnDevice(CBufferFloat *first, CBufferFloat *second,
                                             CBufferFloat *out)
     {
      return(ElementMult(first, second, out));
     }
   //--- Elementwise activation-derivative scaling: inputs_gr[i] =  Deactivation(output_gr[i], inputs[i], activation).  For SIGMOID  every component gets its OWN (1 - a_i) factor (elementwise), not  an identity-row subtraction like IdentDifferent.
   bool              DeActivationOnDevice(CBufferFloat *inputs, CBufferFloat *inputs_gr,
                                          CBufferFloat *output_gr, const int activat)
     {
      return(DeActivation(inputs, inputs_gr, output_gr, activat));
     }
  };
//+------------------------------------------------------------------+
//| Creates and manages the Copy device operation.                   |
//+------------------------------------------------------------------+
bool CD2SkillDeviceOps::Copy(CBufferFloat *source, CBufferFloat *destination, const uint total)
  {
   if(!source || !destination || source.Total() != int(total) ||
      destination.Total() != int(total) || source.GetIndex() < 0 ||
      destination.GetIndex() < 0 || source.GetOpenCL() != OpenCL ||
      destination.GetOpenCL() != OpenCL)
      ReturnFalse;
//---
   return(CopyBufferRaw(source, destination, total));
  }
//+------------------------------------------------------------------+
//| Concatenates two device buffers into one destination.            |
//+------------------------------------------------------------------+
bool CD2SkillDeviceOps::Join2(CBufferFloat *first, const uint first_total, CBufferFloat *second,
                              const uint second_total, CBufferFloat *dest)
  {
   if(!first || !second || !dest || first.Total() != int(first_total) ||
      second.Total() != int(second_total) ||
      dest.Total() != int(first_total + second_total) || first.GetIndex() < 0 ||
      second.GetIndex() < 0 || dest.GetIndex() < 0)
      ReturnFalse;
   return(Concat(first, second, dest, first_total, second_total, 1));
  }
//+------------------------------------------------------------------+
//| Concatenates four device buffers into one destination.           |
//+------------------------------------------------------------------+
bool CD2SkillDeviceOps::Join4(CBufferFloat *first, CBufferFloat *second, CBufferFloat *third,
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
//---
CNet                 SkillMarket;
CNet                 SkillTarget;
CNeuronScenarioForecast *SkillForecast = NULL;
//+------------------------------------------------------------------+
//| Resolves RankTCM at its fixed ACSRM graph slot.                  |
//+------------------------------------------------------------------+
CNeuronBaseOCL *GetRankTCM(void)
  {
//--- Resolve RankTCM only when the fixed ACSRM graph slot has its declared type.
   CNeuronBaseOCL *layer = SkillMarket.Layer(4);
//--- Return the typed predecessor or NULL for a stale or incompatible graph.
   return(layer && layer.Type() == defNeuronCogDriverRankTCM ? layer : NULL);
  }
//+------------------------------------------------------------------+
//| Resolves ACSRM at its fixed MarketEncoder graph slot.            |
//+------------------------------------------------------------------+
CNeuronOMPBOCL *GetACSRM(void)
  {
//--- Resolve ACSRM only when the fixed MarketEncoder graph slot has its declared type.
   CNeuronBaseOCL *layer = SkillMarket.Layer(5);
//--- Return the typed layer or NULL for a stale or incompatible graph.
   return(layer && layer.Type() == defNeuronOMPBOCL ? (CNeuronOMPBOCL *)layer : NULL);
  }
//+------------------------------------------------------------------+
//| Sets an ACSRM runtime mode after clearing source-anchor state.   |
//+------------------------------------------------------------------+
bool SetACSRMMode(const ENUM_OMPB_MODE mode)
  {
//--- Clear source-anchor state before selecting the requested ACSRM runtime mode.
   CNeuronOMPBOCL *layer = GetACSRM();
//--- Finalize the mode change only for a validated ACSRM layer.
   return(layer && layer.SetSourceAnchor(false) && layer.SetMode(mode));
  }
//+------------------------------------------------------------------+
//| Configures ACSRM mode and BYPASS training safeguards.            |
//+------------------------------------------------------------------+
bool ConfigureACSRM(const ENUM_OMPB_MODE mode)
  {
//--- Resolve the layer and install the requested ACSRM runtime mode.
   CNeuronOMPBOCL *layer = GetACSRM();
   if(!layer || !SetACSRMMode(mode))
      ReturnFalse;
//--- BYPASS must not retain a trainable ACSRM layer during MarketEncoder training.
   if(mode == OMPB_BYPASS)
      layer.TrainMode(false);
//--- Finalize the ACSRM configuration after the runtime safeguards are applied.
   return(true);
  }
//+------------------------------------------------------------------+
//| Resets only the ACSRM immutable Reference history.               |
//+------------------------------------------------------------------+
bool ResetACSRMReference(void)
  {
//--- Resolve ACSRM before discarding its immutable Reference history explicitly.
   CNeuronOMPBOCL *layer = GetACSRM();
//--- Finalize only when the validated layer reset succeeds.
   return(layer && layer.ResetReference());
  }
//+------------------------------------------------------------------+
//| Reads ACSRM runtime diagnostic counters and regularizer values.  |
//+------------------------------------------------------------------+
bool ReadACSRMDiagnostics(uint &reference_count, uint &current_count,
                         float &disagreement, float &kl, float &alpha_prior,
                         uint &invalid_fallbacks, uint &kl_rejects)
  {
//--- Resolve ACSRM before exporting its runtime-only diagnostic counters.
   CNeuronOMPBOCL *layer = GetACSRM();
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
//| Checks that BYPASS leaves ACSRM diagnostics unchanged.           |
//+------------------------------------------------------------------+
bool SkillCheckACSRMBypassInvariant(const uint reference_count_before,
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
   if(!ReadACSRMDiagnostics(reference_count_after, current_count_after, disagreement_after,
                           kl_after, alpha_prior_after, invalid_fallbacks_after, kl_rejects_after))
     {
      PrintFormat("ACSRM_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=%s reason=read_after", scope);
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
      PrintFormat("ACSRM_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=%s " +
                  "before=(%u,%u,%.9g,%.9g,%.9g,%u,%u) " +
                  "after=(%u,%u,%.9g,%.9g,%.9g,%u,%u)",
                  scope, reference_count_before, current_count_before, disagreement_before,
                  kl_before, alpha_prior_before, invalid_fallbacks_before, kl_rejects_before,
                  reference_count_after, current_count_after, disagreement_after, kl_after,
                  alpha_prior_after, invalid_fallbacks_after, kl_rejects_after);
      return(false);
     }
   PrintFormat("ACSRM_STAGE01_BYPASS_DIAGNOSTIC_PASS scope=%s state_unchanged=true", scope);
   return(true);
  }
//---
CBufferFloat         SkillState;
CBufferFloat         SkillTime;
CBufferFloat         SkillFuture;
CBufferFloat         SkillLatentTarget;
CBufferFloat         SkillLatentDelta;
CBufferFloat         SkillProbeState;
CBufferFloat         SkillProbeTime;
CD2SkillBufferView     SkillFutureView;
CNeuronTransposeRCDOCL SkillFutureTranspose;
CBufferFloat         SkillLatentZero;
CBufferFloat         SkillLatentNegativeMarket;
//---
ulong                SkillInvalidBatches = 0;
ulong                SkillBatches = 0;
ulong                SkillResponsibilityMicroseconds = 0;
uint                 SkillCompletedEpochs = 0;
ulong                SkillLastSignature = 0;
ulong                SkillProductionBaseFingerprint = 0;
ulong                SkillProductionACSRMFingerprint = 0;
bool                 SkillProductionSignatureReady = false;
bool                 SkillReady = false;
CBufferFloat         SkillFrozenGeneratorWeights;
CBufferFloat         SkillFrozenRouterWeights;
CBufferFloat         SkillFrozenConfidenceWeights;
CBufferFloat         SkillFrozenPrototypes;
CBufferFloat         SkillFrozenEMASums;
CBufferFloat         SkillFrozenEMACounts;
CBufferFloat         SkillFrozenUsage;
CBufferFloat         SkillFrozenInactive;
CBufferFloat         SkillFrozenInactivityAge;
bool                 SkillFrozenBaselineReady = false;
CD2SkillDeviceOps      SkillDevice;
//--- Deal-evaluation horizon (bars) pinned by each Stage 03 expert from its
//--- own InpDealHorizon input.  The manifest writer and validator share this
//--- value so a later stage that rewrites the tuple preserves the fixed key.
int                    SkillManifestDealHorizon = 24;
double                 SkillManifestOnlineDiscount = 0.5;
double                 SkillManifestTargetTau = 0.0;
int                    SkillManifestTargetUpdatePeriod = 0;
//--- Manifest-quantiles symmetry (rank-sensitive SRM rearrangement, canonical N=32)
uint                   SkillManifestQuantiles = 32;
//--- Device scratch tensors for the final-SIGMOID policy gradient conversion.
CBufferFloat           ActorDerivOneMinus;
CBufferFloat           ActorDeriv;
CBufferFloat           ActorGradScratch;
//+------------------------------------------------------------------+
//| Configures optional forecast recovery diagnostics.               |
//+------------------------------------------------------------------+
bool SkillConfigureForecastRun(const bool recovery_smoke,
                                 const uint recovery_batches, const uint recovery_age)
  {
   if(recovery_smoke && (recovery_batches == 0 || recovery_age == 0))
      ReturnFalse;
   SkillRecoverySmoke = recovery_smoke;
   SkillRecoverySmokeBatches = recovery_batches;
   SkillRecoverySmokeAge = recovery_age;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements ForecastRecoveryAge.                                  |
//+------------------------------------------------------------------+
uint ForecastRecoveryAge(void)
  {
   if(SkillRecoverySmoke)
      return(MathMax(1, int(SkillRecoverySmokeAge)));
   const int seconds = PeriodSeconds(TimeFrame);
   if(seconds <= 0)
      return(6240);
//--- 52 five-day trading weeks: one calendar-equivalent year on H1.
   return(uint)MathMax(1.0, MathRound(6240.0 * PeriodSeconds(PERIOD_H1) / seconds));
  }
//+-------------------------------------------------------------------+
//| Creates and manages object lifecycle for ConfigureForecastRec...  |
//+-------------------------------------------------------------------+
bool ConfigureForecastRecoveryAge(void)
  {
   if(!SkillForecast || !SkillForecast.SetRecoveryAge(ForecastRecoveryAge()))
      ReturnFalse;
   PrintFormat("%s %s recovery_age=%u bars", ACSRM_LOG_PREFIX, (SkillRecoverySmoke ? "smoke" : "forecast"),
               SkillForecast.RecoveryAge());
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillInitTrainingBuffers.                             |
//+------------------------------------------------------------------+
bool SkillInitTrainingBuffers(void)
  {
   COpenCLMy *open_cl = SkillMarket.GetOpenCL();
   if(!open_cl || !SkillDevice.Bind(open_cl))
      ReturnFalse;
   if(!SkillFutureTranspose.Init(0, 0, open_cl, NForecast, BarDescr, 1, ADAM, BatchSize) ||
      !SkillFuture.BufferInit((NForecast * BarDescr), 0) ||
      !SkillFuture.BufferCreate(open_cl) ||
      !SkillLatentZero.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !SkillLatentZero.BufferCreate(open_cl) ||
      !SkillLatentNegativeMarket.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !SkillLatentNegativeMarket.BufferCreate(open_cl) ||
      !SkillLatentTarget.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !SkillLatentTarget.BufferCreate(open_cl) ||
      !SkillLatentDelta.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !SkillLatentDelta.BufferCreate(open_cl))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillAddBase.                                         |
//+------------------------------------------------------------------+
bool SkillAddBase(CArrayObj *description, const uint count)
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
//| Implements SkillNormalizeDescriptionBatch.                       |
//+------------------------------------------------------------------+
bool SkillNormalizeDescriptionBatch(CArrayObj *description)
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
//| Implements SkillCreateDescriptions.                              |
//+------------------------------------------------------------------+
bool SkillCreateDescriptions(CArrayObj *&market, CArrayObj *&target)
  {
   CArrayObj *legacy_decoder = NULL;
//--- Function if.
   if(!CreateStateDescriptions(market, legacy_decoder))
     {
      DeleteObj(legacy_decoder);
      ReturnFalse;
     }
   DeleteObj(legacy_decoder);
//--- Keep exactly CogDriver layers 0..4 (RankTCM is the boundary).
   int layer = market.Total() - 1;
//--- Function while.
   while(layer >= 0)
     {
      CLayerDescription* descr = market.At(layer);
      //--- Function if.
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
   if(!SkillNormalizeDescriptionBatch(market))
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
   descr.layers = ACSRMSamples;
     {
      uint units[] = {ACSRMReferenceSize, ACSRMCurrentWindow};
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
   if(!SkillAddBase(target, (BarDescr * NForecast)))
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
//| Implements SkillCreateNetworks.                                  |
//+------------------------------------------------------------------+
bool SkillCreateNetworks(void)
  {
   CArrayObj *market = NULL, *target = NULL;
//--- Function if.
   if(!SkillCreateDescriptions(market, target))
     {
      DeleteObj(market);
      DeleteObj(target);
      ReturnFalse;
     }
   const bool market_created = SkillMarket.Create(market);
   PrintFormat("%s Create market=%s batch=%u", ACSRM_LOG_PREFIX, (market_created ? "OK" : "FAIL"), uint(BatchSize));
   const bool target_created = (market_created && SkillTarget.Create(target));
   PrintFormat("%s Create target=%s", ACSRM_LOG_PREFIX, (target_created ? "OK" : "FAIL"));
   const bool created = (market_created && target_created);
   DeleteObj(market);
   DeleteObj(target);
   if(!created)
      ReturnFalse;
//--- CNet::Create initializes its OpenCL program.  Build each network first,
//--- then move its device buffers to the Market context; otherwise a later
//--- Create invalidates buffers belonging to an earlier network.
   if(!SkillTarget.SetOpenCLChecked(SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
//--- Function if.
   if(!SkillInitTrainingBuffers())
     {
      PrintFormat("%s init: training buffers=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   SkillForecast = (CNeuronScenarioForecast*)SkillMarket.Layer(-1);
//--- Function if.
   if(!SkillForecast || SkillForecast.Type() != defNeuronScenarioForecast)
     {
      PrintFormat("%s init: forecast layer=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
   SkillTarget.TrainMode(false);
   SkillMarket.TrainMode(true);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ConfigureACSRM(OMPB_BYPASS))
      ReturnFalse;
   ompb.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillValidateShapes.                                  |
//+------------------------------------------------------------------+
bool SkillValidateShapes(const bool training = true)
  {
   CNeuronBaseOCL *layer = SkillMarket.Layer(0);
   CBufferFloat *buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != HistoryBars * BarDescr)
     {
      PrintFormat("%s shape: market input expected=%d actual=%d", ACSRM_LOG_PREFIX, HistoryBars * BarDescr,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   layer = SkillMarket.Layer(4);
   buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("%s shape: market RankTCM expected=%d actual=%d", ACSRM_LOG_PREFIX, BarDescr * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   layer = GetACSRM();
   buffer = (layer ? layer.getOutput() : NULL);
   if(!buffer || buffer.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("ACSRM shape: bridge expected=%d actual=%d", BarDescr * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
//--- Function if.
   if(training)
     {
      layer = SkillTarget.Layer(0);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: target input expected=%d actual=%d", ACSRM_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = SkillTarget.Layer(1);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: target PeriodNorm expected=%d actual=%d", ACSRM_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = SkillTarget.Layer(2);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target Conv3 expected=%d actual=%d", ACSRM_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = SkillTarget.Layer(3);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * 2 * EmbeddingSize))
        {
         PrintFormat("%s shape: target Conv2 expected=%d actual=%d", ACSRM_LOG_PREFIX,
                     BarDescr * NForecast * 2 * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = SkillTarget.Layer(4);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target final Conv expected=%d actual=%d", ACSRM_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = SkillTarget.Layer(-1);
      buffer = (layer ? layer.getOutput() : NULL);
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("%s shape: target output PeriodNorm expected=%d actual=%d", ACSRM_LOG_PREFIX,
                     BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      buffer = SkillFutureTranspose.getOutput();
      //--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("%s shape: future transpose expected=%d actual=%d", ACSRM_LOG_PREFIX, BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
     }
   layer = SkillMarket.Layer(-1);
   buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize)
     {
      PrintFormat("%s shape: Scenario Z expected=%d actual=%d", ACSRM_LOG_PREFIX,
                  NScenarios * BarDescr * NForecast * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = SkillForecast.GetU();
//--- Function if.
   if(!buffer || buffer.Total() != (NScenarios * BarDescr * NForecast))
     {
      PrintFormat("%s shape: Scenario U expected=%d actual=%d", ACSRM_LOG_PREFIX, NScenarios * BarDescr * NForecast,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = SkillForecast.GetPi();
//--- Function if.
   if(!buffer || buffer.Total() != NScenarios)
     {
      PrintFormat("%s shape: Scenario Pi expected=%d actual=%d", ACSRM_LOG_PREFIX, NScenarios,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   if(SkillForecast.Variables() != BarDescr || SkillForecast.Scenarios() != NScenarios ||
      SkillForecast.Horizon() != NForecast || SkillForecast.Dimension() != EmbeddingSize ||
//--- Function ActiveTrajectories.
      SkillForecast.ActiveTrajectories() > TopK)
     {
      PrintFormat("%s shape: Forecast V=%d/%d K=%d/%d H=%d/%d D=%d/%d active=%d/%d", ACSRM_LOG_PREFIX,
                  SkillForecast.Variables(), BarDescr, SkillForecast.Scenarios(), NScenarios,
                  SkillForecast.Horizon(), NForecast, SkillForecast.Dimension(), EmbeddingSize,
                  SkillForecast.ActiveTrajectories(), TopK);
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillHashUInt.                                        |
//+------------------------------------------------------------------+
ulong SkillHashUInt(ulong hash, const ulong value)
  {
   hash ^= value;
   return(hash * ulong(1099511628211));
  }
//+------------------------------------------------------------------+
//| Implements SkillHashText.                                        |
//+------------------------------------------------------------------+
ulong SkillHashText(ulong hash, const string text)
  {
   for(int i = 0; i < StringLen(text); i++)
      hash = SkillHashUInt(hash, (ulong)StringGetCharacter(text, i));
   return(hash);
  }
//+------------------------------------------------------------------+
//| Implements SkillHashFile.                                        |
//+------------------------------------------------------------------+
ulong SkillHashFile(ulong hash, const string file_name)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return(0);
   const ulong length = FileSize(handle);
   if(length == 0 || length > ulong(INT_MAX))
     { FileClose(handle); return 0; }
   uchar bytes[];
   if(ArrayResize(bytes, (int)length) != (int)length ||
//--- Read file payload as bytes after successful allocation.
      FileReadArray(handle, bytes, 0, (int)length) != (int)length)
     { FileClose(handle); return 0; }
   FileClose(handle);
   for(int i = 0; i < (int)length; i++)
      hash = SkillHashUInt(hash, (ulong)bytes[i]);
   return(hash);
  }
//+-------------------------------------------------------------------+
//| Exact signature of the unified Market/Scenario inference grap...  |
//+-------------------------------------------------------------------+
ulong SkillForecastSignature(const string market_file = "")
  {
   if(!SkillForecast)
      return(0);
   ulong hash = ulong(1469598103934665603);
   const string checkpoint = (market_file == "" ? SkillActiveMarketFile : market_file);
   hash = SkillHashFile(hash, checkpoint);
   if(hash == 0)
      return(0);
   hash = SkillHashUInt(hash, Skill_FORMAT_VERSION);
   hash = SkillHashUInt(hash, BarDescr);
   hash = SkillHashUInt(hash, NScenarios);
   hash = SkillHashUInt(hash, TopK);
   hash = SkillHashUInt(hash, NForecast);
   hash = SkillHashUInt(hash, EmbeddingSize);
   hash = SkillHashUInt(hash, SkillForecast.ContractSignature());
   hash = SkillHashText(hash, "z_layout=K,V,H,D;u_layout=K,V,H;pi_layout=K;codebook_layout=K,V,H,D");
   hash = SkillHashText(hash,
                          "market_layout=RankTCM_then_ACSRM_then_ScenarioForecast;" +
                          "variable_order=BarDescr_feature_series_0_to_8");
   hash = SkillHashText(hash, "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   return(hash);
  }
//+------------------------------------------------------------------+
//| Implements SkillForecastTrainingSignature.                       |
//+------------------------------------------------------------------+
ulong SkillForecastTrainingSignature(const ulong forecast_signature, const string target_file = "")
  {
   if(forecast_signature == 0)
      return(0);
   const string checkpoint = (target_file == "" ? SkillActiveTargetFile : target_file);
   return(SkillHashFile(forecast_signature, checkpoint));
  }
//+------------------------------------------------------------------+
//| Implements SkillWriteManifest.                                   |
//+------------------------------------------------------------------+
bool SkillWriteManifest(const uint completed_epochs)
  {
   if(!SkillForecast)
      ReturnFalse;
   const ulong signature = SkillForecastSignature(Skill_MARKET_FILE);
   const ulong target_hash = SkillHashFile(ulong(1469598103934665603), Skill_TARGET_FILE);
   const ulong training_signature = SkillForecastTrainingSignature(signature, Skill_TARGET_FILE);
   if(signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
   int handle = FileOpen(Skill_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=ACSRM_FORECAST");
   FileWrite(handle, StringFormat("version=%u", Skill_FORMAT_VERSION));
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
   FileWrite(handle, StringFormat("contract_signature=%I64u", SkillForecast.ContractSignature()));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", signature));
   FileWrite(handle, StringFormat("target_hash=%I64u", target_hash));
   FileWrite(handle, StringFormat("training_signature=%I64u", training_signature));
   FileWrite(handle, StringFormat("completed_epochs=%u", completed_epochs));
   FileWrite(handle, StringFormat("training_batches=%I64u", SkillBatches));
   FileWrite(handle, StringFormat("invalid_batches=%I64u", SkillInvalidBatches));
   FileWrite(handle, "normalization=OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   FileClose(handle);
   SkillLastSignature = signature;
   return(true);
  }
bool SkillStage01SaveCheckpoint(const uint completed_epochs);
//+------------------------------------------------------------------+
//| Implements SkillSaveCheckpoint.                                  |
//+------------------------------------------------------------------+
bool SkillSaveCheckpoint(const uint completed_epochs)
  {
   return(SkillStage01SaveCheckpoint(completed_epochs));
  }
//+------------------------------------------------------------------+
//| Implements SkillInitIndicators.                                  |
//+------------------------------------------------------------------+
bool SkillInitIndicators(void)
  {
   return (Symb.Name(_Symbol) && Symb.Refresh() &&
           RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice) &&
           CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice) &&
           ATR.Create(Symb.Name(), TimeFrame, ATRPeriod) &&
           MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice));
  }
//+------------------------------------------------------------------+
//| Loads forecast training data into the Skill caches.              |
//+------------------------------------------------------------------+
bool SkillLoadForecastTraining(void);
//+------------------------------------------------------------------+
//| Captures a frozen forecast baseline for Stage 02 proofs.         |
//+------------------------------------------------------------------+
bool SkillCaptureFrozenForecastBaseline(CNeuronScenarioForecast *forecast);
//+------------------------------------------------------------------+
//| Verifies the frozen forecast baseline is bit-exact.              |
//+------------------------------------------------------------------+
bool SkillVerifyFrozenForecastExact(CNeuronScenarioForecast *forecast);
//+------------------------------------------------------------------+
//| Forwards the Skill forecast chain by one step.                   |
//+------------------------------------------------------------------+
bool SkillForwardForecast(const int position, CBufferFloat *state, CBufferFloat *time,
                          const int shift_bars = 1);
//+------------------------------------------------------------------+
//| Reads one key value from a manifest file.                        |
//+------------------------------------------------------------------+
string SkillManifestValue(const string file_name, const string key);
//+------------------------------------------------------------------+
//| Checks the explicit Stage 02 checkpoint file triplet.            |
//+------------------------------------------------------------------+
bool ACSRMStage02ExplicitCheckpointFilesValid(const string market_file, const string target_file,
      const string manifest_file);
//+------------------------------------------------------------------+
//| Loads the explicit Stage 02 checkpoint file triplet.             |
//+------------------------------------------------------------------+
bool ACSRMStage02LoadExplicitCheckpoint(const string market_file, const string target_file,
                                       const string manifest_file);
//+--------------------------------------------------------------------------+
//| Reloads an explicit Stage 02 checkpoint and proves deterministic  mode.  |
//+--------------------------------------------------------------------------+
bool ACSRMStage02ReloadProofDeterministic(const string market_file, const string target_file,
      const string manifest_file, const int position,
      CBufferFloat *state, CBufferFloat *time);
//+-----------------------------------------------------------------------+
//| Reloads the selected checkpoint and proves deterministic  inference.  |
//+-----------------------------------------------------------------------+
bool ACSRMStage02ReloadCheckpointProof(const string market_file, const string target_file,
                                      const string manifest_file, const ulong expected_market,
                                      const ulong expected_target, const ulong expected_posterior);
//+------------------------------------------------------------------+
//| Finalizes the steady selector after Stage 02 proofs pass.        |
//+------------------------------------------------------------------+
bool ACSRMStage02FinalizeSteadySelector(void);
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Stage 02 preflight and Reference collection only.  This path is  intentionally separate from CreateSkillForecastStudy(): Stage  02 must reject a missing/incompatible Stage 01 checkpoint rather  than creating a new random Market graph.  |
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
datetime             ACSRMStage02ReferenceStart = 0;
datetime             ACSRMStage02ReferenceSplit = 0;
datetime             ACSRMStage02ReferenceEnd = 0;
datetime             ACSRMStage02CalibrationStart = 0;
datetime             ACSRMStage02CalibrationSplit = 0;
datetime             ACSRMStage02CalibrationEnd = 0;
int                  ACSRMStage02ReferenceFirst = -1;
int                  ACSRMStage02ReferenceLast = -1;
int                  ACSRMStage02SourceEvalFirst = -1;
int                  ACSRMStage02SourceEvalLast = -1;
int                  ACSRMStage02TargetCalibrationFirst = -1;
int                  ACSRMStage02TargetCalibrationLast = -1;
int                  ACSRMStage02TargetEvalFirst = -1;
int                  ACSRMStage02TargetEvalLast = -1;
bool                 ACSRMStage02Smoke = false;
uint                 ACSRMStage02SmokeLimit = 0;
uint                 ACSRMStage02SourcePeriod = 0;
uint                 ExtACSRMCalibrationEpochs = 1;
uint                 ExtACSRMCalibrationEpoch = 0;
ulong                ACSRMStage02BaseFingerprint = 0;
ulong                ACSRMStage02ForecastFingerprint = 0;
ulong                ACSRMStage02TargetFingerprint = 0;
ulong                ACSRMStage02PosteriorFingerprint = 0;
bool                 ACSRMStage02SignaturesCaptured = false;
bool                 ACSRMStage02StopReported = false;
//--- The Stage 01 state remains the only production checkpoint until an
//--- accepted Stage 02 transaction has completed its reload smoke.
bool                 ACSRMStage02CheckpointPublished = false;
//+------------------------------------------------------------------------+
//| Checks that every finite value survives the Stage 02 state  encoding.  |
//+------------------------------------------------------------------------+
bool ACSRMStage02Finite(const double value)
  {
   return(MathIsValidNumber(value));
  }
//+------------------------------------------------------------------+
//| Returns whether Stage 02 must terminate after a manual stop.     |
//+------------------------------------------------------------------+
bool ACSRMStage02StopHandled(const string phase = "")
  {
   if(!IsStopped())
      return(false);
//--- Leave both graphs outside training mode without finalizing or saving.
   SkillMarket.TrainMode(false);
   SkillTarget.TrainMode(false);
   if(phase != "" && !ACSRMStage02StopReported)
     {
      PrintFormat("ACSRM_STAGE02_STOP_REQUESTED phase=%s", phase);
      ACSRMStage02StopReported = true;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures Stage 02 runtime constants and mode selectors.        |
//+------------------------------------------------------------------+
bool ACSRMStage02Configure(const datetime reference_start, const datetime reference_end,
                          const datetime calibration_start, const datetime calibration_end,
                          const uint anchor_period, const float tau_value, const float lambda_dis,
                          const float lambda_kl, const float lambda_alpha, const float alpha_prior,
                          const float max_kl, const bool smoke, const uint smoke_limit)
  {
   if(reference_start >= reference_end || reference_end > calibration_start ||
      calibration_start >= calibration_end || tau_value < 0.0f || tau_value > 1.0f ||
      lambda_dis < 0.0f || lambda_kl < 0.0f || lambda_alpha < 0.0f ||
      max_kl < 0.0f ||
      !ACSRMStage02Finite(tau_value) || !ACSRMStage02Finite(lambda_dis) ||
      !ACSRMStage02Finite(lambda_kl) || !ACSRMStage02Finite(lambda_alpha) ||
      !ACSRMStage02Finite(alpha_prior) || !ACSRMStage02Finite(max_kl))
     {
      Print("ACSRM_STAGE02_PREFLIGHT_FAIL reason=input_contract");
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
      Print("ACSRM_STAGE02_PREFLIGHT_FAIL reason=empty_80_20_partition");
      ReturnFalse;
     }
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ompb.SetTau(tau_value) || !ompb.SetDisagreementMult(lambda_dis) ||
      !ompb.SetKLDMult(lambda_kl) || !ompb.SetAlphaMult(lambda_alpha) ||
      !ompb.SetAlphaPrior(alpha_prior) || !ompb.SetMaxKL(max_kl) ||
      !ConfigureACSRM(OMPB_BYPASS))
     {
      Print("ACSRM_STAGE02_PREFLIGHT_FAIL reason=ompb_configuration");
      ReturnFalse;
     }
   ompb.TrainMode(false);
   ACSRMStage02ReferenceStart = reference_start;
   ACSRMStage02ReferenceSplit = reference_split;
   ACSRMStage02ReferenceEnd = reference_end;
   ACSRMStage02CalibrationStart = calibration_start;
   ACSRMStage02CalibrationSplit = calibration_split;
   ACSRMStage02CalibrationEnd = calibration_end;
   ACSRMStage02Smoke = smoke;
   ACSRMStage02SmokeLimit = (smoke ? MathMax(1, int(smoke_limit)) : 0);
   ACSRMStage02SourcePeriod = anchor_period;
   PrintFormat("ACSRM_STAGE02_PREFLIGHT_CONFIG reference=%s..%s split=%s calibration=%s..%s",
               TimeToString(reference_start, TIME_DATE), TimeToString(reference_end, TIME_DATE),
               TimeToString(reference_split, TIME_DATE), TimeToString(calibration_start, TIME_DATE),
               TimeToString(calibration_end, TIME_DATE));
   PrintFormat("ACSRM_STAGE02_PREFLIGHT_CONFIG split=%s source_period=%u smoke=%s",
               TimeToString(calibration_split, TIME_DATE), anchor_period, (smoke ? "true" : "false"));
   PrintFormat("ACSRM_STAGE02_OBJECTIVE tau=%.8g lambda_dis=%.8g lambda_kl=%.8g " +
               "lambda_alpha=%.8g alpha_prior=%.8g max_kl=%.8g",
               tau_value, lambda_dis, lambda_kl, lambda_alpha, alpha_prior, max_kl);
   return(true);
  }
//+------------------------------------------------------------------+
//| Prepares Stage 02 input data and state descriptors.              |
//+------------------------------------------------------------------+
bool ACSRMStage02PrepareData(void)
  {
   const int reference_start = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02ReferenceStart);
   const int reference_split = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02ReferenceSplit);
   const int reference_end = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02ReferenceEnd);
   const int calibration_start = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02CalibrationStart);
   const int calibration_split = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02CalibrationSplit);
   const int calibration_end = iBarShift(Symb.Name(), TimeFrame, ACSRMStage02CalibrationEnd);
   if(reference_start <= 0 || reference_split <= 0 || reference_end <= 0 ||
      calibration_start <= 0 || calibration_split <= 0 || calibration_end <= 0 ||
      reference_start <= reference_split || reference_split <= reference_end ||
      reference_end < calibration_start || calibration_start <= calibration_split ||
      calibration_split <= calibration_end)
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=bar_order shifts=(%d,%d,%d,%d,%d,%d)",
                  reference_start, reference_split, reference_end, calibration_start,
                  calibration_split, calibration_end);
      ReturnFalse;
     }
   const int bars = CopyRates(Symb.Name(), TimeFrame, 0, reference_start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=rates bars=%d error=%d", bars, GetLastError());
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
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=indicators bars=%d error=%d", bars, GetLastError());
      ReturnFalse;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=rates_series error=%d", GetLastError());
      ReturnFalse;
     }
   ACSRMStage02ReferenceFirst = reference_start - HistoryBars - NForecast - 1;
   ACSRMStage02ReferenceLast = reference_split;
   ACSRMStage02SourceEvalFirst = reference_split - 1;
   ACSRMStage02SourceEvalLast = reference_end;
   ACSRMStage02TargetCalibrationFirst = calibration_start - HistoryBars - NForecast - 1;
   ACSRMStage02TargetCalibrationLast = calibration_split;
   ACSRMStage02TargetEvalFirst = calibration_split - 1;
   ACSRMStage02TargetEvalLast = calibration_end;
   const int reference_rows = ACSRMStage02ReferenceFirst - ACSRMStage02ReferenceLast + 1;
   const int source_eval_rows = ACSRMStage02SourceEvalFirst - ACSRMStage02SourceEvalLast + 1;
   const int target_calibration_rows = ACSRMStage02TargetCalibrationFirst - ACSRMStage02TargetCalibrationLast + 1;
   const int target_eval_rows = ACSRMStage02TargetEvalFirst - ACSRMStage02TargetEvalLast + 1;
   if(ACSRMStage02ReferenceFirst < ACSRMStage02ReferenceLast ||
      ACSRMStage02SourceEvalFirst < ACSRMStage02SourceEvalLast ||
      ACSRMStage02TargetCalibrationFirst < ACSRMStage02TargetCalibrationLast ||
      ACSRMStage02TargetEvalFirst < ACSRMStage02TargetEvalLast ||
      reference_rows < int(ACSRMReferenceSize) || source_eval_rows <= 0 ||
      target_calibration_rows < int(ACSRMCurrentWindow) || target_eval_rows <= 0)
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=bars ref=%d source=%d calibration=%d target=%d",
                  reference_rows, source_eval_rows, target_calibration_rows, target_eval_rows);
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_REQUIRED reference=%u target_calibration=%u",
                  ACSRMReferenceSize, ACSRMCurrentWindow);
      ReturnFalse;
     }
   PrintFormat("ACSRM_STAGE02_PREFLIGHT_PASS bars=%d reference=%d source_eval=%d target_calibration=%d target_eval=%d",
               bars, reference_rows, source_eval_rows, target_calibration_rows, target_eval_rows);
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the valid-row quota for one Stage 02 progress phase.     |
//+------------------------------------------------------------------+
uint ACSRMStage02ProgressQuota(const uint rows, const uint cap, const bool smoke,
                              const uint smoke_limit)
  {
   uint quota = (cap == 0 ? rows : (uint)MathMin(rows, cap));
   if(smoke)
      quota = (uint)MathMin(quota, smoke_limit);
   return(quota);
  }
//+------------------------------------------------------------------+
//| Converts valid progress into a bounded Stage 02 percentage.      |
//+------------------------------------------------------------------+
double ACSRMStage02ProgressPercent(const uint done, const uint quota)
  {
   if(quota == 0)
      return(0.0);
   return(100.0 * double(MathMin(done, quota)) / double(quota));
  }
//+------------------------------------------------------------------+
//| Shows a throttled Stage 02 chart status without training work.   |
//+------------------------------------------------------------------+
void ACSRMStage02ShowProgress(const string phase, const uint done, const uint total,
                             const uint attempts, const uint invalid, const bool end,
                             const bool successful, const bool show_epoch,
                             const bool show_ompb, const uint source_done,
                             const uint source_attempts, const uint source_invalid,
                             const string detail)
  {
   static ulong  last_tick = 0;
   static string last_phase = "";
   const ulong now = GetTickCount64();
   const bool phase_changed = (phase != last_phase);
   const bool timer_elapsed = (!phase_changed && last_tick > 0 && now - last_tick >= 1000);
   const bool read_metrics = (!phase_changed && (timer_elapsed || end));
   if(!phase_changed && !end && !timer_elapsed)
      return;
   string lmix_text = "n/a";
   string valid_text = "n/a";
   string diagnostic_invalid_text = "n/a";
   string disagreement_text = "n/a";
   string kl_text = "n/a";
//--- A phase transition deliberately hides the preceding phase's epoch metrics.
   if(read_metrics && show_epoch)
     {
      double lmix, router, trajectory, confidence, latent, observation;
      double valid, diagnostic_invalid, entropy, distance, inactive, recovered;
      if(SkillForecast && SkillForecast.ReadEpochDiagnostics(
            lmix, router, trajectory, confidence, latent, observation,
            valid, diagnostic_invalid, entropy, distance, inactive, recovered) &&
         valid > 0.0 && ACSRMStage02Finite(lmix) && ACSRMStage02Finite(valid) &&
         ACSRMStage02Finite(diagnostic_invalid))
        {
         lmix_text = StringFormat("%.8f", lmix / valid);
         valid_text = StringFormat("%.0f", valid);
         diagnostic_invalid_text = StringFormat("%.0f", diagnostic_invalid);
        }
      if(show_ompb)
        {
         uint reference_count, current_count, invalid_fallbacks, kl_rejects;
         float disagreement, kl, alpha_prior;
         if(ReadACSRMDiagnostics(reference_count, current_count, disagreement, kl, alpha_prior,
                                invalid_fallbacks, kl_rejects) && ACSRMStage02Finite(disagreement) &&
            ACSRMStage02Finite(kl))
           {
            disagreement_text = StringFormat("%.8f", disagreement);
            kl_text = StringFormat("%.8f", kl);
           }
        }
     }
   const double percent = ACSRMStage02ProgressPercent(done, total);
   string state = "running";
   if(end)
     {
      state = (successful && total > 0 && done >= total ? "complete" : "ended");
      if(IsStopped())
         state = "stopped";
     }
   const string phase_label = (phase == "calibration" ?
                               StringFormat("%s epoch %u/%u", phase,
                                     ExtACSRMCalibrationEpoch, ExtACSRMCalibrationEpochs) : phase);
   Comment(StringFormat("%s Stage02 %s %6.2f%% state=%s\n" +
                        "processed %u/%u attempts %u invalid %u L_mix %s valid %s invalid %s\n" +
                        "KL %s disagreement %s source_updates %u source_attempts %u source_invalid %u\n%s",
                        ACSRM_LOG_PREFIX, phase_label, percent, state, done, total, attempts, invalid,
                        lmix_text, valid_text, diagnostic_invalid_text, kl_text, disagreement_text,
                        source_done, source_attempts, source_invalid, detail));
   if(end && IsStopped())
      PrintFormat("ACSRM_STAGE02_PROGRESS_STOP phase=%s valid=%u total=%u attempts=%u invalid=%u",
                  phase, done, total, attempts, invalid);
   last_phase = phase;
   last_tick = now;
  }
//+------------------------------------------------------------------+
//| Collects the immutable Stage 02 Reference history.               |
//+------------------------------------------------------------------+
bool ACSRMStage02CollectReference(void)
  {
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ResetACSRMReference() || !ConfigureACSRM(OMPB_REFERENCE))
     {
      Print("ACSRM_STAGE02_REFERENCE_FAIL reason=setup");
      ReturnFalse;
     }
   ompb.TrainMode(false);
   uint collected = 0;
   const uint rows = uint(ACSRMStage02ReferenceFirst - ACSRMStage02ReferenceLast + 1);
   const uint quota = ACSRMStage02ProgressQuota(rows, ACSRMReferenceSize, ACSRMStage02Smoke,
                      ACSRMStage02SmokeLimit);
   ACSRMStage02ShowProgress("reference", collected, quota, collected, 0, false, false, false,
                           false,
                           0, 0, 0, "");
   for(int position = ACSRMStage02ReferenceFirst;
       position >= ACSRMStage02ReferenceLast && !IsStopped(); position--)
     {
      if(!CreateBuffers(position + NForecast, GetPointer(SkillState),
                        GetPointer(SkillTime), NULL) ||
         !SkillMarket.feedForward(GetPointer(SkillState), 1, false, (CBufferFloat *)NULL))
        {
         PrintFormat("ACSRM_STAGE02_REFERENCE_FAIL reason=forward position=%d line=%d", position, __LINE__);
         ReturnFalse;
        }
      collected++;
      ACSRMStage02ShowProgress("reference", collected, quota, collected, 0, false, false, false,
                              false,
                              0, 0, 0, "");
      if(ACSRMStage02Smoke && collected >= ACSRMStage02SmokeLimit)
         break;
      if(ompb.ReferenceCount() >= ACSRMReferenceSize)
         break;
     }
//--- A partial Reference is never a successful collection.
   if(IsStopped())
     {
      Print("ACSRM_STAGE02_STOP_REQUESTED phase=reference");
      ACSRMStage02StopReported = true;
      return(false);
     }
   if(collected == 0 || (!ACSRMStage02Smoke && ompb.ReferenceCount() != ACSRMReferenceSize))
     {
      PrintFormat("ACSRM_STAGE02_REFERENCE_FAIL reason=count collected=%u reference=%u required=%u",
                  collected, ompb.ReferenceCount(), ACSRMReferenceSize);
      ACSRMStage02ShowProgress("reference", collected, quota, collected, 0, true, false, false,
                              false,
                              0, 0, 0, "result=failed");
      ReturnFalse;
     }
   ACSRMStage02ShowProgress("reference", collected, quota, collected, 0, true, true, false,
                           false,
                           0, 0, 0, "");
   PrintFormat("ACSRM_STAGE02_REFERENCE_PASS collected=%u reference=%u capacity=%u smoke=%s",
               collected, ompb.ReferenceCount(), ACSRMReferenceSize, (ACSRMStage02Smoke ? "true" : "false"));
   return(ConfigureACSRM(OMPB_BYPASS));
  }
//+------------------------------------------------------------------+
//| Computes the frozen BYPASS baseline for one evaluation scope.    |
//+------------------------------------------------------------------+
bool ACSRMStage02Baseline(const int first, const int last, const string scope,
                         double &forecast_loss, uint &valid_batches, uint &invalid_batches)
  {
   forecast_loss = 0.0;
   valid_batches = 0;
   invalid_batches = 0;
   uint attempts = 0;
   const uint rows = uint(first - last + 1);
   const uint quota = ACSRMStage02ProgressQuota(rows, 0, ACSRMStage02Smoke,
                      ACSRMStage02SmokeLimit);
   const string progress_phase = (StringFind(scope, "source") >= 0 ?
                                  "baseline source" : "baseline target");
   if(first < last || !SkillForecast || !ConfigureACSRM(OMPB_BYPASS) ||
      !SkillMarket.Clear() || !SkillTarget.Clear() ||
      !SkillForecast.ResetEpochDiagnostics())
     {
      PrintFormat("ACSRM_STAGE02_BASELINE_FAIL scope=%s reason=setup", scope);
      ReturnFalse;
     }
   ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                           quota, attempts, invalid_batches, false, false, true, false,
                           0, 0, 0, "");
   for(int position = first; position >= last && !IsStopped(); position--)
     {
      attempts++;
      if(!SkillTrainBatch(position))
        {
         invalid_batches++;
         ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                                 quota, attempts, invalid_batches, false, false, true, false,
                                 0, 0, 0, "");
         continue;
        }
      valid_batches++;
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, false, false, true, false,
                              0, 0, 0, "");
      if(ACSRMStage02Smoke && valid_batches >= ACSRMStage02SmokeLimit)
         break;
     }
//--- A stopped partial baseline cannot supply acceptance metrics.
   if(IsStopped())
     {
      if(scope == "source_eval")
         Print("ACSRM_STAGE02_STOP_REQUESTED phase=baseline_source");
      else
         Print("ACSRM_STAGE02_STOP_REQUESTED phase=baseline_target");
      ACSRMStage02StopReported = true;
      return(false);
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(valid_batches == 0 || !SkillForecast.BuildEpochCodebookDiagnostics() ||
      !SkillForecast.ReadEpochDiagnostics(
         lmix, router, trajectory, confidence, latent, observation,
         valid, invalid, entropy, distance, inactive, recovered) ||
      valid < double(valid_batches) || !ACSRMStage02Finite(lmix) ||
      !ACSRMStage02Finite(valid) || !ACSRMStage02Finite(invalid))
     {
      PrintFormat("ACSRM_STAGE02_BASELINE_FAIL scope=%s reason=metrics valid=%u invalid=%u",
                  scope, valid_batches, invalid_batches);
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, true, false, true, false,
                              0, 0, 0, "result=failed");
      ReturnFalse;
     }
   forecast_loss = lmix / valid;
   if(!ACSRMStage02Finite(forecast_loss))
     {
      PrintFormat("ACSRM_STAGE02_BASELINE_FAIL scope=%s reason=nonfinite_loss", scope);
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, true, false, true, false,
                              0, 0, 0, "result=failed");
      ReturnFalse;
     }
   ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                           quota, attempts, invalid_batches, true, true, true, false,
                           0, 0, 0, "");
   PrintFormat("ACSRM_STAGE02_BASELINE_PASS scope=%s loss=%.9f valid=%u invalid=%u",
               scope, forecast_loss, valid_batches, invalid_batches);
   return(true);
  }
//+------------------------------------------------------------------+
//| Fingerprints the selected market layers for Stage 02.            |
//+------------------------------------------------------------------+
bool ACSRMStage02MarketFingerprint(const int first, const int last, ulong &fingerprint)
  {
   if(first < 0 || first > last)
      ReturnFalse;
   fingerprint = ulong(1469598103934665603);
   for(int index = first; index <= last; index++)
     {
      CNeuronBaseOCL *layer = SkillMarket.Layer(index);
      if(!layer)
         ReturnFalse;
      fingerprint = (fingerprint ^ ulong(index + 1)) * ulong(1099511628211);
      if(!layer.AppendParameterFingerprint(fingerprint))
         ReturnFalse;
     }
   return(fingerprint != 0);
  }
//+------------------------------------------------------------------+
//| Captures the states that must remain frozen during ACSRM.        |
//+------------------------------------------------------------------+
bool ACSRMStage02CaptureSignatures(void)
  {
   ACSRMStage02SignaturesCaptured = false;
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ACSRMStage02MarketFingerprint(0, 4, ACSRMStage02BaseFingerprint) ||
      !ACSRMStage02MarketFingerprint(6, 6, ACSRMStage02ForecastFingerprint) ||
      !SkillTarget.ParameterFingerprint(ACSRMStage02TargetFingerprint))
      ReturnFalse;
   ACSRMStage02PosteriorFingerprint = ulong(1469598103934665603);
   if(!ompb.AppendParameterFingerprint(ACSRMStage02PosteriorFingerprint) ||
      ACSRMStage02PosteriorFingerprint == 0)
      ReturnFalse;
   PrintFormat("ACSRM_STAGE02_SIGNATURES_BEFORE base=%I64u forecast=%I64u target=%I64u posterior=%I64u",
               ACSRMStage02BaseFingerprint, ACSRMStage02ForecastFingerprint,
               ACSRMStage02TargetFingerprint, ACSRMStage02PosteriorFingerprint);
   ACSRMStage02SignaturesCaptured = true;
   return(true);
  }
//+-------------------------------------------------------------------+
//| Ensures Stage 02 changed only the posterior/alpha parameter set.  |
//+-------------------------------------------------------------------+
bool ACSRMStage02VerifySignatures(void)
  {
   if(!ACSRMStage02SignaturesCaptured)
     {
      Print("ACSRM_STAGE02_SIGNATURES_FAIL reason=not_captured");
      return(false);
     }
   ulong base = 0, forecast = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ACSRMStage02MarketFingerprint(0, 4, base) ||
      !ACSRMStage02MarketFingerprint(6, 6, forecast) ||
      !SkillTarget.ParameterFingerprint(target) ||
      !ompb.AppendParameterFingerprint(posterior))
      ReturnFalse;
   const bool outer_unchanged = (base == ACSRMStage02BaseFingerprint &&
                                 forecast == ACSRMStage02ForecastFingerprint &&
                                 target == ACSRMStage02TargetFingerprint);
   const bool posterior_changed = (posterior != ACSRMStage02PosteriorFingerprint);
   PrintFormat("ACSRM_STAGE02_SIGNATURES_AFTER base=%I64u forecast=%I64u target=%I64u posterior=%I64u",
               base, forecast, target, posterior);
   if(!outer_unchanged || !posterior_changed)
     {
      PrintFormat("ACSRM_STAGE02_SIGNATURES_FAIL outer_unchanged=%s posterior_changed=%s",
                  (outer_unchanged ? "true" : "false"),
                  (posterior_changed ? "true" : "false"));
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Verifies that pre-calibration inference preserved frozen state.  |
//+------------------------------------------------------------------+
bool ACSRMStage02VerifyPreCalibrationSignatures(void)
  {
   if(!ACSRMStage02SignaturesCaptured)
     {
      Print("ACSRM_STAGE02_PRECALIBRATION_SIGNATURES_FAIL reason=not_captured");
      return(false);
     }
   ulong base = 0, forecast = 0, target = 0;
   ulong ompb = ulong(1469598103934665603);
   CNeuronOMPBOCL *layer = GetACSRM();
//--- Capture the ACSRM and frozen outer-network fingerprints before calibration.
   if(!layer || !ACSRMStage02MarketFingerprint(0, 4, base) ||
      !ACSRMStage02MarketFingerprint(6, 6, forecast) ||
      !SkillTarget.ParameterFingerprint(target) ||
      !layer.AppendParameterFingerprint(ompb))
     {
      Print("ACSRM_STAGE02_PRECALIBRATION_SIGNATURES_FAIL reason=fingerprint_read");
      return(false);
     }
//--- Compare every protected fingerprint with the BYPASS/signature baseline.
   const bool unchanged = (base == ACSRMStage02BaseFingerprint &&
                           forecast == ACSRMStage02ForecastFingerprint &&
                           target == ACSRMStage02TargetFingerprint &&
                           ompb == ACSRMStage02PosteriorFingerprint);
   PrintFormat("ACSRM_STAGE02_PRECALIBRATION_SIGNATURES base=%I64u forecast=%I64u " +
               "target=%I64u ompb=%I64u unchanged=%s", base, forecast, target, ompb,
               (unchanged ? "true" : "false"));
   if(!unchanged)
     {
      Print("ACSRM_STAGE02_PRECALIBRATION_SIGNATURES_FAIL reason=state_changed");
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Computes an evaluation loss through posterior-mean inference.    |
//+------------------------------------------------------------------+
bool ACSRMStage02InferenceEvaluation(const int first, const int last, const string scope,
                                    double &forecast_loss, uint &valid_batches,
                                    uint &invalid_batches)
  {
   forecast_loss = 0.0;
   valid_batches = 0;
   invalid_batches = 0;
   uint attempts = 0;
   const uint rows = uint(first - last + 1);
   const uint quota = ACSRMStage02ProgressQuota(rows, 0, ACSRMStage02Smoke,
                      ACSRMStage02SmokeLimit);
   const string progress_phase = (StringFind(scope, "source") >= 0 ?
                                  "eval source" : "eval target");
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(first < last || !ompb || !SkillForecast || !ConfigureACSRM(OMPB_INFERENCE) ||
      !SkillMarket.Clear() || !SkillTarget.Clear() ||
      !SkillForecast.ResetEpochDiagnostics())
     {
      PrintFormat("ACSRM_STAGE02_EVALUATION_FAIL scope=%s reason=setup", scope);
      ReturnFalse;
     }
   SkillMarket.TrainMode(true);
   SkillTarget.TrainMode(false);
   ompb.TrainMode(false);
   ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                           quota, attempts, invalid_batches, false, false, true, false,
                           0, 0, 0, "");
   for(int position = first; position >= last && !IsStopped(); position--)
     {
      attempts++;
      if(!SkillTrainBatch(position))
        {
         invalid_batches++;
         ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                                 quota, attempts, invalid_batches, false, false, true, false,
                                 0, 0, 0, "");
         continue;
        }
      valid_batches++;
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, false, false, true, false,
                              0, 0, 0, "");
      if(ACSRMStage02Smoke && valid_batches >= ACSRMStage02SmokeLimit)
         break;
     }
//--- A stopped partial inference pass cannot supply evaluation metrics.
   if(IsStopped())
     {
      if(scope == "pre_source_inference")
         Print("ACSRM_STAGE02_STOP_REQUESTED phase=pre_inference_source");
      else
         if(scope == "pre_target_inference")
            Print("ACSRM_STAGE02_STOP_REQUESTED phase=pre_inference_target");
         else
            if(scope == "source_inference")
               Print("ACSRM_STAGE02_STOP_REQUESTED phase=post_inference_source");
            else
               Print("ACSRM_STAGE02_STOP_REQUESTED phase=post_inference_target");
      ACSRMStage02StopReported = true;
      return(false);
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(valid_batches == 0 || !SkillForecast.BuildEpochCodebookDiagnostics() ||
      !SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence, latent,
            observation, valid, invalid, entropy, distance,
            inactive, recovered) ||
      valid < double(valid_batches) || !ACSRMStage02Finite(lmix) ||
      !ACSRMStage02Finite(valid) || !ACSRMStage02Finite(invalid))
     {
      PrintFormat("ACSRM_STAGE02_EVALUATION_FAIL scope=%s reason=metrics valid=%u invalid=%u",
                  scope, valid_batches, invalid_batches);
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, true, false, true, false,
                              0, 0, 0, "result=failed");
      ReturnFalse;
     }
   forecast_loss = lmix / valid;
   if(!ACSRMStage02Finite(forecast_loss) || !SkillVerifyFrozenForecastExact())
     {
      PrintFormat("ACSRM_STAGE02_EVALUATION_FAIL scope=%s reason=finite_or_frozen", scope);
      ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                              quota, attempts, invalid_batches, true, false, true, false,
                              0, 0, 0, "result=failed");
      ReturnFalse;
     }
   ACSRMStage02ShowProgress(progress_phase, (ACSRMStage02Smoke ? valid_batches : attempts),
                           quota, attempts, invalid_batches, true, true, true, false,
                           0, 0, 0, "");
   PrintFormat("ACSRM_STAGE02_EVALUATION_PASS scope=%s loss=%.9f valid=%u invalid=%u",
               scope, forecast_loss, valid_batches, invalid_batches);
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks the frozen acceptance gate without replacing any file.    |
//+------------------------------------------------------------------+
bool ACSRMStage02Accept(const double source_before, const double target_before,
                       const double source_after, const double target_after,
                       const uint source_valid, const uint target_valid)
  {
   const double denominator_source = MathMax(MathAbs(source_before), 1.0e-12);
   const double denominator_target = MathMax(MathAbs(target_before), 1.0e-12);
   const double target_improvement = (target_before - target_after) / denominator_target;
   const double source_degradation = (source_after - source_before) / denominator_source;
   const bool finite = (ACSRMStage02Finite(source_before) && ACSRMStage02Finite(target_before) &&
                        ACSRMStage02Finite(source_after) && ACSRMStage02Finite(target_after));
   const bool accepted = (finite && source_valid > 0 && target_valid > 0 &&
                          target_improvement >= 0.01 && source_degradation <= 0.01);
   PrintFormat("ACSRM_STAGE02_ACCEPTANCE target_improvement=%.8f source_degradation=%.8f " +
               "source_valid=%u target_valid=%u accepted=%s", target_improvement,
               source_degradation, source_valid, target_valid,
               (accepted ? "true" : "false"));
   return(accepted);
  }
//+------------------------------------------------------------------+
//| The temporary graph must never validate a production file.       |
//+------------------------------------------------------------------+
ulong ACSRMStage02ForecastSignature(const string market_file,
                                   CNeuronScenarioForecast *forecast)
  {
   if(!forecast)
      return(0);
   ulong hash = ulong(1469598103934665603);
   hash = SkillHashFile(hash, market_file);
   if(hash == 0)
      return(0);
   hash = SkillHashUInt(hash, Skill_FORMAT_VERSION);
   hash = SkillHashUInt(hash, BarDescr);
   hash = SkillHashUInt(hash, NScenarios);
   hash = SkillHashUInt(hash, TopK);
   hash = SkillHashUInt(hash, NForecast);
   hash = SkillHashUInt(hash, EmbeddingSize);
   hash = SkillHashUInt(hash, forecast.ContractSignature());
   hash = SkillHashText(hash, "z_layout=K,V,H,D;u_layout=K,V,H;pi_layout=K;codebook_layout=K,V,H,D");
   hash = SkillHashText(hash,
                          "market_layout=RankTCM_then_ACSRM_then_ScenarioForecast;" +
                          "variable_order=BarDescr_feature_series_0_to_8");
   return(SkillHashText(hash, "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw"));
  }
//+------------------------------------------------------------------+
//| Writes the candidate manifest for Stage 02 checkpoints.          |
//+------------------------------------------------------------------+
bool ACSRMStage02WriteCandidateManifest(const string market_file, const string target_file,
                                       const string manifest_file, const uint completed_epochs)
  {
   const ulong forecast_signature = ACSRMStage02ForecastSignature(market_file, SkillForecast);
   const ulong target_hash = SkillHashFile(ulong(1469598103934665603), target_file);
   const ulong training_signature = SkillHashFile(forecast_signature, target_file);
   if(!SkillForecast || forecast_signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
   int handle = FileOpen(manifest_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   const bool written = (FileWrite(handle, "format=ACSRM_FORECAST") > 0 &&
                         FileWrite(handle, StringFormat("version=%u", Skill_FORMAT_VERSION)) > 0 &&
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
                                   SkillForecast.ContractSignature())) > 0 &&
                         FileWrite(handle, StringFormat("forecast_signature=%I64u", forecast_signature)) > 0 &&
                         FileWrite(handle, StringFormat("target_hash=%I64u", target_hash)) > 0 &&
                         FileWrite(handle, StringFormat("training_signature=%I64u", training_signature)) > 0 &&
                         FileWrite(handle, StringFormat("completed_epochs=%u", completed_epochs)) > 0 &&
                         FileWrite(handle, StringFormat("training_batches=%I64u", SkillBatches)) > 0 &&
                         FileWrite(handle, StringFormat("invalid_batches=%I64u", SkillInvalidBatches)) > 0 &&
                         FileWrite(handle, "normalization=OHLC_deltas_from_open;" +
                                   "tick_volume_div_1000;RSI_CCI_ATR_MACD_raw") > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   return(written);
  }
//+------------------------------------------------------------------+
//| Validates the candidate manifest structure and fields.           |
//+------------------------------------------------------------------+
bool ACSRMStage02ValidateCandidateManifest(const string market_file, const string target_file,
      const string manifest_file,
      CNeuronScenarioForecast *forecast)
  {
   const ulong signature = ACSRMStage02ForecastSignature(market_file, forecast);
   const ulong target_hash = SkillHashFile(ulong(1469598103934665603), target_file);
   const ulong training_signature = SkillHashFile(signature, target_file);
   if(!forecast || signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
#define ACSRM_STAGE02_MANIFEST_EQ(KEY,VALUE) if(SkillManifestValue(manifest_file, KEY) != (VALUE)) ReturnFalse
   ACSRM_STAGE02_MANIFEST_EQ("format", "ACSRM_FORECAST");
   ACSRM_STAGE02_MANIFEST_EQ("version", IntegerToString(Skill_FORMAT_VERSION));
   ACSRM_STAGE02_MANIFEST_EQ("forecast_type", IntegerToString(defNeuronScenarioForecast));
   ACSRM_STAGE02_MANIFEST_EQ("variables", IntegerToString(BarDescr));
   ACSRM_STAGE02_MANIFEST_EQ("scenarios", IntegerToString(NScenarios));
   ACSRM_STAGE02_MANIFEST_EQ("top_k", IntegerToString(TopK));
   ACSRM_STAGE02_MANIFEST_EQ("horizon", IntegerToString(NForecast));
   ACSRM_STAGE02_MANIFEST_EQ("latent", IntegerToString(EmbeddingSize));
   ACSRM_STAGE02_MANIFEST_EQ("z_layout", "K,V,H,D");
   ACSRM_STAGE02_MANIFEST_EQ("u_layout", "K,V,H");
   ACSRM_STAGE02_MANIFEST_EQ("pi_layout", "K");
   ACSRM_STAGE02_MANIFEST_EQ("codebook_layout", "K,V,H,D");
   ACSRM_STAGE02_MANIFEST_EQ("variable_order", "BarDescr_feature_series_0_to_8");
   ACSRM_STAGE02_MANIFEST_EQ("normalization", "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   ACSRM_STAGE02_MANIFEST_EQ("contract_signature", StringFormat("%I64u", forecast.ContractSignature()));
   ACSRM_STAGE02_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", signature));
   ACSRM_STAGE02_MANIFEST_EQ("target_hash", StringFormat("%I64u", target_hash));
   ACSRM_STAGE02_MANIFEST_EQ("training_signature", StringFormat("%I64u", training_signature));
#undef ACSRM_STAGE02_MANIFEST_EQ
   return(SkillManifestValue(manifest_file, "completed_epochs") != "" &&
          SkillManifestValue(manifest_file, "training_batches") != "" &&
          SkillManifestValue(manifest_file, "invalid_batches") != "");
  }
//+---------------------------------------------------------------------+
//| Reload-probes one immutable candidate before selector  activation.  |
//+---------------------------------------------------------------------+
bool ACSRMStage02ValidateCandidate(const string market_file, const string target_file,
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
   CNeuronOMPBOCL *live_ompb = GetACSRM();
   if(!rank_tcm || rank_tcm.Type() != defNeuronCogDriverRankTCM || !staged_ompb ||
      !staged_forecast ||
      !market.ParameterFingerprint(staged_market) || !target.ParameterFingerprint(staged_target) ||
      !staged_ompb.AppendParameterFingerprint(staged_posterior) || !SkillMarket.ParameterFingerprint(live_market) ||
      !SkillTarget.ParameterFingerprint(live_target) || !live_ompb ||
      !live_ompb.AppendParameterFingerprint(live_posterior) ||
      !ACSRMStage02ValidateCandidateManifest(market_file, target_file, manifest_file,
            staged_forecast))
      ReturnFalse;
   return(staged_market == live_market && staged_target == live_target &&
          staged_posterior == live_posterior);
  }
//+------------------------------------------------------------------+
//--- A malformed active selector is fail-closed.  Its absence deliberately
//--- retains the original Stage 01 fixed-name loading contract.
bool                 ACSRMStage02RecoveryFailed = false;
bool                 ACSRMStage02SelectorActivated = false;
bool                 ACSRMStage02ReloadProofPassed = false;
bool                 ACSRMStage02PreviousProofReady = false;
string               ACSRMStage02StagedGeneration = "";
ulong                ACSRMStage02PreviousMarketFingerprint = 0;
ulong                ACSRMStage02PreviousTargetFingerprint = 0;
ulong                ACSRMStage02PreviousPosteriorFingerprint = 0;
//+-------------------------------------------------------------------+
//| Validates a complete checkpoint set without touching live state.  |
//+-------------------------------------------------------------------+
bool ACSRMStage02ValidateCheckpointSet(const string market_file, const string target_file,
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
          ACSRMStage02ValidateCandidateManifest(market_file, target_file, manifest_file,
                checkpoint_forecast));
  }
//+----------------------------------------------------------------------+
//| Identifies the clean initial state before the first Stage 01  save.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02CanonicalTupleMissing(const string market_file, const string target_file,
                                      const string manifest_file)
  {
//--- A Stage 01 restore miss is safe only when no tuple member exists.
   return(!FileIsExist(market_file, FILE_COMMON) && !FileIsExist(target_file, FILE_COMMON) &&
          !FileIsExist(manifest_file, FILE_COMMON));
  }
//+----------------------------------------------------------------------+
//| Captures the immutable checkpoint fingerprints for rollback  proof.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02CheckpointFingerprints(const string market_file, const string target_file,
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
//+-------------------------------------------------------------------+
//| Restores the Stage 01 fixed-name checkpoint as the default view.  |
//+-------------------------------------------------------------------+
void ACSRMStage02ResetActiveCheckpointFiles(void)
  {
   SkillActiveMarketFile = Skill_MARKET_FILE;
   SkillActiveTargetFile = Skill_TARGET_FILE;
   SkillActiveManifestFile = Skill_MANIFEST_FILE;
  }
//+------------------------------------------------------------------+
//| Accepts only generated identifiers, never arbitrary filenames.   |
//+------------------------------------------------------------------+
bool ACSRMStage02GenerationIdValid(const string generation)
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
//| Maps a selector generation to its immutable checkpoint trio.     |
//+------------------------------------------------------------------+
bool ACSRMStage02GenerationFiles(const string generation, string &market_file,
                                string &target_file, string &manifest_file)
  {
   if(generation == "stage01")
     {
      market_file = Skill_MARKET_FILE;
      target_file = Skill_TARGET_FILE;
      manifest_file = Skill_MANIFEST_FILE;
      return(true);
     }
   if(!ACSRMStage02GenerationIdValid(generation))
      return(false);
   market_file = "ACSRMMarket" + ACSRM_STAGE02_GENERATION_PREFIX + generation + ".nnw";
   target_file = "ACSRMTarget" + ACSRM_STAGE02_GENERATION_PREFIX + generation + ".nnw";
   manifest_file = "ACSRMForecast" + ACSRM_STAGE02_GENERATION_PREFIX + generation + ".manifest";
   return(true);
  }
//+------------------------------------------------------------------+
//| Accepts only the fixed Stage 01 trio or one exact generation.    |
//+------------------------------------------------------------------+
bool ACSRMStage02ExplicitCheckpointFilesValid(const string market_file, const string target_file,
      const string manifest_file)
  {
   string expected_market = "", expected_target = "", expected_manifest = "";
   if(!ACSRMStage02GenerationFiles("stage01", expected_market, expected_target, expected_manifest))
      return(false);
   if(market_file == expected_market && target_file == expected_target &&
      manifest_file == expected_manifest)
      return(true);
   const string market_prefix = "ACSRMMarket" + ACSRM_STAGE02_GENERATION_PREFIX;
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
   if(!ACSRMStage02GenerationFiles(generation, expected_market, expected_target, expected_manifest))
      return(false);
   return(market_file == expected_market && target_file == expected_target &&
          manifest_file == expected_manifest);
  }
//+----------------------------------------------------------------------+
//| Rejects interrupted publish sidecars instead of guessing  recovery.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02RecoverySidecarPresent(string &sidecar_file)
  {
   sidecar_file = "";
   if(FileIsExist(ACSRM_STAGE02_LEGACY_TRANSACTION_FILE, FILE_COMMON))
     {
      sidecar_file = ACSRM_STAGE02_LEGACY_TRANSACTION_FILE;
      return(true);
     }
   if(FileIsExist(ACSRM_STAGE02_LEGACY_TRANSACTION_NEXT_FILE, FILE_COMMON))
     {
      sidecar_file = ACSRM_STAGE02_LEGACY_TRANSACTION_NEXT_FILE;
      return(true);
     }
   if(FileIsExist(ACSRM_STAGE02_SELECTOR_NEXT, FILE_COMMON))
     {
      sidecar_file = ACSRM_STAGE02_SELECTOR_NEXT;
      return(true);
     }
   if(FileIsExist(ACSRM_STAGE02_SELECTOR_RESTORE, FILE_COMMON))
     {
      sidecar_file = ACSRM_STAGE02_SELECTOR_RESTORE;
      return(true);
     }
//--- Previous is valid only in the process that created it and will remove it
//--- after deterministic reload proof.  On restart it is an interrupted
//--- transaction marker and must fail closed rather than activate unproven data.
   if(FileIsExist(ACSRM_STAGE02_SELECTOR_PREVIOUS, FILE_COMMON) &&
      !ACSRMStage02PreviousProofReady)
     {
      sidecar_file = ACSRM_STAGE02_SELECTOR_PREVIOUS;
      return(true);
     }
   return(false);
  }
//+----------------------------------------------------------------------+
//| Reads exactly one strict selector without allowing path  injection.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02ReadSelector(const string selector_file, string &generation)
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
         if(format_seen || value != "ACSRM_STAGE02_SELECTOR")
            valid = false;
         format_seen = true;
        }
      else
         if(key == "version")
           {
            if(version_seen || value != IntegerToString(ACSRM_STAGE02_SELECTOR_VERSION))
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
      (parsed_generation != "stage01" && !ACSRMStage02GenerationIdValid(parsed_generation)))
      return(false);
   generation = parsed_generation;
   return(true);
  }
//+------------------------------------------------------------------+
//| Binds one selector to the exact expected checkpoint generation.  |
//+------------------------------------------------------------------+
bool ACSRMStage02SelectorMatchesExpected(const string selector_file,
                                        const string expected_generation,
                                        const ulong expected_market,
                                        const ulong expected_target,
                                        const ulong expected_posterior)
  {
   string generation = "", market_file = "", target_file = "", manifest_file = "";
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   if(!FileIsExist(selector_file, FILE_COMMON) ||
      !ACSRMStage02ReadSelector(selector_file, generation) || generation != expected_generation ||
      !ACSRMStage02GenerationFiles(generation, market_file, target_file, manifest_file) ||
      !ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !ACSRMStage02CheckpointFingerprints(market_file, target_file, market, target, posterior) ||
      market != expected_market || target != expected_target || posterior != expected_posterior)
      return(false);
   return(true);
  }
//+-------------------------------------------------------------------+
//| Writes and rereads one non-active selector file before a switch.  |
//+-------------------------------------------------------------------+
bool ACSRMStage02WriteSelectorFile(const string selector_file, const string generation)
  {
   if(generation != "stage01" && !ACSRMStage02GenerationIdValid(generation))
      return(false);
   int handle = FileOpen(selector_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return(false);
   const bool written = (FileWrite(handle, "format=ACSRM_STAGE02_SELECTOR") > 0 &&
                         FileWrite(handle, StringFormat("version=%u",
                                   ACSRM_STAGE02_SELECTOR_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("generation=%s", generation)) > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   string reread = "";
   return(written && ACSRMStage02ReadSelector(selector_file, reread) && reread == generation);
  }
//+---------------------------------------------------------------------+
//| Ensures three tuple names describe three distinct physical  files.  |
//+---------------------------------------------------------------------+
bool ACSRMStage02TupleNamesValid(const string market_file, const string target_file,
                                const string manifest_file)
  {
   return(market_file != "" && target_file != "" && manifest_file != "" &&
          market_file != target_file && market_file != manifest_file &&
          target_file != manifest_file);
  }
//+------------------------------------------------------------------+
//| Copies a complete checkpoint tuple with manifest last.           |
//+------------------------------------------------------------------+
bool ACSRMStage02CopyTuple(const string source_market, const string source_target,
                          const string source_manifest, const string destination_market,
                          const string destination_target, const string destination_manifest,
                          const bool rewrite)
  {
   if(!ACSRMStage02TupleNamesValid(source_market, source_target, source_manifest) ||
      !ACSRMStage02TupleNamesValid(destination_market, destination_target,
                                  destination_manifest))
      return(false);
   const uint flags = (rewrite ? FILE_COMMON | FILE_REWRITE : FILE_COMMON);
   return(FileCopy(source_market, FILE_COMMON, destination_market, flags) &&
          FileCopy(source_target, FILE_COMMON, destination_target, flags) &&
          FileCopy(source_manifest, FILE_COMMON, destination_manifest, flags));
  }
//+-------------------------------------------------------------------------------------------------------------------------------------------+
//| Publishes a staged tuple through three canonical active  filenames. The staged and previous tuples remain until reload  proof completes.  |
//+-------------------------------------------------------------------------------------------------------------------------------------------+
bool ACSRMStage02PublishCanonicalTuple(const string market_file, const string target_file,
                                      const string manifest_file, const string next_market,
                                      const string next_target, const string next_manifest,
                                      const string previous_market,
                                      const string previous_target,
                                      const string previous_manifest)
  {
   if(!ACSRMStage02TupleNamesValid(market_file, target_file, manifest_file) ||
      !ACSRMStage02TupleNamesValid(next_market, next_target, next_manifest) ||
      !ACSRMStage02TupleNamesValid(previous_market, previous_target, previous_manifest) ||
      !ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !ACSRMStage02ValidateCheckpointSet(next_market, next_target, next_manifest) ||
      FileIsExist(previous_market, FILE_COMMON) || FileIsExist(previous_target, FILE_COMMON) ||
      FileIsExist(previous_manifest, FILE_COMMON))
      return(false);
   if(!ACSRMStage02CopyTuple(market_file, target_file, manifest_file, previous_market,
                            previous_target, previous_manifest, false) ||
      !ACSRMStage02ValidateCheckpointSet(previous_market, previous_target, previous_manifest))
      return(false);
   return(ACSRMStage02CopyTuple(next_market, next_target, next_manifest, market_file,
                               target_file, manifest_file, true) &&
          ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file));
  }
//+----------------------------------------------------------------------+
//| Restores the retained tuple using the manifest as the final  write.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02RestoreCanonicalTuple(const string market_file, const string target_file,
                                      const string manifest_file, const string previous_market,
                                      const string previous_target,
                                      const string previous_manifest)
  {
   if(!ACSRMStage02TupleNamesValid(market_file, target_file, manifest_file) ||
      !ACSRMStage02TupleNamesValid(previous_market, previous_target, previous_manifest) ||
      !ACSRMStage02ValidateCheckpointSet(previous_market, previous_target, previous_manifest))
      return(false);
   return(ACSRMStage02CopyTuple(previous_market, previous_target, previous_manifest, market_file,
                               target_file, manifest_file, true) &&
          ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file));
  }
//+------------------------------------------------------------------+
//| Classifies the retained tuple required by one Stage 01 phase.    |
//+------------------------------------------------------------------+
bool SkillStage01TupleStateValid(const bool market_exists,
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
//| Removes only files owned by a completed Stage 01 transaction.    |
//+------------------------------------------------------------------+
bool SkillStage01DeleteFile(const string file_name)
  {
   return(!FileIsExist(file_name, FILE_COMMON) || FileDelete(file_name, FILE_COMMON));
  }
//+------------------------------------------------------------------+
//| Releases Stage 01 resources and resets stage state.              |
//+------------------------------------------------------------------+
bool SkillStage01Cleanup(void)
  {
//--- Remove the commit marker last; its presence must survive every partial cleanup.
   return(SkillStage01DeleteFile(ACSRM_STAGE01_MANIFEST_NEXT_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_TARGET_NEXT_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_MARKET_NEXT_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_TARGET_PREVIOUS_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_MARKET_PREVIOUS_FILE) &&
          SkillStage01DeleteFile(ACSRM_STAGE01_TRANSACTION_FILE));
  }
//+-------------------------------------------------------------------+
//| Writes one durable marker before canonical Stage 01 publication.  |
//+-------------------------------------------------------------------+
bool SkillStage01WriteTransaction(const bool had_previous, const string phase,
                                    const ulong market, const ulong target,
                                    const ulong posterior)
  {
   if((phase != "prepared" && phase != "previous_ready") ||
      FileIsExist(ACSRM_STAGE01_TRANSACTION_FILE, FILE_COMMON))
      return(false);
   int handle = FileOpen(ACSRM_STAGE01_TRANSACTION_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return(false);
//--- The candidate fingerprints identify an already-published complete tuple.
   const bool written = (FileWrite(handle, "format=ACSRM_STAGE01_TRANSACTION") > 0 &&
                         FileWrite(handle, StringFormat("version=%u", ACSRM_STAGE01_TRANSACTION_VERSION)) > 0 &&
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
//| Rewrites only a complete transaction marker state transition.    |
//+------------------------------------------------------------------+
bool SkillStage01AdvanceTransaction(const bool had_previous, const string phase,
                                      const ulong market, const ulong target,
                                      const ulong posterior)
  {
   if(!SkillStage01DeleteFile(ACSRM_STAGE01_TRANSACTION_FILE))
      return(false);
   return(SkillStage01WriteTransaction(had_previous, phase, market, target, posterior));
  }
//+------------------------------------------------------------------+
//| Reads one strict Stage 01 transaction marker.                    |
//+------------------------------------------------------------------+
bool SkillStage01ReadTransaction(bool &had_previous, string &phase,
                                   string &market, string &target, string &posterior)
  {
   had_previous = false;
   phase = "";
   market = "";
   target = "";
   posterior = "";
   if(!FileIsExist(ACSRM_STAGE01_TRANSACTION_FILE, FILE_COMMON))
      return(false);
   const string format = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "format");
   const string version = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "version");
   const string previous = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "had_previous");
   phase = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "phase");
   market = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "candidate_market");
   target = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "candidate_target");
   posterior = SkillManifestValue(ACSRM_STAGE01_TRANSACTION_FILE, "candidate_posterior");
   if(format != "ACSRM_STAGE01_TRANSACTION" ||
      version != IntegerToString(ACSRM_STAGE01_TRANSACTION_VERSION) ||
      (previous != "true" && previous != "false") ||
      (phase != "prepared" && phase != "previous_ready") ||
      market == "" || target == "" || posterior == "")
      return(false);
   had_previous = (previous == "true");
   return(true);
  }
//+------------------------------------------------------------------+
//| Compares a complete tuple against its transaction marker.        |
//+------------------------------------------------------------------+
bool SkillStage01TupleMatches(const string market_file, const string target_file,
                                const string manifest_file, const string market,
                                const string target, const string posterior)
  {
   ulong actual_market = 0, actual_target = 0;
   ulong actual_posterior = ulong(1469598103934665603);
   return(ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file) &&
          ACSRMStage02CheckpointFingerprints(market_file, target_file, actual_market,
                actual_target, actual_posterior) &&
          StringFormat("%I64u", actual_market) == market &&
          StringFormat("%I64u", actual_target) == target &&
          StringFormat("%I64u", actual_posterior) == posterior);
  }
//+------------------------------------------------------------------+
//| Recovers only an interrupted Stage 01-owned publication.         |
//+------------------------------------------------------------------+
bool SkillStage01RecoverInterruptedCheckpoint(void)
  {
   const bool marker_exists = FileIsExist(ACSRM_STAGE01_TRANSACTION_FILE, FILE_COMMON);
   const bool temporary_exists = (FileIsExist(ACSRM_STAGE01_MARKET_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(ACSRM_STAGE01_TARGET_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(ACSRM_STAGE01_MANIFEST_NEXT_FILE, FILE_COMMON) ||
                                  FileIsExist(ACSRM_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
                                  FileIsExist(ACSRM_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
                                  FileIsExist(ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON));
   if(!marker_exists)
     {
      //--- Uncommitted staging never changed canonical names and is safe to discard.
      if(!temporary_exists)
         return(true);
      if(!ACSRMStage02ValidateCheckpointSet(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                           Skill_MANIFEST_FILE) &&
         !ACSRMStage02CanonicalTupleMissing(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                           Skill_MANIFEST_FILE))
         return(false);
      return(SkillStage01Cleanup());
     }
   bool had_previous = false;
   string phase = "", market = "", target = "", posterior = "";
   if(!SkillStage01ReadTransaction(had_previous, phase, market, target, posterior))
      return(false);
//--- A complete candidate published before cleanup is already the committed epoch.
   if(SkillStage01TupleMatches(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                 Skill_MANIFEST_FILE, market, target, posterior))
      return(SkillStage01Cleanup());
//--- Before previous-ready canonical names are still the earlier valid tuple.
   if(phase == "prepared" && ACSRMStage02ValidateCheckpointSet(Skill_MARKET_FILE,
         Skill_TARGET_FILE, Skill_MANIFEST_FILE))
      return(SkillStage01Cleanup());
   if(had_previous)
     {
      if(!SkillStage01TupleStateValid(FileIsExist(ACSRM_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON),
                                        FileIsExist(ACSRM_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON),
                                        FileIsExist(ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON), true, phase) ||
         !ACSRMStage02RestoreCanonicalTuple(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                           Skill_MANIFEST_FILE,
                                           ACSRM_STAGE01_MARKET_PREVIOUS_FILE,
                                           ACSRM_STAGE01_TARGET_PREVIOUS_FILE,
                                           ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE))
         return(false);
      return(SkillStage01Cleanup());
     }
//--- The first publication has no prior checkpoint; remove only marker-owned partial data.
   if(!SkillStage01DeleteFile(Skill_MANIFEST_FILE) ||
      !SkillStage01DeleteFile(Skill_TARGET_FILE) ||
      !SkillStage01DeleteFile(Skill_MARKET_FILE))
      return(false);
   return(SkillStage01Cleanup());
  }
//+------------------------------------------------------------------+
//| Publishes one fully validated Stage 01 checkpoint transaction.   |
//+------------------------------------------------------------------+
bool SkillStage01SaveCheckpoint(const uint completed_epochs)
  {
   if(FileIsExist(ACSRM_STAGE01_TRANSACTION_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_MARKET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_TARGET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE, FILE_COMMON))
      return(false);
//--- Classify the pre-save canonical state before staging any new model bytes.
   const bool had_previous = ACSRMStage02ValidateCheckpointSet(Skill_MARKET_FILE,
                             Skill_TARGET_FILE,
                             Skill_MANIFEST_FILE);
   if(!had_previous && !ACSRMStage02CanonicalTupleMissing(Skill_MARKET_FILE,
         Skill_TARGET_FILE, Skill_MANIFEST_FILE))
      return(false);
//--- The next tuple is self-contained and validated before a commit marker exists.
   const datetime now = TimeCurrent();
   if(!SkillMarket.Save(ACSRM_STAGE01_MARKET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !SkillTarget.Save(ACSRM_STAGE01_TARGET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !ACSRMStage02WriteCandidateManifest(ACSRM_STAGE01_MARKET_NEXT_FILE,
                                         ACSRM_STAGE01_TARGET_NEXT_FILE, ACSRM_STAGE01_MANIFEST_NEXT_FILE, completed_epochs) ||
      !ACSRMStage02ValidateCheckpointSet(ACSRM_STAGE01_MARKET_NEXT_FILE,
                                        ACSRM_STAGE01_TARGET_NEXT_FILE,
                                        ACSRM_STAGE01_MANIFEST_NEXT_FILE))
      return(false);
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   if(!ACSRMStage02CheckpointFingerprints(ACSRM_STAGE01_MARKET_NEXT_FILE,
                                         ACSRM_STAGE01_TARGET_NEXT_FILE,
                                         market, target, posterior) ||
      !SkillStage01WriteTransaction(had_previous, "prepared", market, target, posterior))
      return(false);
//--- Retain the old complete tuple before any short canonical file is replaced.
   if(had_previous &&
      (!ACSRMStage02CopyTuple(Skill_MARKET_FILE, Skill_TARGET_FILE, Skill_MANIFEST_FILE,
                             ACSRM_STAGE01_MARKET_PREVIOUS_FILE,
                             ACSRM_STAGE01_TARGET_PREVIOUS_FILE,
                             ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE, false) ||
       !ACSRMStage02ValidateCheckpointSet(ACSRM_STAGE01_MARKET_PREVIOUS_FILE,
                                         ACSRM_STAGE01_TARGET_PREVIOUS_FILE,
                                         ACSRM_STAGE01_MANIFEST_PREVIOUS_FILE)))
      return(false);
   if(!SkillStage01AdvanceTransaction(had_previous, "previous_ready", market, target, posterior) ||
      !ACSRMStage02CopyTuple(ACSRM_STAGE01_MARKET_NEXT_FILE, ACSRM_STAGE01_TARGET_NEXT_FILE,
                            ACSRM_STAGE01_MANIFEST_NEXT_FILE, Skill_MARKET_FILE,
                            Skill_TARGET_FILE, Skill_MANIFEST_FILE, true) ||
      !SkillStage01TupleMatches(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                  Skill_MANIFEST_FILE, StringFormat("%I64u", market),
                                  StringFormat("%I64u", target), StringFormat("%I64u", posterior)))
      return(false);
   if(!SkillStage01Cleanup())
      return(false);
   SkillLastSignature = SkillForecastSignature(Skill_MARKET_FILE);
   PrintFormat("ACSRM_STAGE01_CHECKPOINT_PASS epoch=%u", completed_epochs);
   return(true);
  }
//+---------------------------------------------------------------------+
//| Proves deterministic serialized parameters without live  mutation.  |
//+---------------------------------------------------------------------+
bool ACSRMStage02ReloadProofDeterministicTuple(const string market_file,
      const string target_file,
      const string manifest_file)
  {
   ulong first_market = 0, first_target = 0;
   ulong first_posterior = ulong(1469598103934665603);
   ulong second_market = 0, second_target = 0;
   ulong second_posterior = ulong(1469598103934665603);
   return(ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file) &&
          ACSRMStage02CheckpointFingerprints(market_file, target_file, first_market,
                first_target, first_posterior) &&
          ACSRMStage02CheckpointFingerprints(market_file, target_file, second_market,
                second_target, second_posterior) &&
          first_market != 0 && first_target != 0 && first_posterior != 0 &&
          first_market == second_market && first_target == second_target &&
          first_posterior == second_posterior);
  }
//+------------------------------------------------------------------+
//| Detects any interrupted canonical or legacy publication state.   |
//+------------------------------------------------------------------+
bool ACSRMStage02CanonicalSidecarPresent(string &sidecar_file)
  {
   sidecar_file = "";
   const string sidecars[] =
     {
      ACSRM_STAGE02_MARKET_NEXT_FILE,
      ACSRM_STAGE02_TARGET_NEXT_FILE,
      ACSRM_STAGE02_MANIFEST_NEXT_FILE,
      ACSRM_STAGE02_MARKET_PREVIOUS_FILE,
      ACSRM_STAGE02_TARGET_PREVIOUS_FILE,
      ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE,
      ACSRM_STAGE02_ACTIVE_SELECTOR,
      ACSRM_STAGE02_SELECTOR_NEXT,
      ACSRM_STAGE02_SELECTOR_PREVIOUS,
      ACSRM_STAGE02_SELECTOR_RESTORE,
      ACSRM_STAGE02_LEGACY_TRANSACTION_FILE,
      ACSRM_STAGE02_LEGACY_TRANSACTION_NEXT_FILE
     };
   for(int index = 0; index < ArraySize(sidecars); ++index)
      if(FileIsExist(sidecars[index], FILE_COMMON))
        {
         sidecar_file = sidecars[index];
         return(true);
        }
   return(false);
  }
//+--------------------------------------------------------------------------------------------------------------------------------------------------+
//| Resolves only the canonical production checkpoint or fails  closed. Lightweight mode is valid only when the caller validates  the loaded graph.  |
//+--------------------------------------------------------------------------------------------------------------------------------------------------+
bool ACSRMStage02ResolveActiveCheckpoint(const bool validate_tuple = true)
  {
   ACSRMStage02RecoveryFailed = false;
   ACSRMStage02ResetActiveCheckpointFiles();
//--- Complete a Stage 01-owned recovery before interpreting canonical files.
   if(!SkillStage01RecoverInterruptedCheckpoint())
     {
      ACSRMStage02RecoveryFailed = true;
      Print("ACSRM_STAGE01_CHECKPOINT_FAIL reason=recovery");
      return(false);
     }
   string sidecar_file = "";
   if(ACSRMStage02CanonicalSidecarPresent(sidecar_file))
     {
      ACSRMStage02RecoveryFailed = true;
      Print("ACSRM_STAGE02_CANONICAL_FAIL reason=incomplete_or_legacy_transaction");
      return(false);
     }
//--- No tuple exists before the first Stage 01 run; its caller creates a new graph.
   if(ACSRMStage02CanonicalTupleMissing(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                       Skill_MANIFEST_FILE))
     {
      Print("ACSRM_STAGE02_CANONICAL_MISS reason=initial_checkpoint_absent");
      return(false);
     }
//--- A partial or unreadable tuple is not a new-model state and remains fail-closed.
   if(!validate_tuple)
     {
      return(true);
     }
   if(!ACSRMStage02ValidateCheckpointSet(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                        Skill_MANIFEST_FILE))
     {
      ACSRMStage02RecoveryFailed = true;
      Print("ACSRM_STAGE02_CANONICAL_FAIL reason=invalid_active_tuple");
      return(false);
     }
   Print("ACSRM_STAGE02_CANONICAL_PASS checkpoint=active");
   return(true);
  }
//+------------------------------------------------------------------+
//| Retains the previous selection before the one-file activation.   |
//+------------------------------------------------------------------+
bool ACSRMStage02PreparePreviousSelector(void)
  {
   string generation = "", market_file = "", target_file = "", manifest_file = "";
   string sidecar_file = "";
   ACSRMStage02PreviousProofReady = false;
   if(ACSRMStage02RecoverySidecarPresent(sidecar_file) ||
      FileIsExist(ACSRM_STAGE02_SELECTOR_PREVIOUS, FILE_COMMON))
      return(false);
   if(!FileIsExist(ACSRM_STAGE02_ACTIVE_SELECTOR, FILE_COMMON))
     {
      generation = "stage01";
      market_file = Skill_MARKET_FILE;
      target_file = Skill_TARGET_FILE;
      manifest_file = Skill_MANIFEST_FILE;
     }
   else
      if(!ACSRMStage02ReadSelector(ACSRM_STAGE02_ACTIVE_SELECTOR, generation) ||
         !ACSRMStage02GenerationFiles(generation, market_file, target_file, manifest_file))
         return(false);
   if(!ACSRMStage02ValidateCheckpointSet(market_file, target_file, manifest_file) ||
      !ACSRMStage02CheckpointFingerprints(market_file, target_file,
                                         ACSRMStage02PreviousMarketFingerprint,
                                         ACSRMStage02PreviousTargetFingerprint,
                                         ACSRMStage02PreviousPosteriorFingerprint))
      return(false);
   if(generation == "stage01")
     {
      if(!ACSRMStage02WriteSelectorFile(ACSRM_STAGE02_SELECTOR_PREVIOUS, generation))
         return(false);
     }
   else
      if(!FileCopy(ACSRM_STAGE02_ACTIVE_SELECTOR, FILE_COMMON, ACSRM_STAGE02_SELECTOR_PREVIOUS,
                   FILE_COMMON | FILE_REWRITE))
         return(false);
   string copied = "";
   if(!ACSRMStage02ReadSelector(ACSRM_STAGE02_SELECTOR_PREVIOUS, copied) || copied != generation)
      return(false);
   ACSRMStage02PreviousProofReady = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Switches exactly one active file after full generation proof.    |
//+------------------------------------------------------------------+
bool ACSRMStage02ActivateGeneration(const string generation)
  {
   if(FileIsExist(ACSRM_STAGE02_SELECTOR_NEXT, FILE_COMMON) ||
      !ACSRMStage02WriteSelectorFile(ACSRM_STAGE02_SELECTOR_NEXT, generation))
      return(false);
   if(!FileMove(ACSRM_STAGE02_SELECTOR_NEXT, FILE_COMMON, ACSRM_STAGE02_ACTIVE_SELECTOR,
                FILE_COMMON | FILE_REWRITE))
     {
      Print("ACSRM_STAGE02_SELECTOR_FAIL reason=activation_move next_retained=true");
      return(false);
     }
   ACSRMStage02SelectorActivated = true;
   PrintFormat("ACSRM_STAGE02_SELECTOR_SWITCH_PASS generation=%s", generation);
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores a retained canonical tuple after failed reload proof.   |
//+------------------------------------------------------------------+
bool ACSRMStage02RestorePreviousSelector(void)
  {
   ulong previous_market = 0, previous_target = 0;
   ulong previous_posterior = ulong(1469598103934665603);
   if(!ACSRMStage02PreviousProofReady ||
      !ACSRMStage02CheckpointFingerprints(ACSRM_STAGE02_MARKET_PREVIOUS_FILE,
                                         ACSRM_STAGE02_TARGET_PREVIOUS_FILE,
                                         previous_market, previous_target, previous_posterior) ||
      previous_market != ACSRMStage02PreviousMarketFingerprint ||
      previous_target != ACSRMStage02PreviousTargetFingerprint ||
      previous_posterior != ACSRMStage02PreviousPosteriorFingerprint)
      return(false);
   if(!ACSRMStage02RestoreCanonicalTuple(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                        Skill_MANIFEST_FILE,
                                        ACSRM_STAGE02_MARKET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_TARGET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   ACSRMStage02SelectorActivated = false;
   ACSRMStage02ReloadProofPassed = false;
   if(!ACSRMStage02ReloadCheckpointProof(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                        Skill_MANIFEST_FILE,
                                        ACSRMStage02PreviousMarketFingerprint,
                                        ACSRMStage02PreviousTargetFingerprint,
                                        ACSRMStage02PreviousPosteriorFingerprint) ||
      !ACSRMStage02FinalizeSteadySelector())
      return(false);
   return(true);
  }
//+-------------------------------------------------------------------+
//| Stages and reload-probes the complete temporary canonical tuple.  |
//+-------------------------------------------------------------------+
bool ACSRMStage02StageCandidate(void)
  {
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ompb.AppendParameterFingerprint(posterior) ||
      !ACSRMStage02ResolveActiveCheckpoint())
      return(false);
   if(FileIsExist(ACSRM_STAGE02_MARKET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE02_TARGET_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE02_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE02_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE02_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      FileIsExist(ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE, FILE_COMMON))
      return(false);
   const datetime now = TimeCurrent();
   if(!SkillMarket.Save(ACSRM_STAGE02_MARKET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !SkillTarget.Save(ACSRM_STAGE02_TARGET_NEXT_FILE, 0.0f, 0.0f, 0.0f, now, true) ||
      !ACSRMStage02WriteCandidateManifest(ACSRM_STAGE02_MARKET_NEXT_FILE,
                                         ACSRM_STAGE02_TARGET_NEXT_FILE,
                                         ACSRM_STAGE02_MANIFEST_NEXT_FILE,
                                         SkillCompletedEpochs) ||
      !ACSRMStage02ValidateCandidate(ACSRM_STAGE02_MARKET_NEXT_FILE,
                                    ACSRM_STAGE02_TARGET_NEXT_FILE,
                                    ACSRM_STAGE02_MANIFEST_NEXT_FILE))
     {
      Print("ACSRM_STAGE02_CANONICAL_FAIL reason=stage_or_reload_probe");
      return(false);
     }
   ACSRMStage02StagedGeneration = "canonical";
   PrintFormat("ACSRM_STAGE02_CANONICAL_STAGE_PASS posterior=%I64u", posterior);
   return(true);
  }
//+------------------------------------------------------------------+
//| Publishes only a fully validated temporary canonical tuple.      |
//+------------------------------------------------------------------+
bool ACSRMStage02PublishCandidate(void)
  {
   if(!ACSRMStage02CheckpointFingerprints(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                         ACSRMStage02PreviousMarketFingerprint,
                                         ACSRMStage02PreviousTargetFingerprint,
                                         ACSRMStage02PreviousPosteriorFingerprint) ||
      !ACSRMStage02PublishCanonicalTuple(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                        Skill_MANIFEST_FILE,
                                        ACSRM_STAGE02_MARKET_NEXT_FILE,
                                        ACSRM_STAGE02_TARGET_NEXT_FILE,
                                        ACSRM_STAGE02_MANIFEST_NEXT_FILE,
                                        ACSRM_STAGE02_MARKET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_TARGET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   ACSRMStage02PreviousProofReady = true;
   ACSRMStage02SelectorActivated = true;
   return(true);
  }
//+-----------------------------------------------------------------------+
//| Loads exactly the supplied checkpoint without resolving a  selector.  |
//+-----------------------------------------------------------------------+
bool ACSRMStage02LoadExplicitCheckpoint(const string market_file, const string target_file,
                                       const string manifest_file)
  {
   if(!ACSRMStage02ExplicitCheckpointFilesValid(market_file, target_file, manifest_file))
      return(false);
   SkillFrozenBaselineReady = false;
   SkillForecast = NULL;
   if(!FileIsExist(market_file, FILE_COMMON) || !FileIsExist(target_file, FILE_COMMON) ||
      !FileIsExist(manifest_file, FILE_COMMON))
      return(false);
//--- The explicit loader retains this validated tuple as its current in-memory
//--- view for its caller's fingerprint/finalize work; it never resolves selectors.
   SkillActiveMarketFile = market_file;
   SkillActiveTargetFile = target_file;
   SkillActiveManifestFile = manifest_file;
   if(!SkillValidateForecastManifestHeader())
      return(false);
   float error = 0.0f, undefine = 0.0f, forecast = 0.0f;
   datetime studied = 0;
   if(!SkillMarket.Load(market_file, error, undefine, forecast, studied, true) ||
      !SkillTarget.Load(target_file, error, undefine, forecast, studied, true))
      return(false);
   if(!SkillTarget.SetOpenCLChecked(SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   SkillForecast = (CNeuronScenarioForecast *)SkillMarket.Layer(-1);
   if(!SkillForecast || SkillForecast.Type() != defNeuronScenarioForecast ||
      !ConfigureForecastRecoveryAge() || !SkillInitTrainingBuffers() ||
      !SkillValidateShapes())
      return(false);
   const ulong signature = SkillForecastSignature();
   const string completed = SkillManifestValue(manifest_file, "completed_epochs");
   const string batches = SkillManifestValue(manifest_file, "training_batches");
   const string invalid = SkillManifestValue(manifest_file, "invalid_batches");
   if(!SkillValidateForecastManifest(signature, true) || completed == "" || batches == "" ||
      invalid == "")
      return(false);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !SkillMarket.SetWeightsUpdate(false) || !SkillTarget.SetWeightsUpdate(false) ||
      !SkillForecast.SetCodebookUpdate(false) || !ConfigureACSRM(OMPB_INFERENCE) || !ompb.Clear() ||
      !SkillCaptureFrozenForecastBaseline(SkillForecast))
      return(false);
   SkillCompletedEpochs = (uint)StringToInteger(completed);
   SkillBatches = (ulong)StringToInteger(batches);
   SkillInvalidBatches = (ulong)StringToInteger(invalid);
   SkillMarket.TrainMode(false);
   SkillTarget.TrainMode(false);
   ompb.TrainMode(false);
   SkillLastSignature = signature;
   return(true);
  }
//+---------------------------------------------------------------------+
//| Compares one immutable reload output and names the failed  tensor.  |
//+---------------------------------------------------------------------+
bool ACSRMStage02ReloadTensorEqual(const string tensor, CBufferFloat *actual,
                                  CBufferFloat &expected)
  {
   if(!actual || !actual.BufferRead() || actual.Total() != expected.Total())
     {
      PrintFormat("ACSRM_STAGE02_RELOAD_TENSOR_FAIL tensor=%s reason=shape actual=%d expected=%d",
                  tensor, (actual ? actual.Total() : -1), expected.Total());
      return(false);
     }
   for(int index = 0; index < actual.Total(); index++)
     {
      if(!MathIsValidNumber(actual[index]) || !MathIsValidNumber(expected[index]) ||
         actual[index] != expected[index])
        {
         PrintFormat("ACSRM_STAGE02_RELOAD_TENSOR_FAIL tensor=%s index=%d first=%.9g second=%.9g",
                     tensor, index, expected[index], actual[index]);
         return(false);
        }
     }
   return(true);
  }
//+---------------------------------------------------------------------+
//| Proves the first inference result after two clean explicit  loads.  |
//+---------------------------------------------------------------------+
bool ACSRMStage02ReloadProofDeterministic(const string market_file, const string target_file,
      const string manifest_file, const int position,
      CBufferFloat *state, CBufferFloat *time)
  {
   CBufferFloat z, u, pi;
   if(position < 0 || !state || !time ||
      !ACSRMStage02LoadExplicitCheckpoint(market_file, target_file, manifest_file) ||
      !SkillForwardForecast(position, state, time) ||
      !SkillCaptureFrozenBuffer(SkillForecast.GetZ(), z) ||
      !SkillCaptureFrozenBuffer(SkillForecast.GetU(), u) ||
      !SkillCaptureFrozenBuffer(SkillForecast.GetPi(), pi) ||
      !ACSRMStage02LoadExplicitCheckpoint(market_file, target_file, manifest_file) ||
      !SkillForwardForecast(position, state, time) ||
      !ACSRMStage02ReloadTensorEqual("Z", SkillForecast.GetZ(), z) ||
      !ACSRMStage02ReloadTensorEqual("U", SkillForecast.GetU(), u) ||
      !ACSRMStage02ReloadTensorEqual("Pi", SkillForecast.GetPi(), pi) ||
      !SkillVerifyFrozenForecastExact(SkillForecast))
      return(false);
   return(true);
  }
//+----------------------------------------------------------------------------+
//| Reloads one selected checkpoint and proves deterministic mean  inference.  |
//+----------------------------------------------------------------------------+
bool ACSRMStage02ReloadCheckpointProof(const string market_file, const string target_file,
                                      const string manifest_file, const ulong expected_market,
                                      const ulong expected_target, const ulong expected_posterior)
  {
   ACSRMStage02ReloadProofPassed = false;
   if(!ACSRMStage02ReloadProofDeterministic(market_file, target_file, manifest_file,
                                           ACSRMStage02TargetEvalFirst, GetPointer(SkillState),
                                           GetPointer(SkillTime)))
      return(false);
   CNeuronOMPBOCL *ompb = GetACSRM();
   ulong market = 0, target = 0, posterior = ulong(1469598103934665603);
   if(!ompb || !SkillMarket.ParameterFingerprint(market) ||
      !SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      market != expected_market || target != expected_target || posterior != expected_posterior)
      return(false);
   ACSRMStage02ReloadProofPassed = true;
   Print("ACSRM_STAGE02_RELOAD_SMOKE_PASS mode=inference deterministic=true");
   return(true);
  }
//+------------------------------------------------------------------------+
//| Reloads the canonical tuple and proves deterministic mean  inference.  |
//+------------------------------------------------------------------------+
bool ACSRMStage02ReloadSmoke(const ulong expected_market, const ulong expected_target,
                            const ulong expected_posterior)
  {
   if(!ACSRMStage02SelectorActivated)
      return(false);
   return(ACSRMStage02ReloadCheckpointProof(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                           Skill_MANIFEST_FILE,
                                           expected_market, expected_target, expected_posterior));
  }
//+----------------------------------------------------------------------+
//| Cleans temporary tuples only after deterministic canonical  reload.  |
//+----------------------------------------------------------------------+
bool ACSRMStage02FinalizeSteadySelector(void)
  {
   ulong market = 0, target = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ACSRMStage02ReloadProofPassed ||
      !ACSRMStage02ValidateCheckpointSet(Skill_MARKET_FILE, Skill_TARGET_FILE,
                                        Skill_MANIFEST_FILE) ||
      !ACSRMStage02ValidateCheckpointSet(ACSRM_STAGE02_MARKET_NEXT_FILE,
                                        ACSRM_STAGE02_TARGET_NEXT_FILE,
                                        ACSRM_STAGE02_MANIFEST_NEXT_FILE) ||
      !ACSRMStage02ValidateCheckpointSet(ACSRM_STAGE02_MARKET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_TARGET_PREVIOUS_FILE,
                                        ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE))
      return(false);
   if(!ompb || !SkillMarket.ParameterFingerprint(market) ||
      !SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      market == 0 || target == 0 || posterior == 0)
      return(false);
   if(!FileDelete(ACSRM_STAGE02_MANIFEST_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(ACSRM_STAGE02_TARGET_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(ACSRM_STAGE02_MARKET_NEXT_FILE, FILE_COMMON) ||
      !FileDelete(ACSRM_STAGE02_MANIFEST_PREVIOUS_FILE, FILE_COMMON) ||
      !FileDelete(ACSRM_STAGE02_TARGET_PREVIOUS_FILE, FILE_COMMON) ||
      !FileDelete(ACSRM_STAGE02_MARKET_PREVIOUS_FILE, FILE_COMMON) ||
      !ACSRMStage02ResolveActiveCheckpoint())
      return(false);
   ACSRMStage02PreviousProofReady = false;
   return(true);
  }
//+-------------------------------------------------------------------+
//| Publishes only after canonical staging and deterministic reload.  |
//+-------------------------------------------------------------------+
bool ACSRMStage02SaveAcceptedCheckpoint(void)
  {
   ulong market = 0, target = 0, posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetACSRM();
   ACSRMStage02ReloadProofPassed = false;
   ACSRMStage02PreviousProofReady = false;
   ACSRMStage02SelectorActivated = false;
   if(!ompb || !SkillMarket.ParameterFingerprint(market) ||
      !SkillTarget.ParameterFingerprint(target) || !ompb.AppendParameterFingerprint(posterior) ||
      !ACSRMStage02StageCandidate() || !ACSRMStage02PublishCandidate())
      return(false);
   if(!ACSRMStage02ReloadSmoke(market, target, posterior) || !ACSRMStage02FinalizeSteadySelector())
     {
      const bool restored = ACSRMStage02RestorePreviousSelector();
      PrintFormat("ACSRM_STAGE02_CANONICAL_FAIL reason=reload_or_finalize restored=%s",
                  (restored ? "true" : "false"));
      return(false);
     }
   ACSRMStage02CheckpointPublished = true;
   Print("ACSRM_STAGE02_CANONICAL_PASS checkpoint=published");
   return(true);
  }
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Runs one frozen-graph ACSRM update. CNet owns full gradient  propagation but has parameter writes disabled; the explicit  ACSRM call below is therefore the only update in Stage 02.  |
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool ACSRMStage02RunBatch(const int position, const bool source_anchor)
  {
   CNeuronOMPBOCL *ompb = GetACSRM();
   CNeuronBaseOCL *rank_tcm = GetRankTCM();
   if(!ompb || !rank_tcm)
      ReturnFalseEx("Stage 02 batch layers are null");
   if(!ConfigureACSRM(OMPB_CALIBRATE))
      ReturnFalseEx("Stage 02 batch configuration");
   if(!ompb.SetSourceAnchor(source_anchor))
      ReturnFalseEx("Stage 02 source-anchor setup");
   ompb.TrainMode(true);
//--- Target batches publish Current only after the complete update succeeds.
   if(!source_anchor && !ompb.BeginCurrentTransaction())
      ReturnFalseEx("Stage 02 Current transaction begin");
   if(!SkillTrainBatch(position))
     {
      if(source_anchor)
        {
         if(!ompb.SetSourceAnchor(false))
            ReturnFalseEx("Stage 02 source-anchor cleanup after training failure");
        }
      else
        {
         if(!ompb.RollbackCurrentTransaction())
            ReturnFalseEx("Stage 02 Current rollback after training failure");
        }
      ReturnFalseEx("Stage 02 batch training");
     }
//--- A source anchor has already created its mean Forecast gradient.  Its
//--- forward/backward phase must retain the anchor flag so Current and both
//--- regularizers remain untouched.  Clear only the update guard afterwards
//--- to apply that already-computed mean posterior gradient.
   if(source_anchor)
     {
      const bool accepted = ompb.ForwardAccepted();
      if(!ompb.SetSourceAnchor(false))
         ReturnFalseEx("Stage 02 source-anchor cleanup");
      if(!accepted)
         ReturnFalseEx("Stage 02 source-anchor forward rejected");
      if(!ompb.UpdateInputWeights(rank_tcm))
         ReturnFalseEx("Stage 02 source-anchor update");
      return(true);
     }
//--- A fallback output cannot update weights or advance Current accounting.
   if(!ompb.ForwardAccepted())
     {
      if(!ompb.RollbackCurrentTransaction())
         ReturnFalseEx("Stage 02 Current rollback after rejected forward");
      ReturnFalseEx("Stage 02 target forward rejected");
     }
   if(!ompb.UpdateInputWeights(rank_tcm))
     {
      if(!ompb.RollbackCurrentTransaction())
         ReturnFalseEx("Stage 02 Current rollback after update failure");
      ReturnFalseEx("Stage 02 target update");
     }
   if(!ompb.CommitCurrentTransaction())
     {
      if(!ompb.RollbackCurrentTransaction())
         ReturnFalseEx("Stage 02 Current rollback after commit failure");
      ReturnFalseEx("Stage 02 Current transaction commit");
     }
   return(true);
  }
//+-----------------------------------------------------------------------------------------------------------------------------+
//| Stage 02 online calibration. The outer network stays frozen;  only the ACSRM posterior and alpha receive explicit updates.  |
//+-----------------------------------------------------------------------------------------------------------------------------+
bool ACSRMStage02CalibrateEpoch(void)
  {
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !SkillForecast ||
      !SkillMarket.SetWeightsUpdate(false) || !SkillTarget.SetWeightsUpdate(false) ||
      !SkillForecast.SetCodebookUpdate(false) || !SkillMarket.Clear() ||
      !SkillTarget.Clear() || !SkillForecast.ResetEpochDiagnostics() ||
      !ConfigureACSRM(OMPB_CALIBRATE))
     {
      Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=setup");
      ReturnFalse;
     }
   SkillMarket.TrainMode(true);
   SkillTarget.TrainMode(false);
   ompb.TrainMode(true);
   uint target_attempts = 0;
   uint target_updates = 0;
   uint target_invalid = 0;
   uint target_fallbacks_epoch = 0;
   uint target_kl_rejects_epoch = 0;
   uint source_attempts = 0;
   uint source_updates = 0;
   uint source_invalid = 0;
   uint source_fallbacks_epoch = 0;
   uint source_kl_rejects_epoch = 0;
   uint target_metric_count = 0;
   double target_kl_sum = 0.0;
   double target_disagreement_sum = 0.0;
   double target_alpha_prior_sum = 0.0;
   double target_kl_max = 0.0;
   double target_disagreement_max = 0.0;
   double target_alpha_prior_max = 0.0;
   const uint source_rows = uint(ACSRMStage02ReferenceFirst - ACSRMStage02ReferenceLast + 1);
   const uint target_rows = uint(ACSRMStage02TargetCalibrationFirst -
                                 ACSRMStage02TargetCalibrationLast + 1);
   const uint target_quota = ACSRMStage02ProgressQuota(target_rows, 0, ACSRMStage02Smoke,
                             ACSRMStage02SmokeLimit);
   if(source_rows == 0)
     {
      Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=source_rows");
      ReturnFalse;
     }
   ACSRMStage02ShowProgress("calibration", target_attempts, target_quota, target_attempts,
                           target_invalid, false, false, true, true, source_updates, source_attempts,
                           source_invalid, StringFormat("target_updates=%u", target_updates));
   for(int position = ACSRMStage02TargetCalibrationFirst;
       position >= ACSRMStage02TargetCalibrationLast && !IsStopped(); position--)
     {
      target_attempts++;
      uint target_reference_before, target_current_before;
      uint target_invalid_fallbacks_before, target_kl_rejects_before;
      float target_disagreement_before, target_kl_before, target_alpha_prior_before;
      if(!ReadACSRMDiagnostics(target_reference_before, target_current_before,
                              target_disagreement_before, target_kl_before,
                              target_alpha_prior_before, target_invalid_fallbacks_before,
                              target_kl_rejects_before))
        {
         Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=target_diagnostics_before");
         ReturnFalse;
        }
      const bool target_committed = ACSRMStage02RunBatch(position, false);
      uint target_reference_after, target_current_after;
      uint target_invalid_fallbacks_after, target_kl_rejects_after;
      float target_disagreement_preupdate, target_kl_preupdate, target_alpha_prior_preupdate;
      if(!ReadACSRMDiagnostics(target_reference_after, target_current_after,
                              target_disagreement_preupdate, target_kl_preupdate,
                              target_alpha_prior_preupdate, target_invalid_fallbacks_after,
                              target_kl_rejects_after) ||
         target_invalid_fallbacks_after < target_invalid_fallbacks_before ||
         target_kl_rejects_after < target_kl_rejects_before)
        {
         PrintFormat("ACSRM_STAGE02_CALIBRATION_FAIL reason=target_diagnostics_after position=%d", position);
         ReturnFalse;
        }
      target_fallbacks_epoch += target_invalid_fallbacks_after - target_invalid_fallbacks_before;
      target_kl_rejects_epoch += target_kl_rejects_after - target_kl_rejects_before;
      if(!target_committed)
        {
         target_invalid++;
        }
      else
        {
         if(!ACSRMStage02Finite(target_disagreement_preupdate) ||
            !ACSRMStage02Finite(target_kl_preupdate) ||
            !ACSRMStage02Finite(target_alpha_prior_preupdate))
           {
            PrintFormat("ACSRM_STAGE02_CALIBRATION_FAIL reason=target_diagnostics position=%d", position);
            ReturnFalse;
           }
         if(target_metric_count == 0)
           {
            target_kl_max = double(target_kl_preupdate);
            target_disagreement_max = double(target_disagreement_preupdate);
            target_alpha_prior_max = double(target_alpha_prior_preupdate);
            PrintFormat("ACSRM_STAGE02_INITIAL_KL epoch=%u value=%.9g", ExtACSRMCalibrationEpoch,
                        double(target_kl_preupdate));
           }
         target_kl_sum += double(target_kl_preupdate);
         target_disagreement_sum += double(target_disagreement_preupdate);
         target_alpha_prior_sum += double(target_alpha_prior_preupdate);
         target_kl_max = MathMax(target_kl_max, double(target_kl_preupdate));
         target_disagreement_max = MathMax(target_disagreement_max, double(target_disagreement_preupdate));
         target_alpha_prior_max = MathMax(target_alpha_prior_max, double(target_alpha_prior_preupdate));
         target_metric_count++;
         target_updates++;
         if(ACSRMStage02SourcePeriod > 0 && target_updates % ACSRMStage02SourcePeriod == 0)
           {
            const int source_position = ACSRMStage02ReferenceFirst - int(source_updates % source_rows);
            uint source_reference_before, source_current_before;
            uint source_invalid_fallbacks_before, source_kl_rejects_before;
            float source_disagreement_before, source_kl_before, source_alpha_prior_before;
            if(!ReadACSRMDiagnostics(source_reference_before, source_current_before,
                                    source_disagreement_before, source_kl_before,
                                    source_alpha_prior_before, source_invalid_fallbacks_before,
                                    source_kl_rejects_before))
              {
               Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=anchor_diagnostics_before");
               ReturnFalse;
              }
            source_attempts++;
            const bool source_committed = ACSRMStage02RunBatch(source_position, true);
            uint source_reference_after, source_current_after;
            uint source_invalid_fallbacks_after, source_kl_rejects_after;
            float source_disagreement_after, source_kl_after, source_alpha_prior_after;
            if(!ReadACSRMDiagnostics(source_reference_after, source_current_after,
                                    source_disagreement_after, source_kl_after,
                                    source_alpha_prior_after, source_invalid_fallbacks_after,
                                    source_kl_rejects_after) ||
               source_invalid_fallbacks_after < source_invalid_fallbacks_before ||
               source_kl_rejects_after < source_kl_rejects_before)
              {
               PrintFormat("ACSRM_STAGE02_CALIBRATION_FAIL reason=anchor_diagnostics_after position=%d", source_position);
               ReturnFalse;
              }
            source_fallbacks_epoch += source_invalid_fallbacks_after - source_invalid_fallbacks_before;
            source_kl_rejects_epoch += source_kl_rejects_after - source_kl_rejects_before;
            if(!source_committed)
              {
               source_invalid++;
              }
            else
              {
               if(source_current_after != source_current_before ||
                  source_reference_after != source_reference_before ||
                  !ACSRMStage02Finite(source_disagreement_after) ||
                  !ACSRMStage02Finite(source_kl_after) ||
                  !ACSRMStage02Finite(source_alpha_prior_after))
                 {
                  PrintFormat("ACSRM_STAGE02_CALIBRATION_FAIL reason=anchor_contract position=%d", source_position);
                  ReturnFalse;
                 }
               source_updates++;
              }
           }
        }
      ACSRMStage02ShowProgress("calibration", target_attempts, target_quota, target_attempts,
                              target_invalid, false, false, true, true, source_updates, source_attempts,
                              source_invalid, StringFormat("target_updates=%u", target_updates));
      if(ACSRMStage02Smoke && target_attempts >= ACSRMStage02SmokeLimit)
         break;
     }
//--- A stopped epoch is terminal and cannot be reported as calibration failure.
   if(IsStopped())
     {
      ACSRMStage02ShowProgress("calibration", target_attempts, target_quota, target_attempts,
                              target_invalid, true, false, true, true, source_updates,
                              source_attempts, source_invalid,
                              StringFormat("target_updates=%u result=stopped", target_updates));
      return(false);
     }
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(target_attempts < target_quota || target_updates == 0 || target_metric_count == 0 ||
      !SkillForecast.BuildEpochCodebookDiagnostics() ||
      !SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence, latent,
            observation, valid, invalid, entropy, distance,
            inactive, recovered) ||
      valid < double(target_updates) || !ACSRMStage02Finite(lmix) ||
      !ACSRMStage02Finite(valid) || !ACSRMStage02Finite(invalid) ||
      !SkillVerifyFrozenForecastExact())
     {
      PrintFormat("ACSRM_STAGE02_CALIBRATION_FAIL reason=final_metrics target=%u source=%u",
                  target_updates, source_updates);
      ACSRMStage02ShowProgress("calibration", target_attempts, target_quota, target_attempts,
                              target_invalid, true, false, true, true, source_updates, source_attempts,
                              source_invalid, StringFormat("target_updates=%u result=failed", target_updates));
      ReturnFalse;
     }
   ACSRMStage02ShowProgress("calibration", target_attempts, target_quota, target_attempts,
                           target_invalid, true, true, true, true, source_updates, source_attempts,
                           source_invalid, StringFormat("target_updates=%u", target_updates));
   PrintFormat("ACSRM_STAGE02_CALIBRATION_PASS target_attempts=%u target_updates=%u target_invalid=%u " +
               "source_attempts=%u source_updates=%u source_invalid=%u loss=%.9f reference=%u current=%u smoke=%s",
               target_attempts, target_updates, target_invalid, source_attempts, source_updates,
               source_invalid, lmix / valid, ompb.ReferenceCount(), ompb.CurrentCount(),
               (ACSRMStage02Smoke ? "true" : "false"));
   PrintFormat("ACSRM_STAGE02_EPOCH_PASS epoch=%u epochs=%u loss_all_calls=%.9f " +
               "target_kl_preupdate_mean=%.9g target_kl_preupdate_max=%.9g " +
               "target_disagreement_preupdate_mean=%.9g target_disagreement_preupdate_max=%.9g " +
               "target_alpha_prior_preupdate_mean=%.9g target_alpha_prior_preupdate_max=%.9g " +
               "target_failed_batches=%u source_failed_batches=%u " +
               "target_fallbacks_epoch=%u source_fallbacks_epoch=%u " +
               "target_kl_rejects_epoch=%u source_kl_rejects_epoch=%u source_updates=%u",
               ExtACSRMCalibrationEpoch, ExtACSRMCalibrationEpochs, lmix / valid,
               target_kl_sum / double(target_metric_count), target_kl_max,
               target_disagreement_sum / double(target_metric_count), target_disagreement_max,
               target_alpha_prior_sum / double(target_metric_count), target_alpha_prior_max,
               target_invalid, source_invalid, target_fallbacks_epoch, source_fallbacks_epoch,
               target_kl_rejects_epoch, source_kl_rejects_epoch, source_updates);
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs chronological calibration epochs with retained parameters.  |
//+------------------------------------------------------------------+
bool ACSRMStage02Calibrate(void)
  {
//--- Clear in the epoch setup retains posterior, alpha and optimizer state.
   for(uint epoch = 0; epoch < ExtACSRMCalibrationEpochs; epoch++)
     {
      if(IsStopped())
         return(false);
      ExtACSRMCalibrationEpoch = epoch + 1;
      PrintFormat("ACSRM_STAGE02_EPOCH_BEGIN epoch=%u epochs=%u",
                  ExtACSRMCalibrationEpoch, ExtACSRMCalibrationEpochs);
      ACSRMStage02ShowProgress("epoch setup", 0, 1, 0, 0, false, false, false, false,
                              0, 0, 0, StringFormat("epoch=%u/%u", ExtACSRMCalibrationEpoch,
                                    ExtACSRMCalibrationEpochs));
      if(!ACSRMStage02CalibrateEpoch())
         return(false);
     }
//--- Only a complete run may proceed to held-out evaluation and publication.
   return(!IsStopped() && ExtACSRMCalibrationEpoch == ExtACSRMCalibrationEpochs);
  }
//+------------------------------------------------------------------+
//| Creates the Stage 02 reference study fixtures.                   |
//+------------------------------------------------------------------+
bool CreateACSRMStage02ReferenceStudy(const datetime reference_start, const datetime reference_end,
                                     const datetime calibration_start, const datetime calibration_end,
                                     const uint anchor_period, const float tau_value,
                                     const float lambda_dis, const float lambda_kl,
                                     const float lambda_alpha, const float alpha_prior,
                                     const float max_kl, const bool smoke, const uint smoke_limit,
                                     const uint calibration_epochs = 1)
  {
   if(calibration_epochs == 0)
      ReturnFalseEx("calibration epochs is zero");
   ExtACSRMCalibrationEpochs = (smoke ? 1 : calibration_epochs);
   ExtACSRMCalibrationEpoch = 0;
   ACSRMStage02SignaturesCaptured = false;
   ACSRMStage02StopReported = false;
   PrintFormat("ACSRM_STAGE02_EPOCH_CONFIG requested=%u effective=%u smoke=%s",
               calibration_epochs, ExtACSRMCalibrationEpochs, (smoke ? "true" : "false"));
   ResetLastError();
   SkillReady = false;
   SkillForecast = NULL;
   SkillCompletedEpochs = 0;
   SkillBatches = 0;
   SkillInvalidBatches = 0;
   if(!SkillLoadForecastTraining())
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=stage01_checkpoint error=%d", GetLastError());
      ReturnFalse;
     }
   CNeuronBaseOCL *rank_tcm = SkillMarket.Layer(4);
   CNeuronBaseOCL *ompb_layer = SkillMarket.Layer(5);
   CNeuronBaseOCL *forecast_layer = SkillMarket.Layer(6);
   if(!SkillForecast || !rank_tcm || !ompb_layer || !forecast_layer ||
      rank_tcm.Type() != defNeuronCogDriverRankTCM || ompb_layer.Type() != defNeuronOMPBOCL ||
      forecast_layer.Type() != defNeuronScenarioForecast || !GetACSRM() ||
      !SkillValidateShapes() || !SkillInitIndicators() ||
      !SkillMarket.SetWeightsUpdate(false) || !SkillTarget.SetWeightsUpdate(false) ||
      !SkillForecast.SetGradientNormalization(true))
     {
      Print("ACSRM_STAGE02_PREFLIGHT_FAIL reason=loaded_contract");
      ReturnFalse;
     }
   if(!ACSRMStage02Configure(reference_start, reference_end, calibration_start, calibration_end,
                            anchor_period, tau_value, lambda_dis, lambda_kl, lambda_alpha,
                            alpha_prior, max_kl, smoke, smoke_limit))
      ReturnFalse;
   SkillTarget.TrainMode(false);
   SkillMarket.TrainMode(true);
   GetACSRM().TrainMode(false);
   SkillReady = true;
   if(!EventChartCustom(ChartID(), 1, 0, 0, "ACSRM_STAGE02_INIT"))
     {
      PrintFormat("ACSRM_STAGE02_PREFLIGHT_FAIL reason=chart_event error=%d", GetLastError());
      SkillReady = false;
      ReturnFalse;
     }
   Print("ACSRM_STAGE02_CHECKPOINT_PASS mode=load_only production_save=forbidden");
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the Stage 02 reference study end to end.                    |
//+------------------------------------------------------------------+
void RunACSRMStage02ReferenceStudy(void)
  {
   ACSRMStage02ShowProgress("preparing", 0, 1, 0, 0, false, false, false, false,
                           0, 0, 0, "");
   if(!SkillReady || !ACSRMStage02PrepareData() || !ACSRMStage02CollectReference())
     {
      if(ACSRMStage02StopHandled("reference"))
         return;
      ACSRMStage02ShowProgress("preparing", 0, 1, 0, 0, true, false, false, false,
                              0, 0, 0,
                              "result=failed");
      ExpertRemove();
      return;
     }
   if(!SkillForecast.SetCodebookUpdate(false))
     {
      Print("ACSRM_STAGE02_REFERENCE_FAIL reason=codebook_freeze");
      ExpertRemove();
      return;
     }
   if(!SkillCaptureFrozenForecastBaseline())
     {
      Print("ACSRM_STAGE02_REFERENCE_FAIL reason=frozen_baseline");
      ExpertRemove();
      return;
     }
   double source_loss = 0.0, target_loss = 0.0;
   uint source_valid, source_invalid, target_valid, target_invalid;
   const bool source_ok = ACSRMStage02Baseline(ACSRMStage02SourceEvalFirst,
                          ACSRMStage02SourceEvalLast, "source_eval",
                          source_loss, source_valid, source_invalid);
   if(ACSRMStage02StopHandled("baseline_source"))
      return;
   const bool target_ok = (source_ok &&
                           ACSRMStage02Baseline(ACSRMStage02TargetEvalFirst,
                                 ACSRMStage02TargetEvalLast, "target_eval",
                                 target_loss, target_valid, target_invalid));
   if(ACSRMStage02StopHandled("baseline_target"))
      return;
   bool calibration_ok = false;
   bool evaluation_ok = false;
   bool acceptance_ok = false;
   bool checkpoint_saved = false;
    double pre_source_inference_loss = 0.0, pre_target_inference_loss = 0.0;
    uint pre_source_inference_valid = 0, pre_source_inference_invalid = 0;
    uint pre_target_inference_valid = 0, pre_target_inference_invalid = 0;
    double post_source_inference_loss = 0.0, post_target_inference_loss = 0.0;
    uint post_source_inference_valid = 0, post_source_inference_invalid = 0;
    uint post_target_inference_valid = 0, post_target_inference_invalid = 0;
   if(target_ok && SkillVerifyFrozenForecastExact() && ACSRMStage02CaptureSignatures())
     {
      PrintFormat("ACSRM_STAGE02_REFERENCE_READY source_loss=%.9f target_loss=%.9f source_valid=%u target_valid=%u",
                  source_loss, target_loss, source_valid, target_valid);
      const bool pre_source_ok =
         (ConfigureACSRM(OMPB_INFERENCE) &&
          ACSRMStage02InferenceEvaluation(ACSRMStage02SourceEvalFirst, ACSRMStage02SourceEvalLast,
                                           "pre_source_inference", pre_source_inference_loss,
                                           pre_source_inference_valid, pre_source_inference_invalid));
      if(ACSRMStage02StopHandled("pre_inference_source"))
         return;
      const bool pre_target_ok =
         (pre_source_ok && ConfigureACSRM(OMPB_INFERENCE) &&
          ACSRMStage02InferenceEvaluation(ACSRMStage02TargetEvalFirst, ACSRMStage02TargetEvalLast,
                                          "pre_target_inference", pre_target_inference_loss,
                                          pre_target_inference_valid, pre_target_inference_invalid));
      if(ACSRMStage02StopHandled("pre_inference_target"))
         return;
      const bool pre_signatures_ok = ACSRMStage02VerifyPreCalibrationSignatures();
      const bool pre_inference_ok = (pre_source_ok && pre_target_ok && pre_signatures_ok);
      PrintFormat("ACSRM_STAGE02_PREUPDATE_INFERENCE mode=inference source_loss=%.9f " +
                  "target_loss=%.9f source_valid=%u source_invalid=%u target_valid=%u " +
                  "target_invalid=%u signatures_unchanged=%s status=%s",
                   pre_source_inference_loss, pre_target_inference_loss, pre_source_inference_valid,
                   pre_source_inference_invalid, pre_target_inference_valid, pre_target_inference_invalid,
                  (pre_signatures_ok ? "true" : "false"),
                  (pre_inference_ok ? "pass" : "fail"));
      if(pre_inference_ok && SkillForecast.SetGradientNormalization(false))
        {
         Print("ACSRM_STAGE02_GRADIENT_MODE mode=raw_forecast");
         calibration_ok = ACSRMStage02Calibrate();
         if(ACSRMStage02StopHandled("calibration"))
            return;
        }
      else
        {
         if(!pre_inference_ok)
            Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=preupdate_inference");
         else
            Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=gradient_mode");
        }
      if(!SkillForecast.SetGradientNormalization(true))
        {
         Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=gradient_mode_restore");
         calibration_ok = false;
        }
     }
   else
      Print("ACSRM_STAGE02_REFERENCE_FAIL reason=baseline_frozen_or_signatures");
//--- Evaluation is intentionally distinct from the frozen BYPASS baselines.
//--- It uses posterior-mean ACSRM inference and must be complete before any
//--- candidate checkpoint can be staged.
   if(calibration_ok && ConfigureACSRM(OMPB_INFERENCE))
     {
      const bool source_inference_ok =
         ACSRMStage02InferenceEvaluation(ACSRMStage02SourceEvalFirst, ACSRMStage02SourceEvalLast,
                                         "source_inference", post_source_inference_loss,
                                         post_source_inference_valid, post_source_inference_invalid);
      if(ACSRMStage02StopHandled("post_inference_source"))
         return;
      const bool target_inference_ok =
         (source_inference_ok &&
          ACSRMStage02InferenceEvaluation(ACSRMStage02TargetEvalFirst, ACSRMStage02TargetEvalLast,
                                          "target_inference", post_target_inference_loss,
                                          post_target_inference_valid, post_target_inference_invalid));
      if(ACSRMStage02StopHandled("post_inference_target"))
         return;
       evaluation_ok = (target_inference_ok && ACSRMStage02VerifySignatures());
       if(evaluation_ok)
         {
          const double continuation_source_denominator = MathAbs(pre_source_inference_loss);
          const double continuation_target_denominator = MathAbs(pre_target_inference_loss);
          const bool continuation_finite =
             (ACSRMStage02Finite(pre_source_inference_loss) &&
              ACSRMStage02Finite(pre_target_inference_loss) &&
              ACSRMStage02Finite(post_source_inference_loss) &&
              ACSRMStage02Finite(post_target_inference_loss) &&
              continuation_source_denominator > 0.0 && continuation_target_denominator > 0.0);
          const double continuation_target_improvement =
             (continuation_finite ?
              (pre_target_inference_loss - post_target_inference_loss) /
              continuation_target_denominator : 0.0);
          const double continuation_source_degradation =
             (continuation_finite ?
              (post_source_inference_loss - pre_source_inference_loss) /
              continuation_source_denominator : 0.0);
          const bool continuation_status =
             (continuation_finite && pre_source_inference_valid > 0 &&
              pre_target_inference_valid > 0 && post_source_inference_valid > 0 &&
              post_target_inference_valid > 0);
          PrintFormat("ACSRM_STAGE02_CONTINUATION target_improvement=%.8f " +
                      "source_degradation=%.8f pre_source_loss=%.9f " +
                      "post_source_loss=%.9f pre_target_loss=%.9f " +
                      "post_target_loss=%.9f pre_source_valid=%u " +
                      "pre_source_invalid=%u post_source_valid=%u " +
                      "post_source_invalid=%u pre_target_valid=%u " +
                      "pre_target_invalid=%u post_target_valid=%u " +
                      "post_target_invalid=%u finite=%s status=%s",
                      continuation_target_improvement, continuation_source_degradation,
                      pre_source_inference_loss, post_source_inference_loss,
                      pre_target_inference_loss, post_target_inference_loss,
                      pre_source_inference_valid, pre_source_inference_invalid,
                      post_source_inference_valid, post_source_inference_invalid,
                      pre_target_inference_valid, pre_target_inference_invalid,
                      post_target_inference_valid, post_target_inference_invalid,
                      (continuation_finite ? "true" : "false"),
                      (continuation_status ? "pass" : "fail"));
          acceptance_ok = ACSRMStage02Accept(source_loss, target_loss,
                                            post_source_inference_loss,
                                            post_target_inference_loss,
                                            post_source_inference_valid,
                                            post_target_inference_valid);
         }
      else
         Print("ACSRM_STAGE02_ACCEPTANCE_FAIL reason=evaluation_or_signatures");
     }
   else
      Print("ACSRM_STAGE02_ACCEPTANCE_FAIL reason=calibration");
//--- Switching to deterministic inference and clearing ACSRM drops only Current
//--- and transient buffers.  Clear deliberately preserves posterior parameters.
   CNeuronOMPBOCL *ompb = GetACSRM();
   const bool finalized = (ompb && ConfigureACSRM(OMPB_INFERENCE) && ompb.Clear());
   if(!finalized)
      Print("ACSRM_STAGE02_CALIBRATION_FAIL reason=finalize");
   else
     {
      ompb.TrainMode(false);
      //--- Quality acceptance is diagnostic; unchanged frozen parameters remain mandatory.
      const bool checkpoint_integrity = ACSRMStage02VerifySignatures();
      ACSRMStage02ShowProgress("saving", 0, 1, 0, 0, false, false, false, false,
                              0, 0, 0,
                              StringFormat("accepted=%s smoke=%s",
                                           (acceptance_ok ? "true" : "false"),
                                           (ACSRMStage02Smoke ? "true" : "false")));
      if(calibration_ok && checkpoint_integrity && !ACSRMStage02Smoke && !IsStopped())
         checkpoint_saved = ACSRMStage02SaveAcceptedCheckpoint();
      else
         if(ACSRMStage02Smoke)
            PrintFormat("ACSRM_STAGE02_TRANSACTION_SKIPPED reason=smoke accepted=%s",
                        (acceptance_ok ? "true" : "false"));
         else
            Print("ACSRM_STAGE02_TRANSACTION_SKIPPED reason=integrity");
      ACSRMStage02ShowProgress("saving", (checkpoint_saved ? 1 : 0), 1, 1, 0, true,
                              checkpoint_saved || ACSRMStage02Smoke, false, false,
                              0, 0, 0,
                              StringFormat("accepted=%s saved=%s smoke=%s",
                                           (acceptance_ok ? "true" : "false"),
                                           (checkpoint_saved ? "true" : "false"),
                                           (ACSRMStage02Smoke ? "true" : "false")));
      CNeuronOMPBOCL *final_ompb = GetACSRM();
      if(!final_ompb)
         Print("ACSRM_STAGE02_INFERENCE_READY_FAIL reason=ompb_after_checkpoint");
      else
         PrintFormat("ACSRM_STAGE02_INFERENCE_READY calibration=%s evaluation=%s accepted=%s saved=%s " +
                     "reference=%u current=%u", (calibration_ok ? "true" : "false"),
                     (evaluation_ok ? "true" : "false"), (acceptance_ok ? "true" : "false"),
                     (checkpoint_saved ? "true" : "false"), final_ompb.ReferenceCount(),
                     final_ompb.CurrentCount());
     }
   SkillMarket.TrainMode(false);
   SkillTarget.TrainMode(false);
   const uint final_rows = uint(ACSRMStage02TargetEvalFirst - ACSRMStage02TargetEvalLast + 1);
   const uint final_quota = ACSRMStage02ProgressQuota(final_rows, 0, ACSRMStage02Smoke,
                            ACSRMStage02SmokeLimit);
   string final_detail = StringFormat("source_degradation=n/a target_improvement=n/a accepted=%s saved=%s",
                                      (acceptance_ok ? "true" : "false"),
                                      (checkpoint_saved ? "true" : "false"));
   if(target_ok && evaluation_ok)
     {
      const double source_denominator = MathMax(MathAbs(source_loss), 1.0e-12);
      const double target_denominator = MathMax(MathAbs(target_loss), 1.0e-12);
       const double source_degradation = (post_source_inference_loss - source_loss) /
                                         source_denominator;
       const double target_improvement = (target_loss - post_target_inference_loss) /
                                         target_denominator;
      final_detail = StringFormat("source_degradation=%.2f%% target_improvement=%.2f%% accepted=%s saved=%s",
                                  100.0 * source_degradation, 100.0 * target_improvement,
                                  (acceptance_ok ? "true" : "false"),
                                  (checkpoint_saved ? "true" : "false"));
     }
    const uint final_done = (ACSRMStage02Smoke ? post_target_inference_valid :
                             post_target_inference_valid + post_target_inference_invalid);
    ACSRMStage02ShowProgress("final", final_done, final_quota,
                            post_target_inference_valid + post_target_inference_invalid,
                            post_target_inference_invalid, true, finalized && calibration_ok && evaluation_ok,
                            false, false, post_source_inference_valid,
                            post_source_inference_valid + post_source_inference_invalid,
                            post_source_inference_invalid,
                           final_detail);
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Creates or initializes SkillForecastStudy.                       |
//+------------------------------------------------------------------+
bool CreateSkillForecastStudy(void)
  {
   ResetLastError();
   SkillCompletedEpochs = 0;
   SkillBatches = 0;
   SkillInvalidBatches = 0;
//--- Always prefer an existing compatible checkpoint for continued training.
//--- Only a clean initial checkpoint miss falls back to a new randomized graph.
//--- Partial, incompatible and transaction states remain fail-closed.
   bool resumed = SkillLoadForecastTraining();
//--- Function if.
   if(!resumed)
     {
      if(ACSRMStage02RecoveryFailed)
        {
         Print("ACSRM checkpoint recovery=FAIL before=training_create fail_closed=true");
         ReturnFalse;
        }
      const int load_error = GetLastError();
      SkillForecast = NULL;
      PrintFormat("%s init: forecast restore=FAIL error=%d; creating new random model", ACSRM_LOG_PREFIX, load_error);
      ResetLastError();
      //--- Function if.
      if(!SkillCreateNetworks())
        {
         PrintFormat("%s init: forecast create=FAIL error=%d", ACSRM_LOG_PREFIX, GetLastError());
         ReturnFalse;
        }
     }
//--- Function if.
   if(!SkillValidateShapes())
     {
      PrintFormat("%s init: shapes=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
//--- Stage 01 keeps ACSRM a non-trainable identity bridge.  Its posterior,
//--- alpha and both histories remain untouched while ScenarioForecast trains.
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!ompb || !ConfigureACSRM(OMPB_BYPASS))
      ReturnFalse;
   ompb.TrainMode(false);
   if(!SkillInitIndicators())
     {
      PrintFormat("%s init: indicators=FAIL error=%d", ACSRM_LOG_PREFIX, GetLastError());
      ReturnFalse;
     }
   PrintFormat("%s init: forecast=%s completed_epochs=%u batches=%I64u invalid=%I64u", ACSRM_LOG_PREFIX,
               (resumed ? "RESUMED" : "NEW"), SkillCompletedEpochs, SkillBatches, SkillInvalidBatches);
   SkillReady = true;
//--- Function if.
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("%s init: chart event=FAIL error=%d", ACSRM_LOG_PREFIX, GetLastError());
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements ReleaseSkillForecastStudy.                            |
//+------------------------------------------------------------------+
void ReleaseSkillForecastStudy(const int reason)
  {
//--- Complete epochs save transactionally in TrainSkillForecast(). Never
//--- overwrite a valid checkpoint with a fresh, partial or failed run here.
   SkillForecast = NULL;
   SkillReady = false;
  }
//+------------------------------------------------------------------+
//| Implements SkillPrepareLatentTarget.                             |
//+------------------------------------------------------------------+
bool SkillPrepareLatentTarget(CBufferFloat *market_latent)
  {
   if(!market_latent || market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0 ||
      SkillFuture.Total() != (NForecast * BarDescr) || SkillFuture.GetIndex() < 0 ||
      SkillLatentTarget.Total() != (BarDescr * NForecast * EmbeddingSize) || SkillLatentTarget.GetIndex() < 0 ||
      SkillLatentDelta.Total() != (BarDescr * NForecast * EmbeddingSize) || SkillLatentDelta.GetIndex() < 0)
      ReturnFalse;
//--- SkillFuture is decoder-native [H,B].  The existing transpose changes
//--- only this device layout to Target-native [B,H]; Market keeps its original
//--- [BarDescr,HistoryBars] representation and is never transposed.
   if(!SkillFutureView.Bind(GetPointer(SkillFuture)))
      ReturnFalse;
   const bool transposed = SkillFutureTranspose.FeedForward(SkillFutureView.AsObject());
   SkillFutureView.Unbind();
   if(!transposed)
      ReturnFalse;
   CNeuronBaseOCL *target_input = SkillTarget.Layer(0);
   CNeuronBaseOCL *future_latent = SkillTarget.Layer(-1);
   if(!target_input || target_input.getOutput().Total() != (BarDescr * NForecast) ||
      target_input.getOutputIndex() < 0 ||
      !SkillDevice.Copy(SkillFutureTranspose.getOutput(), target_input.getOutput(),
                          BarDescr * NForecast) ||
      !SkillTarget.feedForward(GetPointer(SkillTarget), 0, (CBufferFloat*)NULL))
      ReturnFalse;
   if(!future_latent || future_latent.getOutput().Total() != (BarDescr * NForecast * EmbeddingSize) ||
      future_latent.getOutputIndex() < 0)
      ReturnFalse;
//--- Forecast loss compares full future Z against full generator Z. Codebook
//--- EMA alone receives the detached delta future-current.
   if(!SkillDevice.Copy(future_latent.getOutput(), GetPointer(SkillLatentTarget),
                          BarDescr * NForecast * EmbeddingSize) ||
      !SkillDevice.Subtract(GetPointer(SkillLatentZero), market_latent,
                              GetPointer(SkillLatentNegativeMarket), EmbeddingSize))
      ReturnFalse;
   if(!SkillDevice.BroadcastSum(GetPointer(SkillLatentNegativeMarket),
                                  future_latent.getOutput(), GetPointer(SkillLatentDelta),
                                  EmbeddingSize, BarDescr))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillProbeNet.                                        |
//+------------------------------------------------------------------+
bool SkillProbeNet(CNet &net, CBufferFloat *probe_state, double &latent[], const bool restore_training)
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
//| Implements SkillProbeDrift.                                      |
//+------------------------------------------------------------------+
double SkillProbeDrift(const double &before[], const double &after[])
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
bool SkillCollectRecoverySmoke(ulong &hits[], double &responsibility_sum[], double &scale_sum[],
                                 double &absolute_sum[], double &absolute_over_scale_sum[])
  {
   if(!SkillForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios)
      ReturnFalse;
   CBufferFloat *responsibilities = SkillForecast.GetResponsibilities();
   CBufferFloat *confidence = SkillForecast.GetConfidence();
   CBufferFloat *target = SkillForecast.GetTarget();
   CBufferFloat *forecast = SkillForecast.getOutput();
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
//| Logs the recovery-smoke responsibility and scale statistics.     |
//+------------------------------------------------------------------+
bool SkillLogRecoverySmoke(const ulong &hits[], const double &responsibility_sum[],
                             const double &scale_sum[], const double &absolute_sum[],
                             const double &absolute_over_scale_sum[],
                             const uint &recovered[], const uint &age_zero[])
  {
   if(!SkillForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios ||
      ArraySize(recovered) != NScenarios || ArraySize(age_zero) != NScenarios)
      ReturnFalse;
   CScenarioCodebook *codebook = SkillForecast.GetCodebook();
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
                  ACSRM_LOG_PREFIX, scenario, hits[scenario], age_zero[scenario],
                  responsibility_sum[scenario], mean_scale, mean_absolute,
                  mean_absolute_over_scale, ages[scenario], counts[scenario], usage[scenario],
                  int(MathRound(inactive[scenario])), recovered[scenario]);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillTrainBatch.                                      |
//+------------------------------------------------------------------+
bool SkillTrainBatch(const int position)
  {
//--- Build exactly the same X<=t state as CreateBuffers(position,...,future),
//--- but do not touch any future sample until Market and Scenario are complete.
   if(!CreateBuffers(position + NForecast, GetPointer(SkillState), GetPointer(SkillTime), NULL))
      ReturnFalse;
   if(!SkillMarket.feedForward(GetPointer(SkillState), 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
//--- Target supervision remains bound to raw RankTCM z_t.  ACSRM sits between
//--- it and ScenarioForecast, so relative indexing would select the bridge.
   CNeuronBaseOCL *market_layer = GetRankTCM();
   CBufferFloat *market_latent = (market_layer ? market_layer.getOutput() : NULL);
   if(!market_latent ||
      market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0)
      ReturnFalse;
//--- Only now is any future window presented to the detached Target Encoder.
   if(!CreateBuffers(position, GetPointer(SkillState), GetPointer(SkillTime), GetPointer(SkillFuture)))
      ReturnFalse;
   if(!SkillPrepareLatentTarget(market_latent))
      ReturnFalse;
   const ulong responsibility_started = GetMicrosecondCount();
//--- Function if.
   if(!SkillForecast.BuildResponsibilities(GetPointer(SkillLatentTarget), GetPointer(SkillLatentDelta)))
     {
      //--- Commit only a completed device validation.  Infrastructure failures leave
      //--- the transient control-valid flag false, so stale batch_control is ignored.
      if(SkillForecast.LastBatchControlValid() && !SkillForecast.CommitBatchDiagnostics())
         PrintFormat("%s -> %d invalid diagnostic commit failed", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   const ulong responsibility_elapsed = GetMicrosecondCount() - responsibility_started;
   if(!SkillForecast.CommitBatchDiagnostics())
      ReturnFalse;
//--- The unified CNet updates ScenarioForecast and all preceding Market layers.
   if(!SkillMarket.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, -1, true))
      ReturnFalse;
   SkillResponsibilityMicroseconds += responsibility_elapsed;
   return(true);
  }
//+-------------------------------------------------------------------+
//| Current-epoch progress uses the existing 12-float device summ...  |
//+-------------------------------------------------------------------+
void SkillShowForecastProgress(const double percent, const ulong failed_batches)
  {
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(SkillForecast && SkillForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence,
         latent, observation, valid, invalid, entropy, distance, inactive, recovered) && valid > 0.0)
     {
      Comment(StringFormat("%s Forecast %6.2f%% L_mix %.8f latent %.8f invalid(current epoch) %I64u",
                           ACSRM_LOG_PREFIX, percent, lmix / valid, latent / valid, failed_batches));
      return;
     }
   Comment(StringFormat("%s Forecast %6.2f%% L_mix n/a latent n/a invalid(current epoch) %I64u",
                        ACSRM_LOG_PREFIX, percent, failed_batches));
  }
//+------------------------------------------------------------------+
//| Implements TrainSkillForecast.                                   |
//+------------------------------------------------------------------+
#ifndef Online
void TrainSkillForecast(void)
  {
   if(!SkillReady)
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
   if(!CreateBuffers(probe_position + NForecast, GetPointer(SkillProbeState),
//--- Gets Pointer.
                     GetPointer(SkillProbeTime), NULL))
     { PrintFormat("%s -> %d fixed probe unavailable", __FUNCTION__, __LINE__); return; }
   uint ticks = GetTickCount();
   bool stop = false;
   const uint passes = (SkillRecoverySmoke ? 1 : uint(MathMax(Epochs, 0)));
   const ulong smoke_limit = (SkillRecoverySmoke ? ulong(MathMax(1, int(SkillRecoverySmokeBatches))) : 0);
   for(uint pass = 0; pass < passes && !IsStopped() && !stop; pass++)
     {
      const uint epoch = SkillCompletedEpochs;
      ulong epoch_failures = 0;
      ulong smoke_batches = 0;
      ulong smoke_hits[];
      double smoke_responsibility_sum[];
      double smoke_scale_sum[];
      double smoke_absolute_sum[];
      double smoke_absolute_over_scale_sum[];
      if(SkillRecoverySmoke &&
         (ArrayResize(smoke_hits, NScenarios) != NScenarios ||
          ArrayResize(smoke_responsibility_sum, NScenarios) != NScenarios ||
          ArrayResize(smoke_scale_sum, NScenarios) != NScenarios ||
          ArrayResize(smoke_absolute_sum, NScenarios) != NScenarios ||
          //--- All temporary smoke buffers must resize to NScenarios.
          ArrayResize(smoke_absolute_over_scale_sum, NScenarios) != NScenarios))
        { PrintFormat("%s -> %d smoke counter allocation failed", __FUNCTION__, __LINE__); break; }
      ArrayInitialize(smoke_hits, 0);
      ArrayInitialize(smoke_responsibility_sum, 0.0);
      ArrayInitialize(smoke_scale_sum, 0.0);
      ArrayInitialize(smoke_absolute_sum, 0.0);
      ArrayInitialize(smoke_absolute_over_scale_sum, 0.0);
      if(!SkillMarket.Clear() || !SkillTarget.Clear())
        { PrintFormat("%s -> %d clear failed", __FUNCTION__, __LINE__); break; }
      SkillTarget.TrainMode(false);
      double latent_before[], latent_after[];
      if(!SkillProbeNet(SkillMarket, GetPointer(SkillProbeState), latent_before, true) ||
         !ConfigureACSRM(OMPB_BYPASS))
        { PrintFormat("%s -> %d fixed probe forward failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!SkillForecast.ResetEpochDiagnostics())
        { PrintFormat("%s -> %d diagnostic reset failed", __FUNCTION__, __LINE__); stop = true; break; }
      uint ompb_reference_count_before, ompb_current_count_before;
      uint ompb_invalid_fallbacks_before, ompb_kl_rejects_before;
      float ompb_disagreement_before, ompb_kl_before, ompb_alpha_prior_before;
      if(!ReadACSRMDiagnostics(ompb_reference_count_before, ompb_current_count_before,
                              ompb_disagreement_before, ompb_kl_before,
                              ompb_alpha_prior_before, ompb_invalid_fallbacks_before,
                              ompb_kl_rejects_before))
        {
         PrintFormat("ACSRM_STAGE01_BYPASS_DIAGNOSTIC_FAIL scope=epoch-%u reason=read_before", epoch + 1);
         stop = true;
         break;
        }
      SkillResponsibilityMicroseconds = 0;
      for(int position = start - HistoryBars - NForecast - 1; position >= end && !IsStopped() && !stop; position--)
        {
         //--- Function if.
         if(!SkillTrainBatch(position))
           {
            epoch_failures++;
            PrintFormat("%s invalid batch epoch=%d position=%d line=%d", ACSRM_LOG_PREFIX, epoch, position, __LINE__);
            //--- Function if.
            if(GetTickCount() - ticks > 500)
              {
               const double percent = (double(pass) + 1.0 -
                                       double(position - end) /
                                       MathMax(start - end - HistoryBars - NForecast, 1)) *
                                      100.0 / Epochs;
               SkillShowForecastProgress(percent, epoch_failures);
               ticks = GetTickCount();
              }
            continue;
           }
         //--- Function if.
         if(SkillRecoverySmoke)
           {
            if(!SkillCollectRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
                                            smoke_absolute_sum, smoke_absolute_over_scale_sum))
              { PrintFormat("%s -> %d smoke responsibility read failed", __FUNCTION__, __LINE__); stop = true; break; }
            smoke_batches++;
            if(smoke_batches >= smoke_limit)
               break;
           }
         //--- Function if.
         if(GetTickCount() - ticks > 500)
           {
            const double percent = (double(pass) + 1.0 -
                                    double(position - end) /
                                    MathMax(start - end - HistoryBars - NForecast, 1)) *
                                   100.0 / Epochs;
            SkillShowForecastProgress(percent, epoch_failures);
            ticks = GetTickCount();
           }
        }
      double lmix, lrouter, ltrajectory, lconfidence, latent, observation;
      double valid_value, invalid_value, entropy, distance, inactive_value, recovered_value;
      if(!SkillForecast.BuildEpochCodebookDiagnostics() ||
         !SkillForecast.ReadEpochDiagnostics(lmix, lrouter, ltrajectory, lconfidence, latent, observation,
               valid_value, invalid_value, entropy, distance, inactive_value, recovered_value))
        { PrintFormat("%s -> %d diagnostic read failed", __FUNCTION__, __LINE__); stop = true; break; }
      const ulong valid = (ulong)MathRound(valid_value);
      const ulong invalid = (ulong)MathRound(invalid_value);
      const uint inactive = (uint)MathRound(inactive_value);
      const uint recovered = (uint)MathRound(recovered_value);
      uint recovered_by_scenario[];
      uint age_zero_by_scenario[];
      if(!SkillForecast.ReadEpochRecoveryEvents(recovered_by_scenario))
        { PrintFormat("%s -> %d recovery-event read failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!SkillForecast.ReadEpochAgeZeroEvents(age_zero_by_scenario))
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
      if(SkillRecoverySmoke && valid != smoke_batches)
        {
         PrintFormat("%s -> %d smoke valid mismatch valid=%I64u collected=%I64u",
                     __FUNCTION__, __LINE__, valid, smoke_batches);
         stop = true;
         break;
        }
      SkillBatches += valid;
      SkillInvalidBatches += invalid;
      if(valid == 0)
        { stop = true; break; }
      if(!SkillCheckACSRMBypassInvariant(ompb_reference_count_before, ompb_current_count_before,
                                          ompb_disagreement_before, ompb_kl_before, ompb_alpha_prior_before,
                                          ompb_invalid_fallbacks_before, ompb_kl_rejects_before,
                                          StringFormat("epoch-%u", epoch + 1)))
        { stop = true; break; }
      //--- Target is an independently initialized inference-only encoder.  It is
      //--- never copied from Market and is absent from every backward/update path.
      if(!SkillProbeNet(SkillMarket, GetPointer(SkillProbeState), latent_after, true) ||
         !ConfigureACSRM(OMPB_BYPASS))
        { PrintFormat("%s -> %d Market epoch probe failed", __FUNCTION__, __LINE__); stop = true; break; }
      const double latent_drift = SkillProbeDrift(latent_before, latent_after);
      PrintFormat("%s epoch=%d batches=%I64u L_mix=%.8f L_router=%.8f L_trajectory=%.8f " +
                  "L_confidence=%.8f NLL_confidence=%.8f usage_entropy=%.8f " +
                  "pairwise_codebook=%.8f invalid=%I64u latent_drift=%.8f inactive=%u " +
                  "recovered=%u recovered_by_scenario=%s responsibility_ms=%.3f",
                  ACSRM_LOG_PREFIX, epoch + 1, valid, lmix / valid, lrouter / valid,
                  ltrajectory / valid, latent / valid, lconfidence / valid, entropy,
                  distance, SkillInvalidBatches, latent_drift, inactive, recovered,
                  FormatScenarioCounts(recovered_by_scenario),
                  double(SkillResponsibilityMicroseconds) / (1000.0 * double(valid)));
      if(SkillRecoverySmoke && !SkillLogRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
            smoke_absolute_sum, smoke_absolute_over_scale_sum,
            recovered_by_scenario, age_zero_by_scenario))
        { PrintFormat("%s -> %d smoke state log failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!SkillRecoverySmoke && !SkillSaveCheckpoint(epoch + 1))
        { PrintFormat("%s -> %d checkpoint failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(SkillRecoverySmoke)
         PrintFormat("%s smoke complete valid=%I64u recovery_age=%u checkpoint=SKIPPED", ACSRM_LOG_PREFIX, valid,
                     SkillForecast.RecoveryAge());
      SkillCompletedEpochs = epoch + 1;
     }
   Comment("");
//--- Function if.
   if(!stop)
     {
      //--- Successful forecast stage is sealed for the later Actor-Critic stage.
      SkillMarket.TrainMode(false);
      SkillTarget.TrainMode(false);
      SkillLastSignature = SkillForecastSignature();
      PrintFormat("%s forecast inference-only signature=%I64u batches=%I64u invalid=%I64u", ACSRM_LOG_PREFIX,
                  SkillLastSignature, SkillBatches, SkillInvalidBatches);
     }
   ExpertRemove();
  }
#endif
//+-------------------------------------------------------------------+
//| Actor-Critic inference and composition helpers Implements ORI...  |
//+-------------------------------------------------------------------+
string SkillManifestValue(const string file_name, const string key)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return("");
   const string prefix = key + "=";
   string value = "";
//--- Function while.
   while(!FileIsEnding(handle))
     {
      const string line = FileReadString(handle);
      //--- Function if.
      if(StringFind(line, prefix) == 0)
        {
         value = StringSubstr(line, StringLen(prefix));
         break;
        }
     }
   FileClose(handle);
   return(value);
  }
//+-------------------------------------------------------------------+
//| Implements SkillValidateForecastManifestHeader. Static checkp...  |
//+-------------------------------------------------------------------+
bool SkillValidateForecastManifestHeader(void)
  {
#define Skill_MANIFEST_HEADER_EQ(KEY,VALUE) if(SkillManifestValue(SkillActiveManifestFile,KEY)!=(VALUE)) ReturnFalse
   Skill_MANIFEST_HEADER_EQ("format", "ACSRM_FORECAST");
   Skill_MANIFEST_HEADER_EQ("version", IntegerToString(Skill_FORMAT_VERSION));
   Skill_MANIFEST_HEADER_EQ("forecast_type", IntegerToString(defNeuronScenarioForecast));
   Skill_MANIFEST_HEADER_EQ("variables", IntegerToString(BarDescr));
   Skill_MANIFEST_HEADER_EQ("scenarios", IntegerToString(NScenarios));
   Skill_MANIFEST_HEADER_EQ("top_k", IntegerToString(TopK));
   Skill_MANIFEST_HEADER_EQ("horizon", IntegerToString(NForecast));
   Skill_MANIFEST_HEADER_EQ("latent", IntegerToString(EmbeddingSize));
   Skill_MANIFEST_HEADER_EQ("z_layout", "K,V,H,D");
   Skill_MANIFEST_HEADER_EQ("u_layout", "K,V,H");
   Skill_MANIFEST_HEADER_EQ("pi_layout", "K");
   Skill_MANIFEST_HEADER_EQ("codebook_layout", "K,V,H,D");
   Skill_MANIFEST_HEADER_EQ("variable_order", "BarDescr_feature_series_0_to_8");
   Skill_MANIFEST_HEADER_EQ("normalization", "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
#undef Skill_MANIFEST_HEADER_EQ
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillValidateForecastManifest.                        |
//+------------------------------------------------------------------+
bool SkillValidateForecastManifest(const ulong signature, const bool training = false)
  {
   if(signature == 0)
      ReturnFalse;
   if(!SkillValidateForecastManifestHeader())
      ReturnFalse;
#define Skill_MANIFEST_EQ(KEY,VALUE) if(SkillManifestValue(SkillActiveManifestFile,KEY)!=(VALUE)) ReturnFalse
   Skill_MANIFEST_EQ("contract_signature", StringFormat("%I64u", SkillForecast.ContractSignature()));
   Skill_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", signature));
//--- Function if.
   if(training)
     {
      const ulong target_hash = SkillHashFile(ulong(1469598103934665603), SkillActiveTargetFile);
      const ulong training_signature = SkillForecastTrainingSignature(signature);
      Skill_MANIFEST_EQ("target_hash", StringFormat("%I64u", target_hash));
      Skill_MANIFEST_EQ("training_signature", StringFormat("%I64u", training_signature));
      if(target_hash == 0 || training_signature == 0 ||
         SkillManifestValue(SkillActiveManifestFile, "completed_epochs") == "" ||
         SkillManifestValue(SkillActiveManifestFile, "training_batches") == "" ||
         SkillManifestValue(SkillActiveManifestFile, "invalid_batches") == "")
         ReturnFalse;
     }
#undef Skill_MANIFEST_EQ
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillCaptureFrozenBuffer.                             |
//+------------------------------------------------------------------+
bool SkillCaptureFrozenBuffer(CBufferFloat *source, CBufferFloat &baseline)
  {
   return(source && source.BufferRead() && source.Total() > 0 && baseline.AssignArray(source));
  }
//+------------------------------------------------------------------+
//| Implements SkillFrozenBufferEqual.                               |
//+------------------------------------------------------------------+
bool SkillFrozenBufferEqual(CBufferFloat *source, CBufferFloat &baseline)
  {
   if(!source || !source.BufferRead() || source.Total() != baseline.Total())
      ReturnFalse;
   for(int i = 0; i < source.Total(); i++)
      if(source[i] != baseline[i])
         ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillCaptureFrozenForecastBaseline.                   |
//+------------------------------------------------------------------+
bool SkillCaptureFrozenForecastBaseline(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : SkillForecast);
   SkillFrozenBaselineReady = false;
   if(!current || !current.GetCodebook())
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   if(trainable <= 0 || trainable != int(current.TrainableWeights()) ||
      !SkillCaptureFrozenBuffer(generator, SkillFrozenGeneratorWeights) ||
      !SkillCaptureFrozenBuffer(router, SkillFrozenRouterWeights) ||
      !SkillCaptureFrozenBuffer(confidence, SkillFrozenConfidenceWeights) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetPrototypes(), SkillFrozenPrototypes) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetEMASums(), SkillFrozenEMASums) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetEMACounts(), SkillFrozenEMACounts) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetUsage(), SkillFrozenUsage) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetInactive(), SkillFrozenInactive) ||
      !SkillCaptureFrozenBuffer(current.GetCodebook().GetInactivityAge(), SkillFrozenInactivityAge))
      ReturnFalse;
   SkillFrozenBaselineReady = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements SkillVerifyFrozenWeightsExact.                        |
//+------------------------------------------------------------------+
bool SkillVerifyFrozenWeightsExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : SkillForecast);
   if(!SkillFrozenBaselineReady || !current)
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   return (trainable > 0 && trainable == int(current.TrainableWeights()) &&
           SkillFrozenBufferEqual(generator, SkillFrozenGeneratorWeights) &&
           SkillFrozenBufferEqual(router, SkillFrozenRouterWeights) &&
           SkillFrozenBufferEqual(confidence, SkillFrozenConfidenceWeights));
  }
//+------------------------------------------------------------------+
//| Implements SkillVerifyFrozenCodebookExact.                       |
//+------------------------------------------------------------------+
bool SkillVerifyFrozenCodebookExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : SkillForecast);
   if(!SkillFrozenBaselineReady || !current || !current.GetCodebook())
      ReturnFalse;
   return (SkillFrozenBufferEqual(current.GetCodebook().GetPrototypes(), SkillFrozenPrototypes) &&
           SkillFrozenBufferEqual(current.GetCodebook().GetEMASums(), SkillFrozenEMASums) &&
           SkillFrozenBufferEqual(current.GetCodebook().GetEMACounts(), SkillFrozenEMACounts) &&
           SkillFrozenBufferEqual(current.GetCodebook().GetUsage(), SkillFrozenUsage) &&
           SkillFrozenBufferEqual(current.GetCodebook().GetInactive(), SkillFrozenInactive) &&
           SkillFrozenBufferEqual(current.GetCodebook().GetInactivityAge(), SkillFrozenInactivityAge));
  }
//+------------------------------------------------------------------+
//| Implements SkillVerifyFrozenForecastExact.                       |
//+------------------------------------------------------------------+
bool SkillVerifyFrozenForecastExact(CNeuronScenarioForecast *forecast = NULL)
  {
   return(SkillVerifyFrozenWeightsExact(forecast) && SkillVerifyFrozenCodebookExact(forecast));
  }
//+------------------------------------------------------------------+
//| Implements SkillLoadForecastInference.                           |
//+------------------------------------------------------------------+
bool SkillLoadForecastInference(void)
  {
   SkillFrozenBaselineReady = false;
   SkillForecast = NULL;
   if(!ACSRMStage02ResolveActiveCheckpoint(false))
     {
      Print("ACSRM checkpoint selector=FAIL before=inference_load");
      ReturnFalse;
     }
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
//--- Function if.
   if(!SkillMarket.Load(SkillActiveMarketFile, error, undefine, forecast, studied, true))
     {
      PrintFormat("%s inference: model load=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   SkillForecast = (CNeuronScenarioForecast*)SkillMarket.Layer(-1);
//--- Function if.
   if(!SkillForecast || SkillForecast.Type() != defNeuronScenarioForecast)
     {
      PrintFormat("%s inference: forecast layer=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//--- Function if.
   if(SkillForecast.GetTopK() != TopK)
     {
      PrintFormat("%s inference: TopK expected=%d actual=%d", ACSRM_LOG_PREFIX, TopK, SkillForecast.GetTopK());
      ReturnFalse;
     }
//--- only the Market/Forecast path. Target and its future-window b...
   if(!SkillValidateShapes(false))
     {
      PrintFormat("%s inference: shape audit=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   SkillMarket.TrainMode(false);
   const ulong signature = SkillForecastSignature();
//--- Function if.
   if(!SkillValidateForecastManifest(signature))
     {
      PrintFormat("%s inference: manifest=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
//--- Function if.
   if(!SkillCaptureFrozenForecastBaseline())
     {
      PrintFormat("%s inference: frozen baseline=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   SkillLastSignature = signature;
   return(true);
  }
//+-------------------------------------------------------------------+
//| Full forecast-training restore. Target is required here becau...  |
//+-------------------------------------------------------------------+
bool SkillLoadForecastTraining(void)
  {
   if(!ACSRMStage02ResolveActiveCheckpoint())
     {
      Print("ACSRM checkpoint selector=FAIL before=training_load");
      ReturnFalse;
     }
   if(!FileIsExist(SkillActiveManifestFile, FILE_COMMON) ||
      !FileIsExist(SkillActiveMarketFile, FILE_COMMON) ||
      !FileIsExist(SkillActiveTargetFile, FILE_COMMON))
      ReturnFalse;
//--- Avoid a partial CNet::Load before the fallback random graph i...
   if(!SkillValidateForecastManifestHeader())
     {
      PrintFormat("%s restore: manifest static contract=FAIL", ACSRM_LOG_PREFIX);
      ReturnFalse;
     }
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   if(!SkillMarket.Load(SkillActiveMarketFile, error, undefine, forecast, studied, true) ||
      !SkillTarget.Load(SkillActiveTargetFile, error, undefine, forecast, studied, true))
      ReturnFalse;
   if(!SkillTarget.SetOpenCLChecked(SkillMarket.GetOpenCL()))
      ReturnFalseEx("target OpenCL transfer failed");
   SkillForecast = (CNeuronScenarioForecast*)SkillMarket.Layer(-1);
   if(!SkillForecast || SkillForecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//--- These are transient training tensors and are intentionally absent from
//--- *.nnw. Recreate them before the common shape audit.
   if(!SkillInitTrainingBuffers() || !SkillValidateShapes())
      ReturnFalse;
   const ulong signature = SkillForecastSignature();
   if(!SkillValidateForecastManifest(signature, true))
      ReturnFalse;
   const string completed = SkillManifestValue(SkillActiveManifestFile, "completed_epochs");
   const string batches = SkillManifestValue(SkillActiveManifestFile, "training_batches");
   const string invalid = SkillManifestValue(SkillActiveManifestFile, "invalid_batches");
   if(completed == "" || batches == "" || invalid == "")
      ReturnFalse;
   SkillCompletedEpochs = (uint)StringToInteger(completed);
   SkillBatches = (ulong)StringToInteger(batches);
   SkillInvalidBatches = (ulong)StringToInteger(invalid);
   SkillTarget.TrainMode(false);
   SkillMarket.TrainMode(true);
   SkillLastSignature = signature;
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures the loaded Stage 02 graph for frozen Stage 03 use.    |
//+------------------------------------------------------------------+
bool SkillConfigureProductionACSRMCheckpoint(void)
  {
   CNeuronOMPBOCL *ompb = GetACSRM();
   CNeuronScenarioForecast *forecast = SkillForecast;
   if(!ompb || !forecast || !SetACSRMMode(OMPB_INFERENCE) ||
      !SkillMarket.SetWeightsUpdate(false) ||
      !forecast.SetCodebookUpdate(false))
      ReturnFalse;
   SkillMarket.TrainMode(false);
   ompb.TrainMode(false);
   forecast.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks the strict published Stage 02 contract before Stage 03.   |
//+------------------------------------------------------------------+
bool SkillValidateProductionACSRMCheckpoint(void)
  {
   const string checkpoint = "canonical";
   CNeuronBaseOCL *rank_tcm = SkillMarket.Layer(4);
   CNeuronOMPBOCL *ompb = GetACSRM();
   CNeuronBaseOCL *forecast_layer = SkillMarket.Layer(6);
   CNeuronBaseOCL *market_tail = SkillMarket.Layer(-1);
   CNeuronScenarioForecast *forecast = SkillForecast;
   CNeuronBaseOCL *ompb_base = (CNeuronBaseOCL *)ompb;
   CNeuronBaseOCL *forecast_base = (CNeuronBaseOCL *)forecast;
   CLayerDescription *ompb_info = (ompb ? ompb.GetLayerInfo() : NULL);
   bool layers_frozen = true;
   for(int index = 0; index <= 6; ++index)
     {
      CNeuronBaseOCL *layer = SkillMarket.Layer(index);
      if(!layer || layer.TrainMode())
        {
         layers_frozen = false;
         break;
        }
     }
   const bool descriptor_valid = (ompb_info &&
                                  ompb_info.window == EmbeddingSize &&
                                  ompb_info.count == BarDescr &&
                                  ompb_info.layers == ACSRMSamples &&
                                  ompb_info.units.Size() == 2 &&
                                  ompb_info.units[0] == ACSRMReferenceSize &&
                                  ompb_info.units[1] == ACSRMCurrentWindow &&
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
   const ulong signature = SkillForecastSignature();
   const bool valid = (ACSRMStage02ResolveActiveCheckpoint(false) &&
                       !ACSRMStage02RecoveryFailed &&
                       SkillActiveMarketFile == Skill_MARKET_FILE &&
                       SkillActiveTargetFile == Skill_TARGET_FILE &&
                       SkillActiveManifestFile == Skill_MANIFEST_FILE &&
                       SkillValidateForecastManifest(signature) &&
                       rank_tcm != NULL && rank_tcm.Type() == defNeuronCogDriverRankTCM &&
                       ompb != NULL && ompb.Mode() == OMPB_INFERENCE &&
                       ompb_base != NULL && !ompb_base.TrainMode() &&
                       forecast != NULL && forecast_base != NULL && !forecast_base.TrainMode() &&
                       !forecast.CodebookUpdate() &&
                       SkillMarket.WeightsUpdateEnabled() == false && layers_frozen &&
                       descriptor_valid);
   if(!valid)
     {
      PrintFormat("%s_STAGE03_PRODUCTION_GATE_FAIL checkpoint=%s", FileName, checkpoint);
      return(false);
     }
   PrintFormat("%s_STAGE03_PRODUCTION_GATE_PASS checkpoint=%s", FileName, checkpoint);
   return(true);
  }
//+-----------------------------------------------------------------------+
//| Captures Market-base and ACSRM parameter fingerprints for Stage  03.  |
//+-----------------------------------------------------------------------+
bool SkillCaptureProductionACSRMFingerprints(void)
  {
   CNeuronOMPBOCL *ompb = GetACSRM();
   SkillProductionSignatureReady = false;
   SkillProductionBaseFingerprint = 0;
   SkillProductionACSRMFingerprint = ulong(1469598103934665603);
   if(!ompb || !ACSRMStage02MarketFingerprint(0, 4, SkillProductionBaseFingerprint) ||
      !ompb.AppendParameterFingerprint(SkillProductionACSRMFingerprint) ||
      SkillProductionBaseFingerprint == 0 || SkillProductionACSRMFingerprint == 0)
      ReturnFalse;
   SkillProductionSignatureReady = true;
   PrintFormat("%s_STAGE03_SIGNATURES_BEFORE base=%I64u posterior=%I64u", FileName,
               SkillProductionBaseFingerprint, SkillProductionACSRMFingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Verifies that Stage 03 preserved the frozen Market and ACSRM.    |
//+------------------------------------------------------------------+
bool SkillVerifyProductionACSRMFingerprints(void)
  {
   ulong base = 0;
   ulong posterior = ulong(1469598103934665603);
   CNeuronOMPBOCL *ompb = GetACSRM();
   if(!SkillProductionSignatureReady || !ompb ||
      !ACSRMStage02MarketFingerprint(0, 4, base) ||
      !ompb.AppendParameterFingerprint(posterior) ||
      !SkillValidateProductionACSRMCheckpoint())
      ReturnFalse;
   const bool unchanged = (base == SkillProductionBaseFingerprint &&
                           posterior == SkillProductionACSRMFingerprint);
   PrintFormat("%s_STAGE03_SIGNATURES_AFTER base=%I64u posterior=%I64u unchanged=%s", FileName,
               base, posterior, (unchanged ? "true" : "false"));
   return(unchanged);
  }
//+------------------------------------------------------------------+
//| Implements SkillAddCross.                                        |
//+------------------------------------------------------------------+
bool SkillAddCross(CArrayObj *description, const bool critic)
  {
   CLayerDescription *descr = new CLayerDescription();
   if(!description || !descr)
     { DeleteObj(descr); ReturnFalse; }
   descr.type = defNeuronScenarioCrossAttention;
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
//| Appends one convolutional layer description to the graph.        |
//+------------------------------------------------------------------+
bool SkillAddConv(CArrayObj *description, const uint count, const uint window, const uint output,
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
//| Implements SkillForwardForecast.                                 |
//+------------------------------------------------------------------+
bool SkillForwardForecast(const int position, CBufferFloat *state, CBufferFloat *time,
                          const int shift_bars = 1)
  {
//--- Causality: the decision state window ends one bar BEFORE the execution
//--- bar, so close/high/low of the entry bar are realized when its open is
//--- chosen as the post-decision execution price.
   if(!SkillForecast || !CreateBuffers(position + shift_bars, state, time, NULL) ||
      !SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = SkillForecast.GetZ(), *u = SkillForecast.GetU(), *pi = SkillForecast.GetPi();
   return (z != NULL && u != NULL && pi != NULL && z.Total() == NScenarios * BarDescr * NForecast * EmbeddingSize &&
           u.Total() == NScenarios * BarDescr * NForecast &&
           pi.Total() == NScenarios && z.GetIndex() >= 0 && u.GetIndex() >= 0 && pi.GetIndex() >= 0);
  }
//+------------------------------------------------------------------+
//| Stable Owner index of one offline training example.              |
//+------------------------------------------------------------------+
int SkillObservationOwner(const int position, const int slot, const int seed)
  {
//--- A deal (position, slot) is the example; the Owner is a stable function
//--- of the example id and the replay seed.  Re-reading the same record or
//--- reordering episodes never transfers its label to the other Critic.
   return(int((ulong(MathMax(position, 0)) * ulong(2654435761u) +
               ulong(MathMax(slot, 0)) * ulong(40503u) +
               ulong(MathMax(seed, 0))) & 1));
  }
//+------------------------------------------------------------------------------------------------------------------+
//| Converts the actor gradient to the pre-activation gradient:  dU/dz = dU/da * a * (1 - a) for the final SIGMOID.  |
//+------------------------------------------------------------------------------------------------------------------+
bool ApplyActorSigmoidDerivative(CNeuronBaseOCL *output_layer)
  {
   if(!output_layer || output_layer.getOutput().Total() != (int)NActions ||
      output_layer.getGradient().Total() != (int)NActions ||
      output_layer.getOutput().GetIndex() < 0 || output_layer.getGradient().GetIndex() < 0)
      ReturnFalse;
   if(!ActorDerivOneMinus.BufferInit(NActions, 0) ||
      !ActorDerivOneMinus.BufferCreate(SkillMarket.GetOpenCL()) ||
      !ActorDerivOneMinus.BufferWrite() ||
      !ActorDeriv.BufferInit(NActions, 0) ||
      !ActorDeriv.BufferCreate(SkillMarket.GetOpenCL()) ||
      !ActorDeriv.BufferWrite() ||
      !ActorGradScratch.BufferInit(NActions, 0) ||
      !ActorGradScratch.BufferCreate(SkillMarket.GetOpenCL()) ||
      !ActorGradScratch.BufferWrite() ||
      !SkillDevice.Bind(SkillMarket.GetOpenCL()))
      ReturnFalse;
//--- dU/dz = dU/da * a * (1 - a) through the genuine per-element
//--- DeActivation primitive: every component scales by its OWN (1 - a_i).
//--- IdentDifferent (identity-row subtraction) is NOT used: with
//--- dimension == NActions it produces (-a_1, -a_2, ..., 1 - a_k) instead
//--- of elementwise (1 - a_1, ..., 1 - a_n).
   if(!SkillDevice.DeActivationOnDevice(output_layer.getOutput(), GetPointer(ActorGradScratch),
                                        output_layer.getGradient(), SIGMOID) ||
      !SkillDevice.Copy(GetPointer(ActorGradScratch), output_layer.getGradient(), NActions))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Builds CriticInput.                                              |
//+------------------------------------------------------------------+
bool BuildCriticInput(CBufferFloat *account, CBufferFloat *action, CBufferFloat *combined)
  {
   if(!account || !action || !combined || account.Total() != AccountDescr || action.Total() != NActions ||
      account.GetIndex() < 0 || action.GetIndex() < 0)
      ReturnFalse;
//--- GPU Join/Copy transfer: both sources bind the Market context and the
//--- concatenation is assembled on device from the CURRENT tensors, so the
//--- target-action can never arrive from a stale host snapshot.  The final
//--- read-back refreshes the host copy that the first CNet layer consumes.
   if(!combined.BufferInit(AccountDescr + NActions, 0))
      ReturnFalse;
   if(combined.GetIndex() < 0 && !combined.BufferCreate(SkillMarket.GetOpenCL()))
      ReturnFalse;
   if(!combined.BufferWrite() || !account.BufferWrite())
      ReturnFalse;
   if(!SkillDevice.Bind(SkillMarket.GetOpenCL()))
      ReturnFalse;
   if(!SkillDevice.Join2(account, AccountDescr, action, NActions, combined))
      ReturnFalse;
   return(combined.BufferRead());
  }
//+-------------------------------------------------------------------+
//| Implements EvaluateAction. lot instead of the former balance-...  |
//+-------------------------------------------------------------------+
double EvaluateAction(CBufferFloat *action, const double balance, const uint position,
                      const int horizon_bars, int &outcome_tag)
  {
//--- CheckAction already floors sub-minimum lot fractions at LotsMin(), so
//--- the returned value carries a real outcome label (TP/SL/HORIZON) even
//--- for a cold-start Actor with tiny lots.  Passthrough only.
   const double reward = CheckAction(action, balance, position, horizon_bars, outcome_tag);
   if(!MathIsValidNumber(reward))
      return(reward);
   return(reward);
  }
//+-------------------------------------------------------------------+
//| Builds TeacherAction. raw future window here; the live Market...  |
//+-------------------------------------------------------------------+
bool BuildTeacherAction(const int position, CBufferFloat *account, CBufferFloat *action,
                        double &reward, const int horizon_bars, int &outcome_tag)
  {
   reward = 0;
   if(!account || !action || position < 0 || position + HistoryBars + NForecast > int(Rates.Size()) ||
//--- Function Total.
      account.Total() != AccountDescr)
     {
      PrintFormat("BuildTeacherAction input position=%d rates=%d account=%d action=%d",
                  position, Rates.Size(), (account ? account.Total() : -1), (action ? action.Total() : -1));
      ReturnFalse;
     }
   CBufferFloat teacher_state, teacher_time;
//--- Function if.
   if(!CreateBuffers(position, GetPointer(teacher_state), GetPointer(teacher_time), GetPointer(SkillFuture)))
     {
      PrintFormat("BuildTeacherAction CreateBuffers position=%d future=%d index=%d",
                  position, SkillFuture.Total(), SkillFuture.GetIndex());
      ReturnFalse;
     }
   vector<float> account_values;
//--- Function if.
   if(account.GetData(account_values) != AccountDescr)
     {
      PrintFormat("BuildTeacherAction account data total=%d index=%d", account.Total(), account.GetIndex());
      ReturnFalse;
     }
   const vector<float> teacher = OraculAction(account_values, GetPointer(SkillFuture));
//--- Function if.
   if(teacher.Size() != NActions)
     {
      PrintFormat("BuildTeacherAction oracle size=%d expected=%d", teacher.Size(), NActions);
      ReturnFalse;
     }
//--- Function if.
   if(!action.AssignArray(teacher))
     {
      Print("BuildTeacherAction action AssignArray");
      ReturnFalse;
     }
//--- Function if.
   if(action.GetIndex() < 0 && !action.BufferCreate(SkillMarket.GetOpenCL()))
     {
      PrintFormat("BuildTeacherAction action BufferCreate total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
//--- Function if.
   if(action.GetIndex() >= 0 && !action.BufferWrite())
     {
      PrintFormat("BuildTeacherAction action BufferWrite total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
   reward = EvaluateAction(action, MathMax(0.0, double(account_values[0]) * EtalonBalance),
                           (uint)position, horizon_bars, outcome_tag);
//--- Function if.
   if(!MathIsValidNumber(reward))
     {
      PrintFormat("BuildTeacherAction reward invalid %.8f", reward);
      ReturnFalse;
     }
   return(true);
  }
//+-------------------------------------------------------------------+
//| A valid random policy action expands Critic coverage only. Bu...  |
//+-------------------------------------------------------------------+
bool BuildRandomAction(const int position, CBufferFloat *account, CBufferFloat *action,
                       double &reward, const int horizon_bars, int &outcome_tag)
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
   if(action.GetIndex() < 0 && !action.BufferCreate(SkillMarket.GetOpenCL()))
      ReturnFalse;
   if(action.GetIndex() >= 0 && !action.BufferWrite())
      ReturnFalse;
   reward = EvaluateAction(action, balance, (uint)position, horizon_bars, outcome_tag);
   return(MathIsValidNumber(reward));
  }
//+------------------------------------------------------------------+
//| Implements SkillClampAction.                                     |
//+------------------------------------------------------------------+
bool SkillClampAction(CBufferFloat *source, CBufferFloat *target)
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
//| Checks the canonical AC tuple file names.                        |
//+------------------------------------------------------------------+
bool SkillACTupleNamesValid(const string actor_file, const string q1_file,
                              const string q2_file, const string manifest_file)
  {
   return(actor_file != "" && q1_file != "" && q2_file != "" && manifest_file != "" &&
          actor_file != q1_file && actor_file != q2_file && actor_file != manifest_file &&
          q1_file != q2_file && q1_file != manifest_file && q2_file != manifest_file);
  }
//+------------------------------------------------------------------+
//| Deletes one canonical AC tuple file.                             |
//+------------------------------------------------------------------+
bool SkillACDeleteFile(const string file_name)
  {
   return(!FileIsExist(file_name, FILE_COMMON) || FileDelete(file_name, FILE_COMMON));
  }
//+------------------------------------------------------------------+
//| Copies the AC tuple files to a new canonical set.                |
//+------------------------------------------------------------------+
bool SkillACCopyTuple(const string source_actor, const string source_q1,
                        const string source_q2, const string source_manifest,
                        const string destination_actor, const string destination_q1,
                        const string destination_q2, const string destination_manifest,
                        const bool rewrite)
  {
   if(!SkillACTupleNamesValid(source_actor, source_q1, source_q2, source_manifest) ||
      !SkillACTupleNamesValid(destination_actor, destination_q1, destination_q2,
                                destination_manifest))
      ReturnFalseEx("Actor-Critic tuple names");
   const uint flags = (rewrite ? FILE_COMMON | FILE_REWRITE : FILE_COMMON);
   if(!FileCopy(source_actor, FILE_COMMON, destination_actor, flags))
      ReturnFalseEx("Actor-Critic Actor copy");
   if(!FileCopy(source_q1, FILE_COMMON, destination_q1, flags))
      ReturnFalseEx("Actor-Critic Q1 copy");
   if(!FileCopy(source_q2, FILE_COMMON, destination_q2, flags))
      ReturnFalseEx("Actor-Critic Q2 copy");
//--- Write the manifest last so a complete tuple has an unambiguous commit point.
   if(!FileCopy(source_manifest, FILE_COMMON, destination_manifest, flags))
      ReturnFalseEx("Actor-Critic manifest copy");
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks that an AC tuple file triplet is complete.                |
//+------------------------------------------------------------------+
bool SkillACTupleComplete(const string actor_file, const string q1_file,
                            const string q2_file, const string manifest_file)
  {
   const ulong seed = ulong(1469598103934665603);
   return(SkillACTupleNamesValid(actor_file, q1_file, q2_file, manifest_file) &&
          SkillHashFile(seed, actor_file) != 0 && SkillHashFile(seed, q1_file) != 0 &&
          SkillHashFile(seed, q2_file) != 0 && SkillHashFile(seed, manifest_file) != 0);
  }
//+------------------------------------------------------------------+
//| Compares two AC tuples byte-for-byte.                            |
//+------------------------------------------------------------------+
bool SkillACTuplesEqual(const string first_actor, const string first_q1,
                          const string first_q2, const string first_manifest,
                          const string second_actor, const string second_q1,
                          const string second_q2, const string second_manifest)
  {
   const ulong seed = ulong(1469598103934665603);
   return(SkillACTupleComplete(first_actor, first_q1, first_q2, first_manifest) &&
          SkillACTupleComplete(second_actor, second_q1, second_q2, second_manifest) &&
          SkillHashFile(seed, first_actor) == SkillHashFile(seed, second_actor) &&
          SkillHashFile(seed, first_q1) == SkillHashFile(seed, second_q1) &&
          SkillHashFile(seed, first_q2) == SkillHashFile(seed, second_q2) &&
          SkillHashFile(seed, first_manifest) == SkillHashFile(seed, second_manifest));
  }
//+------------------------------------------------------------------+
//| Writes one transaction line to the AC transaction file.          |
//+------------------------------------------------------------------+
bool SkillACWriteTransactionLine(const int handle, const string value,
                                   const string context)
  {
   if(FileWrite(handle, value) <= 0)
     {
      PrintFormat("Skill policy transaction write failed field=%s error=%d",
                  context, GetLastError());
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Writes the AC transaction record set.                            |
//+------------------------------------------------------------------+
bool SkillACWriteTransaction(const string transaction_file, const bool had_previous,
                               const string phase, const ulong actor_hash,
                               const ulong q1_hash, const ulong q2_hash,
                               const ulong manifest_hash)
  {
   const string staged_file = transaction_file + ".next";
   if(transaction_file == "" || (phase != "prepared" && phase != "previous_ready") ||
      actor_hash == 0 || q1_hash == 0 || q2_hash == 0 || manifest_hash == 0 ||
      FileIsExist(transaction_file, FILE_COMMON) || FileIsExist(staged_file, FILE_COMMON))
      ReturnFalseEx("Actor-Critic transaction arguments");
   int handle = FileOpen(staged_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalseEx("Actor-Critic transaction open");
   if(!SkillACWriteTransactionLine(handle, "format=ACSRM_ACTOR_CRITIC_TRANSACTION", "format") ||
      !SkillACWriteTransactionLine(handle,
                                     StringFormat("version=%u", Skill_AC_TRANSACTION_VERSION),
                                     "version") ||
      !SkillACWriteTransactionLine(handle, StringFormat("phase=%s", phase), "phase") ||
      !SkillACWriteTransactionLine(handle, StringFormat("had_previous=%s",
                                     (had_previous ? "true" : "false")), "had_previous") ||
      !SkillACWriteTransactionLine(handle, StringFormat("actor_hash=%I64u", actor_hash),
                                     "actor_hash") ||
      !SkillACWriteTransactionLine(handle, StringFormat("q1_hash=%I64u", q1_hash), "q1_hash") ||
      !SkillACWriteTransactionLine(handle, StringFormat("q2_hash=%I64u", q2_hash), "q2_hash") ||
      !SkillACWriteTransactionLine(handle, StringFormat("manifest_hash=%I64u", manifest_hash),
                                     "manifest_hash"))
     {
      FileClose(handle);
      SkillACDeleteFile(staged_file);
      ReturnFalse;
     }
   FileFlush(handle);
   FileClose(handle);
//--- Publish the complete marker in one rename after all fields are flushed.
   if(!FileMove(staged_file, FILE_COMMON, transaction_file, FILE_COMMON | FILE_REWRITE))
      ReturnFalseEx("Actor-Critic transaction publish");
   return(true);
  }
//+------------------------------------------------------------------+
//| Reads the AC transaction record set.                             |
//+------------------------------------------------------------------+
bool SkillACReadTransaction(const string transaction_file, bool &had_previous,
                              string &phase, ulong &actor_hash, ulong &q1_hash,
                              ulong &q2_hash, ulong &manifest_hash)
  {
   had_previous = false;
   phase = "";
   actor_hash = 0;
   q1_hash = 0;
   q2_hash = 0;
   manifest_hash = 0;
   if(!FileIsExist(transaction_file, FILE_COMMON) ||
      SkillManifestValue(transaction_file, "format") != "ACSRM_ACTOR_CRITIC_TRANSACTION" ||
      SkillManifestValue(transaction_file, "version") != IntegerToString(Skill_AC_TRANSACTION_VERSION))
      return(false);
   const string previous = SkillManifestValue(transaction_file, "had_previous");
   phase = SkillManifestValue(transaction_file, "phase");
   const string actor = SkillManifestValue(transaction_file, "actor_hash");
   const string q1 = SkillManifestValue(transaction_file, "q1_hash");
   const string q2 = SkillManifestValue(transaction_file, "q2_hash");
   const string manifest = SkillManifestValue(transaction_file, "manifest_hash");
   if((previous != "true" && previous != "false") ||
      (phase != "prepared" && phase != "previous_ready") || actor == "" || q1 == "" ||
      q2 == "" || manifest == "")
      return(false);
   had_previous = (previous == "true");
   actor_hash = (ulong)StringToInteger(actor);
   q1_hash = (ulong)StringToInteger(q1);
   q2_hash = (ulong)StringToInteger(q2);
   manifest_hash = (ulong)StringToInteger(manifest);
   return(actor_hash != 0 && q1_hash != 0 && q2_hash != 0 && manifest_hash != 0);
  }
//+------------------------------------------------------------------+
//| Advances the AC transaction selector.                            |
//+------------------------------------------------------------------+
bool SkillACAdvanceTransaction(const string transaction_file, const bool had_previous,
                                 const string phase, const ulong actor_hash,
                                 const ulong q1_hash, const ulong q2_hash,
                                 const ulong manifest_hash)
  {
   if(!SkillACDeleteFile(transaction_file))
      ReturnFalse;
   return(SkillACWriteTransaction(transaction_file, had_previous, phase,
                                    actor_hash, q1_hash, q2_hash, manifest_hash));
  }
//+------------------------------------------------------------------+
//| Cleans up the AC transaction files.                              |
//+------------------------------------------------------------------+
bool SkillACCleanupTransaction(const string transaction_file,
                                 const string next_actor, const string next_q1,
                                 const string next_q2, const string next_manifest,
                                 const string previous_actor, const string previous_q1,
                                 const string previous_q2, const string previous_manifest)
  {
   const string staged_transaction = transaction_file + ".next";
   const string files[] = {next_manifest, next_q2, next_q1, next_actor,
                           previous_manifest, previous_q2, previous_q1, previous_actor,
                           staged_transaction, transaction_file
                          };
   for(int i = 0; i < ArraySize(files); i++)
      if(!SkillACDeleteFile(files[i]))
        {
         PrintFormat("Skill policy transaction cleanup failed file=%s error=%d",
                     files[i], GetLastError());
         return(false);
        }
   return(true);
  }
//+------------------------------------------------------------------+
//| Recovers the AC transaction state after a failed write.          |
//+------------------------------------------------------------------+
bool SkillACRecoverTransaction(const string canonical_actor, const string canonical_q1,
                                 const string canonical_q2, const string canonical_manifest,
                                 const string next_actor, const string next_q1,
                                 const string next_q2, const string next_manifest,
                                 const string previous_actor, const string previous_q1,
                                 const string previous_q2, const string previous_manifest,
                                 const string transaction_file)
  {
   const bool marker_exists = FileIsExist(transaction_file, FILE_COMMON);
   const bool staged_marker_exists = FileIsExist(transaction_file + ".next", FILE_COMMON);
   const bool sidecars_exist = (FileIsExist(next_actor, FILE_COMMON) ||
                                FileIsExist(next_q1, FILE_COMMON) ||
                                FileIsExist(next_q2, FILE_COMMON) ||
                                FileIsExist(next_manifest, FILE_COMMON) ||
                                FileIsExist(previous_actor, FILE_COMMON) ||
                                FileIsExist(previous_q1, FILE_COMMON) ||
                                FileIsExist(previous_q2, FILE_COMMON) ||
                                FileIsExist(previous_manifest, FILE_COMMON) ||
                                staged_marker_exists);
   if(!marker_exists)
     {
      if(!sidecars_exist)
         return(true);
      if(SkillACTupleComplete(canonical_actor, canonical_q1, canonical_q2, canonical_manifest))
         return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                            next_manifest, previous_actor, previous_q1,
                                            previous_q2, previous_manifest));
      if(SkillACTupleComplete(previous_actor, previous_q1, previous_q2, previous_manifest) &&
         SkillACCopyTuple(previous_actor, previous_q1, previous_q2, previous_manifest,
                            canonical_actor, canonical_q1, canonical_q2, canonical_manifest, true) &&
         SkillACTuplesEqual(previous_actor, previous_q1, previous_q2, previous_manifest,
                              canonical_actor, canonical_q1, canonical_q2, canonical_manifest))
         return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                            next_manifest, previous_actor, previous_q1,
                                            previous_q2, previous_manifest));
      if(!FileIsExist(canonical_actor, FILE_COMMON) &&
         !FileIsExist(canonical_q1, FILE_COMMON) &&
         !FileIsExist(canonical_q2, FILE_COMMON) &&
         !FileIsExist(canonical_manifest, FILE_COMMON))
         return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                            next_manifest, previous_actor, previous_q1,
                                            previous_q2, previous_manifest));
      ReturnFalse;
     }
   bool had_previous = false;
   string phase = "";
   ulong actor_hash = 0, q1_hash = 0, q2_hash = 0, manifest_hash = 0;
   if(!SkillACReadTransaction(transaction_file, had_previous, phase, actor_hash, q1_hash,
                                q2_hash, manifest_hash))
      ReturnFalse;
   if(SkillHashFile(ulong(1469598103934665603), canonical_actor) == actor_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_q1) == q1_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_q2) == q2_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_manifest) == manifest_hash)
      return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                         next_manifest, previous_actor, previous_q1,
                                         previous_q2, previous_manifest));
   if(phase == "prepared" && had_previous &&
      SkillACTupleComplete(canonical_actor, canonical_q1, canonical_q2, canonical_manifest))
      return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                         next_manifest, previous_actor, previous_q1,
                                         previous_q2, previous_manifest));
   if(had_previous && SkillACTupleComplete(previous_actor, previous_q1, previous_q2, previous_manifest) &&
      SkillACCopyTuple(previous_actor, previous_q1, previous_q2, previous_manifest,
                         canonical_actor, canonical_q1, canonical_q2, canonical_manifest, true) &&
      SkillACTuplesEqual(previous_actor, previous_q1, previous_q2, previous_manifest,
                           canonical_actor, canonical_q1, canonical_q2, canonical_manifest))
      return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                         next_manifest, previous_actor, previous_q1,
                                         previous_q2, previous_manifest));
   if(!had_previous && phase == "prepared" &&
      SkillACTupleComplete(next_actor, next_q1, next_q2, next_manifest) &&
      SkillACCopyTuple(next_actor, next_q1, next_q2, next_manifest,
                         canonical_actor, canonical_q1, canonical_q2, canonical_manifest, true) &&
      SkillHashFile(ulong(1469598103934665603), canonical_actor) == actor_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_q1) == q1_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_q2) == q2_hash &&
      SkillHashFile(ulong(1469598103934665603), canonical_manifest) == manifest_hash)
      return(SkillACCleanupTransaction(transaction_file, next_actor, next_q1, next_q2,
                                         next_manifest, previous_actor, previous_q1,
                                         previous_q2, previous_manifest));
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Writes the AC manifest file for a forecast signature.            |
//+------------------------------------------------------------------+
bool SkillWriteACManifestFile(const ulong forecast_signature, const string actor_file,
                                const string q1_file, const string q2_file,
                                const string manifest_file)
  {
   if(forecast_signature == 0 || forecast_signature != SkillLastSignature ||
      !SkillACTupleNamesValid(actor_file, q1_file, q2_file, manifest_file))
      ReturnFalse;
   const ulong seed = ulong(1469598103934665603);
   const ulong actor_hash = SkillHashFile(seed, actor_file);
   const ulong q1_hash = SkillHashFile(seed, q1_file);
   const ulong q2_hash = SkillHashFile(seed, q2_file);
   if(actor_hash == 0 || q1_hash == 0 || q2_hash == 0)
      ReturnFalse;
   int handle = FileOpen(manifest_file, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   const bool written = (FileWrite(handle, "format=ACSRM_ACTOR_CRITIC") > 0 &&
                         FileWrite(handle, StringFormat("version=%u", Skill_AC_FORMAT_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("forecast_signature=%I64u", forecast_signature)) > 0 &&
                         FileWrite(handle, StringFormat("actor_context=%u", (3 * EmbeddingSize))) > 0 &&
                         FileWrite(handle, StringFormat("critic_context=%u", (5 * EmbeddingSize))) > 0 &&
                         FileWrite(handle, "action_order=BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL") > 0 &&
                         FileWrite(handle, "scenario_policy=preserve_KV_no_probability_aggregation") > 0 &&
                         FileWrite(handle, "teacher_policy=profitable_realized_teacher_and_random_supervised") > 0 &&
                         FileWrite(handle, "no_trade_penalty=zero_lot_only_sub_min_lot_evaluated_at_stated_lot") > 0 &&
                         FileWrite(handle, "account_execution=target_position_tp_sl_bar_lifecycle") > 0 &&
                         FileWrite(handle, "critic_target=deal_outcome_tp_sl_horizon") > 0 &&
                         FileWrite(handle, "critic_pretrain_target=deal_outcome_tp_sl_horizon") > 0 &&
                         FileWrite(handle, "offline_target=tp_sl_horizon_discounted_by_elapsed_bars") > 0 &&
                         FileWrite(handle, "critic_online_target=cross_target_td") > 0 &&
                         FileWrite(handle, "critic_final_semantics=discounted_live_return") > 0 &&
                         FileWrite(handle, "online_bootstrap_state=next_closed_bar") > 0 &&
                         FileWrite(handle, StringFormat("deal_horizon_bars=%d", SkillManifestDealHorizon)) > 0 &&
                         FileWrite(handle, "ambiguous_level_rule=sl_before_tp_within_same_bar") > 0 &&
                         FileWrite(handle, "actor_policy_gradient=executable_single_critic_sigmoid_derivative") > 0 &&
                         FileWrite(handle, "critic_coverage=owner_split_seed_stable_offline_separate_online") > 0 &&
                         FileWrite(handle, "actor_supervision=policy_srm_single_owner_critic_plus_teacher_random") > 0 &&
                         FileWrite(handle, "offline_units=account_currency_money") > 0 &&
                         FileWrite(handle, "online_reward=account_currency_equity_delta") > 0 &&
                         FileWrite(handle, "online_target=cross_target_td_y_r_plus_gamma_sample_targetq") > 0 &&
                         FileWrite(handle, StringFormat("online_discount=%.4f", SkillManifestOnlineDiscount)) > 0 &&
                         FileWrite(handle, "actor_supervision_gradient=combined_single_update_before_weight_change") > 0 &&
                         FileWrite(handle, "policy_critic_selector=actor_event_hash_seed_balanced") > 0 &&
                         FileWrite(handle, "risk_variable=return") > 0 &&
                         FileWrite(handle, "cvar_convention=lower_tail_return") > 0 &&
                         FileWrite(handle, "srm_ordering=weighted_monotone_rearrangement") > 0 &&
                         FileWrite(handle, "srm_ordering_key=return_ascending_index_tiebreak") > 0 &&
                         FileWrite(handle, "sample_semantics=discrete_fqf_representative") > 0 &&
                         FileWrite(handle, StringFormat("quantiles=%u", SkillManifestQuantiles)) > 0 &&
                         FileWrite(handle, "srm_invalid_policy=skip_actor_update") > 0 &&
                         FileWrite(handle, "target_network_type=sgd_shadow") > 0 &&
                         FileWrite(handle, "target_update=periodic_hard_copy") > 0 &&
                         FileWrite(handle, "target_crossing=q1_from_q2_q2_from_q1") > 0 &&
                         FileWrite(handle, StringFormat("target_tau=%.4f", SkillManifestTargetTau)) > 0 &&
                         FileWrite(handle, StringFormat("target_update_period=%d", SkillManifestTargetUpdatePeriod)) > 0 &&
                         FileWrite(handle, StringFormat("config_version=%u", Skill_AC_CONFIG_VERSION)) > 0 &&
                         FileWrite(handle, StringFormat("actor_hash=%I64u", actor_hash)) > 0 &&
                         FileWrite(handle, StringFormat("q1_hash=%I64u", q1_hash)) > 0 &&
                         FileWrite(handle, StringFormat("q2_hash=%I64u", q2_hash)) > 0 &&
                         FileWrite(handle, StringFormat("generation=%I64u", (ulong)TimeCurrent())) > 0);
   if(written)
      FileFlush(handle);
   FileClose(handle);
   return(written);
  }
//+------------------------------------------------------------------+
//| Writes the full AC manifest record.                              |
//+------------------------------------------------------------------+
bool SkillWriteACManifest(const ulong forecast_signature)
  {
   return(SkillWriteACManifestFile(forecast_signature, Skill_ACTOR_FILE,
                                     Skill_Q1_FILE, Skill_Q2_FILE,
                                     Skill_AC_MANIFEST_FILE));
  }
//+------------------------------------------------------------------+
//| Validates the AC manifest tuple presence and fields.             |
//+------------------------------------------------------------------+
bool SkillValidateACManifestTuple(const bool required, const string manifest_file,
                                    const string actor_file, const string q1_file,
                                    const string q2_file, const bool allow_bank_reset)
  {
   if(!SkillACTupleNamesValid(actor_file, q1_file, q2_file, manifest_file) ||
      SkillManifestValue(manifest_file, "format") == "")
      return(!required);
#define Skill_AC_MANIFEST_EQ(KEY,VALUE) \
   { const string actual=SkillManifestValue(manifest_file,KEY); const string expected=(VALUE); \
     if(actual!=expected) { PrintFormat("Skill policy manifest: %s expected=%s actual=%s",KEY,expected,actual); ReturnFalse; } }
   Skill_AC_MANIFEST_EQ("format", "ACSRM_ACTOR_CRITIC");
   Skill_AC_MANIFEST_EQ("version", IntegerToString(Skill_AC_FORMAT_VERSION));
   Skill_AC_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", SkillLastSignature));
   Skill_AC_MANIFEST_EQ("actor_context", IntegerToString((3 * EmbeddingSize)));
   Skill_AC_MANIFEST_EQ("critic_context", IntegerToString((5 * EmbeddingSize)));
   Skill_AC_MANIFEST_EQ("action_order", "BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL");
   Skill_AC_MANIFEST_EQ("scenario_policy", "preserve_KV_no_probability_aggregation");
   Skill_AC_MANIFEST_EQ("teacher_policy", "profitable_realized_teacher_and_random_supervised");
   Skill_AC_MANIFEST_EQ("no_trade_penalty", "zero_lot_only_sub_min_lot_evaluated_at_stated_lot");
   Skill_AC_MANIFEST_EQ("account_execution", "target_position_tp_sl_bar_lifecycle");
   Skill_AC_MANIFEST_EQ("critic_target", "deal_outcome_tp_sl_horizon");
   Skill_AC_MANIFEST_EQ("offline_target", "tp_sl_horizon_discounted_by_elapsed_bars");
   { const string bootstrap = SkillManifestValue(manifest_file, "online_bootstrap_state");
     if(bootstrap != "" && bootstrap != "next_closed_bar")
       { PrintFormat("Skill policy manifest: online_bootstrap_state expected=next_closed_bar actual=%s", bootstrap); ReturnFalse; } }
   { const string pretrain = SkillManifestValue(manifest_file, "critic_pretrain_target");
     if(pretrain != "" && pretrain != "deal_outcome_tp_sl_horizon")
       { PrintFormat("Skill policy manifest: critic_pretrain_target expected=deal_outcome_tp_sl_horizon actual=%s", pretrain); ReturnFalse; } }
   { const string online_target = SkillManifestValue(manifest_file, "critic_online_target");
     if(online_target != "" && online_target != "cross_target_td")
       { PrintFormat("Skill policy manifest: critic_online_target expected=cross_target_td actual=%s", online_target); ReturnFalse; } }
   { const string final_semantics = SkillManifestValue(manifest_file, "critic_final_semantics");
     if(final_semantics != "" && final_semantics != "discounted_live_return")
       { PrintFormat("Skill policy manifest: critic_final_semantics expected=discounted_live_return actual=%s", final_semantics); ReturnFalse; } }
   Skill_AC_MANIFEST_EQ("deal_horizon_bars", IntegerToString(SkillManifestDealHorizon));
   Skill_AC_MANIFEST_EQ("ambiguous_level_rule", "sl_before_tp_within_same_bar");
   Skill_AC_MANIFEST_EQ("actor_policy_gradient", "executable_single_critic_sigmoid_derivative");
   Skill_AC_MANIFEST_EQ("critic_coverage", "owner_split_seed_stable_offline_separate_online");
   Skill_AC_MANIFEST_EQ("actor_supervision", "policy_srm_single_owner_critic_plus_teacher_random");
   Skill_AC_MANIFEST_EQ("offline_units", "account_currency_money");
   Skill_AC_MANIFEST_EQ("online_reward", "account_currency_equity_delta");
   Skill_AC_MANIFEST_EQ("online_target", "cross_target_td_y_r_plus_gamma_sample_targetq");
   Skill_AC_MANIFEST_EQ("online_discount", StringFormat("%.4f", SkillManifestOnlineDiscount));
   Skill_AC_MANIFEST_EQ("actor_supervision_gradient", "combined_single_update_before_weight_change");
   Skill_AC_MANIFEST_EQ("policy_critic_selector", "actor_event_hash_seed_balanced");
   Skill_AC_MANIFEST_EQ("risk_variable", "return");
   Skill_AC_MANIFEST_EQ("cvar_convention", "lower_tail_return");
   Skill_AC_MANIFEST_EQ("srm_ordering", "weighted_monotone_rearrangement");
   Skill_AC_MANIFEST_EQ("srm_ordering_key", "return_ascending_index_tiebreak");
   Skill_AC_MANIFEST_EQ("sample_semantics", "discrete_fqf_representative");
   Skill_AC_MANIFEST_EQ("quantiles", IntegerToString(SkillManifestQuantiles));
   Skill_AC_MANIFEST_EQ("srm_invalid_policy", "skip_actor_update");
   { const string version_actual = SkillManifestValue(manifest_file, "config_version");
     if(version_actual != IntegerToString(Skill_AC_CONFIG_VERSION) && version_actual != "7" && version_actual != "8" && version_actual != "9")
       { PrintFormat("Skill policy manifest: config_version expected=%s or 7 actual=%s",
                     IntegerToString(Skill_AC_CONFIG_VERSION), version_actual); ReturnFalse; } }
   Skill_AC_MANIFEST_EQ("actor_hash",
                          StringFormat("%I64u",
                                       SkillHashFile(ulong(1469598103934665603), actor_file)));
   Skill_AC_MANIFEST_EQ("q1_hash",
                          StringFormat("%I64u",
                                       SkillHashFile(ulong(1469598103934665603), q1_file)));
   Skill_AC_MANIFEST_EQ("q2_hash",
                          StringFormat("%I64u",
                                       SkillHashFile(ulong(1469598103934665603), q2_file)));
#undef Skill_AC_MANIFEST_EQ
//--- Function if.
   if(SkillManifestValue(manifest_file, "generation") == "")
     {
      Print("Skill policy manifest: generation is absent");
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates one AC manifest file.                                  |
//+------------------------------------------------------------------+
bool SkillValidateACManifestFile(const bool required, const string manifest_file,
                                   const bool allow_bank_reset)
  {
   return(SkillValidateACManifestTuple(required, manifest_file, Skill_ACTOR_FILE,
                                         Skill_Q1_FILE, Skill_Q2_FILE, allow_bank_reset));
  }
//+------------------------------------------------------------------+
//| Validates the complete AC manifest state.                        |
//+------------------------------------------------------------------+
bool SkillValidateACManifest(const bool required)
  {
   return(SkillValidateACManifestFile(required, Skill_AC_MANIFEST_FILE, false));
  }
//+------------------------------------------------------------------+
//| Binds Actor and both Critics to the Market OpenCL context.       |
//+------------------------------------------------------------------+
bool SkillBindPolicyOpenCLChecked(CNet &actor, CNet &q1, CNet &q2)
  {
   COpenCLMy *target = SkillMarket.GetOpenCL();
   if(CheckPointer(target) == POINTER_INVALID ||
      !actor.SupportsOpenCLChecked() || !q1.SupportsOpenCLChecked() ||
      !q2.SupportsOpenCLChecked())
      ReturnFalse;
   if(!actor.SetOpenCLChecked(target) || !q1.SetOpenCLChecked(target) ||
      !q2.SetOpenCLChecked(target))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Loads a policy net checkpoint from file.                         |
//+------------------------------------------------------------------+
bool SkillLoadPolicyNet(CNet &net, const string file_name)
  {
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   return(net.Load(file_name, error, undefine, forecast, studied, true));
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Closes the offline episode through paired utility then reset.    |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| SkillCanPublishACCheckpoint validates replacement permission     |
//+------------------------------------------------------------------+
bool SkillCanPublishACCheckpoint(const bool canonical_exists,
                                   const bool canonical_valid)
  {
   if(!canonical_exists || canonical_valid)
      return(true);
   return(false);
  }
//+------------------------------------------------------------------+
//| Checks whether a policy checkpoint can be created.               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Loads or creates the policy set file.                            |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Loads or creates the full policy set.                            |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Implements SkillSavePolicySet.                                   |
//+------------------------------------------------------------------+
bool SkillSavePolicySet(CNet &actor, CNet &q1, CNet &q2)
  {
   if(!SkillVerifyFrozenWeightsExact() || !SkillVerifyFrozenCodebookExact() ||
      SkillForecastSignature() != SkillLastSignature)
      ReturnFalse;
   if(!SkillACRecoverTransaction(Skill_ACTOR_FILE, Skill_Q1_FILE, Skill_Q2_FILE,
                                   Skill_AC_MANIFEST_FILE, Skill_ACTOR_NEXT_FILE,
                                   Skill_Q1_NEXT_FILE, Skill_Q2_NEXT_FILE,
                                   Skill_AC_MANIFEST_NEXT_FILE, Skill_ACTOR_PREVIOUS_FILE,
                                   Skill_Q1_PREVIOUS_FILE, Skill_Q2_PREVIOUS_FILE,
                                   Skill_AC_MANIFEST_PREVIOUS_FILE,
                                   Skill_AC_TRANSACTION_FILE))
      ReturnFalse;
   const bool canonical_exists = (FileIsExist(Skill_ACTOR_FILE, FILE_COMMON) ||
                                  FileIsExist(Skill_Q1_FILE, FILE_COMMON) ||
                                  FileIsExist(Skill_Q2_FILE, FILE_COMMON) ||
                                  FileIsExist(Skill_AC_MANIFEST_FILE, FILE_COMMON));
   const bool canonical_valid = SkillValidateACManifestFile(true, Skill_AC_MANIFEST_FILE, false);
   const bool had_previous = canonical_valid;
   if(!SkillCanPublishACCheckpoint(canonical_exists, canonical_valid))
     {
      Print("Skill policy checkpoint=FAIL reason=canonical_tuple_invalid");
      ReturnFalse;
     }
   if(canonical_exists && !canonical_valid)
      Print("Skill policy checkpoint=RECREATE reason=canonical_tuple_invalid");
   const datetime now = TimeCurrent();
//--- Stage every model and write the candidate manifest without touching the active tuple.
   if(!actor.Save(Skill_ACTOR_NEXT_FILE, 0, 0, 0, now, true))
      ReturnFalseEx("Actor-Critic Actor candidate");
   if(!q1.Save(Skill_Q1_NEXT_FILE, q1.getRecentAverageError(), 0, 0, now, true))
      ReturnFalseEx("Actor-Critic Q1 candidate");
   if(!q2.Save(Skill_Q2_NEXT_FILE, q2.getRecentAverageError(), 0, 0, now, true))
      ReturnFalseEx("Actor-Critic Q2 candidate");
   if(!SkillWriteACManifestFile(SkillLastSignature, Skill_ACTOR_NEXT_FILE,
                                  Skill_Q1_NEXT_FILE, Skill_Q2_NEXT_FILE,
                                  Skill_AC_MANIFEST_NEXT_FILE))
      ReturnFalseEx("Actor-Critic candidate manifest");
   if(!SkillValidateACManifestTuple(true, Skill_AC_MANIFEST_NEXT_FILE,
                                      Skill_ACTOR_NEXT_FILE, Skill_Q1_NEXT_FILE,
                                      Skill_Q2_NEXT_FILE, false))
      ReturnFalseEx("Actor-Critic candidate validation");
   const ulong seed = ulong(1469598103934665603);
   const ulong actor_hash = SkillHashFile(seed, Skill_ACTOR_NEXT_FILE);
   const ulong q1_hash = SkillHashFile(seed, Skill_Q1_NEXT_FILE);
   const ulong q2_hash = SkillHashFile(seed, Skill_Q2_NEXT_FILE);
   const ulong manifest_hash = SkillHashFile(seed, Skill_AC_MANIFEST_NEXT_FILE);
   if(!SkillACWriteTransaction(Skill_AC_TRANSACTION_FILE, had_previous, "prepared",
                                 actor_hash, q1_hash, q2_hash, manifest_hash))
      ReturnFalse;
   if(had_previous)
     {
      if(!SkillACCopyTuple(Skill_ACTOR_FILE, Skill_Q1_FILE, Skill_Q2_FILE,
                             Skill_AC_MANIFEST_FILE, Skill_ACTOR_PREVIOUS_FILE,
                             Skill_Q1_PREVIOUS_FILE, Skill_Q2_PREVIOUS_FILE,
                             Skill_AC_MANIFEST_PREVIOUS_FILE, false) ||
         !SkillValidateACManifestTuple(true, Skill_AC_MANIFEST_PREVIOUS_FILE,
                                         Skill_ACTOR_PREVIOUS_FILE, Skill_Q1_PREVIOUS_FILE,
                                         Skill_Q2_PREVIOUS_FILE, false) ||
         !SkillACAdvanceTransaction(Skill_AC_TRANSACTION_FILE, true, "previous_ready",
                                      actor_hash, q1_hash, q2_hash, manifest_hash))
         ReturnFalse;
     }
//--- Publish the candidate tuple and validate it before removing recovery sidecars.
   if(!SkillACCopyTuple(Skill_ACTOR_NEXT_FILE, Skill_Q1_NEXT_FILE, Skill_Q2_NEXT_FILE,
                          Skill_AC_MANIFEST_NEXT_FILE, Skill_ACTOR_FILE,
                          Skill_Q1_FILE, Skill_Q2_FILE, Skill_AC_MANIFEST_FILE, true))
      ReturnFalseEx("Actor-Critic canonical publication");
   if(!SkillValidateACManifestFile(true, Skill_AC_MANIFEST_FILE, false))
      ReturnFalseEx("Actor-Critic canonical validation");
   if(SkillHashFile(seed, Skill_ACTOR_FILE) != actor_hash ||
      SkillHashFile(seed, Skill_Q1_FILE) != q1_hash ||
      SkillHashFile(seed, Skill_Q2_FILE) != q2_hash ||
      SkillHashFile(seed, Skill_AC_MANIFEST_FILE) != manifest_hash)
      ReturnFalseEx("Actor-Critic canonical hash");
   if(!SkillACCleanupTransaction(Skill_AC_TRANSACTION_FILE, Skill_ACTOR_NEXT_FILE,
                                   Skill_Q1_NEXT_FILE, Skill_Q2_NEXT_FILE,
                                   Skill_AC_MANIFEST_NEXT_FILE, Skill_ACTOR_PREVIOUS_FILE,
                                   Skill_Q1_PREVIOUS_FILE, Skill_Q2_PREVIOUS_FILE,
                                   Skill_AC_MANIFEST_PREVIOUS_FILE))
      ReturnFalseEx("Actor-Critic transaction cleanup");
   return(true);
  }
//+---------------------------------------------------------------------+
//| Saves a Stage 03 checkpoint only for the active production  ACSRM.  |
//+---------------------------------------------------------------------+
bool SkillSaveStage03StopCheckpoint(CNet &actor, CNet &q1, CNet &q2)
  {
   if(!SkillVerifyProductionACSRMFingerprints())
      ReturnFalse;
   return(SkillSavePolicySet(actor, q1, q2));
  }
//+------------------------------------------------------------------+
//| Implements SkillLoadInferenceActor.                              |
//+------------------------------------------------------------------+
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
//| Implements SkillAdvanceAccountTime.                              |
//+------------------------------------------------------------------+
bool SkillAdvanceAccountTime(CBufferFloat *current, const datetime next_time, CBufferFloat *next)
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
//| Implements SkillForwardForecastState.                            |
//+------------------------------------------------------------------+
bool SkillForwardForecastState(CBufferFloat *state)
  {
//--- CNet accepts a host-only input buffer and uploads it through its first
//--- layer.  Do not require a device allocation from the caller here.
   if(!SkillForecast || !state || state.Total() != HistoryBars * BarDescr)
      ReturnFalse;
   if(!SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = SkillForecast.GetZ(), *u = SkillForecast.GetU(), *pi = SkillForecast.GetPi();
   if(!z || !u || !pi || z.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize ||
      u.Total() != NScenarios * BarDescr * NForecast || pi.Total() != NScenarios ||
      z.GetIndex() < 0 || u.GetIndex() < 0 || pi.GetIndex() < 0)
      ReturnFalse;
   return(true);
  }
//+-----------------------------------------------------------------------+
//| Builds the live-account feature vector from balance/equity  changes.  |
//+-----------------------------------------------------------------------+
bool SkillBuildLiveAccount(const double previous_balance, const double previous_equity,
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
//| Implements SkillRefreshLiveMarket.                               |
//+------------------------------------------------------------------+
bool SkillRefreshLiveMarket(CBufferFloat *state, CBufferFloat *time_state)
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
//+-------------------------------------------------------------------+
//| Checks ExecutableOrder. One execution contract for historical...  |
//+-------------------------------------------------------------------+
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
//| Implements SkillValidateAction.                                  |
//+------------------------------------------------------------------+
bool SkillValidateAction(CBufferFloat *action)
  {
   if(!action || action.Total() != NActions ||
      (action.GetIndex() >= 0 && !action.BufferRead()))
      ReturnFalse;
   for(uint i = 0; i < NActions; i++)
      if(!MathIsValidNumber(action[i]) || action[i] < 0 || action[i] > 1)
         ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------------+
//| Executes the actor action, places the real deal and returns  margins.  |
//+------------------------------------------------------------------------+
bool SkillExecuteAction(CBufferFloat *action, double buy_value, double sell_value, double &margin_penalty,
                          bool &market_closed)
  {
   margin_penalty = 0;
   market_closed = false;
   if(!SkillValidateAction(action))
      ReturnFalse;
//--- Canonical CogDriver mutual exclusion and broker constraints.
   if(action[0] >= action[3])
     { action.Update(0, (action[0] - action[3])); action.Update(3, 0); }
   else
     { action.Update(3, (action[3] - action[0])); action.Update(0, 0); }
//---
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
         //--- Function if.
         if((lot - buy_value) >= min_lot && !Trade.Buy(lot - buy_value, Symb.Name(), Symb.Ask(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
            //--- Function if.
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return(true);
              }
            //--- Function if.
            if(retcode != 10019)
              {
               PrintFormat("Skill buy execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
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
         //--- Function if.
         if((lot - sell_value) >= min_lot && !Trade.Sell(lot - sell_value, Symb.Name(), Symb.Bid(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
            //--- Function if.
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return(true);
              }
            //--- Function if.
            if(retcode != 10019)
              {
               PrintFormat("Skill sell execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
               ReturnFalse;
              }
            margin_penalty -= 100.0 * (lot - sell_value);
           }
     }
   return(true);
  }
//+-------------------------------------------------------------------+
//| Implements SkillExecuteAction. Inference callers do not train...  |
//+-------------------------------------------------------------------+
bool SkillExecuteAction(CBufferFloat *action, double buy_value, double sell_value)
  {
   double margin_penalty = 0;
   bool market_closed = false;
   return(SkillExecuteAction(action, buy_value, sell_value, margin_penalty, market_closed));
  }
#endif
//+----------------------------------------------------------------------+
//| Finalizes projected balance before writing terminal account  state.  |
//+----------------------------------------------------------------------+
bool SkillFinalizeAccountBalance(const double balance, const double realized,
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
   const double buy = MathMax(0.0, double(action[0] - action[3]));
   const double sell = MathMax(0.0, double(action[3] - action[0]));
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Rates[position].open, margin))
      ReturnFalse;
   const double min_lot = Symb.LotsMin();
//--- Function if.
   if(balance <= min_balance || balance < margin * min_lot)
     {
      terminal = true;
      return(SkillAdvanceAccountTime(current, Rates[position - 1].time, next));
     }
//--- Action is the target position for the next bar.  A valid same-direction
//--- action modifies its TP/SL; a valid opposite action closes then reopens;
//--- an invalid action closes both positions.
   const bool open_buy = IsExecutableOrder(buy, action[1], action[2]);
   const bool open_sell = IsExecutableOrder(sell, action[4], action[5]);
//--- Unified volume contract: the environment evaluates the STATED
//--- continuous volume (money P&L is linear in the lot), never rounded up
//--- to LotsMin.  Broker executability stays the IsExecutableOrder gate and
//--- a separate Test-stage statistic, never a learning-label property.
   const double target_buy = (open_buy ? buy : 0.0);
   const double target_sell = (open_sell ? sell : 0.0);
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
//--- An invalid target liquidates every open position at the curre...
   if(!open_buy && !open_sell)
     {
      realized = current_buy_profit + current_sell_profit;
     }
   else
      //--- Function if.
      if(open_buy)
        {
         //--- A reverse target closes the Sell; a same-side reduction realizes only its
         //--- proportional carried P/L.  The retained Buy keeps its marked-to-market P/L.
         realized += current_sell_profit;
         //--- Function if.
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
         const MqlRates bar = Rates[position];
         if(sl >= bar.low)
            realized += current_buy_profit + (sl - entry) * point_cost * target_buy;
         else
            if(tp <= bar.high)
               realized += current_buy_profit + (tp - entry) * point_cost * target_buy;
            else
              {
               next_buy = target_buy;
               next_buy_profit = current_buy_profit + (Rates[position - 1].open - entry) * point_cost * target_buy;
              }
        }
      else
        {
         //--- Symmetric Sell lifecycle.  The SL-before-TP order matches CheckAction.
         realized += current_buy_profit;
         //--- Function if.
         if(current_sell > target_sell && current_sell > 0)
           {
            const double closed = current_sell - target_sell;
            realized += current_sell_profit * closed / current_sell;
            current_sell_profit -= current_sell_profit * closed / current_sell;
           }
         const double added = MathMax(0.0, target_sell - current_sell);
         current_sell_profit -= spread * point_cost * added;
         const double tp = entry - (action[4] * MaxTP + Symb.Spread()) * Symb.Point();
         const double sl = entry + (action[5] * MaxSL + Symb.Spread()) * Symb.Point();
         const MqlRates bar = Rates[position];
         if(sl <= bar.high)
            realized += current_sell_profit + (entry - sl) * point_cost * target_sell;
         else
            if(tp >= bar.low)
               realized += current_sell_profit + (entry - tp) * point_cost * target_sell;
            else
              {
               next_sell = target_sell;
               next_sell_profit = current_sell_profit + (entry - Rates[position - 1].open) * point_cost * target_sell;
              }
        }
   double next_balance = 0.0;
   bool balance_terminal = false;
   if(!SkillFinalizeAccountBalance(balance, realized, min_balance, next_balance, balance_terminal))
      ReturnFalse;
   const double current_equity = balance + double(current[6]) * balance + double(current[7]) * balance;
   const double next_equity = next_balance + next_buy_profit + next_sell_profit;
//--- Causal one-step equity change in ACCOUNT-CURRENCY units is the
//--- transition reward.  The offline deal label is also account currency,
//--- so the online bootstrap y = r + gamma * Sample(TargetQ_i) keeps one
//--- explicit dimension base everywhere (an earlier relative reward was
//--- added to money-dimensioned distribution samples).
   reward = (next_equity - current_equity);
   if(!MathIsValidNumber(reward))
      ReturnFalse;
   if(!SkillAdvanceAccountTime(current, Rates[position - 1].time, next) ||
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
//+---+