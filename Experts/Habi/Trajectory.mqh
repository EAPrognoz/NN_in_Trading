//+------------------------------------------------------------------+
//|                                                  Trajectory.mqh  |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#ifndef HABI_TRAJECTORY_MQH
#define HABI_TRAJECTORY_MQH
#define NN_HABI_INIT_DIAGNOSTICS
string NNHabiInitModel="Startup";
#include "..\NeuroNet_DNG\NeuroNet.mqh"

input group                "---- Indicators ----"
input ENUM_TIMEFRAMES      TimeFrame=PERIOD_CURRENT;
input group                "---- RSI ----"
input int                  RSIPeriod=14;
input ENUM_APPLIED_PRICE   RSIPrice=PRICE_CLOSE;
input group                "---- CCI ----"
input int                  CCIPeriod=14;
input ENUM_APPLIED_PRICE   CCIPrice=PRICE_TYPICAL;
input group                "---- ATR ----"
input int                  ATRPeriod=14;
input group                "---- MACD ----"
input int                  FastPeriod=12;
input int                  SlowPeriod=26;
input int                  SignalPeriod=9;
input ENUM_APPLIED_PRICE   MACDPrice=PRICE_CLOSE;

//--- Common configuration; edit here when changing the experiment.
const string            FileName="Habi";
const int               HistoryBars=24;                       // bars of history per sample
const int               LatentSize=256;                       // latent vector size for new models
const int               Candidates=64; // fixed experiment setting
const uint              normalization_batch=100000;           // normalization batch for new models
const uint              TrainingBatch=10000;                  // offline training batch size
const int               PlanningHorizonBars=24;               // native planning horizon in bars
const int               NForecast=12;                         // forecast steps for native layer
const double            DiscountGamma=0.5;                    // discount factor for returns
const double            VolumeCap=1;                          // maximum volume cap in lots
const double            MaxTP=1000;                           // maximum take profit in points
const double            MaxSL=1000;                           // maximum stop loss in points
const double            EtalonBalance=10000;                  // reference balance for normalization
const double            InitialOfflineBalance=0;              // initial balance for offline replay
const double            PenaltyInvalidCurrency=1; // fixed experiment setting
const bool              enable_critic_td=true;                // enables critic TD training
const bool              enable_policy=true;                   // enables policy training
const bool              train_prior=true;                     // trains the prior network
const bool              train_decoder=true;                   // trains the decoder network
const int               policy_period=5; // fixed experiment setting
const int               target_period=24*5; // fixed experiment setting
const bool              AdaptiveBeta=true;                    // adaptive beta instead of fixed value
const double            BetaValue=0.001;                          // initial adaptive beta or fixed beta
const double            BetaLearningRate=0.001;               // log-beta learning rate
const double            KTarget=5.042;                        // target summed KL(q||p)
const double            BetaRho1=0.9;                         // log-beta first moment decay
const double            BetaRho2=0.999;                       // log-beta second moment decay
const double            BetaEpsilon=1e-8;                     // log-beta optimizer epsilon
const uint              RandomSeed=0;                         // random seed, zero means time based
const string            PriorFile=FileName + "_Prior.nnw";
const string            PosteriorFile=FileName + "_Posterior.nnw";
const string            DecoderFile=FileName + "_Decoder.nnw";
const string            CriticFile=FileName + "_Critic.nnw";
const bool              CommonFiles=true;                     // use the common terminal folder
const ulong             MagicBase=100000; // fixed experiment setting
const ulong             MagicTest=200000; // fixed experiment setting
const ulong             AllowedDeviationPoints=0;              // allowed deviation in points
const datetime          OnlineEpisodeEnd=0;                    // optional end of a known online episode
const string            PriorImportFile="";                    // prior import file name
const string            PriorImportLayers="";                  // prior import layer indices
const string            PosteriorImportFile="";                // posterior import file name
const string            PosteriorImportLayers="";              // posterior import layer indices
const string            DecoderImportFile="";                  // decoder import file name
const string            DecoderImportLayers="";                // decoder import layer indices
const string            CriticImportFile="";                   // critic import file name
const string            CriticImportLayers="";                 // critic import layer indices
const bool              ResetLoadedBeta=false;                 // resets beta after a model load

//--- Offline inputs are declared by Study before this header.
#ifndef HABI_OFFLINE_STUDY
const double            MinBalance=0;                  // offline balance lower bound, zero unset
const double            MaxBalance=0;                  // offline balance upper bound, zero unset
const datetime          Start=0;                           // offline fit start time
const datetime          End=0;                             // offline fit end time
const int               Epochs=1;                           // number of offline fit passes
#endif

//--- compile time sizes of the native graph
#define BarDescription 9
#define AccountState 13
#define NActions 6
#define NRewards 1

//+------------------------------------------------------------------+
//| SState - one sample state passed to the model                    |
//+------------------------------------------------------------------+
struct SState
  {
   float             state[];      // observation buffer
   float             account[9];   // account features, HistoryBars*9
   float             action[6];    // action vector
   float             rewards[1];   // reward vector
  };
//+------------------------------------------------------------------+
//| SHabiGeometry - sizes of the native model graph                  |
//+------------------------------------------------------------------+
struct SHabiGeometry
  {
   uint              M;
   uint              O;
   uint              A;
   uint              L;
   uint              N;
  };
//+------------------------------------------------------------------+
//| HabiCommonConfig - validates the shared configuration            |
//+------------------------------------------------------------------+
bool HabiCommonConfig(bool training,bool online)
  {
   if(HistoryBars<1 || (training && (LatentSize<1 || TrainingBatch<1 || normalization_batch<2)) || VolumeCap<=0 || MaxTP<=0 || MaxSL<=0 || EtalonBalance<=0 || !MathIsValidNumber(VolumeCap+MaxTP+MaxSL+EtalonBalance+DiscountGamma) || DiscountGamma<0 || DiscountGamma>1)
      return(false);

   if(training && (!online || enable_critic_td) && (!MathIsValidNumber(PenaltyInvalidCurrency) || PenaltyInvalidCurrency<=0))
      return(false);

   if(training && online && ((enable_policy && policy_period<1) || (enable_critic_td && target_period<1)))
      return(false);

   return(true);
  }

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
