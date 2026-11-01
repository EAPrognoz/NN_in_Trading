//+------------------------------------------------------------------+
//|                                              Habi_Settings.mqh   |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#ifndef HABI_SETTINGS_MQH
#define HABI_SETTINGS_MQH

//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
input int            HistoryBars=5;                        // bars of history per sample
input int            LatentSize=256;                       // latent vector size for new models
input int            Candidates=0;                         // candidate count, zero means default
input uint           normalization_batch=100000;           // normalization batch for new models
input uint           TrainingBatch=10000;                  // offline training batch size
input int            PlanningHorizonBars=24;               // native planning horizon in bars
input int            NForecast=12;                         // forecast steps for native layer
input double         DiscountGamma=0.5;                    // discount factor for returns
input double         VolumeCap=1;                          // maximum volume cap in lots
input double         MaxTP=1000;                           // maximum take profit in points
input double         MaxSL=1000;                           // maximum stop loss in points
input double         EtalonBalance=10000;                  // reference balance for normalization
input double         InitialOfflineBalance=0;              // initial balance for offline replay
input double         OfflineBalanceMin=0;                  // offline balance lower bound, zero unset
input double         OfflineBalanceMax=0;                  // offline balance upper bound, zero unset
input double         PenaltyInvalidCurrency=0;             // invalid currency penalty, must be positive
input bool           enable_critic_td=true;                // enables critic TD training
input bool           enable_policy=true;                   // enables policy training
input bool           train_prior=true;                     // trains the prior network
input bool           train_decoder=true;                   // trains the decoder network
input int            policy_period=0;                      // policy update period in bars
input int            target_period=0;                      // target update period in bars
input bool           AdaptiveBeta=false;                   // adaptive beta instead of fixed value
input double         BetaValue=0;                          // fixed beta value
input double         BetaLearningRate=0;                   // beta learning rate, zero disables
input double         KTarget=0;                            // target coefficient, zero means default
input double         BetaRho1=-1;                          // beta EMA rho one, negative unset
input double         BetaRho2=-1;                          // beta EMA rho two, negative unset
input double         BetaEpsilon=0;                        // beta epsilon, zero means default
input uint           RandomSeed=0;                         // random seed, zero means time based
input string         PriorFile="";                         // prior model file name
input string         PosteriorFile="";                     // posterior model file name
input string         DecoderFile="";                       // decoder model file name
input string         CriticFile="";                        // critic model file name
input string         TargetPriorFile="";                   // target prior file name
input string         TargetDecoderFile="";                 // target decoder file name
input string         TargetCriticFile="";                  // target critic file name
input string         OutputPrefix="Habi";                  // prefix for saved model files
input bool           CommonFiles=true;                     // use the common terminal folder
input datetime       FitStart=0;                           // offline fit start time
input datetime       FitEnd=0;                             // offline fit end time
input int            FitPasses=1;                           // number of offline fit passes
input string         HistoricalCostsFile="";               // historical costs csv file name
input ulong          MagicBase=0;                           // base magic for online slots
input ulong          MagicTest=0;                           // magic for the single slot test
input ulong          AllowedDeviationPoints=0;              // allowed deviation in points
input ENUM_TIMEFRAMES DataTimeframe=PERIOD_CURRENT;         // working timeframe, current by default
input datetime       OnlineEpisodeEnd=0;                    // optional end of a known online episode
input string         PriorImportFile="";                    // prior import file name
input string         PriorImportLayers="";                  // prior import layer indices
input string         PosteriorImportFile="";                // posterior import file name
input string         PosteriorImportLayers="";              // posterior import layer indices
input string         DecoderImportFile="";                  // decoder import file name
input string         DecoderImportLayers="";                // decoder import layer indices
input string         CriticImportFile="";                   // critic import file name
input string         CriticImportLayers="";                 // critic import layer indices
input bool           ResetLoadedBeta=false;                 // resets beta after a model load

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
#endif

//+------------------------------------------------------------------+
