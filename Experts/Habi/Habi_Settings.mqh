#ifndef HABI_SETTINGS_MQH
#define HABI_SETTINGS_MQH
input int HistoryBars=5;
#define BarDescription 9
#define AccountState 13
#define NActions 6
#define NRewards 1
input int LatentSize=256;
input int Candidates=0;
// Required for new models; loaded geometry has priority.
input uint normalization_batch=100000;
input uint TrainingBatch=10000;
// Native description default; independent of N.
input int PlanningHorizonBars=24;
input int NForecast=12;
input double DiscountGamma=0.5;
input double VolumeCap=1;
input double MaxTP=1000;
input double MaxSL=1000;
input double EtalonBalance=10000;
input double InitialOfflineBalance=0;
input double OfflineBalanceMin=0;
input double OfflineBalanceMax=0;
input double PenaltyInvalidCurrency=0;
// Unset; must be explicitly positive.
input bool enable_critic_td=true;
input bool enable_policy=true;
input bool train_prior=true;
input bool train_decoder=true;
input int policy_period=0;
// Required only for enabled POLICY.
input int target_period=0;
// Required only for enabled TD.
input bool AdaptiveBeta=false;
input double BetaValue=0;
// Fixed zero is legal.
input double BetaLearningRate=0;
input double KTarget=0;
input double BetaRho1=-1;
input double BetaRho2=-1;
input double BetaEpsilon=0;
input uint RandomSeed=0;
input string PriorFile="";
input string PosteriorFile="";
input string DecoderFile="";
input string CriticFile="";
input string TargetPriorFile="";
input string TargetDecoderFile="";
input string TargetCriticFile="";
input string OutputPrefix="Habi";
input bool CommonFiles=true;
input datetime FitStart=0;
input datetime FitEnd=0;
input int FitPasses=1;
input string HistoricalCostsFile="";
input ulong MagicBase=0;
input ulong MagicTest=0;
input ulong AllowedDeviationPoints=0;
input ENUM_TIMEFRAMES DataTimeframe=PERIOD_CURRENT;
struct SState
  {
   float state[];
// Runtime HistoryBars: allocate exactly HistoryBars*9.
   float account[9];
   float action[6];
   float rewards[1];
  };
struct SHabiGeometry
  {
   uint M,O,A,L,N;
  };

//+------------------------------------------------------------------+
//| HabiCommonConfig.
//+------------------------------------------------------------------+
bool HabiCommonConfig(bool training,bool online)
  {
   if(HistoryBars<1 || (training && (LatentSize<1 || TrainingBatch<1 || normalization_batch<2)) || VolumeCap<=0 || MaxTP<=0 || MaxSL<=0 || EtalonBalance<=0 || !MathIsValidNumber(VolumeCap+MaxTP+MaxSL+EtalonBalance+DiscountGamma) || DiscountGamma<0 || DiscountGamma>1)
      return false;
   if(training && (!online || enable_critic_td) && (!MathIsValidNumber(PenaltyInvalidCurrency) || PenaltyInvalidCurrency<=0))
      return false;
   if(training && online && ((enable_policy && policy_period<1) || (enable_critic_td && target_period<1)))
      return false;
   return true;
  }
// Optional actual end of a known online episode; OnDeinit alone is never terminal.
input datetime OnlineEpisodeEnd=0;
// Narrow native affine import, applied over a valid Habi graph. Explicit source indices.
input string PriorImportFile="";
input string PriorImportLayers="";
input string PosteriorImportFile="";
input string PosteriorImportLayers="";
input string DecoderImportFile="";
input string DecoderImportLayers="";
input string CriticImportFile="";
input string CriticImportLayers="";
input bool ResetLoadedBeta=false;
#endif
