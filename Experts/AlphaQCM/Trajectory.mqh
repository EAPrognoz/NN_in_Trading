//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                                         AlphaQCM |
//+------------------------------------------------------------------+
#ifndef ALPHAQCM_TRAJECTORY_MQH
#define ALPHAQCM_TRAJECTORY_MQH
#define NN_ALPHAQCM
#include "../NeuroNet_DNG/NeuroNet.mqh"

//+------------------------------------------------------------------+
//| AlphaQCM market and execution settings.                          |
//+------------------------------------------------------------------+
#define AG_WINDOW 512
#define AG_CHANNELS 5
#define AG_ACCOUNT_SIZE 13
#define AG_OUTPUT_SIZE 6
#define AG_LOOKBACK 120
#define AG_HORIZON 1
#define AG_IC_WINDOW 256
#define AG_NORM_WINDOW 256
#define AG_POOL_SIZE 100
#define AG_MAX_TOKENS 20
#define AG_CONTEXT_BARS 256
#define AG_WARMUP_BARS 631
#define AG_WARMUP_TRIALS 256
#define AG_SEARCH_TRIALS 8
#define AG_EMBEDDING 32
#define AG_SEARCH_HIDDEN 64
#define AG_TRADE_HIDDEN 32
#define AG_MLP_HIDDEN 32
#define AG_TRADING_ROLLOUT 128
#define AG_TOKEN_ROLLOUT 2048
#define AG_PPO_PASSES 4
#define AG_CLIP 0.2
#define AG_TRADE_GAMMA 0.99
#define AG_TOKEN_GAMMA 1.0
#define AG_LAMBDA 0.95
#define AG_VALUE_COEF 0.5
#define AG_ENTROPY_COEF 0.01
#define AG_EPSILON_VAR 0.000001
#define AG_ABSOLUTE_TOL 0.00001
#define AG_RELATIVE_TOL 0.0001
#define AG_RIDGE 0.0001
#define AG_SIGNAL_CLIP 10.0
#define AG_SIGMA_FLOOR 0.01
#define AG_VOLUME_CAP 1.0
#define AG_MAX_TP 1000.0
#define AG_MAX_SL 1000.0
#define AG_INVALID_PENALTY 0.0001
#define AG_RUIN_PENALTY 1.0
#define AG_EQUITY_FLOOR 0.10
#define AG_ETALON_BALANCE 10000.0
#define AG_MIN_TRADING_BALANCE 50.0
#define AG_LOT_EPSILON 0.0000000001
#define AG_AGE_SECONDS 36000.0
#define AG_YEAR_SECONDS 31536000.0
#define AG_PI 3.14159265358979323846
#define AG_POSITION_SCALE 10000.0
#define AG_INVALID_TOKEN_REWARD -1.0
#define AG_TIMER_MS 100
#define AG_CONTEXT_METADATA 5
#define AG_LIMB_BITS 16
#define AG_LIMB_MASK 65535
#define AG_TIME_LIMBS 4
#define AG_PROTECTION_MULTIPLIER 2.0
#define AG_CONST_NEGATIVE_ONE -1.0
#define AG_CONST_NEGATIVE_HALF -0.5
#define AG_CONST_HALF 0.5
#define AG_CONST_ONE 1.0
#define AG_CONST_TWO 2.0
#define AG_INTERVAL_ONE 1
#define AG_INTERVAL_FIVE 5
#define AG_INTERVAL_TEN 10
#define AG_INTERVAL_TWENTY 20
#define AG_INTERVAL_SIXTY 60
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES TimeFrame=PERIOD_H1;
input string          ModelFile="AlphaQCM\\AlphaQCM.nn";
input ulong           Seed=0;
input bool            SelectiveOpenCL=true;
#ifdef ALPHAGEN_STUDY
input datetime Start=0;
input datetime End=0;
input int      Epochs=1;
input double   MinBalance=100;
input double   MaxBalance=1500;
#endif

#define AQCM_PROFILE_BATCH 128
#define AQCM_PROFILE_CANDIDATES 8
#define AQCM_PROFILE_QUANTILES 64
#define AQCM_PROFILE_TARGET_QUANTILES 64
#define AQCM_PROFILE_POINTS 64
#define AQCM_PROFILE_REPLAY 100000
#define AQCM_PROFILE_WARMUP 10000
#define AQCM_PROFILE_CREDIT 32
#define AQCM_PROFILE_SYNC 160
#define AQCM_PROFILE_GAMMA 1.0
#define AQCM_PROFILE_KAPPA 1.0
#define AQCM_PROFILE_EPSILON 0.1
#define AQCM_PROFILE_BONUS 1.0
#define AQCM_PROFILE_FREQUENCIES 64
#define AQCM_PROFILE_PER_EPSILON 1.0e-6
#define AQCM_PROFILE_PER_BETA 0.4
#define AQCM_PROFILE_PER_BETA_UPDATES 200000
#define AQCM_PROFILE_PER_ALPHA 0.5

#include "../NeuroNet_DNG/NeuroNet_AlphaQCMApplication.mqh"
#endif
