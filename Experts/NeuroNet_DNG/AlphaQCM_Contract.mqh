//+------------------------------------------------------------------+
//|                                            AlphaQCM_Contract.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#ifndef ALPHAQCM_CONTRACT_MQH
#define ALPHAQCM_CONTRACT_MQH

#define defNeuronAlphaQCMEncoderOCL 0x9002
#define defNeuronAlphaQCMOCL        0x9003
#define AQCM_FLAG_PUBLICATION_ONLY 128

enum ENUM_AQCM_LOAD_MODE
  {
   AQCM_TRAIN,
   AQCM_INFER,
   AQCM_VERIFY
  };

int AlphaQCMLoadMode=AQCM_TRAIN;
float AlphaQCMExpectedConfig[];

enum ENUM_AQCM_CONFIG
  {
   AQCM_CFG_BATCH=AG_CFG_SIZE,
   AQCM_CFG_CANDIDATES,
   AQCM_CFG_QUANTILES,
   AQCM_CFG_TARGET_QUANTILES,
   AQCM_CFG_POINTS,
   AQCM_CFG_REPLAY,
   AQCM_CFG_WARMUP,
   AQCM_CFG_CREDIT,
   AQCM_CFG_SYNC,
   AQCM_CFG_GAMMA,
   AQCM_CFG_KAPPA,
   AQCM_CFG_EPSILON,
   AQCM_CFG_BONUS,
   AQCM_CFG_FREQUENCIES,
   AQCM_CFG_PER_EPSILON,
   AQCM_CFG_PER_BETA,
   AQCM_CFG_PER_BETA_UPDATES,
   AQCM_CFG_PER_ALPHA,
   AQCM_CFG_SIZE
  };

enum ENUM_AQCM_STATUS
  {
   AQCM_RESOURCE_UNSUPPORTED=21,
   AQCM_NUMERIC_ERROR,
   AQCM_UPDATE_INCONSISTENT,
   AQCM_INCOMPLETE_DATA,
   AQCM_OPENCL_ERROR
  };

#endif // ALPHAQCM_CONTRACT_MQH
