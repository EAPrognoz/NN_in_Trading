//+------------------------------------------------------------------+
//|                                     AlphaQCM_NetworkContract.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef ALPHAQCM_NETWORK_CONTRACT_MQH
#define ALPHAQCM_NETWORK_CONTRACT_MQH

#define def_k_AQCNetInit 570
#define def_k_AQCEmbed 571
#define def_k_AQCRecur 572
#define def_k_AQCBPTT 573
#define def_k_AQCRecurOuter 574
#define def_k_AQCEmbedOuter 575
#define def_k_AQCCosine 576
#define def_k_AQCDenseRows 577
#define def_k_AQCProduct 578
#define def_k_AQCDenseInput 579
#define def_k_AQCDenseOuter 580
#define def_k_AQCProductGrad 581
#define def_k_AQCParameterCheck 582
#define def_k_AQCSpanCheck 583
#define def_k_AQCTargetSelect 584
#define def_k_AQCQResidual 585
#define def_k_AQCQuantileTarget 586
#define def_k_AQCPairLoss 587
#define def_k_AQCExpandGrad 588
#define def_k_AQCPriorityLoss 589
#define def_k_AQCZeroGradient 590
#define def_k_AQCStageBatch 591

string AlphaQCMNetworkKernelName(const int id)
  {
   switch(id)
     {
      case def_k_AQCNetInit: return("AQCNetInit");
      case def_k_AQCEmbed: return("AQCEmbed");
      case def_k_AQCRecur: return("AQCRecur");
      case def_k_AQCBPTT: return("AQCBPTT");
      case def_k_AQCRecurOuter: return("AQCRecurOuter");
      case def_k_AQCEmbedOuter: return("AQCEmbedOuter");
      case def_k_AQCCosine: return("AQCCosine");
      case def_k_AQCDenseRows: return("AQCDenseRows");
      case def_k_AQCProduct: return("AQCProduct");
      case def_k_AQCDenseInput: return("AQCDenseInput");
      case def_k_AQCDenseOuter: return("AQCDenseOuter");
      case def_k_AQCProductGrad: return("AQCProductGrad");
      case def_k_AQCParameterCheck: return("AQCParameterCheck");
      case def_k_AQCSpanCheck: return("AQCSpanCheck");
      case def_k_AQCTargetSelect: return("AQCTargetSelect");
      case def_k_AQCQResidual: return("AQCQResidual");
      case def_k_AQCQuantileTarget: return("AQCQuantileTarget");
      case def_k_AQCPairLoss: return("AQCPairLoss");
      case def_k_AQCExpandGrad: return("AQCExpandGrad");
      case def_k_AQCPriorityLoss: return("AQCPriorityLoss");
      case def_k_AQCZeroGradient: return("AQCZeroGradient");
      case def_k_AQCStageBatch: return("AQCStageBatch");
     }
   return("");
  }
#endif
