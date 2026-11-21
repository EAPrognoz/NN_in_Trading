//+------------------------------------------------------------------+
//|                             NeuroNet_AlphaGenSelectiveOpenCL.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGEN_SELECTIVE_OPENCL_MQH
#define NEURONET_ALPHAGEN_SELECTIVE_OPENCL_MQH

#include "NeuroNet_SelectiveOpenCL.mqh"

//+------------------------------------------------------------------+
//| CAlphaGenSelectiveOpenCL - Collect the composite's GPU kernels   |
//+------------------------------------------------------------------+
class CAlphaGenSelectiveOpenCL:public CSelectiveOpenCL
  {
protected:
   virtual string ProgramSource(void) const override
     {
      return(cl_program+"\n"+ag_contract+"\n"+ag_program+"\n"+ag_pool_program);
     }

public:
   virtual bool RegisterSelectiveLayer(const int type,const ENUM_OPTIMIZATION optimizer) override
     {
      if(!IsCollectingSelectiveKernels()||optimizer!=ADAM)
         ReturnFalse;

      //--- The composite owns the search and trading policies and the pool.
      if(type==defNeuronBaseOCL||type==defNeuronAlphaGenEvaluatorOCL||type==defNeuronLSTMOCL)
         return(true);
      if(type!=defNeuronAlphaGenOCL)
         ReturnFalseEx(StringFormat("Unsupported AlphaGen layer type %d",type));

      if(!Require(def_k_AGContextMetadata,"AGContextMetadata")||
         !Require(def_k_AGCopySlice,"AGCopySlice")||
         !Require(def_k_AGDenseGradient,"AGDenseGradient")||
         !Require(def_k_AGDistribution,"AGDistribution")||
         !Require(def_k_AGEmbedding,"AGEmbedding")||
         !Require(def_k_AGEmbeddingBackward,"AGEmbeddingBackward")||
         !Require(def_k_AGEvaluate,"AGEvaluate")||
         !Require(def_k_AGGAE,"AGGAE")||
         !Require(def_k_AGLSTMBackward,"AGLSTMBackward")||
         !Require(def_k_AGLSTMForward,"AGLSTMForward")||
         !Require(def_k_AGLSTMWeightGradient,"AGLSTMWeightGradient")||
         !Require(def_k_AGMask,"AGMask")||
         !Require(def_k_AGNumeric,"AGNumeric")||
         !Require(def_k_AGNumericFinish,"AGNumericFinish")||
         !Require(def_k_AGPoolCommitBatch,"AGPoolCommitBatch")||
         !Require(def_k_AGPoolEvaluateBatch,"AGPoolEvaluateBatch")||
         !Require(def_k_AGPoolEvaluateHistory,"AGPoolEvaluateHistory")||
         !Require(def_k_AGPoolSignal,"AGPoolSignal")||
         !Require(def_k_AGPoolTransaction,"AGPoolTransaction")||
         !Require(def_k_AGReLU,"AGReLU")||
         !Require(def_k_AGReplayPrepare,"AGReplayPrepare")||
         !Require(def_k_AGRidge,"AGRidge")||
         !Require(def_k_AGSearchGenerate,"AGSearchGenerate")||
         !Require(def_k_AGSearchRecord,"AGSearchRecord")||
         !Require(def_k_AGSequence,"AGSequence")||
         !Require(def_k_AGStandardize,"AGStandardize")||
         !Require(def_k_CalcHiddenGradient,"CalcHiddenGradient")||
         !Require(def_k_CopyBufferRaw,"CopyBufferRaw")||
         !Require(def_k_FeedForward,"FeedForward")||
         !Require(def_k_LSTM_UpdateWeightsAdam,"LSTM_UpdateWeightsAdam")||
         !Require(def_k_MatMult,"MatMult"))
         ReturnFalse;

      return(true);
     }
  };

#endif // NEURONET_ALPHAGEN_SELECTIVE_OPENCL_MQH
