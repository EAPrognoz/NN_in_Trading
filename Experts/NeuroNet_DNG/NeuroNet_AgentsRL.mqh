//+------------------------------------------------------------------+
//|                                             NeuroNet_AgentsRL.mqh|
//|           Reinforcement learning agents: FQF, SAC, PLR, MacroHFT.|
//|                                               Copyright 2024, DNG|
//|                                 https://www.mql5.com/ru/users/dng|
//+------------------------------------------------------------------+

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from                    |
//| NeuroNet.mqh: AgentsRL.                                          |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronFQF layer defaults.          |
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronFQF layer defaults.          |
//+------------------------------------------------------------------+
CNeuronFQF::CNeuronFQF(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronFQF GPU resources.               |
//+------------------------------------------------------------------+
CNeuronFQF::~CNeuronFQF(void)
  {
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronFQF layer and allocates its GPU buffers.  |
//+------------------------------------------------------------------+
bool CNeuronFQF::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint actions, uint quantiles, uint numInputs, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, actions, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   if(!cFraction.Init(0, myIndex, open_cl, actions * quantiles, optimization, batch))
      ReturnFalse;
   cFraction.SetActivationFunction(None);
//---
   if(!cSoftMax.Init(0, myIndex, open_cl, actions * quantiles, optimization, batch))
      ReturnFalse;
   cSoftMax.SetHeads(actions);
   cSoftMax.SetActivationFunction(None);
//---
   if(!cCosine.Init(numInputs, myIndex, open_cl, actions * quantiles, optimization, batch))
      ReturnFalse;
   cCosine.SetActivationFunction(None);
//---
   if(!cCosineEmbeding.Init(0, myIndex, open_cl, numInputs, optimization, batch))
      ReturnFalse;
   cCosineEmbeding.SetActivationFunction(LReLU);
//---
   if(!cQuantile0.Init(4 * actions * quantiles, myIndex, open_cl, numInputs, optimization, batch))
      ReturnFalse;
   cQuantile0.SetActivationFunction(None);
//---
   if(!cQuantile1.Init(actions * quantiles, myIndex, open_cl, 4 * actions * quantiles, optimization, batch))
      ReturnFalse;
   cQuantile1.SetActivationFunction(LReLU);
//---
   if(!cQuantile2.Init(0, myIndex, open_cl, actions * quantiles, optimization, batch))
      ReturnFalse;
   cQuantile2.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronFQF feed-forward pass and writes the layer       |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronFQF::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cFraction.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSoftMax.FeedForward(GetPointer(cFraction)))
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2];
      global_work_size[1] = Output.Total();
      global_work_size[0] = cSoftMax.Neurons() / global_work_size[1];
      setBuffer(def_k_FQF_Cosine, def_k_fqf_cosine_softmax, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_Cosine, def_k_fqf_cosine_outputs, cCosine.getOutputIndex());
      kernelExecute(def_k_FQF_Cosine, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cCosine.getOutput().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cCosineEmbeding.FeedForward(GetPointer(cCosine)))
      ReturnFalse;
//---
   if(!ElementMult(NeuronOCL.getOutput(), cCosineEmbeding.getOutput(), cQuantile0.getOutput()))
      ReturnFalse;
//---
   if(!cQuantile1.FeedForward(GetPointer(cQuantile0)))
      ReturnFalse;
//---
   if(!cQuantile2.FeedForward(GetPointer(cQuantile1)))
      ReturnFalse;
//---
     {
      uint global_work_offset[1] = {0};
      uint global_work_size[1] = { Neurons() };
      setBuffer(def_k_FQF_Output, def_k_fqfout_quantiles, cQuantile2.getOutputIndex());
      setBuffer(def_k_FQF_Output, def_k_fqfout_delta_taus, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_Output, def_k_fqfout_output, getOutputIndex());
      setArgument(def_k_FQF_Output, def_k_fqfout_total, (int)(cQuantile2.Neurons() / global_work_size[0]));
      kernelExecute(def_k_FQF_Output, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!Output.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronFQF layer.|
//+------------------------------------------------------------------+
bool CNeuronFQF::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !Gradient || !Output)
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cSoftMax.Neurons() / Neurons(), Neurons() };
      setBuffer(def_k_FQF_OutputGradient, def_k_fqfoutgr_quantiles, cQuantile2.getOutputIndex());
      setBuffer(def_k_FQF_OutputGradient, def_k_fqfoutgr_taus, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_OutputGradient, def_k_fqfoutgr_output_gr, getGradientIndex());
      setBuffer(def_k_FQF_OutputGradient, def_k_fqfoutgr_quantiles_gr, cQuantile2.getGradientIndex());
      setBuffer(def_k_FQF_OutputGradient, def_k_fqfoutgr_taus_gr, cSoftMax.getGradientIndex());
      kernelExecute(def_k_FQF_OutputGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cQuantile2.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cQuantile1.CalcHiddenGradients(cQuantile2.AsObject()))
      ReturnFalse;
   if(!cQuantile0.CalcHiddenGradients(cQuantile1.AsObject()))
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cCosineEmbeding.Neurons(), 1 };
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_state_enbeding, NeuronOCL.getOutputIndex());
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_taus_embedding, cCosineEmbeding.getOutputIndex());
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_quantiles_gr, cQuantile0.getGradientIndex());
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_state_gr, NeuronOCL.getGradientIndex());
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_taus_gr, cCosineEmbeding.getGradientIndex());
      kernelExecute(def_k_FQF_QuantileGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!NeuronOCL.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cCosineEmbeding.CalcHiddenGradients(cCosine.AsObject()))
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cSoftMax.Neurons() / Neurons(), Neurons() };
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_softmax, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_output_gr, cCosine.getGradientIndex());
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_softmax_gr, cSoftMax.getGradientIndex());
      kernelExecute(def_k_FQF_CosineGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cSoftMax.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cFraction.CalcHiddenGradients(cSoftMax.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronFQF trainable weights from the accumulated    |
//| gradients.                                                       |
//+------------------------------------------------------------------+
bool CNeuronFQF::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cFraction.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cCosineEmbeding.UpdateInputWeights(GetPointer(cCosine)))
      ReturnFalse;
   if(!cQuantile1.UpdateInputWeights(GetPointer(cQuantile0)))
      ReturnFalse;
   if(!cQuantile2.UpdateInputWeights(GetPointer(cQuantile1)))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFQF state to the file handle.                   |
//+------------------------------------------------------------------+
bool CNeuronFQF::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cFraction.Save(file_handle))
      ReturnFalse;
   if(!cSoftMax.Save(file_handle))
      ReturnFalse;
   if(!cCosine.Save(file_handle))
      ReturnFalse;
   if(!cCosineEmbeding.Save(file_handle))
      ReturnFalse;
   if(!cQuantile0.Save(file_handle))
      ReturnFalse;
   if(!cQuantile1.Save(file_handle))
      ReturnFalse;
   if(!cQuantile2.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFQF state from the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronFQF::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cFraction.Type() ||
      !cFraction.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cSoftMax.Type() ||
      !cSoftMax.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cCosine.Type() ||
      !cCosine.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cCosineEmbeding.Type() ||
      !cCosineEmbeding.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cQuantile0.Type() ||
      !cQuantile0.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cQuantile1.Type() ||
      !cQuantile1.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cQuantile2.Type() ||
      !cQuantile2.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Returns the layer info of the CNeuronFQF.                        |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronFQF::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
   result.window_out = cSoftMax.Neurons() / Neurons();
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronFQF GPU buffers to the given OpenCL context.  |
//+------------------------------------------------------------------+
void CNeuronFQF::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cFraction.SetOpenCL(OpenCL);
   cSoftMax.SetOpenCL(OpenCL);
   cCosine.SetOpenCL(OpenCL);
   cCosineEmbeding.SetOpenCL(OpenCL);
   cQuantile0.SetOpenCL(OpenCL);
   cQuantile1.SetOpenCL(OpenCL);
   cQuantile2.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronSoftActorCritic layer and allocates its   |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint actions, uint quantiles, uint numInputs, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronFQF::Init(numOutputs, myIndex, open_cl, actions, quantiles, numInputs, optimization_type, batch))
      ReturnFalse;
//---
   if(!cAlphas.Init(0, 0, OpenCL, actions, numInputs, cQuantile2.Neurons(), optimization_type, batch))
      ReturnFalse;
   cAlphas.SetActivationFunction(SIGMOID);
//---
   if(!cLogProbs.BufferInit(actions, 0) || !cLogProbs.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!cRandomize.BufferInit(actions, 0) || !cRandomize.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronSoftActorCritic feed-forward pass and writes the |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronFQF::feedForward(NeuronOCL))
      ReturnFalse;
   if(!cAlphas.FeedForward(GetPointer(cQuantile0), cQuantile2.getOutput()))
      ReturnFalse;
//---
   int actions = cRandomize.Total();
   for(int i = 0; i < actions; i++)
     {
      float probability = (float)MathRand() / 32767.0f;
      cRandomize.Update(i, probability);
     }
   if(!cRandomize.BufferWrite())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_alphas, cAlphas.getOutputIndex())
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_log_probs, cLogProbs.GetIndex())
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_outputs, getOutputIndex())
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_probs, cSoftMax.getOutputIndex())
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_quantiles, cQuantile2.getOutputIndex())
   setBuffer(def_k_SAC_AlphaLogProbs, def_k_sac_alp_random, cRandomize.GetIndex())
   setArgument(def_k_SAC_AlphaLogProbs, def_k_sac_alp_count_quants, (int)(cSoftMax.Neurons() / global_work_size[0]))
   setArgument(def_k_SAC_AlphaLogProbs, def_k_sac_alp_activation, (int)activation)
   kernelExecute(def_k_SAC_AlphaLogProbs, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cRandomize.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronSoftActorCritic calc alpha gradients          |
//| operation.                                                       |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::calcAlphaGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || !NeuronOCL.getGradient() ||
      NeuronOCL.getGradientIndex() < 0)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   setBuffer(def_k_SAC_AlphaGradients, def_k_sac_alg_outputs, cAlphas.getOutputIndex())
   setBuffer(def_k_SAC_AlphaGradients, def_k_sac_alg_alphas_grad, cAlphas.getGradientIndex())
   setBuffer(def_k_SAC_AlphaGradients, def_k_sac_alg_gradient, NeuronOCL.getGradientIndex())
   setBuffer(def_k_SAC_AlphaGradients, def_k_sac_alg_log_probs, cLogProbs.GetIndex())
   setArgument(def_k_SAC_AlphaGradients, def_k_sac_alg_activation, (int)cAlphas.Activation())
   kernelExecute(def_k_SAC_AlphaGradients, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cLogProbs.BufferRead())
      ReturnFalse;
#endif
//---
   return cQuantile0.CalcHiddenGradients((CObject*)GetPointer(cAlphas), cQuantile2.getOutput(), cQuantile2.getGradient());
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronSoftActorCritic trainable weights from the    |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronFQF::updateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return cAlphas.UpdateInputWeights(cQuantile0.AsObject(), cQuantile2.getOutput());
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronSoftActorCritic weights toward the source    |
//| model with factor tau.                                           |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronFQF::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSoftActorCritic *Source = source;
//---
   return cAlphas.WeightsUpdate(GetPointer(Source.cAlphas), tau);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronSoftActorCritic GPU buffers to the given      |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronSoftActorCritic::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronFQF::SetOpenCL(obj);
   cAlphas.SetOpenCL(OpenCL);
   if(cLogProbs.Total() != Neurons())
      cLogProbs.BufferInit(Neurons(), 0);
   cLogProbs.BufferCreate(OpenCL);
   if(cRandomize.Total() != Neurons())
      cRandomize.BufferInit(Neurons(), 0);
   cRandomize.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSoftActorCritic state to the file handle.       |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::Save(const int file_handle)
  {
   if(!CNeuronFQF::Save(file_handle))
      ReturnFalse;
   return cAlphas.Save(file_handle);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSoftActorCritic state from the file handle.  |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::Load(const int file_handle)
  {
   if(!CNeuronFQF::Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cAlphas.Type() ||
      !cAlphas.Load(file_handle))
      ReturnFalse;
//---
   if(cLogProbs.Total() != Neurons())
     {
      cLogProbs.BufferFree();
      if(!cLogProbs.BufferInit(Neurons(), 0) ||
         (!!OpenCL && !cLogProbs.BufferCreate(OpenCL)))
         ReturnFalse;
     }
//---
   if(cRandomize.Total() != Neurons())
     {
      cRandomize.BufferFree();
      if(!cRandomize.BufferInit(Neurons(), 0) ||
         (!!OpenCL && !cRandomize.BufferCreate(OpenCL)))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronSoftActorCritic calc log probs operation.     |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::CalcLogProbs(CBufferFloat* buffer)
  {
   if(!buffer || buffer.Total() < Neurons())
      ReturnFalse;
   if(buffer.GetIndex() < 0 && !buffer.BufferCreate(OpenCL))
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_alphas, cAlphas.getOutputIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_log_probs, buffer.GetIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_outputs, buffer.GetIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_probs, cSoftMax.getOutputIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_quantiles, cQuantile2.getOutputIndex())
   setArgument(def_k_SAC_CalcLogProbs, def_k_sacclp_count_quants, (int)(cSoftMax.Neurons() / global_work_size[0]))
   setArgument(def_k_SAC_CalcLogProbs, def_k_sacclp_activation, (int)Activation())
   kernelExecute(def_k_SAC_CalcLogProbs, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cQuantile2.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronSoftActorCritic re calc log probs operation.  |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::ReCalcLogProbs(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_alphas, cAlphas.getOutputIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_log_probs, cLogProbs.GetIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_outputs, Output.GetIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_probs, cSoftMax.getOutputIndex())
   setBuffer(def_k_SAC_CalcLogProbs, def_k_sacclp_quantiles, cQuantile2.getOutputIndex())
   setArgument(def_k_SAC_CalcLogProbs, def_k_sacclp_count_quants, (int)(cSoftMax.Neurons() / global_work_size[0]))
   setArgument(def_k_SAC_CalcLogProbs, def_k_sacclp_activation, (int)Activation())
   kernelExecute(def_k_SAC_CalcLogProbs, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cQuantile2.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronSoftActorCritic layer.                                    |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !Gradient || !Output)
      ReturnFalse;
//---
     {
      uint global_work_offset[1] = {0};
      uint global_work_size[1] = { Neurons() };
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_quantiles, cQuantile2.getOutputIndex())
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_taus, cSoftMax.getOutputIndex())
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_output_gr, getGradientIndex())
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_quantiles_gr, cQuantile2.getGradientIndex())
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_taus_gr, cSoftMax.getGradientIndex())
      setBuffer(def_k_SAC_OutputGradient, def_k_sacoutgr_outputs, getOutputIndex())
      setArgument(def_k_SAC_OutputGradient, def_k_sacoutgr_activation, (int)Activation())
      setArgument(def_k_SAC_OutputGradient, def_k_sacoutgr_count_quants, (int)(cSoftMax.Neurons() / Neurons()))
      kernelExecute(def_k_SAC_OutputGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cSoftMax.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cQuantile1.CalcHiddenGradients((CObject *)GetPointer(cQuantile2)))
      ReturnFalse;
   if(!cQuantile0.CalcHiddenGradients((CObject *)GetPointer(cQuantile1)))
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cCosineEmbeding.Neurons(), 1 };
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_state_enbeding, NeuronOCL.getOutputIndex())
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_taus_embedding, cCosineEmbeding.getOutputIndex())
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_quantiles_gr, cQuantile0.getGradientIndex())
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_state_gr, NeuronOCL.getGradientIndex())
      setBuffer(def_k_FQF_QuantileGradient, def_k_fqfqgr_taus_gr, cCosineEmbeding.getGradientIndex())
      kernelExecute(def_k_FQF_QuantileGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cCosineEmbeding.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   if(!cCosine.CalcHiddenGradients((CObject *)GetPointer(cCosineEmbeding)))
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cSoftMax.Neurons() / Neurons(), Neurons() };
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_softmax, cSoftMax.getOutputIndex())
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_output_gr, cCosine.getGradientIndex())
      setBuffer(def_k_FQF_CosineGradient, def_k_fqfcosgr_softmax_gr, cSoftMax.getGradientIndex())
      kernelExecute(def_k_FQF_CosineGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cSoftMax.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
   if(!cSoftMax.CalcHiddenGradients((CObject *)GetPointer(cFraction)))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Returns the check data of the CNeuronSoftActorCritic.            |
//+------------------------------------------------------------------+
void CNeuronSoftActorCritic::getCheckData(vector<float> &output, vector<float> &grad, vector<float> &quantiles, vector<float> &probability, vector<float> &alphas)
  {
   Output.GetData(output);
   Gradient.GetData(grad);
   cQuantile2.getGradient().GetData(quantiles);
   cSoftMax.getGradient().GetData(probability);
   cAlphas.getGradient().GetData(alphas);
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronFQF weights toward the source model with     |
//| factor tau.                                                      |
//+------------------------------------------------------------------+
bool CNeuronFQF::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronFQF *temp = source;
   if(!cFraction.WeightsUpdate(GetPointer(temp.cFraction), tau))
      ReturnFalse;
   if(!cSoftMax.WeightsUpdate(GetPointer(temp.cSoftMax), tau))
      ReturnFalse;
   if(!cCosine.WeightsUpdate(GetPointer(temp.cCosine), tau))
      ReturnFalse;
   if(!cCosineEmbeding.WeightsUpdate(GetPointer(temp.cCosineEmbeding), tau))
      ReturnFalse;
   if(!cQuantile0.WeightsUpdate(GetPointer(temp.cQuantile0), tau))
      ReturnFalse;
   if(!cQuantile1.WeightsUpdate(GetPointer(temp.cQuantile1), tau))
      ReturnFalse;
   if(!cQuantile2.WeightsUpdate(GetPointer(temp.cQuantile2), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPLROCL layer and allocates its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window_in, uint units_count, bool transpose, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_in * units_count, optimization_type, batch))
      ReturnFalse;
//---
   iVariables = (int)window_in;
   iCount = (int)units_count;
   bTranspose = transpose;
//---
   icIsTTP = OpenCL.AddBuffer(sizeof(int) * Neurons(), CL_MEM_READ_WRITE);
   if(icIsTTP < 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronPLROCL feed-forward pass and writes the layer    |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {iCount, iVariables};
   uint local_work_size[2] = {iCount, 1};
   ResetLastError();
   setBuffer(def_k_PLR, def_k_plr_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_PLR, def_k_plr_outputs, getOutputIndex())
   setBuffer(def_k_PLR, def_k_plt_isttp, icIsTTP)
   setArgument(def_k_PLR, def_k_plr_transpose, (int)bTranspose)
   setArgument(def_k_PLR, def_k_plr_step, (float)0.2)
//---
   kernelExecuteLoc(def_k_PLR, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronPLROCL    |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!OpenCL || !prevLayer || !prevLayer.getGradient())
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {iCount, iVariables};
   ResetLastError();
   setBuffer(def_k_PLRGrad, def_k_plrg_inputs_gr, prevLayer.getGradientIndex())
   setBuffer(def_k_PLRGrad, def_k_plrg_outputs, getOutputIndex())
   setBuffer(def_k_PLRGrad, def_k_plrg_outputs_gr, getGradientIndex())
   setArgument(def_k_PLRGrad, def_k_plr_transpose, (int)bTranspose)
//---
   kernelExecute(def_k_PLRGrad, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!prevLayer.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronPLROCL state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iCount) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, iVariables) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(bTranspose)) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronPLROCL state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCount = FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   bTranspose = (bool)FileReadInteger(file_handle);
//---
   OpenCL.BufferFree(icIsTTP);
   icIsTTP = OpenCL.AddBuffer(sizeof(int) * Neurons(), CL_MEM_READ_WRITE);
   if(icIsTTP < 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronPLROCL GPU buffers to the given OpenCL        |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronPLROCL::SetOpenCL(COpenCLMy * obj)
  {
   if(!!OpenCL && icIsTTP >= 0)
      OpenCL.BufferFree(icIsTTP);
//---
   CNeuronBaseOCL::SetOpenCL(obj);
   icIsTTP = OpenCL.AddBuffer(sizeof(float) * Neurons(), CL_MEM_READ_WRITE);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMarketObserver layer and allocates its GPU|
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMarketObserver::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                 uint window, uint window_key, uint units_count,
                                 uint heads, uint layers, uint forecast,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- Init parent object
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * forecast, optimization_type, batch))
      ReturnFalse;
//--- Clear layers' array
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
//--- Tranpose input data
   int lay_count = 0;
   CNeuronTransposeOCL *transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, lay_count, OpenCL, units_count, window, optimization, iBatch) ||
      !cLayers.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
//--- Piecewise linear representation
   lay_count++;
   CNeuronPLROCL *plr = new CNeuronPLROCL();
   if(!plr ||
      !plr.Init(0, lay_count, OpenCL, units_count, window, false, optimization, iBatch) ||
      !cLayers.Add(plr))
     {
      DeleteObj(plr);
      ReturnFalse;
     }
//--- Self-Attention for Variables
   lay_count++;
   CNeuronRMAT *att = new CNeuronRMAT();
   if(!att ||
      !att.Init(0, lay_count, OpenCL, units_count, window_key, window, heads, layers, optimization, iBatch) ||
      !cLayers.Add(att))
     {
      DeleteObj(att);
      ReturnFalse;
     }
//--- Forecast mapping
   lay_count++;
   CResidualConv *conv = new CResidualConv();
   if(!conv ||
      !conv.Init(0, lay_count, OpenCL, units_count, forecast, window, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//--- Back transpose forecast
   lay_count++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, lay_count, OpenCL, window, forecast, optimization, iBatch) ||
      !cLayers.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
//---
   if(!SetOutput(transp.getOutput(), true) ||
      !SetGradient(transp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronRLAgent layer and allocates its GPU       |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronRLAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                          uint window, uint units_count, uint segments,
                          float rho, uint layers, uint n_actions,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- Init parent object
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, n_actions, optimization_type, batch))
      ReturnFalse;
//--- Clear layers' array
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
//--- State observation
   int lay_count = 0;
   for(uint i = 0; i < layers; i++)
     {
      CNeuronPSformer *psf = new CNeuronPSformer();
      if(!psf ||
         !psf.Init(0, lay_count, OpenCL, window, units_count, segments, rho, optimization, iBatch) ||
         !cLayers.Add(psf))
        {
         DeleteObj(psf);
         ReturnFalse;
        }
      lay_count++;
     }
//---
   CNeuronConvSAMOCL *conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(n_actions, lay_count, OpenCL, window, window, 1, units_count, 1, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(GELU);
   lay_count++;
   CNeuronBaseSAMOCL *flat = new CNeuronBaseSAMOCL();
   if(!flat ||
      !flat.Init(0, lay_count, OpenCL, n_actions, optimization, iBatch) ||
      !cLayers.Add(flat))
     {
      DeleteObj(flat);
      ReturnFalse;
     }
   SetActivationFunction(SIGMOID);
//---
   if(!SetOutput(flat.getOutput(), true) ||
      !SetGradient(flat.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronControlAgent layer and allocates its GPU  |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronControlAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                               uint window, uint window_key, uint units_count,
                               uint heads, uint window_kv, uint units_kv, uint layers,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- Init parent object
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
//--- Clear layers' array
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
//--- Second buffer prepare
   int lay_count = 0;
   CNeuronBaseOCL *flat = new CNeuronBaseOCL();
   if(!flat ||
      !flat.Init(0, lay_count, OpenCL, window_kv * units_kv, optimization, iBatch) ||
      !cLayers.Add(flat))
     {
      DeleteObj(flat);
      ReturnFalse;
     }
   lay_count++;
   CNeuronTransposeOCL *transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, lay_count, OpenCL, units_kv, window_kv, optimization, iBatch) ||
      !cLayers.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   lay_count++;
//--- Attention Action To Observation
   for(uint i = 0; i < layers; i++)
     {
      if(units_count > 1)
        {
         CNeuronRelativeSelfAttention *self = new CNeuronRelativeSelfAttention();
         if(!self ||
            !self.Init(0, lay_count, OpenCL, window, window_key, units_count, heads, optimization, iBatch) ||
            !cLayers.Add(self))
           {
            DeleteObj(self);
            ReturnFalse;
           }
         lay_count++;
        }
      CNeuronRelativeCrossAttention *cross = new CNeuronRelativeCrossAttention();
      if(!cross ||
         !cross.Init(0, lay_count, OpenCL, window, window_key, units_count, heads, units_kv, window_kv, optimization, iBatch) ||
         !cLayers.Add(cross))
        {
         DeleteObj(cross);
         ReturnFalse;
        }
      lay_count++;
      CResidualConv *ffn = new CResidualConv();
      if(!ffn ||
         !ffn.Init(0, lay_count, OpenCL, window, window, units_count, optimization, iBatch) ||
         !cLayers.Add(ffn))
        {
         DeleteObj(ffn);
         ReturnFalse;
        }
      lay_count++;
     }
   CNeuronConvSAMOCL *conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, lay_count, OpenCL, window, window, window, units_count, 1, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   SetActivationFunction(SIGMOID);
//---
   if(!SetOutput(conv.getOutput(), true) ||
      !SetGradient(conv.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronControlAgent feed-forward pass and writes the    |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronControlAgent::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!SecondInput)
      ReturnFalse;
//---
   CNeuronBaseOCL *second = cLayers[0];
   if(!second)
      ReturnFalse;
   if(!second.SetOutput(SecondInput, true))
      ReturnFalse;
//---
   second = cLayers[1];
   if(!second || !second.FeedForward(cLayers[0]))
      ReturnFalse;
//---
   CNeuronBaseOCL *first = NeuronOCL;
   CNeuronBaseOCL *main = NULL;
   for(int i = 2; i < cLayers.Total(); i++)
     {
      main = cLayers[i];
      if(!main ||
         !main.FeedForward(first, second.getOutput()))
         ReturnFalse;
      first = main;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronControlAgent layer.                                       |
//+------------------------------------------------------------------+
bool CNeuronControlAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
//---
   CNeuronBaseOCL *main = cLayers[0];
   if(!main)
      ReturnFalse;
   if(!main.SetGradient(SecondGradient, true))
      ReturnFalse;
   main.SetActivationFunction(SecondActivation);
//---
   CNeuronBaseOCL *second = cLayers[1];
   if(!second)
      ReturnFalse;
   second.SetActivationFunction(SecondActivation);
   CBufferFloat *second_out = second.getOutput();
   CBufferFloat *second_gr = second.getGradient();
   CBufferFloat *temp = second.getPrevOutput();
   if(!second_gr.Fill(0))
      ReturnFalse;
//---
   for(int i = cLayers.Total() - 2; i >= 2; i--)
     {
      main = cLayers[i];
      if(!main)
         ReturnFalse;
      if(cLayers[i + 1].Type() == defNeuronRelativeCrossAttention)
        {
         if(!main.CalcHiddenGradients(cLayers[i + 1], second_out, temp, SecondActivation) ||
            !SumAndNormalize(temp, second_gr, second_gr, 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      else
        {
         if(!main.CalcHiddenGradients(cLayers[i + 1]))
            ReturnFalse;
        }
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(main.AsObject(), second_out, temp, SecondActivation))
      ReturnFalse;
   if(main.Type() == defNeuronRelativeCrossAttention)
     {
      if(!SumAndNormalize(temp, second_gr, second_gr, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   main = cLayers[0];
   if(!main.CalcHiddenGradients(second.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronControlAgent trainable weights from the       |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronControlAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   CNeuronBaseOCL *second = cLayers[1];
   CNeuronBaseOCL *main = NULL;
   for(int i = cLayers.Total() - 1; i > 2; i--)
     {
      main = cLayers[i];
      if(!main ||
         !main.UpdateInputWeights(cLayers[i - 1], second.getOutput()))
         ReturnFalse;
     }
   main = cLayers[2];
   if(!main ||
      !main.UpdateInputWeights(NeuronOCL, second.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMASA layer and allocates its GPU buffers. |
//+------------------------------------------------------------------+
bool CNeuronMASA::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                       uint window, uint window_key, uint units_count,
                       uint heads_mo, uint layers_mo, uint forecast, uint segments_rl,
                       float rho, uint layers_rl, uint n_actions, uint heads_contr,
                       uint layers_contr, int NormLayer, CNeuronBatchNormOCL * normLayer,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseSAMOCL::Init(numOutputs, myIndex, open_cl, n_actions, rho, optimization_type, batch))
      ReturnFalse;
//--- Market Observation
   if(!cMarketObserver.Init(0, 0, OpenCL, window, window_key, units_count, heads_mo, layers_mo, forecast, optimization, iBatch))
      ReturnFalse;
   if(!cRevIN.Init(0, 1, OpenCL, cMarketObserver.Neurons(), NormLayer, normLayer))
      ReturnFalse;
//--- RL Agent
   if(!cRLAgent.Init(0, 2, OpenCL, window, units_count, segments_rl, fRho, layers_rl, n_actions, optimization, iBatch))
      ReturnFalse;
//---
   if(!cControlAgent.Init(0, 3, OpenCL, 3, window_key, n_actions / 3, heads_contr, window, forecast, layers_contr, optimization, iBatch))
      ReturnFalse;
//---
   if(!SetOutput(cControlAgent.getOutput(), true) ||
      !SetGradient(cControlAgent.getGradient(), true))
      ReturnFalse;
   SetActivationFunction(SIGMOID);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Sets the activation function of the CNeuronMASA.                 |
//+------------------------------------------------------------------+
void CNeuronMASA::SetActivationFunction(ENUM_ACTIVATION value)
  {
   cControlAgent.SetActivationFunction(value);
   cRLAgent.SetActivationFunction((ENUM_ACTIVATION)cControlAgent.Activation());
   CNeuronBaseSAMOCL::SetActivationFunction((ENUM_ACTIVATION)cControlAgent.Activation());
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMASA feed-forward pass and writes the layer      |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMASA::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cMarketObserver.FeedForward(NeuronOCL.AsObject()))
      ReturnFalse;
   if(!cRLAgent.FeedForward(NeuronOCL.AsObject()))
      ReturnFalse;
   if(!cControlAgent.FeedForward(cRLAgent.AsObject(), cMarketObserver.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMASA      |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronMASA::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cRevIN.FeedForward(cMarketObserver.AsObject()))
      ReturnFalse;
   float error = 1.0f;
   if(!cRevIN.calcOutputGradients(SecondGradient, error))
      ReturnFalse;
   if(!cMarketObserver.CalcHiddenGradients(cRevIN.AsObject()))
      ReturnFalse;
   if(!cRLAgent.CalcHiddenGradients(cControlAgent.AsObject(), cMarketObserver.getOutput(),
                                    cMarketObserver.getPrevOutput(),
                                    (ENUM_ACTIVATION)cMarketObserver.Activation()) ||
      !SumAndNormalize(cMarketObserver.getGradient(), cMarketObserver.getPrevOutput(), cMarketObserver.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   CBufferFloat *temp = cRLAgent.getGradient();
   if(!cRLAgent.SetGradient(cRLAgent.getPrevOutput(), false) ||
      !cRLAgent.calcOutputGradients(cControlAgent.getOutput(), error) ||
      !SumAndNormalize(temp, cRLAgent.getPrevOutput(), temp, 1, false, 0, 0, 0, 1) ||
      !cRLAgent.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cMarketObserver.AsObject()))
      ReturnFalse;
   temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.calcOutputGradients(cRLAgent.getOutput(), error) ||
      !SumAndNormalize(temp, NeuronOCL.getPrevOutput(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMASA trainable weights from the accumulated   |
//| gradients.                                                       |
//+------------------------------------------------------------------+
bool CNeuronMASA::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cControlAgent.UpdateInputWeights(cRLAgent.AsObject(), cMarketObserver.getOutput()))
      ReturnFalse;
   if(!cRLAgent.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMarketObserver.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMASA GPU buffers to the given OpenCL context. |
//+------------------------------------------------------------------+
void CNeuronMASA::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseSAMOCL::SetOpenCL(obj);
   cMarketObserver.SetOpenCL(OpenCL);
   cRevIN.SetOpenCL(OpenCL);
   cRLAgent.SetOpenCL(OpenCL);
   cControlAgent.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMASA state to the file handle.                  |
//+------------------------------------------------------------------+
bool CNeuronMASA::Save(const int file_handle)
  {
   if(!CNeuronBaseSAMOCL::Save(file_handle))
      ReturnFalse;
   if(!cMarketObserver.Save(file_handle))
      ReturnFalse;
   if(!cRevIN.Save(file_handle))
      ReturnFalse;
   if(!cRLAgent.Save(file_handle))
      ReturnFalse;
   if(!cControlAgent.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMASA state from the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronMASA::Load(const int file_handle)
  {
   if(!CNeuronBaseSAMOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMarketObserver.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRevIN.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRLAgent.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cControlAgent.AsObject()))
      ReturnFalse;
//---
   if(!SetOutput(cControlAgent.getOutput(), true) ||
      !SetGradient(cControlAgent.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Sets the norm layer of the CNeuronMASA.                          |
//+------------------------------------------------------------------+
bool CNeuronMASA::SetNormLayer(int NormLayer, CNeuronBatchNormOCL * normLayer)
  {
   return cRevIN.Init(0, 1, OpenCL, cMarketObserver.Neurons(), NormLayer, normLayer);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPLRMultiAgentsOCL layer and allocates its |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronPLRMultiAgentsOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                    uint window_in, uint units_count, bool transpose,
                                    vector<float> &min_distance,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   iAgents = (int)min_distance.Size();
   if(iAgents <= 0)
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_in * units_count * iAgents, optimization_type, batch))
      ReturnFalse;
//---
   iVariables = (int)window_in;
   iCount = (int)units_count;
   bTranspose = transpose;
//---
   icIsTTP = OpenCL.AddBuffer(sizeof(int) * Neurons(), CL_MEM_READ_WRITE);
   if(icIsTTP < 0)
      ReturnFalse;
//---
   if(!cMinDistance.AssignArray(min_distance) ||
      !cMinDistance.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronPLRMultiAgentsOCL feed-forward pass and writes   |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronPLRMultiAgentsOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iCount, iVariables, iAgents};
   uint local_work_size[3] = {iCount, 1, 1};
   ResetLastError();
   const int kernel = def_k_PLRMultiAgents;
   setBuffer(kernel, def_k_plr_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_plr_outputs, getOutputIndex())
   setBuffer(kernel, def_k_plt_isttp, icIsTTP)
   setArgument(kernel, def_k_plr_transpose, (int)bTranspose)
   setBuffer(kernel, def_k_plr_step, cMinDistance.GetIndex())
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronPLRMultiAgentsOCL layer.                                  |
//+------------------------------------------------------------------+
bool CNeuronPLRMultiAgentsOCL::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!OpenCL || !prevLayer || !prevLayer.getGradient())
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {iCount, iVariables};
   ResetLastError();
   const int kernel = def_k_PLRMultiAgentsGrad;
   setBuffer(kernel, def_k_plrg_inputs_gr, prevLayer.getGradientIndex())
   setBuffer(kernel, def_k_plrg_outputs, getOutputIndex())
   setBuffer(kernel, def_k_plrg_outputs_gr, getGradientIndex())
   setArgument(kernel, def_k_plrg_transpose, (int)bTranspose)
   setArgument(kernel, def_k_plrg_agents, (int)iAgents)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!prevLayer.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronPLRMultiAgentsOCL state to the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronPLRMultiAgentsOCL::Save(const int file_handle)
  {
   if(!CNeuronPLROCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iAgents) < INT_VALUE)
      ReturnFalse;
   if(!cMinDistance.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronPLRMultiAgentsOCL state from the file handle.|
//+------------------------------------------------------------------+
bool CNeuronPLRMultiAgentsOCL::Load(const int file_handle)
  {
   if(!CNeuronPLROCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iAgents = FileReadInteger(file_handle);
   if(!cMinDistance.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronPLRMultiAgentsOCL GPU buffers to the given    |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronPLRMultiAgentsOCL::SetOpenCL(COpenCLMy * obj)
  {
//---
   CNeuronPLROCL::SetOpenCL(obj);
   cMinDistance.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronCrossSectionalAnalysis layer and allocates|
//| its GPU buffers.                                                 |
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
      uint window, uint window_key, uint heads, uint heads_kv,
      uint units_count, uint layers, uint layers_to_one_kv, uint variables,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMVMHAttentionMLKV::Init(numOutputs, myIndex, open_cl, window_key, window_key, heads, heads_kv,
                                      variables, layers, layers_to_one_kv, units_count, optimization_type, batch))
      ReturnFalse;
//---
   if(!cEmbeding.Init(0, 0, OpenCL, window, window, window_key, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cEmbeding.SetActivationFunction(GELU);
   if(!cTransposeRCD.Init(0, 1, OpenCL, variables, units_count, window_key, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronCrossSectionalAnalysis feed-forward pass and     |
//| writes the layer output.                                         |
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cEmbeding.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cTransposeRCD.FeedForward(cEmbeding.AsObject()))
      ReturnFalse;
//---
   return CNeuronMVMHAttentionMLKV::feedForward(cTransposeRCD.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronCrossSectionalAnalysis layer.                             |
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   if(!CNeuronMVMHAttentionMLKV::calcInputGradients(cTransposeRCD.AsObject()))
      ReturnFalse;
   if(!cEmbeding.CalcHiddenGradients(cTransposeRCD.AsObject()))
      ReturnFalse;
   if(cEmbeding.Activation() != None)
     {
      if(!DeActivation(cEmbeding.getOutput(), cEmbeding.getGradient(), cEmbeding.getGradient(), cEmbeding.Activation()))
         ReturnFalse;
     }
   if(!prevLayer.CalcHiddenGradients(cEmbeding.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronCrossSectionalAnalysis trainable weights from |
//| the accumulated gradients.                                       |
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronMVMHAttentionMLKV::updateInputWeights(cTransposeRCD.AsObject()))
      ReturnFalse;
   if(!cEmbeding.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronCrossSectionalAnalysis GPU buffers to the     |
//| given OpenCL context.                                            |
//+------------------------------------------------------------------+
void CNeuronCrossSectionalAnalysis::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMVMHAttentionMLKV::SetOpenCL(obj);
   cEmbeding.SetOpenCL(OpenCL);
   cTransposeRCD.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronCrossSectionalAnalysis state to the file handle.|
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::Save(const int file_handle)
  {
   if(!CNeuronMVMHAttentionMLKV::Save(file_handle))
      ReturnFalse;
   if(!cEmbeding.Save(file_handle))
      ReturnFalse;
   if(!cTransposeRCD.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronCrossSectionalAnalysis state from the file   |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronCrossSectionalAnalysis::Load(const int file_handle)
  {
   if(!CNeuronMVMHAttentionMLKV::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEmbeding.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeRCD.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronTemporalAnalysis layer and allocates its  |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                   uint window, uint window_key, uint heads, uint heads_kv,
                                   uint units_count, uint layers, uint layers_to_one_kv, uint variables,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronCrossSectionalAnalysis::Init(numOutputs, myIndex, open_cl, 3 * units_count, window_key, heads, heads_kv,
                                           window / 3, layers, layers_to_one_kv, variables, optimization_type, batch))
      ReturnFalse;
   if(!cTranspose.Init(0, 0, OpenCL, variables, units_count, window, optimization_type, batch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronTemporalAnalysis feed-forward pass and writes the|
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronCrossSectionalAnalysis::feedForward(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronTemporalAnalysis layer.                                   |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   if(!CNeuronCrossSectionalAnalysis::calcInputGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return prevLayer.CalcHiddenGradients(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronTemporalAnalysis trainable weights from the   |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return CNeuronCrossSectionalAnalysis::updateInputWeights(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronTemporalAnalysis GPU buffers to the given     |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronTemporalAnalysis::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronCrossSectionalAnalysis::SetOpenCL(obj);
   cTranspose.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronTemporalAnalysis state to the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::Save(const int file_handle)
  {
   if(!CNeuronCrossSectionalAnalysis::Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronTemporalAnalysis state from the file handle. |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::Load(const int file_handle)
  {
   if(!CNeuronCrossSectionalAnalysis::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPortfolioGenerator layer and allocates its|
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                     uint assets, uint time_points, uint dimension,
                                     uint agents, uint projection,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(assets <= 0 || time_points <= 0 || dimension <= 0 || agents <= 0)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, projection, optimization_type, batch))
      ReturnFalse;
//---
   iAssets = assets;
   iTimePoints = time_points;
   iDimension = dimension;
   iAgents = agents;
//---
   if(!cTransposeVRC.Init(0, 0, OpenCL, iAgents, iTimePoints, iDimension, optimization, iBatch))
      ReturnFalse;
   if(!cAssetTime[0].Init(0, 1, OpenCL, iAssets * iTimePoints * iAgents, optimization, iBatch))
      ReturnFalse;
   cAssetTime[0].SetActivationFunction(None);
   if(!cSoftMax.Init(0, 2, OpenCL, cAssetTime[0].Neurons(), optimization, iBatch))
      ReturnFalse;
   cSoftMax.SetHeads(iAssets * iAgents);
   if(!cAssetTime[1].Init(Neurons(), 3, OpenCL, iAssets * iDimension * iAgents, optimization, iBatch))
      ReturnFalse;
   cAssetTime[1].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronPortfolioGenerator feed-forward pass and writes  |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!SecondInput)
      ReturnFalse;
//---
   if(!cTransposeVRC.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!MatMul(SecondInput, cTransposeVRC.getOutput(), cAssetTime[0].getOutput(), iAssets, iDimension, iTimePoints, iAgents))
      ReturnFalse;
   if(!cSoftMax.FeedForward(cAssetTime[0].AsObject()))
      ReturnFalse;
   if(!MatMul(cSoftMax.getOutput(), NeuronOCL.getOutput(), cAssetTime[1].getOutput(), iAssets, iTimePoints, iDimension, iAgents))
      ReturnFalse;
//---
   return CNeuronBaseOCL::feedForward(cAssetTime[1].AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronPortfolioGenerator layer.                                 |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient || !SecondInput)
      ReturnFalse;
   if(!CNeuronBaseOCL::calcInputGradients(cAssetTime[1].AsObject()))
      ReturnFalse;
   if(!MatMulGrad(cSoftMax.getOutput(), cSoftMax.getGradient(),
                  NeuronOCL.getOutput(), cTransposeVRC.getPrevOutput(),
                  cAssetTime[1].getGradient(),
                  iAssets, iTimePoints, iDimension, iAgents))
      ReturnFalse;
   if(!cAssetTime[0].CalcHiddenGradients(cSoftMax.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(SecondInput, SecondGradient,
                  cTransposeVRC.getOutput(), cTransposeVRC.getGradient(),
                  cAssetTime[0].getGradient(),
                  iAssets, iDimension, iTimePoints, iAgents))
      ReturnFalse;
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTransposeVRC.AsObject()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cTransposeVRC.getPrevOutput(), NeuronOCL.getGradient(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), cTransposeVRC.getPrevOutput(), cTransposeVRC.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronPortfolioGenerator trainable weights from the |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return CNeuronBaseOCL::updateInputWeights(cAssetTime[1].AsObject());
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronPortfolioGenerator GPU buffers to the given   |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronPortfolioGenerator::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(uint i = 0; i < cAssetTime.Size(); i++)
      cAssetTime[i].SetOpenCL(OpenCL);
   cTransposeVRC.SetOpenCL(OpenCL);
   cSoftMax.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronPortfolioGenerator state to the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cAssetTime.Size(); i++)
      if(!cAssetTime[i].Save(file_handle))
         ReturnFalse;
   if(!cTransposeVRC.Save(file_handle))
      ReturnFalse;
   if(!cSoftMax.Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iAssets) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iTimePoints) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iAgents) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iDimension) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronPortfolioGenerator state from the file       |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cAssetTime.Size(); i++)
      if(!LoadInsideLayer(file_handle, cAssetTime[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeVRC.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSoftMax.AsObject()))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iAssets = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iTimePoints = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iAgents = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimension = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMASAAT layer and allocates its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                         uint window, uint window_key, uint heads, uint units_cout,
                         uint layers, vector<float> &min_distance, uint projection,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronPortfolioGenerator::Init(numOutputs, myIndex, open_cl, window, units_cout / 3,
                                       window_key, (uint)min_distance.Size() + 1, projection, optimization_type, batch))
      ReturnFalse;
   if(!cTranspose.Init(0, 0, OpenCL, units_cout, window, optimization, iBatch))
      ReturnFalse;
   if(!cPLR.Init(0, 1, OpenCL, window, units_cout, false, min_distance, optimization, iBatch))
      ReturnFalse;
   if(!cConcat.Init(0, 2, OpenCL, cTranspose.Neurons() + cPLR.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cCrossSectionalAnalysis.Init(0, 3, OpenCL, units_cout, window_key, heads, heads / 2, window, layers, 1, iAgents, optimization, iBatch))
      ReturnFalse;
   if(!cTemporalAnalysis.Init(0, 4, OpenCL, units_cout, window_key, heads, heads / 2, window, layers, 1, iAgents, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMASAAT feed-forward pass and writes the layer    |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cPLR.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!Concat(cTranspose.getOutput(), cPLR.getOutput(), cConcat.getOutput(), cTranspose.Neurons(), cPLR.Neurons(), 1))
      ReturnFalse;
   if(!cCrossSectionalAnalysis.FeedForward(cConcat.AsObject()))
      ReturnFalse;
   if(!cTemporalAnalysis.FeedForward(cConcat.AsObject()))
      ReturnFalse;
//---
   return CNeuronPortfolioGenerator::feedForward(cTemporalAnalysis.AsObject(), cCrossSectionalAnalysis.getOutput());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMASAAT    |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   if(!CNeuronPortfolioGenerator::calcInputGradients(cTemporalAnalysis.AsObject(),
         cCrossSectionalAnalysis.getOutput(),
         cCrossSectionalAnalysis.getGradient(),
         (ENUM_ACTIVATION)cCrossSectionalAnalysis.Activation()))
      ReturnFalse;
   if(!cConcat.CalcHiddenGradients(cCrossSectionalAnalysis.AsObject()))
      ReturnFalse;
   CBufferFloat *grad = cConcat.getGradient();
   if(!cConcat.SetGradient(cConcat.getPrevOutput(), false) ||
      !cConcat.CalcHiddenGradients(cTemporalAnalysis.AsObject()) ||
      !SumAndNormalize(grad, cConcat.getGradient(), grad, 1, 0, 0, 0, 0, 1) ||
      !cConcat.SetGradient(grad, false))
      ReturnFalse;
//---
   if(!DeConcat(cTranspose.getPrevOutput(), cPLR.getGradient(), cConcat.getGradient(), cTranspose.Neurons(), cPLR.Neurons(), 1))
      ReturnFalse;
   if(cPLR.Activation() != None)
      if(!DeActivation(cPLR.getOutput(), cPLR.getGradient(), cPLR.getGradient(), cPLR.Activation()))
         ReturnFalse;
//---
   if(!cTranspose.CalcHiddenGradients(cPLR.AsObject()) ||
      !SumAndNormalize(cTranspose.getGradient(), cTranspose.getPrevOutput(), cTranspose.getGradient(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(cTranspose.Activation() != None)
      if(!DeActivation(cTranspose.getOutput(), cTranspose.getGradient(), cTranspose.getGradient(), cTranspose.Activation()))
         ReturnFalse;
   if(!prevLayer.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMASAAT trainable weights from the accumulated |
//| gradients.                                                       |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cCrossSectionalAnalysis.UpdateInputWeights(cConcat.AsObject()))
      ReturnFalse;
   if(!cTemporalAnalysis.UpdateInputWeights(cConcat.AsObject()))
      ReturnFalse;
//---
   return CNeuronPortfolioGenerator::updateInputWeights(cTemporalAnalysis.AsObject());
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMASAAT GPU buffers to the given OpenCL        |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronMASAAT::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronPortfolioGenerator::SetOpenCL(obj);
   cTranspose.SetOpenCL(OpenCL);
   cPLR.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
   cCrossSectionalAnalysis.SetOpenCL(OpenCL);
   cTemporalAnalysis.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMASAAT state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::Save(const int file_handle)
  {
   if(!CNeuronPortfolioGenerator::Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cPLR.Save(file_handle))
      ReturnFalse;
   if(!cCrossSectionalAnalysis.Save(file_handle))
      ReturnFalse;
   if(!cTemporalAnalysis.Save(file_handle))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMASAAT state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronMASAAT::Load(const int file_handle)
  {
   if(!CNeuronPortfolioGenerator::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPLR.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossSectionalAnalysis.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTemporalAnalysis.AsObject()))
      ReturnFalse;
//---
   if(!cConcat.Init(0, 2, OpenCL, cTranspose.Neurons() + cPLR.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMemory layer and allocates its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMemory::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                         uint window, uint window_key, uint units_count, uint heads,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl, window, window_key,
                                           units_count, heads, window, units_count, optimization_type, batch))
      ReturnFalse;
   if(!cLSTM.Init(0, 0, OpenCL, iWindow, iUnits, optimization, iBatch))
      ReturnFalse;
   if(!cMamba.Init(0, 1, OpenCL, iWindow, 2 * iWindow, iUnits, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMemory feed-forward pass and writes the layer    |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMemory::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cLSTM.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMamba.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronRelativeCrossAttention::feedForward(cMamba.AsObject(), cLSTM.getOutput());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMemory    |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronMemory::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cMamba.AsObject(), cLSTM.getOutput(),
         cLSTM.getGradient(), (ENUM_ACTIVATION)cLSTM.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMamba.AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cMamba.getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cLSTM.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false) ||
      !SumAndNormalize(temp, cMamba.getPrevOutput(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMemory trainable weights from the accumulated |
//| gradients.                                                       |
//+------------------------------------------------------------------+
bool CNeuronMemory::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronRelativeCrossAttention::updateInputWeights(cMamba.AsObject(), cLSTM.getOutput()))
      ReturnFalse;
   if(!cLSTM.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMamba.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMemory GPU buffers to the given OpenCL        |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronMemory::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronRelativeCrossAttention::SetOpenCL(obj);
   cLSTM.SetOpenCL(OpenCL);
   cMamba.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMemory state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronMemory::Save(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cLSTM.Save(file_handle))
      ReturnFalse;
   if(!cMamba.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMemory state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronMemory::Load(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLSTM.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMamba.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMemory execution state and its GPU buffers.    |
//+------------------------------------------------------------------+
bool CNeuronMemory::Clear(void)
  {
   if(!CNeuronRelativeCrossAttention::Clear())
      ReturnFalse;
   if(!cLSTM.Clear())
      ReturnFalse;
   if(!cMamba.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronFinMem layer and allocates its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinMem::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                         uint window, uint window_key, uint units_count, uint heads,
                         uint account_descr, uint nactions,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl,
                                           nactions / 2, window_key, 2, heads, window,
                                           units_count, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cTransposeState.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMemory[0].Init(0, index, OpenCL, window, window_key, units_count, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMemory[1].Init(0, index, OpenCL, units_count, window_key, window, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cCrossMemory.Init(0, index, OpenCL, window, window_key, units_count, heads, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMemoryToAccount.Init(0, index, OpenCL, window, window_key, units_count, heads, account_descr, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionToAccount.Init(0, index, OpenCL, nactions / 2, window_key, 2, heads, account_descr, 1, optimization, iBatch))
      ReturnFalse;
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronFinMem feed-forward pass and writes the layer    |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronFinMem::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!cTransposeState.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMemory[0].FeedForward(NeuronOCL) ||
      !cMemory[1].FeedForward(cTransposeState.AsObject()))
      ReturnFalse;
   if(!cCrossMemory.FeedForward(cMemory[0].AsObject(), cMemory[1].getOutput()))
      ReturnFalse;
   if(!cMemoryToAccount.FeedForward(cCrossMemory.AsObject(), SecondInput))
      ReturnFalse;
   if(!cActionToAccount.FeedForward(this.AsObject(), SecondInput))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!CNeuronRelativeCrossAttention::feedForward(cActionToAccount.AsObject(), cMemoryToAccount.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronFinMem    |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronFinMem::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
                                       CBufferFloat* SecondInput,
                                       CBufferFloat* SecondGradient,
                                       ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
//---
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cActionToAccount.AsObject(),
         cMemoryToAccount.getOutput(),
         cMemoryToAccount.getGradient(),
         (ENUM_ACTIVATION)cMemoryToAccount.Activation()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   CBufferFloat *temp = Gradient;
   if(!SetGradient(cMemoryToAccount.getPrevOutput(), false))
      ReturnFalse;
   if(!calcHiddenGradients(cActionToAccount.AsObject(), SecondInput, SecondGradient, SecondActivation))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   Gradient = temp;
   if(!cCrossMemory.CalcHiddenGradients(cMemoryToAccount.AsObject(), SecondInput, cMemoryToAccount.getPrevOutput(), SecondActivation))
      ReturnFalse;
   if(!SumAndNormalize(SecondGradient, cMemoryToAccount.getPrevOutput(), SecondGradient, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cMemory[0].CalcHiddenGradients(cCrossMemory.AsObject(), cMemory[1].getOutput(), cMemory[1].getGradient(), (ENUM_ACTIVATION)cMemory[1].Activation()))
      ReturnFalse;
   if(!cTransposeState.CalcHiddenGradients(cMemory[1].AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMemory[0].AsObject()))
      ReturnFalse;
   temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cTransposeState.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cTransposeState.AsObject()) ||
      !NeuronOCL.SetGradient(temp, false) ||
      !SumAndNormalize(temp, cTransposeState.getPrevOutput(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronFinMem trainable weights from the accumulated |
//| gradients.                                                       |
//+------------------------------------------------------------------+
bool CNeuronFinMem::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
//---
   if(!CNeuronRelativeCrossAttention::updateInputWeights(cActionToAccount.AsObject(),
         cMemoryToAccount.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!cActionToAccount.UpdateInputWeights(this.AsObject(), SecondInput))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!cMemoryToAccount.UpdateInputWeights(cCrossMemory.AsObject(), SecondInput))
      ReturnFalse;
   if(!cCrossMemory.UpdateInputWeights(cMemory[0].AsObject(), cMemory[1].getOutput()))
      ReturnFalse;
   if(!cMemory[1].UpdateInputWeights(cTransposeState.AsObject()))
      ReturnFalse;
   if(!cMemory[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronFinMem execution state and its GPU buffers.    |
//+------------------------------------------------------------------+
bool CNeuronFinMem::Clear(void)
  {
   if(!Output ||
      !Output.Fill(0))
      ReturnFalse;
   for(uint i = 0; i < cMemory.Size(); i++)
      if(!cMemory[i].Clear())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronFinMem GPU buffers to the given OpenCL        |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronFinMem::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronRelativeCrossAttention::SetOpenCL(obj);
   cTransposeState.SetOpenCL(OpenCL);
   for(uint i = 0; i < cMemory.Size(); i++)
      cMemory[i].SetOpenCL(OpenCL);
   cCrossMemory.SetOpenCL(OpenCL);
   cMemoryToAccount.SetOpenCL(OpenCL);
   cActionToAccount.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFinMem state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronFinMem::Save(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cTransposeState.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cMemory.Size(); i++)
      if(!cMemory[i].Save(file_handle))
         ReturnFalse;
   if(!cCrossMemory.Save(file_handle))
      ReturnFalse;
   if(!cMemoryToAccount.Save(file_handle))
      ReturnFalse;
   if(!cActionToAccount.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFinMem state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronFinMem::Load(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeState.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cMemory.Size(); i++)
      if(!LoadInsideLayer(file_handle, cMemory[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossMemory.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMemoryToAccount.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionToAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronLowLevelReflection layer and allocates its|
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                     uint window, uint window_key, uint units_count, uint heads,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMemory::Init(numOutputs, myIndex, open_cl, window, window_key, units_count, heads,
                           optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cChangeLSTM.Init(0, index, OpenCL, window, units_count, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cChangeMamba.Init(0, index, OpenCL, window, 2 * window, units_count, optimization, iBatch))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      index++;
      if(!cCrossAttention[i].Init(0, index, OpenCL, window, window_key, units_count, heads, window, units_count, optimization, iBatch))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronLowLevelReflection feed-forward pass and writes  |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cChangeLSTM.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cChangeMamba.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cCrossAttention[0].FeedForward(NeuronOCL, cChangeLSTM.getOutput()))
      ReturnFalse;
   if(!cCrossAttention[1].FeedForward(cCrossAttention[0].AsObject(), cChangeMamba.getOutput()))
      ReturnFalse;
//---
   return CNeuronMemory::feedForward(cCrossAttention[1].AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronLowLevelReflection layer.                                 |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronMemory::calcInputGradients(cCrossAttention[1].AsObject()))
      ReturnFalse;
   if(!cCrossAttention[0].CalcHiddenGradients(cCrossAttention[1].AsObject(), cChangeMamba.getOutput(),
         cChangeMamba.getGradient(), (ENUM_ACTIVATION)cChangeMamba.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cCrossAttention[0].AsObject(), cChangeLSTM.getOutput(),
                                     cChangeLSTM.getGradient(), (ENUM_ACTIVATION)cChangeLSTM.Activation()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cChangeMamba.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cChangeMamba.AsObject()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), temp, temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cChangeLSTM.AsObject()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), temp, temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronLowLevelReflection trainable weights from the |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronMemory::updateInputWeights(cCrossAttention[1].AsObject()))
      ReturnFalse;
   if(!cCrossAttention[1].UpdateInputWeights(cCrossAttention[0].AsObject(), cChangeMamba.getOutput()))
      ReturnFalse;
   if(!cCrossAttention[0].UpdateInputWeights(NeuronOCL, cChangeLSTM.getOutput()))
      ReturnFalse;
   if(!cChangeMamba.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cChangeLSTM.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronLowLevelReflection state to the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::Save(const int file_handle)
  {
   if(!CNeuronMemory::Save(file_handle))
      ReturnFalse;
   if(!cChangeLSTM.Save(file_handle))
      ReturnFalse;
   if(!cChangeMamba.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cCrossAttention.Size(); i++)
      if(!cCrossAttention[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronLowLevelReflection state from the file       |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::Load(const int file_handle)
  {
   if(!CNeuronMemory::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cChangeLSTM.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cChangeMamba.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cCrossAttention.Size(); i++)
      if(!LoadInsideLayer(file_handle, cCrossAttention[i].AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronLowLevelReflection GPU buffers to the given   |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronLowLevelReflection::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMemory::SetOpenCL(obj);
   cChangeLSTM.SetOpenCL(OpenCL);
   cChangeMamba.SetOpenCL(OpenCL);
   for(uint i = 0; i < cCrossAttention.Size(); i++)
      cCrossAttention[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronLowLevelReflection execution state and its GPU |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronLowLevelReflection::Clear(void)
  {
   if(!CNeuronMemory::Clear())
      ReturnFalse;
   if(!cChangeLSTM.Clear())
      ReturnFalse;
   if(!cChangeMamba.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHighLevelReflection layer and allocates   |
//| its GPU buffers.                                                 |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                      uint window, uint window_key, uint units_count,
                                      uint heads, uint desc_account, uint actions_state,
                                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMemory::Init(numOutputs, myIndex, open_cl, 3, window_key, actions_state / 3,
                           heads, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cAccount.Init(0, index, OpenCL, desc_account, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHistoryAccount.Init(0, index, OpenCL, desc_account, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionReason.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads,
                          window, units_count, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHistoryActions.Init(0, index, OpenCL, iWindow, iUnits, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionResult.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads,
                          desc_account, 1, optimization, iBatch))
      ReturnFalse;;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHighLevelReflection feed-forward pass and writes |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
//---
   if(cAccount.getOutput() != SecondInput)
     {
      if(cAccount.Neurons() != SecondInput.Total())
         if(!cAccount.Init(0, 0, OpenCL, SecondInput.Total(), optimization, iBatch))
            ReturnFalse;
      if(!cAccount.SetOutput(SecondInput, true))
         ReturnFalse;
     }
//---
   if(!cHistoryAccount.FeedForward(cAccount.AsObject()))
      ReturnFalse;
   if(!cActionReason.FeedForward(this.AsObject(), NeuronOCL.getOutput()))
      ReturnFalse;
   if(!cHistoryActions.FeedForward(cActionReason.AsObject()))
      ReturnFalse;
   if(!cActionResult.FeedForward(cHistoryActions.AsObject(), cHistoryAccount.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//---
   return CNeuronMemory::feedForward(cActionResult.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHighLevelReflection layer.                                |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput,
      CBufferFloat* SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
//---
   if(cAccount.getGradient() != SecondGradient)
      if(!cAccount.SetGradient(SecondGradient, true))
         ReturnFalse;
   cAccount.SetActivationFunction(SecondActivation);
//---
   if(!CNeuronMemory::calcInputGradients(cActionResult.AsObject()))
      ReturnFalse;
   if(!cHistoryActions.CalcHiddenGradients(cActionResult.AsObject(),
                                           cHistoryAccount.getOutput(),
                                           cHistoryAccount.getGradient(),
                                           (ENUM_ACTIVATION)cHistoryAccount.Activation()))
      ReturnFalse;
   if(!cActionReason.CalcHiddenGradients(cHistoryActions.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   CBufferFloat *temp = Gradient;
   Gradient = cActionReason.getPrevOutput();
   if(!calcHiddenGradients(cActionReason.AsObject(), NeuronOCL.getOutput(),
                           NeuronOCL.getGradient(), (ENUM_ACTIVATION)NeuronOCL.Activation()))
      ReturnFalse;
//---
   Gradient = temp;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//---
   if(!cAccount.CalcHiddenGradients(cHistoryAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHighLevelReflection trainable weights from the|
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!CNeuronMemory::updateInputWeights(cActionResult.AsObject()))
      ReturnFalse;
   if(!cActionResult.UpdateInputWeights(cHistoryActions.AsObject(), cHistoryAccount.getOutput()))
      ReturnFalse;
   if(!cHistoryActions.UpdateInputWeights(cActionReason.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!cActionReason.UpdateInputWeights(this.AsObject(), NeuronOCL.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!cHistoryAccount.UpdateInputWeights(cAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHighLevelReflection state to the file handle.   |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::Save(const int file_handle)
  {
   if(!CNeuronMemory::Save(file_handle))
      ReturnFalse;
   if(!cAccount.Save(file_handle))
      ReturnFalse;
   if(!cHistoryAccount.Save(file_handle))
      ReturnFalse;
   if(!cActionReason.Save(file_handle))
      ReturnFalse;
   if(!cHistoryActions.Save(file_handle))
      ReturnFalse;
   if(!cActionResult.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHighLevelReflection state from the file      |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::Load(const int file_handle)
  {
   if(!CNeuronMemory::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAccount.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHistoryAccount.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionReason.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHistoryActions.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionResult.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronHighLevelReflection GPU buffers to the given  |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronHighLevelReflection::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMemory::SetOpenCL(obj);
   cAccount.SetOpenCL(OpenCL);
   cHistoryAccount.SetOpenCL(OpenCL);
   cActionReason.SetOpenCL(OpenCL);
   cHistoryActions.SetOpenCL(OpenCL);
   cActionResult.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronHighLevelReflection execution state and its GPU|
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronHighLevelReflection::Clear(void)
  {
   if(!CNeuronMemory::Clear())
      ReturnFalse;
   if(!cAccount.Clear())
      ReturnFalse;
   if(!cHistoryAccount.Clear())
      ReturnFalse;
   if(!cActionReason.Clear())
      ReturnFalse;
   if(!cHistoryActions.Clear())
      ReturnFalse;
   if(!cActionResult.Clear())
      ReturnFalse;;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMoreLessEqual feed-forward pass and writes the   |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronMoreLessEqual::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
   uint global_work_offset[1] = { 0 };
   uint global_work_size[1] = { Neurons() };
   ResetLastError();
   const int kernel = def_k_MoreLessEqual;
   setBuffer(kernel, def_k_mle_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_mle_outputs, getOutputIndex())
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMoreLessEqual layer.                                      |
//+------------------------------------------------------------------+
bool CNeuronMoreLessEqual::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getGradient())
      ReturnFalse;
   return NeuronOCL.getGradient().Fill(0);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronFinAgent layer and allocates its GPU      |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                           uint window, uint window_key, uint units_count, uint heads,
                           uint account_descr, uint nactions, uint segments,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl, 3,
                                           window_key, nactions / 3, heads,
                                           window, units_count,
                                           optimization_type, batch))
      ReturnFalse;
   int index = 0;
   if(!cMarketIntelligence.Init(0, index, OpenCL, window, units_count, segments, 0.2f, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMarketMemory.Init(0, index, OpenCL, window, window_key, units_count, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTransposeState.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowLevelReflection[0].Init(0, index, OpenCL, window, window_key, units_count, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowLevelReflection[1].Init(0, index, OpenCL, units_count, window_key, window, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHighLevelReflection.Init(0, index, OpenCL, window, window_key, units_count, heads, account_descr, nactions, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTools.Init(0, index, OpenCL, window * units_count, optimization, iBatch))
      ReturnFalse;
   cTools.SetActivationFunction(None);
   index++;
   if(!cCrossLowLevel.Init(0, index, OpenCL, window, window_key, units_count, heads, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMarketToLowLevel.Init(0, index, OpenCL, window, window_key, units_count, heads, window, units_count, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMarketToTools.Init(0, index, OpenCL, window, window_key, units_count, heads, window, units_count, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronFinAgent feed-forward pass and writes the layer  |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!cMarketIntelligence.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMarketMemory.FeedForward(cMarketIntelligence.AsObject()))
      ReturnFalse;
   if(!cTransposeState.FeedForward(cMarketIntelligence.AsObject()))
      ReturnFalse;
   if(!cLowLevelReflection[0].FeedForward(cMarketIntelligence.AsObject()))
      ReturnFalse;
   if(!cLowLevelReflection[1].FeedForward(cTransposeState.AsObject()))
      ReturnFalse;
   if(!cHighLevelReflection.FeedForward(cMarketIntelligence.AsObject(), SecondInput))
      ReturnFalse;
   if(!cTools.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cCrossLowLevel.FeedForward(cLowLevelReflection[0].AsObject(), cLowLevelReflection[1].getOutput()))
      ReturnFalse;
   if(!cMarketToLowLevel.FeedForward(cMarketMemory.AsObject(), cCrossLowLevel.getOutput()))
      ReturnFalse;
   if(!cMarketToTools.FeedForward(cMarketToLowLevel.AsObject(), cTools.getOutput()))
      ReturnFalse;
//---
   return CNeuronRelativeCrossAttention::feedForward(cHighLevelReflection.AsObject(), cMarketToTools.getOutput());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronFinAgent  |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput,
      CBufferFloat* SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
//---
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cHighLevelReflection.AsObject(),
         cMarketToTools.getOutput(),
         cMarketToTools.getGradient(),
         (ENUM_ACTIVATION)cMarketToTools.Activation()))
      ReturnFalse;
//---
   if(!cMarketToLowLevel.CalcHiddenGradients(cMarketToTools.AsObject(),
         cTools.getOutput(),
         cTools.getGradient(),
         (ENUM_ACTIVATION)cTools.Activation()))
      ReturnFalse;
//---
   if(!cMarketMemory.CalcHiddenGradients(cMarketToLowLevel.AsObject(),
                                         cCrossLowLevel.getOutput(),
                                         cCrossLowLevel.getGradient(),
                                         (ENUM_ACTIVATION)cCrossLowLevel.Activation()))
      ReturnFalse;
//---
   if(!cLowLevelReflection[0].CalcHiddenGradients(cCrossLowLevel.AsObject(),
         cLowLevelReflection[1].getOutput(),
         cLowLevelReflection[1].getGradient(),
         (ENUM_ACTIVATION)cLowLevelReflection[1].Activation()))
      ReturnFalse;
   if(!cTransposeState.CalcHiddenGradients(cLowLevelReflection[1].AsObject()))
      ReturnFalse;
//---
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).CalcHiddenGradients(cMarketMemory.AsObject()))
      ReturnFalse;
   CBufferFloat *temp = cMarketIntelligence.getGradient();
   if(!cMarketIntelligence.SetGradient(cMarketIntelligence.getPrevOutput(), false) ||
      !((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).CalcHiddenGradients(cHighLevelReflection.AsObject(),
            SecondInput, SecondGradient, SecondActivation) ||
      !SumAndNormalize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).CalcHiddenGradients(cLowLevelReflection[0].AsObject()) ||
      !SumAndNormalize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).CalcHiddenGradients(cTransposeState.AsObject()) ||
      !SumAndNormalize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !cMarketIntelligence.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cMarketIntelligence.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronFinAgent trainable weights from the           |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
//---
   if(!CNeuronRelativeCrossAttention::updateInputWeights(cHighLevelReflection.AsObject(),
         cMarketToTools.getOutput()))
      ReturnFalse;
//---
   if(!cMarketToTools.UpdateInputWeights(cMarketToLowLevel.AsObject(),
                                         cTools.getOutput()))
      ReturnFalse;
//---
   if(!cMarketToLowLevel.UpdateInputWeights(cMarketMemory.AsObject(),
         cCrossLowLevel.getOutput()))
      ReturnFalse;
   if(!cMarketMemory.UpdateInputWeights(cMarketIntelligence.AsObject()))
      ReturnFalse;
//---
   if(!cCrossLowLevel.UpdateInputWeights(cLowLevelReflection[0].AsObject(),
                                         cLowLevelReflection[1].getOutput()))
      ReturnFalse;
//---
   if(!cLowLevelReflection[0].UpdateInputWeights(cMarketIntelligence.AsObject()))
      ReturnFalse;
   if(!cLowLevelReflection[1].UpdateInputWeights(cTransposeState.AsObject()))
      ReturnFalse;
//---
   if(!cHighLevelReflection.UpdateInputWeights(cMarketIntelligence.AsObject(), SecondInput))
      ReturnFalse;
//---
   if(!cMarketIntelligence.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFinAgent state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::Save(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cMarketIntelligence.Save(file_handle))
      ReturnFalse;
   if(!cMarketMemory.Save(file_handle))
      ReturnFalse;
   if(!cTransposeState.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cLowLevelReflection.Size(); i++)
      if(!cLowLevelReflection[i].Save(file_handle))
         ReturnFalse;
   if(!cHighLevelReflection.Save(file_handle))
      ReturnFalse;
   if(!cTools.Save(file_handle))
      ReturnFalse;
   if(!cCrossLowLevel.Save(file_handle))
      ReturnFalse;
   if(!cMarketToLowLevel.Save(file_handle))
      ReturnFalse;
   if(!cMarketToTools.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFinAgent state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::Load(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMarketIntelligence.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMarketMemory.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeState.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cLowLevelReflection.Size(); i++)
      if(!LoadInsideLayer(file_handle, cLowLevelReflection[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHighLevelReflection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTools.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossLowLevel.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMarketToLowLevel.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMarketToTools.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronFinAgent GPU buffers to the given OpenCL      |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronFinAgent::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronRelativeCrossAttention::SetOpenCL(obj);
   cMarketIntelligence.SetOpenCL(OpenCL);
   cMarketMemory.SetOpenCL(OpenCL);
   cTransposeState.SetOpenCL(OpenCL);
   for(uint i = 0; i < cLowLevelReflection.Size(); i++)
      cLowLevelReflection[i].SetOpenCL(OpenCL);
   cHighLevelReflection.SetOpenCL(OpenCL);
   cTools.SetOpenCL(OpenCL);
   cCrossLowLevel.SetOpenCL(OpenCL);
   cMarketToLowLevel.SetOpenCL(OpenCL);
   cMarketToTools.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronFinAgent execution state and its GPU buffers.  |
//+------------------------------------------------------------------+
bool CNeuronFinAgent::Clear(void)
  {
   if(!CNeuronRelativeCrossAttention::Clear())
      ReturnFalse;
   if(!cMarketIntelligence.Clear())
      ReturnFalse;
   if(!cMarketMemory.Clear())
      ReturnFalse;
   if(!cTransposeState.Clear())
      ReturnFalse;
   for(uint i = 0; i < cLowLevelReflection.Size(); i++)
      if(!cLowLevelReflection[i].Clear())
         ReturnFalse;
   if(!cHighLevelReflection.Clear())
      ReturnFalse;
   if(!cTools.Clear())
      ReturnFalse;
   if(!cCrossLowLevel.Clear())
      ReturnFalse;
   if(!cMarketToLowLevel.Clear())
      ReturnFalse;
   if(!cMarketToTools.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMemoryDistil layer and allocates its GPU  |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                               uint window, uint window_key, uint units_count,
                               uint heads, uint stack_size,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl, window, window_key,
                                           units_count, heads, window, 2 * stack_size, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cLSTM.Init(0, index, OpenCL, iWindow, iUnits, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMamba.Init(0, index, OpenCL, iWindow, 2 * iWindow, iUnits, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, 2 * iWindow * iUnits, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
   index++;
   if(!cTransposeConc.Init(0, index, OpenCL, iUnits, 2 * iWindow, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConvolution.Init(0, index, OpenCL, iUnits, iUnits, iWindowKey, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cConvolution.SetActivationFunction(GELU);
   index++;
   uint windows[] = {iWindowKey * iWindow, iWindowKey * iWindow};
   if(!cDistilMemory.Init(0, index, OpenCL, iUnitsKV / 2, iWindow, windows))
      ReturnFalse;
//---
   index++;
   if(!cCrossAttention.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads, iWindow, 2 * iUnits, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMemoryDistil feed-forward pass and writes the    |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cLSTM.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMamba.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!Concat(cLSTM.getOutput(), cMamba.getOutput(), cConcatenated.getOutput(), iWindow, iWindow, iUnits))
      ReturnFalse;
   if(!cTransposeConc.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   if(!cConvolution.FeedForward(cTransposeConc.AsObject()))
      ReturnFalse;
   if(!cDistilMemory.FeedForward(cConvolution.AsObject()))
      ReturnFalse;
//---
   if(!cCrossAttention.FeedForward(NeuronOCL, cConcatenated.getOutput()))
      ReturnFalse;
//---
   return CNeuronRelativeCrossAttention::feedForward(cCrossAttention.AsObject(), cDistilMemory.getOutput());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMemoryDistil layer.                                       |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cCrossAttention.AsObject(),
         cDistilMemory.getOutput(),
         cDistilMemory.getGradient(),
         (ENUM_ACTIVATION)cDistilMemory.Activation()))
      ReturnFalse;
//---
   if(!cConvolution.CalcHiddenGradients(cDistilMemory.AsObject()))
      ReturnFalse;
   if(!cTransposeConc.CalcHiddenGradients(cConvolution.AsObject()))
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cTransposeConc.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cCrossAttention.AsObject(),
                                     cConcatenated.getOutput(),
                                     cConcatenated.getPrevOutput(),
                                     (ENUM_ACTIVATION)cConcatenated.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(cConcatenated.getGradient(), cConcatenated.getPrevOutput(),
                       cConcatenated.getGradient(), iWindow, false, 0, 0, 0, 1) ||
      !DeConcat(cLSTM.getGradient(), cMamba.getGradient(), cConcatenated.getGradient(),
                iWindow, iWindow, iUnits))
      ReturnFalse;
//---
   if(cLSTM.Activation() != None)
      if(!DeActivation(cLSTM.getOutput(), cLSTM.getGradient(), cLSTM.getGradient(), cLSTM.Activation()))
         ReturnFalse;
   if(cMamba.Activation() != None)
      if(!DeActivation(cMamba.getOutput(), cMamba.getGradient(), cMamba.getGradient(), cMamba.Activation()))
         ReturnFalse;
//---
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cConcatenated.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cLSTM.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMamba.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iWindow, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMemoryDistil trainable weights from the       |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronRelativeCrossAttention::updateInputWeights(cCrossAttention.AsObject(), cDistilMemory.getOutput()))
      ReturnFalse;
   if(!cCrossAttention.UpdateInputWeights(NeuronOCL, cConcatenated.getOutput()))
      ReturnFalse;
//---
   if(!cDistilMemory.UpdateInputWeights(cConvolution.AsObject()))
      ReturnFalse;
   if(!cConvolution.UpdateInputWeights(cTransposeConc.AsObject()))
      ReturnFalse;
//---
   if(!cLSTM.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMamba.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMemoryDistil GPU buffers to the given OpenCL  |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronMemoryDistil::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMemory::SetOpenCL(obj);
   cConcatenated.SetOpenCL(OpenCL);
   cTransposeConc.SetOpenCL(OpenCL);
   cConvolution.SetOpenCL(OpenCL);
   cDistilMemory.SetOpenCL(OpenCL);
   cCrossAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMemoryDistil execution state and its GPU       |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::Clear(void)
  {
   if(!CNeuronMemory::Clear())
      ReturnFalse;
   if(!cConcatenated.Clear())
      ReturnFalse;
   if(!cTransposeConc.Clear())
      ReturnFalse;
   if(!cConvolution.Clear())
      ReturnFalse;
   if(!cDistilMemory.Clear())
      ReturnFalse;
   if(!cCrossAttention.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMemoryDistil state to the file handle.          |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::Save(const int file_handle)
  {
   if(!CNeuronMemory::Save(file_handle))
      ReturnFalse;
   if(!cConcatenated.Save(file_handle))
      ReturnFalse;
   if(!cTransposeConc.Save(file_handle))
      ReturnFalse;
   if(!cConvolution.Save(file_handle))
      ReturnFalse;
   if(!cDistilMemory.Save(file_handle))
      ReturnFalse;
   if(!cCrossAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMemoryDistil state from the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronMemoryDistil::Load(const int file_handle)
  {
   if(!CNeuronMemory::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenated.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeConc.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConvolution.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDistilMemory.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronFinConAgent layer and allocates its GPU   |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                              uint window, uint window_key, uint units_count,
                              uint heads, uint stack_size, uint action_space,
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl,
                                           3, window_key, action_space / 3,
                                           heads, window, units_count,
                                           optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cStatesMemory.Init(0, index, OpenCL, window, iWindowKey, iUnitsKV,
                          iHeads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionsMemory.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits,
                           iHeads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!caRole[0].Init(10 * iWindow, index, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   CBufferFloat *out = caRole[0].getOutput();
   if(!out ||
      !out.Fill(1))
      ReturnFalse;
   index++;
   if(!caRole[1].Init(0, index, OpenCL, 10 * iWindow, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cStateToRole.Init(0, index, OpenCL, window, iWindowKey, iUnitsKV,
                         iHeads, iWindow, 10, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronFinConAgent feed-forward pass and writes the     |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(bTrain && !caRole[1].FeedForward(caRole[0].AsObject()))
      ReturnFalse;
//---
   if(!cStateToRole.FeedForward(NeuronOCL, caRole[1].getOutput()))
      ReturnFalse;
   if(!cStatesMemory.FeedForward(cStateToRole.AsObject()))
      ReturnFalse;
//---
   if(!cActionsMemory.FeedForward(this.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//---
   return CNeuronRelativeCrossAttention::feedForward(cActionsMemory.AsObject(), cStatesMemory.getOutput());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronFinConAgent layer.                                        |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cActionsMemory.AsObject(),
         cStatesMemory.getOutput(),
         cStatesMemory.getGradient(),
         (ENUM_ACTIVATION)cStatesMemory.Activation()))
      ReturnFalse;
//---
   CBufferFloat *temp = Gradient;
   if(!SwapBuffers(Output, PrevOutput) ||
      !SetGradient(cActionsMemory.getPrevOutput(), false))
      ReturnFalse;
   if(!calcHiddenGradients(cActionsMemory.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   Gradient = temp;
//---
   if(!cStateToRole.CalcHiddenGradients(cStatesMemory.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cStateToRole.AsObject(),
                                     caRole[1].getOutput(),
                                     caRole[1].getGradient(),
                                     (ENUM_ACTIVATION)caRole[1].Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronFinConAgent trainable weights from the        |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronRelativeCrossAttention::updateInputWeights(cActionsMemory.AsObject(),
         cStatesMemory.getOutput()))
      ReturnFalse;
//---
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   if(!cActionsMemory.UpdateInputWeights(this.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//---
   if(!cStatesMemory.UpdateInputWeights(cStateToRole.AsObject()))
      ReturnFalse;
   if(!cStateToRole.UpdateInputWeights(NeuronOCL, caRole[1].getOutput()))
      ReturnFalse;
//---
   if(!caRole[1].UpdateInputWeights(caRole[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronFinConAgent GPU buffers to the given OpenCL   |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronFinConAgent::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronRelativeCrossAttention::SetOpenCL(obj);
   cStatesMemory.SetOpenCL(OpenCL);
   cActionsMemory.SetOpenCL(OpenCL);
   caRole[0].SetOpenCL(OpenCL);
   caRole[1].SetOpenCL(OpenCL);
   cStateToRole.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronFinConAgent execution state and its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::Clear(void)
  {
   if(!CNeuronRelativeCrossAttention::Clear())
      ReturnFalse;
   if(!cStatesMemory.Clear())
      ReturnFalse;
   if(!cActionsMemory.Clear())
      ReturnFalse;
   if(!caRole[0].Clear())
      ReturnFalse;
   if(!caRole[1].Clear())
      ReturnFalse;
   if(!cStateToRole.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFinConAgent state to the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::Save(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cStatesMemory.Save(file_handle))
      ReturnFalse;
   if(!cActionsMemory.Save(file_handle))
      ReturnFalse;
   if(!caRole[0].Save(file_handle))
      ReturnFalse;
   if(!caRole[1].Save(file_handle))
      ReturnFalse;
   if(!cStateToRole.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFinConAgent state from the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronFinConAgent::Load(const int file_handle)
  {
   if(!CNeuronRelativeCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStatesMemory.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionsMemory.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, caRole[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, caRole[1].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStateToRole.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronFinConManager layer and allocates its GPU |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                uint window, uint window_key, uint units_count, uint heads,
                                uint stack_size, uint account_descr, uint action_space,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronFinConAgent::Init(numOutputs, myIndex, open_cl, action_space, window_key,
                                caAgents.Size() + caTrAgents.Size() + 1,
                                heads, stack_size, action_space, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cTransposeState.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
//---
   for(uint i = 0; i < caAgents.Size(); i++)
     {
      index++;
      if(!caAgents[i].Init(0, index, OpenCL, window, iWindowKey, units_count, iHeads, stack_size, action_space, optimization, iBatch))
         ReturnFalse;
     }
   for(uint i = 0; i < caTrAgents.Size(); i++)
     {
      index++;
      if(!caTrAgents[i].Init(0, index, OpenCL, units_count, iWindowKey, window, iHeads, stack_size, action_space, optimization, iBatch))
         ReturnFalse;
     }
   index++;
   if(!cRiskAgent.Init(0, index, OpenCL, account_descr, iWindowKey, 1, iHeads, stack_size, action_space, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcatenatedAgents.Init(0, index, OpenCL, caAgents.Size()*caAgents[0].Neurons() +
                                caTrAgents.Size()*caTrAgents[0].Neurons() +
                                cRiskAgent.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cAccount.Init(0, index, OpenCL, account_descr, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronFinConManager feed-forward pass and writes the   |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(cAccount.getOutput() != SecondInput)
     {
      if(!cAccount.SetOutput(SecondInput, true))
         ReturnFalse;
     }
   if(!cTransposeState.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Agents
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].FeedForward(NeuronOCL))
         ReturnFalse;
   for(uint i = 0; i < caTrAgents.Size(); i++)
      if(!caTrAgents[i].FeedForward(cTransposeState.AsObject()))
         ReturnFalse;
   if(!cRiskAgent.FeedForward(cAccount.AsObject()))
      ReturnFalse;
//--- Concatenate
   if(!Concat(caAgents[0].getOutput(), caAgents[1].getOutput(), caAgents[2].getOutput(), cRiskAgent.getOutput(),
              cConcatenatedAgents.getPrevOutput(), Neurons(), Neurons(), Neurons(), Neurons(), 1) ||
      !Concat(caTrAgents[0].getOutput(), caTrAgents[1].getOutput(), caTrAgents[2].getOutput(), cConcatenatedAgents.getPrevOutput(),
              cConcatenatedAgents.getOutput(), Neurons(), Neurons(), Neurons(), 4 * Neurons(), 1))
      ReturnFalse;
//--- Manager
   return CNeuronFinConAgent::feedForward(cConcatenatedAgents.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronFinConManager layer.                                      |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput,
      CBufferFloat* SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
   if(cAccount.getGradient() != SecondGradient)
     {
      if(!cAccount.SetGradient(SecondGradient, true))
         ReturnFalse;
      cAccount.SetActivationFunction(SecondActivation);
     }
//---
   if(!CNeuronFinConAgent::calcInputGradients(cConcatenatedAgents.AsObject()))
      ReturnFalse;
//---
   if(!DeConcat(caTrAgents[0].getGradient(), caTrAgents[1].getGradient(), caTrAgents[2].getGradient(), cConcatenatedAgents.getPrevOutput(),
                cConcatenatedAgents.getGradient(), Neurons(), Neurons(), Neurons(), 4 * Neurons(), 1) ||
      !DeConcat(caAgents[0].getGradient(), caAgents[1].getGradient(), caAgents[2].getGradient(), cRiskAgent.getGradient(),
                cConcatenatedAgents.getPrevOutput(), Neurons(), Neurons(), Neurons(), Neurons(), 1))
      ReturnFalse;
//---
   if(!cAccount.CalcHiddenGradients(cRiskAgent.AsObject()))
      ReturnFalse;
   if(!cTransposeState.CalcHiddenGradients(caTrAgents[0].AsObject()))
      ReturnFalse;
   CBufferFloat *temp = cTransposeState.getGradient();
   if(!cTransposeState.SetGradient(cTransposeState.getPrevOutput(), false))
      ReturnFalse;
   for(uint i = 1; i < caTrAgents.Size(); i++)
     {
      if(!cTransposeState.CalcHiddenGradients(caTrAgents[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(cTransposeState.getGradient(), temp, temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   if(!cTransposeState.SetGradient(temp, false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTransposeState.AsObject()))
      ReturnFalse;
   temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cTransposeState.getPrevOutput(), false))
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
     {
      if(!NeuronOCL.CalcHiddenGradients(caAgents[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(NeuronOCL.getGradient(), temp, temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronFinConManager trainable weights from the      |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::updateInputWeights(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput)
  {
//--- Manager
   if(!CNeuronFinConAgent::updateInputWeights(cConcatenatedAgents.AsObject()))
      ReturnFalse;
//--- Agents
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
   for(uint i = 0; i < caTrAgents.Size(); i++)
      if(!caTrAgents[i].UpdateInputWeights(cTransposeState.AsObject()))
         ReturnFalse;
   if(!cRiskAgent.UpdateInputWeights(cAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronFinConManager GPU buffers to the given OpenCL |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronFinConManager::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronFinConAgent::SetOpenCL(obj);
   cTransposeState.SetOpenCL(OpenCL);
   for(uint i = 0; i < caAgents.Size(); i++)
      caAgents[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < caTrAgents.Size(); i++)
      caTrAgents[i].SetOpenCL(OpenCL);
   cRiskAgent.SetOpenCL(OpenCL);
   cConcatenatedAgents.SetOpenCL(OpenCL);
   cAccount.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronFinConManager layer.        |
//+------------------------------------------------------------------+
void CNeuronFinConManager::TrainMode(bool flag)
  {
   CNeuronFinConAgent::TrainMode(flag);
   cTransposeState.TrainMode(flag);
   for(uint i = 0; i < caAgents.Size(); i++)
      caAgents[i].TrainMode(flag);
   for(uint i = 0; i < caTrAgents.Size(); i++)
      caTrAgents[i].TrainMode(flag);
   cRiskAgent.TrainMode(flag);
   cConcatenatedAgents.TrainMode(flag);
   cAccount.TrainMode(flag);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFinConManager state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::Save(const int file_handle)
  {
   if(!CNeuronFinConAgent::Save(file_handle))
      ReturnFalse;
   if(!cTransposeState.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caTrAgents.Size(); i++)
      if(!caTrAgents[i].Save(file_handle))
         ReturnFalse;
   if(!cRiskAgent.Save(file_handle))
      ReturnFalse;
   if(!cConcatenatedAgents.Save(file_handle))
      ReturnFalse;
   if(!cAccount.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFinConManager state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::Load(const int file_handle)
  {
   if(!CNeuronFinConAgent::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeState.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!LoadInsideLayer(file_handle, caAgents[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caTrAgents.Size(); i++)
      if(!LoadInsideLayer(file_handle, caTrAgents[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRiskAgent.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenatedAgents.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronFinConManager execution state and its GPU      |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFinConManager::Clear(void)
  {
   if(!CNeuronFinConAgent::Clear())
      ReturnFalse;
   if(!cTransposeState.Clear())
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].Clear())
         ReturnFalse;
   for(uint i = 0; i < caTrAgents.Size(); i++)
      if(!caTrAgents[i].Clear())
         ReturnFalse;
   if(!cRiskAgent.Clear())
      ReturnFalse;
   if(!cConcatenatedAgents.Clear())
      ReturnFalse;
   if(!cAccount.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMacroHFTHyperAgent layer and allocates its|
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                     uint window, uint window_key, uint units_count,
                                     uint heads, uint layers, uint agents, uint stack_size,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSoftMaxOCL::Init(numOutputs, myIndex, open_cl, agents, optimization_type, batch))
      ReturnFalse;
   SetHeads(1);
//---
   int index = 0;
   if(!cMemory.Init(0, 0, OpenCL, window, window_key, units_count, heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cStatePrepare.Init(0, index, OpenCL, window, window_key, units_count, heads, layers,
                          optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cScale.Init(4 * agents, index, OpenCL, 3, 1, 1, units_count - 2, window, optimization, iBatch))
      ReturnFalse;
   cScale.SetActivationFunction(TANH);
   index++;
   if(!cMLP[0].Init(agents, index, OpenCL, 4 * agents, optimization, iBatch))
      ReturnFalse;
   cMLP[0].SetActivationFunction(LReLU);
   index++;
   if(!cMLP[1].Init(0, index, OpenCL, agents, optimization, iBatch))
      ReturnFalse;
   cMLP[0].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMacroHFTHyperAgent feed-forward pass and writes  |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cMemory.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cStatePrepare.FeedForward(cMemory.AsObject()))
      ReturnFalse;
   if(!cTranspose.FeedForward(cStatePrepare.AsObject()))
      ReturnFalse;
   if(!cScale.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cMLP[0].FeedForward(cScale.AsObject()))
      ReturnFalse;
   if(!cMLP[1].FeedForward(cMLP[0].AsObject()))
      ReturnFalse;
//---
   return CNeuronSoftMaxOCL::feedForward(cMLP[1].AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMacroHFTHyperAgent layer.                                 |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSoftMaxOCL::calcInputGradients(cMLP[1].AsObject()))
      ReturnFalse;
   if(!cMLP[0].CalcHiddenGradients(cMLP[1].AsObject()))
      ReturnFalse;
   if(!cScale.CalcHiddenGradients(cMLP[0].AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cScale.AsObject()))
      ReturnFalse;
   if(!cStatePrepare.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!cMemory.CalcHiddenGradients(cStatePrepare.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMemory.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMacroHFTHyperAgent trainable weights from the |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cMLP[1].UpdateInputWeights(cMLP[0].AsObject()))
      ReturnFalse;
   if(!cMLP[0].UpdateInputWeights(cScale.AsObject()))
      ReturnFalse;
   if(!cScale.UpdateInputWeights(cTranspose.AsObject()))
      ReturnFalse;
   if(!cStatePrepare.UpdateInputWeights(cMemory.AsObject()))
      ReturnFalse;
   if(!cMemory.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMacroHFTHyperAgent state to the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::Save(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Save(file_handle))
      ReturnFalse;
   if(!cStatePrepare.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cScale.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cMLP.Size(); i++)
      if(!cMLP[i].Save(file_handle))
         ReturnFalse;
   if(!cMemory.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMacroHFTHyperAgent state from the file       |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::Load(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStatePrepare.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScale.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cMLP.Size(); i++)
      if(!LoadInsideLayer(file_handle, cMLP[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMemory.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMacroHFTHyperAgent GPU buffers to the given   |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronMacroHFTHyperAgent::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronSoftMaxOCL::SetOpenCL(obj);
   cStatePrepare.SetOpenCL(obj);
   cTranspose.SetOpenCL(obj);
   cScale.SetOpenCL(obj);
   for(uint i = 0; i < cMLP.Size(); i++)
      cMLP[i].SetOpenCL(obj);
   cMemory.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMacroHFTHyperAgent execution state and its GPU |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTHyperAgent::Clear(void)
  {
   if(!CNeuronSoftMaxOCL::Clear())
      ReturnFalse;
   if(!cStatePrepare.Clear())
      ReturnFalse;
   if(!cTranspose.Clear())
      ReturnFalse;
   if(!cScale.Clear())
      ReturnFalse;
   for(uint i = 0; i < cMLP.Size(); i++)
      if(!cMLP[i].Clear())
         ReturnFalse;
   if(!cMemory.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMacroHFT layer and allocates its GPU      |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                           uint window, uint window_key, uint units_count,
                           uint heads, uint layers, uint stack_size, uint nactions,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, nactions, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cTranspose.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   uint half = (caAgents.Size() + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      index++;
      if(!caAgents[i].Init(0, index, OpenCL, window, window_key, units_count, heads,
                           stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   for(uint i = half; i < caAgents.Size(); i++)
     {
      index++;
      if(!caAgents[i].Init(0, index, OpenCL, units_count, window_key, window, heads,
                           stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   index++;
   if(!cHyperAgent.Init(0, index, OpenCL, window, window_key, units_count, heads, layers,
                        caAgents.Size(), stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, caAgents.Size()*nactions, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMacroHFT feed-forward pass and writes the layer  |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint total = caAgents.Size();
   uint half = (total + 1) / 2;
   for(uint i = 0; i < half; i++)
      if(!caAgents[i].FeedForward(NeuronOCL))
         ReturnFalse;
   for(uint i = half; i < total; i++)
      if(!caAgents[i].FeedForward(cTranspose.AsObject()))
         ReturnFalse;
   if(!cHyperAgent.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!Concat(caAgents[0].getOutput(), caAgents[1].getOutput(), caAgents[2].getOutput(),
              caAgents[3].getOutput(), cConcatenated.getPrevOutput(), Neurons(), Neurons(),
              Neurons(), Neurons(), 1) ||
      !Concat(cConcatenated.getPrevOutput(), caAgents[4].getOutput(), caAgents[5].getOutput(),
              cConcatenated.getOutput(), 4 * Neurons(), Neurons(), Neurons(), 1))
      ReturnFalse;
   if(!MatMul(cHyperAgent.getOutput(), cConcatenated.getOutput(), Output, 1, total, Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMacroHFT  |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint total = caAgents.Size();
   if(!MatMulGrad(cHyperAgent.getOutput(), cHyperAgent.getGradient(), cConcatenated.getOutput(), cConcatenated.getGradient(), Gradient, 1, total, Neurons(), 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cHyperAgent.AsObject()))
      ReturnFalse;
//---
   if(!DeConcat(cConcatenated.getPrevOutput(), caAgents[4].getGradient(), caAgents[5].getGradient(),
                cConcatenated.getGradient(), 4 * Neurons(), Neurons(), Neurons(), 1) ||
      !DeConcat(caAgents[0].getGradient(), caAgents[1].getGradient(), caAgents[2].getGradient(),
                caAgents[3].getGradient(), cConcatenated.getPrevOutput(), Neurons(), Neurons(),
                Neurons(), Neurons(), 1))
      ReturnFalse;
//---
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!temp ||
      !NeuronOCL.SetGradient(cTranspose.getPrevOutput(), false))
      ReturnFalse;
   uint half = (total + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      if(!NeuronOCL.CalcHiddenGradients(caAgents[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   for(uint i = half; i < total; i++)
     {
      if(!cTranspose.CalcHiddenGradients(caAgents[i].AsObject()) ||
         !NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMacroHFT trainable weights from the           |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cHyperAgent.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   uint total = caAgents.Size();
   uint half = (total + 1) / 2;
   for(uint i = 0; i < half; i++)
      if(!caAgents[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
   for(uint i = half; i < total; i++)
      if(!caAgents[i].UpdateInputWeights(cTranspose.AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMacroHFT state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].Save(file_handle))
         ReturnFalse;
   if(!cHyperAgent.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMacroHFT state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!LoadInsideLayer(file_handle, caAgents[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHyperAgent.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.Init(0, 8, OpenCL, caAgents.Size()*Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMacroHFT GPU buffers to the given OpenCL      |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronMacroHFT::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cTranspose.SetOpenCL(obj);
   for(uint i = 0; i < caAgents.Size(); i++)
      caAgents[i].SetOpenCL(obj);
   cHyperAgent.SetOpenCL(obj);
   cConcatenated.SetOpenCL(obj);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMacroHFT execution state and its GPU buffers.  |
//+------------------------------------------------------------------+
bool CNeuronMacroHFT::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cTranspose.Clear())
      ReturnFalse;
   for(uint i = 0; i < caAgents.Size(); i++)
      if(!caAgents[i].Clear())
         ReturnFalse;
   if(!cHyperAgent.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMacroHFTvsRiskManager layer and allocates |
//| its GPU buffers.                                                 |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                        uint window, uint window_key, uint units_count, uint heads,
                                        uint stack_size, uint nactions, uint account_decr,
                                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CResidualConv::Init(numOutputs, myIndex, open_cl, 3, 3, (nactions + 2) / 3, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!caAccountProjection[0].Init(0, index, OpenCL, account_decr, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!caAccountProjection[1].Init(0, index, OpenCL, window * units_count, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMemoryAccount.Init(caAccountProjection[1].Neurons(), index, OpenCL, account_decr,
                           window_key, 1, heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMemoryAction.Init(0, index, OpenCL, 3, window_key, (nactions + 2) / 3,
                          heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cCrossAttention.Init(0, index, OpenCL, 3, window_key, (nactions + 2) / 3,
                            heads, window, units_count, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMacroHFTvsRiskManager feed-forward pass and      |
//| writes the layer output.                                         |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(caAccountProjection[0].getOutput() != SecondInput)
     {
      if(!caAccountProjection[0].SetOutput(SecondInput, true))
         ReturnFalse;
     }
//---
   if(!cMemoryAccount.FeedForward(caAccountProjection[0].AsObject()))
      ReturnFalse;
   if(!caAccountProjection[1].FeedForward(cMemoryAccount.AsObject()))
      ReturnFalse;
   if(!cMemoryAction.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cCrossAttention.FeedForward(cMemoryAction.AsObject(), caAccountProjection[1].getOutput()))
      ReturnFalse;
//---
   return CResidualConv::feedForward(cCrossAttention.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMacroHFTvsRiskManager layer.                              |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput,
      CBufferFloat* SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
   if(caAccountProjection[0].getGradient() != SecondGradient)
     {
      if(!caAccountProjection[0].SetGradient(SecondGradient, true))
         ReturnFalse;
      caAccountProjection[0].SetActivationFunction(SecondActivation);
     }
//---
   if(!CResidualConv::calcInputGradients(cCrossAttention.AsObject()))
      ReturnFalse;
   if(!cMemoryAction.CalcHiddenGradients(cCrossAttention.AsObject(),
                                         caAccountProjection[1].getOutput(),
                                         caAccountProjection[1].getGradient(),
                                         (ENUM_ACTIVATION)caAccountProjection[1].Activation()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cMemoryAction.AsObject()))
      ReturnFalse;
//---
   if(!cMemoryAccount.CalcHiddenGradients(caAccountProjection[1].AsObject()))
      ReturnFalse;
   if(!caAccountProjection[0].CalcHiddenGradients(cMemoryAccount.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMacroHFTvsRiskManager trainable weights from  |
//| the accumulated gradients.                                       |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::updateInputWeights(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput)
  {
   if(!CResidualConv::updateInputWeights(cCrossAttention.AsObject()))
      ReturnFalse;
   if(!cCrossAttention.UpdateInputWeights(cMemoryAction.AsObject(), caAccountProjection[1].getOutput()))
      ReturnFalse;
//---
   if(!cMemoryAction.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   if(!caAccountProjection[1].UpdateInputWeights(cMemoryAccount.AsObject()))
      ReturnFalse;
   if(!cMemoryAccount.UpdateInputWeights(caAccountProjection[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMacroHFTvsRiskManager state to the file handle. |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::Save(const int file_handle)
  {
   if(!CResidualConv::Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caAccountProjection.Size(); i++)
      if(!caAccountProjection[i].Save(file_handle))
         ReturnFalse;
   if(!cMemoryAccount.Save(file_handle))
      ReturnFalse;
   if(!cMemoryAction.Save(file_handle))
      ReturnFalse;
   if(!cCrossAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMacroHFTvsRiskManager state from the file    |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::Load(const int file_handle)
  {
   if(!CResidualConv::Load(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caAccountProjection.Size(); i++)
      if(!LoadInsideLayer(file_handle, caAccountProjection[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMemoryAccount.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMemoryAction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMacroHFTvsRiskManager GPU buffers to the given|
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronMacroHFTvsRiskManager::SetOpenCL(COpenCLMy * obj)
  {
   CResidualConv::SetOpenCL(obj);
   for(uint i = 0; i < caAccountProjection.Size(); i++)
      caAccountProjection[i].SetOpenCL(OpenCL);
   cMemoryAccount.SetOpenCL(OpenCL);
   cMemoryAction.SetOpenCL(OpenCL);
   cCrossAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMacroHFTvsRiskManager execution state and its  |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronMacroHFTvsRiskManager::Clear(void)
  {
   if(!CResidualConv::Clear())
      ReturnFalse;
   for(uint i = 0; i < caAccountProjection.Size(); i++)
      if(!caAccountProjection[i].Clear())
         ReturnFalse;
   if(!cMemoryAccount.Clear())
      ReturnFalse;
   if(!cMemoryAction.Clear())
      ReturnFalse;
   if(!cCrossAttention.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHidformerTSAgent layer and allocates its  |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                   uint window, uint window_key, uint units_count,
                                   uint heads, uint stack_size, uint action_space,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CResidualConv::Init(numOutputs, myIndex, open_cl, 3, 3, (action_space + 2) / 3, optimization_type, batch))
      ReturnFalse;
//--- Role
   int index = 0;
   if(!caRole[0].Init(10 * window_key, index, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   caRole[0].getOutput().Fill(1);
   index++;
   if(!caRole[1].Init(0, index, OpenCL, 10 * window_key, optimization, iBatch))
      ReturnFalse;
//--- State to Role
   index++;
   if(!cStateToRole.Init(0, index, OpenCL, window, window_key, units_count, heads, window_key, 10, optimization, iBatch))
      ReturnFalse;
//--- State
   index++;
   if(!cShuffle.Init(0, index, OpenCL, window, window * units_count, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cRecursiveState.Init(0, index, OpenCL, window, window_key, units_count, heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cResidualState.Init(0, index, OpenCL, window, window, units_count, optimization, iBatch))
      ReturnFalse;
//--- Action
   index++;
   if(!cRecursiveAction.Init(0, index, OpenCL, 3, window_key, (action_space + 2) / 3, heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionToState.Init(0, index, OpenCL, 3, window_key, (action_space + 2) / 3, heads, window, units_count, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHidformerTSAgent feed-forward pass and writes the|
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(bTrain && !caRole[1].FeedForward(caRole[0].AsObject()))
      ReturnFalse;
//--- State to Role
   if(!cStateToRole.FeedForward(NeuronOCL, caRole[1].getOutput()))
      ReturnFalse;
//--- State
   if(!cShuffle.FeedForward(cStateToRole.AsObject()))
      ReturnFalse;
   if(!cRecursiveState.FeedForward(cShuffle.AsObject()))
      ReturnFalse;
   if(!cResidualState.FeedForward(cRecursiveState.AsObject()))
      ReturnFalse;
//--- Action
   if(!cRecursiveAction.FeedForward(AsObject()))
      ReturnFalse;
   if(!cActionToState.FeedForward(cRecursiveAction.AsObject(), cResidualState.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//---
   return CResidualConv::feedForward(cActionToState.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHidformerTSAgent layer.                                   |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CResidualConv::calcInputGradients(cActionToState.AsObject()))
      ReturnFalse;
   if(!cRecursiveAction.CalcHiddenGradients(cActionToState.AsObject(),
         cResidualState.getOutput(),
         cResidualState.getGradient(),
         (ENUM_ACTIVATION)cResidualState.Activation()))
      ReturnFalse;
//--- Action
   CBufferFloat *temp = Gradient;
   if(!SwapBuffers(Output, PrevOutput) ||
      !SetGradient(cRecursiveAction.getPrevOutput(), false))
      ReturnFalse;
   if(!calcHiddenGradients(cRecursiveAction.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   Gradient = temp;
//--- State
   if(!cRecursiveState.CalcHiddenGradients(cResidualState.AsObject()))
      ReturnFalse;
   if(!cShuffle.CalcHiddenGradients(cRecursiveState.AsObject()))
      ReturnFalse;
   if(!cStateToRole.CalcHiddenGradients(cShuffle.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cStateToRole.AsObject(),
                                     caRole[1].getOutput(),
                                     caRole[1].getGradient(),
                                     (ENUM_ACTIVATION)caRole[1].Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHidformerTSAgent trainable weights from the   |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CResidualConv::updateInputWeights(cActionToState.AsObject()))
      ReturnFalse;
//--- Action
   if(!cActionToState.UpdateInputWeights(cRecursiveAction.AsObject()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput) ||
      !cRecursiveAction.UpdateInputWeights(AsObject()) ||
      !SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//--- State
   if(!cResidualState.UpdateInputWeights(cRecursiveState.AsObject()))
      ReturnFalse;
   if(!cResidualState.UpdateInputWeights(cShuffle.AsObject()))
      ReturnFalse;
   if(!cShuffle.UpdateInputWeights(cStateToRole.AsObject()))
      ReturnFalse;
   if(!cStateToRole.UpdateInputWeights(NeuronOCL, caRole[1].getOutput()))
      ReturnFalse;
//--- Role
   if(!caRole[1].UpdateInputWeights(caRole[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHidformerTSAgent state to the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::Save(const int file_handle)
  {
   if(!CResidualConv::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caRole.Size(); i++)
      if(!caRole[i].Save(file_handle))
         ReturnFalse;
   if(!cStateToRole.Save(file_handle))
      ReturnFalse;
   if(!cShuffle.Save(file_handle))
      ReturnFalse;
   if(!cRecursiveState.Save(file_handle))
      ReturnFalse;
   if(!cResidualState.Save(file_handle))
      ReturnFalse;
   if(!cRecursiveAction.Save(file_handle))
      ReturnFalse;
   if(!cActionToState.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHidformerTSAgent state from the file handle. |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::Load(const int file_handle)
  {
   if(!CResidualConv::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caRole.Size(); i++)
      if(!LoadInsideLayer(file_handle, caRole[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStateToRole.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShuffle.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRecursiveState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidualState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRecursiveAction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionToState.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronHidformerTSAgent GPU buffers to the given     |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronHidformerTSAgent::SetOpenCL(COpenCLMy * obj)
  {
   CResidualConv::SetOpenCL(obj);
//---
   for(uint i = 0; i < caRole.Size(); i++)
      caRole[i].SetOpenCL(OpenCL);
   cStateToRole.SetOpenCL(OpenCL);
   cShuffle.SetOpenCL(OpenCL);
   cRecursiveState.SetOpenCL(OpenCL);
   cResidualState.SetOpenCL(OpenCL);
   cRecursiveAction.SetOpenCL(OpenCL);
   cActionToState.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronHidformerTSAgent execution state and its GPU   |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronHidformerTSAgent::Clear(void)
  {
   if(!CResidualConv::Clear())
      ReturnFalse;
//---
   for(uint i = 0; i < caRole.Size(); i++)
      if(!caRole[i].Clear())
         ReturnFalse;
   if(!cStateToRole.Clear())
      ReturnFalse;
   if(!cShuffle.Clear())
      ReturnFalse;
   if(!cRecursiveState.Clear())
      ReturnFalse;
   if(!cResidualState.Clear())
      ReturnFalse;
   if(!cRecursiveAction.Clear())
      ReturnFalse;
   if(!cActionToState.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHidformerFreqAgent layer and allocates its|
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                     uint window, uint filters, uint units_count,
                                     uint heads, uint stack_size, uint action_space,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CResidualConv::Init(numOutputs, myIndex, open_cl, 3, 3, (action_space + 2) / 3, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cTranspose.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   index++;
   uint wind = (units_count >= 20 ? (units_count + 3) / 4 : units_count);
   uint units = (units_count + wind - 1) / wind;
   if(!cLegendre.Init(0, index, OpenCL, wind, wind, units, filters, window, optimization, batch))
      ReturnFalse;
   index++;
   if(!cHLState.Init(0, index, OpenCL, units * window, 2, filters, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAttentionState.Init(0, index, OpenCL, filters, filters, units * window, 2, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cResidualState.Init(0, index, OpenCL, filters, filters, 2 * units * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cShuffle.Init(0, index, OpenCL, filters, cResidualState.Neurons(), optimization, iBatch))
      ReturnFalse;
//--- Action
   index++;
   if(!cRecursiveAction.Init(0, index, OpenCL, 3, filters, (action_space + 2) / 3, heads, stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cActionToState.Init(0, index, OpenCL, 3, filters, (action_space + 2) / 3, heads, filters, 2 * units * window, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHidformerFreqAgent feed-forward pass and writes  |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- State
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cLegendre.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cHLState.FeedForward(cLegendre.AsObject()))
      ReturnFalse;
   if(!cAttentionState.FeedForward(cHLState.AsObject()))
      ReturnFalse;
   if(!cResidualState.FeedForward(cAttentionState.AsObject()))
      ReturnFalse;
   if(!cShuffle.FeedForward(cResidualState.AsObject()))
      ReturnFalse;
//--- Action
   if(!cRecursiveAction.FeedForward(AsObject()))
      ReturnFalse;
   if(!cActionToState.FeedForward(cRecursiveAction.AsObject(), cShuffle.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   return CResidualConv::feedForward(cActionToState.AsObject());
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHidformerFreqAgent layer.                                 |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Action
   if(!CResidualConv::calcInputGradients(cActionToState.AsObject()))
      ReturnFalse;
   if(!cRecursiveAction.CalcHiddenGradients(cActionToState.AsObject(),
         cShuffle.getOutput(),
         cShuffle.getGradient(),
         (ENUM_ACTIVATION)cShuffle.Activation()))
      ReturnFalse;
   CBufferFloat *temp = Gradient;
   if(!SwapBuffers(Output, PrevOutput) ||
      !SetGradient(cActionToState.getPrevOutput(), false) ||
      !calcHiddenGradients(cRecursiveAction.AsObject()) ||
      !SwapBuffers(Output, PrevOutput))
      ReturnFalse;
   Gradient = temp;
//--- State
   if(!cResidualState.CalcHiddenGradients(cShuffle.AsObject()))
      ReturnFalse;
   if(!cAttentionState.CalcHiddenGradients(cResidualState.AsObject()))
      ReturnFalse;
   if(!cHLState.CalcHiddenGradients(cAttentionState.AsObject()))
      ReturnFalse;
   if(!cLegendre.CalcHiddenGradients(cHLState.AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cLegendre.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHidformerFreqAgent trainable weights from the |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- Action
   if(!CResidualConv::updateInputWeights(cActionToState.AsObject()))
      ReturnFalse;
   if(!cActionToState.UpdateInputWeights(cRecursiveAction.AsObject(),
                                         cShuffle.getOutput()))
      ReturnFalse;
   if(!SwapBuffers(Output, PrevOutput) ||
      !cRecursiveAction.UpdateInputWeights(AsObject()) ||
      !SwapBuffers(Output, PrevOutput))
      ReturnFalse;
//--- State
   if(!cShuffle.UpdateInputWeights(cResidualState.AsObject()))
      ReturnFalse;
   if(!cResidualState.UpdateInputWeights(cAttentionState.AsObject()))
      ReturnFalse;
   if(!cAttentionState.UpdateInputWeights(cHLState.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHidformerFreqAgent state to the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::Save(const int file_handle)
  {
   if(!CResidualConv::Save(file_handle))
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cLegendre.Save(file_handle))
      ReturnFalse;
   if(!cHLState.Save(file_handle))
      ReturnFalse;
   if(!cAttentionState.Save(file_handle))
      ReturnFalse;
   if(!cResidualState.Save(file_handle))
      ReturnFalse;
   if(!cShuffle.Save(file_handle))
      ReturnFalse;
//--- Action
   if(!cRecursiveAction.Save(file_handle))
      ReturnFalse;
   if(!cActionToState.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHidformerFreqAgent state from the file       |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::Load(const int file_handle)
  {
   if(!CResidualConv::Load(file_handle))
      ReturnFalse;
//--- State
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLegendre.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHLState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttentionState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidualState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShuffle.AsObject()))
      ReturnFalse;
//--- Action
   if(!LoadInsideLayer(file_handle, cRecursiveAction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActionToState.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronHidformerFreqAgent execution state and its GPU |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronHidformerFreqAgent::Clear(void)
  {
   if(!CResidualConv::Clear())
      ReturnFalse;
//---
   if(!cTranspose.Clear())
      ReturnFalse;
   if(!cLegendre.Clear())
      ReturnFalse;
   if(!cHLState.Clear())
      ReturnFalse;
   if(!cAttentionState.Clear())
      ReturnFalse;
   if(!cResidualState.Clear())
      ReturnFalse;
   if(!cShuffle.Clear())
      ReturnFalse;
//--- Action
   if(!cRecursiveAction.Clear())
      ReturnFalse;
   if(!cActionToState.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronHidformerFreqAgent GPU buffers to the given   |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronHidformerFreqAgent::SetOpenCL(COpenCLMy * obj)
  {
   CResidualConv::SetOpenCL(obj);
//---
   cTranspose.SetOpenCL(OpenCL);
   cLegendre.SetOpenCL(OpenCL);
   cHLState.SetOpenCL(OpenCL);
   cAttentionState.SetOpenCL(OpenCL);
   cResidualState.SetOpenCL(OpenCL);
   cShuffle.SetOpenCL(OpenCL);
//--- Action
   cRecursiveAction.SetOpenCL(OpenCL);
   cActionToState.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHiSSDLowLevelControler layer and allocates|
//| its GPU buffers.                                                 |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
      uint time_step, uint variables, uint task_skills,
      uint common_skills, uint n_actions, uint window,
      uint step, uint window_key, uint heads,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window_key, window_key,
                            n_actions, 1, variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(SIGMOID);
//---
   int index = 0;
   if(!cTaskSpecificSkillsEncoder.Init(0, index, OpenCL, time_step, variables, task_skills,
                                       window, step, window_key, heads, optimization, iBatch))
      ReturnFalse;
   cTaskSpecificSkillsEncoder.SetActivationFunction(None);
//---
   iTaskSkills = task_skills;
   iCommonSkills = MathMax(common_skills, 1);
//---
   index++;
   if(!cTranspose.Init(0, index, OpenCL, time_step, variables, optimization, iBatch))
      ReturnFalse;
//---
   uint window_size = (time_step + iTaskSkills + iCommonSkills);
   index++;
   if(!cObservAndSkillsConcat.Init(0, index, OpenCL, window_size * iVariables,
                                   optimization, iBatch))
      ReturnFalse;
   cObservAndSkillsConcat.SetActivationFunction(None);
   index++;
   if(!cNormalizarion.Init(0, index, OpenCL, cObservAndSkillsConcat.Neurons(),
                           iBatch, optimization))
      ReturnFalse;
   cNormalizarion.SetActivationFunction(None);
   for(uint i = 0; i < cActionDecoder.Size(); i++)
     {
      index++;
      if(!cActionDecoder[i].Init(0, index, OpenCL, window_size, window_size, window_key,
                                 1, iVariables, optimization, iBatch))
         ReturnFalse;
      cActionDecoder[i].SetActivationFunction(SoftPlus);
      window_size = window_key;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHiSSDLowLevelControler feed-forward pass and     |
//| writes the layer output.                                         |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!SecondInput)
      ReturnFalse;
   if(!cTaskSpecificSkillsEncoder.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!Concat(cTranspose.getOutput(), cTaskSpecificSkillsEncoder.getOutput(), SecondInput,
              cObservAndSkillsConcat.getOutput(), cTranspose.GetCount(),
              iTaskSkills, iCommonSkills, iVariables))
      ReturnFalse;
   if(!cNormalizarion.FeedForward(cObservAndSkillsConcat.AsObject()))
      ReturnFalse;
   CNeuronBaseOCL *neuron = cNormalizarion.AsObject();
   for(uint i = 0; i < cActionDecoder.Size(); i++)
     {
      if(!cActionDecoder[i].FeedForward(neuron))
         ReturnFalse;
      neuron = cActionDecoder[i].AsObject();
     }
//---
   return CNeuronConvOCL::feedForward(neuron);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHiSSDLowLevelControler layer.                             |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
      CBufferFloat* SecondInput,
      CBufferFloat* SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
//---
   uint total = cActionDecoder.Size();
   if(total <= 0)
      ReturnFalse;
   CObject *neuron = cActionDecoder[total - 1].AsObject();
//---
   if(!CNeuronConvOCL::calcInputGradients(neuron))
      ReturnFalse;
   for(int i = int(total - 2); i >= 0; i--)
     {
      if(!cActionDecoder[i].CalcHiddenGradients(neuron))
         ReturnFalse;
      neuron = cActionDecoder[i].AsObject();
     }
   if(!cNormalizarion.CalcHiddenGradients(neuron))
      ReturnFalse;
   if(!cObservAndSkillsConcat.CalcHiddenGradients(cNormalizarion.AsObject()))
      ReturnFalse;
   if(!DeConcat(cTranspose.getGradient(), cTaskSpecificSkillsEncoder.getGradient(), SecondGradient,
                cObservAndSkillsConcat.getGradient(), cTranspose.GetCount(),
                iTaskSkills, iCommonSkills, iVariables))
      ReturnFalse;
//---
   if(SecondActivation != None)
     {
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
     }
   if(NeuronOCL.Activation() != None)
     {
      if(!DeActivation(cTranspose.getOutput(), cTranspose.getGradient(),
                       cTranspose.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
     }
   if(cTaskSpecificSkillsEncoder.Activation() != None)
     {
      if(!DeActivation(cTaskSpecificSkillsEncoder.getOutput(), cTaskSpecificSkillsEncoder.getGradient(),
                       cTaskSpecificSkillsEncoder.getGradient(), cTaskSpecificSkillsEncoder.Activation()))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(cTaskSpecificSkillsEncoder.AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cTranspose.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iVariables, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHiSSDLowLevelControler trainable weights from |
//| the accumulated gradients.                                       |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* second)
  {
   if(!cTaskSpecificSkillsEncoder.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cNormalizarion.UpdateInputWeights(cObservAndSkillsConcat.AsObject()))
      ReturnFalse;
   CNeuronBaseOCL *neuron = cNormalizarion.AsObject();
   for(uint i = 0; i < cActionDecoder.Size(); i++)
     {
      if(!cActionDecoder[i].UpdateInputWeights(neuron))
         ReturnFalse;
      neuron = cActionDecoder[i].AsObject();
     }
//---
   return CNeuronConvOCL::updateInputWeights(neuron);
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronHiSSDLowLevelControler weights toward the    |
//| source model with factor tau.                                    |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronHiSSDLowLevelControler *Source = source;
   if(!cTaskSpecificSkillsEncoder.WeightsUpdate(GetPointer(Source.cTaskSpecificSkillsEncoder), tau))
      ReturnFalse;
   if(!cNormalizarion.WeightsUpdate(GetPointer(Source.cNormalizarion), tau))
      ReturnFalse;
   for(uint i = 0; i < cActionDecoder.Size(); i++)
      if(!cActionDecoder[i].WeightsUpdate(GetPointer(Source.cActionDecoder[i]), tau))
         ReturnFalse;
//---
   return CNeuronConvOCL::WeightsUpdate(source, tau);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHiSSDLowLevelControler state to the file handle.|
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iTaskSkills)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iCommonSkills)) < INT_VALUE)
      ReturnFalse;
//---
   if(!cTaskSpecificSkillsEncoder.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cNormalizarion.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cActionDecoder.Size(); i++)
      if(!cActionDecoder[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHiSSDLowLevelControler state from the file   |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronHiSSDLowLevelControler::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iTaskSkills = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCommonSkills = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cTaskSpecificSkillsEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormalizarion.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cActionDecoder.Size(); i++)
      if(!LoadInsideLayer(file_handle, cActionDecoder[i].AsObject()))
         ReturnFalse;
//---
   if(!cObservAndSkillsConcat.Init(0, 2, OpenCL, cNormalizarion.Neurons(),
                                   optimization, iBatch))
      ReturnFalse;
   cObservAndSkillsConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronHiSSDLowLevelControler GPU buffers to the     |
//| given OpenCL context.                                            |
//+------------------------------------------------------------------+
void CNeuronHiSSDLowLevelControler::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
//---
   cTaskSpecificSkillsEncoder.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
   cObservAndSkillsConcat.SetOpenCL(OpenCL);
   cNormalizarion.SetOpenCL(OpenCL);
   for(uint i = 0; i < cActionDecoder.Size(); i++)
      cActionDecoder[i].SetOpenCL(OpenCL);
  }


//+------------------------------------------------------------------+
