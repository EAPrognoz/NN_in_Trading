//+------------------------------------------------------------------+
//|                                            NeuroNet_AgentsRL.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh:      |
//| AgentsRL.                                                        |
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
bool CNeuronFQF::Init(uint numOutputs,
                      uint myIndex,
                      COpenCLMy * open_cl,
                      uint actions,
                      uint quantiles,
                      uint numInputs,
                      ENUM_OPTIMIZATION optimization_type,
                      uint batch
                     )
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronFQF       |
//| layer.                                                           |
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
   if(!cCosine.CalcHiddenGradients(cCosineEmbeding.AsObject()))
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Transfers the accumulated quantile-branch gradients through the  |
//| FQF backward tail.                                               |
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves FQF/ACSRM learned weights without runtime or optimizer     |
//| data.                                                            |
//+------------------------------------------------------------------+
bool CNeuronFQF::SaveParametersOnly(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE ||
      (Type() != defNeuronFQF && Type() != defNeuronACSRM) ||
      !CNeuronBaseOCL::SaveDPCCBaseParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, 7, INT_VALUE) != INT_VALUE ||
      FileWriteInteger(file_handle, (int)cSoftMax.Heads(), INT_VALUE) != INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, cFraction.Type(), INT_VALUE) != INT_VALUE ||
      !cFraction.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cSoftMax.Type(), INT_VALUE) != INT_VALUE ||
      !cSoftMax.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cCosine.Type(), INT_VALUE) != INT_VALUE ||
      !cCosine.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cCosineEmbeding.Type(), INT_VALUE) != INT_VALUE ||
      !cCosineEmbeding.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile0.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile0.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile1.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile1.SaveParametersOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile2.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile2.SaveParametersOnly(file_handle))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates and restores the learned FQF/ACSRM child weight graph. |
//+------------------------------------------------------------------+
bool CNeuronFQF::LoadParametersOnly(const int file_handle, const bool apply)
  {
   if(file_handle == INVALID_HANDLE ||
      (Type() != defNeuronFQF && Type() != defNeuronACSRM) ||
      !CNeuronBaseOCL::LoadDPCCBaseParametersOnly(file_handle, apply))
      ReturnFalse;
   ResetLastError();
   if(FileReadInteger(file_handle, INT_VALUE) != 7 ||
      FileReadInteger(file_handle, INT_VALUE) != (int)cSoftMax.Heads() || GetLastError() != 0)
      ReturnFalse;
   if(FileReadInteger(file_handle, INT_VALUE) != cFraction.Type() ||
      !cFraction.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cSoftMax.Type() ||
      !cSoftMax.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cCosine.Type() ||
      !cCosine.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cCosineEmbeding.Type() ||
      !cCosineEmbeding.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile0.Type() ||
      !cQuantile0.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile1.Type() ||
      !cQuantile1.LoadParametersOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile2.Type() ||
      !cQuantile2.LoadParametersOnly(file_handle, apply) || GetLastError() != 0)
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the FQF/ACSRM trainable children and Adam clocks only.     |
//+------------------------------------------------------------------+
bool CNeuronFQF::SaveDPCCPolicyTrainingStateOnly(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE ||
      (Type() != defNeuronFQF && Type() != defNeuronACSRM) ||
      !SaveDPCCPolicyTrainingBaseOnly(file_handle) ||
      FileWriteInteger(file_handle, 7, INT_VALUE) != INT_VALUE ||
      FileWriteInteger(file_handle, (int)cSoftMax.Heads(), INT_VALUE) != INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, cFraction.Type(), INT_VALUE) != INT_VALUE ||
      !cFraction.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cSoftMax.Type(), INT_VALUE) != INT_VALUE ||
      !cSoftMax.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cCosine.Type(), INT_VALUE) != INT_VALUE ||
      !cCosine.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cCosineEmbeding.Type(), INT_VALUE) != INT_VALUE ||
      !cCosineEmbeding.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile0.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile0.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile1.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile1.SaveDPCCPolicyTrainingStateOnly(file_handle) ||
      FileWriteInteger(file_handle, cQuantile2.Type(), INT_VALUE) != INT_VALUE ||
      !cQuantile2.SaveDPCCPolicyTrainingStateOnly(file_handle))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates/restores the recursive FQF/ACSRM Adam training state.  |
//+------------------------------------------------------------------+
bool CNeuronFQF::LoadDPCCPolicyTrainingStateOnly(const int file_handle, const bool apply)
  {
   if(file_handle == INVALID_HANDLE ||
      (Type() != defNeuronFQF && Type() != defNeuronACSRM) ||
      !LoadDPCCPolicyTrainingBaseOnly(file_handle, apply))
      ReturnFalse;
   ResetLastError();
   if(FileReadInteger(file_handle, INT_VALUE) != 7 ||
      FileReadInteger(file_handle, INT_VALUE) != (int)cSoftMax.Heads() || GetLastError() != 0)
      ReturnFalse;
   if(FileReadInteger(file_handle, INT_VALUE) != cFraction.Type() ||
      !cFraction.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cSoftMax.Type() ||
      !cSoftMax.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cCosine.Type() ||
      !cCosine.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cCosineEmbeding.Type() ||
      !cCosineEmbeding.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile0.Type() ||
      !cQuantile0.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile1.Type() ||
      !cQuantile1.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) ||
      FileReadInteger(file_handle, INT_VALUE) != cQuantile2.Type() ||
      !cQuantile2.LoadDPCCPolicyTrainingStateOnly(file_handle, apply) || GetLastError() != 0)
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Includes every FQF child weight and Adam moment in owner         |
//| binding.                                                         |
//+------------------------------------------------------------------+
bool CNeuronFQF::AppendDPCCPolicyTrainingFingerprint(ulong &fingerprint)
  {
   if((Type() != defNeuronFQF && Type() != defNeuronACSRM) ||
      !AppendDPCCPolicyTrainingBaseFingerprint(fingerprint))
      ReturnFalse;
   fingerprint = (fingerprint ^ ulong(7)) * ulong(1099511628211);
   fingerprint = (fingerprint ^ ulong(cSoftMax.Heads())) * ulong(1099511628211);
   if(!cFraction.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cSoftMax.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cCosine.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cCosineEmbeding.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cQuantile0.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cQuantile1.AppendDPCCPolicyTrainingFingerprint(fingerprint) ||
      !cQuantile2.AppendDPCCPolicyTrainingFingerprint(fingerprint))
      ReturnFalse;
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the layer info of the CNeuronFQF.                        |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronFQF::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return(result);
   result.window_out = cSoftMax.Neurons() / Neurons();
//---
   return(result);
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
//| Constructor: initializes the CNeuronACSRM layer defaults.        |
//+------------------------------------------------------------------+
CNeuronACSRM::CNeuronACSRM(void)
  {
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
  }
//+------------------------------------------------------------------+
//| CPU mirror of the kernel cumulative spectral mass Phi(u).        |
//+------------------------------------------------------------------+
//| Used ONLY by ::ValidateDistributionForSRM to check the           |
//| production spectral-monotonicity contract before an SRM launch:  |
//| the same clamps as SRMCumulative in NeuroNet.cl (tau in [0,1],   |
//| parameters clamped to their domains, SRM_PARAMETER_EPS = 1e-6).  |
//+------------------------------------------------------------------+
float ACSSRMCumulativeCPU(const float u, const int risk_type, const float risk_parameter, const float mean_weight)
  {
   const float tau = (float)MathMin(MathMax(u, 0.0), 1.0);
   switch(risk_type)
     {
      case defSRM_MEAN:
         return(tau);
      case defSRM_CVAR:
        {
         const float alpha = (float)MathMin(MathMax(risk_parameter, 1.0e-6f), 1.0f);
         return((float)MathMin(MathMax(MathMin(tau, alpha) / alpha, 0.0), 1.0));
        }
      case defSRM_MEAN_CVAR:
        {
         const float alpha = (float)MathMin(MathMax(risk_parameter, 1.0e-6f), 1.0f);
         const float omega = (float)MathMin(MathMax(mean_weight, 0.0f), 1.0f);
         return((float)MathMin(MathMax(omega * tau + (1.0f - omega) * MathMin(tau, alpha) / alpha, 0.0), 1.0));
        }
      case defSRM_EXPONENTIAL:
        {
         const float lambda = MathMax(risk_parameter, 0.0f);
         if(lambda <= 1.0e-6f)
            return(tau);
         const float denominator = 1.0f - (float)MathExp(-lambda);
         const float numerator   = 1.0f - (float)MathExp(-lambda * tau);
         return((float)MathMin(MathMax(numerator / denominator, 0.0), 1.0));
        }
      case defSRM_DUAL_POWER:
        {
         const float nu = MathMax(risk_parameter, 1.0f);
         return((float)MathMin(MathMax(1.0f - MathPow(1.0f - tau, nu), 0.0), 1.0));
        }
     }
   return(0.0f);
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronACSRM GPU resources (FQF head    |
//| and the transient rearrangement state).                          |
//+------------------------------------------------------------------+
CNeuronACSRM::~CNeuronACSRM(void)
  {
   m_input_gradient_scratch.BufferFree();
   m_ordered_quantiles.BufferFree();
   m_ordered_probabilities.BufferFree();
   m_rearrange_order.BufferFree();
   m_ordered_quantiles_gradient.BufferFree();
   m_ordered_probabilities_gradient.BufferFree();
   m_rearrange_diagnostics.BufferFree();
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronACSRM layer over the FQF backbone.        |
//+------------------------------------------------------------------+
bool CNeuronACSRM::Init(uint numOutputs,
                        uint myIndex,
                        COpenCLMy * open_cl,
                        uint actions,
                        uint quantiles,
                        uint numInputs,
                        ENUM_OPTIMIZATION optimization_type,
                        uint batch
                       )
  {
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
   m_last_invalid_quantile = 0;
   m_last_invalid_probability = 0;
   m_last_invalid_mass = 0;
   m_last_spectral_break = 0;
   m_last_srm_invalid = false;
   if(!CNeuronFQF::Init(numOutputs, myIndex, open_cl, actions, quantiles, numInputs, optimization_type, batch))
      ReturnFalse;
   if(!m_input_gradient_scratch.BufferInit(cCosineEmbeding.Neurons(), 0) ||
      !m_input_gradient_scratch.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronACSRM feed-forward pass.                         |
//+------------------------------------------------------------------+
bool CNeuronACSRM::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
   return(CNeuronFQF::feedForward(NeuronOCL));
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronACSRM gradient-propagation dispatch.          |
//+------------------------------------------------------------------+
bool CNeuronACSRM::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!m_distribution_gradient_ready)
      return(CNeuronFQF::calcInputGradients(NeuronOCL));
//--- specialized AC-SRM backward over the prepared internal gradients
   const bool result = propagateACSRMGradients(NeuronOCL);
   m_distribution_gradient_ready = false;
   return(result);
  }
//+------------------------------------------------------------------+
//| Transfers the AC-SRM prepared gradients through the FQF tail.    |
//+------------------------------------------------------------------+
bool CNeuronACSRM::propagateACSRMGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !Gradient || !Output)
      ReturnFalse;
//--- quantile branch: cQuantile2 -> cQuantile1 -> cQuantile0
   if(!cQuantile1.CalcHiddenGradients(cQuantile2.AsObject()))
      ReturnFalse;
   if(!cQuantile0.CalcHiddenGradients(cQuantile1.AsObject()))
      ReturnFalse;
//--- value-path input gradient and the embedding feature gradients
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
//--- apply the cCosineEmbeding LReLU derivative BEFORE the cosine
//--- feature backward: e = LReLU(a), g_a = g_e * LReLU'(a). The gradient
//--- of e (from the value path) sits in cCosineEmbeding.Gradient; the
//--- DeActivation is exact because sign(e) == sign(a) for LReLU.
   if(!DeActivation(cCosineEmbeding.getOutput(),
                    cCosineEmbeding.getGradient(),
                    cCosineEmbeding.getGradient(),
                    cCosineEmbeding.Activation()))
      ReturnFalse;
//--- cosine features backward (the connection matrix lives in cCosine)
   if(!cCosine.CalcHiddenGradients(cCosineEmbeding.AsObject()))
      ReturnFalse;
//--- probability gradient at the soft-max output: ::FQF_CosineGradientSRM
//--- accumulates the quantile-branch contribution on top of the direct
//--- SRM/distributional gradient in cSoftMax.Gradient
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2] = { cSoftMax.Neurons() / Neurons(), Neurons() };
      setBuffer(def_k_FQF_CosineGradientSRM, def_k_fqfcosgr_softmax, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_CosineGradientSRM, def_k_fqfcosgr_output_gr, cCosine.getGradientIndex());
      setBuffer(def_k_FQF_CosineGradientSRM, def_k_fqfcosgr_softmax_gr, cSoftMax.getGradientIndex());
      kernelExecute(def_k_FQF_CosineGradientSRM, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cSoftMax.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//--- fraction head: soft-max input gradient (cFraction.Gradient) plus the
//--- cFraction weight gradients
   if(!cFraction.CalcHiddenGradients(cSoftMax.AsObject()))
      ReturnFalse;
//--- probability branch to the shared input. The full derivative is
//---   d rho / d x = (value path) + (dP/dx)^T g_P
//--- The FQF feed-forward convention maps the fraction logits through the
//--- PREVIOUS layer weight matrix (cFraction.FeedForward(NeuronOCL) reads
//--- NeuronOCL.getWeights(); cFraction itself holds no weights), so the
//--- probability branch to the shared input is W_prev^T * g_logits. It is
//--- computed exactly, without the per-layer activation clipping of
//--- ::CalcHiddenGradient, by ::FQF_FractionGradient and both contributions
//--- are summed into NeuronOCL.Gradient by SumMatrix.
     {
      const uint inputs = cCosineEmbeding.Neurons();
      const uint outputs = cFraction.Neurons();
      if(inputs == 0 || outputs == 0 ||
         (uint)m_input_gradient_scratch.Total() < inputs ||
         NeuronOCL.Neurons() != inputs ||
         CheckPointer(NeuronOCL.getWeights()) == POINTER_INVALID)
         ReturnFalse;
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2];
      global_work_size[0] = inputs;
      //--- spec 16: FQF_FractionGradient local_size[0] == 1 (reduction
      //--- strictly along dimension 1) and local_size[1] <= LOCAL_ARRAY_SIZE
      global_work_size[1] = (uint)MathMin(MathMin((outputs + 3) / 4, OpenCL.GetMaxLocalSize(1)),
                                          defFQF_LOCAL_ARRAY_SIZE);
      uint local_work_size[2] = { 1, global_work_size[1] };
      setBuffer(def_k_FQF_FractionGradient, def_k_fqffg_matrix_w, NeuronOCL.getWeightsIndex());
      setBuffer(def_k_FQF_FractionGradient, def_k_fqffg_matrix_g, cFraction.getGradientIndex());
      setBuffer(def_k_FQF_FractionGradient, def_k_fqffg_matrix_ig, m_input_gradient_scratch.GetIndex());
      setArgument(def_k_FQF_FractionGradient, def_k_fqffg_outputs, (int)outputs);
      kernelExecuteLoc(def_k_FQF_FractionGradient, global_work_offset, global_work_size, local_work_size)
      //--- NeuronOCL.Gradient = value_path + fraction_branch
        {
         uint sum_work_offset[1] = {0};
         uint sum_work_size[1] = { inputs };
         setBuffer(def_k_MatrixSum, def_k_sum_matrix1, NeuronOCL.getGradientIndex());
         setBuffer(def_k_MatrixSum, def_k_sum_matrix2, m_input_gradient_scratch.GetIndex());
         setBuffer(def_k_MatrixSum, def_k_sum_matrix_out, NeuronOCL.getGradientIndex());
         setArgument(def_k_MatrixSum, def_k_sum_dimension, 1);
         setArgument(def_k_MatrixSum, def_k_sum_multiplyer, 1.0f);
         setArgument(def_k_MatrixSum, def_k_sum_shift_in1, 0);
         setArgument(def_k_MatrixSum, def_k_sum_shift_in2, 0);
         setArgument(def_k_MatrixSum, def_k_sum_shift_out, 0);
         setBuffer(def_k_MatrixSum, def_k_sum_numeric_flags, NeuronOCL.getGradientIndex());
         setArgument(def_k_MatrixSum, def_k_sum_flags_enabled, 0);
         kernelExecute(def_k_MatrixSum, sum_work_offset, sum_work_size)
        }
#ifdef  _DEBUG
      if(!NeuronOCL.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Computes the distributional target gradients for the AC-SRM      |
//| layer and marks them prepared for the specialized FQF tail.      |
//+------------------------------------------------------------------+
bool CNeuronACSRM::calcDistributionTargetGradients(CNeuronBaseOCL* NeuronOCL,
                                                   float target,
                                                   float probability_weight,
                                                   float value_weight
                                                  )
  {
   m_distribution_gradient_ready = false;
   if(!NeuronOCL || !OpenCL || !Gradient || !Output)
      ReturnFalse;
   if(probability_weight < 0 || value_weight < 0)
      ReturnFalse;
   if(!MathIsValidNumber(target) || !MathIsValidNumber(probability_weight) || !MathIsValidNumber(value_weight))
      ReturnFalse;
   if(CheckPointer(cSoftMax.getOutput()) == POINTER_INVALID || CheckPointer(cSoftMax.getGradient()) == POINTER_INVALID)
      ReturnFalse;
   if(CheckPointer(cQuantile2.getOutput()) == POINTER_INVALID || CheckPointer(cQuantile2.getGradient()) == POINTER_INVALID)
      ReturnFalse;
   if(cSoftMax.Neurons() != cQuantile2.Neurons() || cSoftMax.Neurons() == 0)
      ReturnFalse;
   if(Neurons() != 1)
      ReturnFalse;
   const uint quantiles = cSoftMax.Neurons() / Neurons();
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE)
      ReturnFalse;
   const long max_work_group = OpenCL.GetMaxWorkGroupSize();
   if(max_work_group > 0 && (long)quantiles > max_work_group)
      ReturnFalse;
//--- the kernel performs local reductions over get_local_size(0); the local
//--- group must cover the whole distribution so the geometry is [N,1]/[N,1]
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, Neurons() };
      uint local_work_size[2]   = { quantiles, 1 };
      setBuffer(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_quantiles, cQuantile2.getOutputIndex());
      setBuffer(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_probabilities, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_quantiles_gr, cQuantile2.getGradientIndex());
      setBuffer(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_probabilities_gr, cSoftMax.getGradientIndex());
      setArgument(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_target, target);
      setArgument(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_probability_weight, probability_weight);
      setArgument(def_k_FQF_DistributionTargetGradient, def_k_fqfdtg_value_weight, value_weight);
      kernelExecuteLoc(def_k_FQF_DistributionTargetGradient, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!cQuantile2.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   m_distribution_gradient_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Samples scalar realizations from the AC-SRM distributions.       |
//+------------------------------------------------------------------+
//| One work-group per task computes the local inclusive CDF of its  |
//| N probabilities, locates the task random value u_t inside the    |
//| CDF and writes the selected centroid of that distribution.       |
//+------------------------------------------------------------------+
bool CNeuronACSRM::Sample(CBufferFloat *result, CBufferFloat *random, const uint tasks)
  {
   if(!OpenCL || !result || !random)
      ReturnFalse;
   if(result.GetOpenCL() != OpenCL || random.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(tasks == 0)
      ReturnFalse;
//--- N quantiles inside one independent distribution
   const uint quantiles = (Neurons() > 0 ? cQuantile2.Neurons() / Neurons() : 0);
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE)
      ReturnFalse;
   const long max_work_group = OpenCL.GetMaxWorkGroupSize();
   if(max_work_group > 0 && (long)quantiles > max_work_group)
      ReturnFalse;
//--- both distribution buffers must cover exactly tasks * N elements
   if((uint)cSoftMax.getOutput().Total() != tasks * quantiles ||
      (uint)cQuantile2.getOutput().Total() != tasks * quantiles)
      ReturnFalse;
   if((uint)random.Total() < tasks || (uint)result.Total() < tasks)
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, tasks };
      uint local_work_size[2]   = { quantiles, 1 };
      setBuffer(def_k_FQF_Sample, def_k_fqfs_quantiles, cQuantile2.getOutputIndex());
      setBuffer(def_k_FQF_Sample, def_k_fqfs_probabilities, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_Sample, def_k_fqfs_random, random.GetIndex());
      setBuffer(def_k_FQF_Sample, def_k_fqfs_result, result.GetIndex());
      kernelExecuteLoc(def_k_FQF_Sample, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!result.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Converts the AC-SRM distributions into scalar spectral risk      |
//| scores.                                                          |
//+------------------------------------------------------------------+
//| One work-group per task builds the local inclusive CDF of its N  |
//| probabilities, normalizes the CDF axis by the total mass and     |
//| evaluates the exact spectral mass Phi(tau_high) - Phi(tau_low)   |
//| of every adaptive interval; the risk score is sum_i w_i z_i.     |
//+------------------------------------------------------------------+
bool CNeuronACSRM::SRMForward(CBufferFloat *result,
                              const uint tasks,
                              const int risk_type,
                              const float risk_parameter,
                              const float mean_weight
                             )
  {
   m_last_srm_invalid = false;
   if(!OpenCL || !result)
      ReturnFalse;
   if(result.GetOpenCL() != OpenCL)
      ReturnFalse;
//--- spec 33: parameter validation per risk measure
   switch(risk_type)
     {
      case defSRM_MEAN:
         break;
      case defSRM_CVAR:
         if(risk_parameter <= 0.0f || risk_parameter > 1.0f)
            ReturnFalse;
         break;
      case defSRM_MEAN_CVAR:
         if(risk_parameter <= 0.0f || risk_parameter > 1.0f)
            ReturnFalse;
         if(mean_weight < 0.0f || mean_weight > 1.0f)
            ReturnFalse;
         break;
      case defSRM_EXPONENTIAL:
         if(risk_parameter < 0.0f)
            ReturnFalse;
         break;
      case defSRM_DUAL_POWER:
         if(risk_parameter < 1.0f)
            ReturnFalse;
         break;
      default:
         ReturnFalse;
     }
   if(!MathIsValidNumber(risk_parameter) || !MathIsValidNumber(mean_weight))
      ReturnFalse;
   if(tasks == 0)
      ReturnFalse;
//--- N quantiles and result sizing
   if((uint)result.Total() < tasks)
      ReturnFalse;
   const uint quantiles = (Neurons() > 0 ? cSoftMax.Neurons() / Neurons() : 0);
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE)
      ReturnFalse;
   const long max_work_group = OpenCL.GetMaxWorkGroupSize();
   if(max_work_group > 0 && (long)quantiles > max_work_group)
      ReturnFalse;
//--- spec 17: AC-SRM Stage 03 pins Actions == 1, distribution head == 1
   if(Neurons() != 1)
      ReturnFalse;
//--- rank-sensitive measures consume the monotone rearrangement; the
//--- bitonic v1 network requires N = 2^k
   const bool rank_sensitive = (risk_type != defSRM_MEAN);
   if(rank_sensitive && ((quantiles & (quantiles - 1)) != 0))
      ReturnFalse;
//--- spec 12/18: invalid distributions are rejected BEFORE the SRM launch
//--- and never interpreted as a valid protective zero
   m_last_srm_invalid = !ValidateDistributionForSRM(tasks, risk_type, risk_parameter, mean_weight, true);
   if(m_last_srm_invalid)
      ReturnFalse;
   if(rank_sensitive)
     {
      if(!EnsureRearrangeBuffers(tasks, quantiles))
         ReturnFalse;
      if(!MonotoneRearrange(GetPointer(m_ordered_quantiles), GetPointer(m_ordered_probabilities),
                            GetPointer(m_rearrange_order), GetPointer(m_rearrange_diagnostics), tasks))
         ReturnFalse;
     }
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, tasks };
      uint local_work_size[2]   = { quantiles, 1 };
      const int q_index = (rank_sensitive ? m_ordered_quantiles.GetIndex() : cQuantile2.getOutputIndex());
      const int p_index = (rank_sensitive ? m_ordered_probabilities.GetIndex() : cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_SRMForward, def_k_fqfsrm_quantiles, q_index);
      setBuffer(def_k_FQF_SRMForward, def_k_fqfsrm_probabilities, p_index);
      setBuffer(def_k_FQF_SRMForward, def_k_fqfsrm_result, result.GetIndex());
      setArgument(def_k_FQF_SRMForward, def_k_fqfsrm_risk_type, (int)risk_type);
      setArgument(def_k_FQF_SRMForward, def_k_fqfsrm_risk_parameter, risk_parameter);
      setArgument(def_k_FQF_SRMForward, def_k_fqfsrm_mean_weight, mean_weight);
      kernelExecuteLoc(def_k_FQF_SRMForward, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!result.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the SRM risk-score gradients.                    |
//+------------------------------------------------------------------+
//| One work-group per task runs ::FQF_SRMGradient: the direct       |
//| centroid gradient g_t * w_i writes cQuantile2.Gradient and the   |
//| direct probability gradient (A_i - B) / M writes                 |
//| cSoftMax.Gradient. The existing FQF backward tail then adds the  |
//| quantile-branch contribution to cSoftMax.Gradient via            |
//| ::FQF_CosineGradient before cSoftMax.calcInputGradients.         |
//+------------------------------------------------------------------+
bool CNeuronACSRM::SRMGradient(CBufferFloat *result_gradient,
                               const uint tasks,
                               const int risk_type,
                               const float risk_parameter,
                               const float mean_weight
                              )
  {
   m_distribution_gradient_ready = false;
   m_last_srm_invalid = false;
   if(!OpenCL || !result_gradient ||
      !cQuantile2.getGradient() || !cSoftMax.getGradient())
      ReturnFalse;
   if(result_gradient.GetOpenCL() != OpenCL)
      ReturnFalse;
//--- spec 40: parameter validation per risk measure (same as SRMForward)
   switch(risk_type)
     {
      case defSRM_MEAN:
         break;
      case defSRM_CVAR:
         if(risk_parameter <= 0.0f || risk_parameter > 1.0f)
            ReturnFalse;
         break;
      case defSRM_MEAN_CVAR:
         if(risk_parameter <= 0.0f || risk_parameter > 1.0f)
            ReturnFalse;
         if(mean_weight < 0.0f || mean_weight > 1.0f)
            ReturnFalse;
         break;
      case defSRM_EXPONENTIAL:
         if(risk_parameter < 0.0f)
            ReturnFalse;
         break;
      case defSRM_DUAL_POWER:
         if(risk_parameter < 1.0f)
            ReturnFalse;
         break;
      default:
         ReturnFalse;
     }
   if(!MathIsValidNumber(risk_parameter) || !MathIsValidNumber(mean_weight))
      ReturnFalse;
   if((uint)result_gradient.Total() < tasks)
      ReturnFalse;
   const uint quantiles = (Neurons() > 0 ? cSoftMax.Neurons() / Neurons() : 0);
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE)
      ReturnFalse;
   const long max_work_group = OpenCL.GetMaxWorkGroupSize();
   if(max_work_group > 0 && (long)quantiles > max_work_group)
      ReturnFalse;
//--- spec 17: AC-SRM Stage 03 pins Actions == 1, distribution head == 1
   if(Neurons() != 1)
      ReturnFalse;
   const bool rank_sensitive = (risk_type != defSRM_MEAN);
   if(rank_sensitive && ((quantiles & (quantiles - 1)) != 0))
      ReturnFalse;
//--- spec 12/18: invalid distributions (and an invalid upstream gradient)
//--- are rejected BEFORE the SRM gradient launch; a protective zero is
//--- never treated as a valid break-even outcome
   if(result_gradient.GetIndex() >= 0 && !result_gradient.BufferRead())
      ReturnFalse;
   for(uint u = 0; u < tasks && u < (uint)result_gradient.Total(); u++)
      if(!MathIsValidNumber(result_gradient[u]))
        { m_last_srm_invalid = true; ReturnFalse; }
   m_last_srm_invalid = !ValidateDistributionForSRM(tasks, risk_type, risk_parameter, mean_weight, false);
   if(m_last_srm_invalid)
      ReturnFalse;
   if(rank_sensitive)
     {
      if(!EnsureRearrangeBuffers(tasks, quantiles))
         ReturnFalse;
      if(!MonotoneRearrange(GetPointer(m_ordered_quantiles), GetPointer(m_ordered_probabilities),
                            GetPointer(m_rearrange_order), GetPointer(m_rearrange_diagnostics), tasks))
         ReturnFalse;
     }
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, tasks };
      uint local_work_size[2]   = { quantiles, 1 };
      const int q_index = (rank_sensitive ? m_ordered_quantiles.GetIndex() : cQuantile2.getOutputIndex());
      const int p_index = (rank_sensitive ? m_ordered_probabilities.GetIndex() : cSoftMax.getOutputIndex());
      const int qg_index = (rank_sensitive ? m_ordered_quantiles_gradient.GetIndex() : cQuantile2.getGradientIndex());
      const int pg_index = (rank_sensitive ? m_ordered_probabilities_gradient.GetIndex() : cSoftMax.getGradientIndex());
      setBuffer(def_k_FQF_SRMGradient, def_k_fqfsrmg_quantiles, q_index);
      setBuffer(def_k_FQF_SRMGradient, def_k_fqfsrmg_probabilities, p_index);
      setBuffer(def_k_FQF_SRMGradient, def_k_fqfsrmg_result_gradient, result_gradient.GetIndex());
      setBuffer(def_k_FQF_SRMGradient, def_k_fqfsrmg_quantiles_gradient, qg_index);
      setBuffer(def_k_FQF_SRMGradient, def_k_fqfsrmg_probabilities_gradient, pg_index);
      setArgument(def_k_FQF_SRMGradient, def_k_fqfsrmg_risk_type, (int)risk_type);
      setArgument(def_k_FQF_SRMGradient, def_k_fqfsrmg_risk_parameter, risk_parameter);
      setArgument(def_k_FQF_SRMGradient, def_k_fqfsrmg_mean_weight, mean_weight);
      kernelExecuteLoc(def_k_FQF_SRMGradient, global_work_offset, global_work_size, local_work_size)
      if(rank_sensitive &&
         !RearrangeGradient(GetPointer(m_ordered_quantiles_gradient), GetPointer(m_ordered_probabilities_gradient),
                            GetPointer(m_rearrange_order), cQuantile2.getGradient(), cSoftMax.getGradient(), tasks))
        { m_last_srm_invalid = true; ReturnFalse; }
#ifdef  _DEBUG
      if(!cQuantile2.getGradient().BufferRead())
         ReturnFalse;
      if(!cSoftMax.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   m_distribution_gradient_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocates the transient rearrangement buffers (tasks*quantiles   |
//| for the ordered pairs/order/gradients, tasks*6 diagnostics).     |
//+------------------------------------------------------------------+
bool CNeuronACSRM::EnsureRearrangeBuffers(const uint tasks, const uint quantiles)
  {
   const int need = (int)(tasks * quantiles);
   const int need_diag = (int)(tasks * defFQF_DIAGNOSTIC_FLOATS);
   CBufferFloat *parts[6];
   parts[0] = GetPointer(m_ordered_quantiles);
   parts[1] = GetPointer(m_ordered_probabilities);
   parts[2] = GetPointer(m_rearrange_order);
   parts[3] = GetPointer(m_ordered_quantiles_gradient);
   parts[4] = GetPointer(m_ordered_probabilities_gradient);
   parts[5] = GetPointer(m_rearrange_diagnostics);
   int sizes[6];
   sizes[0] = need;
   sizes[1] = need;
   sizes[2] = need;
   sizes[3] = need;
   sizes[4] = need;
   sizes[5] = need_diag;
   for(int k = 0; k < 6; k++)
     {
      if(parts[k].Total() == sizes[k] && parts[k].GetIndex() >= 0)
         continue;
      if(!parts[k].BufferInit(sizes[k], 0))
         return(false);
      if(!parts[k].BufferCreate(OpenCL))
         return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs ::FQF_MonotoneRearrange on the current distribution.        |
//+------------------------------------------------------------------+
bool CNeuronACSRM::MonotoneRearrange(CBufferFloat *ordered_quantiles, CBufferFloat *ordered_probabilities,
                                     CBufferFloat *order, CBufferFloat *diagnostics, const uint tasks)
  {
   if(!OpenCL || !ordered_quantiles || !ordered_probabilities || !order || !diagnostics)
      ReturnFalse;
   if(ordered_quantiles.GetOpenCL() != OpenCL || ordered_probabilities.GetOpenCL() != OpenCL ||
      order.GetOpenCL() != OpenCL || diagnostics.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(tasks == 0)
      ReturnFalse;
   if(Neurons() != 1)
      ReturnFalse;
   const uint quantiles = (Neurons() > 0 ? cQuantile2.Neurons() / Neurons() : 0);
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE ||
      (quantiles & (quantiles - 1)) != 0)
      ReturnFalse;
   const long max_work_group = OpenCL.GetMaxWorkGroupSize();
   if(max_work_group > 0 && (long)quantiles > max_work_group)
      ReturnFalse;
   if((uint)cSoftMax.getOutput().Total() != tasks * quantiles ||
      (uint)cQuantile2.getOutput().Total() != tasks * quantiles)
      ReturnFalse;
   if((uint)ordered_quantiles.Total() < tasks * quantiles ||
      (uint)ordered_probabilities.Total() < tasks * quantiles ||
      (uint)order.Total() < tasks * quantiles ||
      (uint)diagnostics.Total() < tasks * defFQF_DIAGNOSTIC_FLOATS)
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, tasks };
      uint local_work_size[2]   = { quantiles, 1 };
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_quantiles, cQuantile2.getOutputIndex());
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_probabilities, cSoftMax.getOutputIndex());
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_ordered_quantiles, ordered_quantiles.GetIndex());
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_ordered_probabilities, ordered_probabilities.GetIndex());
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_order, order.GetIndex());
      setBuffer(def_k_FQF_MonotoneRearrange, def_k_fqfmon_diagnostics, diagnostics.GetIndex());
      kernelExecuteLoc(def_k_FQF_MonotoneRearrange, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!ordered_quantiles.BufferRead() || !ordered_probabilities.BufferRead() ||
         !order.BufferRead() || !diagnostics.BufferRead())
         ReturnFalse;
#endif
     }
   m_rearrangement_ready = (order == GetPointer(m_rearrange_order));
   return(true);
  }
//+------------------------------------------------------------------+
//| Scatters the sorted-pair gradients back to the raw FQF indexing. |
//+------------------------------------------------------------------+
bool CNeuronACSRM::RearrangeGradient(CBufferFloat *ordered_quantiles_gradient,
                                     CBufferFloat *ordered_probabilities_gradient,
                                     CBufferFloat *order,
                                     CBufferFloat *quantiles_gradient,
                                     CBufferFloat *probabilities_gradient,
                                     const uint tasks
                                    )
  {
   if(!OpenCL || !ordered_quantiles_gradient || !ordered_probabilities_gradient ||
      !order || !quantiles_gradient || !probabilities_gradient)
      ReturnFalse;
   if(ordered_quantiles_gradient.GetOpenCL() != OpenCL || ordered_probabilities_gradient.GetOpenCL() != OpenCL ||
      order.GetOpenCL() != OpenCL || quantiles_gradient.GetOpenCL() != OpenCL ||
      probabilities_gradient.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(tasks == 0)
      ReturnFalse;
   const uint quantiles = (Neurons() > 0 ? cQuantile2.Neurons() / Neurons() : 0);
   if(quantiles == 0 || quantiles > defFQF_LOCAL_ARRAY_SIZE)
      ReturnFalse;
//--- the permutation must be exactly [0..N-1] (checked before launch)
   if((uint)order.Total() < tasks * quantiles)
      ReturnFalse;
   if((uint)quantiles_gradient.Total() < tasks * quantiles ||
      (uint)probabilities_gradient.Total() < tasks * quantiles)
      ReturnFalse;
   if((uint)ordered_quantiles_gradient.Total() < tasks * quantiles ||
      (uint)ordered_probabilities_gradient.Total() < tasks * quantiles)
      ReturnFalse;
//--- permutation validity contract: order entries are a bijection of 0..N-1
   if(order.GetIndex() >= 0 && !order.BufferRead())
      ReturnFalse;
   bool seen[];
   if(!ArrayResize(seen, (int)quantiles))
      ReturnFalse;
   for(uint tt = 0; tt < tasks; tt++)
     {
      for(uint i = 0; i < quantiles; i++)
         seen[i] = false;
      for(uint k = 0; k < quantiles; k++)
        {
         const int idx = (int)(tt * quantiles + k);
         const double raw_order = double(order[idx]);
         if(!MathIsValidNumber(raw_order))
           { m_last_srm_invalid = true; ReturnFalse; }
         const int o = (int)MathRound(raw_order);
         if(MathAbs(raw_order - double(o)) > 1.0e-6 ||
            o < 0 || o >= (int)quantiles || seen[o])
           { m_last_srm_invalid = true; ReturnFalse; }
         seen[o] = true;
        }
     }
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint global_work_size[2]  = { quantiles, tasks };
      setBuffer(def_k_FQF_RearrangeGradient, def_k_fqfrg_ordered_quantiles_gradient, ordered_quantiles_gradient.GetIndex());
      setBuffer(def_k_FQF_RearrangeGradient, def_k_fqfrg_ordered_probabilities_gradient, ordered_probabilities_gradient.GetIndex());
      setBuffer(def_k_FQF_RearrangeGradient, def_k_fqfrg_order, order.GetIndex());
      setBuffer(def_k_FQF_RearrangeGradient, def_k_fqfrg_quantiles_gradient, quantiles_gradient.GetIndex());
      setBuffer(def_k_FQF_RearrangeGradient, def_k_fqfrg_probabilities_gradient, probabilities_gradient.GetIndex());
      kernelExecute(def_k_FQF_RearrangeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!quantiles_gradient.BufferRead() || !probabilities_gradient.BufferRead())
         ReturnFalse;
#endif
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Probability/numerical contract validation before an SRM launch.  |
//+------------------------------------------------------------------+
bool CNeuronACSRM::ValidateDistributionForSRM(const uint tasks, const int risk_type,
      const float risk_parameter, const float mean_weight,
      const bool spectral_break_diagnostics)
  {
   m_last_invalid_quantile = 0;
   m_last_invalid_probability = 0;
   m_last_invalid_mass = 0;
   m_last_spectral_break = 0;
   if(!OpenCL || !cSoftMax.getOutput() || !cQuantile2.getOutput())
      ReturnFalse;
   if(tasks == 0)
      ReturnFalse;
   const uint quantiles = (Neurons() > 0 ? cSoftMax.Neurons() / Neurons() : 0);
   if(quantiles == 0)
      ReturnFalse;
   CBufferFloat *z = cQuantile2.getOutput();
   CBufferFloat *p = cSoftMax.getOutput();
   if(z.Total() < (int)(tasks * quantiles) || p.Total() < (int)(tasks * quantiles))
      ReturnFalse;
   if(z.GetIndex() >= 0 && !z.BufferRead())
      ReturnFalse;
   if(p.GetIndex() >= 0 && !p.BufferRead())
      ReturnFalse;
   bool valid = true;
   for(uint tt = 0; tt < tasks; tt++)
     {
      double mass = 0.0;
      for(uint k = 0; k < quantiles; k++)
        {
         const int idx = (int)(tt * quantiles + k);
         const double q = double(z[idx]);
         const double pi = double(p[idx]);
         if(!MathIsValidNumber(q))
           { m_last_invalid_quantile++; valid = false; }
         if(!MathIsValidNumber(pi) || pi < 0.0)
           { m_last_invalid_probability++; valid = false; }
         if(MathIsValidNumber(pi) && pi > 0.0)
            mass += pi;
        }
      if(MathAbs(mass - 1.0) > defFQF_PROBABILITY_TOLERANCE)
        { m_last_invalid_mass++; valid = false; }
     }
   if(!valid)
      ReturnFalse;
//--- spectral monotonicity: Phi(tau_high) >= Phi(tau_low) - eps for the
//--- same normalized CDF consumed by ::FQF_SRMForward; CVaR/Mean-CVaR
//--- additionally count the non-differentiable break-point hits |tau-alpha|
   const bool count_kink = spectral_break_diagnostics &&
                           (risk_type == defSRM_CVAR || risk_type == defSRM_MEAN_CVAR);
   for(uint tt = 0; tt < tasks; tt++)
     {
      double S = 0.0;
      for(uint k = 0; k < quantiles; k++)
         S += MathMax(0.0, double(p[tt * quantiles + k]));
      if(!(S > 1.0e-37))
        { m_last_invalid_mass++; valid = false; break; }
      double tau_low = 0.0;
      double cum = 0.0;
      for(uint k = 0; k < quantiles; k++)
        {
         cum += MathMax(0.0, double(p[tt * quantiles + k]));
         const float tau_high = (float)MathMin(MathMax(cum / S, 0.0), 1.0);
         const float phi_high = ACSSRMCumulativeCPU(tau_high, risk_type, risk_parameter, mean_weight);
         const float phi_low  = ACSSRMCumulativeCPU((float)tau_low, risk_type, risk_parameter, mean_weight);
         if(phi_high < phi_low - defFQF_SPECTRAL_EPS)
           { valid = false; break; }
         if(count_kink && MathAbs(double(tau_high) - double(risk_parameter)) < defFQF_KINK_EPS)
            m_last_spectral_break++;
         tau_low = double(tau_high);
        }
      if(!valid)
         break;
     }
   return(valid);
  }
//+------------------------------------------------------------------+
//| Read-only diagnostic access to the last SRM validation counters. |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetDistributionDiagnostics(int &invalid_quantile, int &invalid_probability,
      int &invalid_mass, int &spectral_break) const
  {
   invalid_quantile = m_last_invalid_quantile;
   invalid_probability = m_last_invalid_probability;
   invalid_mass = m_last_invalid_mass;
   spectral_break = m_last_spectral_break;
   return(true);
  }
//+------------------------------------------------------------------+
//| Read-only diagnostic access to the transient rearrangement       |
//| state.                                                           |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetRearrangeBuffers(CBufferFloat *&ordered_quantiles, CBufferFloat *&ordered_probabilities,
                                       CBufferFloat *&order, CBufferFloat *&diagnostics)
  {
   ordered_quantiles = GetPointer(m_ordered_quantiles);
   ordered_probabilities = GetPointer(m_ordered_probabilities);
   order = GetPointer(m_rearrange_order);
   diagnostics = GetPointer(m_rearrange_diagnostics);
   return(m_rearrangement_ready);
  }
//+------------------------------------------------------------------+
//| Read-only diagnostic access to the ordered gradient buffers.     |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetOrderedGradientBuffers(CBufferFloat *&ordered_quantiles_gr,
                                             CBufferFloat *&ordered_probabilities_gr
                                            )
  {
   ordered_quantiles_gr = GetPointer(m_ordered_quantiles_gradient);
   ordered_probabilities_gr = GetPointer(m_ordered_probabilities_gradient);
   return(ordered_quantiles_gr != NULL && ordered_probabilities_gr != NULL);
  }
//+------------------------------------------------------------------+
//| Read-only diagnostic access to the FQF centroids and             |
//| probabilities (cQuantile2/cSoftMax output buffers).              |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetDistributionBuffers(CBufferFloat *&quantiles, CBufferFloat *&probabilities)
  {
   quantiles = cQuantile2.getOutput();
   probabilities = cSoftMax.getOutput();
   return(quantiles != NULL && probabilities != NULL);
  }
//+------------------------------------------------------------------+
//| Read-only diagnostic access to the prepared FQF                  |
//| value/probability gradient buffers (cQuantile2/cSoftMax).        |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetDistributionGradientBuffers(CBufferFloat *&quantiles_gr, CBufferFloat *&probabilities_gr)
  {
   quantiles_gr = cQuantile2.getGradient();
   probabilities_gr = cSoftMax.getGradient();
   return(quantiles_gr != NULL && probabilities_gr != NULL);
  }
//+------------------------------------------------------------------+
//| Compute the AC-SRM actor head weight norm (weight_norm, L2)      |
//+------------------------------------------------------------------+
bool CNeuronACSRM::GetHeadWeightNorm(double &weight_norm)
  {
   weight_norm = 0.0;
   CBufferFloat *parts[6];
   parts[0] = cFraction.getWeights();
   parts[1] = cCosine.getWeights();
   parts[2] = cCosineEmbeding.getWeights();
   parts[3] = cQuantile0.getWeights();
   parts[4] = cQuantile1.getWeights();
   parts[5] = cQuantile2.getWeights();
   for(int k = 0; k < 6; k++)
     {
      CBufferFloat *w = parts[k];
      if(!w)
         continue;
      if(w.GetIndex() >= 0 && !w.BufferRead())
         return(false);
      double n = 0.0;
      for(int i = 0; i < w.Total(); i++)
         n += double(w[i]) * double(w[i]);
      weight_norm += n;
     }
   weight_norm = MathSqrt(weight_norm);
   return(MathIsValidNumber(weight_norm));
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronACSRM state from the file handle; the        |
//| transient rearrangement buffers are rebuilt on demand.           |
//+------------------------------------------------------------------+
bool CNeuronACSRM::Load(const int file_handle)
  {
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
   if(!CNeuronFQF::Load(file_handle))
      ReturnFalse;
   m_input_gradient_scratch.BufferFree();
   if(!m_input_gradient_scratch.BufferInit(cCosineEmbeding.Neurons(), 0) ||
      !m_input_gradient_scratch.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronACSRM GPU buffers (FQF head plus the          |
//| transient rearrangement state) to the given OpenCL context.      |
//+------------------------------------------------------------------+
void CNeuronACSRM::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronFQF::SetOpenCL(obj);
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
   CBufferFloat *transient[6];
   transient[0] = GetPointer(m_ordered_quantiles);
   transient[1] = GetPointer(m_ordered_probabilities);
   transient[2] = GetPointer(m_rearrange_order);
   transient[3] = GetPointer(m_ordered_quantiles_gradient);
   transient[4] = GetPointer(m_ordered_probabilities_gradient);
   transient[5] = GetPointer(m_rearrange_diagnostics);
   for(int k = 0; k < 6; k++)
     {
      if(!!OpenCL)
         transient[k].BufferCreate(OpenCL);
      else
         transient[k].BufferFree();
     }
   if(!!OpenCL)
      m_input_gradient_scratch.BufferCreate(OpenCL);
   else
      m_input_gradient_scratch.BufferFree();
  }
//+------------------------------------------------------------------+
//| Reports checked-transfer support for the CNeuronACSRM layer.     |
//+------------------------------------------------------------------+
bool CNeuronACSRM::SupportsOpenCLChecked(void) const
  {
//--- Type() pins the AC-SRM leaf; the protected FQF child tree is checked
//--- through the base contract without modifying the shared FQF classes.
   if(Type() != defNeuronACSRM)
      return(false);
   return(cFraction.SupportsOpenCLChecked() &&
          cSoftMax.SupportsOpenCLChecked() &&
          cCosine.SupportsOpenCLChecked() &&
          cCosineEmbeding.SupportsOpenCLChecked() &&
          cQuantile0.SupportsOpenCLChecked() &&
          cQuantile1.SupportsOpenCLChecked() &&
          cQuantile2.SupportsOpenCLChecked());
  }
//+------------------------------------------------------------------+
//| Transfers every CNeuronACSRM owned buffer to the target context. |
//+------------------------------------------------------------------+
bool CNeuronACSRM::SetOpenCLChecked(COpenCLMy *obj)
  {
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      ReturnFalse;
   m_distribution_gradient_ready = false;
   m_rearrangement_ready = false;
   if(!CNeuronBaseOCL::SetOpenCLChecked(obj))
      ReturnFalse;
   if(!cFraction.SetOpenCLChecked(obj) || !cSoftMax.SetOpenCLChecked(obj) ||
      !cCosine.SetOpenCLChecked(obj) || !cCosineEmbeding.SetOpenCLChecked(obj) ||
      !cQuantile0.SetOpenCLChecked(obj) || !cQuantile1.SetOpenCLChecked(obj) ||
      !cQuantile2.SetOpenCLChecked(obj))
      ReturnFalse;
//--- Move the transient scratch only when it is materialized.
   if(m_input_gradient_scratch.Total() > 0)
     {
      if(m_input_gradient_scratch.GetIndex() >= 0 &&
         m_input_gradient_scratch.GetOpenCL() != obj &&
         !m_input_gradient_scratch.BufferRead())
         ReturnFalse;
      if((m_input_gradient_scratch.GetOpenCL() != obj ||
          m_input_gradient_scratch.GetIndex() < 0) &&
         !m_input_gradient_scratch.BufferCreate(obj))
         ReturnFalse;
     }
//--- Move the transient rearrangement buffers when materialized.
   CBufferFloat *transient[6];
   transient[0] = GetPointer(m_ordered_quantiles);
   transient[1] = GetPointer(m_ordered_probabilities);
   transient[2] = GetPointer(m_rearrange_order);
   transient[3] = GetPointer(m_ordered_quantiles_gradient);
   transient[4] = GetPointer(m_ordered_probabilities_gradient);
   transient[5] = GetPointer(m_rearrange_diagnostics);
   for(int k = 0; k < 6; k++)
     {
      if(transient[k].Total() <= 0)
         continue;
      if(transient[k].GetIndex() >= 0 &&
         transient[k].GetOpenCL() != obj &&
         !transient[k].BufferRead())
         ReturnFalse;
      if((transient[k].GetOpenCL() != obj ||
          transient[k].GetIndex() < 0) &&
         !transient[k].BufferCreate(obj))
         ReturnFalse;
     }
   m_rearrangement_ready = false;
//--- Finalize the checked AC-SRM transfer.
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronSoftActorCritic layer and allocates its   |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronSoftActorCritic::Init(uint numOutputs,
                                  uint myIndex,
                                  COpenCLMy * open_cl,
                                  uint actions,
                                  uint quantiles,
                                  uint numInputs,
                                  ENUM_OPTIMIZATION optimization_type,
                                  uint batch
                                 )
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
   return(true);
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
   return(true);
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
   return(cQuantile0.CalcHiddenGradients((CObject*)GetPointer(cAlphas), cQuantile2.getOutput(), cQuantile2.getGradient()));
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
   return(cAlphas.UpdateInputWeights(cQuantile0.AsObject(), cQuantile2.getOutput()));
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
   return(cAlphas.WeightsUpdate(GetPointer(Source.cAlphas), tau));
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
   return(cAlphas.Save(file_handle));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the check data of the CNeuronSoftActorCritic.            |
//+------------------------------------------------------------------+
void CNeuronSoftActorCritic::getCheckData(vector<float> &output,
                                          vector<float> &grad,
                                          vector<float> &quantiles,
                                          vector<float> &probability,
                                          vector<float> &alphas
                                         )
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPLROCL layer and allocates its GPU        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronPLROCL::Init(uint numOutputs,
                         uint myIndex,
                         COpenCLMy * open_cl,
                         uint window_in,
                         uint units_count,
                         bool transpose,
                         ENUM_OPTIMIZATION optimization_type,
                         uint batch
                        )
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
//| Initializes the CNeuronMarketObserver layer and allocates its    |
//| GPU buffers.                                                     |
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronControlAgent layer.                                       |
//+------------------------------------------------------------------+
bool CNeuronControlAgent::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
                                             CBufferFloat* SecondInput,
                                             CBufferFloat* SecondGradient,
                                             ENUM_ACTIVATION SecondActivation = None
                                            )
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMASA      |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronMASA::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
                                     CBufferFloat* SecondInput,
                                     CBufferFloat* SecondGradient,
                                     ENUM_ACTIVATION SecondActivation = None
                                    )
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the norm layer of the CNeuronMASA.                          |
//+------------------------------------------------------------------+
bool CNeuronMASA::SetNormLayer(int NormLayer, CNeuronBatchNormOCL * normLayer)
  {
   return(cRevIN.Init(0, 1, OpenCL, cMarketObserver.Neurons(), NormLayer, normLayer));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronPLRMultiAgentsOCL state from the file        |
//| handle.                                                          |
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
   return(true);
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
//| Initializes the CNeuronCrossSectionalAnalysis layer and          |
//| allocates its GPU buffers.                                       |
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
   return(true);
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
   return(CNeuronMVMHAttentionMLKV::feedForward(cTransposeRCD.AsObject()));
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
   return(true);
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
   return(true);
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
//| Saves the CNeuronCrossSectionalAnalysis state to the file        |
//| handle.                                                          |
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronTemporalAnalysis feed-forward pass and writes    |
//| the layer output.                                                |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   return(CNeuronCrossSectionalAnalysis::feedForward(cTranspose.AsObject()));
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
   return(prevLayer.CalcHiddenGradients(cTranspose.AsObject()));
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronTemporalAnalysis trainable weights from the   |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronTemporalAnalysis::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return(CNeuronCrossSectionalAnalysis::updateInputWeights(cTranspose.AsObject()));
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPortfolioGenerator layer and allocates    |
//| its GPU buffers.                                                 |
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
   return(true);
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
   return(CNeuronBaseOCL::feedForward(cAssetTime[1].AsObject()));
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronPortfolioGenerator layer.                                 |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
                                                   CBufferFloat* SecondInput,
                                                   CBufferFloat* SecondGradient,
                                                   ENUM_ACTIVATION SecondActivation = None
                                                  )
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronPortfolioGenerator trainable weights from the |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronPortfolioGenerator::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return(CNeuronBaseOCL::updateInputWeights(cAssetTime[1].AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronPortfolioGenerator::feedForward(cTemporalAnalysis.AsObject(), cCrossSectionalAnalysis.getOutput()));
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
   return(true);
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
   return(CNeuronPortfolioGenerator::updateInputWeights(cTemporalAnalysis.AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronRelativeCrossAttention::feedForward(cMamba.AsObject(), cLSTM.getOutput()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronLowLevelReflection layer and allocates    |
//| its GPU buffers.                                                 |
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
   return(true);
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
   return(CNeuronMemory::feedForward(cCrossAttention[1].AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronMemory::feedForward(cActionResult.AsObject()));
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHighLevelReflection trainable weights from    |
//| the accumulated gradients.                                       |
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
   return(true);
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
   return(true);
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
   return(true);
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
//| Clears the CNeuronHighLevelReflection execution state and its    |
//| GPU buffers.                                                     |
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMoreLessEqual layer.                                      |
//+------------------------------------------------------------------+
bool CNeuronMoreLessEqual::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getGradient())
      ReturnFalse;
   return(NeuronOCL.getGradient().Fill(0));
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
   return(true);
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
   return(CNeuronRelativeCrossAttention::feedForward(cHighLevelReflection.AsObject(), cMarketToTools.getOutput()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronRelativeCrossAttention::feedForward(cCrossAttention.AsObject(), cDistilMemory.getOutput()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronRelativeCrossAttention::feedForward(cActionsMemory.AsObject(), cStatesMemory.getOutput()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CNeuronFinConAgent::feedForward(cConcatenatedAgents.AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMacroHFTHyperAgent layer and allocates    |
//| its GPU buffers.                                                 |
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
   return(true);
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
   return(CNeuronSoftMaxOCL::feedForward(cMLP[1].AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(CResidualConv::feedForward(cCrossAttention.AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMacroHFTvsRiskManager GPU buffers to the      |
//| given OpenCL context.                                            |
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHidformerTSAgent feed-forward pass and writes    |
//| the layer output.                                                |
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
   return(CResidualConv::feedForward(cActionToState.AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHidformerFreqAgent layer and allocates    |
//| its GPU buffers.                                                 |
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
   return(true);
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
   return(CResidualConv::feedForward(cActionToState.AsObject()));
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
   return(true);
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
//| Initializes the CNeuronHiSSDLowLevelControler layer and          |
//| allocates its GPU buffers.                                       |
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
   return(true);
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
   return(CNeuronConvOCL::feedForward(neuron));
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
   return(true);
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
   return(CNeuronConvOCL::updateInputWeights(neuron));
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
   return(CNeuronConvOCL::WeightsUpdate(source, tau));
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHiSSDLowLevelControler state to the file        |
//| handle.                                                          |
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
   return(true);
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
   return(true);
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
//| DPCC checked sizes and stable profile hashing                    |
//+------------------------------------------------------------------+
bool DPCCRolloutProduct(const ulong first, const ulong second, ulong &result)
  {
   if(first == 0 || second == 0 || first > (ulong)INT_MAX / second)
      return(false);
   result = first * second;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function DPCCRolloutHashWord                                     |
//+------------------------------------------------------------------+
void DPCCRolloutHashWord(const ulong value, ulong &hash)
  {
   for(uint byte_index = 0; byte_index < 8; byte_index++)
      hash = (hash ^ ((value >> (byte_index * 8)) & 255)) * ulong(1099511628211);
  }
//+------------------------------------------------------------------+
//| Function DPCCRolloutHashDouble                                   |
//+------------------------------------------------------------------+
void DPCCRolloutHashDouble(const double value, ulong &hash)
  {
   UDPCCDoubleBits raw;
   raw.value = value;
   DPCCRolloutHashWord(raw.bits, hash);
  }
//+------------------------------------------------------------------+
//| Function DPCCBuildRolloutLayout                                  |
//+------------------------------------------------------------------+
bool DPCCBuildRolloutLayout(const SDPCCShape &shape, SDPCCLayout &layout)
  {
   const uint dimensions[] = {shape.K, shape.V, shape.H, shape.D, shape.M, shape.J, shape.E,
                              shape.PlanBatch, shape.Embedding, shape.Heads, shape.KeyDimension
                             };
   for(uint i = 0; i < (uint)ArraySize(dimensions); i++)
      if(dimensions[i] == 0 || dimensions[i] > 16777216)
         return(false);
   if(shape.K > 64 || shape.J > 64 || shape.NActions != 6 || shape.StateSize != 13 || shape.Depth != 0)
      return(false);
   ulong kh, mk, width, temporary;
   if(!DPCCRolloutProduct(shape.K, shape.H, kh))
      return(false);
   if(!DPCCRolloutProduct(shape.M, shape.K, mk))
      return(false);
   if(!DPCCRolloutProduct(shape.Heads, shape.KeyDimension, width))
      return(false);
   if(!DPCCRolloutProduct(mk, (ulong)shape.H + 1, temporary))
      return(false);
   if(!DPCCRolloutProduct(temporary, 24, temporary))
      return(false);
   if(!DPCCRolloutProduct(mk, (ulong)shape.H * 4, temporary))
      return(false);
   if(!DPCCRolloutProduct((ulong)shape.M * 9, shape.K, temporary))
      return(false);
   if(!DPCCRolloutProduct(temporary, ((ulong)shape.H + 1) * 24, temporary))
      return(false);
   ZeroMemory(layout);
   layout.version = DPCC_CONTRACT_VERSION;
   if(!DPCCRolloutProduct(kh, 4, layout.lengths[0]))
      return(false);
   layout.lengths[1] = 24;
   layout.lengths[2] = 48;
   layout.lengths[3] = 24;
   if(!DPCCRolloutProduct(kh, 24, layout.lengths[4]))
      return(false);
   layout.lengths[5] = shape.H;
   if(!DPCCRolloutProduct(kh, (ulong)shape.E * 6, layout.lengths[6]))
      return(false);
   layout.lengths[7] = shape.M;
   for(uint section = 0; section < DPCC_CONTEXT_SECTIONS; section++)
     {
      layout.offsets[section] = layout.total;
      if(layout.lengths[section] > (ulong)INT_MAX - layout.total)
         return(false);
      layout.total += layout.lengths[section];
     }
   return(layout.total <= INT_MAX);
  }
//+------------------------------------------------------------------+
//| Function DPCCSameShape                                           |
//+------------------------------------------------------------------+
bool DPCCSameShape(const SDPCCShape &first, const SDPCCShape &second)
  {
   return(first.K == second.K && first.V == second.V && first.H == second.H && first.D == second.D &&
          first.M == second.M && first.J == second.J && first.E == second.E &&
          first.PlanBatch == second.PlanBatch && first.NActions == second.NActions &&
          first.StateSize == second.StateSize && first.Embedding == second.Embedding &&
          first.Heads == second.Heads && first.KeyDimension == second.KeyDimension && first.Depth == second.Depth);
  }
//+------------------------------------------------------------------+
//| Function DPCCReadFinancialDescriptor                             |
//+------------------------------------------------------------------+
bool DPCCReadFinancialDescriptor(CLayerDescription *desc, SDPCCShape &shape, const int type, const bool plans_output)
  {
   if(!desc || desc.type != type || ArraySize(desc.windows) != 9 ||
      ArraySize(desc.units) != 5 || ArraySize(desc.heads) != 1 || desc.windows[0] != DPCC_CONTRACT_VERSION ||
      desc.activation != None || desc.layers != 0 || desc.window != 6 || desc.window_out != 13 ||
      desc.batch == 0 || (desc.optimization != SGD && desc.optimization != ADAM && desc.optimization != ADAM_MINI))
      return(false);
   shape.K = desc.windows[1];
   shape.V = desc.windows[2];
   shape.H = desc.windows[3];
   shape.D = desc.windows[4];
   shape.M = desc.windows[5];
   shape.J = desc.windows[6];
   shape.E = desc.windows[7];
   shape.PlanBatch = desc.windows[8];
   shape.Embedding = desc.units[0];
   shape.Heads = desc.units[1];
   shape.KeyDimension = desc.units[2];
   shape.NActions = desc.units[3];
   shape.StateSize = desc.units[4];
   shape.Depth = desc.layers;
   SDPCCLayout checked_layout;
   if(!DPCCBuildRolloutLayout(shape, checked_layout))
      return(false);
   return(desc.count == (plans_output ? (ulong)shape.M * shape.H * 6 : (ulong)shape.M * shape.K) && desc.variables == shape.V && desc.step == shape.H &&
          desc.heads[0] == shape.Heads && desc.window == shape.NActions && desc.window_out == shape.StateSize);
  }
//+------------------------------------------------------------------+
//| Function DPCCValidateFixedConfig                                 |
//+------------------------------------------------------------------+
bool DPCCValidateFixedConfig(const SDPCCShape &shape, const SDPCCConfig &config)
  {
   if(!DPCCSameShape(shape, config.shape) || config.contract_version != DPCC_CONTRACT_VERSION ||
      config.binary_layer_version != DPCC_BINARY_LAYER_VERSION ||
      config.margin_fx_profile > DPCC_MARGIN_HISTORICAL_CLOSE_V1 ||
      (config.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 && config.account_currency_digits != 0) ||
      (config.margin_fx_profile != DPCC_MARGIN_EXTERNAL_V2 &&
       (config.account_currency_digits > 8 || config.price_basis != DPCC_PRICE_BID ||
        config.quote_adapter != DPCC_QUOTE_BID_IDENTITY_V1)) ||
      (config.mode != DPCC_P0 && config.mode != DPCC_P1 && config.mode != DPCC_P2) ||
      config.projection_max_iterations == 0 || config.projection_max_iterations > 16 ||
      config.projection_max_evaluations == 0 || config.projection_max_evaluations > 160 ||
      config.projection_lookback_steps > shape.H || config.risk_type < defSRM_MEAN || config.risk_type > defSRM_DUAL_POWER ||
      config.price_basis > DPCC_PRICE_LAST || config.quote_adapter != config.price_basis ||
      config.event_model != DPCC_EVENT_LOCAL_TWO_PATH_V1 ||
      config.normalization_mode != DPCC_NORMALIZATION_CAUSAL_AFFINE_V2 ||
      config.loss_coordinates != DPCC_LOSS_NORMALIZED || config.randomness_version != DPCC_RANDOM_INDEXED_NORMAL_V1)
      return(false);
   const double nonnegative[] = {config.projection_improvement_atol, config.projection_distance_tie_atol,
                                 config.max_quote_age_seconds, config.max_last_age_seconds
                                };
   for(uint i = 0; i < (uint)ArraySize(nonnegative); i++)
      if(!MathIsValidNumber(nonnegative[i]) || nonnegative[i] < 0 ||
         !MathIsValidNumber((float)nonnegative[i]))
         return(false);
   const double scales[] = {config.volume_scale, config.price_scale, config.money_scale};
   for(uint i = 0; i < (uint)ArraySize(scales); i++)
      if(!MathIsValidNumber(scales[i]) || scales[i] <= 0 ||
         !MathIsValidNumber((float)scales[i]) || (float)scales[i] <= 0)
         return(false);
   if(!MathIsValidNumber(config.gamma) || config.gamma <= 0 || config.gamma > 1 ||
      !MathIsValidNumber(config.risk_parameter) || !MathIsValidNumber(config.mean_weight) ||
      config.mean_weight < 0 || config.mean_weight > 1 || !MathIsValidNumber(config.probability_tolerance) ||
      config.probability_tolerance < 0 || config.probability_tolerance >= 1)
      return(false);
   if((config.risk_type == defSRM_CVAR || config.risk_type == defSRM_MEAN_CVAR) &&
      (config.risk_parameter <= 0 || config.risk_parameter > 1))
      return(false);
   if((config.risk_type == defSRM_EXPONENTIAL || config.risk_type == defSRM_DUAL_POWER) &&
      config.risk_parameter <= 0)
      return(false);
   for(uint action = 0; action < 6; action++)
      if(!MathIsValidNumber(config.action_bias[action]) || !MathIsValidNumber(config.action_scale[action]) ||
         config.action_scale[action] <= 0)
         return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function DPCCFixedSignature                                      |
//+------------------------------------------------------------------+
ulong DPCCFixedSignature(const SDPCCConfig &config, const uint type)
  {
   ulong hash = ulong(14695981039346656037);
   const uint integers[] = {type, config.contract_version, config.binary_layer_version,
                            config.shape.K, config.shape.V, config.shape.H, config.shape.D,
                            config.shape.M, config.shape.J, config.shape.E, config.shape.PlanBatch,
                            config.shape.NActions, config.shape.StateSize, config.shape.Embedding,
                            config.shape.Heads, config.shape.KeyDimension, config.shape.Depth,
                            config.projection_max_iterations, config.projection_max_evaluations,
                            config.projection_lookback_steps, config.price_basis, config.quote_adapter,
                            config.event_model, (uint)config.risk_type, config.normalization_mode,
                            config.loss_coordinates, config.randomness_version
                           };
   for(uint i = 0; i < (uint)ArraySize(integers); i++)
      DPCCRolloutHashWord(integers[i], hash);
   const double reals[] = {config.projection_improvement_atol, config.projection_distance_tie_atol,
                           config.volume_scale, config.price_scale, config.money_scale,
                           config.max_quote_age_seconds, config.max_last_age_seconds,
                           (double)config.risk_parameter, (double)config.mean_weight, (double)config.gamma,
                           (double)config.probability_tolerance
                          };
   for(uint i = 0; i < (uint)ArraySize(reals); i++)
      DPCCRolloutHashDouble(reals[i], hash);
   for(uint action = 0; action < 6; action++)
     {
      DPCCRolloutHashDouble(config.action_bias[action], hash);
      DPCCRolloutHashDouble(config.action_scale[action], hash);
     }
//--- Legacy profile hashes exactly the original words. Financial profile1
//--- binds both its selected semantics and actual account money precision.
   if(config.margin_fx_profile != DPCC_MARGIN_EXTERNAL_V2)
     {
      DPCCRolloutHashWord(0x46585333, hash);
      DPCCRolloutHashWord(config.margin_fx_profile, hash);
      DPCCRolloutHashWord(config.account_currency_digits, hash);
     }
   return(hash);
  }
//+------------------------------------------------------------------+
//| Constructs and releases exclusively owned scratch                |
//+------------------------------------------------------------------+
CNeuronScenarioRolloutOCL::CNeuronScenarioRolloutOCL(void)
  {
   ZeroMemory(m_shape);
   ZeroMemory(m_layout);
   ZeroMemory(m_config);
   m_phase = DPCC_LAYER_CONSTRUCTED;
   m_status = DPCC_CONTRACT_MISMATCH;
   m_configured = false;
   m_fixed = false;
   m_public_valid = false;
   m_states = NULL;
   m_validity = NULL;
   m_violations = NULL;
   m_diagnostics = NULL;
   m_return_scratch = NULL;
   m_state_scratch = NULL;
   m_violation_scratch = NULL;
   m_validity_scratch = NULL;
   m_diagnostic_scratch = NULL;
   m_system = NULL;
   m_aggregate = NULL;
   m_slice_active = NULL;
   m_batch_validity = NULL;
   m_batch_diagnostics = NULL;
   m_open_context = NULL;
   m_open_state = NULL;
   m_open_risk = NULL;
   m_open_generation = 0;
   m_open_step = 0;
   m_open_horizon = 0;
   m_open_pending = false;
   m_open_actionable = false;
   m_open_observed = false;
   bTrain = false;
  }
//+------------------------------------------------------------------+
//| Function ~CNeuronScenarioRolloutOCL                              |
//+------------------------------------------------------------------+
CNeuronScenarioRolloutOCL::~CNeuronScenarioRolloutOCL(void)
  {
   DeleteObj(m_states);
   DeleteObj(m_validity);
   DeleteObj(m_violations);
   DeleteObj(m_diagnostics);
   DeleteObj(m_return_scratch);
   DeleteObj(m_state_scratch);
   DeleteObj(m_violation_scratch);
   DeleteObj(m_validity_scratch);
   DeleteObj(m_diagnostic_scratch);
   DeleteObj(m_system);
   DeleteObj(m_aggregate);
   DeleteObj(m_slice_active);
   DeleteObj(m_batch_validity);
   DeleteObj(m_batch_diagnostics);
   DeleteObj(m_open_context);
   DeleteObj(m_open_state);
   DeleteObj(m_open_risk);
  }
//+------------------------------------------------------------------+
//| Records a refused operation without publishing stale state       |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Fail(const ENUM_DPCC_STATUS status, const bool fatal)
  {
   m_status = status;
   if(fatal)
     {
      m_phase = DPCC_LAYER_FAILED;
      m_public_valid = false;
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::EnsureBuffer                 |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EnsureBuffer(CBufferFloat *&buffer, const uint count, const float value)
  {
   if(count == 0 || count > INT_MAX || !OpenCL)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!buffer)
      buffer = new CBufferFloat();
   if(!buffer)
      return(Fail(DPCC_OPENCL_ERROR));
   if(buffer.Total() == (int)count && buffer.GetOpenCL() == OpenCL && buffer.GetIndex() >= 0 &&
      OpenCL.GetBufferSize(buffer.GetIndex()) == (ulong)count * sizeof(float))
      return(true);
   if(buffer.GetIndex() >= 0 && !buffer.BufferFree())
      return(Fail(DPCC_OPENCL_ERROR));
   if(!buffer.BufferInit(count, value))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!buffer.BufferCreate(OpenCL))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates exact tensor geometry and constructs SP-04 layout      |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::BuildLayout(const SDPCCShape &shape, SDPCCLayout &layout) const
  {
   return(DPCCBuildRolloutLayout(shape, layout));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::SameShape                    |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::SameShape(const SDPCCShape &first, const SDPCCShape &second) const
  {
   return(DPCCSameShape(first, second));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ReadDescriptor               |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ReadDescriptor(CLayerDescription *desc, SDPCCShape &shape) const
  {
   return(DPCCReadFinancialDescriptor(desc, shape, defNeuronScenarioRolloutOCL, false));
  }
//+------------------------------------------------------------------+
//| Initializes a parameter-free CNet layer using exact descriptor   |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                     CLayerDescription *desc)
  {
   m_public_valid = false;
   m_open_pending = false;
   m_open_actionable = false;
   m_open_observed = false;
   SDPCCShape candidate;
   if(!open_cl || !ReadDescriptor(desc, candidate))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
//--- Init discards transient state and releases each allocation through its actual owner.
   CBufferFloat *old_buffers[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                                  m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                                  m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                                  m_slice_active, m_batch_validity, m_batch_diagnostics,
                                  m_open_context, m_open_state, m_open_risk
                                 };
   if(OpenCL != open_cl)
      for(uint i = 0; i < (uint)ArraySize(old_buffers); i++)
         if(CheckPointer(old_buffers[i]) != POINTER_INVALID && old_buffers[i].GetIndex() >= 0 && !old_buffers[i].BufferFree())
            return(Fail(DPCC_OPENCL_ERROR, true));
   OpenCL = open_cl;
   m_shape = candidate;
   if(!BuildLayout(m_shape, m_layout))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
//--- numOutputs represents graph connections, not financial-layer parameters.
   if(!CNeuronBaseOCL::Init(0, myIndex, open_cl, desc.count, desc.optimization, desc.batch))
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_myIndex = (int)myIndex;
   activation = None;
   bTrain = false;
   m_configured = false;
   m_fixed = false;
   ZeroMemory(m_config);
   if(!EnsureBuffer(m_states, m_shape.M * m_shape.K * (m_shape.H + 1) * 24))
      return(Fail(m_status, true));
   if(!EnsureBuffer(m_validity, m_shape.M))
      return(Fail(m_status, true));
   if(!EnsureBuffer(m_violations, m_shape.M * m_shape.K * 8))
      return(Fail(m_status, true));
   if(!EnsureBuffer(m_diagnostics, m_shape.M * m_shape.K * m_shape.H * 4))
      return(Fail(m_status, true));
   if(!EnsureBuffer(m_slice_active, 1, 1))
      return(Fail(m_status, true));
   SDPCCShape phase_shape = m_shape;
   phase_shape.H = 1;
   phase_shape.M = 1;
   SDPCCLayout phase_layout;
   if(!BuildLayout(phase_shape, phase_layout) ||
      !EnsureBuffer(m_open_context, (uint)phase_layout.total) ||
      !EnsureBuffer(m_open_state, m_shape.K * DPCC_CONTINUATION_FLOATS) ||
      !EnsureBuffer(m_open_risk, m_shape.K * DPCC_NATIVE_PENDING_PHASE_FLOATS))
      return(Fail(m_status, true));
   if(!PrepareScratch(m_shape.M))
      return(Fail(m_status, true));
   m_phase = DPCC_LAYER_SHAPED;
   m_status = DPCC_OK;
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Validates all fixed profile fields without implicit defaults     |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ValidateConfig(const SDPCCConfig &config) const
  {
   return(DPCCValidateFixedConfig(m_shape, config));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::Configure                    |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Configure(const SDPCCConfig &config)
  {
   m_status = DPCC_OK;
   if(m_open_pending)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED || !ValidateConfig(config))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_fixed && ConfigSignature(config) != ConfigSignature(m_config))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_config = config;
   m_configured = true;
   m_phase = DPCC_LAYER_CONFIGURED;
   m_public_valid = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates ownership, byte sizes, and actual device aliases       |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ValidateBuffer(CBufferFloat *buffer, const ulong count, const bool exact) const
  {
   if(!buffer || count == 0 || count > INT_MAX || buffer.GetIndex() < 0 || buffer.GetOpenCL() != OpenCL ||
      (exact ? (ulong)buffer.Total() != count : (ulong)buffer.Total() < count))
      return(false);
   return(OpenCL.GetBufferSize(buffer.GetIndex()) == (ulong)buffer.Total() * sizeof(float));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::SharesDevice                 |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::SharesDevice(CBufferFloat *first, CBufferFloat *second) const
  {
   return(first && second && (first == second ||
                              (first.GetIndex() >= 0 && first.GetIndex() == second.GetIndex() &&
                               first.GetOpenCL() == second.GetOpenCL())));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ValidateInputAliases         |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ValidateInputAliases(CBufferFloat *plans, CBufferFloat *active,
      CBufferFloat *context) const
  {
   CBufferFloat *inputs[] = {plans, active, context};
   CBufferFloat *owned[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                            m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                            m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                            m_slice_active, m_batch_validity, m_batch_diagnostics
                           };
   for(uint i = 0; i < (uint)ArraySize(inputs); i++)
      for(uint j = 0; j < (uint)ArraySize(owned); j++)
         if(SharesDevice(inputs[i], owned[j]) && !(i == 1 && inputs[i] == m_slice_active && owned[j] == m_slice_active))
            return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ValidateAliases              |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ValidateAliases(CBufferFloat *plans, CBufferFloat *active, CBufferFloat *context,
      CBufferFloat *returns_out, CBufferFloat *states_out,
      CBufferFloat *violations_out) const
  {
   CBufferFloat *outputs[] = {returns_out, states_out, violations_out};
   CBufferFloat *inputs[] = {plans, active, context, Output, PrevOutput, Gradient, m_states, m_validity,
                             m_violations, m_diagnostics, m_return_scratch, m_state_scratch,
                             m_violation_scratch, m_validity_scratch, m_diagnostic_scratch,
                             m_system, m_aggregate, m_slice_active, m_batch_validity, m_batch_diagnostics,
                             m_open_context, m_open_state, m_open_risk
                            };
   for(uint i = 0; i < 3; i++)
     {
      for(uint j = 0; j < (uint)ArraySize(inputs); j++)
         if(SharesDevice(outputs[i], inputs[j]))
            return(false);
      for(uint j = 0; j < i; j++)
         if(SharesDevice(outputs[i], outputs[j]))
            return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ValidateContext              |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ValidateContext(CBufferFloat *context) const
  {
   return(ValidateBuffer(context, m_layout.total));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::PrepareScratch               |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::PrepareScratch(const uint slots)
  {
   const ulong mk = (ulong)m_shape.M * m_shape.K;
   const ulong mp = (ulong)m_shape.M * 9;
   const ulong mpk = mp * m_shape.K;
   if(slots == 0 || slots > mp || mpk * ((ulong)m_shape.H + 1) * 24 > INT_MAX ||
      mpk * m_shape.H * 4 > INT_MAX)
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- Both fixed generations are allocated by Init before financial execution.
   if(!EnsureBuffer(m_return_scratch, (uint)mk) ||
      !EnsureBuffer(m_state_scratch, (uint)(mk * (m_shape.H + 1) * 24)) ||
      !EnsureBuffer(m_violation_scratch, (uint)(mk * 8)) ||
      !EnsureBuffer(m_validity_scratch, m_shape.M) ||
      !EnsureBuffer(m_diagnostic_scratch, (uint)(mk * m_shape.H * 4)) ||
      !EnsureBuffer(m_batch_validity, (uint)mp) ||
      !EnsureBuffer(m_batch_diagnostics, (uint)(mpk * m_shape.H * 4)) ||
      !EnsureBuffer(m_system, (uint)mpk) || !EnsureBuffer(m_aggregate, 1))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the common financial GPU engine for full, slice and batch   |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::RolloutPlans(CBufferFloat *plans, const uint plan_offset, const uint slots,
      CBufferFloat *active, const uint active_offset, CBufferFloat *context,
      CBufferFloat *returns_out, CBufferFloat *states_out,
      CBufferFloat *violations_out, CBufferFloat *diagnostics_out,
      CBufferFloat *continuation, uint episode_start,
      uint episode_horizon, float training_floor,
      CBufferFloat *step_rewards, CBufferFloat *executed_mask)
  {
   const int kernel = def_k_DPCCRollout;
   CBufferFloat *buffers[] = {plans, context, active, returns_out, states_out, violations_out,
                              m_system, diagnostics_out
                             };
   for(uint arg = 0; arg < 8; arg++)
      if(!OpenCL.SetArgumentBuffer(kernel, arg, buffers[arg].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint integers[] = {plan_offset, slots, m_shape.K, m_shape.H, m_shape.E, m_shape.M,
                            (uint)m_layout.offsets[1], (uint)m_layout.offsets[2],
                            (uint)m_layout.offsets[3], (uint)m_layout.offsets[4],
                            (uint)m_layout.offsets[5], (uint)m_layout.offsets[6],
                            (uint)m_layout.offsets[7], (uint)m_layout.total
                           };
   for(uint arg = 0; arg < (uint)ArraySize(integers); arg++)
      if(!OpenCL.SetArgument(kernel, arg + 8, integers[arg]))
         return(Fail(DPCC_OPENCL_ERROR));
   const float reals[] = {m_config.gamma, (float)m_config.volume_scale, (float)m_config.price_scale,
                          (float)m_config.max_quote_age_seconds, (float)m_config.max_last_age_seconds
                         };
   for(uint arg = 0; arg < (uint)ArraySize(reals); arg++)
      if(!OpenCL.SetArgument(kernel, arg + 22, reals[arg]))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 27, active_offset))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 28, m_config.price_basis))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 29, m_config.quote_adapter))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 30, (float)m_config.money_scale) ||
      !OpenCL.SetArgument(kernel, 31, (uint)(m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? 4 :
                          (m_config.margin_fx_profile == DPCC_MARGIN_FOREX_SPOT_V1 ? 3 : 2))) ||
      !OpenCL.SetArgument(kernel, 32, m_config.account_currency_digits))
      return(Fail(DPCC_OPENCL_ERROR));
   CBufferFloat *state_source = continuation ? continuation : context;
   if(!OpenCL.SetArgumentBuffer(kernel, 33, state_source.GetIndex()) ||
      !OpenCL.SetArgument(kernel, 34, (uint)(continuation != NULL)) ||
      !OpenCL.SetArgument(kernel, 35, episode_start) ||
      !OpenCL.SetArgument(kernel, 36, continuation ? episode_horizon : m_shape.H) ||
      !OpenCL.SetArgument(kernel, 37, training_floor))
      return(Fail(DPCC_OPENCL_ERROR));
   CBufferFloat *reward_destination = step_rewards ? step_rewards : context;
   CBufferFloat *mask_destination = executed_mask ? executed_mask : context;
   if(!OpenCL.SetArgumentBuffer(kernel, 38, reward_destination.GetIndex()) ||
      !OpenCL.SetArgumentBuffer(kernel, 39, mask_destination.GetIndex()))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0, 0};
   const uint global[] = {m_shape.K, slots};
   if(!OpenCL.Execute(kernel, 2, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ReduceValidity               |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::ReduceValidity(CBufferFloat *active, const uint active_offset,
      const uint slots, CBufferFloat *violations_out, CBufferFloat *validity_out)
  {
   const int kernel = def_k_DPCCRolloutReduce;
   CBufferFloat *buffers[] = {m_system, violations_out, active, validity_out, m_aggregate};
   for(uint arg = 0; arg < 5; arg++)
      if(!OpenCL.SetArgumentBuffer(kernel, arg, buffers[arg].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 5, m_shape.K))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 6, slots))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 7, active_offset))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0};
   const uint global[] = {slots + 1};
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
//--- Only the single GPU system verdict crosses the host boundary.
   if(!m_aggregate.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   const float status = m_aggregate[0];
   if(status == 3)
     {
      if(violations_out.BufferRead())
         for(uint pair = 0; pair < slots * m_shape.K; pair++)
            if(violations_out[pair * 8 + 5] == (float)DPCC_INVALID_INPUT)
              {
               const uint failure_line = (uint)violations_out[pair * 8 + 6];
               if(failure_line > 0)
                  PrintFormat("DPCC_ROLLOUT_CONTEXT_INVALID check_line=%u pair=%u", failure_line, pair);
               break;
              }
      return(Fail(DPCC_INVALID_INPUT));
     }
//--- Status 3 has already returned above.
   if(status == 5)
      return(Fail(DPCC_NUMERIC_ERROR));
   if(status != 0)
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluates CNet inputs and publishes all public buffers together  |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::feedForward(CNeuronBaseOCL *source)
  {
   m_public_valid = false;
   if(m_diagnostics)
      m_diagnostics.Fill(0);
   return(Fail(DPCC_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::feedForward                  |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::feedForward(CNeuronBaseOCL *source, CBufferFloat *context)
  {
   m_status = DPCC_OK;
   m_public_valid = false;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || !source ||
      !ValidateBuffer(source.getOutput(), (ulong)m_shape.M * m_shape.H * 6) || !ValidateContext(context) ||
      !ValidateInputAliases(source.getOutput(), context, context))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_diagnostics && !m_diagnostics.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!PrepareScratch(m_shape.M))
      return(false);
   m_fixed = true;
   if(!RolloutPlans(source.getOutput(), 0, m_shape.M, context, (uint)m_layout.offsets[7], context,
                    m_return_scratch, m_state_scratch, m_violation_scratch, m_diagnostic_scratch))
      return(false);
   if(!ReduceValidity(context, (uint)m_layout.offsets[7], m_shape.M, m_violation_scratch, m_validity_scratch))
      return(false);
//--- Pointer publication cannot fail halfway through a GPU copy sequence.
   CBufferFloat *temporary = Output;
   Output = m_return_scratch;
   m_return_scratch = temporary;
   temporary = m_states;
   m_states = m_state_scratch;
   m_state_scratch = temporary;
   temporary = m_validity;
   m_validity = m_validity_scratch;
   m_validity_scratch = temporary;
   temporary = m_violations;
   m_violations = m_violation_scratch;
   m_violation_scratch = temporary;
   temporary = m_diagnostics;
   m_diagnostics = m_diagnostic_scratch;
   m_diagnostic_scratch = temporary;
   m_public_valid = true;
   m_phase = DPCC_LAYER_READY;
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluates caller-owned proposal workspace without public         |
//| mutation                                                         |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EvaluatePlanBatch(CBufferFloat *plans, uint plan_offset, uint slot_capacity,
      CBufferFloat *slot_active, CBufferFloat *rollout_context,
      CBufferFloat *returns_out, CBufferFloat *states_out,
      CBufferFloat *violations_out)
  {
   m_status = DPCC_OK;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || slot_capacity == 0 ||
      slot_capacity > (ulong)m_shape.M * 9)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong plan_end = (ulong)plan_offset + (ulong)slot_capacity * m_shape.H * 6;
   const ulong mk = (ulong)slot_capacity * m_shape.K;
   if(!ValidateBuffer(plans, plan_end, false) || !ValidateBuffer(slot_active, slot_capacity) ||
      !ValidateContext(rollout_context) || !ValidateBuffer(returns_out, mk) ||
      !ValidateBuffer(states_out, mk * (m_shape.H + 1) * 24) || !ValidateBuffer(violations_out, mk * 8) ||
      !ValidateInputAliases(plans, slot_active, rollout_context) ||
      !ValidateAliases(plans, slot_active, rollout_context, returns_out, states_out, violations_out))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!PrepareScratch(slot_capacity))
      return(false);
   m_fixed = true;
   if(!RolloutPlans(plans, plan_offset, slot_capacity, slot_active, 0, rollout_context,
                    returns_out, states_out, violations_out, m_batch_diagnostics))
      return(false);
   return(ReduceValidity(slot_active, 0, slot_capacity, violations_out, m_batch_validity));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::EvaluatePlanSlice            |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EvaluatePlanSlice(CBufferFloat *plans, uint plan_offset,
      CBufferFloat *rollout_context, CBufferFloat *returns_out,
      CBufferFloat *states_out, CBufferFloat *violations_out)
  {
   return(EvaluatePlanBatch(plans, plan_offset, 1, m_slice_active, rollout_context,
                            returns_out, states_out, violations_out));
  }
//+------------------------------------------------------------------+
//| Batch exact native continuations without mutating public outputs |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EvaluateContinuedPlanBatch(CBufferFloat *plans, uint plan_offset,
      uint slot_capacity, CBufferFloat *slot_active,
      CBufferFloat *rollout_context, CBufferFloat *continuation,
      uint episode_start, uint episode_horizon, float training_floor,
      CBufferFloat *returns_out, CBufferFloat *states_out,
      CBufferFloat *violations_out, CBufferFloat *diagnostics_out,
      CBufferFloat *step_rewards, CBufferFloat *executed_mask)
  {
   m_status = DPCC_OK;
   const ulong pairs = (ulong)slot_capacity * m_shape.K;
   const ulong proposal_capacity = (ulong)m_shape.M * 9;
   const ulong history_capacity = 1 + (ulong)m_shape.H * 12;
   const ulong capacity = (proposal_capacity > history_capacity ? proposal_capacity : history_capacity);
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || m_open_pending || slot_capacity == 0 ||
      slot_capacity > capacity || pairs > (ulong)INT_MAX ||
      episode_horizon == 0 || episode_horizon > 16777216 ||
      (ulong)episode_start + m_shape.H > episode_horizon || !MathIsValidNumber(training_floor) || training_floor < 0 ||
      !ValidateBuffer(plans, (ulong)plan_offset + (ulong)slot_capacity * m_shape.H * 6, false) ||
      !ValidateBuffer(slot_active, slot_capacity) || !ValidateContext(rollout_context) ||
      !ValidateBuffer(continuation, pairs * DPCC_CONTINUATION_FLOATS) || OwnsBufferOrDeviceSlot(continuation) ||
      !ValidateBuffer(returns_out, pairs) || !ValidateBuffer(states_out, pairs * (m_shape.H + 1) * 24) ||
      !ValidateBuffer(violations_out, pairs * 8) || !ValidateBuffer(diagnostics_out, pairs * m_shape.H * 4) ||
      !ValidateBuffer(step_rewards, pairs * m_shape.H) || !ValidateBuffer(executed_mask, pairs * m_shape.H) ||
      !ValidateInputAliases(plans, slot_active, rollout_context) ||
      !ValidateInputAliases(continuation, NULL, NULL) ||
      !ValidateAliases(plans, slot_active, rollout_context, returns_out, states_out, violations_out) ||
      !ValidateAliases(continuation, NULL, NULL, returns_out, states_out, violations_out) ||
      !ValidateAliases(plans, slot_active, rollout_context, diagnostics_out, step_rewards, executed_mask))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *inputs[] = {plans, slot_active, rollout_context, continuation};
   CBufferFloat *outputs[] = {returns_out, states_out, violations_out, diagnostics_out, step_rewards, executed_mask};
   for(uint i = 0; i < (uint)ArraySize(inputs); i++)
      for(uint j = 0; j < i; j++)
         if(SharesDevice(inputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < (uint)ArraySize(outputs); i++)
     {
      for(uint j = 0; j < (uint)ArraySize(inputs); j++)
         if(SharesDevice(outputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(outputs[i], outputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
//--- Reuse continued-batch reduction buffers without resizing the proposal workspace.
   if(!ValidateBuffer(m_system, pairs, false) && !EnsureBuffer(m_system, (uint)pairs))
      return(false);
   if(!ValidateBuffer(m_batch_validity, slot_capacity, false) &&
      !EnsureBuffer(m_batch_validity, slot_capacity))
      return(false);
   if(!ValidateBuffer(m_aggregate, 1) && !EnsureBuffer(m_aggregate, 1))
      return(false);
   m_fixed = true;
   if(!RolloutPlans(plans, plan_offset, slot_capacity, slot_active, 0, rollout_context,
                    returns_out, states_out, violations_out, diagnostics_out,
                    continuation, episode_start, episode_horizon, training_floor, step_rewards, executed_mask))
      return(false);
   return(ReduceValidity(slot_active, 0, slot_capacity, violations_out, m_batch_validity));
  }
//+------------------------------------------------------------------+
//| Runs one native financial transition with exact continuation     |
//| state                                                            |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EvaluateOneBar(CBufferFloat *plans, const uint plan_offset,
      CBufferFloat *rollout_context, const uint episode_step,
      const uint episode_horizon, const bool initialize,
      CBufferFloat *state_in, CBufferFloat *state_out,
      CBufferFloat *rewards_out, CBufferFloat *status_out,
      CBufferFloat *diagnostics_out)
  {
   m_status = DPCC_OK;
   SDPCCShape step_shape = m_shape;
   step_shape.H = 1;
   step_shape.M = 1;
   SDPCCLayout step_layout;
   const ulong state_count = (ulong)m_shape.K * DPCC_CONTINUATION_FLOATS;
   const ulong diagnostic_count = (ulong)m_shape.K * 4;
//--- Episode indices are carried by IEEE-754 binary32 continuation/source-action fields.
   const uint max_exact_float_index = 16777216;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || m_shape.M != 1 || episode_horizon == 0 ||
      episode_horizon > max_exact_float_index || episode_step >= episode_horizon ||
      (initialize ? (state_in != NULL) : (state_in == NULL)) || !BuildLayout(step_shape, step_layout) ||
      !ValidateBuffer(plans, (ulong)plan_offset + 6, false) ||
      !ValidateBuffer(rollout_context, step_layout.total, true) ||
      (!initialize && !ValidateBuffer(state_in, state_count, true)) ||
      !ValidateBuffer(state_out, state_count, true) || !ValidateBuffer(rewards_out, m_shape.K, true) ||
      !ValidateBuffer(status_out, m_shape.K, true) || !ValidateBuffer(diagnostics_out, diagnostic_count, true))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *inputs[] = {plans, rollout_context, state_in};
   CBufferFloat *outputs[] = {state_out, rewards_out, status_out, diagnostics_out};
   CBufferFloat *owned[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                            m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                            m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                            m_slice_active, m_batch_validity, m_batch_diagnostics
                           };
   for(uint input_index = 0; input_index < (uint)ArraySize(inputs); input_index++)
      for(uint previous = 0; previous < input_index; previous++)
         if(SharesDevice(inputs[input_index], inputs[previous]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint output = 0; output < (uint)ArraySize(outputs); output++)
     {
      for(uint input_index = 0; input_index < (uint)ArraySize(inputs); input_index++)
         if(SharesDevice(outputs[output], inputs[input_index]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint previous = 0; previous < output; previous++)
         if(SharesDevice(outputs[output], outputs[previous]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint internal = 0; internal < (uint)ArraySize(owned); internal++)
         if(SharesDevice(outputs[output], owned[internal]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   if(!ValidateInputAliases(plans, state_in, rollout_context))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const int kernel = def_k_DPCCRolloutStep;
   CBufferFloat *state_source = (initialize ? rollout_context : state_in);
   CBufferFloat *buffers[] = {plans, rollout_context, state_source, state_out, rewards_out,
                              status_out, diagnostics_out
                             };
   for(uint argument = 0; argument < (uint)ArraySize(buffers); argument++)
      if(!OpenCL.SetArgumentBuffer(kernel, argument, buffers[argument].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint integers[] = {plan_offset, m_shape.K, m_shape.E, step_shape.M,
                            (uint)step_layout.offsets[1], (uint)step_layout.offsets[2],
                            (uint)step_layout.offsets[3], (uint)step_layout.offsets[4],
                            (uint)step_layout.offsets[5], (uint)step_layout.offsets[6],
                            (uint)step_layout.offsets[7], (uint)step_layout.total,
                            episode_step, episode_horizon
                           };
   for(uint argument = 0; argument < (uint)ArraySize(integers); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 7, integers[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   const float reals[] = {(float)m_config.gamma, (float)m_config.volume_scale,
                          (float)m_config.price_scale, (float)m_config.max_quote_age_seconds,
                          (float)m_config.max_last_age_seconds, (float)m_config.money_scale
                         };
   for(uint argument = 0; argument < (uint)ArraySize(reals); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 21, reals[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 27, m_config.price_basis) ||
      !OpenCL.SetArgument(kernel, 28, m_config.quote_adapter) ||
      !OpenCL.SetArgument(kernel, 29, (uint)(m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? 4 :
                          (m_config.margin_fx_profile == DPCC_MARGIN_FOREX_SPOT_V1 ? 3 : 2))) ||
      !OpenCL.SetArgument(kernel, 30, m_config.account_currency_digits) ||
      !OpenCL.SetArgument(kernel, 31, (uint)(initialize ? 1 : 0)))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0};
   const uint global[] = {m_shape.K};
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   m_fixed = true;
   if(!status_out.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   for(uint scenario = 0; scenario < m_shape.K; scenario++)
     {
      const float status = status_out[scenario];
      if(status == 3.0f)
         return(Fail(DPCC_INVALID_INPUT));
      if(status == 5.0f)
         return(Fail(DPCC_NUMERIC_ERROR));
      if(status != 0.0f)
         return(Fail(DPCC_OPENCL_ERROR));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Advances one factual continuation without a new action           |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::EvaluateOneBarWithoutAction(CBufferFloat *rollout_context,
      const uint episode_step,
      const uint episode_horizon,
      CBufferFloat *state_in, CBufferFloat *state_out,
      CBufferFloat *rewards_out, CBufferFloat *status_out,
      CBufferFloat *diagnostics_out)
  {
//--- Validate caller-owned inputs and outputs before preparing the open.
   m_status = DPCC_OK;
   SDPCCShape step_shape = m_shape;
   step_shape.H = 1;
   step_shape.M = 1;
   SDPCCLayout step_layout;
   const ulong state_count = (ulong)m_shape.K * DPCC_CONTINUATION_FLOATS;
   if(m_open_pending || !m_configured || m_phase == DPCC_LAYER_FAILED || m_shape.M != 1 ||
      episode_step == 0 || episode_horizon == 0 || episode_horizon > 16777216 ||
      episode_step >= episode_horizon || !BuildLayout(step_shape, step_layout) ||
      !ValidateBuffer(rollout_context, step_layout.total, true) ||
      !ValidateBuffer(state_in, state_count, true) || !ValidateBuffer(state_out, state_count, true) ||
      !ValidateBuffer(rewards_out, m_shape.K, true) || !ValidateBuffer(status_out, m_shape.K, true) ||
      !ValidateBuffer(diagnostics_out, (ulong)m_shape.K * 4, true))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *inputs[] = {rollout_context, state_in};
   CBufferFloat *outputs[] = {state_out, rewards_out, status_out, diagnostics_out};
   for(uint i = 0; i < (uint)ArraySize(inputs); i++)
     {
      if(OwnsBufferOrDeviceSlot(inputs[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(inputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   for(uint i = 0; i < (uint)ArraySize(outputs); i++)
     {
      if(OwnsBufferOrDeviceSlot(outputs[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < (uint)ArraySize(inputs); j++)
         if(SharesDevice(outputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(outputs[i], outputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
//--- Keep preparation reward separate from the native intrabar suffix.
   CBufferFloat prepared_state, preparation_rewards;
   if(!prepared_state.BufferInit((uint)state_count, 0.0f))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!prepared_state.BufferCreate(OpenCL))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!preparation_rewards.BufferInit(m_shape.K, 0.0f))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!preparation_rewards.BufferCreate(OpenCL))
      return(Fail(DPCC_OPENCL_ERROR));
   ulong token = 0;
   if(!PrepareOpen(rollout_context, episode_step, episode_horizon, false, state_in,
                   GetPointer(prepared_state), GetPointer(preparation_rewards), status_out,
                   diagnostics_out, token))
      return(false);
   if(token == 0)
     {
      if(!CopyBufferRaw(GetPointer(prepared_state), state_out, (uint)state_count))
         return(Fail(DPCC_OPENCL_ERROR));
      if(!CopyBufferRaw(GetPointer(preparation_rewards), rewards_out, m_shape.K))
         return(Fail(DPCC_OPENCL_ERROR));
      return(true);
     }
   if(!CommitPreparedNoAction(token, state_out, rewards_out, status_out, diagnostics_out))
      return(false);
   if(!SumAndNormalize(GetPointer(preparation_rewards), rewards_out, rewards_out, 1, false, 0, 0, 0, 1.0f))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!rewards_out.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   for(uint scenario = 0; scenario < m_shape.K; scenario++)
      if(!MathIsValidNumber(rewards_out[scenario]))
         return(Fail(DPCC_NUMERIC_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Pins one observed bar after the causal open preparation          |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::PrepareOpenInternal(CBufferFloat *rollout_context,
      const uint episode_step, const uint episode_horizon,
      const bool initialize, CBufferFloat *state_in,
      CBufferFloat *prepared_state_out, CBufferFloat *prep_rewards_out,
      CBufferFloat *status_out, CBufferFloat *diagnostics_out,
      ulong &token, const bool observed_open)
  {
   if(!m_open_pending)
      token = 0;
   m_status = DPCC_OK;
   if(m_open_pending)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   SDPCCShape step_shape = m_shape;
   step_shape.H = 1;
   step_shape.M = 1;
   SDPCCLayout step_layout;
   const ulong state_count = (ulong)m_shape.K * DPCC_CONTINUATION_FLOATS;
   const ulong risk_count = (ulong)m_shape.K * DPCC_NATIVE_PENDING_PHASE_FLOATS;
   const ulong diagnostic_count = (ulong)m_shape.K * 4;
   const uint max_exact_float_index = 16777216;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || m_shape.M != 1 || episode_horizon == 0 ||
      episode_horizon > max_exact_float_index || episode_step >= episode_horizon ||
      (initialize ? (state_in != NULL) : (state_in == NULL)) || !BuildLayout(step_shape, step_layout) ||
      !ValidateBuffer(rollout_context, step_layout.total, true) ||
      (!initialize && !ValidateBuffer(state_in, state_count, true)) ||
      !ValidateBuffer(prepared_state_out, state_count, true) ||
      !ValidateBuffer(prep_rewards_out, m_shape.K, true) || !ValidateBuffer(status_out, m_shape.K, true) ||
      !ValidateBuffer(diagnostics_out, diagnostic_count, true) ||
      !ValidateBuffer(m_open_context, step_layout.total, true) ||
      !ValidateBuffer(m_open_state, state_count, true) || !ValidateBuffer(m_open_risk, risk_count, true))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *inputs[] = {rollout_context, state_in};
   CBufferFloat *outputs[] = {prepared_state_out, prep_rewards_out, status_out, diagnostics_out};
   CBufferFloat *owned[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                            m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                            m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                            m_slice_active, m_batch_validity, m_batch_diagnostics,
                            m_open_context, m_open_state, m_open_risk
                           };
   for(uint i = 0; i < (uint)ArraySize(inputs); i++)
     {
      if(i == 1 && initialize)
         continue;
      for(uint j = 0; j < (uint)ArraySize(owned); j++)
         if(SharesDevice(inputs[i], owned[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(inputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   for(uint i = 0; i < (uint)ArraySize(outputs); i++)
     {
      for(uint j = 0; j < (uint)ArraySize(inputs); j++)
         if(!(j == 1 && initialize) && SharesDevice(outputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(outputs[i], outputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < (uint)ArraySize(owned); j++)
         if(SharesDevice(outputs[i], owned[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   if(!CopyBufferRaw(rollout_context, m_open_context, (uint)step_layout.total))
      return(Fail(DPCC_OPENCL_ERROR));
   const int kernel = def_k_DPCCRolloutPrepareOpen;
   CBufferFloat *state_source = (initialize ? m_open_context : state_in);
   CBufferFloat *buffers[] = {m_open_context, state_source, prepared_state_out, m_open_risk,
                              prep_rewards_out, status_out, diagnostics_out
                             };
   for(uint argument = 0; argument < (uint)ArraySize(buffers); argument++)
      if(!OpenCL.SetArgumentBuffer(kernel, argument, buffers[argument].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint integers[] = {m_shape.K, m_shape.E, step_shape.M,
                            (uint)step_layout.offsets[1], (uint)step_layout.offsets[2],
                            (uint)step_layout.offsets[3], (uint)step_layout.offsets[4],
                            (uint)step_layout.offsets[5], (uint)step_layout.offsets[6],
                            (uint)step_layout.offsets[7], (uint)step_layout.total,
                            episode_step, episode_horizon
                           };
   for(uint argument = 0; argument < (uint)ArraySize(integers); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 7, integers[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   const float reals[] = {(float)m_config.gamma, (float)m_config.volume_scale,
                          (float)m_config.price_scale, (float)m_config.max_quote_age_seconds,
                          (float)m_config.max_last_age_seconds, (float)m_config.money_scale
                         };
   for(uint argument = 0; argument < (uint)ArraySize(reals); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 20, reals[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 26, m_config.price_basis) ||
      !OpenCL.SetArgument(kernel, 27, m_config.quote_adapter) ||
      !OpenCL.SetArgument(kernel, 28, (uint)(m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? 4 :
                          (m_config.margin_fx_profile == DPCC_MARGIN_FOREX_SPOT_V1 ? 3 : 2))) ||
      !OpenCL.SetArgument(kernel, 29, m_config.account_currency_digits) ||
      !OpenCL.SetArgument(kernel, 30, (uint)(initialize ? 1 : 0)) ||
      !OpenCL.SetArgument(kernel, 31, (uint)(observed_open ? 1 : 0)))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0};
   const uint global[] = {m_shape.K};
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!status_out.BufferRead() || !m_open_risk.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   bool actionable = false;
   for(uint scenario = 0; scenario < m_shape.K; scenario++)
     {
      const float status = status_out[scenario];
      if(status == 3.0f)
         return(Fail(DPCC_INVALID_INPUT));
      if(status == 5.0f)
         return(Fail(DPCC_NUMERIC_ERROR));
      if(status != 0.0f)
         return(Fail(DPCC_OPENCL_ERROR));
      if(m_open_risk[(int)((ulong)scenario * DPCC_NATIVE_PENDING_PHASE_FLOATS + 20)] == 1.0f)
         actionable = true;
     }
   m_fixed = true;
   if(!prepared_state_out.BufferRead() || !prep_rewards_out.BufferRead() || !diagnostics_out.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   if(!actionable)
      return(true);
   if(!CopyBufferRaw(prepared_state_out, m_open_state, (uint)state_count))
      return(Fail(DPCC_OPENCL_ERROR));
   m_open_generation++;
   if(m_open_generation == 0)
      m_open_generation = 1;
   m_open_step = episode_step;
   m_open_horizon = episode_horizon;
   m_open_actionable = true;
   m_open_pending = true;
   m_open_observed = observed_open;
   token = m_open_generation;
   return(true);
  }
//+------------------------------------------------------------------+
//| Prepares one legacy factual open and preserves current behavior  |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::PrepareOpen(CBufferFloat *rollout_context,
      const uint episode_step, const uint episode_horizon,
      const bool initialize, CBufferFloat *state_in,
      CBufferFloat *prepared_state_out, CBufferFloat *prep_rewards_out,
      CBufferFloat *status_out, CBufferFloat *diagnostics_out,
      ulong &token)
  {
   return(PrepareOpenInternal(rollout_context, episode_step, episode_horizon, initialize, state_in,
                              prepared_state_out, prep_rewards_out, status_out, diagnostics_out, token, false));
  }
//+------------------------------------------------------------------+
//| Prepares an observed open without exposing an intrabar suffix    |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::PrepareObservedOpen(CBufferFloat *rollout_context,
      const uint episode_step, const uint episode_horizon,
      const bool initialize, CBufferFloat *state_in,
      CBufferFloat *prepared_state_out, CBufferFloat *prep_rewards_out,
      CBufferFloat *status_out, CBufferFloat *diagnostics_out,
      ulong &token)
  {
   return(PrepareOpenInternal(rollout_context, episode_step, episode_horizon, initialize, state_in,
                              prepared_state_out, prep_rewards_out, status_out, diagnostics_out, token, true));
  }
//+------------------------------------------------------------------+
//| Consumes a pinned legacy factual context and prepared            |
//| continuation                                                     |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::CommitPreparedActionInternal(CBufferFloat *plans, const uint plan_offset,
      CBufferFloat *factual_context, const ulong token,
      CBufferFloat *close_state_out,
      CBufferFloat *suffix_rewards_out,
      CBufferFloat *status_out,
      CBufferFloat *diagnostics_out,
      const bool observed_commit, const bool apply_action)
  {
   m_status = DPCC_OK;
   if(token == 0 || !m_open_pending || !m_open_actionable || token != m_open_generation ||
      observed_commit != m_open_observed)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong state_count = (ulong)m_shape.K * DPCC_CONTINUATION_FLOATS;
   const ulong risk_count = (ulong)m_shape.K * DPCC_NATIVE_PENDING_PHASE_FLOATS;
   const ulong diagnostic_count = (ulong)m_shape.K * 4;
   SDPCCShape step_shape = m_shape;
   step_shape.H = 1;
   step_shape.M = 1;
   SDPCCLayout step_layout;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || !BuildLayout(step_shape, step_layout) ||
      (apply_action && !ValidateBuffer(plans, (ulong)plan_offset + 6, false)) ||
      !ValidateBuffer(close_state_out, state_count, true) ||
      !ValidateBuffer(suffix_rewards_out, m_shape.K, true) ||
      !ValidateBuffer(status_out, m_shape.K, true) ||
      !ValidateBuffer(diagnostics_out, diagnostic_count, true) ||
      !ValidateBuffer(m_open_context, step_layout.total, true) ||
      !ValidateBuffer(factual_context, step_layout.total, true) ||
      !ValidateBuffer(m_open_state, state_count, true) || !ValidateBuffer(m_open_risk, risk_count, true))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if((observed_commit && SharesDevice(factual_context, m_open_context)) ||
      (!observed_commit && factual_context != m_open_context))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *inputs[] = {plans, m_open_context, m_open_state, m_open_risk, factual_context};
   CBufferFloat *outputs[] = {close_state_out, suffix_rewards_out, status_out, diagnostics_out};
   CBufferFloat *owned[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                            m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                            m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                            m_slice_active, m_batch_validity, m_batch_diagnostics
                           };
   for(uint i = 0; i < (uint)ArraySize(inputs); i++)
     {
      if(i == 0 && !apply_action)
         continue;
      for(uint j = 0; j < i; j++)
         if(SharesDevice(inputs[i], inputs[j]) &&
            !(i == 4 && j == 1 && !observed_commit && factual_context == m_open_context))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < (uint)ArraySize(owned); j++)
         if(SharesDevice(inputs[i], owned[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   for(uint i = 0; i < (uint)ArraySize(outputs); i++)
     {
      for(uint j = 0; j < (uint)ArraySize(inputs); j++)
         if(SharesDevice(outputs[i], inputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(SharesDevice(outputs[i], outputs[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < (uint)ArraySize(owned); j++)
         if(SharesDevice(outputs[i], owned[j]))
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   const int kernel = def_k_DPCCRolloutCommitAction;
   CBufferFloat *buffers[] = {(apply_action ? plans : m_open_state), m_open_context, m_open_state, m_open_risk, factual_context,
                              close_state_out, suffix_rewards_out, status_out, diagnostics_out
                             };
   for(uint argument = 0; argument < (uint)ArraySize(buffers); argument++)
      if(!OpenCL.SetArgumentBuffer(kernel, argument, buffers[argument].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint integers[] = {plan_offset, m_shape.K, m_shape.E, step_shape.M,
                            (uint)step_layout.offsets[1], (uint)step_layout.offsets[2],
                            (uint)step_layout.offsets[3], (uint)step_layout.offsets[4],
                            (uint)step_layout.offsets[5], (uint)step_layout.offsets[6],
                            (uint)step_layout.offsets[7], (uint)step_layout.total,
                            m_open_step, m_open_horizon
                           };
   for(uint argument = 0; argument < (uint)ArraySize(integers); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 9, integers[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   const float reals[] = {(float)m_config.gamma, (float)m_config.volume_scale,
                          (float)m_config.price_scale, (float)m_config.max_quote_age_seconds,
                          (float)m_config.max_last_age_seconds, (float)m_config.money_scale
                         };
   for(uint argument = 0; argument < (uint)ArraySize(reals); argument++)
      if(!OpenCL.SetArgument(kernel, argument + 23, reals[argument]))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(kernel, 29, m_config.price_basis) ||
      !OpenCL.SetArgument(kernel, 30, m_config.quote_adapter) ||
      !OpenCL.SetArgument(kernel, 31, (uint)(m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? 4 :
                          (m_config.margin_fx_profile == DPCC_MARGIN_FOREX_SPOT_V1 ? 3 : 2))) ||
      !OpenCL.SetArgument(kernel, 32, m_config.account_currency_digits) ||
      !OpenCL.SetArgument(kernel, 33, (uint)(observed_commit ? 1 : 0)) ||
      !OpenCL.SetArgument(kernel, 34, (uint)(apply_action ? 1 : 0)))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0};
   const uint global[] = {m_shape.K};
//--- A launched commit is consumed even if execution reports an error.
   m_open_pending = false;
   m_open_actionable = false;
   m_open_observed = false;
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!status_out.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   for(uint scenario = 0; scenario < m_shape.K; scenario++)
     {
      const float status = status_out[scenario];
      if(status == 3.0f)
         return(Fail(DPCC_INVALID_INPUT));
      if(status == 5.0f)
         return(Fail(DPCC_NUMERIC_ERROR));
      if(status != 0.0f)
         return(Fail(DPCC_OPENCL_ERROR));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Commits a legacy prepared action against its pinned context      |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::CommitPreparedAction(CBufferFloat *plans, const uint plan_offset,
      const ulong token, CBufferFloat *close_state_out,
      CBufferFloat *suffix_rewards_out,
      CBufferFloat *status_out,
      CBufferFloat *diagnostics_out)
  {
   return(CommitPreparedActionInternal(plans, plan_offset, m_open_context, token, close_state_out,
                                       suffix_rewards_out, status_out, diagnostics_out, false));
  }
//+------------------------------------------------------------------+
//| Commits the prepared suffix without reading or applying an       |
//| action                                                           |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::CommitPreparedNoAction(const ulong token, CBufferFloat *close_state_out,
      CBufferFloat *suffix_rewards_out,
      CBufferFloat *status_out,
      CBufferFloat *diagnostics_out)
  {
   return(CommitPreparedActionInternal(NULL, 0, m_open_context, token, close_state_out,
                                       suffix_rewards_out, status_out, diagnostics_out, false, false));
  }
//+------------------------------------------------------------------+
//| Commits factual observed H1 data after the action is selected    |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::CommitObservedAction(CBufferFloat *plans, const uint plan_offset,
      CBufferFloat *factual_context, const ulong token,
      CBufferFloat *close_state_out,
      CBufferFloat *suffix_rewards_out,
      CBufferFloat *status_out,
      CBufferFloat *diagnostics_out)
  {
   return(CommitPreparedActionInternal(plans, plan_offset, factual_context, token, close_state_out,
                                       suffix_rewards_out, status_out, diagnostics_out, true));
  }
//+------------------------------------------------------------------+
//| Explicitly refuses financial backward and parameter updates      |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::calcInputGradients(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::calcInputGradients           |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::calcInputGradients(CNeuronBaseOCL *source, CBufferFloat *second,
      CBufferFloat *second_gradient,
      ENUM_ACTIVATION second_activation)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::calcHiddenGradients          |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::calcHiddenGradients(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::calcHiddenGradients          |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::calcHiddenGradients(CNeuronBaseOCL *source, CBufferFloat *second,
      CBufferFloat *second_gradient,
      ENUM_ACTIVATION second_activation)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::updateInputWeights           |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::updateInputWeights(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::updateInputWeights           |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::updateInputWeights(CNeuronBaseOCL *source, CBufferFloat *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::WeightsUpdate                |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::numOutputs                   |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::numOutputs(const uint outputs, ENUM_OPTIMIZATION optimization_type)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Blocks all host output readers until a successful public forward |
//+------------------------------------------------------------------+
int CNeuronScenarioRolloutOCL::getOutputVal(float &values[])
  {
   if(m_public_valid)
      return(Output.GetData(values));
   ArrayResize(values, 0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::getOutputVal                 |
//+------------------------------------------------------------------+
int CNeuronScenarioRolloutOCL::getOutputVal(CArrayFloat *values)
  {
   if(m_public_valid)
      return(Output.GetData(values));
   if(values)
      values.Clear();
   return(0);
  }
//+------------------------------------------------------------------+
//| Returns the public rollout values when available.                |
//+------------------------------------------------------------------+
int CNeuronScenarioRolloutOCL::getOutputVal(vector<float> &values)
  {
   if(m_public_valid)
      return(Output.GetData(values));
   values.Resize(0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Hashes fixed geometry and profile, excluding projection mode     |
//+------------------------------------------------------------------+
ulong CNeuronScenarioRolloutOCL::ConfigSignature(const SDPCCConfig &config) const
  {
   return(DPCCFixedSignature(config, defNeuronScenarioRolloutOCL));
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::ContractSignature            |
//+------------------------------------------------------------------+
ulong CNeuronScenarioRolloutOCL::ContractSignature(void) const
  {
   return(m_configured && m_phase != DPCC_LAYER_FAILED ? ConfigSignature(m_config) : 0);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::AppendParameterFingerprint   |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(!m_configured || m_phase == DPCC_LAYER_FAILED)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   DPCCRolloutHashWord(Type(), fingerprint);
   DPCCRolloutHashWord(ConfigSignature(m_config), fingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Serializes version 3 geometry and complete fixed configuration   |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !m_configured || m_phase == DPCC_LAYER_FAILED ||
      !ValidateConfig(m_config))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint header[] = {defNeuronScenarioRolloutOCL, (m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? DPCC_HISTORICAL_FX_ENVELOPE_VERSION :
                          (m_config.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 ? DPCC_BINARY_LAYER_VERSION : DPCC_FINANCIAL_ENVELOPE_VERSION)), DPCC_CONTRACT_VERSION,
                          m_shape.K, m_shape.V, m_shape.H, m_shape.D, m_shape.M, m_shape.J, m_shape.E,
                          m_shape.PlanBatch, m_shape.NActions, m_shape.StateSize, m_shape.Embedding,
                          m_shape.Heads, m_shape.KeyDimension, m_shape.Depth,
                          (uint)optimization, iBatch, (uint)m_myIndex
                         };
   for(uint i = 0; i < (uint)ArraySize(header); i++)
      if(FileWriteInteger(file_handle, (int)header[i], INT_VALUE) != INT_VALUE)
         return(Fail(DPCC_INVALID_INPUT));
   const uint integers[] = {(uint)m_config.mode, m_config.projection_max_iterations,
                            m_config.projection_max_evaluations, m_config.projection_lookback_steps,
                            m_config.price_basis, m_config.quote_adapter, m_config.event_model,
                            (uint)m_config.risk_type, m_config.normalization_mode,
                            m_config.loss_coordinates, m_config.randomness_version
                           };
   for(uint i = 0; i < (uint)ArraySize(integers); i++)
      if(FileWriteInteger(file_handle, (int)integers[i], INT_VALUE) != INT_VALUE)
         return(Fail(DPCC_INVALID_INPUT));
   if(m_config.margin_fx_profile != DPCC_MARGIN_EXTERNAL_V2 &&
      (FileWriteInteger(file_handle, (int)m_config.margin_fx_profile, INT_VALUE) != INT_VALUE ||
       FileWriteInteger(file_handle, (int)m_config.account_currency_digits, INT_VALUE) != INT_VALUE))
      return(Fail(DPCC_INVALID_INPUT));
   const double reals[] = {m_config.projection_improvement_atol, m_config.projection_distance_tie_atol,
                           m_config.volume_scale, m_config.price_scale, m_config.money_scale,
                           m_config.max_quote_age_seconds, m_config.max_last_age_seconds
                          };
   for(uint i = 0; i < (uint)ArraySize(reals); i++)
      if(FileWriteDouble(file_handle, reals[i]) != sizeof(double))
         return(Fail(DPCC_INVALID_INPUT));
   const float scalars[] = {m_config.risk_parameter, m_config.mean_weight, m_config.gamma,
                            m_config.probability_tolerance
                           };
   for(uint i = 0; i < (uint)ArraySize(scalars); i++)
      if(FileWriteFloat(file_handle, scalars[i]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   for(uint action = 0; action < 6; action++)
      if(FileWriteFloat(file_handle, m_config.action_bias[action]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   for(uint action = 0; action < 6; action++)
      if(FileWriteFloat(file_handle, m_config.action_scale[action]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   if(FileWriteLong(file_handle, (long)ConfigSignature(m_config)) != sizeof(long))
      return(Fail(DPCC_INVALID_INPUT));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::Load                         |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Load(const int file_handle)
  {
   m_public_valid = false;
   m_phase = DPCC_LAYER_FAILED;
   m_configured = false;
//--- The factory has already consumed the exact Type identifier.
   if(file_handle == INVALID_HANDLE || !OpenCL || FileTell(file_handle) > FileSize(file_handle) ||
      FileSize(file_handle) - FileTell(file_handle) < 248)
      return(Fail(DPCC_INVALID_INPUT, true));
   uint header[19];
   for(uint i = 0; i < 19; i++)
      header[i] = (uint)FileReadInteger(file_handle, INT_VALUE);
   if((header[0] != DPCC_BINARY_LAYER_VERSION && header[0] != DPCC_FINANCIAL_ENVELOPE_VERSION &&
       header[0] != DPCC_HISTORICAL_FX_ENVELOPE_VERSION) ||
      header[1] != DPCC_CONTRACT_VERSION || header[18] > INT_MAX ||
      (header[0] != DPCC_BINARY_LAYER_VERSION &&
       FileSize(file_handle) - FileTell(file_handle) < 248 - 19 * INT_VALUE + 2 * INT_VALUE))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   CLayerDescription desc;
   desc.type = defNeuronScenarioRolloutOCL;
   desc.window = 6;
   desc.window_out = 13;
   desc.layers = header[15];
   desc.optimization = (ENUM_OPTIMIZATION)header[16];
   desc.batch = header[17];
   desc.activation = None;
   if(ArrayResize(desc.windows, 9) != 9 || ArrayResize(desc.units, 5) != 5 || ArrayResize(desc.heads, 1) != 1)
      return(Fail(DPCC_INVALID_INPUT, true));
   desc.windows[0] = DPCC_CONTRACT_VERSION;
   for(uint i = 0; i < 8; i++)
      desc.windows[i + 1] = header[i + 2];
   desc.units[0] = header[12];
   desc.units[1] = header[13];
   desc.units[2] = header[14];
   desc.units[3] = header[10];
   desc.units[4] = header[11];
   desc.heads[0] = desc.units[1];
   desc.variables = header[3];
   desc.step = header[4];
   const ulong count = (ulong)header[2] * header[6];
   if(count > INT_MAX)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   desc.count = (uint)count;
   SDPCCConfig candidate;
   ZeroMemory(candidate);
   if(!ReadDescriptor(GetPointer(desc), candidate.shape))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   candidate.contract_version = DPCC_CONTRACT_VERSION;
   candidate.binary_layer_version = DPCC_BINARY_LAYER_VERSION;
   candidate.mode = (ENUM_DPCC_MODE)FileReadInteger(file_handle);
   candidate.projection_max_iterations = (uint)FileReadInteger(file_handle);
   candidate.projection_max_evaluations = (uint)FileReadInteger(file_handle);
   candidate.projection_lookback_steps = (uint)FileReadInteger(file_handle);
   candidate.price_basis = (uint)FileReadInteger(file_handle);
   candidate.quote_adapter = (uint)FileReadInteger(file_handle);
   candidate.event_model = (uint)FileReadInteger(file_handle);
   candidate.risk_type = FileReadInteger(file_handle);
   candidate.normalization_mode = (uint)FileReadInteger(file_handle);
   candidate.loss_coordinates = (uint)FileReadInteger(file_handle);
   candidate.randomness_version = (uint)FileReadInteger(file_handle);
   if(header[0] != DPCC_BINARY_LAYER_VERSION)
     {
      candidate.margin_fx_profile = (uint)FileReadInteger(file_handle);
      candidate.account_currency_digits = (uint)FileReadInteger(file_handle);
      if(candidate.margin_fx_profile != (header[0] == DPCC_HISTORICAL_FX_ENVELOPE_VERSION ?
                                         DPCC_MARGIN_HISTORICAL_CLOSE_V1 : DPCC_MARGIN_FOREX_SPOT_V1))
         return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   candidate.projection_improvement_atol = FileReadDouble(file_handle);
   candidate.projection_distance_tie_atol = FileReadDouble(file_handle);
   candidate.volume_scale = FileReadDouble(file_handle);
   candidate.price_scale = FileReadDouble(file_handle);
   candidate.money_scale = FileReadDouble(file_handle);
   candidate.max_quote_age_seconds = FileReadDouble(file_handle);
   candidate.max_last_age_seconds = FileReadDouble(file_handle);
   candidate.risk_parameter = FileReadFloat(file_handle);
   candidate.mean_weight = FileReadFloat(file_handle);
   candidate.gamma = FileReadFloat(file_handle);
   candidate.probability_tolerance = FileReadFloat(file_handle);
   for(uint action = 0; action < 6; action++)
      candidate.action_bias[action] = FileReadFloat(file_handle);
   for(uint action = 0; action < 6; action++)
      candidate.action_scale[action] = FileReadFloat(file_handle);
   const ulong expected_signature = (ulong)FileReadLong(file_handle);
   if(expected_signature == 0 || expected_signature != ConfigSignature(candidate))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(!Init(0, header[18], OpenCL, GetPointer(desc)))
      return(Fail(m_status, true));
   if(!Configure(candidate))
      return(Fail(m_status, true));
   m_fixed = true;
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Clears transient state while retaining frozen configuration      |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::Clear(void)
  {
   m_public_valid = false;
   m_open_pending = false;
   m_open_actionable = false;
   m_open_observed = false;
   m_open_step = 0;
   m_open_horizon = 0;
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {Output, PrevOutput, Gradient, m_states, m_validity, m_violations,
                              m_diagnostics, m_return_scratch, m_state_scratch, m_violation_scratch,
                              m_validity_scratch, m_diagnostic_scratch, m_system, m_aggregate,
                              m_batch_validity, m_batch_diagnostics, m_open_context, m_open_state, m_open_risk
                             };
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!buffers[i] || !buffers[i].Fill(0))
         return(Fail(DPCC_OPENCL_ERROR, true));
   if(!m_slice_active || !m_slice_active.Fill(1))
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_phase = (m_configured ? DPCC_LAYER_CONFIGURED : DPCC_LAYER_SHAPED);
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Transfers every owned slot through its creating context          |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::SupportsOpenCLChecked(void) const
  {
   return(Type() == defNeuronScenarioRolloutOCL);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::SetOpenCLChecked             |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::SetOpenCLChecked(COpenCLMy *obj)
  {
   m_public_valid = false;
   if(!obj || !SupportsOpenCLChecked() || m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_states, m_validity, m_violations, m_diagnostics, m_return_scratch,
                              m_state_scratch, m_violation_scratch, m_validity_scratch,
                              m_diagnostic_scratch, m_system, m_aggregate, m_slice_active,
                              m_batch_validity, m_batch_diagnostics, m_open_context, m_open_state, m_open_risk
                             };
//--- Read all foreign allocations before releasing any old-owner slot.
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!buffers[i] || (buffers[i].GetIndex() >= 0 && buffers[i].GetOpenCL() != obj &&
                         !buffers[i].BufferRead()))
         return(Fail(DPCC_OPENCL_ERROR, true));
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if((buffers[i].GetOpenCL() != obj || buffers[i].GetIndex() < 0) && !buffers[i].BufferCreate(obj))
         return(Fail(DPCC_OPENCL_ERROR, true));
   if(!CNeuronBaseOCL::SetOpenCLChecked(obj))
      return(Fail(DPCC_OPENCL_ERROR, true));
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::SetOpenCL                    |
//+------------------------------------------------------------------+
void CNeuronScenarioRolloutOCL::SetOpenCL(COpenCLMy *obj)
  {
   SetOpenCLChecked(obj);
  }
//+------------------------------------------------------------------+
//| Reconstructs all descriptor copies from serialized geometry      |
//+------------------------------------------------------------------+
CLayerDescription *CNeuronScenarioRolloutOCL::GetLayerInfo(void)
  {
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED)
      return(NULL);
   CLayerDescription *desc = new CLayerDescription();
   if(!desc)
      return(NULL);
   desc.type = Type();
   desc.count = m_shape.M * m_shape.K;
   desc.window = 6;
   desc.window_out = 13;
   desc.variables = m_shape.V;
   desc.step = m_shape.H;
   desc.layers = 0;
   desc.batch = iBatch;
   desc.optimization = optimization;
   desc.activation = None;
   if(ArrayResize(desc.windows, 9) != 9 || ArrayResize(desc.units, 5) != 5 || ArrayResize(desc.heads, 1) != 1)
     {
      delete desc;
      return(NULL);
     }
   const uint windows[] = {DPCC_CONTRACT_VERSION, m_shape.K, m_shape.V, m_shape.H, m_shape.D,
                           m_shape.M, m_shape.J, m_shape.E, m_shape.PlanBatch
                          };
   for(uint i = 0; i < 9; i++)
      desc.windows[i] = windows[i];
   desc.units[0] = m_shape.Embedding;
   desc.units[1] = m_shape.Heads;
   desc.units[2] = m_shape.KeyDimension;
   desc.units[3] = m_shape.NActions;
   desc.units[4] = m_shape.StateSize;
   desc.heads[0] = m_shape.Heads;
   return(desc);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::MatchesProjectionContract    |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::MatchesProjectionContract(const SDPCCShape &shape,
      const SDPCCConfig &config,
      COpenCLMy *owner) const
  {
   return(CheckPointer(owner) != POINTER_INVALID && OpenCL == owner && m_configured &&
          m_phase != DPCC_LAYER_CONSTRUCTED && m_phase != DPCC_LAYER_FAILED &&
          DPCCSameShape(m_shape, shape) && DPCCValidateFixedConfig(shape, config) &&
          DPCCFixedSignature(config, defNeuronScenarioRolloutOCL) == ContractSignature());
  }
//+------------------------------------------------------------------+
//| Constructs exclusively owned fixed projection workspace          |
//+------------------------------------------------------------------+
CNeuronPlanProjectionOCL::CNeuronPlanProjectionOCL(void)
  {
   ZeroMemory(m_shape);
   ZeroMemory(m_layout);
   ZeroMemory(m_config);
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      m_work[i] = NULL;
   m_rollout = NULL;
   m_phase = DPCC_LAYER_CONSTRUCTED;
   m_status = DPCC_CONTRACT_MISMATCH;
   m_configured = false;
   m_fixed = false;
   m_public_valid = false;
   bTrain = false;
  }
//+------------------------------------------------------------------+
//| Releases projector ownership without deleting the borrowed layer |
//+------------------------------------------------------------------+
CNeuronPlanProjectionOCL::~CNeuronPlanProjectionOCL(void)
  {
   m_rollout = NULL;
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      DeleteObj(m_work[i]);
  }
//+------------------------------------------------------------------+
//| Invalidates failed public results and records unsupported routes |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Fail(const ENUM_DPCC_STATUS status, const bool fatal)
  {
   m_status = status;
   if(status != DPCC_BACKWARD_UNSUPPORTED)
      m_public_valid = false;
   if(fatal)
      m_phase = DPCC_LAYER_FAILED;
   return(false);
  }
//+------------------------------------------------------------------+
//| Maps all fixed buffer sizes before any device allocation         |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::WorkspaceSizes(ulong &counts[]) const
  {
   SDPCCLayout checked;
   if(!DPCCBuildRolloutLayout(m_shape, checked) || ArrayResize(counts, DPCC_PW_COUNT) != DPCC_PW_COUNT)
      return(false);
   const ulong mh6 = (ulong)m_shape.M * m_shape.H * 6;
   const ulong mk = (ulong)m_shape.M * m_shape.K;
   const ulong mp = (ulong)m_shape.M * 9;
   const ulong mpk = mp * m_shape.K;
   const ulong values[] = {mh6, mh6, mh6, m_shape.M, m_shape.M, mk,
                           mk * (m_shape.H + 1) * 24, mk * 8, (ulong)m_shape.M * 12,
                           mh6 * 9, mp, mpk, mpk * (m_shape.H + 1) * 24, mpk * 8,
                           mp * 12, (ulong)m_shape.M * m_shape.H, (ulong)m_shape.M * 24,
                           1, 6, m_shape.M, m_shape.M, (ulong)m_shape.M * 12,
                           m_shape.M, m_shape.M, (ulong)m_shape.M * 12
                          };
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
     {
      if(values[i] == 0 || values[i] > INT_MAX)
         return(false);
      counts[i] = values[i];
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocates complete M and MP generations before the first forward |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::EnsureWorkspace(void)
  {
   ulong counts[];
   if(!WorkspaceSizes(counts))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      if(!EnsureBuffer(m_work[i], (uint)counts[i]))
         return(Fail(m_status, true));
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks actual owned allocation identity before any kernel writes |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ValidateWorkspace(CBufferFloat *plans, CBufferFloat *context) const
  {
   ulong counts[];
   if(!WorkspaceSizes(counts) || !ValidateBuffer(Output, counts[DPCC_PW_RESULT]) ||
      !ValidateBuffer(PrevOutput, counts[DPCC_PW_RESULT]) || !ValidateBuffer(Gradient, counts[DPCC_PW_RESULT]))
      return(false);
   CBufferFloat *owned[];
   if(ArrayResize(owned, DPCC_PW_COUNT + 3) != DPCC_PW_COUNT + 3)
      return(false);
   owned[0] = Output;
   owned[1] = PrevOutput;
   owned[2] = Gradient;
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
     {
      if(!ValidateBuffer(m_work[i], counts[i]))
         return(false);
      owned[i + 3] = m_work[i];
     }
   for(uint i = 0; i < (uint)ArraySize(owned); i++)
     {
      if(SharesDevice(plans, owned[i]) || SharesDevice(context, owned[i]))
         return(false);
      for(uint j = 0; j < i; j++)
         if(SharesDevice(owned[i], owned[j]))
            return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Uploads the six fixed positive normalized action scales          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::WriteScales(void)
  {
   if(!m_configured)
      return(true);
   for(uint action = 0; action < 6; action++)
      if(!m_work[DPCC_PW_SCALES].Update((int)action, m_config.action_scale[action]))
         return(Fail(DPCC_OPENCL_ERROR));
   return(m_work[DPCC_PW_SCALES].BufferWrite() || Fail(DPCC_OPENCL_ERROR));
  }
//+------------------------------------------------------------------+
//| Releases transient tensors through their actual creating owners  |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ReleaseOwned(void)
  {
   CBufferFloat *base[] = {Output, PrevOutput, Gradient, Weights, DeltaWeights, FirstMomentum, SecondMomentum};
   for(uint i = 0; i < (uint)ArraySize(base); i++)
      if(CheckPointer(base[i]) != POINTER_INVALID && base[i].GetIndex() >= 0 && !base[i].BufferFree())
         return(Fail(DPCC_OPENCL_ERROR, true));
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      if(CheckPointer(m_work[i]) != POINTER_INVALID && m_work[i].GetIndex() >= 0 && !m_work[i].BufferFree())
         return(Fail(DPCC_OPENCL_ERROR, true));
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes an exact parameter-free plan-shaped CNet layer       |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    CLayerDescription *desc)
  {
   m_public_valid = false;
   m_rollout = NULL;
   SDPCCShape candidate;
   if(CheckPointer(open_cl) == POINTER_INVALID || !ReadDescriptor(desc, candidate))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(!ReleaseOwned())
      return(false);
   OpenCL = open_cl;
   m_shape = candidate;
   if(!DPCCBuildRolloutLayout(m_shape, m_layout) ||
      !CNeuronBaseOCL::Init(0, myIndex, open_cl, desc.count, desc.optimization, desc.batch))
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_myIndex = (int)myIndex;
   activation = None;
   bTrain = false;
   m_configured = false;
   m_fixed = false;
   ZeroMemory(m_config);
   if(!EnsureWorkspace())
      return(false);
   m_phase = DPCC_LAYER_SHAPED;
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Freezes exact fixed profile and requires explicit rollout        |
//| binding                                                          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Configure(const SDPCCConfig &config)
  {
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED || !ValidateWorkspace() ||
      !ValidateConfig(config) ||
      (m_fixed && ConfigSignature(config) != ConfigSignature(m_config)))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_config = config;
   m_configured = true;
   m_rollout = NULL;
   m_public_valid = false;
   m_phase = DPCC_LAYER_CONFIGURED;
   m_status = DPCC_OK;
   return(WriteScales());
  }
//+------------------------------------------------------------------+
//| Binds a nonowning exact configured financial model on this owner |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::BindRollout(CNeuronScenarioRolloutOCL *rollout)
  {
   m_public_valid = false;
   m_rollout = NULL;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || CheckPointer(rollout) == POINTER_INVALID ||
      rollout.Type() != defNeuronScenarioRolloutOCL ||
      !rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) || !ValidateWorkspace())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_rollout = rollout;
   m_phase = DPCC_LAYER_READY;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Detaches only the borrowed financial model and public readiness  |
//+------------------------------------------------------------------+
void CNeuronPlanProjectionOCL::UnbindRollout(void)
  {
   m_rollout = NULL;
   m_public_valid = false;
   if(m_configured && m_phase != DPCC_LAYER_FAILED)
      m_phase = DPCC_LAYER_CONFIGURED;
   m_status = DPCC_CONTRACT_MISMATCH;
  }
//+------------------------------------------------------------------+
//| Binds each exact GPU argument without implicit host tensor reads |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::BindBuffers(const int kernel, CBufferFloat *&buffers[])
  {
   for(uint arg = 0; arg < (uint)ArraySize(buffers); arg++)
     {
      if(CheckPointer(buffers[arg]) == POINTER_INVALID)
        {
         PrintFormat("DPCC_PROJECTION_FAIL operation=bind_buffer kernel=%s argument=%u cause=invalid_buffer",
                     OpenCL.GetKernelName(kernel), arg);
         return(Fail(DPCC_OPENCL_ERROR));
        }
      if(!OpenCL.SetArgumentBuffer(kernel, arg, buffers[arg].GetIndex()))
        {
         PrintFormat("DPCC_PROJECTION_FAIL operation=bind_buffer kernel=%s argument=%u error=%d",
                     OpenCL.GetKernelName(kernel), arg, GetLastError());
         return(Fail(DPCC_OPENCL_ERROR));
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Binds checked integer scalar arguments in ABI order              |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::BindIntegers(const int kernel, const uint start, const uint &values[])
  {
   for(uint arg = 0; arg < (uint)ArraySize(values); arg++)
      if(!OpenCL.SetArgument(kernel, start + arg, values[arg]))
        {
         PrintFormat("DPCC_PROJECTION_FAIL operation=bind_integer kernel=%s argument=%u error=%d",
                     OpenCL.GetKernelName(kernel), start + arg, GetLastError());
         return(Fail(DPCC_OPENCL_ERROR));
        }
   return(true);
  }
//+------------------------------------------------------------------+
//| Launches one independent writer per candidate or proposal slot   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ExecuteRows(const int kernel, const uint rows)
  {
   const uint offset[] = {0};
   const uint global[] = {rows};
   if(OpenCL.Execute(kernel, 1, offset, global))
      return(true);
   string error = "";
   CLGetInfoString(OpenCL.GetContext(), CL_ERROR_DESCRIPTION, error);
   PrintFormat("DPCC_PROJECTION_FAIL operation=execute_kernel kernel=%s rows=%u error=%s",
               OpenCL.GetKernelName(kernel), rows, error);
   return(Fail(DPCC_OPENCL_ERROR));
  }
//+------------------------------------------------------------------+
//| Reads only the one common GPU completion or system verdict       |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ReadCommon(const uint mode, uint &verdict)
  {
   const int kernel = def_k_DPCCProjectionReduce;
   CBufferFloat *buffers[] = {m_work[DPCC_PW_CONTROL], m_work[DPCC_PW_AGGREGATE]};
   const uint values[] = {m_shape.M, mode};
   if(!BindBuffers(kernel, buffers) || !BindIntegers(kernel, 2, values) || !ExecuteRows(kernel, 1))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!m_work[DPCC_PW_AGGREGATE].BufferRead())
     {
      PrintFormat("DPCC_PROJECTION_FAIL operation=readback kernel=%s", OpenCL.GetKernelName(kernel));
      return(Fail(DPCC_OPENCL_ERROR));
     }
   const float common = m_work[DPCC_PW_AGGREGATE][0];
   if(common == 3)
      return(Fail(DPCC_INVALID_INPUT));
   if(common == 5)
      return(Fail(DPCC_NUMERIC_ERROR));
   if(common != 0 && common != (mode == 0 ? 1 : 2))
      return(Fail(DPCC_OPENCL_ERROR));
   verdict = (uint)common;
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates finiteness and snapshots the immutable original once   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ValidatePlans(CBufferFloat *plans, CBufferFloat *context)
  {
   const int kernel = def_k_DPCCProjectionInit;
   CBufferFloat *buffers[] = {plans, context, m_work[DPCC_PW_ORIGINAL], m_work[DPCC_PW_CURRENT],
                              m_work[DPCC_PW_BASE_ACTIVE], m_work[DPCC_PW_FINAL_ACTIVE],
                              m_work[DPCC_PW_CONTROL]
                             };
   const uint values[] = {m_shape.M, m_shape.H, (uint)m_layout.offsets[7], 0};
   uint common = 0;
   return(BindBuffers(kernel, buffers) && BindIntegers(kernel, 7, values) &&
          ExecuteRows(kernel, m_shape.M) && ReadCommon(0, common));
  }
//+------------------------------------------------------------------+
//| Reduces every K record and immutable-original distance on device |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Summarize(const bool proposals, CBufferFloat *plans, CBufferFloat *active)
  {
   const int kernel = def_k_DPCCProjectionSummary;
   CBufferFloat *buffers[] = {plans, m_work[DPCC_PW_ORIGINAL], active,
                              m_work[proposals ? DPCC_PW_PROPOSAL_RETURN : DPCC_PW_BASE_RETURN],
                              m_work[proposals ? DPCC_PW_PROPOSAL_STATE : DPCC_PW_BASE_STATE],
                              m_work[proposals ? DPCC_PW_PROPOSAL_VIOLATION : DPCC_PW_BASE_VIOLATION],
                              m_work[proposals ? DPCC_PW_PROPOSAL_SUMMARY : DPCC_PW_BASE_SUMMARY],
                              m_work[DPCC_PW_SCALES]
                             };
   const uint values[] = {m_shape.M, m_shape.K, m_shape.H, (proposals ? 9 : 1)};
   return(BindBuffers(kernel, buffers) && BindIntegers(kernel, 8, values) &&
          ExecuteRows(kernel, m_shape.M * (proposals ? 9 : 1)));
  }
//+------------------------------------------------------------------+
//| Consumes original or canonical summary and builds causal order   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::BuildRepairIndexOrder(CBufferFloat *context, const uint stage)
  {
   const int kernel = def_k_DPCCProjectionPrepare;
   CBufferFloat *buffers[] = {m_work[DPCC_PW_CURRENT], context, m_work[DPCC_PW_BASE_SUMMARY],
                              m_work[DPCC_PW_CONTROL], m_work[DPCC_PW_BASE_ACTIVE],
                              m_work[DPCC_PW_FINAL_ACTIVE], m_work[DPCC_PW_ORDER]
                             };
   const uint values[] = {m_shape.M, m_shape.H, stage, (uint)m_layout.offsets[1],
                          (uint)m_layout.offsets[2], (uint)m_layout.offsets[3],
                          m_config.projection_max_evaluations, m_config.projection_max_iterations,
                          m_config.projection_lookback_steps
                         };
   return(BindBuffers(kernel, buffers) && BindIntegers(kernel, 7, values) && ExecuteRows(kernel, m_shape.M));
  }
//+------------------------------------------------------------------+
//| Builds nine independent alternatives with reserve and prefix     |
//| mask                                                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::BuildLocalProposals(CBufferFloat *context)
  {
   const int kernel = def_k_DPCCProjectionProposals;
   CBufferFloat *buffers[] = {m_work[DPCC_PW_CURRENT], context, m_work[DPCC_PW_CONTROL],
                              m_work[DPCC_PW_ORDER], m_work[DPCC_PW_PROPOSAL],
                              m_work[DPCC_PW_PROPOSAL_ACTIVE]
                             };
   const uint values[] = {m_shape.M, m_shape.H, (uint)m_layout.offsets[1],
                          (uint)m_layout.offsets[2], (uint)m_layout.offsets[3],
                          m_config.projection_max_evaluations, m_config.projection_max_iterations
                         };
   return(BindBuffers(kernel, buffers) && BindIntegers(kernel, 6, values) && ExecuteRows(kernel, m_shape.M));
  }
//+------------------------------------------------------------------+
//| Evaluates one fixed MP batch using the bound common financial    |
//| model                                                            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::EvaluateProposals(CBufferFloat *context)
  {
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_PROPOSAL], 0, m_shape.M * 9,
                                   m_work[DPCC_PW_PROPOSAL_ACTIVE], context,
                                   m_work[DPCC_PW_PROPOSAL_RETURN], m_work[DPCC_PW_PROPOSAL_STATE],
                                   m_work[DPCC_PW_PROPOSAL_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   return(Summarize(true, m_work[DPCC_PW_PROPOSAL], m_work[DPCC_PW_PROPOSAL_ACTIVE]));
  }
//+------------------------------------------------------------------+
//| Accepts at most one ordered strict risk improvement per row      |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SelectImprovement(void)
  {
   const int kernel = def_k_DPCCProjectionSelect;
   CBufferFloat *buffers[] = {m_work[DPCC_PW_PROPOSAL], m_work[DPCC_PW_PROPOSAL_ACTIVE],
                              m_work[DPCC_PW_PROPOSAL_SUMMARY], m_work[DPCC_PW_CURRENT],
                              m_work[DPCC_PW_CONTROL], m_work[DPCC_PW_ORDER],
                              m_work[DPCC_PW_FINAL_ACTIVE]
                             };
   const uint values[] = {m_shape.M, m_shape.H, m_config.projection_max_evaluations,
                          m_config.projection_max_iterations, m_config.projection_lookback_steps
                         };
   if(!BindBuffers(kernel, buffers) || !BindIntegers(kernel, 7, values) ||
      !OpenCL.SetArgument(kernel, 12, (float)m_config.projection_improvement_atol) ||
      !OpenCL.SetArgument(kernel, 13, (float)m_config.projection_distance_tie_atol))
      return(Fail(DPCC_OPENCL_ERROR));
   return(ExecuteRows(kernel, m_shape.M));
  }
//+------------------------------------------------------------------+
//| Verifies repaired rows by reserved full rollout before           |
//| publication                                                      |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::VerifyProjection(CBufferFloat *context)
  {
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_CURRENT], 0, m_shape.M,
                                   m_work[DPCC_PW_FINAL_ACTIVE], context,
                                   m_work[DPCC_PW_BASE_RETURN], m_work[DPCC_PW_BASE_STATE],
                                   m_work[DPCC_PW_BASE_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   if(!Summarize(false, m_work[DPCC_PW_CURRENT], m_work[DPCC_PW_FINAL_ACTIVE]))
      return(false);
   const int kernel = def_k_DPCCProjectionPublish;
   CBufferFloat *buffers[] = {m_work[DPCC_PW_ORIGINAL], m_work[DPCC_PW_CURRENT],
                              m_work[DPCC_PW_BASE_SUMMARY], m_work[DPCC_PW_CONTROL],
                              m_work[DPCC_PW_RESULT], m_work[DPCC_PW_COST_TEMP],
                              m_work[DPCC_PW_STATUS_TEMP], m_work[DPCC_PW_DIAGNOSTICS_TEMP]
                             };
   const uint values[] = {m_shape.M, m_shape.H, m_config.projection_max_evaluations};
   uint common = 0;
   if(!BindBuffers(kernel, buffers) || !BindIntegers(kernel, 8, values) ||
      !ExecuteRows(kernel, m_shape.M) || !ReadCommon(1, common))
      return(false);
//--- Every public generation becomes visible only after all system checks succeed.
   CBufferFloat *temporary = Output;
   Output = m_work[DPCC_PW_RESULT];
   m_work[DPCC_PW_RESULT] = temporary;
   temporary = m_work[DPCC_PW_COST];
   m_work[DPCC_PW_COST] = m_work[DPCC_PW_COST_TEMP];
   m_work[DPCC_PW_COST_TEMP] = temporary;
   temporary = m_work[DPCC_PW_STATUS];
   m_work[DPCC_PW_STATUS] = m_work[DPCC_PW_STATUS_TEMP];
   m_work[DPCC_PW_STATUS_TEMP] = temporary;
   temporary = m_work[DPCC_PW_DIAGNOSTICS];
   m_work[DPCC_PW_DIAGNOSTICS] = m_work[DPCC_PW_DIAGNOSTICS_TEMP];
   m_work[DPCC_PW_DIAGNOSTICS_TEMP] = temporary;
   m_status = (common == 2 ? DPCC_SEARCH_EXHAUSTED : DPCC_OK);
   m_public_valid = true;
   m_phase = DPCC_LAYER_READY;
   return(true);
  }
//+------------------------------------------------------------------+
//| Rejects a missing financial context before touching workspace    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::feedForward(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| Runs the complete bounded causal GPU projection state machine    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::feedForward(CNeuronBaseOCL *source, CBufferFloat *context)
  {
   m_public_valid = false;
   m_status = DPCC_OK;
   if(!m_configured || m_phase == DPCC_LAYER_FAILED || CheckPointer(source) == POINTER_INVALID ||
      CheckPointer(m_rollout) == POINTER_INVALID ||
      !m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) ||
      !ValidateBuffer(source.getOutput(), (ulong)m_shape.M * m_shape.H * 6) ||
      !ValidateBuffer(context, m_layout.total) || !ValidateWorkspace(source.getOutput(), context))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_fixed = true;
   if(!ValidatePlans(source.getOutput(), context))
      return(false);
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_CURRENT], 0, m_shape.M,
                                   m_work[DPCC_PW_BASE_ACTIVE], context,
                                   m_work[DPCC_PW_BASE_RETURN], m_work[DPCC_PW_BASE_STATE],
                                   m_work[DPCC_PW_BASE_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   if(!Summarize(false, m_work[DPCC_PW_CURRENT], m_work[DPCC_PW_BASE_ACTIVE]) ||
      !BuildRepairIndexOrder(context, 1))
      return(false);
   uint common = 0;
   if(!ReadCommon(0, common))
      return(false);
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_CURRENT], 0, m_shape.M,
                                   m_work[DPCC_PW_BASE_ACTIVE], context,
                                   m_work[DPCC_PW_BASE_RETURN], m_work[DPCC_PW_BASE_STATE],
                                   m_work[DPCC_PW_BASE_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   if(!Summarize(false, m_work[DPCC_PW_CURRENT], m_work[DPCC_PW_BASE_ACTIVE]) ||
      !BuildRepairIndexOrder(context, 2) || !ReadCommon(0, common))
      return(false);
//--- Each rejection advances one causal cursor; every acceptance consumes an iteration.
   const ulong wave_bound = ((ulong)m_config.projection_max_iterations + 1) *
                            m_config.projection_lookback_steps;
   ulong wave = 0;
   while(common == 1 && wave < wave_bound)
     {
      if(!BuildLocalProposals(context) || !EvaluateProposals(context) || !SelectImprovement() ||
         !ReadCommon(0, common))
         return(false);
      wave++;
     }
   if(common == 1)
      return(Fail(DPCC_OPENCL_ERROR));
   return(VerifyProjection(context));
  }
//+------------------------------------------------------------------+
//| Clears transient generations without changing the frozen profile |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Clear(void)
  {
   m_public_valid = false;
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED || !ValidateWorkspace())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *base[] = {Output, PrevOutput, Gradient};
   for(uint i = 0; i < (uint)ArraySize(base); i++)
      if(!base[i].Fill(0))
         return(Fail(DPCC_OPENCL_ERROR, true));
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      if(!m_work[i].Fill(0))
         return(Fail(DPCC_OPENCL_ERROR, true));
   if(!WriteScales())
      return(Fail(m_status, true));
   m_phase = (m_configured ? DPCC_LAYER_CONFIGURED : DPCC_LAYER_SHAPED);
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Exposes checked transfer only for this exact financial layer     |
//| type                                                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SupportsOpenCLChecked(void) const
  {
   return(Type() == defNeuronPlanProjectionOCL);
  }
//+------------------------------------------------------------------+
//| Discards all transient tensors and requires binding on new owner |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SetOpenCLChecked(COpenCLMy *obj)
  {
   m_public_valid = false;
   m_rollout = NULL;
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked() ||
      m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!ReleaseOwned())
      return(false);
   OpenCL = obj;
   const ulong count = (ulong)m_shape.M * m_shape.H * 6;
   if(!EnsureBuffer(Output, (uint)count) || !EnsureBuffer(PrevOutput, (uint)count) ||
      !EnsureBuffer(Gradient, (uint)count) || !EnsureWorkspace())
      return(Fail(m_status, true));
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Delegates legacy transfer to the checked transient-discard route |
//+------------------------------------------------------------------+
void CNeuronPlanProjectionOCL::SetOpenCL(COpenCLMy *obj)
  {
   SetOpenCLChecked(obj);
  }
//+------------------------------------------------------------------+
//| Dispatches valid objects while invalidating rejected input calls |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::FeedForward(CObject *source, CBufferFloat *second)
  {
   if(CheckPointer(source) == POINTER_INVALID)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CNeuronBaseOCL *typed = dynamic_cast<CNeuronBaseOCL *>(source);
   if(CheckPointer(typed) == POINTER_INVALID)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   return(feedForward(typed, second));
  }
//+------------------------------------------------------------------+
//| Uses the object-aware second-input route without host conversion |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::FeedForwardSecond(CObject *source, CNeuronBaseOCL *second)
  {
   return(FeedForward(source, (CheckPointer(second) != POINTER_INVALID ? second.getOutput() : NULL)));
  }
//+------------------------------------------------------------------+
//| Projection EnsureBuffer contract                                 |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::EnsureBuffer(CBufferFloat *&buffer, const uint count, const float value)
  {
   if(count == 0 || count > INT_MAX || CheckPointer(OpenCL) == POINTER_INVALID)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(CheckPointer(buffer) == POINTER_INVALID)
      buffer = new CBufferFloat();
   if(CheckPointer(buffer) == POINTER_INVALID)
      return(Fail(DPCC_OPENCL_ERROR));
   if(buffer.Total() == (int)count && buffer.GetOpenCL() == OpenCL && buffer.GetIndex() >= 0 &&
      OpenCL.GetBufferSize(buffer.GetIndex()) == (ulong)count * sizeof(float))
      return(true);
   if(buffer.GetIndex() >= 0 && !buffer.BufferFree())
      return(Fail(DPCC_OPENCL_ERROR));
   if(!buffer.BufferInit(count, value))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!buffer.BufferCreate(OpenCL))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Projection ValidateBuffer contract                               |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ValidateBuffer(CBufferFloat *buffer, const ulong count, const bool exact) const
  {
   if(CheckPointer(buffer) == POINTER_INVALID || CheckPointer(OpenCL) == POINTER_INVALID || count == 0 || count > INT_MAX || buffer.GetIndex() < 0 || buffer.GetOpenCL() != OpenCL ||
      (exact ? (ulong)buffer.Total() != count : (ulong)buffer.Total() < count))
      return(false);
   return(OpenCL.GetBufferSize(buffer.GetIndex()) == (ulong)buffer.Total() * sizeof(float));
  }
//+------------------------------------------------------------------+
//| Projection SharesDevice contract                                 |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SharesDevice(CBufferFloat *first, CBufferFloat *second) const
  {
   return(CheckPointer(first) != POINTER_INVALID && CheckPointer(second) != POINTER_INVALID && (first == second ||
          (first.GetIndex() >= 0 && first.GetIndex() == second.GetIndex() &&
           first.GetOpenCL() == second.GetOpenCL())));
  }
//+------------------------------------------------------------------+
//| Projection AppendParameterFingerprint contract                   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(!m_configured || m_phase == DPCC_LAYER_FAILED)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   DPCCRolloutHashWord(Type(), fingerprint);
   DPCCRolloutHashWord(ConfigSignature(m_config), fingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Projection Save contract                                         |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !m_configured || m_phase == DPCC_LAYER_FAILED ||
      !ValidateConfig(m_config))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint header[] = {defNeuronPlanProjectionOCL, (m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? DPCC_HISTORICAL_FX_ENVELOPE_VERSION :
                          (m_config.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 ? DPCC_BINARY_LAYER_VERSION : DPCC_FINANCIAL_ENVELOPE_VERSION)), DPCC_CONTRACT_VERSION,
                          m_shape.K, m_shape.V, m_shape.H, m_shape.D, m_shape.M, m_shape.J, m_shape.E,
                          m_shape.PlanBatch, m_shape.NActions, m_shape.StateSize, m_shape.Embedding,
                          m_shape.Heads, m_shape.KeyDimension, m_shape.Depth,
                          (uint)optimization, iBatch, (uint)m_myIndex
                         };
   for(uint i = 0; i < (uint)ArraySize(header); i++)
      if(FileWriteInteger(file_handle, (int)header[i], INT_VALUE) != INT_VALUE)
         return(Fail(DPCC_INVALID_INPUT));
   const uint integers[] = {(uint)m_config.mode, m_config.projection_max_iterations,
                            m_config.projection_max_evaluations, m_config.projection_lookback_steps,
                            m_config.price_basis, m_config.quote_adapter, m_config.event_model,
                            (uint)m_config.risk_type, m_config.normalization_mode,
                            m_config.loss_coordinates, m_config.randomness_version
                           };
   for(uint i = 0; i < (uint)ArraySize(integers); i++)
      if(FileWriteInteger(file_handle, (int)integers[i], INT_VALUE) != INT_VALUE)
         return(Fail(DPCC_INVALID_INPUT));
   if(m_config.margin_fx_profile != DPCC_MARGIN_EXTERNAL_V2 &&
      (FileWriteInteger(file_handle, (int)m_config.margin_fx_profile, INT_VALUE) != INT_VALUE ||
       FileWriteInteger(file_handle, (int)m_config.account_currency_digits, INT_VALUE) != INT_VALUE))
      return(Fail(DPCC_INVALID_INPUT));
   const double reals[] = {m_config.projection_improvement_atol, m_config.projection_distance_tie_atol,
                           m_config.volume_scale, m_config.price_scale, m_config.money_scale,
                           m_config.max_quote_age_seconds, m_config.max_last_age_seconds
                          };
   for(uint i = 0; i < (uint)ArraySize(reals); i++)
      if(FileWriteDouble(file_handle, reals[i]) != sizeof(double))
         return(Fail(DPCC_INVALID_INPUT));
   const float scalars[] = {m_config.risk_parameter, m_config.mean_weight, m_config.gamma,
                            m_config.probability_tolerance
                           };
   for(uint i = 0; i < (uint)ArraySize(scalars); i++)
      if(FileWriteFloat(file_handle, scalars[i]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   for(uint action = 0; action < 6; action++)
      if(FileWriteFloat(file_handle, m_config.action_bias[action]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   for(uint action = 0; action < 6; action++)
      if(FileWriteFloat(file_handle, m_config.action_scale[action]) != sizeof(float))
         return(Fail(DPCC_INVALID_INPUT));
   if(FileWriteLong(file_handle, (long)ConfigSignature(m_config)) != sizeof(long))
      return(Fail(DPCC_INVALID_INPUT));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Projection Load contract                                         |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::Load(const int file_handle)
  {
   m_public_valid = false;
   m_phase = DPCC_LAYER_FAILED;
   m_configured = false;
//--- The factory has already consumed the exact Type identifier.
   if(file_handle == INVALID_HANDLE || !OpenCL || FileTell(file_handle) > FileSize(file_handle) ||
      FileSize(file_handle) - FileTell(file_handle) < 248)
      return(Fail(DPCC_INVALID_INPUT, true));
   uint header[19];
   for(uint i = 0; i < 19; i++)
      header[i] = (uint)FileReadInteger(file_handle, INT_VALUE);
   if((header[0] != DPCC_BINARY_LAYER_VERSION && header[0] != DPCC_FINANCIAL_ENVELOPE_VERSION &&
       header[0] != DPCC_HISTORICAL_FX_ENVELOPE_VERSION) ||
      header[1] != DPCC_CONTRACT_VERSION || header[18] > INT_MAX ||
      (header[0] != DPCC_BINARY_LAYER_VERSION &&
       FileSize(file_handle) - FileTell(file_handle) < 248 - 19 * INT_VALUE + 2 * INT_VALUE))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   CLayerDescription desc;
   desc.type = defNeuronPlanProjectionOCL;
   desc.window = 6;
   desc.window_out = 13;
   desc.layers = header[15];
   desc.optimization = (ENUM_OPTIMIZATION)header[16];
   desc.batch = header[17];
   desc.activation = None;
   if(ArrayResize(desc.windows, 9) != 9 || ArrayResize(desc.units, 5) != 5 || ArrayResize(desc.heads, 1) != 1)
      return(Fail(DPCC_INVALID_INPUT, true));
   desc.windows[0] = DPCC_CONTRACT_VERSION;
   for(uint i = 0; i < 8; i++)
      desc.windows[i + 1] = header[i + 2];
   desc.units[0] = header[12];
   desc.units[1] = header[13];
   desc.units[2] = header[14];
   desc.units[3] = header[10];
   desc.units[4] = header[11];
   desc.heads[0] = desc.units[1];
   desc.variables = header[3];
   desc.step = header[4];
   const ulong count = (ulong)header[6] * header[4] * 6;
   if(count > INT_MAX)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   desc.count = (uint)count;
   SDPCCConfig candidate;
   ZeroMemory(candidate);
   if(!ReadDescriptor(GetPointer(desc), candidate.shape))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   candidate.contract_version = DPCC_CONTRACT_VERSION;
   candidate.binary_layer_version = DPCC_BINARY_LAYER_VERSION;
   candidate.mode = (ENUM_DPCC_MODE)FileReadInteger(file_handle);
   candidate.projection_max_iterations = (uint)FileReadInteger(file_handle);
   candidate.projection_max_evaluations = (uint)FileReadInteger(file_handle);
   candidate.projection_lookback_steps = (uint)FileReadInteger(file_handle);
   candidate.price_basis = (uint)FileReadInteger(file_handle);
   candidate.quote_adapter = (uint)FileReadInteger(file_handle);
   candidate.event_model = (uint)FileReadInteger(file_handle);
   candidate.risk_type = FileReadInteger(file_handle);
   candidate.normalization_mode = (uint)FileReadInteger(file_handle);
   candidate.loss_coordinates = (uint)FileReadInteger(file_handle);
   candidate.randomness_version = (uint)FileReadInteger(file_handle);
   if(header[0] != DPCC_BINARY_LAYER_VERSION)
     {
      candidate.margin_fx_profile = (uint)FileReadInteger(file_handle);
      candidate.account_currency_digits = (uint)FileReadInteger(file_handle);
      if(candidate.margin_fx_profile != (header[0] == DPCC_HISTORICAL_FX_ENVELOPE_VERSION ?
                                         DPCC_MARGIN_HISTORICAL_CLOSE_V1 : DPCC_MARGIN_FOREX_SPOT_V1))
         return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   candidate.projection_improvement_atol = FileReadDouble(file_handle);
   candidate.projection_distance_tie_atol = FileReadDouble(file_handle);
   candidate.volume_scale = FileReadDouble(file_handle);
   candidate.price_scale = FileReadDouble(file_handle);
   candidate.money_scale = FileReadDouble(file_handle);
   candidate.max_quote_age_seconds = FileReadDouble(file_handle);
   candidate.max_last_age_seconds = FileReadDouble(file_handle);
   candidate.risk_parameter = FileReadFloat(file_handle);
   candidate.mean_weight = FileReadFloat(file_handle);
   candidate.gamma = FileReadFloat(file_handle);
   candidate.probability_tolerance = FileReadFloat(file_handle);
   for(uint action = 0; action < 6; action++)
      candidate.action_bias[action] = FileReadFloat(file_handle);
   for(uint action = 0; action < 6; action++)
      candidate.action_scale[action] = FileReadFloat(file_handle);
   const ulong expected_signature = (ulong)FileReadLong(file_handle);
   if(expected_signature == 0 || expected_signature != ConfigSignature(candidate))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(!Init(0, header[18], OpenCL, GetPointer(desc)))
      return(Fail(m_status, true));
   if(!Configure(candidate))
      return(Fail(m_status, true));
   m_fixed = true;
   if(!Clear())
      return(false);
   UnbindRollout();
   return(true);
  }
//+------------------------------------------------------------------+
//| Projection GetLayerInfo contract                                 |
//+------------------------------------------------------------------+
CLayerDescription *CNeuronPlanProjectionOCL::GetLayerInfo(void)
  {
   if(m_phase == DPCC_LAYER_CONSTRUCTED || m_phase == DPCC_LAYER_FAILED)
      return(NULL);
   CLayerDescription *desc = new CLayerDescription();
   if(!desc)
      return(NULL);
   desc.type = Type();
   desc.count = m_shape.M * m_shape.H * 6;
   desc.window = 6;
   desc.window_out = 13;
   desc.variables = m_shape.V;
   desc.step = m_shape.H;
   desc.layers = 0;
   desc.batch = iBatch;
   desc.optimization = optimization;
   desc.activation = None;
   if(ArrayResize(desc.windows, 9) != 9 || ArrayResize(desc.units, 5) != 5 || ArrayResize(desc.heads, 1) != 1)
     {
      delete desc;
      return(NULL);
     }
   const uint windows[] = {DPCC_CONTRACT_VERSION, m_shape.K, m_shape.V, m_shape.H, m_shape.D,
                           m_shape.M, m_shape.J, m_shape.E, m_shape.PlanBatch
                          };
   for(uint i = 0; i < 9; i++)
      desc.windows[i] = windows[i];
   desc.units[0] = m_shape.Embedding;
   desc.units[1] = m_shape.Heads;
   desc.units[2] = m_shape.KeyDimension;
   desc.units[3] = m_shape.NActions;
   desc.units[4] = m_shape.StateSize;
   desc.heads[0] = m_shape.Heads;
   return(desc);
  }
//+------------------------------------------------------------------+
//| Projection getOutputVal contract                                 |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getOutputVal(float &values[])
  {
   if(m_public_valid)
      return(Output.GetData(values));
   ArrayResize(values, 0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection getOutputVal contract                                 |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getOutputVal(CArrayFloat *values)
  {
   if(m_public_valid)
      return(Output.GetData(values));
   if(values)
      values.Clear();
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection getOutputVal contract                                 |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getOutputVal(vector<float> &values)
  {
   if(m_public_valid)
      return(Output.GetData(values));
   values.Resize(0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection ReadDescriptor contract                               |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ReadDescriptor(CLayerDescription *desc, SDPCCShape &shape) const
  {
   return(DPCCReadFinancialDescriptor(desc, shape, defNeuronPlanProjectionOCL, true));
  }
//+------------------------------------------------------------------+
//| Projection ValidateConfig contract                               |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::ValidateConfig(const SDPCCConfig &config) const
  {
   return(DPCCValidateFixedConfig(m_shape, config));
  }
//+------------------------------------------------------------------+
//| Projection ConfigSignature contract                              |
//+------------------------------------------------------------------+
ulong CNeuronPlanProjectionOCL::ConfigSignature(const SDPCCConfig &config) const
  {
   return(DPCCFixedSignature(config, defNeuronPlanProjectionOCL));
  }
//+------------------------------------------------------------------+
//| Projection ContractSignature contract                            |
//+------------------------------------------------------------------+
ulong CNeuronPlanProjectionOCL::ContractSignature(void) const
  {
   return(m_configured && m_phase != DPCC_LAYER_FAILED ? ConfigSignature(m_config) : 0);
  }
//+------------------------------------------------------------------+
//| Projection getOutputIndex contract                               |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getOutputIndex(void) const
  {
   return(m_public_valid && CheckPointer(Output) != POINTER_INVALID ? Output.GetIndex() : INVALID_HANDLE);
  }
//+------------------------------------------------------------------+
//| Projection getPrevVal contract                                   |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getPrevVal(float &values[])
  {
   ArrayResize(values, 0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection getGradient contract                                  |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getGradient(float &values[])
  {
   ArrayResize(values, 0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection getGradient contract                                  |
//+------------------------------------------------------------------+
int CNeuronPlanProjectionOCL::getGradient(vector<float> &values)
  {
   values.Resize(0);
   return(0);
  }
//+------------------------------------------------------------------+
//| Projection calcInputGradients contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcInputGradients(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcInputGradients contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcInputGradients(CNeuronBaseOCL *source, CBufferFloat *second,
      CBufferFloat *second_gradient, ENUM_ACTIVATION second_activation)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcHiddenGradients contract                          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcHiddenGradients(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcHiddenGradients contract                          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcHiddenGradients(CNeuronBaseOCL *source, CBufferFloat *second,
      CBufferFloat *second_gradient, ENUM_ACTIVATION second_activation)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection updateInputWeights contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::updateInputWeights(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection updateInputWeights contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::updateInputWeights(CNeuronBaseOCL *source, CBufferFloat *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection WeightsUpdateAdam contract                            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::WeightsUpdateAdam(CNeuronBaseOCL *source, float tau)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection WeightsUpdate contract                                |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection numOutputs contract                                   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::numOutputs(const uint outputs, ENUM_OPTIMIZATION optimization_type)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection UpdateInputWeights contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::UpdateInputWeights(CObject *source, CBufferFloat *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection UpdateInputWeightsSecond contract                     |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::UpdateInputWeightsSecond(CObject *source, CNeuronBaseOCL *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection UpdateInputState contract                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::UpdateInputState(CObject *source, CBufferFloat *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection UpdateInputStateSecond contract                       |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::UpdateInputStateSecond(CObject *source, CNeuronBaseOCL *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection CalcHiddenGradients contract                          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::CalcHiddenGradients(CObject *source, CBufferFloat *second,
      CBufferFloat *second_gradient, ENUM_ACTIVATION second_activation)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection CalcHiddenGradientsSecond contract                    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::CalcHiddenGradientsSecond(CObject *source, CNeuronBaseOCL *second)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcOutputGradients contract                          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcOutputGradients(CBufferFloat *target, float &error)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcAlignmentGradient contract                        |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcAlignmentGradient(const CNeuronBaseOCL *source, const bool add)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection calcAlphaGradients contract                           |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::calcAlphaGradients(CNeuronBaseOCL *source)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection SetGradient contract                                  |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SetGradient(CBufferFloat *buffer, bool delete_prev)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection SetGradientIndex contract                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SetGradientIndex(int index)
  {
   return(Fail(DPCC_BACKWARD_UNSUPPORTED));
  }
//+------------------------------------------------------------------+
//| Projection SetOutput contract                                    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SetOutput(CBufferFloat *buffer, bool delete_prev)
  {
   return(Fail(DPCC_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| Projection SwapOutputs contract                                  |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::SwapOutputs(void)
  {
   return(Fail(DPCC_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| TRAIN-01: indexed normal noise for shared DPCC consumers         |
//+------------------------------------------------------------------+
//--- Keys use exact bytes, independent of projection mode and call order.
bool DPCCIndexedNormalV1(const ulong seed, const long decision_time,
                         const uint repeat, const uint candidate,
                         const uint step, const uint element, float &value)
  {
   value = 0;
   uchar bytes[], empty_key[], digest[];
   if(ArrayResize(bytes, 36) != 36)
      return(false);
   bytes[0] = 68;
   bytes[1] = 80;
   bytes[2] = 67;
   bytes[3] = 50;
   const ulong time_word = (ulong)decision_time;
   for(uint i = 0; i < 8; i++)
     {
      bytes[4 + i] = (uchar)((seed >> (8 * i)) & 255);
      bytes[12 + i] = (uchar)((time_word >> (8 * i)) & 255);
     }
   const uint fields[] = {repeat, candidate, step, element};
   for(uint field = 0; field < 4; field++)
      for(uint i = 0; i < 4; i++)
         bytes[20 + field * 4 + i] = (uchar)((fields[field] >> (8 * i)) & 255);
   if(CryptEncode(CRYPT_HASH_SHA256, bytes, empty_key, digest) != 32 || ArraySize(digest) != 32)
      return(false);
   uint first = 0, second = 0;
   for(uint i = 0; i < 4; i++)
     {
      first |= (uint)digest[i] << (8 * i);
      second |= (uint)digest[4 + i] << (8 * i);
     }
   const double u1 = ((double)first + 0.5) / 4294967296.0;
   const double u2 = ((double)second + 0.5) / 4294967296.0;
   const double normal = MathSqrt(-2.0 * MathLog(u1)) * MathCos(6.283185307179586476925286766559 * u2);
   if(!MathIsValidNumber(normal))
      return(false);
   value = (float)normal;
   if(!MathIsValidNumber(value))
     {
      value = 0;
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Build indexed deterministic noise                                |
//+------------------------------------------------------------------+
//--- Canonical C.Noise[m,j,h,a], j=0..J; element=h*6+a.
//--- Prepare once before the device-resident diffusion loop.
bool DPCCBuildIndexedNoiseV1(const ulong seed, const long decision_time,
                             const uint repeat, const uint candidates,
                             const uint steps, const uint horizon, float &values[])
  {
   ArrayResize(values, 0);
   if(candidates == 0 || steps == 0 || horizon == 0)
      return(false);
   const ulong rows = (ulong)horizon * 6;
   const ulong step_count = (ulong)steps + 1;
   if(rows > INT_MAX || step_count > INT_MAX ||
      (ulong)candidates > (ulong)INT_MAX / step_count ||
      (ulong)candidates * step_count > (ulong)INT_MAX / rows)
      return(false);
   const uint count = (uint)((ulong)candidates * step_count * rows);
   if(ArrayResize(values, (int)count) != (int)count)
     {
      ArrayResize(values, 0);
      return(false);
     }
   uint position = 0;
   for(uint m = 0; m < candidates; m++)
      for(uint j = 0; j <= steps; j++)
         for(uint element = 0; element < (uint)rows; element++)
           {
            if(!DPCCIndexedNormalV1(seed, decision_time, repeat, m, j, element, values[position]))
              {
               ArrayResize(values, 0);
               return(false);
              }
            position++;
           }
   return(position == count);
  }
//+------------------------------------------------------------------+
//| Function CNeuronScenarioRolloutOCL::OwnsBufferOrDeviceSlot       |
//+------------------------------------------------------------------+
bool CNeuronScenarioRolloutOCL::OwnsBufferOrDeviceSlot(CBufferFloat *candidate)
  {
   if(CNeuronBaseOCL::OwnsBufferOrDeviceSlot(candidate))
      return(true);
   if(BufferAliases(candidate, m_states))
      return(true);
   if(BufferAliases(candidate, m_validity))
      return(true);
   if(BufferAliases(candidate, m_violations))
      return(true);
   if(BufferAliases(candidate, m_diagnostics))
      return(true);
   if(BufferAliases(candidate, m_return_scratch))
      return(true);
   if(BufferAliases(candidate, m_state_scratch))
      return(true);
   if(BufferAliases(candidate, m_violation_scratch))
      return(true);
   if(BufferAliases(candidate, m_validity_scratch))
      return(true);
   if(BufferAliases(candidate, m_diagnostic_scratch))
      return(true);
   if(BufferAliases(candidate, m_system))
      return(true);
   if(BufferAliases(candidate, m_aggregate))
      return(true);
   if(BufferAliases(candidate, m_slice_active))
      return(true);
   if(BufferAliases(candidate, m_batch_validity))
      return(true);
   if(BufferAliases(candidate, m_batch_diagnostics))
      return(true);
   if(BufferAliases(candidate, m_open_context))
      return(true);
   if(BufferAliases(candidate, m_open_state))
      return(true);
   if(BufferAliases(candidate, m_open_risk))
      return(true);
   return(false);
  }
//+------------------------------------------------------------------+
//| Function CNeuronPlanProjectionOCL::OwnsBufferOrDeviceSlot        |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionOCL::OwnsBufferOrDeviceSlot(CBufferFloat *candidate)
  {
   if(CNeuronBaseOCL::OwnsBufferOrDeviceSlot(candidate))
      return(true);
   for(uint i = 0; i < DPCC_PW_COUNT; i++)
      if(BufferAliases(candidate, m_work[i]))
         return(true);
   return(false);
  }
//+------------------------------------------------------------------+
//| Initializes the independent projection trace lifecycle           |
//+------------------------------------------------------------------+
CNeuronPlanProjectionTraceOCL::CNeuronPlanProjectionTraceOCL(void)
  {
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      m_trace[i] = NULL;
   m_trace_capacity = 0;
   m_trace_frames = 0;
   m_trace_next = 0;
   m_trace_slot = 0;
   m_trace_decision = 0;
   m_trace_live = false;
   m_trace_failed = false;
   m_trace_constructing = false;
   m_trace_executing = false;
   m_trace_record = false;
  }
//+------------------------------------------------------------------+
//| Destroys only the trace-owned tensors                            |
//+------------------------------------------------------------------+
CNeuronPlanProjectionTraceOCL::~CNeuronPlanProjectionTraceOCL(void)
  {
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      DeleteObj(m_trace[i]);
  }
//+------------------------------------------------------------------+
//| Invalidates every frame after a trace or device contract failure |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::TraceFail(const ENUM_DPCC_STATUS status)
  {
   m_public_valid = false;
   m_trace_failed = true;
   m_trace_executing = false;
   ArrayInitialize(m_trace_slots, 0);
   return(Fail(status));
  }
//+------------------------------------------------------------------+
//| Checks all products before allocating transient frame storage    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::TraceSizes(ulong &counts[]) const
  {
   if(m_trace_capacity == 0 || m_trace_capacity > INT_MAX ||
      ArrayResize(counts, DPCC_TW_COUNT) != DPCC_TW_COUNT)
      return(false);
   const ulong scalars = (ulong)m_shape.M * m_shape.H * 6;
   if(scalars == 0 || scalars > (ulong)INT_MAX / 2 / m_trace_capacity)
      return(false);
   const ulong sizes[] = {scalars * 2, scalars * 2, (ulong)m_shape.M * 9 * 6 * 2,
                          (ulong)m_shape.M * 9 * 6 * 2, scalars * 2 * m_trace_capacity,
                          scalars * 2 * m_trace_capacity, (ulong)m_shape.M * 2, scalars, 1
                         };
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
     {
      if(sizes[i] == 0 || sizes[i] > INT_MAX)
         return(false);
      counts[i] = sizes[i];
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocates int32 map slots and float32 coefficients on the device |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::EnsureTrace(void)
  {
   ulong counts[];
   if(!TraceSizes(counts))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      if(!EnsureBuffer(m_trace[i], (uint)counts[i]))
         return(TraceFail(m_status));
   if(ArrayResize(m_trace_slots, (int)m_trace_capacity) != (int)m_trace_capacity)
      return(TraceFail(DPCC_OPENCL_ERROR));
   ArrayInitialize(m_trace_slots, 0);
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates trace ownership and excludes caller and parent aliases |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::ValidateTrace(CBufferFloat *first, CBufferFloat *second) const
  {
   ulong counts[];
   if(Type() != defNeuronPlanProjectionTraceOCL || !TraceSizes(counts) ||
      ArraySize(m_trace_slots) != (int)m_trace_capacity)
      return(false);
   CBufferFloat *base[] = {Output, PrevOutput, Gradient, Weights,
                           DeltaWeights, FirstMomentum, SecondMomentum
                          };
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
     {
      if(!ValidateBuffer(m_trace[i], counts[i]) ||
         SharesDevice(first, m_trace[i]) || SharesDevice(second, m_trace[i]))
         return(false);
      for(uint j = 0; j < i; j++)
         if(SharesDevice(m_trace[i], m_trace[j]))
            return(false);
      for(uint j = 0; j < DPCC_PW_COUNT; j++)
         if(SharesDevice(m_trace[i], m_work[j]))
            return(false);
      for(uint j = 0; j < (uint)ArraySize(base); j++)
         if(SharesDevice(m_trace[i], base[j]))
            return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Releases buffers through their creating OpenCL owners            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::FreeTraceBuffers(void)
  {
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      if(CheckPointer(m_trace[i]) != POINTER_INVALID && m_trace[i].GetIndex() >= 0)
         if(!m_trace[i].BufferFree())
            return(TraceFail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Adapts the new descriptor without altering the object's Type     |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::Init(uint numOutputs, uint index, COpenCLMy *open_cl,
      CLayerDescription *desc)
  {
   if(CheckPointer(desc) == POINTER_INVALID || ArraySize(desc.windows) != 9)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   return(InitTrace(index, open_cl, desc, desc.windows[6]));
  }
//+------------------------------------------------------------------+
//| Initializes the parent financial path and independent trace tape |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::InitTrace(uint index, COpenCLMy *open_cl,
      CLayerDescription *desc, const uint max_frames)
  {
   if(m_trace_live || m_trace_executing || Type() != defNeuronPlanProjectionTraceOCL ||
      CheckPointer(desc) == POINTER_INVALID || desc.type != defNeuronPlanProjectionTraceOCL ||
      CheckPointer(open_cl) == POINTER_INVALID || max_frames == 0 || max_frames > INT_MAX)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   CLayerDescription parent;
   if(!parent.Copy(desc))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   parent.type = defNeuronPlanProjectionOCL;
   SDPCCShape candidate;
   if(!ReadDescriptor(GetPointer(parent), candidate))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   const ulong scalars = (ulong)candidate.M * candidate.H * 6;
   if(scalars == 0 || scalars > (ulong)INT_MAX / 2 / max_frames ||
      (ulong)candidate.M * 9 * 6 * 2 > INT_MAX)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
//--- Parent Init calls virtual Clear before the trace buffers exist.
   if(!FreeTraceBuffers())
      return(false);
   m_trace_capacity = max_frames;
   m_trace_constructing = true;
   const bool initialized = CNeuronPlanProjectionOCL::Init(0, index, open_cl, GetPointer(parent));
   m_trace_constructing = false;
   if(!initialized)
      return(TraceFail(m_status));
   m_trace_failed = false;
   if(!EnsureTrace())
      return(false);
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Prevents financial reconfiguration while decision frames are     |
//| live                                                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::Configure(const SDPCCConfig &config)
  {
   if(m_trace_live || m_trace_failed || !ValidateTrace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   return(CNeuronPlanProjectionOCL::Configure(config));
  }
//+------------------------------------------------------------------+
//| Binds the published financial evaluator only outside a live      |
//| trace                                                            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::BindRollout(CNeuronScenarioRolloutOCL *rollout)
  {
   if(m_trace_live || m_trace_failed || !ValidateTrace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   return(CNeuronPlanProjectionOCL::BindRollout(rollout));
  }
//+------------------------------------------------------------------+
//| Detaches the borrowed evaluator and invalidates outstanding maps |
//+------------------------------------------------------------------+
void CNeuronPlanProjectionTraceOCL::UnbindRollout(void)
  {
   if(m_trace_live)
      TraceFail(DPCC_TRACE_CONTRACT_MISMATCH);
   CNeuronPlanProjectionOCL::UnbindRollout();
  }
//+------------------------------------------------------------------+
//| Starts one decision with explicitly bounded frame slots          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::BeginTrace(const ulong decision_id, const uint frames)
  {
   m_public_valid = false;
   if(m_trace_live || m_trace_failed || m_trace_executing || decision_id == 0 || frames == 0 ||
      frames > m_trace_capacity || !m_configured || m_phase != DPCC_LAYER_READY)
     {
      PrintFormat("DPCC_TRACE_BEGIN_REJECT cause=state live=%d failed=%d executing=%d decision_id=%I64u frames=%u capacity=%u configured=%d phase=%d",
                  m_trace_live, m_trace_failed, m_trace_executing, decision_id, frames, m_trace_capacity,
                  m_configured, (int)m_phase);
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
     }
   if(CheckPointer(m_rollout) == POINTER_INVALID)
     {
      PrintFormat("DPCC_TRACE_BEGIN_REJECT cause=rollout_pointer decision_id=%I64u", decision_id);
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
     }
   if(!m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL))
     {
      PrintFormat("DPCC_TRACE_BEGIN_REJECT cause=rollout_contract decision_id=%I64u", decision_id);
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
     }
   if(!ValidateWorkspace())
     {
      PrintFormat("DPCC_TRACE_BEGIN_REJECT cause=workspace decision_id=%I64u", decision_id);
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
     }
   if(!ValidateTrace())
     {
      PrintFormat("DPCC_TRACE_BEGIN_REJECT cause=trace_buffers decision_id=%I64u", decision_id);
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
     }
   ArrayInitialize(m_trace_slots, 0);
   if(!m_trace[DPCC_TW_ERROR].Fill(0))
      return(TraceFail(DPCC_OPENCL_ERROR));
   m_trace_decision = decision_id;
   m_trace_frames = frames;
   m_trace_next = 0;
   m_trace_live = true;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs one trace kernel without writing financial state            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::TraceStage(const uint stage, CBufferFloat *context)
  {
   const int kernel = def_k_DPCCProjectionTrace;
   CBufferFloat *buffers[] = {m_trace[DPCC_TW_CURRENT_INDEX], m_trace[DPCC_TW_CURRENT_COEFFICIENT],
                              m_trace[DPCC_TW_PROPOSAL_INDEX], m_trace[DPCC_TW_PROPOSAL_COEFFICIENT],
                              m_trace[DPCC_TW_FRAME_INDEX], m_trace[DPCC_TW_FRAME_COEFFICIENT],
                              m_trace[DPCC_TW_WAVE], m_trace[DPCC_TW_ERROR], m_trace[DPCC_TW_COMMON],
                              m_work[DPCC_PW_CURRENT], context, m_work[DPCC_PW_CONTROL],
                              m_work[DPCC_PW_ORDER], m_work[DPCC_PW_BASE_SUMMARY]
                             };
   const uint values[] = {m_shape.M, m_shape.H, stage, m_trace_slot,
                          (uint)m_layout.offsets[1], (uint)m_layout.offsets[2],
                          (uint)m_layout.offsets[3], m_config.projection_max_evaluations
                         };
   if(!BindBuffers(kernel, buffers))
      return(TraceFail(m_status));
   if(!BindIntegers(kernel, 14, values))
      return(TraceFail(m_status));
   if(!ExecuteRows(kernel, m_shape.M))
      return(TraceFail(m_status));
   return(true);
  }
//+------------------------------------------------------------------+
//| Reads a single device-reduced trace contract verdict             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::CheckTrace(void)
  {
   if(!TraceStage(5, m_work[DPCC_PW_CURRENT]))
      return(false);
   if(!m_trace[DPCC_TW_COMMON].BufferRead())
      return(TraceFail(DPCC_OPENCL_ERROR));
   if(m_trace[DPCC_TW_COMMON][0] != 0)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   return(true);
  }
//+------------------------------------------------------------------+
//| Publishes one frame only after its financial and trace checks    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::ForwardFrame(const uint slot, CNeuronBaseOCL *plans,
      CBufferFloat *context, const bool record_gradient)
  {
//--- Frozen inference uses the same financial projector without creating a training trace.
   if(!record_gradient && !m_trace_live && !m_trace_failed && !m_trace_executing)
     {
      m_public_valid = false;
      return(CNeuronPlanProjectionOCL::feedForward(plans, context));
     }
   if(!m_trace_live || m_trace_failed || m_trace_executing || slot >= m_trace_frames ||
      slot != m_trace_next || m_trace_slots[slot] != 0 || CheckPointer(plans) == POINTER_INVALID ||
      !ValidateTrace(plans.getOutput(), context))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   m_trace_slot = slot;
   m_trace_record = record_gradient;
   m_trace_executing = true;
   const bool accepted = feedForward(plans, context);
   m_trace_executing = false;
   if(!accepted)
      return(TraceFail(m_status));
   m_trace_slots[slot] = (record_gradient ? 2 : 1);
   m_trace_next++;
   return(true);
  }
//+------------------------------------------------------------------+
//| Rejects context-free or unframed entry into the trace path       |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::feedForward(CNeuronBaseOCL *source)
  {
   return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| Orchestrates the identical published financial stage sequence    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::feedForward(CNeuronBaseOCL *source, CBufferFloat *context)
  {
   m_public_valid = false;
   m_status = DPCC_OK;
   if(!m_trace_executing || !m_trace_live || m_trace_failed || !m_configured ||
      m_phase == DPCC_LAYER_FAILED || CheckPointer(source) == POINTER_INVALID ||
      CheckPointer(m_rollout) == POINTER_INVALID ||
      !m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) ||
      !ValidateBuffer(source.getOutput(), (ulong)m_shape.M * m_shape.H * 6) ||
      !ValidateBuffer(context, m_layout.total) || !ValidateWorkspace(source.getOutput(), context) ||
      !ValidateTrace(source.getOutput(), context))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   m_fixed = true;
   if(!ValidatePlans(source.getOutput(), context))
      return(false);
   if(m_trace_record && !TraceStage(0, context))
      return(false);
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_CURRENT], 0, m_shape.M,
                                   m_work[DPCC_PW_BASE_ACTIVE], context,
                                   m_work[DPCC_PW_BASE_RETURN], m_work[DPCC_PW_BASE_STATE],
                                   m_work[DPCC_PW_BASE_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   if(!Summarize(false, m_work[DPCC_PW_CURRENT], m_work[DPCC_PW_BASE_ACTIVE]))
      return(false);
   if(m_trace_record && !TraceStage(1, context))
      return(false);
   if(!BuildRepairIndexOrder(context, 1))
      return(false);
   uint common = 0;
   if(!ReadCommon(0, common))
      return(false);
   if(!m_rollout.EvaluatePlanBatch(m_work[DPCC_PW_CURRENT], 0, m_shape.M,
                                   m_work[DPCC_PW_BASE_ACTIVE], context,
                                   m_work[DPCC_PW_BASE_RETURN], m_work[DPCC_PW_BASE_STATE],
                                   m_work[DPCC_PW_BASE_VIOLATION]))
      return(Fail(m_rollout.LastStatus()));
   if(!Summarize(false, m_work[DPCC_PW_CURRENT], m_work[DPCC_PW_BASE_ACTIVE]) ||
      !BuildRepairIndexOrder(context, 2) || !ReadCommon(0, common))
      return(false);
//--- Trace before the parent can replace the cursor/order on acceptance.
   const ulong wave_bound = ((ulong)m_config.projection_max_iterations + 1) *
                            m_config.projection_lookback_steps;
   ulong wave = 0;
   while(common == 1 && wave < wave_bound)
     {
      if(m_trace_record && !TraceStage(2, context))
         return(false);
      if(!BuildLocalProposals(context) || !EvaluateProposals(context) || !SelectImprovement())
         return(false);
      if(m_trace_record && (!TraceStage(3, context) || !CheckTrace()))
         return(false);
      if(!ReadCommon(0, common))
         return(false);
      wave++;
     }
   if(common == 1)
      return(Fail(DPCC_OPENCL_ERROR));
   if(!VerifyProjection(context))
      return(false);
   if(m_trace_record && (!TraceStage(4, context) || !CheckTrace()))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Replaces caller input gradients with the deterministic map VJP   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::BackwardFrame(const uint slot, CBufferFloat *gradient_out,
      CBufferFloat *gradient_in)
  {
   const ulong count = (ulong)m_shape.M * m_shape.H * 6;
   if(!m_trace_live || m_trace_failed || m_trace_executing || slot >= m_trace_frames ||
      m_trace_slots[slot] != 2 || !ValidateBuffer(gradient_out, count) ||
      !ValidateBuffer(gradient_in, count) || SharesDevice(gradient_out, gradient_in) ||
      !ValidateWorkspace(gradient_out, gradient_in) || !ValidateTrace(gradient_out, gradient_in))
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   m_trace_slot = slot;
   const int kernel = def_k_DPCCProjectionTraceVJP;
   CBufferFloat *buffers[] = {m_trace[DPCC_TW_FRAME_INDEX], m_trace[DPCC_TW_FRAME_COEFFICIENT],
                              gradient_out, gradient_in, m_trace[DPCC_TW_ERROR]
                             };
   const uint values[] = {m_shape.M, m_shape.H, slot};
   if(!BindBuffers(kernel, buffers))
      return(TraceFail(m_status));
   if(!BindIntegers(kernel, 5, values))
      return(TraceFail(m_status));
   if(!ExecuteRows(kernel, (uint)count))
      return(TraceFail(m_status));
   if(!CheckTrace())
      return(false);
   m_trace_slots[slot] = 3;
   return(true);
  }
//+------------------------------------------------------------------+
//| Invalidates runtime maps without changing financial              |
//| configuration                                                    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::ReleaseTrace(void)
  {
   if(m_trace_executing)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   ArrayInitialize(m_trace_slots, 0);
   m_trace_decision = 0;
   m_trace_frames = 0;
   m_trace_next = 0;
   m_trace_slot = 0;
   m_trace_live = false;
   m_trace_failed = false;
   m_trace_record = false;
   m_public_valid = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears parent and trace tensors only outside live frames         |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::Clear(void)
  {
   if(Type() != defNeuronPlanProjectionTraceOCL || m_trace_live || m_trace_executing)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   if(m_trace_constructing)
      return(CNeuronPlanProjectionOCL::Clear());
   if(!ValidateTrace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   const bool rollout_ready = (m_configured && m_phase == DPCC_LAYER_READY &&
                               CheckPointer(m_rollout) != POINTER_INVALID &&
                               m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL));
   if(!CNeuronPlanProjectionOCL::Clear())
      return(TraceFail(m_status));
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      if(!m_trace[i].Fill(0))
         return(TraceFail(DPCC_OPENCL_ERROR));
   if(!ReleaseTrace())
      return(false);
   if(rollout_ready)
      m_phase = DPCC_LAYER_READY;
   return(true);
  }
//+------------------------------------------------------------------+
//| Advertises checked transfer for the exact trace class            |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::SupportsOpenCLChecked(void) const
  {
   return(Type() == defNeuronPlanProjectionTraceOCL);
  }
//+------------------------------------------------------------------+
//| Recreates all transient tensors and requires a fresh rollout     |
//| bind                                                             |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::SetOpenCLChecked(COpenCLMy *obj)
  {
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked() || m_trace_live ||
      m_trace_failed || m_trace_executing || m_phase == DPCC_LAYER_CONSTRUCTED ||
      m_phase == DPCC_LAYER_FAILED || !ValidateWorkspace() || !ValidateTrace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   m_public_valid = false;
   m_rollout = NULL;
   if(!FreeTraceBuffers())
      return(false);
   if(!ReleaseOwned())
      return(TraceFail(m_status));
   OpenCL = obj;
   const uint count = m_shape.M * m_shape.H * 6;
   if(!EnsureBuffer(Output, count))
      return(TraceFail(m_status));
   if(!EnsureBuffer(PrevOutput, count))
      return(TraceFail(m_status));
   if(!EnsureBuffer(Gradient, count))
      return(TraceFail(m_status));
   if(!EnsureWorkspace())
      return(TraceFail(m_status));
   if(!EnsureTrace())
      return(false);
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Delegates the legacy entry to the checked trace transfer         |
//+------------------------------------------------------------------+
void CNeuronPlanProjectionTraceOCL::SetOpenCL(COpenCLMy *obj)
  {
   SetOpenCLChecked(obj);
  }
//+------------------------------------------------------------------+
//| Saves only the trace ABI/capacity and native financial model     |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || m_trace_live || m_trace_failed ||
      m_trace_executing || !m_configured || m_phase == DPCC_LAYER_FAILED ||
      !ValidateConfig(m_config) || !ValidateTrace() || !ValidateWorkspace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   const uint header[] = {defNeuronPlanProjectionTraceOCL, DPCC_PROJECTION_TRACE_ABI, m_trace_capacity};
   for(uint i = 0; i < (uint)ArraySize(header); i++)
      if(FileWriteInteger(file_handle, (int)header[i], INT_VALUE) != INT_VALUE)
         return(TraceFail(DPCC_INVALID_INPUT));
   return(CNeuronPlanProjectionOCL::Save(file_handle));
  }
//+------------------------------------------------------------------+
//| Loads an explicit parent envelope and recreates an empty trace   |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::Load(const int file_handle)
  {
   m_public_valid = false;
   if(file_handle == INVALID_HANDLE || Type() != defNeuronPlanProjectionTraceOCL ||
      m_trace_live || m_trace_executing || CheckPointer(OpenCL) == POINTER_INVALID ||
      FileTell(file_handle) > FileSize(file_handle) || FileSize(file_handle) - FileTell(file_handle) < 12)
      return(TraceFail(DPCC_INVALID_INPUT));
//--- CNet consumes the outer Type in CreateElement; direct callers may leave it unread.
   const ulong position = FileTell(file_handle);
   uint wrapper_type = (uint)FileReadInteger(file_handle, INT_VALUE);
   if(wrapper_type != defNeuronPlanProjectionTraceOCL)
     {
      if(position < INT_VALUE || !FileSeek(file_handle, (long)(position - INT_VALUE), SEEK_SET))
         return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
      wrapper_type = (uint)FileReadInteger(file_handle, INT_VALUE);
     }
   if(wrapper_type != defNeuronPlanProjectionTraceOCL)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   const uint abi = (uint)FileReadInteger(file_handle, INT_VALUE);
   const uint frames = (uint)FileReadInteger(file_handle, INT_VALUE);
   if(abi != DPCC_PROJECTION_TRACE_ABI || frames == 0 || frames > INT_MAX ||
      FileTell(file_handle) > FileSize(file_handle) || FileSize(file_handle) - FileTell(file_handle) < INT_VALUE)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
//--- The parent loader receives its payload after its validated nested Type.
   const uint parent_type = (uint)FileReadInteger(file_handle, INT_VALUE);
   if(parent_type != defNeuronPlanProjectionOCL)
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   if(!FreeTraceBuffers())
      return(false);
   m_trace_capacity = frames;
   m_trace_constructing = true;
   const bool loaded = CNeuronPlanProjectionOCL::Load(file_handle);
   m_trace_constructing = false;
   if(!loaded)
      return(TraceFail(m_status));
   m_trace_failed = false;
   if(!EnsureTrace())
      return(false);
   if(!Clear())
      return(false);
   CNeuronPlanProjectionOCL::UnbindRollout();
   return(true);
  }
//+------------------------------------------------------------------+
//| Extends model identity without hashing transient runtime maps    |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(m_trace_failed || !ValidateTrace())
      return(TraceFail(DPCC_TRACE_CONTRACT_MISMATCH));
   if(!CNeuronPlanProjectionOCL::AppendParameterFingerprint(fingerprint))
      return(false);
   DPCCRolloutHashWord(DPCC_PROJECTION_TRACE_ABI, fingerprint);
   DPCCRolloutHashWord(m_trace_capacity, fingerprint);
   return(true);
  }
//+------------------------------------------------------------------+
//| Includes the sparse maps in the library ownership audit          |
//+------------------------------------------------------------------+
bool CNeuronPlanProjectionTraceOCL::OwnsBufferOrDeviceSlot(CBufferFloat *candidate)
  {
   for(uint i = 0; i < DPCC_TW_COUNT; i++)
      if(BufferAliases(candidate, m_trace[i]))
         return(true);
   return(CNeuronPlanProjectionOCL::OwnsBufferOrDeviceSlot(candidate));
  }
//+------------------------------------------------------------------+
//| DPCCBuildStepPositionalEncoding follows Context_Dictionary.json  |
//+------------------------------------------------------------------+
bool DPCCBuildStepPositionalEncoding(const uint j, const uint J, float &encoding[], string &reason)
  {
//--- A failed request leaves no stale positional values for its caller.
   ArrayResize(encoding, 0);
   reason = "";
   if(J == 0)
      reason = "J";
   else
      if(j == 0 || j > J)
         reason = "j";
   if(StringLen(reason) != 0)
      return(false);
   if(ArrayResize(encoding, 16) != 16)
     {
      ArrayResize(encoding, 0);
      reason = "allocation";
      return(false);
     }
   const double u = (double)j / (double)J;
   const double two_pi = 6.28318530717958647692;
   for(int i = 0; i < 8; i++)
     {
      const double frequency = (double)(1 << i);
      const double angle = two_pi * u * frequency;
      const double sine = MathSin(angle);
      const double cosine = MathCos(angle);
      const float sine_value = (float)sine;
      const float cosine_value = (float)cosine;
      if(!MathIsValidNumber(angle) || !MathIsValidNumber(sine) || !MathIsValidNumber(cosine) ||
         !MathIsValidNumber(sine_value) || !MathIsValidNumber(cosine_value))
        {
         ArrayResize(encoding, 0);
         reason = "numeric";
         return(false);
        }
      encoding[2 * i] = sine_value;
      encoding[2 * i + 1] = cosine_value;
     }
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::OwnsBufferOrDeviceSlot                  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::OwnsBufferOrDeviceSlot(CBufferFloat *candidate)
  {
   if(CNeuronBaseOCL::OwnsBufferOrDeviceSlot(candidate))
      return(true);
   if(m_actor_projector && m_actor_projector.OwnsBufferOrDeviceSlot(candidate))
      return(true);
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      if(BufferAliases(candidate, m_work[i]))
         return(true);
   CBufferFloat *action_buffers[] = {m_action_context, m_action_condition, m_action_noise, m_action_current,
                                     m_action_next, m_action_physical, m_action_epsilon, m_action_probability_frames,
                                     m_action_projected_frames, m_action_flags, m_action_scalar, m_action_status,
                                     m_action_candidate_status, m_action_returns, m_action_account13, m_action_q2_scores, m_action_rank_scores,
                                     m_action_numeric_flags
                                    };
   for(uint i = 0; i < (uint)ArraySize(action_buffers); i++)
      if(BufferAliases(candidate, action_buffers[i]))
         return(true);
   CNeuronBaseOCL *children[] = {m_decoder, m_denoiser, m_rollout, m_projector, m_normalizer, m_denorm, m_decoder_source, m_plan_source};
   for(uint i = 0; i < (uint)ArraySize(children); i++)
      if(CheckPointer(children[i]) != POINTER_INVALID && children[i].OwnsBufferOrDeviceSlot(candidate))
         return(true);
   return(false);
  }
//--- WORK-ONLY. Protected CNeuronDPCCOCL methods, not a new primitive class.
//--- Root adds these declarations to the actual composite class. Three tentative
//--- kernel macros 353..355 are supplied in composite_registration.fragment.mqh.
//--- Composite owner allocates flags=max(C.total,R.total,KHF+61,KH4), scalar=1,
//--- All work/result buffers are exclusively owned.
/*
   bool CompositeBuffers(CBufferFloat *&buffers[], const ulong &counts[],
                         const bool flags_last = true);
   bool CompositeDispatch(const int kernel, CBufferFloat *&buffers[],
                          const uint &integers[], const float &reals[], const uint count);
   bool CompositeReduce(CBufferFloat *flags, CBufferFloat *scalar, const uint count);
   bool SnapshotCompositeContext(CBufferFloat *source, CBufferFloat *snapshot,
                                 CBufferFloat *flags, CBufferFloat *scalar, const uint count);
   bool PrepareCompositeNorm(CBufferFloat *context, CNeuronBaseOCL *decoder,
                             CNeuronRevINDenormOCL *denorm, CBufferFloat *flags,
                             CBufferFloat *scalar, const SDPCCLayout &c_layout,
                             const SDPCCShape &shape);
   bool PackCompositeR(CBufferFloat *context, CBufferFloat *native,
                       CBufferFloat *result, CBufferFloat *flags, CBufferFloat *scalar,
                       const SDPCCLayout &c_layout, const SDPCCLayout &r_layout,
                       const SDPCCConfig &config);
   bool PackCompositeCondition(CBufferFloat *context, CBufferFloat *account13,
                               CBufferFloat *condition, CBufferFloat *flags,
                               CBufferFloat *scalar, const SDPCCLayout &c_layout,
                               const SDPCCConfig &config);
   bool SetCompositeStep(const uint j, const uint J, CBufferFloat *condition,
                         const uint tokens);
*/
//+------------------------------------------------------------------+
//| Calculate the shared composite layouts                           |
//+------------------------------------------------------------------+
//--- Host-only layout. Share this single mapping with StudyForecast/Study and
//--- production composite; do not create a second producer feature mapping.
bool DPCCCompositeLayouts(const SDPCCShape &shape, SDPCCLayout &c_layout,
                          SDPCCLayout &r_layout, uint &tokens)
  {
   ZeroMemory(c_layout);
   ZeroMemory(r_layout);
   tokens = 0;
   const uint dims[] = {shape.K, shape.V, shape.H, shape.D, shape.M, shape.J, shape.E};
   for(uint i = 0; i < (uint)ArraySize(dims); i++)
      if(dims[i] == 0 || dims[i] > 16777216)
         return(false);
   if(shape.K > 64 || shape.J > 64 || shape.NActions != 6 || shape.StateSize != 13)
      return(false);
   ulong kvh, kh, z, noise, calendar, count;
   if(!DPCCRolloutProduct(shape.K, shape.H, kh) ||
      !DPCCRolloutProduct(kh, shape.V, kvh) ||
      !DPCCRolloutProduct(kvh, shape.D, z) ||
      !DPCCRolloutProduct(kh, (ulong)shape.E * 6, calendar) ||
      !DPCCRolloutProduct((ulong)shape.M * (shape.J + 1), (ulong)shape.H * 6, noise) ||
      !DPCCRolloutProduct(kh, (ulong)shape.V * (shape.D + 1) + 1, count) ||
      count > (ulong)INT_MAX - 61)
      return(false);
   tokens = (uint)count;
   c_layout.version = DPCC_CONTRACT_VERSION;
   const ulong sizes[] = {z, kvh, shape.K, 16, 24, 48, 24, kh * 24, shape.H, calendar, noise};
   for(uint i = 0; i < 11; i++)
     {
      c_layout.offsets[i] = c_layout.total;
      c_layout.lengths[i] = sizes[i];
      if(sizes[i] > (ulong)INT_MAX - c_layout.total)
         return(false);
      c_layout.total += sizes[i];
     }
   SDPCCShape financial = shape;
   financial.Depth = 0;
   if(!DPCCBuildRolloutLayout(financial, r_layout))
      return(false);
//--- Byte bound for legacy owned RevIN: nine float options per KH4 element.
   const ulong float_limit = (ulong)UINT_MAX / sizeof(float);
   return(kh * 4 <= (ulong)UINT_MAX / (9 * sizeof(float)) &&
          c_layout.total <= float_limit && r_layout.total <= float_limit &&
          (ulong)tokens + 61 <= float_limit);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeBuffers                        |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeBuffers(CBufferFloat *&buffers[], const ulong &counts[],
                                      const bool flags_last)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || ArraySize(buffers) != ArraySize(counts))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(int i = 0; i < ArraySize(buffers); i++)
     {
      if(CheckPointer(buffers[i]) == POINTER_INVALID || counts[i] == 0 || counts[i] > INT_MAX ||
         buffers[i].GetOpenCL() != OpenCL || buffers[i].GetIndex() < 0 ||
         (ulong)buffers[i].Total() < counts[i] ||
         (!(flags_last && i + 1 == ArraySize(buffers)) && (ulong)buffers[i].Total() != counts[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
      for(int k = 0; k < i; k++)
         if(buffers[i] == buffers[k] || buffers[i].GetIndex() == buffers[k].GetIndex())
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeDispatch                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeDispatch(const int kernel, CBufferFloat *&buffers[],
                                       const uint &integers[], const float &reals[], const uint count)
  {
   uint arg = 0;
   for(int i = 0; i < ArraySize(buffers); i++, arg++)
      if(!OpenCL.SetArgumentBuffer(kernel, arg, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   for(int i = 0; i < ArraySize(integers); i++, arg++)
      if(!OpenCL.SetArgument(kernel, arg, integers[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   for(int i = 0; i < ArraySize(reals); i++, arg++)
      if(!OpenCL.SetArgument(kernel, arg, reals[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {count};
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeReduce                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeReduce(CBufferFloat *flags, CBufferFloat *scalar, const uint count)
  {
   if(!flags || !scalar || scalar.Total() != 1 || flags.Total() < (int)count || count == 0 ||
      scalar.GetOpenCL() != OpenCL || flags.GetOpenCL() != OpenCL ||
      scalar.GetIndex() < 0 || flags.GetIndex() < 0 || flags.GetIndex() == scalar.GetIndex())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {flags, scalar};
   const uint integers[] = {count};
   float reals[];
   if(!CompositeDispatch(def_k_DPCCOperatorReduce, buffers, integers, reals, 1) || !scalar.BufferRead())
      return(Fail(DPCC_OPENCL_ERROR));
   const float value = scalar.At(0);
   if(value == 0.0f)
      return(true);
   if(value == 3.0f)
      return(Fail(DPCC_INVALID_INPUT));
   if(value == 5.0f)
      return(Fail(DPCC_NUMERIC_ERROR));
   return(Fail(DPCC_CONTRACT_MISMATCH));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SnapshotCompositeContext                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SnapshotCompositeContext(CBufferFloat *source, CBufferFloat *snapshot,
      CBufferFloat *flags, CBufferFloat *scalar, const uint count)
  {
   CBufferFloat *buffers[] = {source, snapshot, flags};
   const ulong sizes[] = {count, count, count};
   if(!CompositeBuffers(buffers, sizes))
      return(false);
   const uint integers[] = {uint(None), count};
   float reals[];
   return(CompositeDispatch(def_k_DPCCActivationForward, buffers, integers, reals, count) &&
          CompositeReduce(flags, scalar, count));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::PrepareCompositeNorm                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PrepareCompositeNorm(CBufferFloat *context, CNeuronBaseOCL *decoder,
      CNeuronRevINDenormOCL *denorm, CBufferFloat *flags,
      CBufferFloat *scalar, const SDPCCLayout &c_layout,
      const SDPCCShape &shape)
  {
   SDPCCLayout checked_c, checked_r;
   uint tokens;
   if(!DPCCCompositeLayouts(shape, checked_c, checked_r, tokens) ||
      c_layout.total != checked_c.total || c_layout.offsets[3] != checked_c.offsets[3] ||
      !decoder || !denorm || !denorm.OwnedReady() || decoder.Activation() != None)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint count = (uint)checked_r.lengths[0];
   CBufferFloat *buffers[] = {context, decoder.getOutput(), denorm.getBatchOptions(), flags};
   const ulong sizes[] = {checked_c.total, count, (ulong)count * 9, count};
   if(!CompositeBuffers(buffers, sizes))
      return(false);
   const uint integers[] = {(uint)checked_c.offsets[3], count};
   float reals[];
   if(!CompositeDispatch(def_k_DPCCCompositeNormOptions, buffers, integers, reals, count) ||
      !CompositeReduce(flags, scalar, count))
      return(false);
//--- Existing owned RevIN performs the actual affine restoration, unchanged.
   if(!denorm.FeedForward(decoder) || !denorm.OwnedOutputValid())
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::PackCompositeR                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PackCompositeR(CBufferFloat *context, CBufferFloat *native,
                                    CBufferFloat *result, CBufferFloat *flags, CBufferFloat *scalar,
                                    const SDPCCLayout &c_layout, const SDPCCLayout &r_layout,
                                    const SDPCCConfig &config)
  {
   SDPCCLayout c, r;
   uint tokens;
   if(!DPCCCompositeLayouts(config.shape, c, r, tokens) ||
      c_layout.total != c.total || r_layout.total != r.total)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {context, native, result, flags};
   const ulong sizes[] = {c.total, r.lengths[0], r.total, r.total};
   if(!CompositeBuffers(buffers, sizes))
      return(false);
   const uint integers[] = {config.shape.K, config.shape.H, config.shape.M,
                            (uint)c.offsets[4], (uint)c.offsets[7], (uint)r.offsets[7],
                            (uint)r.total, config.price_basis, config.quote_adapter
                           };
   const float reals[] = {(float)config.max_quote_age_seconds, (float)config.max_last_age_seconds};
   return(CompositeDispatch(def_k_DPCCCompositeQuoteR, buffers, integers, reals, (uint)r.total) &&
          CompositeReduce(flags, scalar, (uint)r.total));
  }
//+------------------------------------------------------------------+
//| Function DPCCDispatchConditionV1                                 |
//+------------------------------------------------------------------+
bool DPCCDispatchConditionV1(COpenCLMy *context, CBufferFloat *source, CBufferFloat *account13,
                             CBufferFloat *condition, CBufferFloat *flags, const SDPCCConfig &config)
  {
   SDPCCLayout c, r;
   uint tokens;
   if(!context || !DPCCCompositeLayouts(config.shape, c, r, tokens))
      return(false);
   CBufferFloat *buffers[] = {source, account13, condition, flags};
   const ulong sizes[] = {(ulong)c.offsets[10], 13, (ulong)tokens + 61, (ulong)tokens + 45};
   for(uint i = 0; i < 4; i++)
     {
      if(CheckPointer(buffers[i]) == POINTER_INVALID || buffers[i].GetOpenCL() != context ||
         buffers[i].GetIndex() < 0 || (ulong)buffers[i].Total() < sizes[i] ||
         context.GetBufferSize(buffers[i].GetIndex()) != (ulong)buffers[i].Total() * sizeof(float) ||
         (i == 0 && (ulong)buffers[i].Total() != sizes[i] && (ulong)buffers[i].Total() != c.total) ||
         ((i == 1 || i == 2) && (ulong)buffers[i].Total() != sizes[i]))
         return(false);
      for(uint k = 0; k < i; k++)
         if(buffers[i] == buffers[k] || buffers[i].GetIndex() == buffers[k].GetIndex())
            return(false);
     }
   const uint integers[] = {config.shape.K, config.shape.V, config.shape.H, config.shape.D,
                            (uint)c.offsets[1], (uint)c.offsets[2], (uint)c.offsets[3],
                            (uint)c.offsets[4], (uint)c.offsets[6]
                           };
   const int kernel = def_k_DPCCCompositeCondition;
   for(uint i = 0; i < 4; i++)
      if(!context.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(false);
   for(uint i = 0; i < (uint)ArraySize(integers); i++)
      if(!context.SetArgument(kernel, i + 4, integers[i]))
         return(false);
   const float scales[] = {(float)config.volume_scale, (float)config.price_scale, (float)config.money_scale};
   for(uint i = 0; i < 3; i++)
      if(!context.SetArgument(kernel, i + 13, scales[i]))
         return(false);
   if(!context.SetArgument(kernel, 16, config.price_basis))
      return(false);
   const uint offset[] = {0}, global[] = {tokens + 45};
   return(context.Execute(kernel, 1, offset, global));
  }
//+------------------------------------------------------------------+
//| Function DPCCWriteConditionStepV1                                |
//+------------------------------------------------------------------+
bool DPCCWriteConditionStepV1(COpenCLMy *context, const uint j, const uint J,
                              CBufferFloat *condition, const uint tokens)
  {
   float encoding[];
   string reason;
   const ulong total = (ulong)tokens + 61, byte_offset = ((ulong)tokens + 45) * sizeof(float);
   if(!context || !DPCCBuildStepPositionalEncoding(j, J, encoding, reason) || !condition ||
      condition.GetOpenCL() != context || condition.GetIndex() < 0 || total > INT_MAX ||
      total > (ulong)UINT_MAX / sizeof(float) || (ulong)condition.Total() != total || byte_offset > UINT_MAX)
      return(false);
   return(context.BufferWrite(condition.GetIndex(), encoding, (uint)byte_offset, 0, 16));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::PackCompositeCondition                  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PackCompositeCondition(CBufferFloat *context, CBufferFloat *account13,
      CBufferFloat *condition, CBufferFloat *flags, CBufferFloat *scalar,
      const SDPCCLayout &c_layout, const SDPCCConfig &config)
  {
   SDPCCLayout c, r;
   uint tokens;
   if(!DPCCCompositeLayouts(config.shape, c, r, tokens) || c_layout.total != c.total)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {context, account13, condition, flags};
   const ulong sizes[] = {c.total, 13, (ulong)tokens + 61, (ulong)tokens + 45};
   if(!CompositeBuffers(buffers, sizes))
      return(false);
   if(!DPCCDispatchConditionV1(OpenCL, context, account13, condition, flags, config))
      return(Fail(DPCC_OPENCL_ERROR));
   return(CompositeReduce(flags, scalar, tokens + 45));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SetCompositeStep                        |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SetCompositeStep(const uint j, const uint J, CBufferFloat *condition,
                                      const uint tokens)
  {
   if(j == 0 || J == 0 || j > J)
      return(Fail(DPCC_INVALID_INPUT));
   if(!DPCCWriteConditionStepV1(OpenCL, j, J, condition, tokens))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Return the stable DPCC profile signature                         |
//+------------------------------------------------------------------+
//--- Durable profile hash is independent of owned child instances and readiness.
//--- Operational mode is intentionally omitted like DPCCFixedSignature.
//+------------------------------------------------------------------+
//| Computes a stable signature for the DPCC profile.                |
//+------------------------------------------------------------------+
ulong CNeuronDPCCOCL::ProfileSignature(const SDPCCConfig &c, const SDPCCDecoderComposition &d,
                                       const SDPCCDenoiserComposition &n, const double &beta[],
                                       const double &schedule_alpha[], const double &sigma[]) const
  {
   if(c.shape.J == 0 || c.shape.J > 64 || ArraySize(beta) != (int)c.shape.J + 1 ||
      ArraySize(schedule_alpha) != ArraySize(beta) || ArraySize(sigma) != ArraySize(beta))
      return(0);
   ulong hash = DPCCFixedSignature(c, defNeuronDPCCOCL);
   const SDPCCDenoiserArchitecture g = n.graph;
   const SDPCCParallelExecution p = n.parallel;
   const uint words[] = {d.embedding, g.embedding, g.heads, g.key_dimension, g.depth,
                         (uint)g.padding_mask, (uint)g.temporal_mask, p.version, p.training_plan_batch, p.deployment_plan_batch,
                         p.proposal_slots, p.logical_heads, p.key_dimension, (uint)p.shared_projection_weights, p.binary_layer_version
                        };
   for(uint i = 0; i < (uint)ArraySize(words); i++)
      DPCCRolloutHashWord(words[i], hash);
   const string texts[] = {d.architecture, d.activation_profile, d.backward_profile, d.optimizer_profile, d.graph_digest,
                           g.architecture, g.graph_id, g.graph_sha256, g.normalization, g.feedforward_activation, g.attention_backward,
                           g.backward_profile, n.optimizer_profile, p.program, p.self_attention_fold, p.cross_attention_fold,
                           p.activation_lifetime, p.legacy_slice_use, "CAUSAL_AFFINE_V2/REVIN_GAIN1/OHLC4/ADAM9"
                          };
   for(uint i = 0; i < (uint)ArraySize(texts); i++)
     {
      DPCCRolloutHashWord(StringLen(texts[i]), hash);
      for(int j = 0; j < StringLen(texts[i]); j++)
         DPCCRolloutHashWord(StringGetCharacter(texts[i], j), hash);
     }
   for(uint j = 0; j <= c.shape.J; j++)
     {
      if(!MathIsValidNumber(beta[j]) || !MathIsValidNumber(schedule_alpha[j]) || !MathIsValidNumber(sigma[j]))
         return(0);
      DPCCRolloutHashDouble(beta[j], hash);
      DPCCRolloutHashDouble(schedule_alpha[j], hash);
      DPCCRolloutHashDouble(sigma[j], hash);
     }
   return(hash);
  }
//+------------------------------------------------------------------+
//| Reconstructs the ABI3 Actor descriptor                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::DescribeAction6(CLayerDescription &desc) const
  {
   desc.type = Type();
   desc.count = 6;
   desc.activation = None;
   desc.optimization = ADAM;
   desc.batch = 1;
   desc.window = 6;
   desc.window_out = m_shape.Embedding;
   desc.layers = m_shape.J;
   desc.step = 1;
   desc.variables = 1;
   if(ArrayResize(desc.windows, 0) != 0 || ArrayResize(desc.heads, 0) != 0 ||
      ArrayResize(desc.units, 8) != 8)
      return(false);
   const uint units[] = {m_shape.H, m_shape.M, m_shape.Heads, m_shape.K, m_shape.V,
                         m_actor_market_horizon, m_actor_latent_dimension, 3
                        };
   for(uint i = 0; i < 8; i++)
      desc.units[i] = units[i];
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates a fresh ABI3 frame owner without admitting inference     |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::InitAction6(uint index, COpenCLMy *context, CLayerDescription *desc)
  {
   if(m_failed || m_shaped || m_busy || m_transfer_pending || context == NULL || desc == NULL || m_denoiser != NULL ||
      m_decoder != NULL || m_projector != NULL || m_actor_projector != NULL || m_rollout != NULL || desc.type != Type() ||
      desc.count != 6 || desc.activation != None || desc.optimization != ADAM || desc.batch != 1 ||
      desc.window != 6 || desc.window_out == 0 || desc.layers == 0 || desc.layers > 64 ||
      desc.step != 1 || desc.variables != 1 || ArraySize(desc.units) != 8 ||
      ArraySize(desc.windows) != 0 || ArraySize(desc.heads) != 0 || desc.units[7] != 3)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 7; i++)
      if(desc.units[i] == 0 || desc.units[i] > 16777216)
         return(Fail(DPCC_CONTRACT_MISMATCH));
   if(desc.units[3] > 64 || desc.units[5] < desc.units[0] || desc.window_out % desc.units[2] != 0)
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- Financial shape retains the published four-price contract.
   ZeroMemory(m_shape);
   m_actor_abi = 3;
   m_actor_market_horizon = desc.units[5];
   m_actor_latent_dimension = desc.units[6];
   m_shape.H = desc.units[0];
   m_shape.M = desc.units[1];
   m_shape.Heads = desc.units[2];
   m_shape.K = desc.units[3];
   m_shape.V = desc.units[4];
   m_shape.D = 4;
   m_shape.J = desc.layers;
   m_shape.PlanBatch = m_shape.M;
   m_shape.NActions = 6;
   m_shape.StateSize = 13;
   m_shape.Embedding = desc.window_out;
   m_shape.KeyDimension = desc.window_out / m_shape.Heads;
   if(!CNeuronBaseOCL::Init(0, index, context, 6, ADAM, 1))
      return(Fail(DPCC_OPENCL_ERROR, true));
   activation = None;
   bTrain = false;
   m_denoiser = new CNeuronPlanDenoiserOCL();
   if(!m_denoiser)
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_denoiser.AttachOpenCL(context);
   if(!m_denoiser.InitFrameShape(m_shape.H, m_shape.M, 6, m_shape.Embedding, m_shape.Heads, m_shape.J))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_shaped = true;
   m_status = DPCC_OK;
   return(ValidateAction6() ? true : Fail(DPCC_CONTRACT_MISMATCH, true));
  }
//+------------------------------------------------------------------+
//| Validates isolated ABI3 ownership and immutable financial shape  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidateAction6(void) const
  {
   if(m_actor_abi != 3 || !m_shaped || OpenCL == NULL || m_denoiser == NULL || m_decoder != NULL || m_projector != NULL ||
      m_normalizer != NULL || m_denorm != NULL || m_decoder_source != NULL || m_denoiser.GetOpenCL() != OpenCL ||
      !ValidBuffer(Output, 6, OpenCL) || !ValidBuffer(PrevOutput, 6, OpenCL) ||
      !ValidBuffer(Gradient, 6, OpenCL) || activation != None || optimization != ADAM || iBatch != 1 ||
      CheckPointer(Weights) != POINTER_INVALID || CheckPointer(DeltaWeights) != POINTER_INVALID ||
      CheckPointer(FirstMomentum) != POINTER_INVALID || CheckPointer(SecondMomentum) != POINTER_INVALID)
      return(false);
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      if(m_work[i])
         return(false);
   if(!m_configured)
      return(m_rollout == NULL && m_actor_projector == NULL && m_plan_source == NULL &&
             m_action_context == NULL && m_action_condition == NULL && m_action_noise == NULL &&
             m_action_current == NULL && m_action_next == NULL && m_action_physical == NULL &&
             m_action_epsilon == NULL && m_action_probability_frames == NULL && m_action_projected_frames == NULL &&
             m_action_flags == NULL && m_action_scalar == NULL && m_action_status == NULL &&
             m_action_candidate_status == NULL && m_action_returns == NULL && m_action_account13 == NULL &&
             m_action_q2_scores == NULL && m_action_rank_scores == NULL && m_action_numeric_flags == NULL);
   return(m_rollout != NULL && m_actor_projector != NULL && m_rollout.GetOpenCL() == OpenCL &&
          m_actor_projector.GetOpenCL() == OpenCL &&
          m_actor_projector.TraceCapacity() == m_shape.J &&
          m_plan_source != NULL && ValidateAction6Workspace() &&
          m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) &&
          m_actor_projector.ContractSignature() == DPCCFixedSignature(m_config, defNeuronPlanProjectionOCL));
  }
//+------------------------------------------------------------------+
//| Configures Action6 weights and the published financial children  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Configure(const SDPCCRuntimeConfig &runtime)
  {
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending || m_configured || !ValidateAction6())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   SDPCCShape financial = m_shape;
   financial.E = runtime.financial.shape.E;
   if(!DPCCSameShape(financial, runtime.financial.shape) ||
      !DPCCValidateFixedConfig(financial, runtime.financial) ||
      !ScheduleValid(runtime.beta, runtime.alpha_bar, runtime.sigma))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   SDPCCLayout checked;
   if(!DPCCBuildRolloutLayout(financial, checked))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 109; i++)
      if(!MathIsValidNumber(runtime.condition_scale[i]) || runtime.condition_scale[i] <= 0 ||
         !MathIsValidNumber(runtime.condition_bias[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
//--- The frame owns one weight set; no Market snapshot is attached here.
   if(!m_denoiser.InitFrameCondition(m_shape.K, m_shape.V, m_actor_market_horizon,
                                     m_actor_latent_dimension, runtime.condition_scale,
                                     runtime.condition_bias, runtime.financial.probability_tolerance))
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=denoiser_init status=%d", (int)m_denoiser.LastStatus());
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   m_shape = financial;
   m_rollout = new CNeuronScenarioRolloutOCL();
   m_actor_projector = new CNeuronPlanProjectionTraceOCL();
   if(!m_rollout || !m_actor_projector)
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=child_allocation rollout=%d projector=%d",
                  CheckPointer(m_rollout) != POINTER_INVALID, CheckPointer(m_actor_projector) != POINTER_INVALID);
      return(Fail(DPCC_OPENCL_ERROR, true));
     }
   CLayerDescription desc;
   if(!Describe(desc, defNeuronScenarioRolloutOCL, m_shape.M * m_shape.K, 0) ||
      !m_rollout.Init(0, 0, OpenCL, GetPointer(desc)) || !m_rollout.Configure(runtime.financial))
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=rollout_init_config status=%d",
                  (int)m_rollout.LastStatus());
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   if(!Describe(desc, defNeuronPlanProjectionTraceOCL, m_shape.M * m_shape.H * 6, 0) ||
      !m_actor_projector.InitTrace(0, OpenCL, GetPointer(desc), m_shape.J) ||
      !m_actor_projector.Configure(runtime.financial) || !m_actor_projector.BindRollout(m_rollout))
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=projector_init_config_bind status=%d",
                  (int)m_actor_projector.LastStatus());
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   m_config = runtime.financial;
   if(ArrayCopy(m_beta, runtime.beta) != ArraySize(runtime.beta) ||
      ArrayCopy(m_alpha_bar, runtime.alpha_bar) != ArraySize(runtime.alpha_bar) ||
      ArrayCopy(m_sigma, runtime.sigma) != ArraySize(runtime.sigma) ||
      ArrayCopy(m_actor_scale, runtime.condition_scale) != 109 ||
      ArrayCopy(m_actor_bias, runtime.condition_bias) != 109)
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=runtime_array_copy beta=%d alpha_bar=%d sigma=%d scale=%d bias=%d",
                  ArraySize(m_beta), ArraySize(m_alpha_bar), ArraySize(m_sigma),
                  ArraySize(m_actor_scale), ArraySize(m_actor_bias));
      return(Fail(DPCC_OPENCL_ERROR, true));
     }
   m_configured = true;
   m_imported = false;
   if(!AllocateAction6Workspace())
      return(false);
   if(!Clear())
     {
      PrintFormat("DPCC_ACTION6_CONFIG_FAIL stage=clear status=%d", (int)m_status);
      return(false);
     }
   const ulong contract = ContractSignature();
   if(contract == 0 || (m_admitted && m_admitted_profile != contract))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_admitted_profile = contract;
   m_admitted = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Compare the loaded ABI3 runtime without changing model state     |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::MatchesRuntimeConfig(const SDPCCRuntimeConfig &runtime) const
  {
   const SDPCCConfig expected = runtime.financial;
   if(m_actor_abi != 3 || !m_configured || !m_admitted || m_failed || m_busy || m_transfer_pending)
     {
      Print("DPCC_ACTION6_RUNTIME_MISMATCH field=actor_state");
      return(false);
     }
   if(!DPCCSameShape(m_shape, expected.shape) || !DPCCSameShape(m_config.shape, expected.shape) ||
      !DPCCValidateFixedConfig(m_shape, m_config) || !DPCCValidateFixedConfig(expected.shape, expected))
     {
      Print("DPCC_ACTION6_RUNTIME_MISMATCH field=shape_or_config_validity");
      return(false);
     }
   const long actual_words[] = {m_config.contract_version, m_config.binary_layer_version, (long)m_config.mode,
                                m_config.projection_max_iterations, m_config.projection_max_evaluations, m_config.projection_lookback_steps,
                                m_config.price_basis, m_config.quote_adapter, m_config.event_model, m_config.risk_type, m_config.normalization_mode,
                                m_config.loss_coordinates, m_config.randomness_version, m_config.margin_fx_profile, m_config.account_currency_digits
                               };
   const long expected_words[] = {expected.contract_version, expected.binary_layer_version, (long)expected.mode,
                                  expected.projection_max_iterations, expected.projection_max_evaluations, expected.projection_lookback_steps,
                                  expected.price_basis, expected.quote_adapter, expected.event_model, expected.risk_type, expected.normalization_mode,
                                  expected.loss_coordinates, expected.randomness_version, expected.margin_fx_profile, expected.account_currency_digits
                                 };
//--- The selected projection mode is applied after native restore.
   for(int i = 0; i < ArraySize(actual_words); i++)
      if(i != 2 && actual_words[i] != expected_words[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=integer index=%d actual=%I64d expected=%I64d",
                     i, actual_words[i], expected_words[i]);
         return(false);
        }
   const double actual_values[] = {m_config.projection_improvement_atol, m_config.projection_distance_tie_atol,
                                   m_config.volume_scale, m_config.price_scale, m_config.money_scale, m_config.max_quote_age_seconds,
                                   m_config.max_last_age_seconds, (double)m_config.risk_parameter, (double)m_config.mean_weight,
                                   (double)m_config.gamma, (double)m_config.probability_tolerance
                                  };
   const double expected_values[] = {expected.projection_improvement_atol, expected.projection_distance_tie_atol,
                                     expected.volume_scale, expected.price_scale, expected.money_scale, expected.max_quote_age_seconds,
                                     expected.max_last_age_seconds, (double)expected.risk_parameter, (double)expected.mean_weight,
                                     (double)expected.gamma, (double)expected.probability_tolerance
                                    };
   for(int i = 0; i < ArraySize(actual_values); i++)
      if(!MathIsValidNumber(actual_values[i]) || !MathIsValidNumber(expected_values[i]) ||
         actual_values[i] != expected_values[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=real index=%d", i);
         return(false);
        }
   for(int i = 0; i < 6; i++)
      if(!MathIsValidNumber(m_config.action_bias[i]) || !MathIsValidNumber(expected.action_bias[i]) ||
         !MathIsValidNumber(m_config.action_scale[i]) || !MathIsValidNumber(expected.action_scale[i]) ||
         m_config.action_bias[i] != expected.action_bias[i] || m_config.action_scale[i] != expected.action_scale[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=action_affine index=%d", i);
         return(false);
        }
   const int schedule = (int)expected.shape.J + 1;
   if(ArraySize(m_beta) != schedule || ArraySize(m_alpha_bar) != schedule || ArraySize(m_sigma) != schedule ||
      ArraySize(runtime.beta) != schedule || ArraySize(runtime.alpha_bar) != schedule || ArraySize(runtime.sigma) != schedule ||
      ArraySize(m_actor_scale) != 109 || ArraySize(m_actor_bias) != 109)
     {
      Print("DPCC_ACTION6_RUNTIME_MISMATCH field=schedule_or_condition_shape");
      return(false);
     }
   for(int i = 0; i < schedule; i++)
     {
      if(!MathIsValidNumber(m_beta[i]) || !MathIsValidNumber(m_alpha_bar[i]) || !MathIsValidNumber(m_sigma[i]) ||
         !MathIsValidNumber(runtime.beta[i]) || !MathIsValidNumber(runtime.alpha_bar[i]) ||
         !MathIsValidNumber(runtime.sigma[i]))
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=schedule_nonfinite index=%d", i);
         return(false);
        }
      if(m_beta[i] != runtime.beta[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=beta index=%d", i);
         return(false);
        }
      if(m_alpha_bar[i] != runtime.alpha_bar[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=alpha_bar index=%d", i);
         return(false);
        }
      if(m_sigma[i] != runtime.sigma[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=sigma index=%d", i);
         return(false);
        }
     }
   for(int i = 0; i < 109; i++)
     {
      if(!MathIsValidNumber(m_actor_scale[i]) || !MathIsValidNumber(runtime.condition_scale[i]) ||
         m_actor_scale[i] != runtime.condition_scale[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=condition_scale index=%d", i);
         return(false);
        }
      if(!MathIsValidNumber(m_actor_bias[i]) || !MathIsValidNumber(runtime.condition_bias[i]) ||
         m_actor_bias[i] != runtime.condition_bias[i])
        {
         PrintFormat("DPCC_ACTION6_RUNTIME_MISMATCH field=condition_bias index=%d", i);
         return(false);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::AllocateAction6Workspace                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::AllocateAction6Workspace(void)
  {
   if(CheckPointer(m_action_context) != POINTER_INVALID || CheckPointer(m_action_condition) != POINTER_INVALID ||
      CheckPointer(m_action_noise) != POINTER_INVALID || CheckPointer(m_action_current) != POINTER_INVALID ||
      CheckPointer(m_action_next) != POINTER_INVALID || CheckPointer(m_action_physical) != POINTER_INVALID ||
      CheckPointer(m_action_epsilon) != POINTER_INVALID || CheckPointer(m_action_probability_frames) != POINTER_INVALID ||
      CheckPointer(m_action_projected_frames) != POINTER_INVALID || CheckPointer(m_action_flags) != POINTER_INVALID ||
      CheckPointer(m_action_scalar) != POINTER_INVALID || CheckPointer(m_action_status) != POINTER_INVALID ||
      CheckPointer(m_action_candidate_status) != POINTER_INVALID || CheckPointer(m_action_returns) != POINTER_INVALID ||
      CheckPointer(m_action_account13) != POINTER_INVALID || CheckPointer(m_action_q2_scores) != POINTER_INVALID ||
      CheckPointer(m_action_rank_scores) != POINTER_INVALID || CheckPointer(m_action_numeric_flags) != POINTER_INVALID ||
      CheckPointer(m_plan_source) != POINTER_INVALID)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   ulong tokens = 0, condition_count = 0, noise_count = 0, plan_count = 0, returns_count = 0;
   ulong temporary = 0, frame_count = 0;
   if(!DPCCBuildRolloutLayout(m_shape, m_action_context_layout) ||
      !DPCCRolloutProduct(m_shape.K, m_shape.V, tokens) ||
      !DPCCRolloutProduct(tokens, m_actor_market_horizon, tokens) ||
      !DPCCRolloutProduct(tokens, (ulong)m_actor_latent_dimension + 1, condition_count) ||
      !DPCCRolloutProduct(m_shape.M, (ulong)m_shape.J + 1, temporary) ||
      !DPCCRolloutProduct(temporary, (ulong)m_shape.H * 6, noise_count) ||
      !DPCCRolloutProduct(m_shape.M, (ulong)m_shape.H * 6, plan_count) ||
      !DPCCRolloutProduct(m_shape.J, plan_count, frame_count) ||
      !DPCCRolloutProduct(m_shape.M, m_shape.K, returns_count))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   condition_count += 109 + m_shape.K;
   const ulong flag_count = MathMax(condition_count, MathMax(noise_count, plan_count));
   const ulong limit = (ulong)UINT_MAX / sizeof(float);
   if(m_action_context_layout.total > limit || condition_count > limit || noise_count > limit || frame_count > limit ||
      plan_count > limit || returns_count > limit || flag_count > limit || plan_count > INT_MAX)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
#define DPCC_ACTION6_ALLOC(slot,member,count) \
   member=new CBufferFloat(); \
   if(!member || !member.BufferInit((uint)(count),0) || !member.BufferCreate(OpenCL)) \
     { PrintFormat("DPCC_ACTION6_WORKSPACE_FAIL slot=%d count=%I64u",slot,(ulong)(count)); \
       return(Fail(DPCC_OPENCL_ERROR,true)); }
   DPCC_ACTION6_ALLOC(0, m_action_context, m_action_context_layout.total);
   DPCC_ACTION6_ALLOC(1, m_action_condition, condition_count);
   DPCC_ACTION6_ALLOC(2, m_action_noise, noise_count);
   DPCC_ACTION6_ALLOC(3, m_action_current, plan_count);
   DPCC_ACTION6_ALLOC(4, m_action_next, plan_count);
   DPCC_ACTION6_ALLOC(5, m_action_physical, plan_count);
   DPCC_ACTION6_ALLOC(6, m_action_epsilon, plan_count);
   DPCC_ACTION6_ALLOC(16, m_action_probability_frames, frame_count);
   DPCC_ACTION6_ALLOC(17, m_action_projected_frames, frame_count);
   DPCC_ACTION6_ALLOC(7, m_action_flags, flag_count);
   DPCC_ACTION6_ALLOC(8, m_action_scalar, 1);
   DPCC_ACTION6_ALLOC(9, m_action_status, m_shape.M);
   DPCC_ACTION6_ALLOC(10, m_action_candidate_status, m_shape.M);
   DPCC_ACTION6_ALLOC(11, m_action_returns, returns_count);
   DPCC_ACTION6_ALLOC(12, m_action_account13, 13);
   DPCC_ACTION6_ALLOC(13, m_action_q2_scores, m_shape.M);
   DPCC_ACTION6_ALLOC(14, m_action_rank_scores, m_shape.M);
   DPCC_ACTION6_ALLOC(15, m_action_numeric_flags, m_shape.M);
#undef DPCC_ACTION6_ALLOC
   m_plan_source = new CNeuronBaseOCL();
   if(!m_plan_source || !m_plan_source.Init(0, 0, OpenCL, (uint)plan_count, ADAM, 1))
     {
      PrintFormat("DPCC_ACTION6_WORKSPACE_FAIL slot=16 name=plan_source count=%I64u",
                  plan_count);
      return(Fail(DPCC_OPENCL_ERROR, true));
     }
   m_plan_source.SetActivationFunction(None);
   m_plan_source.TrainMode(false);
   m_action_context_bound = false;
   m_action_candidates_valid = false;
   if(!ValidateAction6Workspace())
     {
      PrintFormat("DPCC_ACTION6_WORKSPACE_FAIL stage=validation plan_count=%I64u",
                  plan_count);
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ValidateAction6Workspace                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidateAction6Workspace(void) const
  {
   if(CheckPointer(m_action_context) == POINTER_INVALID || CheckPointer(m_action_condition) == POINTER_INVALID ||
      CheckPointer(m_action_noise) == POINTER_INVALID || CheckPointer(m_action_current) == POINTER_INVALID ||
      CheckPointer(m_action_next) == POINTER_INVALID || CheckPointer(m_action_physical) == POINTER_INVALID ||
      CheckPointer(m_action_epsilon) == POINTER_INVALID || CheckPointer(m_action_probability_frames) == POINTER_INVALID ||
      CheckPointer(m_action_projected_frames) == POINTER_INVALID || CheckPointer(m_action_flags) == POINTER_INVALID ||
      CheckPointer(m_action_scalar) == POINTER_INVALID || CheckPointer(m_action_status) == POINTER_INVALID ||
      CheckPointer(m_action_candidate_status) == POINTER_INVALID || CheckPointer(m_action_returns) == POINTER_INVALID ||
      CheckPointer(m_action_account13) == POINTER_INVALID || CheckPointer(m_action_q2_scores) == POINTER_INVALID ||
      CheckPointer(m_action_rank_scores) == POINTER_INVALID || CheckPointer(m_action_numeric_flags) == POINTER_INVALID ||
      CheckPointer(m_plan_source) == POINTER_INVALID ||
      m_plan_source.GetOpenCL() != OpenCL || m_plan_source.TrainMode())
      return(false);
   ulong tokens = 0, condition_count = 0, noise_count = 0, plan_count = 0, returns_count = 0;
   ulong temporary = 0, frame_count = 0;
   if(!DPCCRolloutProduct(m_shape.K, m_shape.V, tokens) ||
      !DPCCRolloutProduct(tokens, m_actor_market_horizon, tokens) ||
      !DPCCRolloutProduct(tokens, (ulong)m_actor_latent_dimension + 1, condition_count) ||
      !DPCCRolloutProduct(m_shape.M, (ulong)m_shape.J + 1, temporary) ||
      !DPCCRolloutProduct(temporary, (ulong)m_shape.H * 6, noise_count) ||
      !DPCCRolloutProduct(m_shape.M, (ulong)m_shape.H * 6, plan_count) ||
      !DPCCRolloutProduct(m_shape.J, plan_count, frame_count) ||
      !DPCCRolloutProduct(m_shape.M, m_shape.K, returns_count))
      return(false);
   condition_count += 109 + m_shape.K;
   return(m_action_context_layout.total > 0 &&
          ValidBuffer(m_action_context, m_action_context_layout.total, OpenCL) &&
          ValidBuffer(m_action_condition, condition_count, OpenCL) && ValidBuffer(m_action_noise, noise_count, OpenCL) &&
          ValidBuffer(m_action_current, plan_count, OpenCL) && ValidBuffer(m_action_next, plan_count, OpenCL) &&
          ValidBuffer(m_action_physical, plan_count, OpenCL) && ValidBuffer(m_action_epsilon, plan_count, OpenCL) &&
          ValidBuffer(m_action_probability_frames, frame_count, OpenCL) &&
          ValidBuffer(m_action_projected_frames, frame_count, OpenCL) &&
          CheckPointer(m_action_flags) != POINTER_INVALID && m_action_flags.GetOpenCL() == OpenCL && m_action_flags.GetIndex() >= 0 &&
          m_action_flags.Total() >= (int)MathMax(condition_count, MathMax(noise_count, plan_count)) &&
          ValidBuffer(m_action_scalar, 1, OpenCL) && ValidBuffer(m_action_status, m_shape.M, OpenCL) &&
          ValidBuffer(m_action_candidate_status, m_shape.M, OpenCL) &&
          ValidBuffer(m_action_returns, returns_count, OpenCL) &&
          ValidBuffer(m_action_account13, 13, OpenCL) && ValidBuffer(m_action_q2_scores, m_shape.M, OpenCL) &&
          ValidBuffer(m_action_rank_scores, m_shape.M, OpenCL) && ValidBuffer(m_action_numeric_flags, m_shape.M, OpenCL) &&
          ValidBuffer(m_plan_source.getOutput(), plan_count, OpenCL));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ClearPolicyState                        |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::ClearPolicyState(void)
  {
   m_policy_q1_state.Clear();
   m_policy_q2_state.Clear();
   m_policy_state_valid = false;
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ClearPolicyWorkspace                    |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::ClearPolicyWorkspace(void)
  {
   m_policy_q1_rows.Clear();
   m_policy_q2_rows.Clear();
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ValidatePolicyNetworks                  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidatePolicyNetworks(CNet *market, CNet *q1, CNet *q2) const
  {
   if(m_actor_abi != 3 || !market || !q1 || !q2 || market == q1 || market == q2 || q1 == q2 ||
      !market.IsOpenCLTransferReady() || !q1.IsOpenCLTransferReady() || !q2.IsOpenCLTransferReady() ||
      market.GetOpenCL() != OpenCL || q1.GetOpenCL() != OpenCL || q2.GetOpenCL() != OpenCL ||
      market.WeightsUpdateEnabled())
      return(false);
   CNeuronBaseOCL *market_layer = market.Layer(6);
   CNeuronScenarioForecastPriceOCL *forecast = dynamic_cast<CNeuronScenarioForecastPriceOCL*>(market_layer);
   if(!market_layer || !forecast || forecast.Type() != defNeuronScenarioForecastPriceOCL ||
      forecast.Scenarios() != m_shape.K || forecast.Variables() != m_shape.V ||
      forecast.Horizon() != m_actor_market_horizon || forecast.Dimension() != m_actor_latent_dimension)
      return(false);
   CNet *critics[] = {q1, q2};
   for(uint i = 0; i < (uint)ArraySize(critics); i++)
     {
      if(!NNQCriticGraph(critics[i]))
         return(false);
      CNeuronScenarioCrossAttention *cross = dynamic_cast<CNeuronScenarioCrossAttention*>(critics[i].Layer(1));
      if(!cross || cross.Scenarios() != forecast.Scenarios() || cross.Variables() != forecast.Variables() ||
         cross.Horizon() != forecast.Horizon() || cross.Latent() != forecast.Dimension())
         return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::BindPolicyNetworks                      |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindPolicyNetworks(CNet *market, CNet *q1, CNet *q2)
  {
   if(m_failed || m_busy || m_transfer_pending || m_action_unroll_live || m_action_trace_live ||
      !ValidateAction6Workspace() || !ValidatePolicyNetworks(market, q1, q2))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   ClearPolicyState();
   ClearPolicyWorkspace();
   m_policy_market = market;
   m_policy_q1 = q1;
   m_policy_q2 = q2;
   m_policy_networks_bound = true;
   m_action_market_weights_id = 0;
   m_public_valid = false;
   m_action_candidates_valid = false;
   m_action_selected = UINT_MAX;
   m_action_context_bound = false;
   m_action_market_generation = 0;
   ZeroMemory(m_action_meta);
   if(!Output.Fill(0) || !PrevOutput.Fill(0) || !Gradient.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::UnbindPolicyNetworks                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::UnbindPolicyNetworks(void)
  {
   if(m_busy || m_transfer_pending || m_action_unroll_live || m_action_trace_live)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   ClearPolicyState();
   ClearPolicyWorkspace();
   m_policy_market = NULL;
   m_policy_q1 = NULL;
   m_policy_q2 = NULL;
   m_policy_networks_bound = false;
   m_public_valid = false;
   m_action_candidates_valid = false;
   m_action_selected = UINT_MAX;
   m_action_context_bound = false;
   m_action_market_generation = 0;
   ZeroMemory(m_action_meta);
   if((ValidBuffer(Output, 6, OpenCL) && !Output.Fill(0)) ||
      (ValidBuffer(PrevOutput, 6, OpenCL) && !PrevOutput.Fill(0)) ||
      (ValidBuffer(Gradient, 6, OpenCL) && !Gradient.Fill(0)))
      return(Fail(DPCC_OPENCL_ERROR));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::BindDecisionContext                     |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindDecisionContext(CBufferFloat *context, const SDPCCDecisionMeta &meta)
  {
   if(m_action_unroll_live || m_action_trace_live)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_public_valid = false;
   m_action_candidates_valid = false;
   m_action_context_bound = false;
   m_action_selected = UINT_MAX;
   ClearPolicyState();
   ZeroMemory(m_action_meta);
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending || !m_configured ||
      !ValidateAction6Workspace() || !context || meta.decision_id == 0 || meta.market_state_id == 0 ||
      meta.noise_seed == 0 || meta.decision_server_time <= 0 || meta.bar_index < 0 || meta.mode > DPCC_P2 ||
      StringLen(meta.market_sha256) != 64 || StringLen(meta.profile_sha256) != 64 ||
      context == GetPointer(m_action_context) || context.GetOpenCL() != OpenCL ||
      context.GetIndex() < 0 || (ulong)context.Total() != m_action_context_layout.total ||
      OpenCL.GetBufferSize(context.GetIndex()) != m_action_context_layout.total * sizeof(float) ||
      OwnsBufferOrDeviceSlot(context))
      return(Fail(DPCC_INVALID_INPUT));
   if(!Output.Fill(0) || !PrevOutput.Fill(0) || !Gradient.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!CopyBufferRaw(context, m_action_context, (uint)m_action_context_layout.total))
      return(Fail(DPCC_OPENCL_ERROR));
   m_action_meta = meta;
   m_action_market_generation = 0;
   m_action_context_bound = true;
   m_action_candidates_valid = false;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::GetCheckedPlans                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::GetCheckedPlans(CBufferFloat *&plans) const
  {
   plans = NULL;
   if(m_actor_abi != 3 || !m_action_candidates_valid || m_failed || m_busy ||
      m_action_market_generation == 0 || m_action_market_generation != m_action_meta.market_state_id ||
      !ValidateAction6Workspace())
      return(false);
   plans = m_plan_source.getOutput();
   return(plans != NULL);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::GetCandidateFlags                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::GetCandidateFlags(CBufferFloat *&flags) const
  {
   flags = NULL;
   if(m_actor_abi != 3 || !m_action_candidates_valid || m_failed || m_busy ||
      m_action_market_generation == 0 || m_action_market_generation != m_action_meta.market_state_id ||
      !ValidateAction6Workspace())
      return(false);
   flags = m_action_flags;
   return(flags != NULL);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::GetOracleCandidateMask                  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::GetOracleCandidateMask(uchar &mask[], uint &horizon) const
  {
   horizon = 0;
   ArrayFree(mask);
   const ulong active_offset = (ulong)m_action_context_layout.offsets[7];
   const bool oracle_fallback = (m_status == DPCC_NO_FEASIBLE_CANDIDATE && !m_public_valid &&
                                 m_action_selected == UINT_MAX);
   const bool oracle_decision = (m_status == DPCC_OK && m_public_valid && m_action_selected < m_shape.M);
   const bool workspace_valid = (m_actor_abi == 3 && !m_failed && !m_busy && !m_transfer_pending &&
                                 ValidateAction6Workspace());
   int context_total = -1;
   if(CheckPointer(m_action_context) != POINTER_INVALID)
      context_total = m_action_context.Total();
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending ||
      (!oracle_fallback && !oracle_decision) ||
      !m_action_unroll_live || !m_action_candidates_valid ||
      m_action_market_generation == 0 || m_action_market_generation != m_action_meta.market_state_id ||
      !workspace_valid || active_offset > (ulong)UINT_MAX / sizeof(float) ||
      active_offset + (ulong)m_shape.M > (ulong)m_action_context.Total())
      return(false);
   float projection_status[], active[], candidate_flags[];
   if(ArrayResize(mask, (int)m_shape.M) != (int)m_shape.M ||
      ArrayResize(projection_status, (int)m_shape.M) != (int)m_shape.M ||
      ArrayResize(active, (int)m_shape.M) != (int)m_shape.M ||
      ArrayResize(candidate_flags, (int)m_shape.M) != (int)m_shape.M ||
      !OpenCL.BufferRead(m_action_candidate_status.GetIndex(), projection_status, 0, 0, m_shape.M) ||
      !OpenCL.BufferRead(m_action_context.GetIndex(), active, (uint)(active_offset * sizeof(float)), 0, m_shape.M) ||
      !OpenCL.BufferRead(m_action_flags.GetIndex(), candidate_flags, 0, 0, m_shape.M))
     {
      ArrayFree(mask);
      return(false);
     }
   for(uint i = 0; i < m_shape.M; i++)
     {
      if(!MathIsValidNumber(projection_status[i]) ||
         !MathIsValidNumber(active[i]) || (active[i] != 0.0f && active[i] != 1.0f) ||
         !MathIsValidNumber(candidate_flags[i]) ||
         (candidate_flags[i] != 0.0f && candidate_flags[i] != 1.0f))
        {
         ArrayFree(mask);
           {
            PrintFormat("DPCC_ACTION6_ORACLE_MASK_FAIL stage=invalid_values candidate=%u projection=%g active=%g flags=%g status=%d public=%d selected=%u fallback=%d decision=%d failed=%d busy=%d transfer=%d unroll=%d candidates=%d generation=%I64u/%I64u workspace=%d active_offset=%I64u context_total=%d",
                        i, projection_status[i], active[i], candidate_flags[i], (int)m_status, (int)m_public_valid, m_action_selected, (int)oracle_fallback,
                        (int)oracle_decision, (int)m_failed, (int)m_busy, (int)m_transfer_pending,
                        (int)m_action_unroll_live, (int)m_action_candidates_valid,
                        m_action_market_generation, m_action_meta.market_state_id, (int)workspace_valid,
                        active_offset, context_total);
            return(false);
           }
        }
      //--- Ordinary Actor updates stay on financially valid rollouts; the explicit
      //--- no-feasible fallback may use any projectable candidate for Oracle fit.
      mask[i] = (projection_status[i] == 0.0f && active[i] == 1.0f &&
                 (oracle_fallback || candidate_flags[i] == 0.0f) ? 1 : 0);
     }
   horizon = m_shape.H;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::AbortNoFeasibleAction6Unroll            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::AbortNoFeasibleAction6Unroll(void)
  {
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending ||
      m_status != DPCC_NO_FEASIBLE_CANDIDATE || m_public_valid ||
      m_action_selected != UINT_MAX || !m_action_unroll_live ||
      !m_action_candidates_valid || m_action_update_started)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   return(AbortAction6Unroll());
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::GetSelectedCandidate                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::GetSelectedCandidate(uint &candidate) const
  {
   candidate = UINT_MAX;
   if(m_actor_abi != 3 || !m_public_valid || m_failed || m_busy || m_transfer_pending ||
      m_action_selected >= m_shape.M || m_action_market_generation == 0 ||
      m_action_market_generation != m_action_meta.market_state_id)
      return(false);
   candidate = m_action_selected;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::EvaluateAndPublishAction6               |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::EvaluateAndPublishAction6(CNeuronScenarioForecastPriceOCL *market)
  {
   if(!m_policy_networks_bound || !market || !m_action_candidates_valid ||
      !m_action_meta.decision_id || !m_action_meta.market_state_id ||
      m_action_market_generation != m_action_meta.market_state_id ||
      !ValidatePolicyNetworks(m_policy_market, m_policy_q1, m_policy_q2) ||
      m_policy_market.Layer(6) != (CNeuronBaseOCL*)market || market.Generation() != m_action_meta.market_state_id ||
      !ValidBuffer(m_action_account13, 13, OpenCL) || !m_plan_source ||
      !ValidBuffer(m_plan_source.getOutput(), (ulong)m_shape.M * m_shape.H * 6, OpenCL) ||
      !m_action_candidate_status || !ValidBuffer(m_action_candidate_status, m_shape.M, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   ClearPolicyState();
   m_action_selected = UINT_MAX;
   if(!m_action_numeric_flags.Fill(0) || !m_action_scalar.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!NNQCaptureCritic(m_policy_q1, m_policy_q1_state, m_action_meta.decision_id, m_action_meta.market_state_id) ||
      !NNQCaptureCritic(m_policy_q2, m_policy_q2_state, m_action_meta.decision_id, m_action_meta.market_state_id))
     {
      ClearPolicyState();
      return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   m_policy_state_valid = true;
   const int forecast_type = market.Type();
   if(!NNQEvaluateCriticRows(m_policy_q1, m_action_account13, m_plan_source.getOutput(), m_shape.H, m_shape.M,
                             market.GetZ(), market.GetLogU(), market.GetPi(), forecast_type, m_config.risk_type,
                             m_config.risk_parameter, m_config.mean_weight, m_policy_q1_state, m_policy_q1_rows,
                             m_action_returns, m_action_numeric_flags) ||
      !NNQEvaluateCriticRows(m_policy_q2, m_action_account13, m_plan_source.getOutput(), m_shape.H, m_shape.M,
                             market.GetZ(), market.GetLogU(), market.GetPi(), forecast_type, m_config.risk_type,
                             m_config.risk_parameter, m_config.mean_weight, m_policy_q2_state, m_policy_q2_rows,
                             m_action_q2_scores, m_action_numeric_flags) ||
      !NNQRankCriticRows(OpenCL, m_action_returns, m_action_q2_scores, m_action_flags,
                         m_action_numeric_flags, m_action_rank_scores, m_action_scalar, m_shape.M))
     {
      ClearPolicyState();
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
     }
   float selected_value[1];
   if(!OpenCL.BufferRead(m_action_scalar.GetIndex(), selected_value, 0, 0, 1) ||
      !MathIsValidNumber(selected_value[0]))
     {
      ClearPolicyState();
      return(Fail(DPCC_OPENCL_ERROR));
     }
   const int selected = (int)selected_value[0];
   if(selected_value[0] != (float)selected)
     {
      ClearPolicyState();
      return(Fail(DPCC_NUMERIC_ERROR));
     }
   if(selected == -1)
     {
      ClearPolicyState();
      return(Fail(DPCC_NO_FEASIBLE_CANDIDATE));
     }
   if(selected < 0 || selected >= (int)m_shape.M)
     {
      ClearPolicyState();
      return(Fail(DPCC_NUMERIC_ERROR));
     }
   m_action_selected = (uint)selected;
   float action[6];
   const ulong source_offset = (ulong)m_action_selected * m_shape.H * 6;
   if(source_offset + 6 > (ulong)m_shape.M * m_shape.H * 6 || source_offset > (ulong)UINT_MAX / sizeof(float) ||
      !OpenCL.BufferRead(m_plan_source.getOutput().GetIndex(), action, (uint)(source_offset * sizeof(float)), 0, 6))
     {
      ClearPolicyState();
      m_action_selected = UINT_MAX;
      return(Fail(DPCC_OPENCL_ERROR));
     }
   for(uint i = 0; i < 6; i++)
      if(!MathIsValidNumber(action[i]) || !Output.Update((int)i, action[i]))
        {
         ClearPolicyState();
         m_action_selected = UINT_MAX;
         return(Fail(DPCC_NUMERIC_ERROR));
        }
   if(!Output.BufferWrite())
     {
      ClearPolicyState();
      m_action_selected = UINT_MAX;
      return(Fail(DPCC_OPENCL_ERROR));
     }
   m_public_valid = true;
   m_status = DPCC_OK;
   if(!bTrain)
      ClearPolicyState();
   return(true);
  }
//+------------------------------------------------------------------+
//| Packs one admitted account, physical context and Market          |
//| generation                                                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PackAction6Condition(CNeuronBaseOCL *account,
      CNeuronScenarioForecastPriceOCL *market)
  {
   const ulong tokens = (ulong)m_shape.K * m_shape.V * m_actor_market_horizon;
   const ulong token_width = (ulong)m_actor_latent_dimension + 1;
   const ulong token_values = tokens * token_width;
   const ulong count = 109 + token_values + m_shape.K;
   if(!account || !market || account == GetPointer(this) || account.GetOpenCL() != OpenCL ||
      account.Neurons() != 13 || !account.getOutput() || !ValidBuffer(account.getOutput(), 13, OpenCL) ||
      market.GetOpenCL() != OpenCL || market.ABI() != defScenarioForecastPriceABI ||
      !ValidBuffer(market.GetZ(), tokens * m_actor_latent_dimension, OpenCL) ||
      !ValidBuffer(market.GetLogU(), tokens, OpenCL) || !ValidBuffer(market.GetPi(), m_shape.K, OpenCL) ||
      !ValidBuffer(m_action_context, m_action_context_layout.total, OpenCL) ||
      !ValidBuffer(m_action_condition, count, OpenCL) || !m_action_flags ||
      m_action_flags.GetOpenCL() != OpenCL || m_action_flags.GetIndex() < 0 || m_action_flags.Total() < (int)count)
      return(Fail(DPCC_INVALID_INPUT));
   CBufferFloat *buffers[] = {account.getOutput(), m_action_context, market.GetZ(), market.GetLogU(),
                              market.GetPi(), m_action_condition, m_action_flags
                             };
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
     {
      if(!buffers[i] || buffers[i].GetOpenCL() != OpenCL || buffers[i].GetIndex() < 0 ||
         OpenCL.GetBufferSize(buffers[i].GetIndex()) != (ulong)buffers[i].Total()*sizeof(float))
         return(Fail(DPCC_CONTRACT_MISMATCH));
      for(uint j = 0; j < i; j++)
         if(buffers[i] == buffers[j] || buffers[i].GetIndex() == buffers[j].GetIndex())
            return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   if(!CopyBufferRaw(account.getOutput(), m_action_account13, 13))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint args[] = {m_shape.K, m_shape.V, m_actor_market_horizon, m_actor_latent_dimension,
                        (uint)m_action_context_layout.offsets[1], (uint)m_action_context_layout.offsets[2],
                        (uint)m_action_context_layout.offsets[3], (uint)count
                       };
   const int kernel = def_k_DPCCAction6PackCondition;
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!OpenCL.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   for(uint i = 0; i < (uint)ArraySize(args); i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers)), args[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {(uint)count};
   if(!OpenCL.Execute(kernel, 1, offset, global) || !CompositeReduce(m_action_flags, m_action_scalar, (uint)count))
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
   return(true);
  }
//+------------------------------------------------------------------+
//| Uploads the canonical indexed noise tensor once per decision     |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::InitializeAction6Noise(void)
  {
   float values[];
   const ulong count = (ulong)m_shape.M * (m_shape.J + 1) * m_shape.H * 6;
   if(count > INT_MAX || !DPCCBuildIndexedNoiseV1(m_action_meta.noise_seed, m_action_meta.decision_server_time, 0,
         m_shape.M, m_shape.J, m_shape.H, values) || ArraySize(values) != (int)count ||
      !ValidBuffer(m_action_noise, count, OpenCL) || !ValidBuffer(m_action_current, (ulong)m_shape.M * m_shape.H * 6, OpenCL) ||
      !ValidBuffer(m_action_status, m_shape.M, OpenCL) || !ValidBuffer(m_action_candidate_status, m_shape.M, OpenCL))
      return(Fail(DPCC_INVALID_INPUT));
   for(int i = 0; i < ArraySize(values); i++)
      if(!MathIsValidNumber(values[i]) || !m_action_noise.Update(i, values[i]))
         return(Fail(DPCC_NUMERIC_ERROR));
   if(!m_action_noise.BufferWrite() || !m_action_current.Fill(0) ||
      !m_action_probability_frames.Fill(0) || !m_action_projected_frames.Fill(0) ||
      !m_action_status.Fill(0) || !m_action_candidate_status.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   CBufferFloat *buffers[] = {m_action_noise, m_action_context, m_action_current, m_action_status,
                              m_action_candidate_status, m_action_flags
                             };
   const uint args[] = {m_shape.M, m_shape.H, m_shape.J, 0,
                        (uint)m_action_context_layout.offsets[7]
                       };
   const int kernel = def_k_DPCCAction6InitialNoise;
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!OpenCL.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   for(uint i = 0; i < (uint)ArraySize(args); i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers)), args[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {(uint)(m_shape.M*m_shape.H * 6)};
   return(OpenCL.Execute(kernel, 1, offset, global) &&
          CompositeReduce(m_action_flags, m_action_scalar, global[0]));
  }
//+------------------------------------------------------------------+
//| Runs one reverse diffusion update on the device                  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReverseAction6Step(const uint j)
  {
   if(j == 0 || j > m_shape.J)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_action_current, m_denoiser.getOutput(), m_action_noise, m_action_context,
                              m_action_next, m_action_status, m_action_flags
                             };
   const uint args[] = {m_shape.M, m_shape.H, m_shape.J, j, 0,
                        (uint)m_action_context_layout.offsets[7]
                       };
   const int kernel = def_k_DPCCAction6Reverse;
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!buffers[i] || !OpenCL.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   for(uint i = 0; i < (uint)ArraySize(args); i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers)), args[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const float reals[] = {(float)m_beta[j], (float)m_alpha_bar[j], (float)m_sigma[j]};
   for(uint i = 0; i < (uint)ArraySize(reals); i++)
      if(!MathIsValidNumber(reals[i]) || !OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers) + ArraySize(args)), reals[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint offset[] = {0}, global[] = {(uint)(m_shape.M*m_shape.H * 6)};
   return(OpenCL.Execute(kernel, 1, offset, global) && CompositeReduce(m_action_flags, m_action_scalar, global[0]));
  }
//+------------------------------------------------------------------+
//| Applies the one configured native projector to the current plan  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ProjectAction6Current(const uint trace_slot)
  {
   if(!m_actor_projector.ForwardFrame(trace_slot, m_plan_source, m_action_context, bTrain))
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projector_forward slot=%u status=%d",
                  trace_slot, (int)m_actor_projector.LastStatus());
      return(Fail(m_actor_projector.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_actor_projector.LastStatus()));
     }
   if(!m_actor_projector.getOutput())
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projector_output slot=%u", trace_slot);
      return(Fail(DPCC_OPENCL_ERROR));
     }
   if(!CopyBufferRaw(m_actor_projector.getOutput(), m_action_physical, m_shape.M * m_shape.H * 6))
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projector_output_copy slot=%u", trace_slot);
      return(Fail(DPCC_OPENCL_ERROR));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Shared bounded action mapping for P0/P1/P2                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Action6Affine(CBufferFloat *source, CBufferFloat *destination, const uint direction,
                                  const uint frame_slot)
  {
   const uint count = m_shape.M * m_shape.H * 6;
   if(direction > 1 || !ValidBuffer(source, count, OpenCL) || !ValidBuffer(destination, count, OpenCL) ||
      !ValidBuffer(m_action_context, m_action_context_layout.total, OpenCL) ||
      !m_action_probability_frames || !m_action_projected_frames ||
      !m_action_flags || m_action_flags.Total() < (int)count)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(frame_slot != UINT_MAX &&
      (!bTrain || (m_action_meta.mode != DPCC_P1 && m_action_meta.mode != DPCC_P2) ||
       frame_slot >= (m_action_meta.mode == DPCC_P1 ? 1 : m_shape.J)))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong probability_offset_wide = (frame_slot == UINT_MAX ? (ulong)UINT_MAX : (ulong)frame_slot * count);
   if(probability_offset_wide > UINT_MAX ||
      (frame_slot != UINT_MAX &&
       (probability_offset_wide + count > (ulong)m_action_probability_frames.Total() ||
        probability_offset_wide + count > (ulong)m_action_projected_frames.Total())))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint probability_offset = (uint)probability_offset_wide;
   const uint active_offset = (uint)m_action_context_layout.offsets[7];
   const uint sigmoid_output = 1;
   float scale[6], bias[6];
   for(uint c = 0; c < 6; c++)
     {
      bias[c] = m_config.action_bias[c];
      scale[c] = m_config.action_scale[c];
      if(!MathIsValidNumber(bias[c]) || !MathIsValidNumber(scale[c]) || scale[c] <= 0)
         return(Fail(DPCC_INVALID_INPUT));
     }
   const int kernel = def_k_DPCCCompositeActionAffine;
   CBufferFloat *buffers[] = {source, destination, m_action_context, m_action_flags,
                              m_action_probability_frames, m_action_projected_frames};
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!OpenCL.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint args[] = {m_shape.M, m_shape.H, active_offset, direction,
                        sigmoid_output, probability_offset};
   for(uint i = 0; i < (uint)ArraySize(args); i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers)), args[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   for(uint i = 0; i < 6; i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers) + ArraySize(args)), bias[i]) ||
         !OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers) + ArraySize(args) + 6), scale[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {count};
   if(!OpenCL.Execute(kernel, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   if(CompositeReduce(m_action_flags, m_action_scalar, count))
      return(true);
   if(direction == 1 && source.BufferRead() && m_action_context.BufferRead() && m_action_flags.BufferRead())
     {
      const uint row_size = m_shape.H * 6;
      for(uint i = 0; i < count; i++)
         if(m_action_flags.At((int)i) != 0.0f)
           {
            const uint candidate = i / row_size;
            const uint channel = i % 6;
            const float active = m_action_context.At((int)(active_offset + candidate));
            PrintFormat("DPCC_ACTION6_AFFINE_FAIL mode=%d i=%u m=%u h=%u c=%u value=%.8g active=%.3g flag=%.0f",
                        (int)m_action_meta.mode, i, candidate, (i % row_size) / 6, channel,
                        source.At((int)i), active, m_action_flags.At((int)i));
            break;
           }
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Retains the native projector's exact per-candidate disposition   |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::MergeAction6ProjectionMask(CBufferFloat *corrected)
  {
   if(!m_actor_projector.GetCandidateStatus() || !ValidBuffer(corrected, (ulong)m_shape.M * m_shape.H * 6, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_action_context, m_action_candidate_status, m_actor_projector.GetCandidateStatus(),
                              corrected, m_action_flags
                             };
   const uint args[] = {m_shape.M, m_shape.H, (uint)m_action_context_layout.offsets[7]};
   const int kernel = def_k_DPCCCompositeProjectionMask;
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(!OpenCL.SetArgumentBuffer(kernel, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   for(uint i = 0; i < (uint)ArraySize(args); i++)
      if(!OpenCL.SetArgument(kernel, (int)(i + ArraySize(buffers)), args[i]))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {m_shape.M};
   return(OpenCL.Execute(kernel, 1, offset, global) && CompositeReduce(m_action_flags, m_action_scalar, m_shape.M));
  }
//+------------------------------------------------------------------+
//| Executes the checked native batch rollout and retains private    |
//| data                                                             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::FinalizeAction6Plans(void)
  {
   const uint count = m_shape.M * m_shape.H * 6;
   if(!Action6Affine(m_action_current, m_action_physical, 0) ||
      !CopyBufferRaw(m_action_physical, m_plan_source.getOutput(), count) ||
      (bTrain && (!m_rollout.FeedForward(m_plan_source, m_action_context) || !m_rollout.GetValidity())) ||
      (bTrain && !CopyBufferRaw(m_rollout.getOutput(), m_action_returns, m_shape.M * m_shape.K)) ||
      (bTrain ? !CopyBufferRaw(m_rollout.GetValidity(), m_action_status, m_shape.M) : !m_action_status.Fill(0)))
      return(Fail(m_rollout.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_rollout.LastStatus()));
   CBufferFloat *buffers[] = {m_action_candidate_status, m_action_status, m_action_flags};
   uint integers[] = {m_shape.M, (uint)(bTrain ? 1 : 0)};
   float reals[];
   return(CompositeDispatch(def_k_DPCCAction6CandidateMask, buffers, integers, reals, m_shape.M));
  }
//+------------------------------------------------------------------+
//| ABI3 unranked candidate generation; Output6 remains unpublished  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::RunAction6Forward(CNeuronBaseOCL *account,
                                       CNeuronScenarioForecastPriceOCL *market)
  {
   if(!m_action_context_bound || !market || market.Generation() != m_action_meta.market_state_id ||
      !m_action_meta.market_state_id || !ValidateAction6Workspace())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_action_market_weights_id == 0)
     {
      ulong weights_id = ulong(0xCBF29CE484222325);
      if(StringLen(m_action_meta.market_sha256) != 64)
         return(Fail(DPCC_INVALID_INPUT));
      for(uint i = 0; i < 64; i++)
        {
         const ushort c = StringGetCharacter(m_action_meta.market_sha256, (int)i);
         if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return(Fail(DPCC_INVALID_INPUT));
         weights_id = (weights_id ^ (ulong)c) * ulong(0x100000001B3);
        }
      if(weights_id == 0)
         return(Fail(DPCC_INVALID_INPUT));
      m_action_market_weights_id = weights_id;
     }
   if(!m_denoiser.BindFrameCondition(m_shape.K, m_shape.V, m_actor_market_horizon,
                                     m_actor_latent_dimension, m_actor_scale, m_actor_bias, m_action_meta.market_state_id,
                                     m_action_market_weights_id, m_config.probability_tolerance))
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=bind_frame status=%d denoiser_status=%d",
                  (int)m_status, (int)m_denoiser.LastStatus());
      return(Fail(m_denoiser.LastStatus() == DPCC_OK ? DPCC_CONTRACT_MISMATCH : m_denoiser.LastStatus()));
     }
   if(!PackAction6Condition(account, market))
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=condition_pack status=%d", (int)m_status);
      return(false);
     }
   if(!InitializeAction6Noise())
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=noise_init status=%d", (int)m_status);
      return(false);
     }
   if(!m_denoiser.BeginUnroll(m_action_meta.decision_id, m_shape.J, m_shape.M, bTrain))
     {
      PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=begin_unroll status=%d denoiser_status=%d",
                  (int)m_status, (int)m_denoiser.LastStatus());
      return(Fail(m_denoiser.LastStatus()));
     }
   m_action_unroll_live = true;
   m_action_market_generation = market.Generation();
//--- Frozen inference needs projection, but never records a training trace.
   const uint trace_frames = (bTrain ?
                              (m_action_meta.mode == DPCC_P2 ? m_shape.J :
                               (m_action_meta.mode == DPCC_P1 ? 1 : 0)) : 0);
   bool success = true;
   if(trace_frames > 0)
     {
      success = m_actor_projector.BeginTrace(m_action_meta.decision_id, trace_frames);
      m_action_trace_live = success;
      if(!success)
         PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=begin_trace projector_status=%d",
                     (int)m_actor_projector.LastStatus());
     }
   uint slot = 0;
   for(uint j = m_shape.J; success && j > 0; j--)
     {
      success = m_denoiser.ForwardFrame(m_shape.J - j, j, m_action_current, m_action_condition, m_action_epsilon);
      if(!success)
         PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=denoiser_frame timestep=%u status=%d",
                     j, (int)m_denoiser.LastStatus());
      if(success)
        {
         success = ReverseAction6Step(j);
         if(!success)
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=reverse_step timestep=%u status=%d",
                        j, (int)m_status);
        }
      if(success && m_action_meta.mode == DPCC_P2) // Intermediate frames refine P2; apply feasibility only at j == 1.
        {
         const uint projection_slot = slot;
         const uint cache_slot = (bTrain ? projection_slot : UINT_MAX);
         if(!Action6Affine(m_action_next, m_action_physical, 0, cache_slot))
           {
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projection_to_physical timestep=%u status=%d", j, (int)m_status);
            success = false;
           }
         else if(!CopyBufferRaw(m_action_physical, m_plan_source.getOutput(), m_shape.M * m_shape.H * 6))
           {
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projection_plan_copy timestep=%u", j);
            success = false;
           }
         else if(!ProjectAction6Current(projection_slot))
            success = false;
         else if(!Action6Affine(m_action_physical, m_action_next, 1, cache_slot))
           {
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projection_to_latent timestep=%u status=%d", j, (int)m_status);
            success = false;
           }
         else if(j == 1 && !MergeAction6ProjectionMask(m_action_next))
           {
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=projection_mask timestep=%u status=%d", j, (int)m_status);
            success = false;
           }
         if(success)
            slot++;
        }
      if(success && !CopyBufferRaw(m_action_next, m_action_current, m_shape.M * m_shape.H * 6))
        {
         PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=action_state_copy timestep=%u", j);
         success = false;
        }
     }
   if(success && m_action_meta.mode == DPCC_P1)
     {
      const uint cache_slot = (bTrain ? 0 : UINT_MAX);
      success = Action6Affine(m_action_current, m_action_physical, 0, cache_slot);
      if(!success)
         PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=p1_action_to_physical status=%d", (int)m_status);
      if(success)
        {
         success = CopyBufferRaw(m_action_physical, m_plan_source.getOutput(), m_shape.M * m_shape.H * 6);
         if(!success)
            Print("DPCC_ACTION6_FORWARD_FAIL stage=p1_plan_copy");
        }
      if(success)
         success = ProjectAction6Current(0);
      if(success)
        {
         success = Action6Affine(m_action_physical, m_action_next, 1, cache_slot);
         if(!success)
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=p1_projected_to_latent status=%d",
                        (int)m_status);
        }
      if(success)
        {
         success = MergeAction6ProjectionMask(m_action_next);
         if(!success)
            PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=p1_projection_mask status=%d",
                        (int)m_status);
        }
      if(success)
        {
         success = CopyBufferRaw(m_action_next, m_action_current, m_shape.M * m_shape.H * 6);
         if(!success)
            Print("DPCC_ACTION6_FORWARD_FAIL stage=p1_state_copy");
        }
     }
   if(success)
     {
      success = FinalizeAction6Plans();
      if(!success)
         PrintFormat("DPCC_ACTION6_FORWARD_FAIL stage=rollout_finalize status=%d rollout_status=%d",
                     (int)m_status, (int)m_rollout.LastStatus());
     }
   if(success && !bTrain)
     {
      if(m_action_trace_live)
        {
         success = m_actor_projector.ReleaseTrace();
         if(success)
            m_action_trace_live = false;
        }
      if(success)
        {
         success = m_denoiser.ReleaseUnroll();
         if(success)
            m_action_unroll_live = false;
        }
     }
   if(!success)
      AbortAction6Unroll();
   m_action_candidates_valid = success;
   m_public_valid = false;
   if(!success)
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Aborts an Action6 transaction while its shared weights are       |
//| intact                                                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::AbortAction6Unroll(void)
  {
   if(m_actor_abi != 3 || m_action_update_started)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   bool success = true;
   if(m_action_unroll_live)
      success = m_denoiser.AbortUnroll();
   if(m_action_trace_live)
     {
      if(!m_actor_projector.ReleaseTrace())
         success = false;
     }
   if(!success)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_action_unroll_live = false;
   m_action_trace_live = false;
   m_action_gradient_bound = false;
   m_action_backward_ready = false;
   m_action_candidates_valid = false;
   m_action_selected = UINT_MAX;
   m_public_valid = false;
   ClearPolicyState();
   return(true);
  }
//+------------------------------------------------------------------+
//| Seeds only the selected physical first-action derivative         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindAction6Gradient(const ulong decision_id, const uint candidate,
      CBufferFloat *gradient6)
  {
   if(m_actor_abi != 3 || !bTrain || m_failed || m_busy || m_transfer_pending ||
      !m_action_unroll_live || m_action_gradient_bound || m_action_backward_ready ||
      m_action_update_started || !m_action_candidates_valid || decision_id == 0 ||
      decision_id != m_action_meta.decision_id || candidate >= m_shape.M || candidate != m_action_selected ||
      !ValidateAction6Workspace() || !ValidBuffer(gradient6, 6, OpenCL) ||
      (gradient6 != Gradient && OwnsBufferOrDeviceSlot(gradient6)))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(gradient6 != Gradient && !CopyBufferRaw(gradient6, Gradient, 6))
     {
      const ENUM_DPCC_STATUS reason = m_status;
      AbortAction6Unroll();
      return(Fail(reason));
     }
   CBufferFloat *buffers[] = {Gradient, m_action_candidate_status, m_action_status,
                              m_action_context, m_action_current, m_action_flags
                             };
   const uint integers[] = {m_shape.M, m_shape.H, candidate, (uint)m_action_context_layout.offsets[7]};
   float reals[];
   if(!CompositeDispatch(def_k_DPCCAction6GradientSeed, buffers, integers, reals, m_shape.M * m_shape.H * 6) ||
      !CompositeReduce(m_action_flags, m_action_scalar, m_shape.M * m_shape.H * 6))
     {
      const ENUM_DPCC_STATUS reason = m_status;
      AbortAction6Unroll();
      return(Fail(reason));
     }
   m_action_selected = candidate;
   m_action_gradient_bound = true;
//--- The eligibility scratch becomes gradient status after selection.
   m_action_candidates_valid = false;
   ClearPolicyState();
   return(true);
  }
//+------------------------------------------------------------------+
//| Seed every selected plan coordinate with its physical derivative |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindFullPlanGradient(const ulong decision_id, const uint candidate,
      CBufferFloat *gradient_h6)
  {
   const ulong row = (ulong)m_shape.H * 6;
   const ulong count = (ulong)m_shape.M * row;
   const bool oracle_fallback = (m_status == DPCC_NO_FEASIBLE_CANDIDATE);
   const bool candidate_binding = (oracle_fallback ?
                                   (!m_public_valid && m_action_selected == UINT_MAX) :
                                   (m_public_valid && m_action_selected < m_shape.M));
   if(m_actor_abi != 3 || !bTrain || m_failed || m_busy || m_transfer_pending ||
      !m_action_unroll_live || m_action_gradient_bound || m_action_backward_ready ||
      m_action_update_started || !m_action_candidates_valid || !candidate_binding || decision_id == 0 ||
      decision_id != m_action_meta.decision_id || candidate >= m_shape.M ||
      !ValidateAction6Workspace() || count == 0 || count > (ulong)INT_MAX ||
      !ValidBuffer(gradient_h6, row, OpenCL) || OwnsBufferOrDeviceSlot(gradient_h6))
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- The caller supplies the physical derivative, including any horizon discount.
   CBufferFloat *buffers[] = {gradient_h6, m_action_candidate_status, m_action_status,
                              m_action_context, m_action_current, m_action_flags
                             };
   const uint allow_invalid_rollout = (oracle_fallback ? 1u : 0u);
   const uint integers[] = {m_shape.M, m_shape.H, candidate,
                            (uint)m_action_context_layout.offsets[7], allow_invalid_rollout
                           };
   float reals[];
   bool seeded = CompositeDispatch(def_k_DPCCFullPlanGradientSeed, buffers, integers, reals, (uint)count);
   if(seeded)
      seeded = CompositeReduce(m_action_flags, m_action_scalar, (uint)count);
   if(!seeded)
     {
      const ENUM_DPCC_STATUS reason = m_status;
      AbortAction6Unroll();
      return(Fail(reason));
     }
//--- BackwardAction6 consumes this same candidate-major reverse-unroll buffer.
   m_action_selected = candidate;
   m_action_gradient_bound = true;
   m_action_candidates_valid = false;
   ClearPolicyState();
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::BindSelectedPolicyGradient              |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindSelectedPolicyGradient(const ulong decision_id, const uint critic_index,
      const float upstream)
  {
   if(m_actor_abi != 3 || !bTrain || m_failed || m_busy || m_transfer_pending ||
      !m_action_unroll_live || !m_action_candidates_valid || !m_public_valid ||
      decision_id == 0 || decision_id != m_action_meta.decision_id || m_action_selected >= m_shape.M ||
      critic_index > 1 || !m_policy_networks_bound || !m_policy_state_valid ||
      !ValidatePolicyNetworks(m_policy_market, m_policy_q1, m_policy_q2) || !ValidBuffer(Output, 6, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   bool gradient_ready = false;
   m_busy = true;
   if(critic_index == 0)
      gradient_ready = NNQSelectedPolicyGradient(m_policy_q1, m_policy_market, m_action_account13, Output,
                       m_config.risk_type, m_config.risk_parameter, m_config.mean_weight, upstream,
                       m_policy_q1_state, m_policy_q1_rows, Gradient);
   else
      gradient_ready = NNQSelectedPolicyGradient(m_policy_q2, m_policy_market, m_action_account13, Output,
                       m_config.risk_type, m_config.risk_parameter, m_config.mean_weight, upstream,
                       m_policy_q2_state, m_policy_q2_rows, Gradient);
   m_busy = false;
   if(!gradient_ready)
     {
      ClearPolicyState();
      const ENUM_DPCC_STATUS reason = (m_status == DPCC_OK ? DPCC_NUMERIC_ERROR : m_status);
      AbortAction6Unroll();
      return(Fail(reason));
     }
   if(!BindAction6Gradient(decision_id, m_action_selected, Gradient))
     {
      ClearPolicyState();
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Applies action, reverse-DDPM and branch-sum derivatives          |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Action6GradientStep(const uint mode, CBufferFloat *source,
      CBufferFloat *other, CBufferFloat *result,
      CBufferFloat *epsilon, const uint j, const uint other_offset)
  {
   const uint count = m_shape.M * m_shape.H * 6;
   if(mode > def_k_DPCCAction6GradientStep_final_roundtrip_pullback ||
      (mode == def_k_DPCCAction6GradientStep_ddpm_split && (j == 0 || j > m_shape.J)))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {source, result, epsilon, m_action_flags};
   const ulong counts[] = {count, count, count, count};
   if(!CompositeBuffers(buffers, counts))
      return(false);
   if(!other || other.GetOpenCL() != OpenCL || other.GetIndex() < 0 ||
      (ulong)other_offset + count > (ulong)other.Total() ||
      OpenCL.GetBufferSize(other.GetIndex()) != (ulong)other.Total() * sizeof(float))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < (uint)ArraySize(buffers); i++)
      if(BufferAliases(other, buffers[i]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *kernel_buffers[] = {source, other, result, epsilon, m_action_flags};
   if(other_offset > UINT_MAX - count)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint integers[] = {count, mode, other_offset};
   float reals[8];
   reals[0] = (mode == def_k_DPCCAction6GradientStep_ddpm_split ? (float)m_beta[j] : 0.0f);
   reals[1] = (mode == def_k_DPCCAction6GradientStep_ddpm_split ? (float)m_alpha_bar[j] : 0.0f);
   for(uint c = 0; c < 6; c++)
     {
      reals[c + 2] = m_config.action_scale[c];
      if(!MathIsValidNumber(reals[c + 2]) || reals[c + 2] <= 0)
         return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   if(!CompositeDispatch(def_k_DPCCAction6GradientStep, kernel_buffers, integers, reals, count))
      return(false);
   return(CompositeReduce(m_action_flags, m_action_scalar, count));
  }
//+------------------------------------------------------------------+
//| Reverses saved local projection branches and every denoiser      |
//| frame                                                            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BackwardAction6(CNeuronBaseOCL *account)
  {
   if(m_actor_abi != 3 || !bTrain || m_failed || m_busy || m_transfer_pending ||
      !m_action_unroll_live || !m_action_gradient_bound || m_action_backward_ready ||
      m_action_update_started || m_action_selected >= m_shape.M || !ValidateAction6() ||
      !account || account.GetOpenCL() != OpenCL || account.Neurons() != 13 ||
      !ValidBuffer(account.getGradient(), 13, OpenCL) || OwnsBufferOrDeviceSlot(account.getGradient()))
     {
      PrintFormat("DPCC_A6_BWD_FAIL pre abi=%u train=%d failed=%d busy=%d unroll=%d bound=%d selected=%u valid=%d account=%d",
                  m_actor_abi, (int)bTrain, (int)m_failed, (int)m_busy,
                  (int)m_action_unroll_live, (int)m_action_gradient_bound, m_action_selected,
                  (int)ValidateAction6(), (int)(CheckPointer(account) != POINTER_INVALID));
      return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   m_busy = true;
   uint failure_stage = 1;
   uint failure_frame = 0;
   bool success = m_denoiser.ZeroAccumulatedGradients();
   for(uint j = 1; success && j <= m_shape.J; j++)
     {
      failure_frame = j;
      //--- The first upstream is physical A_checked; later P2 maps include N.
      const bool project = (m_action_meta.mode == DPCC_P2 ||
                            (m_action_meta.mode == DPCC_P1 && j == 1));
      if(project)
        {
         const uint slot = (m_action_meta.mode == DPCC_P2 ? m_shape.J - j : 0);
         const uint probability_offset = slot * (m_shape.M * m_shape.H * 6);
         CBufferFloat *upstream = m_action_next;
         const uint pullback_mode = (j == 1 ?
                                     def_k_DPCCAction6GradientStep_final_roundtrip_pullback :
                                     def_k_DPCCAction6GradientStep_normalization_pullback);
         failure_stage = 2;
         success = Action6GradientStep(pullback_mode, m_action_current, m_action_projected_frames,
                                       m_action_next, m_action_epsilon, 0, probability_offset);
         if(success)
           {
            failure_stage = 3;
            success = m_actor_projector.BackwardFrame(slot, upstream, m_action_physical);
           }
         if(success)
           {
            failure_stage = 4;
            success = Action6GradientStep(def_k_DPCCAction6GradientStep_sigmoid_pullback,
                                          m_action_physical, m_action_probability_frames,
                                          m_action_current, m_action_epsilon, 0, probability_offset);
           }
        }
      else
         if(j == 1)
           {
            failure_stage = 5;
            success = Action6GradientStep(def_k_DPCCAction6GradientStep_sigmoid_pullback, m_action_current, m_action_physical,
                                          m_action_next, m_action_epsilon) &&
                      CopyBufferRaw(m_action_next, m_action_current, m_shape.M * m_shape.H * 6);
           }
      if(!success)
         break;
      //--- Saved frame inputs own the condition; its external buffer is now the gradient sink.
      failure_stage = 6;
      success = Action6GradientStep(def_k_DPCCAction6GradientStep_ddpm_split, m_action_current, m_action_physical,
                                    m_action_next, m_action_epsilon, j);
      if(success)
        {
         failure_stage = 7;
         success = m_denoiser.BackwardFrame(m_shape.J - j, m_action_epsilon,
                                            m_action_physical, m_action_condition);
        }
      if(success)
        {
         failure_stage = 8;
         success = Action6GradientStep(def_k_DPCCAction6GradientStep_branch_sum, m_action_next, m_action_physical,
                                       m_action_current, m_action_epsilon);
        }
     }
   if(success)
     {
      failure_stage = 9;
      //--- Account13 occupies the first 13 coordinates of the accumulated condition VJP.
      CBufferFloat *buffers[] = {m_action_condition, account.getGradient(), m_action_flags};
      const uint integers[] = {uint(None), 13};
      float reals[];
      success = CompositeDispatch(def_k_DPCCActivationForward, buffers, integers, reals, 13);
      if(success)
         success = CompositeReduce(m_action_flags, m_action_scalar, 13);
     }
   const ENUM_DPCC_STATUS reason = (m_status == DPCC_OK ? m_denoiser.LastStatus() : m_status);
   m_busy = false;
   if(!success)
     {
      PrintFormat("DPCC_ACTION6_BACKWARD_FAIL stage=%u frame=%u status=%d denoiser_status=%d",
                  failure_stage, failure_frame, (int)reason, (int)m_denoiser.LastStatus());
      AbortAction6Unroll();
      return(Fail(reason == DPCC_OK ? DPCC_CONTRACT_MISMATCH : reason));
     }
   m_action_backward_ready = true;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| CNet sends a bound Action6 derivative through the owned unroll   |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::calcInputGradients(CNeuronBaseOCL *source)
  {
   return(calcInputGradients(source, NULL, NULL, None));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::calcInputGradients                      |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::calcInputGradients(CNeuronBaseOCL *source, CBufferFloat *second,
                                        CBufferFloat *gradient, ENUM_ACTIVATION function)
  {
   if(m_actor_abi != 3)
      return(Fail(DPCC_BACKWARD_UNSUPPORTED));
   if(second != NULL || gradient != NULL || function != None)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   return(BackwardAction6(source));
  }
//+------------------------------------------------------------------+
//| Applies one Adam step after the complete reverse unroll          |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::UpdateInputWeights(CObject *source, CBufferFloat *second)
  {
   if(m_actor_abi != 3)
      return(Fail(DPCC_BACKWARD_UNSUPPORTED));
   CNeuronBaseOCL *account = source;
   if(!bTrain || m_failed || m_busy || m_transfer_pending || second != NULL ||
      !m_action_unroll_live || !m_action_gradient_bound || !m_action_backward_ready ||
      m_action_update_started || !account || account.GetOpenCL() != OpenCL || account.Neurons() != 13)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_busy = true;
   if(!m_denoiser.ApplyAccumulatedGradients())
     {
      const ENUM_DPCC_STATUS reason = m_denoiser.LastStatus();
      m_busy = false;
      //--- Abort succeeds only before the child's first shared parameter write.
      AbortAction6Unroll();
      return(Fail(reason == DPCC_OK ? DPCC_CONTRACT_MISMATCH : reason, m_failed));
     }
   m_action_update_started = true;
   bool success = m_denoiser.ReleaseUnroll();
   if(success)
      m_action_unroll_live = false;
   if(m_action_trace_live)
     {
      if(!m_actor_projector.ReleaseTrace())
         success = false;
      else
         m_action_trace_live = false;
     }
   m_busy = false;
   if(!success)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_action_update_started = false;
   m_action_gradient_bound = false;
   m_action_backward_ready = false;
   m_action_selected = UINT_MAX;
   m_public_valid = false;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Releases skipped or frozen policy frames without optimizer       |
//| writes                                                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CancelGeneration(void)
  {
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending || m_action_update_started)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!AbortAction6Unroll())
      return(false);
   m_action_context_bound = false;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::UpdateInputState                        |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::UpdateInputState(CObject *source, CBufferFloat *second)
  {
   if(m_actor_abi != 3)
      return(Fail(DPCC_BACKWARD_UNSUPPORTED));
   CNeuronBaseOCL *account = source;
   if(!account || account.GetOpenCL() != OpenCL || account.Neurons() != 13 || second != NULL ||
      !m_action_backward_ready)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   return(CancelGeneration());
  }
//+------------------------------------------------------------------+
//| CNet secondary-input entry binds the Market ABI explicitly       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::feedForwardSecond(CNeuronBaseOCL *source, CNeuronBaseOCL *second)
  {
   if(m_action_unroll_live || m_action_trace_live)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_public_valid = false;
   m_action_candidates_valid = false;
   if(!Output.Fill(0) || !PrevOutput.Fill(0) || !Gradient.Fill(0))
      return(Fail(DPCC_OPENCL_ERROR));
   if(m_actor_abi != 3 || m_failed || m_busy || m_transfer_pending || !m_configured ||
      !m_action_context_bound || !source || !source.CanStartForward() || !second ||
      second.Type() != defNeuronScenarioForecastPriceOCL)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!m_policy_networks_bound || !ValidatePolicyNetworks(m_policy_market, m_policy_q1, m_policy_q2))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_busy = true;
   m_status = DPCC_OK;
   bool success = RunAction6Forward(source, (CNeuronScenarioForecastPriceOCL*)second);
   if(!success && m_status != DPCC_NO_FEASIBLE_CANDIDATE)
      PrintFormat("DPCC_ACTION6_FORWARD_STAGE stage=plan_generation status=%d denoiser_status=%d",
                  (int)m_status, (int)m_denoiser.LastStatus());
   if(success)
     {
      success = EvaluateAndPublishAction6((CNeuronScenarioForecastPriceOCL*)second);
     }
   m_action_context_bound = false;
   m_busy = false;
   if(!success)
     {
      const ENUM_DPCC_STATUS reason = (m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status);
      //--- Treat no feasible plan as a valid no-trade outcome.
      //--- Training preserves Oracle plans; inference publishes no action.
      if(reason == DPCC_NO_FEASIBLE_CANDIDATE && m_action_candidates_valid &&
         m_action_market_generation == m_action_meta.market_state_id &&
         m_action_selected == UINT_MAX &&
         ((bTrain && m_action_unroll_live) ||
          (!bTrain && !m_action_unroll_live && !m_action_trace_live)))
        {
         m_action_context_bound = false;
         m_busy = false;
         m_public_valid = false;
         m_status = DPCC_NO_FEASIBLE_CANDIDATE;
         return(true);
        }
      if(m_action_unroll_live || m_action_trace_live)
         AbortAction6Unroll();
      ClearPolicyState();
      m_public_valid = false;
      return(Fail(reason));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Writes or reads explicitly typed ABI3 profile fields             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Action6Profile(const int f, const bool restore, SDPCCConfig &c,
                                    double &beta[], double &schedule_alpha[], double &sigma[],
                                    float &scale[], float &bias[])
  {
   uint n[] = {c.shape.K, c.shape.V, c.shape.H, c.shape.D, c.shape.M, c.shape.J, c.shape.E,
               c.shape.PlanBatch, c.shape.NActions, c.shape.StateSize, c.shape.Embedding, c.shape.Heads,
               c.shape.KeyDimension, c.shape.Depth, c.contract_version, c.binary_layer_version, (uint)c.mode,
               c.projection_max_iterations, c.projection_max_evaluations, c.projection_lookback_steps,
               c.price_basis, c.quote_adapter, c.event_model, (uint)c.risk_type, c.normalization_mode,
               c.loss_coordinates, c.randomness_version, c.margin_fx_profile, c.account_currency_digits
              };
   double r[] = {c.projection_improvement_atol, c.projection_distance_tie_atol, c.volume_scale,
                 c.price_scale, c.money_scale, c.max_quote_age_seconds, c.max_last_age_seconds,
                 (double)c.risk_parameter, (double)c.mean_weight, (double)c.gamma, (double)c.probability_tolerance
                };
   if(restore)
     {
      if(FileReadArray(f, n, 0, 29) != 29 || FileReadArray(f, r, 0, 11) != 11 || n[5] == 0 || n[5] > 64)
         return(false);
      c.shape.K = n[0];
      c.shape.V = n[1];
      c.shape.H = n[2];
      c.shape.D = n[3];
      c.shape.M = n[4];
      c.shape.J = n[5];
      c.shape.E = n[6];
      c.shape.PlanBatch = n[7];
      c.shape.NActions = n[8];
      c.shape.StateSize = n[9];
      c.shape.Embedding = n[10];
      c.shape.Heads = n[11];
      c.shape.KeyDimension = n[12];
      c.shape.Depth = n[13];
      c.contract_version = n[14];
      c.binary_layer_version = n[15];
      c.mode = (ENUM_DPCC_MODE)n[16];
      c.projection_max_iterations = n[17];
      c.projection_max_evaluations = n[18];
      c.projection_lookback_steps = n[19];
      c.price_basis = n[20];
      c.quote_adapter = n[21];
      c.event_model = n[22];
      c.risk_type = (int)n[23];
      c.normalization_mode = n[24];
      c.loss_coordinates = n[25];
      c.randomness_version = n[26];
      c.margin_fx_profile = n[27];
      c.account_currency_digits = n[28];
      c.projection_improvement_atol = r[0];
      c.projection_distance_tie_atol = r[1];
      c.volume_scale = r[2];
      c.price_scale = r[3];
      c.money_scale = r[4];
      c.max_quote_age_seconds = r[5];
      c.max_last_age_seconds = r[6];
      c.risk_parameter = (float)r[7];
      c.mean_weight = (float)r[8];
      c.gamma = (float)r[9];
      c.probability_tolerance = (float)r[10];
      if(ArrayResize(beta, (int)c.shape.J + 1) != (int)c.shape.J + 1 ||
         ArrayResize(schedule_alpha, (int)c.shape.J + 1) != (int)c.shape.J + 1 ||
         ArrayResize(sigma, (int)c.shape.J + 1) != (int)c.shape.J + 1)
         return(false);
     }
   else
      if(FileWriteArray(f, n) != 29 || FileWriteArray(f, r) != 11)
         return(false);
   if(ArraySize(scale) != 109 || ArraySize(bias) != 109)
      return(false);
   if(restore)
     {
      if(FileReadArray(f, c.action_bias, 0, 6) != 6 || FileReadArray(f, c.action_scale, 0, 6) != 6 ||
         FileReadArray(f, scale, 0, 109) != 109 || FileReadArray(f, bias, 0, 109) != 109 ||
         FileReadArray(f, beta, 0, c.shape.J + 1) != c.shape.J + 1 ||
         FileReadArray(f, schedule_alpha, 0, c.shape.J + 1) != c.shape.J + 1 ||
         FileReadArray(f, sigma, 0, c.shape.J + 1) != c.shape.J + 1)
         return(false);
     }
   else
      if(FileWriteArray(f, c.action_bias) != 6 || FileWriteArray(f, c.action_scale) != 6 ||
         FileWriteArray(f, scale) != 109 || FileWriteArray(f, bias) != 109 ||
         FileWriteArray(f, beta) != c.shape.J + 1 || FileWriteArray(f, schedule_alpha) != c.shape.J + 1 ||
         FileWriteArray(f, sigma) != c.shape.J + 1)
         return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves only a complete FREE ABI3 Actor state                      |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SaveAction6(const int f)
  {
   if(f == INVALID_HANDLE || m_failed || m_busy || m_transfer_pending || !m_configured ||
      m_public_valid || m_action_unroll_live || m_action_trace_live || !ValidateAction6())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong state = WeightSignature();
   const ulong contract = ContractSignature();
   if(state == 0 || contract == 0)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint envelope = m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ?
                         DPCC_HISTORICAL_FX_ENVELOPE_VERSION : (m_config.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 ?
                               DPCC_BINARY_LAYER_VERSION : DPCC_FINANCIAL_ENVELOPE_VERSION);
   const uint header[] = {(uint)Type(), 0x44504343, envelope, 3, (uint)m_myIndex, 1, 1,
                          m_actor_market_horizon, m_actor_latent_dimension
                         };
   SDPCCConfig config = m_config;
   if(FileWriteArray(f, header) != ArraySize(header) || FileWriteLong(f, (long)contract) != sizeof(long) ||
      FileWriteLong(f, (long)state) != sizeof(long) ||
      !Action6Profile(f, false, config, m_beta, m_alpha_bar, m_sigma, m_actor_scale, m_actor_bias) ||
      !m_rollout.Save(f) || !m_actor_projector.Save(f) || !m_denoiser.SaveFrameTrainingState(f) ||
      FileWriteLong(f, (long)state) != sizeof(long) || FileWriteInteger(f, 0x454E4433, INT_VALUE) != INT_VALUE)
      return(Fail(DPCC_INVALID_INPUT));
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores ABI3 shapes, native weights, moments and optimizer      |
//| clocks                                                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::LoadAction6(const int f, const uint envelope, const uint index, const uint batch)
  {
   if(batch != 1 || (m_admitted && m_actor_abi != 3) || FileTell(f) + 2 * INT_VALUE + 2 * sizeof(long) > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const uint market_horizon = (uint)FileReadInteger(f, INT_VALUE);
   const uint latent_dimension = (uint)FileReadInteger(f, INT_VALUE);
   const ulong expected_contract = (ulong)FileReadLong(f);
   const ulong expected_state = (ulong)FileReadLong(f);
   SDPCCRuntimeConfig runtime;
   ZeroMemory(runtime.financial);
   if(expected_contract == 0 || expected_state == 0 ||
      !Action6Profile(f, true, runtime.financial, runtime.beta, runtime.alpha_bar, runtime.sigma,
                      runtime.condition_scale, runtime.condition_bias))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const uint expected_envelope = runtime.financial.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ?
                                  DPCC_HISTORICAL_FX_ENVELOPE_VERSION : (runtime.financial.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 ?
                                        DPCC_BINARY_LAYER_VERSION : DPCC_FINANCIAL_ENVELOPE_VERSION);
   if(envelope != expected_envelope || (m_admitted && m_admitted_profile != expected_contract))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   ReleaseOwned();
   m_failed = false;
   m_transfer_pending = false;
   m_transfer_source = NULL;
   m_transfer_target = NULL;
   m_shape = runtime.financial.shape;
   m_actor_market_horizon = market_horizon;
   m_actor_latent_dimension = latent_dimension;
   CLayerDescription desc;
   if(!DescribeAction6(desc) || !InitAction6(index, OpenCL, GetPointer(desc)) || !Configure(runtime))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_actor_projector.UnbindRollout();
   if(!LoadInsideLayer(f, m_rollout) ||
      !LoadInsideLayer(f, m_actor_projector) ||
      !m_denoiser.LoadFrameTrainingState(f) || FileTell(f) + sizeof(long) + INT_VALUE > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const ulong saved_state = (ulong)FileReadLong(f);
   if(saved_state != expected_state || FileReadInteger(f, INT_VALUE) != 0x454E4433 ||
      !m_actor_projector.BindRollout(m_rollout) || !ValidateAction6() ||
      ContractSignature() != expected_contract || WeightSignature() != expected_state)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_imported = false;
   m_decision_serial = 0;
   m_bar_identity_valid = false;
   m_bar_symbol = "";
   m_bar_timeframe = 0;
   m_bar_time = 0;
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Initialize the DPCC composite layer                              |
//+------------------------------------------------------------------+
//--- Work-only CNeuronDPCCOCL owner/lifecycle implementation.
//--- Merge after shared DPCCCompositeLayouts and before the operators' bodies.
CNeuronDPCCOCL::CNeuronDPCCOCL(void)
  {
   m_actor_abi = 2;
   m_actor_market_horizon = 0;
   m_actor_latent_dimension = 0;
   m_actor_projector = NULL;
   m_action_context = NULL;
   m_action_condition = NULL;
   m_action_noise = NULL;
   m_action_current = NULL;
   m_action_next = NULL;
   m_action_physical = NULL;
   m_action_epsilon = NULL;
   m_action_probability_frames = NULL;
   m_action_projected_frames = NULL;
   m_action_flags = NULL;
   m_action_scalar = NULL;
   m_action_status = NULL;
   m_action_candidate_status = NULL;
   m_action_returns = NULL;
   m_action_account13 = NULL;
   m_action_q2_scores = NULL;
   m_action_rank_scores = NULL;
   m_action_numeric_flags = NULL;
   m_policy_market = NULL;
   m_policy_q1 = NULL;
   m_policy_q2 = NULL;
   m_policy_networks_bound = false;
   m_policy_state_valid = false;
   ZeroMemory(m_action_context_layout);
   ZeroMemory(m_action_meta);
   m_action_market_generation = 0;
   m_action_context_bound = false;
   m_action_candidates_valid = false;
   m_action_market_weights_id = 0;
   m_action_unroll_live = false;
   m_action_trace_live = false;
   m_action_gradient_bound = false;
   m_action_backward_ready = false;
   m_action_update_started = false;
   m_action_selected = UINT_MAX;
   ArrayInitialize(m_actor_scale, 0);
   ArrayInitialize(m_actor_bias, 0);
   ZeroMemory(m_shape);
   ZeroMemory(m_config);
   ZeroMemory(m_c_layout);
   ZeroMemory(m_r_layout);
   ZeroMemory(m_snapshot);
   ZeroMemory(m_decoder_composition);
   ZeroMemory(m_denoiser_composition);
   m_decoder = NULL;
   m_denoiser = NULL;
   m_rollout = NULL;
   m_projector = NULL;
   m_normalizer = NULL;
   m_denorm = NULL;
   m_decoder_source = NULL;
   m_plan_source = NULL;
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
     {
      m_work[i] = NULL;
      m_sizes[i] = 0;
     }
   for(uint i = 0; i < 8; i++)
      m_transferred[i] = false;
   m_bar_identity_valid = false;
   m_bar_symbol = "";
   m_bar_timeframe = 0;
   m_bar_time = 0;
   m_tokens = 0;
   m_padK = 0;
   m_decision_serial = 0;
   m_snapshot_device_hash = 0;
   m_snapshot_staged = false;
   m_admitted_profile = 0;
   m_admitted = false;
   m_loading = false;
   m_shaped = false;
   m_configured = false;
   m_imported = false;
   m_public_valid = false;
   m_failed = false;
   m_snapshot_valid = false;
   m_busy = false;
   m_transfer_pending = false;
   m_transfer_source = NULL;
   m_transfer_target = NULL;
   m_status = DPCC_CONTRACT_MISMATCH;
   bTrain = false;
  }
//+------------------------------------------------------------------+
//| Function ~CNeuronDPCCOCL                                         |
//+------------------------------------------------------------------+
CNeuronDPCCOCL::~CNeuronDPCCOCL(void) { ReleaseOwned(); }
//+------------------------------------------------------------------+
//| Invalidate cached DPCC state                                     |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::Invalidate(void)
  {
   m_public_valid = false;
   m_snapshot_valid = false;
   m_snapshot_device_hash = 0;
   m_snapshot_staged = false;
   ZeroMemory(m_snapshot);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Fail                                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Fail(const ENUM_DPCC_STATUS status, const bool fatal)
  {
   m_public_valid = false;
   m_status = status;
   if(fatal)
     {
      m_failed = true;
      m_imported = false;
      m_loading = false;
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ReleaseOwned                            |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::ReleaseOwned(void)
  {
   Invalidate();
   ClearPolicyState();
   ClearPolicyWorkspace();
   m_policy_market = NULL;
   m_policy_q1 = NULL;
   m_policy_q2 = NULL;
   m_policy_networks_bound = false;
   if(m_actor_projector)
      m_actor_projector.UnbindRollout();
   DeleteObj(m_actor_projector);
   m_actor_projector = NULL;
   if(m_projector)
      m_projector.UnbindRollout();
   DeleteObj(m_projector);
   DeleteObj(m_rollout);
   DeleteObj(m_denoiser);
   DeleteObj(m_decoder);
   m_projector = NULL;
   m_rollout = NULL;
   m_denoiser = NULL;
   m_decoder = NULL;
   DeleteObj(m_denorm);
   DeleteObj(m_normalizer);
   m_denorm = NULL;
   m_normalizer = NULL;
   DeleteObj(m_decoder_source);
   DeleteObj(m_plan_source);
   m_decoder_source = NULL;
   m_plan_source = NULL;
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
     {
      DeleteObj(m_work[i]);
      m_work[i] = NULL;
      m_sizes[i] = 0;
     }
   DeleteObj(m_action_context);
   DeleteObj(m_action_condition);
   DeleteObj(m_action_noise);
   DeleteObj(m_action_current);
   DeleteObj(m_action_next);
   DeleteObj(m_action_physical);
   DeleteObj(m_action_epsilon);
   DeleteObj(m_action_probability_frames);
   DeleteObj(m_action_projected_frames);
   DeleteObj(m_action_flags);
   DeleteObj(m_action_scalar);
   DeleteObj(m_action_status);
   DeleteObj(m_action_candidate_status);
   DeleteObj(m_action_returns);
   DeleteObj(m_action_account13);
   DeleteObj(m_action_q2_scores);
   DeleteObj(m_action_rank_scores);
   DeleteObj(m_action_numeric_flags);
   m_action_context = NULL;
   m_action_condition = NULL;
   m_action_noise = NULL;
   m_action_current = NULL;
   m_action_next = NULL;
   m_action_physical = NULL;
   m_action_epsilon = NULL;
   m_action_probability_frames = NULL;
   m_action_projected_frames = NULL;
   m_action_flags = NULL;
   m_action_scalar = NULL;
   m_action_status = NULL;
   m_action_candidate_status = NULL;
   m_action_returns = NULL;
   m_action_account13 = NULL;
   m_action_q2_scores = NULL;
   m_action_rank_scores = NULL;
   m_action_numeric_flags = NULL;
   ZeroMemory(m_action_context_layout);
   ZeroMemory(m_action_meta);
   m_action_market_generation = 0;
   m_action_context_bound = false;
   m_action_candidates_valid = false;
   m_action_market_weights_id = 0;
   m_action_unroll_live = false;
   m_action_trace_live = false;
   m_action_gradient_bound = false;
   m_action_backward_ready = false;
   m_action_update_started = false;
   m_action_selected = UINT_MAX;
   m_shaped = false;
   m_configured = false;
   m_imported = false;
   m_actor_abi = 2;
   m_actor_market_horizon = 0;
   m_actor_latent_dimension = 0;
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ValidBuffer                             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidBuffer(CBufferFloat *buffer, const ulong count, COpenCLMy *owner) const
  {
   return(buffer && owner && count > 0 && count <= INT_MAX && count <= (ulong)UINT_MAX / sizeof(float) &&
          buffer.Total() == (int)count && buffer.GetOpenCL() == owner && buffer.GetIndex() >= 0 &&
          owner.GetBufferSize(buffer.GetIndex()) == count * sizeof(float));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Geometry                                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Geometry(void)
  {
   if(m_shape.Depth != 0 || m_shape.PlanBatch != m_shape.M ||
      m_shape.Embedding != (ulong)m_shape.Heads * m_shape.KeyDimension ||
      !DPCCCompositeLayouts(m_shape, m_c_layout, m_r_layout, m_tokens))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_padK = 1;
   while(m_padK < m_shape.K)
      m_padK <<= 1;
   if(m_padK > defFQF_LOCAL_ARRAY_SIZE)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong mh6 = (ulong)m_shape.M * m_shape.H * 6;
   const ulong mn = (ulong)m_shape.M * m_padK;
   ulong flags = m_c_layout.total;
   if(flags < m_r_layout.total)
      flags = m_r_layout.total;
   if(flags < (ulong)m_tokens + 61)
      flags = (ulong)m_tokens + 61;
   if(flags < (ulong)m_shape.K * m_shape.H * 4)
      flags = (ulong)m_shape.K * m_shape.H * 4;
   const ulong sizes[] = {m_c_layout.total, m_r_layout.total, (ulong)m_tokens + 61, 13,
                          mh6, (ulong)m_shape.H * 6, (ulong)m_shape.M * m_shape.K,
                          m_shape.M, m_shape.M, m_shape.M, mn, mn, mn, mn, mn,
                          (ulong)m_shape.M * defFQF_DIAGNOSTIC_FLOATS,
                          (ulong)m_shape.M * 12, flags, 1, (ulong)m_shape.H * 4
                         };
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
     {
      if(sizes[i] == 0 || sizes[i] > INT_MAX || sizes[i] > (ulong)UINT_MAX / sizeof(float))
         return(Fail(DPCC_CONTRACT_MISMATCH));
      m_sizes[i] = sizes[i];
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Describe                                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Describe(CLayerDescription &desc, const uint type,
                              const uint count, const uint depth) const
  {
   desc.type = type;
   desc.count = count;
   desc.window = 6;
   desc.window_out = 13;
   desc.variables = m_shape.V;
   desc.step = m_shape.H;
   desc.layers = depth;
   desc.optimization = ADAM;
   desc.activation = None;
   desc.batch = iBatch;
   if(ArrayResize(desc.windows, 9) != 9 || ArrayResize(desc.units, 5) != 5 ||
      ArrayResize(desc.heads, 1) != 1)
      return(false);
   const uint windows[] = {2, m_shape.K, m_shape.V, m_shape.H, m_shape.D, m_shape.M, m_shape.J, m_shape.E, m_shape.PlanBatch};
   const uint units[] = {m_shape.Embedding, m_shape.Heads, m_shape.KeyDimension, 6, 13};
   for(uint i = 0; i < 9; i++)
      desc.windows[i] = windows[i];
   for(uint i = 0; i < 5; i++)
      desc.units[i] = units[i];
   desc.heads[0] = m_shape.Heads;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::AllocateOwned                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::AllocateOwned(void)
  {
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
     {
      m_work[i] = new CBufferFloat();
      if(!m_work[i] || !m_work[i].BufferInit((uint)m_sizes[i], 0) || !m_work[i].BufferCreate(OpenCL))
         return(Fail(DPCC_OPENCL_ERROR, true));
     }
   m_decoder_source = new CNeuronBaseOCL();
   m_plan_source = new CNeuronBaseOCL();
   if(!m_decoder_source || !m_plan_source ||
      !m_decoder_source.Init(0, 0, OpenCL, (uint)m_c_layout.lengths[0], ADAM, 1) ||
      !m_plan_source.Init(0, 0, OpenCL, m_shape.M * m_shape.H * 6, ADAM, 1))
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_decoder_source.SetActivationFunction(None);
   m_plan_source.SetActivationFunction(None);
   m_decoder_source.TrainMode(false);
   m_plan_source.TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::InitializeChildren                      |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::InitializeChildren(void)
  {
   m_decoder = new CNeuronScenarioDecoderOCL();
   m_denoiser = new CNeuronPlanDenoiserOCL();
   m_rollout = new CNeuronScenarioRolloutOCL();
   m_projector = new CNeuronPlanProjectionOCL();
   m_normalizer = new CNeuronBatchNormOCL();
   m_denorm = new CNeuronRevINDenormOCL();
   if(!m_decoder || !m_denoiser || !m_rollout || !m_projector || !m_normalizer || !m_denorm)
      return(Fail(DPCC_OPENCL_ERROR, true));
   CLayerDescription desc;
   if(!Describe(desc, defNeuronScenarioDecoderOCL, m_shape.K * m_shape.H * 4, 0) ||
      !m_decoder.Init(0, 0, OpenCL, GetPointer(desc)) ||
      !Describe(desc, defNeuronPlanDenoiserOCL, m_shape.M * m_shape.H * 6, 1) ||
      !m_denoiser.Init(0, 0, OpenCL, GetPointer(desc)) ||
      !Describe(desc, defNeuronScenarioRolloutOCL, m_shape.M * m_shape.K, 0) ||
      !m_rollout.Init(0, 0, OpenCL, GetPointer(desc)) ||
      !Describe(desc, defNeuronPlanProjectionOCL, m_shape.M * m_shape.H * 6, 0) ||
      !m_projector.Init(0, 0, OpenCL, GetPointer(desc)) ||
      !m_normalizer.Init(0, 0, OpenCL, uint(4), uint(2), ADAM) ||
      !m_denorm.InitOwned(0, OpenCL, m_shape.K * m_shape.H * 4))
      return(Fail(DPCC_OPENCL_ERROR, true));
   TrainMode(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ValidateOwned                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidateOwned(void) const
  {
   if(m_actor_abi == 3)
      return(ValidateAction6());
   if(!m_shaped || !OpenCL || !m_decoder || !m_denoiser || !m_rollout || !m_projector ||
      !m_normalizer || !m_denorm || !m_decoder_source || !m_plan_source ||
      !ValidBuffer(Output, 6, OpenCL) || !ValidBuffer(PrevOutput, 6, OpenCL) ||
      !ValidBuffer(Gradient, 6, OpenCL) || activation != None || optimization != ADAM || bTrain ||
      CheckPointer(Weights) != POINTER_INVALID || CheckPointer(DeltaWeights) != POINTER_INVALID ||
      CheckPointer(FirstMomentum) != POINTER_INVALID || CheckPointer(SecondMomentum) != POINTER_INVALID)
      return(false);
   CNeuronBaseOCL *children[] = {m_decoder, m_denoiser, m_rollout, m_projector, m_normalizer, m_denorm, m_decoder_source, m_plan_source};
   for(uint i = 0; i < 8; i++)
     {
      if(children[i].GetOpenCL() != OpenCL || children[i].TrainMode())
         return(false);
      for(uint j = 0; j < i; j++)
         if(children[i] == children[j])
            return(false);
     }
   if(!m_denorm.OwnedReady() || !ValidBuffer(m_normalizer.getBatchOptions(), 36, OpenCL) ||
      !ValidBuffer(m_denorm.getBatchOptions(), (ulong)m_shape.K * m_shape.H * 4 * 9, OpenCL))
      return(false);
   if(m_configured && (!m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) ||
                       m_projector.ContractSignature() != DPCCFixedSignature(m_config, defNeuronPlanProjectionOCL)))
      return(false);
   CBufferFloat *base[] = {Output, PrevOutput, Gradient, m_decoder_source.getOutput(),
                           m_decoder_source.getPrevOutput(), m_decoder_source.getGradient(),
                           m_plan_source.getOutput(), m_plan_source.getPrevOutput(), m_plan_source.getGradient(),
                           m_normalizer.getBatchOptions(), m_denorm.getBatchOptions()
                          };
   for(uint i = 0; i < (uint)ArraySize(base); i++)
     {
      if(!base[i] || base[i].GetOpenCL() != OpenCL || base[i].GetIndex() < 0)
         return(false);
      for(uint j = 0; j < i; j++)
         if(base[i] == base[j] || base[i].GetIndex() == base[j].GetIndex())
            return(false);
     }
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
     {
      if(!ValidBuffer(m_work[i], m_sizes[i], OpenCL))
         return(false);
      for(uint j = 0; j < i; j++)
         if(m_work[i] == m_work[j] || m_work[i].GetIndex() == m_work[j].GetIndex())
            return(false);
      for(uint j = 0; j < (uint)ArraySize(base); j++)
         if(m_work[i] == base[j] || m_work[i].GetIndex() == base[j].GetIndex())
            return(false);
     }
   return(ValidBuffer(m_decoder_source.getOutput(), m_c_layout.lengths[0], OpenCL) &&
          ValidBuffer(m_plan_source.getOutput(), (ulong)m_shape.M * m_shape.H * 6, OpenCL));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Init                                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Init(uint outputs, uint index, COpenCLMy *context, CLayerDescription *desc)
  {
   if(desc && outputs == 0 && ArraySize(desc.windows) == 0 && ArraySize(desc.units) == 8 &&
      desc.units[7] == 3)
      return(InitAction6(index, context, desc));
   if(m_failed || CheckPointer(m_decoder) != POINTER_INVALID || CheckPointer(m_decoder_source) != POINTER_INVALID || m_busy || m_transfer_pending || m_shaped || !context || !desc || outputs != 0 ||
      desc.type != Type() || desc.count != 6 || desc.layers != 0 || desc.window != 6 || desc.window_out != 13 ||
      desc.activation != None || desc.optimization != ADAM || desc.batch == 0 ||
      ArraySize(desc.windows) != 9 || ArraySize(desc.units) != 5 || ArraySize(desc.heads) != 1 || desc.windows[0] != 2)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_shape.K = desc.windows[1];
   m_shape.V = desc.windows[2];
   m_shape.H = desc.windows[3];
   m_shape.D = desc.windows[4];
   m_shape.M = desc.windows[5];
   m_shape.J = desc.windows[6];
   m_shape.E = desc.windows[7];
   m_shape.PlanBatch = desc.windows[8];
   m_shape.Embedding = desc.units[0];
   m_shape.Heads = desc.units[1];
   m_shape.KeyDimension = desc.units[2];
   m_shape.NActions = desc.units[3];
   m_shape.StateSize = desc.units[4];
   m_shape.Depth = 0;
   if(desc.variables != m_shape.V || desc.step != m_shape.H || desc.heads[0] != m_shape.Heads || !Geometry())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!CNeuronBaseOCL::Init(0, index, context, 6, ADAM, desc.batch))
      return(Fail(DPCC_OPENCL_ERROR, true));
   activation = None;
   bTrain = false;
   if(!AllocateOwned() || !InitializeChildren())
      return(Fail(m_status, true));
   m_shaped = true;
   m_failed = false;
   m_status = DPCC_OK;
   if(!ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(!m_loading)
     {m_decision_serial = 0; m_bar_identity_valid = false; m_bar_symbol = ""; m_bar_timeframe = 0; m_bar_time = 0;}
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ScheduleValid                           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ScheduleValid(const double &beta[], const double &alpha_bar[], const double &sigma[]) const
  {
   if(ArraySize(beta) != (int)m_shape.J + 1 || ArraySize(alpha_bar) != ArraySize(beta) ||
      ArraySize(sigma) != ArraySize(beta) || beta[0] != 0 || alpha_bar[0] != 1 || sigma[0] != 0 || sigma[1] != 0)
      return(false);
   for(uint j = 1; j <= m_shape.J; j++)
     {
      if(!MathIsValidNumber(beta[j]) || beta[j] <= 0 || beta[j] >= 1 ||
         !MathIsValidNumber(alpha_bar[j]) || alpha_bar[j] <= 0 || alpha_bar[j] >= 1 ||
         !MathIsValidNumber(sigma[j]) || sigma[j] < 0 ||
         !MathIsValidNumber((float)beta[j]) || !MathIsValidNumber((float)alpha_bar[j]) ||
         (float)beta[j] <= 0 || (float)beta[j] >= 1 || (float)alpha_bar[j] <= 0 ||
         (float)alpha_bar[j] >= 1 || !MathIsValidNumber((float)sigma[j]))
         return(false);
      const double recurrence = alpha_bar[j - 1] * (1 - beta[j]);
      if(MathAbs(alpha_bar[j] - recurrence) > 1e-12 * MathMax(1.0, MathAbs(recurrence)))
         return(false);
      const double variance = j == 1 ? 0 : beta[j] * (1 - alpha_bar[j - 1]) / (1 - alpha_bar[j]);
      if(MathAbs(sigma[j]*sigma[j] - variance) > 1e-12 * MathMax(1.0, MathAbs(variance)))
         return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Configure                               |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Configure(const SDPCCConfig &config)
  {
   if(m_actor_abi != 2)
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- No architecture/schedule is inferred by the convenience overload.
   if(m_failed || !m_configured || m_busy || m_transfer_pending || !ValidateOwned() ||
      !DPCCValidateFixedConfig(m_shape, config) ||
      DPCCFixedSignature(config, Type()) != DPCCFixedSignature(m_config, Type()))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Configure                               |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Configure(const SDPCCConfig &config, const SDPCCDecoderComposition &decoder,
                               const SDPCCDenoiserComposition &model, const SDPCCDenoiserComposition &training,
                               const double &beta[], const double &alpha_bar[], const double &sigma[])
  {
   if(m_actor_abi != 2)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_failed || !m_shaped || m_configured || m_busy || m_transfer_pending ||
      !ValidateOwned() || !DPCCValidateFixedConfig(m_shape, config) || !ScheduleValid(beta, alpha_bar, sigma))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   SDPCCConfig denoiser_config = config;
   denoiser_config.shape.Depth = 1;
   if(!m_decoder.Configure(config, decoder) || !m_denoiser.Configure(denoiser_config, model, training) ||
      !m_rollout.Configure(config) || !m_projector.Configure(config) || !m_projector.BindRollout(m_rollout))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_config = config;
   m_decoder_composition = decoder;
   m_denoiser_composition = model;
   if(ArrayCopy(m_beta, beta) != ArraySize(beta) || ArrayCopy(m_alpha_bar, alpha_bar) != ArraySize(alpha_bar) ||
      ArrayCopy(m_sigma, sigma) != ArraySize(sigma))
      return(Fail(DPCC_OPENCL_ERROR, true));
   m_configured = true;
   TrainMode(false);
   if(!Clear())
      return(false);
   if(!m_loading)
     {
      const ulong profile = ProfileSignature(config, decoder, model, beta, alpha_bar, sigma);
      if(profile == 0 || (m_admitted && profile != m_admitted_profile))
         return(Fail(DPCC_CONTRACT_MISMATCH, true));
      m_admitted_profile = profile;
      m_admitted = true;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::TrainMode                               |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::TrainMode(bool flag)
  {
   if(m_actor_abi == 3)
     {
      if(m_busy || m_transfer_pending || m_action_unroll_live || m_action_trace_live)
        {
         Fail(DPCC_CONTRACT_MISMATCH);
         return;
        }
      bTrain = flag;
      return;
     }
   bTrain = false;
   if(m_actor_projector)
      m_actor_projector.TrainMode(false);
   CNeuronBaseOCL *children[] = {m_decoder, m_denoiser, m_rollout, m_projector, m_normalizer, m_denorm, m_decoder_source, m_plan_source};
   for(uint i = 0; i < 8; i++)
      if(children[i])
         children[i].TrainMode(false);
  }
//+------------------------------------------------------------------+
//| Updates only an idle ABI3 TargetActor toward the matching Actor  |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(m_actor_abi != 3)
      return(Fail(DPCC_BACKWARD_UNSUPPORTED));
   if(CheckPointer(source) == POINTER_INVALID || source == GetPointer(this) || source.Type() != Type() ||
      !MathIsValidNumber(tau) || tau < 0 || tau > 1)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CNeuronDPCCOCL *actor = source;
   if(actor.m_actor_abi != 3 || bTrain || m_failed || actor.m_failed || m_busy || actor.m_busy ||
      m_transfer_pending || actor.m_transfer_pending || !m_configured || !actor.m_configured ||
      m_action_unroll_live || actor.m_action_unroll_live || m_action_trace_live || actor.m_action_trace_live ||
      m_action_gradient_bound || actor.m_action_gradient_bound ||
      m_action_backward_ready || actor.m_action_backward_ready ||
      m_action_update_started || actor.m_action_update_started || OpenCL != actor.OpenCL ||
      !m_policy_networks_bound || !actor.m_policy_networks_bound ||
      m_policy_market != actor.m_policy_market ||
      !ValidatePolicyNetworks(m_policy_market, m_policy_q1, m_policy_q2) ||
      !actor.ValidatePolicyNetworks(actor.m_policy_market, actor.m_policy_q1, actor.m_policy_q2) ||
      !ValidateAction6() || !actor.ValidateAction6() ||
      !ScheduleValid(m_beta, m_alpha_bar, m_sigma) ||
      !actor.ScheduleValid(actor.m_beta, actor.m_alpha_bar, actor.m_sigma))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong target_contract = ContractSignature();
   const ulong source_contract = actor.ContractSignature();
   if(target_contract == 0 || target_contract != source_contract || !m_admitted || !actor.m_admitted ||
      m_admitted_profile != target_contract || actor.m_admitted_profile != source_contract)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_busy = true;
   const bool updated = m_denoiser.UpdateFrameWeightsFrom(actor.m_denoiser, tau);
   m_busy = false;
   if(!updated)
      return(Fail(m_denoiser.LastStatus(), m_denoiser.FrameUnusable()));
//--- Old plans and their critic publication belong to the previous weights.
   Invalidate();
   ClearPolicyState();
   m_action_context_bound = false;
   m_action_candidates_valid = false;
   m_action_selected = UINT_MAX;
   m_action_market_generation = 0;
   ZeroMemory(m_action_meta);
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Clear                                   |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Clear(void)
  {
   Invalidate();
   if(m_actor_abi == 3)
     {
      if((m_action_unroll_live || m_action_trace_live) && !AbortAction6Unroll())
         return(false);
      if(m_failed || m_busy || m_transfer_pending || !m_configured || !ValidateAction6())
         return(Fail(DPCC_CONTRACT_MISMATCH));
      CBufferFloat *buffers[] = {Output, PrevOutput, Gradient, m_action_context, m_action_condition,
                                 m_action_noise, m_action_current, m_action_next, m_action_physical,
                                 m_action_epsilon, m_action_probability_frames, m_action_projected_frames,
                                 m_action_flags, m_action_scalar, m_action_status,
                                 m_action_candidate_status, m_action_returns, m_action_account13,
                                 m_action_q2_scores, m_action_rank_scores, m_action_numeric_flags
                                };
      for(uint i = 0; i < (uint)ArraySize(buffers); i++)
         if(!buffers[i] || !buffers[i].Fill(0))
           {
            PrintFormat("DPCC_ACTION6_CLEAR_FAIL stage=buffer index=%d", (int)i);
            return(Fail(DPCC_OPENCL_ERROR, true));
           }
      if(!m_plan_source.Clear())
        {
         PrintFormat("DPCC_ACTION6_CLEAR_FAIL stage=plan_source");
         return(Fail(DPCC_OPENCL_ERROR, true));
        }
      if(!m_denoiser.Clear())
        {
         PrintFormat("DPCC_ACTION6_CLEAR_FAIL stage=denoiser status=%d", (int)m_denoiser.LastStatus());
         return(Fail(DPCC_OPENCL_ERROR, true));
        }
      if(!m_rollout.Clear())
        {
         PrintFormat("DPCC_ACTION6_CLEAR_FAIL stage=rollout status=%d", (int)m_rollout.LastStatus());
         return(Fail(DPCC_OPENCL_ERROR, true));
        }
      if(!m_actor_projector.Clear())
        {
         PrintFormat("DPCC_ACTION6_CLEAR_FAIL stage=projector status=%d", (int)m_actor_projector.LastStatus());
         return(Fail(DPCC_OPENCL_ERROR, true));
        }
      ClearPolicyState();
      ZeroMemory(m_action_meta);
      m_action_market_generation = 0;
      m_action_context_bound = false;
      m_action_candidates_valid = false;
      m_action_unroll_live = false;
      m_action_trace_live = false;
      m_action_gradient_bound = false;
      m_action_backward_ready = false;
      m_action_update_started = false;
      m_action_selected = UINT_MAX;
      m_status = DPCC_OK;
      return(true);
     }
   if(m_failed || m_busy || m_transfer_pending || !ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!Output.Fill(0) || !PrevOutput.Fill(0) || !Gradient.Fill(0) ||
      !m_decoder.Clear() || !m_denoiser.Clear() || !m_rollout.Clear() || !m_projector.Clear() ||
      !m_denorm.Clear() || !m_decoder_source.Clear() || !m_plan_source.Clear())
      return(Fail(DPCC_OPENCL_ERROR, true));
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      if(!m_work[i].Fill(0))
         return(Fail(DPCC_OPENCL_ERROR, true));
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SetProjectionMode                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SetProjectionMode(const ENUM_DPCC_MODE mode)
  {
   if(m_failed || !m_configured || m_busy || m_transfer_pending || m_action_unroll_live ||
      m_action_trace_live || mode < DPCC_P0 || mode > DPCC_P2)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   Invalidate();
   m_config.mode = mode;
   m_status = DPCC_OK;
//--- Child projection/rollout numerical contract is mode-independent; their
//--- frozen saved mode need not mutate. Owner records the operational mode.
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SetSnapshotMeta                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SetSnapshotMeta(const SDPCCSnapshotMeta &meta)
  {
   Invalidate();
   if(m_failed || !m_imported || m_busy || m_transfer_pending || meta.decision_key == 0 ||
      meta.decision_time <= 0 || StringLen(meta.symbol) == 0 || StringLen(meta.account_currency) == 0 ||
      meta.timeframe <= 0)
      return(Fail(DPCC_INVALID_INPUT));
   m_snapshot = meta;
   m_snapshot_valid = true;
   m_snapshot_staged = true;
   m_snapshot_device_hash = 0;
   m_status = DPCC_OK;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::DonorDescriptor                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::DonorDescriptor(CNeuronBaseOCL *donor, const uint type, const bool allow_single)
  {
   if(!donor || donor.Type() != (int)type || donor.GetOpenCL() != OpenCL || !donor.CanStartForward())
      return(false);
   CLayerDescription *desc = donor.GetLayerInfo();
   if(!desc)
      return(false);
   const bool result = ArraySize(desc.windows) == 9 && ArraySize(desc.units) == 5 &&
                       desc.windows[0] == 2 && desc.windows[1] == m_shape.K && desc.windows[2] == m_shape.V &&
                       desc.windows[3] == m_shape.H && desc.windows[4] == m_shape.D &&
                       desc.windows[5] == m_shape.M && desc.windows[6] == m_shape.J && desc.windows[7] == m_shape.E &&
                       (desc.windows[8] == m_shape.M || (allow_single && desc.windows[8] == 1)) &&
                       desc.units[0] == m_shape.Embedding && desc.units[1] == m_shape.Heads &&
                       desc.units[2] == m_shape.KeyDimension && desc.units[3] == 6 && desc.units[4] == 13 &&
                       desc.activation == None && desc.optimization == ADAM;
   delete desc;
   return(result);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ReadNormalizer                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReadNormalizer(CNeuronBatchNormOCL *source, float &prepared[])
  {
   ArrayResize(prepared, 0);
   if(!source || source.Type() != defNeuronBatchNormOCL || source.GetOpenCL() != OpenCL ||
      source.Neurons() != 4 || source.Activation() != None || source.Optimization() != ADAM)
      return(false);
   CBufferFloat *options = source.getBatchOptions();
   if(!ValidBuffer(options, 36, OpenCL) || !options.BufferRead() || ArrayResize(prepared, 36) != 36)
      return(false);
   for(uint i = 0; i < 36; i++)
      if(!MathIsValidNumber(options.At(i)))
         return(false);
   for(uint channel = 0; channel < 4; channel++)
     {
      const uint s = channel * 9;
      if(options.At(s + 1) <= 0 || options.At(s + 3) != 1.0f || options.At(s + 4) != 0.0f)
         return(false);
      //--- Fixed normalization values only; no donor moments/normalized input.
      for(uint i = 0; i < 9; i++)
         prepared[s + i] = 0;
      prepared[s] = options.At(s);
      prepared[s + 1] = options.At(s + 1);
      prepared[s + 3] = 1.0f;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::WriteNormalizer                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::WriteNormalizer(const float &prepared[])
  {
   CBufferFloat *options = m_normalizer.getBatchOptions();
   if(ArraySize(prepared) != 36 || !ValidBuffer(options, 36, OpenCL))
      return(false);
   for(uint i = 0; i < 36; i++)
      if(!options.Update((int)i, prepared[i]))
         return(false);
   return(options.BufferWrite() && m_normalizer.SetBatchCount(1));
  }
//+------------------------------------------------------------------+
//| Read the normalization snapshot                                  |
//+------------------------------------------------------------------+
//--- Read owned saved parameters without exposing their mutable buffer or changing decision state.
//+------------------------------------------------------------------+
//| Reads and validates the price normalization snapshot.            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReadNormalizationSnapshot(float &price_norm[])
  {
   ArrayResize(price_norm, 0);
   if(m_actor_abi != 2)
      return(false);
   if(m_failed || !m_configured || !m_imported || m_busy || m_transfer_pending || !ValidateOwned())
      return(false);
   const ulong before = TrainableStateFingerprint(), fixed_before = WeightSignature();
   if(before == 0 || fixed_before == 0)
      return(false);
   float prepared[], snapshot[];
   if(!ReadNormalizer(m_normalizer, prepared) || ArrayResize(snapshot, 16) != 16)
      return(false);
   for(uint channel = 0; channel < 4; channel++)
     {
      const uint source = channel * 9, target = channel * 4;
      const float deviation = (float)MathSqrt((double)prepared[source + 1]);
      if(!MathIsValidNumber(deviation) || deviation <= 0)
         return(false);
      snapshot[target] = prepared[source];
      snapshot[target + 1] = deviation;
      snapshot[target + 2] = prepared[source + 3];
      snapshot[target + 3] = prepared[source + 4];
     }
   const ulong after = TrainableStateFingerprint(), fixed_after = WeightSignature();
   if(after == 0 || fixed_after == 0 || before != after || fixed_before != fixed_after || ArrayCopy(price_norm, snapshot) != 16)
     {
      ArrayResize(price_norm, 0);
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ImportPreparedComponents                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ImportPreparedComponents(CNeuronScenarioDecoderOCL *decoder,
      CNeuronPlanDenoiserOCL *denoiser,
      CNeuronBatchNormOCL *normalizer)
  {
   Invalidate();
   if(m_actor_abi != 2)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(m_failed || !m_configured || m_imported || m_busy || m_transfer_pending || !ValidateOwned() ||
      decoder == m_decoder || denoiser == m_denoiser || normalizer == m_normalizer ||
      !DonorDescriptor(decoder, defNeuronScenarioDecoderOCL, true) ||
      !DonorDescriptor(denoiser, defNeuronPlanDenoiserOCL, true) ||
      m_decoder.UpdateCount() != 0 || m_denoiser.UpdateCount() != 0)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   float prepared[];
   if(!ReadNormalizer(normalizer, prepared))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!m_decoder.CanImportPreparedParameters(decoder) || !m_denoiser.CanImportPreparedParameters(denoiser))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong decoder_before = decoder.TrainableStateFingerprint();
   const ulong denoiser_before = denoiser.TrainableStateFingerprint();
   ulong norm_before = 1469598103934665603;
   if(decoder_before == 0 || denoiser_before == 0 || !normalizer.AppendParameterFingerprint(norm_before))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong decoder_weights = decoder.WeightSignature(), denoiser_weights = denoiser.WeightSignature();
   if(decoder_weights == 0 || denoiser_weights == 0)
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- Existing children validate exact compositions/native optimizer contracts
//--- before their own mutations. Any refusal after the first copy is fatal
//--- for this destination; recreate/reload rather than re-import partial data.
   if(!m_decoder.WeightsUpdate(decoder, 1.0f) || !m_denoiser.WeightsUpdate(denoiser, 1.0f) ||
      !WriteNormalizer(prepared))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(m_decoder.WeightSignature() != decoder_weights || m_denoiser.WeightSignature() != denoiser_weights ||
      m_decoder.UpdateCount() != 0 || m_denoiser.UpdateCount() != 0)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   ulong norm_after = 1469598103934665603;
   if(decoder.TrainableStateFingerprint() != decoder_before || denoiser.TrainableStateFingerprint() != denoiser_before ||
      !normalizer.AppendParameterFingerprint(norm_after) || norm_before != norm_after)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_imported = true;
   TrainMode(false);
   if(!Clear())
      return(Fail(m_status, true));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::AttachOpenCL                            |
//+------------------------------------------------------------------+
void CNeuronDPCCOCL::AttachOpenCL(COpenCLMy *context)
  {
   if(!context || m_shaped || m_busy || m_transfer_pending)
     {
      Fail(DPCC_CONTRACT_MISMATCH);
      return;
     }
   OpenCL = context;
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SupportsOpenCLChecked                   |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SupportsOpenCLChecked(void) const
  {
   if(m_actor_abi == 3)
      return(!m_failed && !m_busy && !m_transfer_pending && !m_action_unroll_live &&
             !m_action_trace_live && ValidateAction6());
   if(Type() != defNeuronDPCCOCL || !m_shaped || m_busy)
      return(false);
   if(m_transfer_pending)
      return(true);
   return(!m_failed && m_decoder.SupportsOpenCLChecked() && m_denoiser.SupportsOpenCLChecked() &&
          m_rollout.SupportsOpenCLChecked() && m_projector.SupportsOpenCLChecked() &&
          m_normalizer.SupportsOpenCLChecked() && m_denorm.SupportsOpenCLChecked() &&
          m_decoder_source.SupportsOpenCLChecked() && m_plan_source.SupportsOpenCLChecked());
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::SetOpenCLChecked                        |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::SetOpenCLChecked(COpenCLMy *target)
  {
   if(m_actor_abi == 3)
      return(target == OpenCL && SupportsOpenCLChecked());
   if(!target || !SupportsOpenCLChecked() || (m_transfer_pending && target != m_transfer_target))
      return(false);
   if(!m_transfer_pending && target == OpenCL)
      return(ValidateOwned());
   Invalidate();
   if(!m_transfer_pending)
     {
      m_transfer_source = OpenCL;
      m_transfer_target = target;
      m_transfer_pending = true;
      for(uint i = 0; i < 8; i++)
         m_transferred[i] = false;
      m_projector.UnbindRollout();
     }
//--- Caller/CNet must retain source and target contexts until this transaction
//--- completes or the owner is destroyed/reloaded. These references do not
//--- authorize mixed-context execution and are not owners of COpenCLMy.
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      if(m_work[i].GetIndex() >= 0 && m_work[i].GetOpenCL() != target && !m_work[i].BufferRead())
         return(Fail(DPCC_OPENCL_ERROR));
   CNeuronBaseOCL *children[] = {m_decoder, m_denoiser, m_rollout, m_projector, m_normalizer, m_denorm, m_decoder_source, m_plan_source};
   for(uint i = 0; i < 8; i++)
      if(!m_transferred[i])
        {
         if(!children[i].SetOpenCLChecked(target))
            return(Fail(DPCC_OPENCL_ERROR));
         m_transferred[i] = true;
        }
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      if((m_work[i].GetOpenCL() != target || m_work[i].GetIndex() < 0) && !m_work[i].BufferCreate(target))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!CNeuronBaseOCL::SetOpenCLChecked(target))
      return(Fail(DPCC_OPENCL_ERROR));
   if(m_configured && !m_projector.BindRollout(m_rollout))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_transfer_pending = false;
   m_transfer_source = NULL;
   m_transfer_target = NULL;
   m_failed = false;
   TrainMode(false);
   return(Clear());
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::GetLayerInfo                            |
//+------------------------------------------------------------------+
CLayerDescription *CNeuronDPCCOCL::GetLayerInfo(void)
  {
   if(!m_shaped || m_failed || m_transfer_pending)
      return(NULL);
   CLayerDescription *description = new CLayerDescription();
   if(!description || (m_actor_abi == 3 ? !DescribeAction6(description) : !Describe(description, Type(), 6, 0)))
     { DeleteObj(description); return(NULL); }
   return(description);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ContractSignature                       |
//+------------------------------------------------------------------+
ulong CNeuronDPCCOCL::ContractSignature(void) const
  {
   if(m_actor_abi == 3)
     {
      if(m_failed || !m_configured || !ValidateAction6())
         return(0);
      ulong hash = DPCCFixedSignature(m_config, Type());
      DPCCRolloutHashWord(3, hash);
      DPCCRolloutHashWord(m_actor_market_horizon, hash);
      DPCCRolloutHashWord(m_actor_latent_dimension, hash);
      for(uint i = 0; i < 109; i++)
        {
         DPCCRolloutHashDouble(m_actor_scale[i], hash);
         DPCCRolloutHashDouble(m_actor_bias[i], hash);
        }
      for(uint j = 0; j <= m_shape.J; j++)
        {
         DPCCRolloutHashDouble(m_beta[j], hash);
         DPCCRolloutHashDouble(m_alpha_bar[j], hash);
         DPCCRolloutHashDouble(m_sigma[j], hash);
        }
      return(hash);
     }
   if(m_failed || !m_configured || !m_decoder || !m_denoiser)
      return(0);
   const ulong decoder_contract = m_decoder.ContractSignature(), denoiser_contract = m_denoiser.ContractSignature();
   if(decoder_contract == 0 || denoiser_contract == 0)
      return(0);
   ulong hash = DPCCFixedSignature(m_config, defNeuronDPCCOCL);
   DPCCRolloutHashWord(decoder_contract, hash);
   DPCCRolloutHashWord(denoiser_contract, hash);
   if(!m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) ||
      m_projector.ContractSignature() != DPCCFixedSignature(m_config, defNeuronPlanProjectionOCL))
      return(0);
   DPCCRolloutHashWord(m_rollout.ContractSignature(), hash);
   DPCCRolloutHashWord(m_projector.ContractSignature(), hash);
   DPCCRolloutHashWord(m_padK, hash);
   DPCCRolloutHashWord(m_tokens, hash);
   for(uint i = 0; i < DPCC_CW_COUNT; i++)
      DPCCRolloutHashWord(m_sizes[i], hash);
   for(uint j = 0; j <= m_shape.J; j++)
     {
      DPCCRolloutHashDouble(m_beta[j], hash);
      DPCCRolloutHashDouble(m_alpha_bar[j], hash);
      DPCCRolloutHashDouble(m_sigma[j], hash);
     }
   return(hash);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::WeightSignature                         |
//+------------------------------------------------------------------+
ulong CNeuronDPCCOCL::WeightSignature(void)
  {
   if(m_actor_abi == 3)
     {
      ulong hash = ContractSignature();
      return(hash != 0 && m_denoiser.AppendFrameTrainingFingerprint(hash) ? hash : 0);
     }
   if(m_failed || m_transfer_pending || !m_configured || !ValidateOwned())
      return(0);
   const ulong first = m_decoder.WeightSignature(), second = m_denoiser.WeightSignature();
   if(first == 0 || second == 0)
      return(0);
   ulong hash = 1469598103934665603;
   DPCCRolloutHashWord(first, hash);
   DPCCRolloutHashWord(second, hash);
//--- BN never executes in the planner. Its fixed options are the imported
//--- parameter contract; per-decision mean/std lives only in owned RevIN/C.
   CBufferFloat *norm = m_normalizer.getBatchOptions();
   if(!norm.BufferRead())
      return(0);
   for(uint c = 0; c < 4; c++)
      for(uint f = 0; f < 5; f++)
         if(f != 2)
            DPCCRolloutHashDouble(norm.At(c * 9 + f), hash);
   return(hash);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::AppendParameterFingerprint              |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(m_actor_abi == 3)
     {
      const ulong contract = ContractSignature();
      if(contract == 0)
         return(false);
      ulong hash = fingerprint;
      DPCCRolloutHashWord(contract, hash);
      if(!m_denoiser.AppendFrameTrainingFingerprint(hash))
         return(false);
      fingerprint = hash;
      return(true);
     }
   if(m_failed || m_transfer_pending || !m_configured || !ValidateOwned())
      return(false);
   return(m_decoder.AppendParameterFingerprint(fingerprint) &&
          m_denoiser.AppendParameterFingerprint(fingerprint) &&
          m_normalizer.AppendParameterFingerprint(fingerprint));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::TrainableStateFingerprint               |
//+------------------------------------------------------------------+
ulong CNeuronDPCCOCL::TrainableStateFingerprint(void)
  {
   ulong hash = 1469598103934665603;
   if(!AppendParameterFingerprint(hash))
      return(0);
   DPCCRolloutHashWord(m_config.randomness_version, hash);
   DPCCRolloutHashWord(m_config.normalization_mode, hash);
   return(hash);
  }
//+------------------------------------------------------------------+
//| Write a typed text field                                         |
//+------------------------------------------------------------------+
//--- The composite's version-3 record has an explicit type/magic/footer, profile
//--- and native child records. Transient buffers/publication are reconstructed,
//--- not serialized as valid results. Native children's own binary formats stay
//--- unchanged. Factory consumes the first Type before calling this Load.
//+------------------------------------------------------------------+
//| Writes a bounded text value to the open file.                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::WriteText(const int file_handle, const string value)
  {
   const int length = StringLen(value);
   return(length > 0 && length <= 128 && FileWriteInteger(file_handle, length, INT_VALUE) == INT_VALUE &&
          FileWriteString(file_handle, value, length) > 0);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ReadText                                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReadText(const int file_handle, string &value)
  {
   value = "";
   if(FileTell(file_handle) + INT_VALUE > FileSize(file_handle))
      return(false);
   const int length = FileReadInteger(file_handle, INT_VALUE);
   if(length <= 0 || length > 128)
      return(false);
   value = FileReadString(file_handle, length);
   return(StringLen(value) == length);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::WriteProfile                            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::WriteProfile(const int f)
  {
   const SDPCCConfig c = m_config;
   const SDPCCDenoiserArchitecture g = m_denoiser_composition.graph;
   const SDPCCParallelExecution p = m_denoiser_composition.parallel;
   const uint integers[] = {c.shape.K, c.shape.V, c.shape.H, c.shape.D, c.shape.M, c.shape.J, c.shape.E,
                            c.shape.PlanBatch, c.shape.NActions, c.shape.StateSize, c.shape.Embedding, c.shape.Heads, c.shape.KeyDimension, c.shape.Depth,
                            c.contract_version, c.binary_layer_version, (uint)c.mode, c.projection_max_iterations, c.projection_max_evaluations,
                            c.projection_lookback_steps, c.price_basis, c.quote_adapter, c.event_model, (uint)c.risk_type,
                            c.normalization_mode, c.loss_coordinates, c.randomness_version
                           };
   if(FileWriteArray(f, integers) != ArraySize(integers))
      return(false);
   if(c.margin_fx_profile != DPCC_MARGIN_EXTERNAL_V2 &&
      (FileWriteInteger(f, (int)c.margin_fx_profile, INT_VALUE) != INT_VALUE ||
       FileWriteInteger(f, (int)c.account_currency_digits, INT_VALUE) != INT_VALUE))
      return(false);
   if(FileWriteInteger(f, (int)m_decoder_composition.embedding, INT_VALUE) != INT_VALUE)
      return(false);
   const uint graph[] = {g.embedding, g.heads, g.key_dimension, g.depth, (uint)g.padding_mask, (uint)g.temporal_mask,
                         p.version, p.training_plan_batch, p.deployment_plan_batch, p.proposal_slots, p.logical_heads, p.key_dimension,
                         (uint)p.shared_projection_weights, p.binary_layer_version
                        };
   if(FileWriteArray(f, graph) != ArraySize(graph))
      return(false);
   const double reals[] = {c.projection_improvement_atol, c.projection_distance_tie_atol, c.volume_scale, c.price_scale,
                           c.money_scale, c.max_quote_age_seconds, c.max_last_age_seconds, (double)c.risk_parameter, (double)c.mean_weight, (double)c.gamma, (double)c.probability_tolerance
                          };
   if(FileWriteArray(f, reals) != ArraySize(reals))
      return(false);
   for(uint i = 0; i < 6; i++)
      if(FileWriteFloat(f, c.action_bias[i]) != sizeof(float) || FileWriteFloat(f, c.action_scale[i]) != sizeof(float))
         return(false);
   if(FileWriteArray(f, m_beta) != ArraySize(m_beta) || FileWriteArray(f, m_alpha_bar) != ArraySize(m_alpha_bar) ||
      FileWriteArray(f, m_sigma) != ArraySize(m_sigma))
      return(false);
   const string texts[] = {m_decoder_composition.architecture, m_decoder_composition.activation_profile,
                           m_decoder_composition.backward_profile, m_decoder_composition.optimizer_profile, m_decoder_composition.graph_digest,
                           g.architecture, g.graph_id, g.graph_sha256, g.normalization, g.feedforward_activation, g.attention_backward,
                           g.backward_profile, m_denoiser_composition.optimizer_profile, p.program, p.self_attention_fold, p.cross_attention_fold,
                           p.activation_lifetime, p.legacy_slice_use, "CAUSAL_AFFINE_V2/REVIN_GAIN1/OHLC4/ADAM9"
                          };
   for(uint i = 0; i < (uint)ArraySize(texts); i++)
      if(!WriteText(f, texts[i]))
         return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ReadProfile                             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReadProfile(const int f,
                                 const uint financial_envelope,
                                 SDPCCConfig &c,
                                 SDPCCDecoderComposition &decoder,
                                 SDPCCDenoiserComposition &denoiser,
                                 double &beta[],
                                 double &schedule_alpha[],
                                 double &sigma[]
                                )
  {
   ZeroMemory(c);
   ZeroMemory(decoder);
   ZeroMemory(denoiser);
   uint n[27], g[14];
   double r[11];
   if(FileReadArray(f, n, 0, 27) != 27 || n[5] == 0 || n[5] > 64)
      return(false);
   c.shape.K = n[0];
   c.shape.V = n[1];
   c.shape.H = n[2];
   c.shape.D = n[3];
   c.shape.M = n[4];
   c.shape.J = n[5];
   c.shape.E = n[6];
   c.shape.PlanBatch = n[7];
   c.shape.NActions = n[8];
   c.shape.StateSize = n[9];
   c.shape.Embedding = n[10];
   c.shape.Heads = n[11];
   c.shape.KeyDimension = n[12];
   c.shape.Depth = n[13];
   c.contract_version = n[14];
   c.binary_layer_version = n[15];
   c.mode = (ENUM_DPCC_MODE)n[16];
   c.projection_max_iterations = n[17];
   c.projection_max_evaluations = n[18];
   c.projection_lookback_steps = n[19];
   c.price_basis = n[20];
   c.quote_adapter = n[21];
   c.event_model = n[22];
   c.risk_type = (int)n[23];
   c.normalization_mode = n[24];
   c.loss_coordinates = n[25];
   c.randomness_version = n[26];
   if(financial_envelope == DPCC_FINANCIAL_ENVELOPE_VERSION ||
      financial_envelope == DPCC_HISTORICAL_FX_ENVELOPE_VERSION)
     {
      if(FileTell(f) + 2 * INT_VALUE > FileSize(f))
         return(false);
      c.margin_fx_profile = (uint)FileReadInteger(f, INT_VALUE);
      c.account_currency_digits = (uint)FileReadInteger(f, INT_VALUE);
      if(c.margin_fx_profile != (financial_envelope == DPCC_HISTORICAL_FX_ENVELOPE_VERSION ?
                                 DPCC_MARGIN_HISTORICAL_CLOSE_V1 : DPCC_MARGIN_FOREX_SPOT_V1))
         return(false);
     }
   else
      if(financial_envelope != DPCC_BINARY_LAYER_VERSION)
         return(false);
   if(FileTell(f) + INT_VALUE > FileSize(f))
      return(false);
   decoder.embedding = (uint)FileReadInteger(f, INT_VALUE);
   if(FileReadArray(f, g, 0, 14) != 14 || g[4] > 1 || g[5] > 1 || g[12] > 1 || FileReadArray(f, r, 0, 11) != 11)
      return(false);
   denoiser.graph.embedding = g[0];
   denoiser.graph.heads = g[1];
   denoiser.graph.key_dimension = g[2];
   denoiser.graph.depth = g[3];
   denoiser.graph.padding_mask = (bool)g[4];
   denoiser.graph.temporal_mask = (bool)g[5];
   denoiser.parallel.version = g[6];
   denoiser.parallel.training_plan_batch = g[7];
   denoiser.parallel.deployment_plan_batch = g[8];
   denoiser.parallel.proposal_slots = g[9];
   denoiser.parallel.logical_heads = g[10];
   denoiser.parallel.key_dimension = g[11];
   denoiser.parallel.shared_projection_weights = (bool)g[12];
   denoiser.parallel.binary_layer_version = g[13];
   c.projection_improvement_atol = r[0];
   c.projection_distance_tie_atol = r[1];
   c.volume_scale = r[2];
   c.price_scale = r[3];
   c.money_scale = r[4];
   c.max_quote_age_seconds = r[5];
   c.max_last_age_seconds = r[6];
   c.risk_parameter = (float)r[7];
   c.mean_weight = (float)r[8];
   c.gamma = (float)r[9];
   c.probability_tolerance = (float)r[10];
   if(FileTell(f) + 12 * sizeof(float) > FileSize(f))
      return(false);
   for(uint i = 0; i < 6; i++)
     {
      c.action_bias[i] = FileReadFloat(f);
      c.action_scale[i] = FileReadFloat(f);
     }
   const int count = (int)c.shape.J + 1;
   if(ArrayResize(beta, count) != count || ArrayResize(schedule_alpha, count) != count || ArrayResize(sigma, count) != count ||
      FileReadArray(f, beta, 0, count) != count || FileReadArray(f, schedule_alpha, 0, count) != count || FileReadArray(f, sigma, 0, count) != count)
      return(false);
   string texts[19];
   for(uint i = 0; i < 19; i++)
      if(!ReadText(f, texts[i]))
         return(false);
   decoder.architecture = texts[0];
   decoder.activation_profile = texts[1];
   decoder.backward_profile = texts[2];
   decoder.optimizer_profile = texts[3];
   decoder.graph_digest = texts[4];
   denoiser.graph.architecture = texts[5];
   denoiser.graph.graph_id = texts[6];
   denoiser.graph.graph_sha256 = texts[7];
   denoiser.graph.normalization = texts[8];
   denoiser.graph.feedforward_activation = texts[9];
   denoiser.graph.attention_backward = texts[10];
   denoiser.graph.backward_profile = texts[11];
   denoiser.optimizer_profile = texts[12];
   denoiser.parallel.program = texts[13];
   denoiser.parallel.self_attention_fold = texts[14];
   denoiser.parallel.cross_attention_fold = texts[15];
   denoiser.parallel.activation_lifetime = texts[16];
   denoiser.parallel.legacy_slice_use = texts[17];
   return(texts[18] == "CAUSAL_AFFINE_V2/REVIN_GAIN1/OHLC4/ADAM9");
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Save                                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Save(const int f)
  {
   if(m_actor_abi == 3)
      return(SaveAction6(f));
   if(f == INVALID_HANDLE || m_failed || !m_configured || !m_imported || m_busy || m_transfer_pending || !ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const ulong weights = WeightSignature(), state = TrainableStateFingerprint(), contract = ContractSignature();
   if(weights == 0 || state == 0 || contract == 0)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const uint header[] = {(uint)Type(), 0x44504343, (m_config.margin_fx_profile == DPCC_MARGIN_HISTORICAL_CLOSE_V1 ? DPCC_HISTORICAL_FX_ENVELOPE_VERSION :
                          (m_config.margin_fx_profile == DPCC_MARGIN_EXTERNAL_V2 ? DPCC_BINARY_LAYER_VERSION : DPCC_FINANCIAL_ENVELOPE_VERSION)), 2, (uint)m_myIndex, iBatch, 1
                         };
   if(FileWriteArray(f, header) != ArraySize(header) || FileWriteLong(f, (long)m_decision_serial) != sizeof(long) ||
      FileWriteInteger(f, (int)m_bar_identity_valid, INT_VALUE) != INT_VALUE ||
      (m_bar_identity_valid ? !WriteText(f, m_bar_symbol) : FileWriteInteger(f, 0, INT_VALUE) != INT_VALUE) ||
      FileWriteInteger(f, m_bar_timeframe, INT_VALUE) != INT_VALUE ||
      FileWriteLong(f, m_bar_time) != sizeof(long) || !WriteProfile(f) || !m_decoder.Save(f) || !m_denoiser.Save(f) || !m_rollout.Save(f) ||
      !m_projector.Save(f) || !m_normalizer.Save(f) || !m_denorm.SaveOwned(f) ||
      FileWriteLong(f, (long)weights) != sizeof(long) || FileWriteLong(f, (long)state) != sizeof(long) ||
      FileWriteLong(f, (long)contract) != sizeof(long) || FileWriteInteger(f, 0x454E4443, INT_VALUE) != INT_VALUE)
      return(Fail(DPCC_INVALID_INPUT));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::Load                                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::Load(const int f)
  {
   Invalidate();
   if(f == INVALID_HANDLE || !OpenCL || m_busy)
      return(Fail(DPCC_INVALID_INPUT, true));
   uint header[6];
   if(FileReadArray(f, header, 0, 6) != 6 || header[0] != 0x44504343 || (header[1] != DPCC_BINARY_LAYER_VERSION && header[1] != DPCC_FINANCIAL_ENVELOPE_VERSION &&
         header[1] != DPCC_HISTORICAL_FX_ENVELOPE_VERSION) || (header[2] != 2 && header[2] != 3) ||
      header[4] == 0 || header[5] != 1 || FileTell(f) + sizeof(long) > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   if(header[2] == 3)
      return(LoadAction6(f, header[1], header[3], header[4]));
   if(m_actor_abi == 3 && m_admitted)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const ulong serial = (ulong)FileReadLong(f);
   if(FileTell(f) + INT_VALUE > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const int bar_valid = FileReadInteger(f, INT_VALUE);
   string bar_symbol;
   if((bar_valid != 0 && bar_valid != 1) ||
      (bar_valid == 1 ? !ReadText(f, bar_symbol) :
       (FileTell(f) + INT_VALUE > FileSize(f) || FileReadInteger(f, INT_VALUE) != 0)) ||
      FileTell(f) + INT_VALUE + sizeof(long) > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const int bar_timeframe = FileReadInteger(f, INT_VALUE);
   const long bar_time = FileReadLong(f);
   if(bar_valid == 1 ? (serial == 0 || StringLen(bar_symbol) == 0 || bar_timeframe <= 0 || bar_time <= 0) :
      (serial != 0 || StringLen(bar_symbol) != 0 || bar_timeframe != 0 || bar_time != 0))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   SDPCCConfig config;
   SDPCCDecoderComposition decoder;
   SDPCCDenoiserComposition denoiser;
   double beta[], schedule_alpha[], sigma[];
   if(!ReadProfile(f, header[1], config, decoder, denoiser, beta, schedule_alpha, sigma))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const ulong candidate_profile = ProfileSignature(config, decoder, denoiser, beta, schedule_alpha, sigma);
   if(candidate_profile == 0 || (m_admitted && candidate_profile != m_admitted_profile))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   ReleaseOwned();
   m_transfer_pending = false;
   m_transfer_source = NULL;
   m_transfer_target = NULL;
   m_failed = false;
   m_shape = config.shape;
   iBatch = header[4];
   m_loading = true;
   CLayerDescription desc;
   if(!Describe(desc, Type(), 6, 0) || !Init(0, header[3], OpenCL, GetPointer(desc)) ||
      !Configure(config, decoder, denoiser, denoiser, beta, schedule_alpha, sigma))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
//--- Configure gives each child the expected admitted contract. Native Load
//--- compares its saved shape/graph and restores moments/t. The shared link
//--- is explicitly revoked until every child record has succeeded.
   m_projector.UnbindRollout();
   if(!LoadInsideLayer(f, m_decoder) ||
      !LoadInsideLayer(f, m_denoiser) ||
      !LoadInsideLayer(f, m_rollout) ||
      !LoadInsideLayer(f, m_projector) ||
      !LoadInsideLayer(f, m_normalizer) ||
      !m_denorm.LoadOwned(f, OpenCL, m_shape.K * m_shape.H * 4) ||
      FileTell(f) + 3 * sizeof(long) + INT_VALUE > FileSize(f))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   const ulong weights = (ulong)FileReadLong(f), state = (ulong)FileReadLong(f), contract = (ulong)FileReadLong(f);
   if(FileReadInteger(f, INT_VALUE) != 0x454E4443 || weights == 0 || state == 0 || contract == 0)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   TrainMode(false);
   if(!m_rollout.MatchesProjectionContract(m_shape, m_config, OpenCL) ||
      m_projector.ContractSignature() != DPCCFixedSignature(m_config, defNeuronPlanProjectionOCL) ||
      !m_projector.BindRollout(m_rollout) || !ValidateOwned() ||
      m_decoder.UpdateCount() != 0 || m_denoiser.UpdateCount() != 0 ||
      WeightSignature() != weights || TrainableStateFingerprint() != state || ContractSignature() != contract)
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   float prepared[];
   if(!ReadNormalizer(m_normalizer, prepared))
      return(Fail(DPCC_CONTRACT_MISMATCH, true));
   m_imported = true;
   if(!Clear())
      return(Fail(m_status, true));
   m_decision_serial = serial;
   m_bar_identity_valid = (bar_valid == 1);
   m_bar_symbol = bar_symbol;
   m_bar_timeframe = bar_timeframe;
   m_bar_time = bar_time;
   m_loading = false;
   m_admitted_profile = candidate_profile;
   m_admitted = true;
   return(true);
  }
//--- Work-only protected CNeuronDPCCOCL declarations/bodies. Root integrates these
//--- into the existing composite owner; no public primitive or model class added.
/*
   bool CompositeArithmeticLayout(SDPCCLayout &c, SDPCCLayout &r, uint &count);
   bool CompositeArithmeticBuffers(CBufferFloat *&buffers[], const ulong &counts[]);
   bool InitializeCompositeNoise(CBufferFloat *normalized);
   bool CompositeReverseArithmetic(CBufferFloat *normalized, CBufferFloat *epsilon,
                                   CBufferFloat *proposal, const uint j);
   bool CompositeActionAffine(CBufferFloat *source, CBufferFloat *destination,
                              const uint direction);
   bool MergeCompositeProjectionMask(CBufferFloat *normalized_corrected);
*/
//+------------------------------------------------------------------+
//| Builds the Action6 composite arithmetic layouts.                 |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeArithmeticLayout(SDPCCLayout &c, SDPCCLayout &r, uint &count)
  {
   uint tokens;
   ulong mh, mh6;
   count = 0;
   if(m_failed || !m_configured || !m_imported || m_transfer_pending || !ValidateOwned() ||
      !DPCCSameShape(m_shape, m_config.shape) || !DPCCCompositeLayouts(m_shape, c, r, tokens) ||
      !DPCCRolloutProduct(m_shape.M, m_shape.H, mh) || !DPCCRolloutProduct(mh, 6, mh6) ||
      mh6 == 0 || mh6 > INT_MAX || mh6 > (ulong)UINT_MAX / sizeof(float) ||
      c.total != m_c_layout.total || r.total != m_r_layout.total)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 11; i++)
      if(c.offsets[i] != m_c_layout.offsets[i] || c.lengths[i] != m_c_layout.lengths[i])
         return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 8; i++)
      if(r.offsets[i] != m_r_layout.offsets[i] || r.lengths[i] != m_r_layout.lengths[i])
         return(Fail(DPCC_CONTRACT_MISMATCH));
   count = (uint)mh6;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeArithmeticBuffers              |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeArithmeticBuffers(CBufferFloat *&buffers[], const ulong &counts[])
  {
   if(!CompositeBuffers(buffers, counts))
      return(false);
   CBufferFloat *scalar = m_work[DPCC_CW_SCALAR];
   if(!ValidBuffer(scalar, 1, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(int i = 0; i < ArraySize(buffers); i++)
     {
      const ulong total = (ulong)buffers[i].Total();
      if(total > (ulong)UINT_MAX / sizeof(float) ||
         OpenCL.GetBufferSize(buffers[i].GetIndex()) != total * sizeof(float) ||
         scalar == buffers[i] || scalar.GetIndex() == buffers[i].GetIndex())
         return(Fail(DPCC_CONTRACT_MISMATCH));
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::InitializeCompositeNoise                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::InitializeCompositeNoise(CBufferFloat *normalized)
  {
   SDPCCLayout c, r;
   uint count;
   if(!CompositeArithmeticLayout(c, r, count))
      return(false);
   CBufferFloat *flags = m_work[DPCC_CW_FLAGS];
   CBufferFloat *buffers[] = {m_work[DPCC_CW_CONTEXT], m_work[DPCC_CW_ROLLOUT_CONTEXT], normalized,
                              m_work[DPCC_CW_STATUS], m_work[DPCC_CW_VALIDITY], flags
                             };
   const ulong counts[] = {c.total, r.total, count, m_shape.M, m_shape.M, count};
   if(!CompositeArithmeticBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.M, m_shape.H, m_shape.J, (uint)c.offsets[10], (uint)r.offsets[7]};
   float reals[];
   return(CompositeDispatch(def_k_DPCCCompositeInitialNoise, buffers, integers, reals, count) &&
          CompositeReduce(flags, m_work[DPCC_CW_SCALAR], count));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeReverseArithmetic              |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeReverseArithmetic(CBufferFloat *normalized, CBufferFloat *epsilon,
      CBufferFloat *proposal, const uint j)
  {
   SDPCCLayout c, r;
   uint count;
   if(!CompositeArithmeticLayout(c, r, count) || j == 0 || j > m_shape.J ||
      ArraySize(m_beta) != (int)m_shape.J + 1 || ArraySize(m_alpha_bar) != (int)m_shape.J + 1 ||
      ArraySize(m_sigma) != (int)m_shape.J + 1)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   const float beta = (float)m_beta[j], schedule_alpha = (float)m_alpha_bar[j], sigma = (float)m_sigma[j];
   if(!MathIsValidNumber(beta) || !MathIsValidNumber(schedule_alpha) || !MathIsValidNumber(sigma) ||
      beta <= 0 || beta >= 1 || schedule_alpha <= 0 || schedule_alpha >= 1 || sigma < 0 || (j == 1 && sigma != 0))
      return(Fail(DPCC_INVALID_INPUT));
   CBufferFloat *flags = m_work[DPCC_CW_FLAGS];
   CBufferFloat *buffers[] = {normalized, epsilon, m_work[DPCC_CW_CONTEXT],
                              m_work[DPCC_CW_ROLLOUT_CONTEXT], proposal, flags
                             };
   const ulong counts[] = {count, count, c.total, r.total, count, count};
   if(!CompositeArithmeticBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.M, m_shape.H, m_shape.J, j, (uint)c.offsets[10], (uint)r.offsets[7]};
   const float reals[] = {beta, schedule_alpha, sigma};
   return(CompositeDispatch(def_k_DPCCCompositeReverse, buffers, integers, reals, count) &&
          CompositeReduce(flags, m_work[DPCC_CW_SCALAR], count));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CompositeActionAffine                   |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CompositeActionAffine(CBufferFloat *source, CBufferFloat *destination,
      const uint direction)
  {
   SDPCCLayout c, r;
   uint count;
   if(!CompositeArithmeticLayout(c, r, count) || direction > 1)
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- Immutable twelve config scalars; PriceNorm[OHLC4] is never consulted.
   float reals[12];
   for(uint channel = 0; channel < 6; channel++)
     {
      reals[channel] = m_config.action_bias[channel];
      reals[channel + 6] = m_config.action_scale[channel];
      if(!MathIsValidNumber(reals[channel]) || !MathIsValidNumber(reals[channel + 6]) ||
         reals[channel + 6] <= 0)
         return(Fail(DPCC_INVALID_INPUT));
     }
   CBufferFloat *flags = m_work[DPCC_CW_FLAGS];
   CBufferFloat *buffers[] = {source, destination, m_work[DPCC_CW_ROLLOUT_CONTEXT], flags};
   const ulong counts[] = {count, count, r.total, count};
   if(!CompositeArithmeticBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.M, m_shape.H, (uint)r.offsets[7], direction};
   return(CompositeDispatch(def_k_DPCCCompositeActionAffine, buffers, integers, reals, count) &&
          CompositeReduce(flags, m_work[DPCC_CW_SCALAR], count));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::MergeCompositeProjectionMask            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::MergeCompositeProjectionMask(CBufferFloat *normalized_corrected)
  {
   SDPCCLayout c, r;
   uint count;
   if(!CompositeArithmeticLayout(c, r, count) || !m_projector || !m_projector.getOutput() ||
      !m_projector.GetCandidateStatus())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *flags = m_work[DPCC_CW_FLAGS];
   CBufferFloat *buffers[] = {m_work[DPCC_CW_ROLLOUT_CONTEXT], m_work[DPCC_CW_STATUS],
                              m_projector.GetCandidateStatus(), normalized_corrected, flags
                             };
   const ulong counts[] = {r.total, m_shape.M, m_shape.M, count, m_shape.M};
   if(!CompositeArithmeticBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.M, m_shape.H, (uint)r.offsets[7]};
   float reals[];
   return(CompositeDispatch(def_k_DPCCCompositeProjectionMask, buffers, integers, reals, m_shape.M) &&
          CompositeReduce(flags, m_work[DPCC_CW_SCALAR], m_shape.M));
  }
//--- Add these exact int64 host fields to existing SDPCCSnapshotMeta.
//--- Interface section 8 requires H open/close times and the previous mark time.
//--- No absolute time is represented in a device float tensor.
/*
   long              bar_open_times[];
   long              bar_close_times[];
   long              previous_mark_time;
*/
//+------------------------------------------------------------------+
//| Validate snapshot time ordering                                  |
//+------------------------------------------------------------------+
//--- Protected CNeuronDPCCOCL method; host admission precedes device writes.
//--- Remaining calendar/Execution relative-time closure is a separate GPU gate.
//+------------------------------------------------------------------+
//| Validates the chronological snapshot timestamps.                 |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidateSnapshotTimes(const SDPCCSnapshotMeta &meta) const
  {
   if(m_shape.H == 0 || ArraySize(meta.bar_open_times) != (int)m_shape.H ||
      ArraySize(meta.bar_close_times) != (int)m_shape.H ||
      meta.decision_time <= 0 || meta.previous_mark_time <= 0 ||
      meta.previous_mark_time > (long)meta.decision_time ||
      meta.bar_open_times[0] != (long)meta.decision_time)
      return(false);
   for(uint h = 0; h < m_shape.H; h++)
     {
      const long opened = meta.bar_open_times[h];
      const long closed = meta.bar_close_times[h];
      if(opened <= 0 || closed <= opened ||
         (h > 0 && opened < meta.bar_close_times[h - 1]))
         return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::BindSnapshotTimes                       |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::BindSnapshotTimes(void)
  {
   if(!ValidateSnapshotTimes(m_snapshot))
      return(Fail(DPCC_INVALID_INPUT));
   CBufferFloat *times = m_work[DPCC_CW_TIME_META];
   if(!ValidBuffer(times, (ulong)m_shape.H * 4, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   long relative[];
   if(ArrayResize(relative, (int)m_shape.H * 2) != (int)m_shape.H * 2)
      return(Fail(DPCC_OPENCL_ERROR));
   for(uint h = 0; h < m_shape.H; h++)
     {
      //--- Positive ordered int64 metadata makes these differences nonoverflowing.
      relative[2 * h] = m_snapshot.bar_close_times[h] - m_snapshot.bar_open_times[h];
      relative[2 * h + 1] = h == 0 ? 0 : m_snapshot.bar_open_times[h] - m_snapshot.bar_close_times[h - 1];
      if(relative[2 * h] <= 0 || relative[2 * h + 1] < 0)
         return(Fail(DPCC_INVALID_INPUT));
     }
//--- CBufferFloat supplies allocation/lifecycle; raw 2H longs occupy 4H floats.
   if(!OpenCL.BufferWrite(times.GetIndex(), relative, 0, 0, m_shape.H * 2))
      return(Fail(DPCC_OPENCL_ERROR));
   CBufferFloat *buffers[] = {m_work[DPCC_CW_CONTEXT], times, m_work[DPCC_CW_FLAGS]};
   const ulong counts[] = {m_c_layout.total, (ulong)m_shape.H * 4, (ulong)m_shape.K * m_shape.H};
   if(!CompositeBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.K, m_shape.H, m_shape.E, (uint)m_c_layout.offsets[7],
                            (uint)m_c_layout.offsets[8], (uint)m_c_layout.offsets[9],
                            uint(m_snapshot.previous_mark_time == m_snapshot.bar_open_times[0])
                           };
   float reals[];
   return(CompositeDispatch(def_k_DPCCCompositeBindTimes, buffers, integers, reals, m_shape.K * m_shape.H) &&
          CompositeReduce(m_work[DPCC_CW_FLAGS], m_work[DPCC_CW_SCALAR], m_shape.K * m_shape.H));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CaptureSnapshotIdentity                 |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CaptureSnapshotIdentity(void)
  {
   if(!OpenCL.SetArgumentBuffer(def_k_DPCCCompositeSnapshotIdentity, 0, m_work[DPCC_CW_CONTEXT].GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_DPCCCompositeSnapshotIdentity, 1, m_work[DPCC_CW_ACCOUNT13].GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_DPCCCompositeSnapshotIdentity, 2, m_work[DPCC_CW_FLAGS].GetIndex()) ||
      !OpenCL.SetArgument(def_k_DPCCCompositeSnapshotIdentity, 3, (uint)m_c_layout.total))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {1};
   ulong value[1];
   if(!OpenCL.Execute(def_k_DPCCCompositeSnapshotIdentity, 1, offset, global) ||
      !OpenCL.BufferRead(m_work[DPCC_CW_FLAGS].GetIndex(), value, 0, 0, 1) || value[0] == 0)
      return(Fail(DPCC_OPENCL_ERROR));
   m_snapshot_device_hash = value[0];
   return(true);
  }
//+------------------------------------------------------------------+
//| Return the snapshot fingerprint                                  |
//+------------------------------------------------------------------+
//--- Replace prior metadata-only owner implementation in the assembled candidate.
//+------------------------------------------------------------------+
//| Computes a fingerprint for the DPCC snapshot state.              |
//+------------------------------------------------------------------+
ulong CNeuronDPCCOCL::SnapshotFingerprint(void)
  {
   if(!m_snapshot_valid || m_failed || m_transfer_pending || m_snapshot_device_hash == 0)
      return(0);
   ulong hash = 1469598103934665603;
   DPCCRolloutHashWord(m_snapshot_device_hash, hash);
   DPCCRolloutHashWord(m_snapshot.decision_key, hash);
   DPCCRolloutHashWord(m_snapshot.base_seed, hash);
   DPCCRolloutHashWord((ulong)m_snapshot.decision_time, hash);
   DPCCRolloutHashWord(m_snapshot.repeat, hash);
   DPCCRolloutHashWord((ulong)m_snapshot.timeframe, hash);
   DPCCRolloutHashWord(m_decision_serial, hash);
   DPCCRolloutHashWord((ulong)m_snapshot.previous_mark_time, hash);
   for(uint h = 0; h < m_shape.H; h++)
     {DPCCRolloutHashWord((ulong)m_snapshot.bar_open_times[h], hash); DPCCRolloutHashWord((ulong)m_snapshot.bar_close_times[h], hash);}
   const string texts[] = {m_snapshot.symbol, m_snapshot.account_currency};
   for(uint text_index = 0; text_index < 2; text_index++)
     {DPCCRolloutHashWord(StringLen(texts[text_index]), hash); for(int i = 0; i < StringLen(texts[text_index]); i++)DPCCRolloutHashWord(StringGetCharacter(texts[text_index], i), hash);}
   return(hash);
  }
//+------------------------------------------------------------------+
//| Update projection diagnostics                                    |
//+------------------------------------------------------------------+
//--- Protected method declared in forward_declarations. No host candidate reads.
//+------------------------------------------------------------------+
//| Updates the projection summary for the selected stage.           |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::UpdateProjectionSummary(const uint stage)
  {
   if(stage > 2 || !ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- For init/finalize the kernel never dereferences projection. Reuse an
//--- existing distinct M*12 input instead of creating another scratch tensor.
   CBufferFloat *projection = (stage == 1 ? m_projector.GetDiagnostics() : m_work[DPCC_CW_SRM_DIAGNOSTICS]);
   if(!projection || (stage == 1 && !ValidBuffer(projection, (ulong)m_shape.M * 12, OpenCL)))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {projection, m_work[DPCC_CW_ROLLOUT_CONTEXT], m_work[DPCC_CW_VALIDITY],
                              m_work[DPCC_CW_DIAGNOSTICS], m_work[DPCC_CW_FLAGS]
                             };
//--- The stage0/2 unused placeholder has its actual native M*6 shape.
   const ulong counts[] = {(ulong)m_shape.M*(stage == 1 ? 12 : 6), m_r_layout.total, m_shape.M,
                           (ulong)m_shape.M * 12, m_shape.M
                          };
   if(!CompositeBuffers(buffers, counts))
      return(false);
   const uint integers[] = {m_shape.M, (uint)m_r_layout.offsets[7], stage};
   float reals[];
   return(CompositeDispatch(def_k_DPCCCompositeSummary, buffers, integers, reals, m_shape.M) &&
          CompositeReduce(m_work[DPCC_CW_FLAGS], m_work[DPCC_CW_SCALAR], m_shape.M));
  }
//+------------------------------------------------------------------+
//| Pad and validate quantile distributions                          |
//+------------------------------------------------------------------+
//--- Protected existing owner methods only. No CNeuronACSRM or public primitive.
//+------------------------------------------------------------------+
//| Pads and validates the rollout distributions.                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PadAndValidateDistributions(void)
  {
   if(m_failed || !m_configured || !ValidateOwned() || !m_rollout.GetValidity() ||
      !ValidBuffer(m_rollout.GetValidity(), m_shape.M, OpenCL))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_work[DPCC_CW_RETURNS], m_work[DPCC_CW_CONTEXT],
                              m_work[DPCC_CW_ROLLOUT_CONTEXT], m_work[DPCC_CW_STATUS], m_rollout.GetValidity(),
                              m_work[DPCC_CW_VALIDITY], m_work[DPCC_CW_SUPPORT], m_work[DPCC_CW_MASS], m_work[DPCC_CW_FLAGS]
                             };
   const ulong counts[] = {(ulong)m_shape.M * m_shape.K, m_c_layout.total, m_r_layout.total, m_shape.M, m_shape.M,
                           m_shape.M, (ulong)m_shape.M * m_padK, (ulong)m_shape.M * m_padK, m_sizes[DPCC_CW_FLAGS]
                          };
   const uint integers[] = {m_shape.M, m_shape.K, m_padK, (uint)m_c_layout.offsets[2], (uint)m_r_layout.offsets[7]};
   const float reals[] = {m_config.probability_tolerance};
   if(!CompositeBuffers(buffers, counts) || !CompositeDispatch(def_k_DPCCCompositePad, buffers, integers, reals, m_shape.M) ||
      !CompositeReduce(m_work[DPCC_CW_FLAGS], m_work[DPCC_CW_SCALAR], m_shape.M))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::RankExternalDistributions               |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::RankExternalDistributions(void)
  {
   if(m_failed || !ValidateOwned() || m_padK == 0 || m_padK > 64 || (m_padK & (m_padK - 1)) != 0 ||
      OpenCL.GetMaxWorkGroupSize() < (long)m_padK || OpenCL.GetMaxLocalSize(0) < m_padK ||
      OpenCL.GetMaxLocalSize(1) < 1 || !DPCCValidateFixedConfig(m_shape, m_config))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   long local_memory = 0;
   if(!OpenCL.GetLocalMemorySize(local_memory) || local_memory < 1024)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_work[DPCC_CW_SUPPORT], m_work[DPCC_CW_MASS], m_work[DPCC_CW_SORTED_SUPPORT],
                              m_work[DPCC_CW_SORTED_MASS], m_work[DPCC_CW_ORDER], m_work[DPCC_CW_SRM_DIAGNOSTICS]
                             };
   const ulong counts[] = {(ulong)m_shape.M * m_padK, (ulong)m_shape.M * m_padK, (ulong)m_shape.M * m_padK,
                           (ulong)m_shape.M * m_padK, (ulong)m_shape.M * m_padK, (ulong)m_shape.M * defFQF_DIAGNOSTIC_FLOATS
                          };
   if(!CompositeBuffers(buffers, counts, false))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 6; i++)
      if(!OpenCL.SetArgumentBuffer(def_k_FQF_MonotoneRearrange, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0, 0}, global[] = {m_padK, m_shape.M}, local[] = {m_padK, 1};
//--- Native kernels statically declare local arrays, no dynamic local args:
//--- rearrangement 4*64*4=1024 bytes, SRM 64*4=256 bytes.
   if(!OpenCL.Execute(def_k_FQF_MonotoneRearrange, 2, offset, global, local) ||
      !OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 0, m_work[DPCC_CW_SORTED_SUPPORT].GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 1, m_work[DPCC_CW_SORTED_MASS].GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 2, m_work[DPCC_CW_SCORES].GetIndex()) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 3, m_config.risk_type) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 4, m_config.risk_parameter) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 5, m_config.mean_weight) ||
      !OpenCL.Execute(def_k_FQF_SRMForward, 2, offset, global, local))
      return(Fail(DPCC_OPENCL_ERROR));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::PublishDecision                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PublishDecision(void)
  {
   m_public_valid = false;
   if(m_failed || !ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {m_plan_source.getOutput(), m_work[DPCC_CW_VALIDITY], m_work[DPCC_CW_SCORES],
                              m_work[DPCC_CW_SORTED_SUPPORT], m_work[DPCC_CW_SORTED_MASS], m_work[DPCC_CW_ORDER],
                              m_work[DPCC_CW_SRM_DIAGNOSTICS], m_work[DPCC_CW_SELECTED], Output, m_work[DPCC_CW_FLAGS]
                             };
   const ulong counts[] = {(ulong)m_shape.M*m_shape.H * 6, m_shape.M, m_shape.M, (ulong)m_shape.M * m_padK,
                           (ulong)m_shape.M * m_padK, (ulong)m_shape.M * m_padK, (ulong)m_shape.M * defFQF_DIAGNOSTIC_FLOATS,
                           (ulong)m_shape.H * 6, 6, m_sizes[DPCC_CW_FLAGS]
                          };
   if(!CompositeBuffers(buffers, counts, false))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   for(uint i = 0; i < 10; i++)
      if(!OpenCL.SetArgumentBuffer(def_k_DPCCCompositePublish, i, buffers[i].GetIndex()))
         return(Fail(DPCC_OPENCL_ERROR));
   if(!OpenCL.SetArgument(def_k_DPCCCompositePublish, 10, m_shape.M) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 11, m_shape.H) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 12, m_padK) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 13, m_config.risk_type) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 14, m_config.risk_parameter) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 15, m_config.mean_weight) ||
      !OpenCL.SetArgument(def_k_DPCCCompositePublish, 16, m_config.probability_tolerance))
      return(Fail(DPCC_OPENCL_ERROR));
   const uint offset[] = {0}, global[] = {1};
   if(!OpenCL.Execute(def_k_DPCCCompositePublish, 1, offset, global))
      return(Fail(DPCC_OPENCL_ERROR));
   float receipt[4];
   if(!OpenCL.BufferRead(m_work[DPCC_CW_FLAGS].GetIndex(), receipt, 0, 0, 4))
      return(Fail(DPCC_OPENCL_ERROR));
   if(!MathIsValidNumber(receipt[0]) || (receipt[0] != DPCC_OK && receipt[0] != DPCC_NO_FEASIBLE_CANDIDATE))
      return(Fail(DPCC_NUMERIC_ERROR));
   m_status = (ENUM_DPCC_STATUS)(int)receipt[0];
//--- Computed business refusal exposes diagnostics only, never a selected plan.
   m_public_valid = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Validate snapshot metadata                                       |
//+------------------------------------------------------------------+
//--- Work-only concrete forward. Replace the two old unavailable-forward bodies.
//--- Required metadata ABI is deliberate, not a missing-fields fallback.
//+------------------------------------------------------------------+
//| Validates the DPCC snapshot identity and timestamps.             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ValidateSnapshotMeta(const SDPCCSnapshotMeta &meta) const
  {
   return(meta.decision_key != 0 && StringLen(meta.symbol) > 0 && StringLen(meta.account_currency) > 0 &&
          meta.timeframe > 0 && ValidateSnapshotTimes(meta));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::CopyFinitePrefix                        |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::CopyFinitePrefix(CBufferFloat *source, CBufferFloat *destination, const uint count)
  {
//--- Existing CopyBufferRaw requires exact equal lengths. Existing activation
//--- None safely copies/checks the first count elements of immutable C.Z.
   if(!source || !destination || count == 0 || source == destination ||
      source.GetOpenCL() != OpenCL || destination.GetOpenCL() != OpenCL ||
      source.GetIndex() < 0 || source.GetIndex() == destination.GetIndex() ||
      source.Total() < (int)count || !ValidBuffer(destination, count, OpenCL) ||
      (ulong)source.Total() > (ulong)UINT_MAX / sizeof(float) ||
      OpenCL.GetBufferSize(source.GetIndex()) != (ulong)source.Total()*sizeof(float))
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *flags = m_work[DPCC_CW_FLAGS], *scalar = m_work[DPCC_CW_SCALAR];
   if(source == flags || destination == flags || source == scalar || destination == scalar ||
      source.GetIndex() == flags.GetIndex() || destination.GetIndex() == flags.GetIndex() ||
      source.GetIndex() == scalar.GetIndex() || destination.GetIndex() == scalar.GetIndex())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   CBufferFloat *buffers[] = {source, destination, flags};
   const uint integers[] = {uint(None), count};
   float reals[];
   return(CompositeDispatch(def_k_DPCCActivationForward, buffers, integers, reals, count) &&
          CompositeReduce(flags, scalar, count));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::PrepareDecision                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::PrepareDecision(CNeuronBaseOCL *source, CBufferFloat *context)
  {
   if(CheckPointer(source) == POINTER_INVALID || CheckPointer(context) == POINTER_INVALID ||
      source == GetPointer(this) || !source.CanStartForward() ||
      source.GetOpenCL() != OpenCL || source.Neurons() != 13 || !source.getOutput() ||
      !ValidBuffer(source.getOutput(), 13, OpenCL) || !ValidBuffer(context, m_c_layout.total, OpenCL) ||
      !m_snapshot_valid || !m_snapshot_staged || !ValidateSnapshotMeta(m_snapshot) || m_c_layout.offsets[0] != 0)
      return(Fail(DPCC_INVALID_INPUT));
   CBufferFloat *inputs[] = {source.getOutput(), context};
   if(inputs[0] == inputs[1] || inputs[0].GetIndex() == inputs[1].GetIndex())
      return(Fail(DPCC_CONTRACT_MISMATCH));
//--- Reject every owned child allocation before snapshots, time writes, or Clear.
   for(uint a = 0; a < 2; a++)
      if(OwnsBufferOrDeviceSlot(inputs[a]))
         return(Fail(DPCC_CONTRACT_MISMATCH));
//--- One immutable Account13 snapshot and one full C snapshot per decision.
   m_snapshot_staged = false;
   m_snapshot_device_hash = 0;
   if(!SnapshotCompositeContext(inputs[0], m_work[DPCC_CW_ACCOUNT13], m_work[DPCC_CW_FLAGS],
                                m_work[DPCC_CW_SCALAR], 13) ||
      !SnapshotCompositeContext(context, m_work[DPCC_CW_CONTEXT], m_work[DPCC_CW_FLAGS],
                                m_work[DPCC_CW_SCALAR], (uint)m_c_layout.total) || !BindSnapshotTimes())
      return(false);
   if(!m_decoder.Clear() || !m_denoiser.Clear() || !m_rollout.Clear() || !m_projector.Clear() ||
      !m_denorm.Clear() || !m_decoder_source.Clear() || !m_plan_source.Clear())
      return(Fail(DPCC_OPENCL_ERROR));
   if(!CopyFinitePrefix(m_work[DPCC_CW_CONTEXT], m_decoder_source.getOutput(), (uint)m_c_layout.lengths[0]))
      return(false);
   if(!m_decoder.FeedForward(m_decoder_source))
      return(Fail(m_decoder.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_decoder.LastStatus()));
   if(!PrepareCompositeNorm(m_work[DPCC_CW_CONTEXT], m_decoder, m_denorm, m_work[DPCC_CW_FLAGS],
                            m_work[DPCC_CW_SCALAR], m_c_layout, m_shape) ||
      !PackCompositeR(m_work[DPCC_CW_CONTEXT], m_denorm.getOutput(), m_work[DPCC_CW_ROLLOUT_CONTEXT],
                      m_work[DPCC_CW_FLAGS], m_work[DPCC_CW_SCALAR], m_c_layout, m_r_layout, m_config) ||
      !PackCompositeCondition(m_work[DPCC_CW_CONTEXT], m_work[DPCC_CW_ACCOUNT13], m_work[DPCC_CW_CONDITION],
                              m_work[DPCC_CW_FLAGS], m_work[DPCC_CW_SCALAR], m_c_layout, m_config))
      return(false);
   const bool same_bar = m_bar_identity_valid && m_bar_symbol == m_snapshot.symbol &&
                         m_bar_timeframe == m_snapshot.timeframe && m_bar_time == (long)m_snapshot.decision_time;
   if(!same_bar && m_decision_serial == ULONG_MAX)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   if(!CaptureSnapshotIdentity())
      return(false);
//--- Commit once per admitted real bar, not per repeat/key/seed or reverse step.
//--- Clear and failure revoke snapshots/caches but preserve this durable identity.
   if(!same_bar)
      m_decision_serial++;
   m_bar_symbol = m_snapshot.symbol;
   m_bar_timeframe = m_snapshot.timeframe;
   m_bar_time = (long)m_snapshot.decision_time;
   m_bar_identity_valid = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::InitializeNoise                         |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::InitializeNoise(void)
  {
//--- Canonical preloaded C.Noise[m,J] only. Host indexed-noise helper owns
//--- SHA/key generation; there is no second device random stream or readback.
   return(InitializeCompositeNoise(m_plan_source.getOutput()) && UpdateProjectionSummary(0));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ReverseStep                             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ReverseStep(const uint j)
  {
   if(!SetCompositeStep(j, m_shape.J, m_work[DPCC_CW_CONDITION], m_tokens))
      return(false);
   if(!m_denoiser.SetContextIdentity(m_snapshot.decision_key, m_decision_serial,
                                     (ulong)m_work[DPCC_CW_CONTEXT].GetIndex() + 1))
      return(Fail(m_denoiser.LastStatus()));
//--- Exactly one real shared, frozen, L=M denoiser invocation for this j.
   if(!m_denoiser.FeedForward(m_plan_source, m_work[DPCC_CW_CONDITION]))
      return(Fail(m_denoiser.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_denoiser.LastStatus()));
   return(CompositeReverseArithmetic(m_plan_source.getOutput(), m_denoiser.getOutput(), m_work[DPCC_CW_NEXT_PLAN], j));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::ProjectCurrent                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::ProjectCurrent(void)
  {
//--- Current source is physical MH6 on entry, borrowed by the real projector.
   if(!m_projector.FeedForward(m_plan_source, m_work[DPCC_CW_ROLLOUT_CONTEXT]))
      return(Fail(m_projector.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_projector.LastStatus()));
   return(UpdateProjectionSummary(1));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::MergeProjectionMask                     |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::MergeProjectionMask(void)
  {
   if(!CompositeActionAffine(m_projector.getOutput(), m_work[DPCC_CW_NEXT_PLAN], 1))
      return(false);
   return(MergeCompositeProjectionMask(m_work[DPCC_CW_NEXT_PLAN]));
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::FinalRollout                            |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::FinalRollout(void)
  {
   const uint count = m_shape.M * m_shape.H * 6;
   if(!CompositeActionAffine(m_plan_source.getOutput(), m_work[DPCC_CW_NEXT_PLAN], 0) ||
      !CopyBufferRaw(m_work[DPCC_CW_NEXT_PLAN], m_plan_source.getOutput(), count))
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
   if(!m_rollout.FeedForward(m_plan_source, m_work[DPCC_CW_ROLLOUT_CONTEXT]))
      return(Fail(m_rollout.LastStatus() == DPCC_OK ? DPCC_OPENCL_ERROR : m_rollout.LastStatus()));
//--- Preserve actual financial returns before native FQF reads owner buffers.
   if(!CopyBufferRaw(m_rollout.getOutput(), m_work[DPCC_CW_RETURNS], m_shape.M * m_shape.K) ||
      !SnapshotCompositeContext(m_rollout.GetValidity(), m_work[DPCC_CW_VALIDITY], m_work[DPCC_CW_FLAGS],
                                m_work[DPCC_CW_SCALAR], m_shape.M))
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronDPCCOCL::feedForward                             |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::feedForward(CNeuronBaseOCL *source) {return(Fail(DPCC_CONTRACT_MISMATCH));}
//+------------------------------------------------------------------+
//| Generate a DPCC decision from the current context                |
//+------------------------------------------------------------------+
bool CNeuronDPCCOCL::feedForward(CNeuronBaseOCL *source, CBufferFloat *context)
  {
   if(m_actor_abi == 3)
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_public_valid = false;
   if(m_failed || m_busy || m_transfer_pending || !m_configured || !m_imported || !ValidateOwned())
      return(Fail(DPCC_CONTRACT_MISMATCH));
   m_busy = true;
   m_status = DPCC_OK;
   bool success = PrepareDecision(source, context) && InitializeNoise();
   const uint count = m_shape.M * m_shape.H * 6;
   for(uint j = m_shape.J; success && j > 0; j--)
     {
      success = ReverseStep(j);
      if(success && m_config.mode == DPCC_P2)
         success = CompositeActionAffine(m_work[DPCC_CW_NEXT_PLAN], m_plan_source.getOutput(), 0) &&
                   ProjectCurrent() && MergeProjectionMask();
      if(success)
         success = CopyBufferRaw(m_work[DPCC_CW_NEXT_PLAN], m_plan_source.getOutput(), count);
     }
   if(success && m_config.mode == DPCC_P1)
      success = CompositeActionAffine(m_plan_source.getOutput(), m_work[DPCC_CW_NEXT_PLAN], 0) &&
                CopyBufferRaw(m_work[DPCC_CW_NEXT_PLAN], m_plan_source.getOutput(), count) &&
                ProjectCurrent() && MergeProjectionMask() &&
                CopyBufferRaw(m_work[DPCC_CW_NEXT_PLAN], m_plan_source.getOutput(), count);
//--- P0 has no projection, P1 exactly one final projection, P2 exactly J.
   if(success)
      success = FinalRollout() && PadAndValidateDistributions() && UpdateProjectionSummary(2) &&
                RankExternalDistributions() && PublishDecision();
   m_busy = false;
   if(!success)
     {
      m_snapshot_valid = false;
      return(Fail(m_status == DPCC_OK ? DPCC_OPENCL_ERROR : m_status));
     }
//--- CandidatePlans remains physical MH6; only exact OK yields a chosen plan.
   return(true);
  }
//+------------------------------------------------------------------+
//| DPCC Actor device loss.                                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::Difference(CBufferFloat *left, CBufferFloat *right,
                                     CBufferFloat *output, const uint count, string &reason)
  {
   COpenCLMy *context = OpenCL;
   if(CheckPointer(context) == POINTER_INVALID || CheckPointer(left) == POINTER_INVALID ||
      CheckPointer(right) == POINTER_INVALID || CheckPointer(output) == POINTER_INVALID || count == 0 ||
      left.GetOpenCL() != context || right.GetOpenCL() != context || output.GetOpenCL() != context ||
      left.GetIndex() < 0 || right.GetIndex() < 0 || output.GetIndex() < 0 || count > (uint)INT_MAX ||
      left.Total() < (int)count || right.Total() < (int)count || output.Total() < (int)count)
     {
      reason = "policy.actor_gradient.difference_buffers";
      return(false);
     }
   const int buffers[] = {left.GetIndex(), right.GetIndex(), output.GetIndex()};
   for(int i = 0; i < 3; i++)
      if(!context.SetArgumentBuffer(def_k_MatrixDif, i, buffers[i]))
        {
         reason = StringFormat("policy.actor_gradient.difference_buffer.%u", i);
         return(false);
        }
   const int zero = 0;
   if(!context.SetArgument(def_k_MatrixDif, def_k_dif_multiplyer, 1.0f) ||
      !context.SetArgument(def_k_MatrixDif, def_k_dif_shift_in1, zero) ||
      !context.SetArgument(def_k_MatrixDif, def_k_dif_shift_in2, zero) ||
      !context.SetArgument(def_k_MatrixDif, def_k_dif_shift_out, zero))
     {
      reason = "policy.actor_gradient.difference_arguments";
      return(false);
     }
   uint offset[2] = {0, 0};
   uint global[2] = {1, count};
   if(!context.Execute(def_k_MatrixDif, 2, offset, global))
     {
      reason = "policy.actor_gradient.difference_execute";
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| DPCC Actor device loss.                                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::Gate(CBufferFloat *input1, CBufferFloat *input2,
                               CBufferFloat *gate, CBufferFloat *output, const uint count, string &reason)
  {
   COpenCLMy *context = OpenCL;
   if(CheckPointer(context) == POINTER_INVALID || CheckPointer(input1) == POINTER_INVALID ||
      CheckPointer(input2) == POINTER_INVALID || CheckPointer(gate) == POINTER_INVALID ||
      CheckPointer(output) == POINTER_INVALID || count == 0 ||
      input1.GetOpenCL() != context || input2.GetOpenCL() != context || gate.GetOpenCL() != context ||
      output.GetOpenCL() != context || input1.GetIndex() < 0 || input2.GetIndex() < 0 ||
      gate.GetIndex() < 0 || output.GetIndex() < 0 || count > (uint)INT_MAX ||
      input1.Total() < (int)count || input2.Total() < (int)count ||
      gate.Total() < (int)count || output.Total() < (int)count)
     {
      reason = "policy.actor_gradient.gate_buffers";
      return(false);
     }
   const int buffers[] = {input1.GetIndex(), input2.GetIndex(), gate.GetIndex(), output.GetIndex(), output.GetIndex()};
   for(int i = 0; i < 5; i++)
      if(!context.SetArgumentBuffer(def_k_GateElementMul, i, buffers[i]))
        {
         reason = StringFormat("policy.actor_gradient.gate_buffer.%u", i);
         return(false);
        }
   if(!context.SetArgument(def_k_GateElementMul, def_k_gem_flags_enabled, 0))
     {
      reason = "policy.actor_gradient.gate_flags";
      return(false);
     }
   uint offset[1] = {0};
   uint global[1] = {count};
   if(!context.Execute(def_k_GateElementMul, 1, offset, global))
     {
      reason = "policy.actor_gradient.gate_execute";
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| DPCC Actor device loss.                                          |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::PrepareOracleGradient(const uint horizon, const uint actions,
      const float &teacher_plan[], const float &valid_steps[],
      const float gamma, string &reason)
  {
   reason = "";
   if(Activation() != SIGMOID || horizon == 0 || actions != 6 ||
      (ulong)horizon * (ulong)actions > (ulong)INT_MAX)
     {
      reason = "policy.oracle.layer_or_axes";
      return(false);
     }
   const uint plan_count = horizon * actions;
   COpenCLMy *context = OpenCL;
   CBufferFloat *destination = Gradient;
   CBufferFloat *actor_values = Output;
   if(CheckPointer(context) == POINTER_INVALID || CheckPointer(destination) == POINTER_INVALID ||
      CheckPointer(actor_values) == POINTER_INVALID || destination.GetOpenCL() != context ||
      actor_values.GetOpenCL() != context || destination.GetIndex() < 0 || actor_values.GetIndex() < 0 ||
      destination.Total() != (int)plan_count || actor_values.Total() != (int)plan_count)
     {
      reason = "policy.oracle.gradient_buffers";
      return(false);
     }
   if(!destination.Fill(0.0f))
     {
      reason = "policy.oracle.gradient_clear";
      return(false);
     }
   if(ArraySize(teacher_plan) != (int)plan_count || ArraySize(valid_steps) != (int)horizon ||
      !MathIsValidNumber(gamma) || gamma < 0.0f || gamma > 1.0f)
     {
      reason = "policy.oracle.target_shape_or_gamma";
      return(false);
     }
   for(uint i = 0; i < plan_count; i++)
      if(!MathIsValidNumber(teacher_plan[i]) || teacher_plan[i] < 0.0f || teacher_plan[i] > 1.0f)
        {
         reason = StringFormat("policy.oracle.target_value.%u", i);
         return(false);
        }
   double discounts[];
   if(ArrayResize(discounts, (int)horizon) != (int)horizon)
     {
      reason = "policy.oracle.discount_capacity";
      return(false);
     }
   double discount = 1.0;
   double normalization = 0.0;
   for(uint h = 0; h < horizon; h++)
     {
      if(valid_steps[h] != 0.0f && valid_steps[h] != 1.0f)
        {
         reason = StringFormat("policy.oracle.valid_step.%u", h);
         return(false);
        }
      discounts[h] = discount;
      normalization += (double)valid_steps[h] * discount;
      discount *= (double)gamma;
     }
   if(!MathIsValidNumber(normalization) || normalization <= 0.0)
     {
      reason = "policy.oracle.no_discounted_valid_steps";
      return(false);
     }
   if(!actor_values.BufferRead())
     {
      reason = "policy.oracle.actor_output_read";
      return(false);
     }
   for(uint i = 0; i < plan_count; i++)
     {
      const float action = actor_values.At((int)i);
      if(!MathIsValidNumber(action) || action < 0.0f || action > 1.0f)
        {
         reason = StringFormat("policy.oracle.actor_output.%u", i);
         return(false);
        }
     }
   const int plan_size = (int)plan_count;
   float weight_values[], one_values[], zero_values[];
   if(ArrayResize(weight_values, plan_size) != plan_size || ArrayResize(one_values, plan_size) != plan_size ||
      ArrayResize(zero_values, plan_size) != plan_size)
     {
      reason = "policy.oracle.scratch_capacity";
      return(false);
     }
   for(int i = 0; i < plan_size; i++)
     {
      const uint plan_step = (uint)i / actions;
      weight_values[i] = (float)((double)valid_steps[plan_step] * discounts[plan_step] / normalization);
      one_values[i] = 1.0f;
      zero_values[i] = 0.0f;
     }
   CBufferFloat target_buffer, weight_buffer, one_buffer, zero_buffer;
   CBufferFloat error_buffer, weighted_buffer, one_minus_buffer, factor_buffer;
   if(!target_buffer.AssignArray(teacher_plan) || !weight_buffer.AssignArray(weight_values) ||
      !one_buffer.AssignArray(one_values) || !zero_buffer.AssignArray(zero_values) ||
      !error_buffer.BufferInit(plan_count, 0.0f) ||
      !weighted_buffer.BufferInit(plan_count, 0.0f) ||
      !one_minus_buffer.BufferInit(plan_count, 0.0f) ||
      !factor_buffer.BufferInit(plan_count, 0.0f))
     {
      reason = "policy.oracle.scratch_init";
      return(false);
     }
   CBufferFloat *scratch[] = {GetPointer(target_buffer), GetPointer(weight_buffer), GetPointer(one_buffer),
                              GetPointer(zero_buffer), GetPointer(error_buffer), GetPointer(weighted_buffer),
                              GetPointer(one_minus_buffer), GetPointer(factor_buffer)
                             };
   for(int i = 0; i < 8; i++)
      if(!scratch[i].BufferCreate(context) || !scratch[i].BufferWrite())
        {
         reason = StringFormat("policy.oracle.scratch_opencl.%d", i);
         return(false);
        }
   if(!Difference(GetPointer(target_buffer), actor_values,
                  GetPointer(error_buffer), plan_count, reason) ||
      !Gate(GetPointer(error_buffer), GetPointer(zero_buffer),
            GetPointer(weight_buffer), GetPointer(weighted_buffer), plan_count, reason) ||
      !Difference(GetPointer(one_buffer), actor_values,
                  GetPointer(one_minus_buffer), plan_count, reason) ||
      !Gate(actor_values, GetPointer(zero_buffer),
            GetPointer(one_minus_buffer), GetPointer(factor_buffer), plan_count, reason) ||
      !Gate(GetPointer(weighted_buffer), GetPointer(zero_buffer),
            GetPointer(factor_buffer), destination, plan_count, reason))
     {
      destination.Fill(0.0f);
      if(reason == "")
         reason = "policy.oracle.gradient_dispatch";
      return(false);
     }
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates convolution owners for inherited checked transfer.     |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::SupportsOpenCLChecked(void) const
  {
   if(CheckPointer(WeightsConv) == POINTER_INVALID)
      return(false);
   if(optimization == SGD)
      return(CheckPointer(DeltaWeightsConv) != POINTER_INVALID);
   return(CheckPointer(FirstMomentumConv) != POINTER_INVALID &&
          CheckPointer(SecondMomentumConv) != POINTER_INVALID);
  }
//+------------------------------------------------------------------+
//| Validates the exact Adam owner layout for Actor continuation.    |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::ValidateTrainingOwners(void) const
  {
   if(optimization != ADAM || iBatch == 0 || CheckPointer(WeightsConv) == POINTER_INVALID ||
      WeightsConv.Total() <= 0 || CheckPointer(FirstMomentumConv) == POINTER_INVALID ||
      CheckPointer(SecondMomentumConv) == POINTER_INVALID ||
      FirstMomentumConv.Total() != WeightsConv.Total() || SecondMomentumConv.Total() != WeightsConv.Total() ||
      (CheckPointer(DeltaWeightsConv) != POINTER_INVALID && DeltaWeightsConv.Total() > 0) ||
      (CheckPointer(DeltaWeights) != POINTER_INVALID && DeltaWeights.Total() > 0))
      return(false);
   const int weights = (CheckPointer(Weights) == POINTER_INVALID ? -1 : Weights.Total());
   const int first = (CheckPointer(FirstMomentum) == POINTER_INVALID ? -1 : FirstMomentum.Total());
   const int second = (CheckPointer(SecondMomentum) == POINTER_INVALID ? -1 : SecondMomentum.Total());
   return(weights >= -1 && first >= -1 && second >= -1 &&
          (weights > 0 ? first == weights && second == weights : first <= 0 && second <= 0));
  }
//+------------------------------------------------------------------+
//| Writes Actor identity, parameters and optional Adam              |
//| continuation.                                                    |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::SaveActorState(const int handle, const bool training)
  {
   if(handle == INVALID_HANDLE || CheckPointer(WeightsConv) == POINTER_INVALID || WeightsConv.Total() <= 0 ||
      (training && !ValidateTrainingOwners()))
      return(false);
   if(FileWriteInteger(handle, Type(), INT_VALUE) != INT_VALUE ||
      FileWriteInteger(handle, Neurons(), INT_VALUE) != INT_VALUE ||
      FileWriteInteger(handle, Activation(), INT_VALUE) != INT_VALUE)
      return(false);
   if(training &&
      (FileWriteInteger(handle, (int)optimization, INT_VALUE) != INT_VALUE ||
       FileWriteInteger(handle, (int)iBatch, INT_VALUE) != INT_VALUE ||
       FileWriteLong(handle, (long)t) != sizeof(long)))
      return(false);
   if(!SaveDPCCParameterBufferOnly(handle, Weights))
      return(false);
   if(training && (!SaveDPCCParameterBufferOnly(handle, FirstMomentum) ||
                   !SaveDPCCParameterBufferOnly(handle, SecondMomentum)))
      return(false);
   const int geometry[] = {(int)GetWindow(), (int)GetInputStride(), (int)GetFilters(),
                           (int)GetVariables(), (int)GetUnits()
                          };
   for(int i = 0; i < ArraySize(geometry); i++)
      if(FileWriteInteger(handle, geometry[i], INT_VALUE) != INT_VALUE)
         return(false);
   if(!SaveDPCCParameterBufferOnly(handle, WeightsConv))
      return(false);
   return(!training || (SaveDPCCParameterBufferOnly(handle, FirstMomentumConv) &&
                        SaveDPCCParameterBufferOnly(handle, SecondMomentumConv)));
  }
//+------------------------------------------------------------------+
//| Validates exact Actor identity before restoring parameters.      |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::LoadActorState(const int handle, const bool training, const bool apply)
  {
   if(handle == INVALID_HANDLE || CheckPointer(WeightsConv) == POINTER_INVALID || WeightsConv.Total() <= 0 ||
      (training && !ValidateTrainingOwners()))
      return(false);
   ResetLastError();
   if(FileReadInteger(handle, INT_VALUE) != Type() || FileReadInteger(handle, INT_VALUE) != Neurons() ||
      FileReadInteger(handle, INT_VALUE) != Activation() || GetLastError() != 0)
      return(false);
   long step = 0;
   if(training)
     {
      if(FileReadInteger(handle, INT_VALUE) != (int)optimization ||
         FileReadInteger(handle, INT_VALUE) != (int)iBatch || GetLastError() != 0)
         return(false);
      step = FileReadLong(handle);
      if(GetLastError() != 0 || step < 0 || step > UINT_MAX)
         return(false);
     }
   if(!LoadDPCCParameterBufferOnly(handle, Weights, apply))
      return(false);
   if(training && (!LoadDPCCParameterBufferOnly(handle, FirstMomentum, apply) ||
                   !LoadDPCCParameterBufferOnly(handle, SecondMomentum, apply)))
      return(false);
   const int geometry[] = {(int)GetWindow(), (int)GetInputStride(), (int)GetFilters(),
                           (int)GetVariables(), (int)GetUnits()
                          };
   for(int i = 0; i < ArraySize(geometry); i++)
      if(FileReadInteger(handle, INT_VALUE) != geometry[i] || GetLastError() != 0)
         return(false);
   if(!LoadDPCCParameterBufferOnly(handle, WeightsConv, apply))
      return(false);
   if(training && (!LoadDPCCParameterBufferOnly(handle, FirstMomentumConv, apply) ||
                   !LoadDPCCParameterBufferOnly(handle, SecondMomentumConv, apply)))
      return(false);
   if(training && apply)
      t = (uint)step;
   return(true);
  }
//+------------------------------------------------------------------+
//| Fingerprints new Actor identity and inherited Adam continuation. |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::AppendDPCCPolicyTrainingFingerprint(ulong &fingerprint)
  {
   if(!ValidateTrainingOwners() || !AppendDPCCPolicyTrainingBaseFingerprint(fingerprint))
      return(false);
   const ulong geometry[] = {ulong(GetWindow()), ulong(GetInputStride()), ulong(GetFilters()),
                             ulong(GetVariables()), ulong(GetUnits())
                            };
   for(int i = 0; i < ArraySize(geometry); i++)
      fingerprint = (fingerprint ^ geometry[i]) * ulong(1099511628211);
   return(AppendParameterBufferFingerprint(WeightsConv, fingerprint, 31) &&
          AppendParameterBufferFingerprint(FirstMomentumConv, fingerprint, 32) &&
          AppendParameterBufferFingerprint(SecondMomentumConv, fingerprint, 33));
  }
//+------------------------------------------------------------------+
//| Converts the first Action6 Critic gradient to sigmoid logits.    |
//+------------------------------------------------------------------+
bool CNeuronDPCCActorOCL::PrepareActionGradient(CBufferFloat *query_gradient, const uint account_width,
      const uint actions, string &reason)
  {
   reason = "";
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(query_gradient) == POINTER_INVALID ||
      CheckPointer(Gradient) == POINTER_INVALID || CheckPointer(Output) == POINTER_INVALID ||
      Activation() != SIGMOID || actions != 6 || account_width == 0 || account_width > (uint)INT_MAX - actions ||
      query_gradient.Total() != (int)(account_width + actions) ||
      query_gradient.GetOpenCL() != OpenCL || query_gradient.GetIndex() < 0 ||
      Output.Total() < (int)actions || Output.Total() != Gradient.Total() ||
      Output.GetOpenCL() != OpenCL || Gradient.GetOpenCL() != OpenCL ||
      Output.GetIndex() < 0 || Gradient.GetIndex() < 0 ||
      OwnsBufferOrDeviceSlot(query_gradient))
     {
      reason = "online.z.actor_gradient_contract";
      return(false);
     }
   if(!Gradient.Fill(0.0f))
     {
      reason = "online.z.actor_gradient_clear";
      return(false);
     }
   const uint count = (uint)Output.Total();
   CBufferFloat account_gradient, action_gradient, one, zero, factor;
   if(!account_gradient.BufferInit(account_width, 0) || !action_gradient.BufferInit(count, 0) ||
      !one.BufferInit(count, 1) || !zero.BufferInit(count, 0) || !factor.BufferInit(count, 0))
     {
      reason = "online.z.actor_gradient_scratch_init";
      return(false);
     }
   CBufferFloat *scratch[] = {GetPointer(account_gradient), GetPointer(action_gradient),
                              GetPointer(one), GetPointer(zero), GetPointer(factor)
                             };
   for(int i = 0; i < ArraySize(scratch); i++)
      if(!scratch[i].BufferCreate(OpenCL))
        {
         reason = "online.z.actor_gradient_scratch_context";
         return(false);
        }
   const int arguments[] = {account_gradient.GetIndex(), (int)account_width, action_gradient.GetIndex(), (int)actions,
                            account_gradient.GetIndex(), 0, Gradient.GetIndex(), 0, query_gradient.GetIndex()
                           };
   for(uint i = 0; i < 9; i++)
      if(!(i % 2 == 0 ? OpenCL.SetArgumentBuffer(def_k_DeconcatenateMatrix, i, arguments[i]) :
           OpenCL.SetArgument(def_k_DeconcatenateMatrix, i, arguments[i])))
        {
         reason = "online.z.actor_gradient_split_argument";
         return(false);
        }
   uint offset[2] = {0, 0};
   uint global[2] = {1, account_width};
   if(!OpenCL.Execute(def_k_DeconcatenateMatrix, 2, offset, global) ||
      !Difference(GetPointer(one), Output, Gradient, count, reason) ||
      !Gate(Output, GetPointer(zero), Gradient, GetPointer(factor), count, reason) ||
      !Gate(GetPointer(action_gradient), GetPointer(zero), GetPointer(factor), Gradient, count, reason))
     {
      Gradient.Fill(0.0f);
      if(reason == "")
         reason = "online.z.actor_gradient_dispatch";
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Native FQF buffers are captured, never enlarged into a fake M    |
//| head.                                                            |
//+------------------------------------------------------------------+
bool CNeuronFQF::CaptureEvaluationState(SLayerReadState &state)
  {
   return(CNeuronBaseOCL::CaptureEvaluationState(state) && cFraction.CaptureEvaluationState(state) &&
          cSoftMax.CaptureEvaluationState(state) && cCosine.CaptureEvaluationState(state) &&
          cCosineEmbeding.CaptureEvaluationState(state) && cQuantile0.CaptureEvaluationState(state) &&
          cQuantile1.CaptureEvaluationState(state) && cQuantile2.CaptureEvaluationState(state));
  }
//+------------------------------------------------------------------+
//| Function CNeuronFQF::RestoreEvaluationState                      |
//+------------------------------------------------------------------+
bool CNeuronFQF::RestoreEvaluationState(SLayerReadState &state)
  {
   return(CNeuronBaseOCL::RestoreEvaluationState(state) && cFraction.RestoreEvaluationState(state) &&
          cSoftMax.RestoreEvaluationState(state) && cCosine.RestoreEvaluationState(state) &&
          cCosineEmbeding.RestoreEvaluationState(state) && cQuantile0.RestoreEvaluationState(state) &&
          cQuantile1.RestoreEvaluationState(state) && cQuantile2.RestoreEvaluationState(state));
  }
//+------------------------------------------------------------------+
//| Function CNeuronFQF::ForwardRowsReadOnly                         |
//+------------------------------------------------------------------+
bool CNeuronFQF::ForwardRowsReadOnly(CNeuronBaseOCL *input_owner,
                                     CBufferFloat *source,
                                     const uint rows,
                                     const SLayerReadState &state,
                                     SLayerRowsWorkspace &work,
                                     CBufferFloat *quantiles,
                                     CBufferFloat *fractions,
                                     CBufferFloat *flags
                                    )
  {
   const uint n = (uint)cFraction.Neurons(), width = (uint)cCosineEmbeding.Neurons();
   if(Neurons() != 1 || !input_owner || input_owner.GetOpenCL() != OpenCL || state.Context() != OpenCL ||
      work.Context() != OpenCL || input_owner.Neurons() != (int)width || n == 0 || n > defFQF_LOCAL_ARRAY_SIZE ||
      !NNQValid(OpenCL, quantiles, (ulong)rows * n) || !NNQValid(OpenCL, fractions, (ulong)rows * n))
      return(false);
   CBufferFloat *logits = work.Take(rows * n), *cosine = work.Take(rows * n), *embedding = work.Take(rows * width);
   CBufferFloat *product = work.Take(rows * width), *hidden = work.Take(rows * (uint)cQuantile1.Neurons());
   if(!logits || !cosine || !embedding || !product || !hidden)
      return(false);
   if(!NNQDense(OpenCL, input_owner.getWeights(), source, logits, flags, rows, width, n, None))
      return(false);
   CBufferFloat *soft_buffers[] = {logits, fractions, flags};
   uint soft_values[] = {n}, soft_global[] = {n, rows};
   float reals[];
   if(!NNQLaunch(OpenCL, def_k_DPCCQRowsSoftmax, soft_buffers, soft_values, reals, soft_global))
      return(false);
   CBufferFloat *cos_buffers[] = {fractions, cosine, flags};
   if(!NNQLaunch(OpenCL, def_k_DPCCQRowsCosine, cos_buffers, soft_values, reals, soft_global))
      return(false);
   if(!NNQDense(OpenCL, cCosine.getWeights(), cosine, embedding, flags, rows, n, width, LReLU) ||
      !NNQMap(OpenCL, source, embedding, product, flags, rows, width, width, width, 0, 0, 2) ||
      !NNQDense(OpenCL, cQuantile0.getWeights(), product, hidden, flags, rows, width, (uint)cQuantile1.Neurons(), LReLU) ||
      !NNQDense(OpenCL, cQuantile1.getWeights(), hidden, quantiles, flags, rows, (uint)cQuantile1.Neurons(), n, None))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronACSRM::CaptureEvaluationState                    |
//+------------------------------------------------------------------+
bool CNeuronACSRM::CaptureEvaluationState(SLayerReadState &state)
  {
   if(!CNeuronFQF::CaptureEvaluationState(state) || !state.PutInt((long)m_distribution_gradient_ready) ||
      !state.PutInt((long)m_rearrangement_ready) || !state.PutInt(m_last_invalid_quantile) ||
      !state.PutInt(m_last_invalid_probability) || !state.PutInt(m_last_invalid_mass) ||
      !state.PutInt(m_last_spectral_break) || !state.PutInt((long)m_last_srm_invalid))
      return(false);
   CBufferFloat *buffers[] = {GetPointer(m_input_gradient_scratch), GetPointer(m_ordered_quantiles),
                              GetPointer(m_ordered_probabilities), GetPointer(m_rearrange_order), GetPointer(m_ordered_quantiles_gradient),
                              GetPointer(m_ordered_probabilities_gradient), GetPointer(m_rearrange_diagnostics)
                             };
   for(int i = 0; i < 7; i++)
      if(!state.PutBuffer(buffers[i]))
         return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function CNeuronACSRM::RestoreEvaluationState                    |
//+------------------------------------------------------------------+
bool CNeuronACSRM::RestoreEvaluationState(SLayerReadState &state)
  {
   if(!CNeuronFQF::RestoreEvaluationState(state))
      return(false);
   long values[7];
   for(int i = 0; i < 7; i++)
      if(!state.GetInt(values[i]))
         return(false);
   m_distribution_gradient_ready = (values[0] != 0);
   m_rearrangement_ready = (values[1] != 0);
   m_last_invalid_quantile = (int)values[2];
   m_last_invalid_probability = (int)values[3];
   m_last_invalid_mass = (int)values[4];
   m_last_spectral_break = (int)values[5];
   m_last_srm_invalid = (values[6] != 0);
   CBufferFloat *buffers[] = {GetPointer(m_input_gradient_scratch), GetPointer(m_ordered_quantiles),
                              GetPointer(m_ordered_probabilities), GetPointer(m_rearrange_order), GetPointer(m_ordered_quantiles_gradient),
                              GetPointer(m_ordered_probabilities_gradient), GetPointer(m_rearrange_diagnostics)
                             };
   for(int i = 0; i < 7; i++)
     {CBufferFloat *original = buffers[i]; if(!state.GetBuffer(buffers[i]) || buffers[i] != original)return(false);}
   return(true);
  }
//+------------------------------------------------------------------+
//| Reuse published rearrangement/SRM kernels on explicit row        |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronACSRM::SRMRowsReadOnly(CBufferFloat *quantiles,
                                   CBufferFloat *fractions,
                                   const uint rows,
                                   const int risk_type,
                                   const float risk_parameter,
                                   const float mean_weight,
                                   SLayerRowsWorkspace &work,
                                   CBufferFloat *scores,
                                   CBufferFloat *flags
                                  )
  {
   const uint n = (uint)cFraction.Neurons();
   const bool valid_risk = (risk_type == defSRM_MEAN ||
                            ((risk_type == defSRM_CVAR || risk_type == defSRM_MEAN_CVAR) && risk_parameter > 0 && risk_parameter <= 1) ||
                            (risk_type == defSRM_EXPONENTIAL && risk_parameter >= 0) || (risk_type == defSRM_DUAL_POWER && risk_parameter >= 1));
   if(!valid_risk || !MathIsValidNumber(risk_parameter) || !MathIsValidNumber(mean_weight) || mean_weight < 0 || mean_weight > 1 ||
      work.Context() != OpenCL || Neurons() != 1 || n == 0 || n > defFQF_LOCAL_ARRAY_SIZE || rows == 0 ||
      !NNQValid(OpenCL, quantiles, (ulong)rows * n) || !NNQValid(OpenCL, fractions, (ulong)rows * n) ||
      !NNQValid(OpenCL, scores, rows) || !NNQValid(OpenCL, flags, rows))
      return(false);
   CBufferFloat *validate[] = {quantiles, fractions, flags};
   uint counts[] = {n}, global_rows[] = {rows};
   float tolerance[] = {defFQF_PROBABILITY_TOLERANCE};
   if(!NNQLaunch(OpenCL, def_k_DPCCQRowsValidate, validate, counts, tolerance, global_rows))
      return(false);
   CBufferFloat *q = quantiles, *p = fractions;
   if(risk_type != defSRM_MEAN)
     {
      if((n & (n - 1)) != 0 || OpenCL.GetMaxWorkGroupSize() < (long)n || OpenCL.GetMaxLocalSize(0) < n)
         return(false);
      q = work.Take(rows * n);
      p = work.Take(rows * n);
      CBufferFloat *order = work.Take(rows * n);
      CBufferFloat *diagnostics = work.Take(rows * defFQF_DIAGNOSTIC_FLOATS);
      if(!q || !p || !order || !diagnostics)
         return(false);
      CBufferFloat *buffers[] = {quantiles, fractions, q, p, order, diagnostics};
      for(uint i = 0; i < 6; i++)
         if(!OpenCL.SetArgumentBuffer(def_k_FQF_MonotoneRearrange, i, buffers[i].GetIndex()))
            return(false);
      uint offset[] = {0, 0}, global[] = {n, rows}, local[] = {n, 1};
      if(!OpenCL.Execute(def_k_FQF_MonotoneRearrange, 2, offset, global, local))
         return(false);
     }
   if(!OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 0, q.GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 1, p.GetIndex()) ||
      !OpenCL.SetArgumentBuffer(def_k_FQF_SRMForward, 2, scores.GetIndex()) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 3, risk_type) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 4, risk_parameter) ||
      !OpenCL.SetArgument(def_k_FQF_SRMForward, 5, mean_weight))
      return(false);
   uint offset[] = {0, 0}, global[] = {n, rows}, local[] = {n, 1};
   if(OpenCL.GetMaxWorkGroupSize() < (long)n || OpenCL.GetMaxLocalSize(0) < n)
      return(false);
   return(OpenCL.Execute(def_k_FQF_SRMForward, 2, offset, global, local));
  }
//+------------------------------------------------------------------+
//| Capture/restore all five native Q layers around selected         |
//| backward.                                                        |
//+------------------------------------------------------------------+
bool NNQCriticGraph(CNet *critic)
  {
   if(CheckPointer(critic) == POINTER_INVALID || !critic.IsOpenCLTransferReady())
      return(false);
   COpenCLMy *context = critic.GetOpenCL();
   if(CheckPointer(context) == POINTER_INVALID)
      return(false);
   CNeuronBaseOCL *layers[5];
   for(int i = 0; i < 5; i++)
     {
      layers[i] = critic.Layer(i);
      if(CheckPointer(layers[i]) == POINTER_INVALID || layers[i].GetOpenCL() != context)
         return(false);
     }
   if(CheckPointer(critic.Layer(5)) != POINTER_INVALID)
      return(false);
   CNeuronScenarioCrossAttention *cross = dynamic_cast<CNeuronScenarioCrossAttention*>(layers[1]);
   CNeuronConvOCL *conv = dynamic_cast<CNeuronConvOCL*>(layers[2]);
   if(CheckPointer(cross) == POINTER_INVALID || CheckPointer(conv) == POINTER_INVALID ||
      cross.Latent() == 0 || cross.Scenarios() == 0 || cross.Variables() == 0 || cross.Horizon() == 0 ||
      conv.GetWindow() != 5 * cross.Latent() || conv.GetInputStride() != 5 * cross.Latent() ||
      conv.GetFilters() != cross.Latent() || conv.GetVariables() != 1 || conv.GetUnits() != 1)
      return(false);
   return(layers[0].Type() == defNeuronBaseOCL && layers[0].Neurons() == 19 &&
          layers[0].Activation() == None && cross.IsCritic() &&
          cross.Queries() == 5 && layers[2].Type() == defNeuronConvOCL && layers[2].Activation() == GELU &&
          layers[2].Neurons() == (int)cross.Latent() && layers[3].Type() == defNeuronBaseOCL &&
          layers[3].Neurons() == 32 && layers[3].Activation() == None && layers[4].Type() == defNeuronACSRM &&
          layers[4].Neurons() == 1);
  }
//+------------------------------------------------------------------+
//| Function NNQOwnedByCritic                                        |
//+------------------------------------------------------------------+
bool NNQOwnedByCritic(CNet *critic, CBufferFloat *buffer)
  {
   if(!critic || !buffer)
      return(true);
   for(int i = 0; i < 5; i++)
     {
      CNeuronBaseOCL *node = critic.Layer(i);
      if(!node || node.OwnsBufferOrDeviceSlot(buffer))
         return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Function NNQCaptureCritic                                        |
//+------------------------------------------------------------------+
bool NNQCaptureCritic(CNet *critic, SLayerReadState &state, const ulong owner, const ulong state_id)
  {
   if(!NNQCriticGraph(critic) || !state.Begin(critic.GetOpenCL(), owner, state_id) ||
      !state.PutInt((long)critic.WeightsUpdateEnabled()))
      return(false);
   for(int i = 0; i < 5; i++)
      if(!critic.Layer(i).CaptureEvaluationState(state))
        {
         state.Clear();
         return(false);
        }
   return(true);
  }
//+------------------------------------------------------------------+
//| Function NNQRestoreCritic                                        |
//+------------------------------------------------------------------+
bool NNQRestoreCritic(CNet *critic, SLayerReadState &state)
  {
   if(!NNQCriticGraph(critic) || state.Context() != critic.GetOpenCL())
      return(false);
   state.Rewind();
   long update = 0;
   if(!state.GetInt(update))
      return(false);
   for(int i = 0; i < 5; i++)
      if(!critic.Layer(i).RestoreEvaluationState(state))
         return(false);
   return(state.Complete() && critic.SetWeightsUpdate(update != 0));
  }
//+------------------------------------------------------------------+
//| Function NNQEvaluateCriticRows                                   |
//+------------------------------------------------------------------+
bool NNQEvaluateCriticRows(CNet *critic,
                           CBufferFloat *account13,
                           CBufferFloat *plans,
                           const uint horizon,
                           const uint rows,
                           CBufferFloat *z,
                           CBufferFloat *log_u,
                           CBufferFloat *pi,
                           const int forecast_type,
                           const int risk_type,
                           const float risk_parameter,
                           const float mean_weight,
                           const SLayerReadState &state,
                           SLayerRowsWorkspace &work,
                           CBufferFloat *scores,
                           CBufferFloat *numeric_flags
                          )
  {
   if(!NNQCriticGraph(critic) || state.Context() != critic.GetOpenCL() || state.Owner() == 0 || state.State() == 0 ||
      !work.Begin(critic.GetOpenCL(), rows, state.Owner(), state.State()) ||
      NNQOwnedByCritic(critic, scores) || NNQOwnedByCritic(critic, numeric_flags))
      return(false);
   COpenCLMy *context = critic.GetOpenCL();
   if(!NNQValid(context, account13, 13) || !NNQValid(context, plans, (ulong)rows * horizon * 6) ||
      !NNQValid(context, numeric_flags, rows) || !NNQValid(context, scores, rows) || horizon == 0)
      return(false);
   CBufferFloat *query = work.Take(rows * 19);
   if(!query)
      return(false);
   CBufferFloat *buffers[] = {account13, plans, query, numeric_flags};
   uint values[] = {horizon}, global[] = {19, rows};
   float reals[];
   if(!NNQLaunch(context, def_k_DPCCQRowsJoin, buffers, values, reals, global))
      return(false);
   CNeuronScenarioCrossAttention *cross = dynamic_cast<CNeuronScenarioCrossAttention*>(critic.Layer(1));
   CNeuronConvOCL *conv = dynamic_cast<CNeuronConvOCL*>(critic.Layer(2));
   CNeuronACSRM *head = dynamic_cast<CNeuronACSRM*>(critic.Layer(4));
   CBufferFloat *trunk = work.Take(rows * (uint)cross.Neurons());
   CBufferFloat *projection = work.Take(rows * (uint)conv.Neurons()), *dense = work.Take(rows * 32);
   CLayerDescription *head_info = head.GetLayerInfo();
   if(!head_info)
      return(false);
   const uint n = head_info.window_out;
   delete head_info;
   CBufferFloat *quantiles = work.Take(rows * n), *fractions = work.Take(rows * n);
   if(!trunk || !projection || !dense || !quantiles || !fractions)
      return(false);
   if(!cross.ForwardRowsReadOnly(query, rows, z, log_u, pi, forecast_type, state, work, trunk, numeric_flags) ||
      !conv.ForwardRowsReadOnly(trunk, rows, state, work, projection, numeric_flags) ||
      !critic.Layer(3).ForwardRowsReadOnly(projection, rows, conv.getWeights(), (uint)conv.Neurons(), work, dense, numeric_flags) ||
      !head.ForwardRowsReadOnly(critic.Layer(3), dense, rows, state, work, quantiles, fractions, numeric_flags))
      return(false);
   return(head.SRMRowsReadOnly(quantiles, fractions, rows, risk_type, risk_parameter, mean_weight, work, scores, numeric_flags));
  }
//+------------------------------------------------------------------+
//| Function NNQRankCriticRows                                       |
//+------------------------------------------------------------------+
bool NNQRankCriticRows(COpenCLMy *context, CBufferFloat *q1, CBufferFloat *q2, CBufferFloat *candidate_flags,
                       CBufferFloat *numeric_flags, CBufferFloat *scores, CBufferFloat *selected, const uint rows)
  {
   if(rows == 0 || rows > 16777216 || !NNQValid(context, q1, rows) || !NNQValid(context, q2, rows) ||
      !NNQValid(context, candidate_flags, rows) || !NNQValid(context, numeric_flags, rows) ||
      !NNQValid(context, scores, rows) || !NNQValid(context, selected, 1))
      return(false);
   CBufferFloat *buffers[] = {q1, q2, candidate_flags, numeric_flags, scores, selected};
   uint values[] = {rows}, global[] = {1};
   float reals[];
   return(NNQLaunch(context, def_k_DPCCQRowsRank, buffers, values, reals, global));
  }
//+------------------------------------------------------------------+
//| Native SRM input gradient; restore Spike/BN/stack runtime on     |
//| exit.                                                            |
//+------------------------------------------------------------------+
bool NNQSelectedPolicyGradient(CNet *critic,
                               CNet *market,
                               CBufferFloat *account13,
                               CBufferFloat *action6,
                               const int risk_type,
                               const float risk_parameter,
                               const float mean_weight,
                               const float upstream,
                               SLayerReadState &state,
                               SLayerRowsWorkspace &work,
                               CBufferFloat *action_gradient
                              )
  {
   if(!NNQCriticGraph(critic) || !market || market.GetOpenCL() != critic.GetOpenCL() ||
      state.Context() != critic.GetOpenCL() || !market.Layer(6) || !MathIsValidNumber(upstream) || !market.IsOpenCLTransferReady() ||
      market.WeightsUpdateEnabled() || state.Owner() == 0 || state.State() == 0)
      return(false);
   CNeuronScenarioForecast *forecast = dynamic_cast<CNeuronScenarioForecast*>(market.Layer(6));
   if(!forecast || (forecast.Type() != defNeuronScenarioForecast && forecast.Type() != defNeuronScenarioForecastPriceOCL))
      return(false);
   COpenCLMy *context = critic.GetOpenCL();
   if(!work.Begin(context, 1, state.Owner(), state.State()) || !NNQValid(context, account13, 13) ||
      !NNQValid(context, action6, 6) || !NNQValid(context, action_gradient, 6) ||
      NNQOwnedByCritic(critic, action_gradient))
      return(false);
   CBufferFloat *query = work.Take(19), *flags = work.Take(1), *srm = work.Take(1), *gradient = work.Take(1);
   if(!query || !flags || !srm || !gradient || !flags.Fill(0) || !gradient.Fill(upstream))
      return(false);
   CBufferFloat *buffers[] = {account13, action6, query, flags};
   uint values[] = {1}, global[] = {19, 1};
   float reals[];
   if(!NNQLaunch(context, def_k_DPCCQRowsJoin, buffers, values, reals, global) || !query.BufferRead() || !flags.BufferRead() || flags.At(0) != 0)
      return(false);
   if(!NNQRestoreCritic(critic, state))
      return(false);
   bool ok = false;
   do
     {
      //--- Virtual inference BN and captured gates match the read-only rows semantics.
      if(!critic.TrainMode(false) || !critic.SetWeightsUpdate(false) ||
         !critic.feedForward((CArrayFloat*)query, 1, false, market, 6))
         break;
      CNeuronACSRM *head = dynamic_cast<CNeuronACSRM*>(critic.Layer(4));
      if(!head.SRMForward(srm, 1, risk_type, risk_parameter, mean_weight) ||
         !head.SRMGradient(gradient, 1, risk_type, risk_parameter, mean_weight) ||
         !critic.backPropGradient(market, 6, -1, false))
         break;
      if(!NNQMap(context, critic.Layer(0).getGradient(), critic.Layer(0).getGradient(),
                 action_gradient, flags, 1, 6, 0, 0, 13, 0, 0) || !flags.BufferRead() || flags.At(0) != 0)
         break;
      ok = true;
     }
   while(false);
//--- Also restore native backward cStack changes, activation roles and diagnostic flags.
   const bool restored = NNQRestoreCritic(critic, state);
   if(!ok || !restored)
     {
      action_gradient.Fill(0);
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
