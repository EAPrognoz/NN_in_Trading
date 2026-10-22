//+------------------------------------------------------------------+
//|                                                NeuroNet_Misc.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh:      |
//| Misc.                                                            |
//+------------------------------------------------------------------+
//| Release the runtime-only posterior samplers.                     |
//+------------------------------------------------------------------+
//| Initializes the CNeuronMultiModel layer and allocates its GPU    |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Init(uint numInputs,
                             uint myIndex,
                             COpenCLMy * open_cl,
                             uint numNeurons,
                             ENUM_OPTIMIZATION optimization_type,
                             int models
                            )
  {
   if(CheckPointer(open_cl) == POINTER_INVALID || numNeurons <= 0)
      ReturnFalse;
   OpenCL = open_cl;
   optimization = ADAM;
   iBatch = 1;
   iModels = models;
//---
   if(CheckPointer(Output) == POINTER_INVALID)
     {
      Output = new CBufferFloat();
      if(CheckPointer(Output) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!Output.BufferInit(numNeurons * models, 0.0))
      ReturnFalse;
   if(!Output.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(PrevOutput) == POINTER_INVALID)
     {
      PrevOutput = new CBufferFloat();
      if(CheckPointer(PrevOutput) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!PrevOutput.BufferInit(numNeurons * models, 1.0))
      ReturnFalse;
   if(!PrevOutput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(Gradient) == POINTER_INVALID)
     {
      Gradient = new CBufferFloat();
      if(CheckPointer(Gradient) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!Gradient.BufferInit((numNeurons + 1)*models, 0.0))
      ReturnFalse;
   if(!Gradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(Weights) == POINTER_INVALID)
     {
      Weights = new CBufferFloat();
      if(CheckPointer(Weights) == POINTER_INVALID)
         ReturnFalse;
     }
   int count = (int)((numInputs + 1) * numNeurons * models);
   if(!Weights.Reserve(count))
      ReturnFalse;
   float k = (float)(1 / sqrt(numInputs + 1));
   for(int i = 0; i < count; i++)
     {
      if(!Weights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!Weights.BufferCreate(OpenCL))
      ReturnFalse;
//---
   DeleteObj(DeltaWeights);
//---
   if(CheckPointer(FirstMomentum) == POINTER_INVALID)
     {
      FirstMomentum = new CBufferFloat();
      if(CheckPointer(FirstMomentum) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!FirstMomentum.BufferInit(count, 0))
      ReturnFalse;
   if(!FirstMomentum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(SecondMomentum) == POINTER_INVALID)
     {
      SecondMomentum = new CBufferFloat();
      if(CheckPointer(SecondMomentum) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!SecondMomentum.BufferInit(count, 0))
      ReturnFalse;
   if(!SecondMomentum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Assemble the posterior parameter tensor.                         |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = Output.Total() / iModels;
   global_work_size[1] = iModels;
   setBuffer(def_k_FFMultiModels, def_k_ff_matrix_w, getWeightsIndex())
   setBuffer(def_k_FFMultiModels, def_k_ff_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(def_k_FFMultiModels, def_k_ff_matrix_o, Output.GetIndex())
   setArgument(def_k_FFMultiModels, def_k_ff_inputs, NeuronOCL.Neurons() / iModels)
   setArgument(def_k_FFMultiModels, def_k_ff_activation, (int)activation)
   kernelExecute(def_k_FFMultiModels, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronMultiModel calc hidden gradients operation.   |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::calcHiddenGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = NeuronOCL.Neurons() / iModels;
   global_work_size[1] = iModels;
   setBuffer(def_k_HGMultiModels, def_k_chg_matrix_w, getWeightsIndex())
   setBuffer(def_k_HGMultiModels, def_k_chg_matrix_g, getGradientIndex())
   setBuffer(def_k_HGMultiModels, def_k_chg_matrix_o, NeuronOCL.getOutputIndex())
   setBuffer(def_k_HGMultiModels, def_k_chg_matrix_ig, NeuronOCL.getGradientIndex())
   setArgument(def_k_HGMultiModels, def_k_chg_outputs, Neurons() / iModels)
   setArgument(def_k_HGMultiModels, def_k_chg_activation, NeuronOCL.Activation())
   iUpdateModel = (int)MathRound(MathRand() / 32767.0 * (iModels - 1));
   setArgument(def_k_HGMultiModels, def_k_chg_model, iUpdateModel)
//---Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_HGMultiModels, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMultiModel trainable weights from the         |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = Neurons() / iModels;
   global_work_size[1] = NeuronOCL.Neurons() / iModels + 1;
   uint rest = 0;
   float lt = lr;
   setBuffer(def_k_UWMultiModels, def_k_uwa_matrix_w, getWeightsIndex())
   setBuffer(def_k_UWMultiModels, def_k_uwa_matrix_g, getGradientIndex())
   setBuffer(def_k_UWMultiModels, def_k_uwa_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(def_k_UWMultiModels, def_k_uwa_matrix_m, getFirstMomentumIndex())
   setBuffer(def_k_UWMultiModels, def_k_uwa_matrix_v, getSecondMomentumIndex())
   lt = (float)(lr * sqrt(1 - pow(b2, (float)t)) / (1 - pow(b1, (float)t)));
   setArgument(def_k_UWMultiModels, def_k_uwa_inputs, NeuronOCL.Neurons() / iModels)
   setArgument(def_k_UWMultiModels, def_k_uwa_l, lt)
   setArgument(def_k_UWMultiModels, def_k_uwa_b1, b1)
   setArgument(def_k_UWMultiModels, def_k_uwa_b2, b2)
   setArgument(def_k_UWMultiModels, def_k_uwa_model, iUpdateModel)
   global_work_size[1] = (global_work_size[1] + 3) / 4;
   ResetLastError();
   kernelExecute(def_k_UWMultiModels, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Weights.BufferRead())
      ReturnFalse;
#endif
   t++;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMultiModel state to the file handle.            |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iModels) <= 0)
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMultiModel state from the file handle.       |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   iModels = FileReadInteger(file_handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronConcatenate layer defaults.  |
//+------------------------------------------------------------------+
CNeuronConcatenate::CNeuronConcatenate(void) : i_SecondInputs(0)
  {
   ConcWeights = new CBufferFloat();
   ConcDeltaWeights = new CBufferFloat();
   ConcFirstMomentum = new CBufferFloat();
   ConcSecondMomentum = new CBufferFloat;
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronConcatenate GPU resources.       |
//+------------------------------------------------------------------+
CNeuronConcatenate::~CNeuronConcatenate()
  {
   DeleteObj(ConcWeights);
   DeleteObj(ConcDeltaWeights);
   DeleteObj(ConcFirstMomentum);
   DeleteObj(ConcSecondMomentum);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronConcatenate layer and allocates its GPU   |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::Init(uint numOutputs,
                              uint myIndex,
                              COpenCLMy * open_cl,
                              uint numNeurons,
                              uint numInputs1,
                              uint numInputs2,
                              ENUM_OPTIMIZATION optimization_type,
                              uint batch
                             )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
//---
   i_SecondInputs = (int)numInputs2;
   if(!ConcWeights)
     {
      ConcWeights = new CBufferFloat();
      if(!ConcWeights)
         ReturnFalse;
     }
   int count = (int)((numInputs1 + numInputs2 + 1) * numNeurons);
   if(!ConcWeights.Reserve(count))
      ReturnFalse;
   float k = (float)(1 / sqrt(numNeurons + 1));
   for(int i = 0; i < count; i++)
     {
      if(!ConcWeights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!ConcWeights.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(optimization == SGD)
     {
      if(!ConcDeltaWeights)
        {
         ConcDeltaWeights = new CBufferFloat();
         if(!ConcDeltaWeights)
            ReturnFalse;
        }
      if(!ConcDeltaWeights.BufferInit(count, 0))
         ReturnFalse;
      if(!ConcDeltaWeights.BufferCreate(OpenCL))
         ReturnFalse;
      DeleteObj(ConcFirstMomentum);
      DeleteObj(ConcSecondMomentum);
     }
   else
     {
      DeleteObj(ConcDeltaWeights);
      //---
      if(!ConcFirstMomentum)
        {
         ConcFirstMomentum = new CBufferFloat();
         if(CheckPointer(ConcFirstMomentum) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!ConcFirstMomentum.BufferInit(count, 0))
         ReturnFalse;
      if(!ConcFirstMomentum.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(!ConcSecondMomentum)
        {
         ConcSecondMomentum = new CBufferFloat();
         if(!ConcSecondMomentum)
            ReturnFalse;
        }
      if(!ConcSecondMomentum.BufferInit(count, 0))
         ReturnFalse;
      if(!ConcSecondMomentum.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronConcatenate feed-forward pass and writes the     |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(SecondInput.Total() < i_SecondInputs)
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   setBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_w, ConcWeights.GetIndex())
   setBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_i1, NeuronOCL.getOutputIndex())
   setBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_i2, SecondInput.GetIndex())
   setBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_o, Output.GetIndex())
   setArgument(def_k_ConcatFeedForward, def_k_cff_inputs1, (int)NeuronOCL.Neurons())
   setArgument(def_k_ConcatFeedForward, def_k_cff_inputs2, (int)i_SecondInputs)
   setArgument(def_k_ConcatFeedForward, def_k_cff_activation, (int)activation)
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Output.Total();
   kernelExecute(def_k_ConcatFeedForward, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronConcatenate layer.                                        |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::calcInputGradients(CNeuronBaseOCL* NeuronOCL,
                                            CBufferFloat* SecondInput,
                                            CBufferFloat* SecondGradient,
                                            ENUM_ACTIVATION SecondActivation = None
                                           )
  {
   if(!OpenCL || !NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
   if(SecondInput.Total() < i_SecondInputs || SecondGradient.Total() < i_SecondInputs)
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
   if(SecondGradient.GetIndex() < 0 && !SecondGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = NeuronOCL.Neurons() + i_SecondInputs;
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_w, ConcWeights.GetIndex())
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_g, Gradient.GetIndex())
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_ig1, NeuronOCL.getGradientIndex())
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_ig2, SecondGradient.GetIndex())
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_o1, NeuronOCL.getOutputIndex())
   setBuffer(def_k_ConcatCalcHiddenGradient, def_k_cchg_matrix_o2, SecondInput.GetIndex())
   setArgument(def_k_ConcatCalcHiddenGradient, def_k_cchg_inputs1, NeuronOCL.Neurons())
   setArgument(def_k_ConcatCalcHiddenGradient, def_k_cchg_inputs2, i_SecondInputs)
   setArgument(def_k_ConcatCalcHiddenGradient, def_k_cchg_outputs, Neurons())
   setArgument(def_k_ConcatCalcHiddenGradient, def_k_cchg_activation1, NeuronOCL.Activation())
   setArgument(def_k_ConcatCalcHiddenGradient, def_k_cchg_activation2, (int)SecondActivation)
   kernelExecute(def_k_ConcatCalcHiddenGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!ConcWeights.BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronConcatenate trainable weights from the        |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(SecondInput.Total() < i_SecondInputs)
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = Neurons();
   global_work_size[1] = NeuronOCL.Neurons() + i_SecondInputs + 1;
   float lt = lr;
   ResetLastError();
   switch(NeuronOCL.Optimization())
     {
      case SGD:
         setBuffer(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_matrix_w, ConcWeights.GetIndex())
         setBuffer(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_matrix_g, getGradientIndex())
         setBuffer(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_matrix_i1, NeuronOCL.getOutputIndex())
         setBuffer(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_matrix_i2, SecondInput.GetIndex())
         setBuffer(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_matrix_dw, ConcDeltaWeights.GetIndex())
         setArgument(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_inputs1, NeuronOCL.Neurons())
         setArgument(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_inputs1, i_SecondInputs)
         setArgument(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_learning_rates, lr)
         setArgument(def_k_ConcatUpdWeightsMomentum, def_k_cuwm_momentum, alpha)
         ResetLastError();
         kernelExecute(def_k_ConcatUpdWeightsMomentum, global_work_offset, global_work_size)
         break;
      case ADAM:
      case ADAM_MINI:
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_w, ConcWeights.GetIndex())
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_g, getGradientIndex())
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_i1, NeuronOCL.getOutputIndex())
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_i2, SecondInput.GetIndex())
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_m, ConcFirstMomentum.GetIndex())
         setBuffer(def_k_ConcatUpdWeightsAdam, def_k_cuwa_matrix_v, ConcSecondMomentum.GetIndex())
         lt = (float)(lr * sqrt(1 - pow(b2, (float)t)) / (1 - pow(b1, (float)t)));
         setArgument(def_k_ConcatUpdWeightsAdam, def_k_cuwa_inputs1, NeuronOCL.Neurons())
         setArgument(def_k_ConcatUpdWeightsAdam, def_k_cuwa_inputs2, i_SecondInputs)
         setArgument(def_k_ConcatUpdWeightsAdam, def_k_cuwa_l, lt)
         setArgument(def_k_ConcatUpdWeightsAdam, def_k_cuwa_b1, b1)
         setArgument(def_k_ConcatUpdWeightsAdam, def_k_cuwa_b2, b2)
         kernelExecute(def_k_ConcatUpdWeightsAdam, global_work_offset, global_work_size)
         t++;
         break;
      default:
         ReturnFalse;
         break;
     }
#ifdef  _DEBUG
   if(!ConcWeights.BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronConcatenate state to the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, i_SecondInputs) < INT_VALUE)
      ReturnFalse;
   if(!ConcWeights.Save(file_handle))
      ReturnFalse;
   if(optimization == SGD)
     {
      if(!ConcDeltaWeights.Save(file_handle))
         ReturnFalse;
     }
   else
     {
      if(!ConcFirstMomentum.Save(file_handle))
         ReturnFalse;
      if(!ConcSecondMomentum.Save(file_handle))
         ReturnFalse;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronConcatenate state from the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   i_SecondInputs = FileReadInteger(file_handle);
   if(!ConcWeights.Load(file_handle))
      ReturnFalse;
   ConcWeights.BufferCreate(OpenCL);
   if(optimization == SGD)
     {
      if(!ConcDeltaWeights.Load(file_handle))
         ReturnFalse;
      ConcDeltaWeights.BufferCreate(OpenCL);
     }
   else
     {
      if(!ConcFirstMomentum.Load(file_handle))
         ReturnFalse;
      if(!ConcSecondMomentum.Load(file_handle))
         ReturnFalse;
      ConcFirstMomentum.BufferCreate(OpenCL);
      ConcSecondMomentum.BufferCreate(OpenCL);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronConcatenate GPU buffers to the given OpenCL   |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronConcatenate::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!ConcWeights)
      return;
   ConcWeights.BufferCreate(OpenCL);
   if(optimization == SGD)
     {
      if(!ConcDeltaWeights)
         return;
      ConcDeltaWeights.BufferCreate(OpenCL);
     }
   else
     {
      if(!ConcFirstMomentum || !ConcSecondMomentum)
         return;
      ConcFirstMomentum.BufferCreate(OpenCL);
      ConcSecondMomentum.BufferCreate(OpenCL);
     }
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronConcatenate weights toward the source model  |
//| with factor tau.                                                 |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronConcatenate *temp = source;
   if(ConcWeights.Total() != temp.ConcWeights.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {ConcWeights.Total()};
   ResetLastError();
   if(tau != 1.0f && optimization == ADAM)
     {
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_target, ConcWeights.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_source, temp.ConcWeights.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_m, ConcFirstMomentum.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_v, ConcSecondMomentum.GetIndex())
      setArgument(def_k_SoftUpdateAdam, def_k_sua_tau, (float)tau)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b1, (float)b1)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b2, (float)b2)
      kernelExecute(def_k_SoftUpdateAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!ConcFirstMomentum.BufferRead())
         ReturnFalse;
#endif
     }
   else
     {
      setBuffer(def_k_SoftUpdate, def_k_su_target, ConcWeights.GetIndex())
      setBuffer(def_k_SoftUpdate, def_k_su_source, temp.ConcWeights.GetIndex())
      setArgument(def_k_SoftUpdate, def_k_su_tau, (float)tau)
      kernelExecute(def_k_SoftUpdate, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!ConcWeights.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronCSCMOCL layer and allocates its GPU       |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::Init(uint numOutputs,
                          uint myIndex,
                          COpenCLMy * open_cl,
                          uint & windows[],
                          uint variables,
                          uint inputs_count,
                          bool need_transpose,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch
                         )
  {
   const uint layers = windows.Size();
   if(layers <= 0)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, inputs_count * variables, optimization_type, batch))
      ReturnFalse;
//---
   ia_Windows.Clear();
   ia_Windows.Reserve(layers);
   for(uint i = 0; i < layers; i++)
      if(!ia_Windows.Add((int)windows[i]))
         ReturnFalse;
   i_Variables = variables;
   i_Count = inputs_count / ia_Windows[0];
   b_NeedTranspose = need_transpose;
//---
   if(b_NeedTranspose)
     {
      CNeuronTransposeOCL *transp = new CNeuronTransposeOCL();
      if(!transp)
         ReturnFalse;
      if(!transp.Init(0, 0, OpenCL, inputs_count, i_Variables, optimization, iBatch))
        {
         DeleteObj(transp);
         ReturnFalse;
        }
      if(!caTranspose.Add(transp))
        {
         DeleteObj(transp);
         ReturnFalse;
        }
      transp = new CNeuronTransposeOCL();
      if(!transp)
         ReturnFalse;
      if(!transp.Init(0, 1, OpenCL, i_Variables, inputs_count, optimization, iBatch))
        {
         DeleteObj(transp);
         ReturnFalse;
        }
      if(!caTranspose.Add(transp))
        {
         DeleteObj(transp);
         ReturnFalse;
        }
      if(!SetOutput(transp.getOutput()) ||
         !SetGradient(transp.getGradient())
        )
         ReturnFalse;
     }
//---
   uint total = ia_Windows[0] * i_Count;
   CNeuronConvOCL *conv = new CNeuronConvOCL();
   if(!conv.Init(0, 0, OpenCL, inputs_count, inputs_count, total, 1, i_Variables, optimization, iBatch))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SIGMOID);
   if(!caConvolutions.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//---
   total = 0;
   for(uint i = 0; i < layers; i++)
     {
      conv = new CNeuronConvOCL();
      if(!conv.Init(0, i + 1, OpenCL, ia_Windows[i], ia_Windows[i], (i < (layers - 1) ? ia_Windows[i + 1] : 1), i_Count, i_Variables, optimization, iBatch))
        {
         DeleteObj(conv);
         ReturnFalse;
        }
      conv.SetActivationFunction(SIGMOID);
      if(!caConvolutions.Add(conv))
        {
         DeleteObj(conv);
         ReturnFalse;
        }
      if(!caConvOutputs.Add(conv.getOutput()) ||
         !caConvGradients.Add(conv.getGradient())
        )
         ReturnFalse;
      total += conv.Neurons();
     }
//---
   CNeuronBaseOCL *comul = new CNeuronBaseOCL();
   if(!comul.Init(0, 0, OpenCL, total, optimization, iBatch))
     {
      DeleteObj(comul);
      ReturnFalse;
     }
   if(!caMLP.Add(comul))
     {
      DeleteObj(comul);
      ReturnFalse;
     }
   if(layers == 1)
     {
      comul.SetOutput(conv.getOutput());
      comul.SetGradient(conv.getGradient());
     }
   conv = new CNeuronConvOCL();
   if(!conv.Init(0, 0, OpenCL, total / i_Variables, total / i_Variables, inputs_count, 1, i_Variables, optimization, iBatch))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   if(!caMLP.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   if(!b_NeedTranspose)
     {
      if(!SetOutput(conv.getOutput()) ||
         !SetGradient(conv.getGradient())
        )
         ReturnFalse;
     }
//---
   CBufferFloat *buf = new CBufferFloat();
   if(!buf)
      ReturnFalse;
   if(!buf.BufferInit(total, 0) || !buf.BufferCreate(OpenCL) ||
      !caTemp.Add(buf))
     {
      DeleteObj(buf);
      ReturnFalse;
     }
   buf = new CBufferFloat();
   if(!buf)
      ReturnFalse;
   if(!buf.BufferInit(total, 0) || !buf.BufferCreate(OpenCL) ||
      !caTemp.Add(buf))
     {
      DeleteObj(buf);
      ReturnFalse;
     }
   buf = new CBufferFloat();
   if(!buf)
      ReturnFalse;
   if(!buf.BufferInit(total, 0) || !buf.BufferCreate(OpenCL) ||
      !caTemp.Add(buf))
     {
      DeleteObj(buf);
      ReturnFalse;
     }
//---
   caConvOutputs.FreeMode(false);
   caConvGradients.FreeMode(false);
   caTranspose.FreeMode(true);
   caConvolutions.FreeMode(true);
   caMLP.FreeMode(true);
   caTemp.FreeMode(true);
   SetActivationFunction(None);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronCSCMOCL feed-forward pass and writes the layer   |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
//---
   if(b_NeedTranspose)
     {
      current = caTranspose.At(0);
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
      inp = current;
     }
//---
   int layers = caConvolutions.Total() - 1;
   for(int l = 0; l <= layers; l++)
     {
      current = caConvolutions.At(l);
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
      inp = current;
     }
//---
   current = caMLP.At(0);
   if(!current)
      ReturnFalse;
   switch(layers)
     {
      case 0:
         ReturnFalse;
      case 1:
         break;
      case 2:
         if(!Concat(caConvOutputs.At(0),
                    caConvOutputs.At(1),
                    current.getOutput(),
                    ia_Windows[1], 1,
                    i_Variables * i_Count))
            ReturnFalse;
         break;
      case 3:
         if(!Concat(caConvOutputs.At(0),
                    caConvOutputs.At(1),
                    caConvOutputs.At(2),
                    current.getOutput(),
                    ia_Windows[1], ia_Windows[2], 1,
                    i_Variables * i_Count))
            ReturnFalse;
         break;
      case 4:
         if(!Concat(caConvOutputs.At(0),
                    caConvOutputs.At(1),
                    caConvOutputs.At(2),
                    caConvOutputs.At(3),
                    current.getOutput(),
                    ia_Windows[1], ia_Windows[2], ia_Windows[3], 1,
                    i_Variables * i_Count))
            ReturnFalse;
         break;
      default:
         if(!Concat(caConvOutputs.At(0),
                    caConvOutputs.At(1),
                    caConvOutputs.At(2),
                    caConvOutputs.At(3),
                    caTemp.At(0),
                    ia_Windows[1], ia_Windows[2], ia_Windows[3], ia_Windows[4],
                    i_Variables * i_Count))
            ReturnFalse;
         break;
     }
   uint last_buf = 0;
   for(int i = 4; i < layers; i += 3)
     {
      uint buf_size = 0;
      for(int j = 1; j <= i; j++)
         buf_size += ia_Windows[j];
      switch(layers - i)
        {
         case 1:
            if(!Concat(caTemp.At(last_buf),
                       caConvOutputs.At(i),
                       current.getOutput(),
                       buf_size, 1,
                       i_Variables * i_Count))
               ReturnFalse;
            break;
         case 2:
            if(!Concat(caTemp.At(last_buf),
                       caConvOutputs.At(i),
                       caConvOutputs.At(i + 1),
                       current.getOutput(),
                       buf_size, ia_Windows[i + 1], 1,
                       i_Variables * i_Count))
               ReturnFalse;
            break;
         case 3:
            if(!Concat(caTemp.At(last_buf),
                       caConvOutputs.At(i),
                       caConvOutputs.At(i + 1),
                       caConvOutputs.At(i + 2),
                       current.getOutput(),
                       buf_size, ia_Windows[i + 1], ia_Windows[i + 2], 1,
                       i_Variables * i_Count))
               ReturnFalse;
            break;
         default:
            if(!Concat(caTemp.At(last_buf),
                       caConvOutputs.At(i),
                       caConvOutputs.At(i + 1),
                       caConvOutputs.At(i + 2),
                       caTemp.At((last_buf + 1) % 2),
                       buf_size, ia_Windows[i + 1], ia_Windows[i + 2], ia_Windows[i + 3],
                       i_Variables * i_Count))
               ReturnFalse;
            break;
        }
      last_buf = (last_buf + 1) % 2;
     }
//---
   inp = current;
   current = caMLP.At(1);
   if(!current ||
      !current.FeedForward(inp))
      ReturnFalse;
//---
   if(b_NeedTranspose)
     {
      inp = current;
      current = caTranspose.At(1);
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronCSCMOCL   |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   CNeuronBaseOCL *current = caMLP.At(0);
   CNeuronBaseOCL *next = caMLP.At(1);
   if(b_NeedTranspose)
     {
      if(!next.CalcHiddenGradients(caTranspose.At(1)))
         ReturnFalse;
     }
   if(!current.CalcHiddenGradients(next.AsObject()))
      ReturnFalse;
   next = current;
//---
   int layers = caConvGradients.Total();
   if(layers == 1)
     {
      next = caConvolutions.At(1);
      if(next.Activation() != None)
        {
         if(!DeActivation(next.getOutput(), next.getGradient(), next.getGradient(), next.Activation()))
            ReturnFalse;
        }
     }
   else
     {
      int prev_window = 0;
      for(int i = 1; i < layers; i++)
         prev_window += int(ia_Windows[i]);
      if(!DeConcat(caTemp.At(0), caConvGradients.At(layers - 1), next.getGradient(), prev_window, 1, i_Variables * i_Count))
         ReturnFalse;
      next = caConvolutions.At(layers);
      int current_buf = 0;
      for(int l = layers; l > 1; l--)
        {
         current = caConvolutions.At(l - 1);
         if(!current.CalcHiddenGradients(next.AsObject()))
            ReturnFalse;
         int window = int(ia_Windows[l - 1]);
         prev_window -= window;
         if(!DeConcat(caTemp.At((current_buf + 1) % 2), caTemp.At(2), caTemp.At(current_buf), prev_window, window, i_Variables * i_Count))
            ReturnFalse;
         if(current.Activation() != None)
           {
            if(!DeActivation(current.getOutput(), caTemp.At(2), caTemp.At(2), current.Activation()))
               ReturnFalse;
           }
         if(!SumAndNormalize(current.getGradient(), caTemp.At(2), current.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
         next = current;
         current_buf = (current_buf + 1) % 2;
        }
     }
   current = caConvolutions.At(0);
   if(!current.CalcHiddenGradients(next.AsObject()))
      ReturnFalse;
   next = current;
//---
   if(b_NeedTranspose)
     {
      current = caTranspose.At(0);
      if(!current.CalcHiddenGradients(next.AsObject()))
         ReturnFalse;
      next = current;
     }
//---
   if(!prevLayer.CalcHiddenGradients(next.AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronCSCMOCL trainable weights from the            |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   CObject *prev = (b_NeedTranspose ? caTranspose.At(0) : NeuronOCL);
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < caConvolutions.Total(); i++)
     {
      current = caConvolutions.At(1);
      if(!current ||
         !current.UpdateInputWeights(prev)
        )
         ReturnFalse;
      prev = current;
     }
   current = caMLP.At(1);
   if(!current ||
      !current.UpdateInputWeights(caMLP.At(0))
     )
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronCSCMOCL state to the file handle.               |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//--- Save constants
   if(FileWriteInteger(file_handle, int(i_Variables)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(i_Count)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(b_NeedTranspose)) < INT_VALUE)
      ReturnFalse;
//--- Objects
   if(!ia_Windows.Save(file_handle))
      ReturnFalse;
   if(b_NeedTranspose)
     {
      for(int i = 0; i < 2; i++)
         if(!caTranspose.At(i).Save(file_handle))
            ReturnFalse;
     }
   int layers = caConvolutions.Total();
   if(FileWriteInteger(file_handle, layers) < INT_VALUE)
      ReturnFalse;
   for(int i = 0; i < layers; i++)
     {
      if(!caConvolutions.At(i).Save(file_handle))
         ReturnFalse;
     }
   for(int i = 0; i < 2; i++)
      if(!caMLP.At(i).Save(file_handle))
         ReturnFalse;
   for(int i = 0; i < 3; i++)
      if(!caTemp.At(i).Save(file_handle))
         ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronCSCMOCL state from the file handle.          |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::Load(const int file_handle)
  {
   CBufferFloat *temp = new CBufferFloat();
   if(!temp)
      ReturnFalse;
   if(!temp.BufferInit(Neurons(), 0) || !temp.BufferCreate(OpenCL) ||
      !SetOutput(temp, false))
     {
      DeleteObj(temp);
      ReturnFalse;
     }
   temp = new CBufferFloat();
   if(!temp)
      ReturnFalse;
   if(!temp.BufferInit(Neurons(), 0) || !temp.BufferCreate(OpenCL) ||
      !SetGradient(temp, false))
     {
      DeleteObj(temp);
      ReturnFalse;
     }
//---
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//--- Load constants
   if(FileIsEnding(file_handle))
      ReturnFalse;
   i_Variables = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   i_Count = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   b_NeedTranspose = bool(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
//--- Objects
   if(!ia_Windows.Load(file_handle))
      ReturnFalse;
   if(b_NeedTranspose)
     {
      caTranspose.Clear();
      CNeuronTransposeOCL *transp = NULL;
      for(int i = 0; i < 2; i++)
        {
         transp = new CNeuronTransposeOCL();
         if(!transp)
            ReturnFalse;
         if(!transp.Init(0, i, OpenCL, 1, 1, optimization, iBatch) ||
            !LoadInsideLayer(file_handle, transp) ||
            !caTranspose.Add(transp))
           {
            DeleteObj(transp);
            ReturnFalse;
           }
        }
     }
   int layers = FileReadInteger(file_handle);
   caConvolutions.Clear();
   caConvOutputs.Clear();
   caConvGradients.Clear();
   CNeuronConvOCL *conv = NULL;
   for(int i = 0; i < layers; i++)
     {
      conv = new CNeuronConvOCL();
      if(!conv)
         ReturnFalse;
      if(!conv.Init(0, i, OpenCL, 1, 1, 1, 1, 1, optimization, iBatch) ||
         !LoadInsideLayer(file_handle, conv) ||
         !caConvolutions.Add(conv)
        )
        {
         DeleteObj(conv);
         ReturnFalse;
        }
      if(i > 0)
        {
         if(!caConvOutputs.Add(conv.getOutput()) ||
            !caConvGradients.Add(conv.getGradient())
           )
            ReturnFalse;
        }
     }
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, caMLP.At(i)))
         ReturnFalse;
   for(int i = 0; i < 3; i++)
      if(!caTemp.At(i).Load(file_handle))
         ReturnFalse;
//---
   CNeuronBaseOCL *neuron = (b_NeedTranspose ? caTranspose.At(1) : caMLP.At(1));
   if(!SetOutput(neuron.getOutput()) ||
      !SetGradient(neuron.getGradient())
     )
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronCSCMOCL GPU buffers to the given OpenCL       |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronCSCMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   CNeuronBaseOCL *current = NULL;
   if(b_NeedTranspose)
      for(int i = 0; i < 2; i++)
        {
         current = caTranspose.At(i);
         current.SetOpenCL(OpenCL);
        }
//---
   for(int i = 0; i < caConvolutions.Total(); i++)
     {
      current = caConvolutions.At(i);
      current.SetOpenCL(OpenCL);
     }
//---
   for(int i = 0; i < 2; i++)
     {
      current = caMLP.At(i);
      current.SetOpenCL(OpenCL);
     }
   for(int i = 0; i < 3; i++)
     {
      CBufferFloat *buf = caTemp.At(i);
      buf.BufferCreate(OpenCL);
     }
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronCSCMOCL weights toward the source model with |
//| factor tau.                                                      |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronCSCMOCL *Source = source;
   if(caConvolutions.Total() != Source.caConvolutions.Total())
      ReturnFalse;
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < caConvolutions.Total(); i++)
     {
      current = caConvolutions.At(i);
      if(!current ||
         !current.WeightsUpdate(Source.caConvolutions.At(i), tau)
        )
         ReturnFalse;
     }
   current = caMLP.At(1);
   if(!current ||
      !current.WeightsUpdate(Source.caMLP.At(1), tau)
     )
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronDiffusion layer and allocates its GPU     |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Init(uint numOutputs,
                            uint myIndex,
                            COpenCLMy * open_cl,
                            uint window,
                            uint window_key,
                            uint heads,
                            uint units_count,
                            uint layers,
                            uint inside_block,
                            ENUM_OPTIMIZATION optimization_type,
                            uint batch
                           )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
   if(!cAddNoise.Init(0, 0, OpenCL, window * units_count, iBatch, optimization))
      ReturnFalse;
   cAddNoise.SetActivationFunction(None);
   if(!cAttention[0].Init(0, 1, OpenCL, window, window_key, heads, units_count, layers, optimization, iBatch))
      ReturnFalse;
   if(!cMergeSplit[0].Init(0, 2, OpenCL, 2 * window, 2 * window, window, (units_count + 1) / 2, optimization, iBatch))
      ReturnFalse;
   if(inside_block > 0)
     {
      CNeuronDiffusion *temp = new CNeuronDiffusion();
      if(!temp)
         ReturnFalse;
      if(!temp.Init(0, 3, OpenCL, window, window_key, heads, (units_count + 1) / 2, layers, inside_block - 1, optimization, iBatch))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
      cNeck = temp;
     }
   else
     {
      CNeuronConvOCL *temp = new CNeuronConvOCL();
      if(!temp)
         ReturnFalse;
      if(!temp.Init(0, 3, OpenCL, window, window, window, (units_count + 1) / 2, optimization, iBatch))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
      cNeck = temp;
     }
   if(!cAttention[1].Init(0, 4, OpenCL, window, window_key, heads, (units_count + 1) / 2, layers, optimization, iBatch))
      ReturnFalse;
   if(!cMergeSplit[1].Init(0, 5, OpenCL, window, window, 2 * window, (units_count + 1) / 2, optimization, iBatch))
      ReturnFalse;
   if(!cResidual.Init(0, 6, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(cMergeSplit[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction((ENUM_ACTIVATION)cMergeSplit[1].Activation());
//---
   if(!cRevIn.Init(0, 7, OpenCL, Neurons(), 0, cAddNoise.AsObject()))
      ReturnFalse;
//---
   if(!SetOutput(cRevIn.getOutput(), true))
      ReturnFalse;
//---
   cRevIn.SetActivationFunction(None);
   SetActivationFunction(None);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronDiffusion feed-forward pass and writes the layer |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cAddNoise.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cAttention[0].FeedForward(cAddNoise.AsObject()))
      ReturnFalse;
   if(!cMergeSplit[0].FeedForward(cAttention[0].AsObject()))
      ReturnFalse;
   if(!cNeck.FeedForward(cMergeSplit[0].AsObject()))
      ReturnFalse;
   if(!cAttention[1].FeedForward(cNeck))
      ReturnFalse;
   if(!cMergeSplit[1].FeedForward(cAttention[1].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cAddNoise.getOutput(), cMergeSplit[1].getOutput(), cResidual.getOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cRevIn.FeedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronDiffusion |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   float error = 1;
   if(!cRevIn.calcOutputGradients(prevLayer.getOutput(), error) ||
      !SumAndNormalize(cRevIn.getGradient(), Gradient, cRevIn.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cResidual.CalcHiddenGradients(cRevIn.AsObject()))
      ReturnFalse;
   if(!cAttention[1].CalcHiddenGradients(cMergeSplit[1].AsObject()))
      ReturnFalse;
   if(!cNeck.CalcHiddenGradients(cAttention[1].AsObject()))
      ReturnFalse;
   if(!cMergeSplit[0].CalcHiddenGradients(cNeck.AsObject()))
      ReturnFalse;
   if(!cAttention[0].CalcHiddenGradients(cMergeSplit[0].AsObject()))
      ReturnFalse;
   if(!cAddNoise.CalcHiddenGradients(cAttention[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cAddNoise.getGradient(), cResidual.getGradient(), cAddNoise.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!prevLayer.CalcHiddenGradients(cAddNoise.AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronDiffusion trainable weights from the          |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cAddNoise.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cAttention[0].UpdateInputWeights(cAddNoise.AsObject()))
      ReturnFalse;
   if(!cMergeSplit[0].UpdateInputWeights(cAttention[0].AsObject()))
      ReturnFalse;
   if(!cNeck.UpdateInputWeights(cMergeSplit[0].AsObject()))
      ReturnFalse;
   if(!cAttention[1].UpdateInputWeights(cNeck))
      ReturnFalse;
   if(!cMergeSplit[1].UpdateInputWeights(cAttention[1].AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronDiffusion GPU buffers to the given OpenCL     |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronDiffusion::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronUShapeAttention::SetOpenCL(obj);
   cAddNoise.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cRevIn.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronDiffusion state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Save(const int file_handle)
  {
   if(!CNeuronUShapeAttention::Save(file_handle))
      ReturnFalse;
   if(!cAddNoise.Save(file_handle))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronDiffusion state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      if(!LoadInsideLayer(file_handle, cAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cMergeSplit[i].AsObject()))
         ReturnFalse;
     }
//---
   int type = FileReadInteger(file_handle);
   if(!!cNeck)
     {
      if(cNeck.Type() != type)
         DeleteObj(cNeck);
     }
//---
   if(!cNeck)
     {
      switch(type)
        {
         case defNeuronUShapeAttention:
            cNeck = new CNeuronUShapeAttention();
            if(!cNeck)
               ReturnFalse;
            break;
         case defNeuronDiffusion:
            cNeck = new CNeuronDiffusion();
            if(!cNeck ||
               !((CNeuronDiffusion*)cNeck).Init(0, 0, OpenCL, 1, 1, 1, 1, 1, 0, ADAM, 1))
               ReturnFalse;
            break;
         case defNeuronConvOCL:
            cNeck = new CNeuronConvOCL();
            if(!cNeck)
               ReturnFalse;
            break;
         default:
            ReturnFalse;
        }
     }
   cNeck.SetOpenCL(OpenCL);
   if(!cNeck.Load(file_handle))
      ReturnFalse;
//---
   if(!cAddNoise.Init(0, 0, OpenCL, Neurons(), iBatch, optimization) ||
      !LoadInsideLayer(file_handle, cAddNoise.AsObject()))
      ReturnFalse;
//---
   if(!cResidual.Init(0, 6, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(cMergeSplit[1].getGradient(), true))
      ReturnFalse;
   if(!cRevIn.Init(0, 7, OpenCL, Neurons(), 0, cAddNoise.AsObject()))
      ReturnFalse;
//---
   if(!SetOutput(cRevIn.getOutput(), true))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHyperProjection layer and allocates its   |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::Init(uint numOutputs,
                                  uint myIndex,
                                  COpenCLMy * open_cl,
                                  uint window,
                                  uint units_count,
                                  ENUM_OPTIMIZATION optimization_type,
                                  uint batch
                                 )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (window + 1)*units_count, optimization_type, batch))
      ReturnFalse;
   iWindow = window;
   iUnits = units_count;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHyperProjection feed-forward pass and writes the |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnits, iWindow};
   uint local_work_size[2] = {1, iWindow};
   int kernel = def_k_HyperProjection;
   ResetLastError();
   setBuffer(kernel, def_k_lp_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_lp_outputs, getOutputIndex())
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHyperProjection layer.                                    |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnits, iWindow};
   int kernel = def_k_HyperProjectionGrad;
   ResetLastError();
   setBuffer(kernel, def_k_lpg_inputs, prevLayer.getOutputIndex())
   setBuffer(kernel, def_k_lpg_inputs_gr, prevLayer.getGradientIndex())
   setBuffer(kernel, def_k_lpg_outputs_gr, getGradientIndex())
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
//| Saves the CNeuronHyperProjection state to the file handle.       |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iWindow)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iUnits)) < INT_VALUE)
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHyperProjection state from the file handle.  |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = (int)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (int)FileReadInteger(file_handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHyperboloids layer and allocates its GPU  |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::Init(uint numOutputs,
                               uint myIndex,
                               COpenCLMy * open_cl,
                               uint window,
                               uint units_count,
                               uint centroids,
                               ENUM_OPTIMIZATION optimization_type,
                               uint batch
                              )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * centroids, optimization_type, batch))
      ReturnFalse;
//---
   iWindows = window;
   iUnits = units_count;
   iCentroids = centroids;
//---
   cHyperCentroids.Clear();
   cHyperCurvatures.Clear();
   cHyperCentroids.SetOpenCL(OpenCL);
   cHyperCurvatures.SetOpenCL(OpenCL);
//---
   CNeuronTransposeOCL *transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, 0, OpenCL, iUnits, iWindows, optimization, iBatch) ||
      !cHyperCentroids.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction(None);
//---
   CNeuronConvOCL *conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, 1, OpenCL, iUnits, iUnits, iCentroids, iWindows, 1, optimization, iBatch) ||
      !cHyperCentroids.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(TANH);
//---
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, 2, OpenCL, iCentroids, iCentroids, iCentroids, 1, iWindows, optimization, iBatch) ||
      !cHyperCentroids.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
//---
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, 3, OpenCL, iWindows, iCentroids, optimization, iBatch) ||
      !cHyperCentroids.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
//---
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, 4, OpenCL, iWindows, iWindows, iWindows, iCentroids, 1, optimization, iBatch) ||
      !cHyperCurvatures.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(TANH);
//---
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, 5, OpenCL, iWindows, iWindows, 1, 1, iCentroids, optimization, iBatch) ||
      !cHyperCurvatures.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
//---
   uint size = iCentroids * iUnits * sizeof(float);
   iProducts = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
   if(iProducts < 0)
      ReturnFalse;
   iDistances = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
   if(iDistances < 0)
      ReturnFalse;
   iNormes = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
   if(iNormes < 0)
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronHyperboloids log map operation.               |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMap(CNeuronBaseOCL* featers,
                                 CNeuronBaseOCL* centroids,
                                 CNeuronBaseOCL* curvatures,
                                 CNeuronBaseOCL* outputs
                                )
  {
   if(!featers || !centroids || !curvatures || !outputs)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iUnits, iCentroids, iWindows};
   uint local_work_size[3] = {1, 1, iWindows};
   int kernel = def_k_LogMap;
   ResetLastError();
   setBuffer(kernel, def_k_logmap_centroids, centroids.getOutputIndex())
   setBuffer(kernel, def_k_logmap_curvatures, curvatures.getOutputIndex())
   setBuffer(kernel, def_k_logmap_features, featers.getOutputIndex())
   setBuffer(kernel, def_k_logmap_outputs, outputs.getOutputIndex())
   setBuffer(kernel, def_k_logmap_distance, iDistances)
   setBuffer(kernel, def_k_logmap_product, iProducts)
   setBuffer(kernel, def_k_logmap_norma, iNormes)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!outputs.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronHyperboloids log map grad operation.          |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMapGrad(CNeuronBaseOCL* featers,
                                     CNeuronBaseOCL* centroids,
                                     CNeuronBaseOCL* curvatures,
                                     CNeuronBaseOCL* outputs
                                    )
  {
   if(!featers || !centroids || !curvatures || !outputs)
      ReturnFalse;
//---
//---
   CBufferFloat *temp = featers.getGradient();
   if(!temp ||
      !temp.Fill(0) ||
      !temp.BufferWrite())
      ReturnFalse;
   temp = centroids.getGradient();
   if(!temp ||
      !temp.Fill(0) ||
      !temp.BufferWrite())
      ReturnFalse;
   temp = curvatures.getGradient();
   if(!temp ||
      !temp.Fill(0) ||
      !temp.BufferWrite())
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iUnits, iCentroids, iWindows};
   uint local_work_size[3] = {1, 1, iWindows};
   int kernel = def_k_LogMapGrad;
   ResetLastError();
   setBuffer(kernel, def_k_logmapgr_centroids, centroids.getOutputIndex())
   setBuffer(kernel, def_k_logmapgr_centroids_gr, centroids.getGradientIndex())
   setBuffer(kernel, def_k_logmapgr_curvatures, curvatures.getOutputIndex())
   setBuffer(kernel, def_k_logmapgr_curvatures_gr, curvatures.getGradientIndex())
   setBuffer(kernel, def_k_logmapgr_features, featers.getOutputIndex())
   setBuffer(kernel, def_k_logmapgr_features_gr, featers.getGradientIndex())
   setBuffer(kernel, def_k_logmapgr_outputs, outputs.getOutputIndex())
   setBuffer(kernel, def_k_logmapgr_outputs_gr, outputs.getGradientIndex())
   setBuffer(kernel, def_k_logmapgr_distance, iDistances)
   setBuffer(kernel, def_k_logmapgr_product, iProducts)
   setBuffer(kernel, def_k_logmapgr_norma, iNormes)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!featers.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronHyperboloids feed-forward pass and writes the    |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *centroids = NULL;
   CNeuronBaseOCL *curvatures = NULL;
//--- Centroids
   for(int i = 0; i < cHyperCentroids.Total(); i++)
     {
      centroids = cHyperCentroids[i];
      if(!centroids ||
         !centroids.FeedForward(prev))
         ReturnFalse;
      prev = centroids;
     }
//--- Curvatures
   for(int i = 0; i < cHyperCurvatures.Total(); i++)
     {
      curvatures = cHyperCurvatures[i];
      if(!curvatures ||
         !curvatures.FeedForward(prev))
         ReturnFalse;
      prev = curvatures;
     }
   if(!LogMap(NeuronOCL, centroids, curvatures, AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronHyperboloids layer.                                       |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   CObject *next = NULL;
   CNeuronBaseOCL *centroids = cHyperCentroids[-1];
   CNeuronBaseOCL *curvatures = cHyperCurvatures[-1];
//---
   if(!LogMapGrad(prevLayer, centroids, curvatures, AsObject()))
      ReturnFalse;
//--- Curvatures
   for(int i = cHyperCurvatures.Total() - 2; i >= 0; i--)
     {
      next = curvatures;
      curvatures = cHyperCurvatures[i];
      if(!curvatures ||
         !curvatures.CalcHiddenGradients(next))
         ReturnFalse;
     }
   CBufferFloat *temp = centroids.getGradient();
   if(centroids.Activation() != None)
      if(!DeActivation(centroids.getOutput(), temp, temp, centroids.Activation()))
         ReturnFalse;
   if(!centroids.SetGradient(centroids.getPrevOutput(), false) ||
      !centroids.CalcHiddenGradients(curvatures.AsObject()) ||
      !SumAndNormalize(temp, centroids.getGradient(), temp, iWindows, false, 0, 0, 0, 1) ||
      !centroids.SetGradient(temp, false)
     )
      ReturnFalse;
//--- Centroids
   for(int i = cHyperCentroids.Total() - 2; i >= 0; i--)
     {
      next = centroids;
      centroids = cHyperCentroids[i];
      if(!centroids ||
         !centroids.CalcHiddenGradients(next))
         ReturnFalse;
     }
//---
   temp = prevLayer.getGradient();
   if(prevLayer.Activation() != None)
      if(!DeActivation(prevLayer.getOutput(), temp, temp, prevLayer.Activation()))
         ReturnFalse;
   if(!prevLayer.SetGradient(prevLayer.getPrevOutput(), false) ||
      !prevLayer.CalcHiddenGradients(centroids.AsObject()) ||
      !SumAndNormalize(temp, prevLayer.getGradient(), temp, iWindows, false, 0, 0, 0, 1) ||
      !prevLayer.SetGradient(temp, false)
     )
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronHyperboloids trainable weights from the       |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *centroids = NULL;
   CNeuronBaseOCL *curvatures = NULL;
//--- Centroids
   for(int i = 0; i < cHyperCentroids.Total(); i++)
     {
      centroids = cHyperCentroids[i];
      if(!centroids ||
         !centroids.UpdateInputWeights(prev))
         ReturnFalse;
      prev = centroids;
     }
//--- Curvatures
   for(int i = 0; i < cHyperCurvatures.Total(); i++)
     {
      curvatures = cHyperCurvatures[i];
      if(!curvatures ||
         !curvatures.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curvatures;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronHyperboloids GPU buffers to the given OpenCL  |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronHyperboloids::SetOpenCL(COpenCLMy * obj)
  {
   if(!!OpenCL)
     {
      OpenCL.BufferFree(iProducts);
      OpenCL.BufferFree(iDistances);
      OpenCL.BufferFree(iNormes);
     }
//---
   CNeuronBaseOCL::SetOpenCL(obj);
   cHyperCentroids.SetOpenCL(OpenCL);
   cHyperCurvatures.SetOpenCL(OpenCL);
//---
   uint size = iCentroids * iUnits * sizeof(float);
   iProducts = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
   iDistances = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
   iNormes = OpenCL.AddBuffer(size, CL_MEM_READ_WRITE);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHyperboloids state to the file handle.          |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cHyperCentroids.Save(file_handle))
      ReturnFalse;
   if(!cHyperCurvatures.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iWindows)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iUnits)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iCentroids)) < INT_VALUE)
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHyperboloids state from the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cHyperCentroids.Load(file_handle))
      ReturnFalse;
   if(!cHyperCurvatures.Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindows = (int)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (int)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCentroids = (int)FileReadInteger(file_handle);
   SetOpenCL(OpenCL);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronHyperboloids layer.         |
//+------------------------------------------------------------------+
void CNeuronHyperboloids::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   CNeuronBaseOCL *neuron = NULL;
//---
   for(int i = 0; i < cHyperCentroids.Total(); i++)
     {
      neuron = cHyperCentroids[i];
      if(!neuron)
         continue;
      neuron.TrainMode(bTrain);
     }
//---
   for(int i = 0; i < cHyperCurvatures.Total(); i++)
     {
      neuron = cHyperCurvatures[i];
      if(!neuron)
         continue;
      neuron.TrainMode(bTrain);
     }
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronHypDiff layer and allocates its GPU       |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronHypDiff::Init(uint numOutputs,
                          uint myIndex,
                          COpenCLMy * open_cl,
                          uint window,
                          uint window_key,
                          uint units_count,
                          uint heads,
                          uint layers,
                          uint centroids,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch
                         )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
//---
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
   int layer = 0;
//--- Projection
   CNeuronHyperProjection *lorenz = new CNeuronHyperProjection();
   if(!lorenz ||
      !lorenz.Init(0, layer, OpenCL, window, units_count, optimization, iBatch) ||
      !cLayers.Add(lorenz))
     {
      DeleteObj(lorenz);
      ReturnFalse;
     }
   layer++;
//--- Encoder
   CNeuronRMAT *rmat = new CNeuronRMAT();
   if(!rmat ||
      !rmat.Init(0, layer, OpenCL, window + 1, window_key, units_count, heads, layers, optimization, iBatch) ||
      !cLayers.Add(rmat))
     {
      DeleteObj(rmat);
      ReturnFalse;
     }
   layer++;
//---
   CNeuronConvOCL *conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, layer, OpenCL, window + 1, window + 1, 2 * window, units_count, 1, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   layer++;
   conv.SetActivationFunction(TANH);
//---
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, layer, OpenCL, 2 * window, 2 * window, 3, units_count, 1, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   layer++;
//--- LogMap projecction
   CNeuronHyperboloids *logmap = new CNeuronHyperboloids();
   if(!logmap ||
      !logmap.Init(0, layer, OpenCL, 3, units_count, centroids, optimization, iBatch) ||
      !cLayers.Add(logmap))
     {
      DeleteObj(logmap);
      ReturnFalse;
     }
   layer++;
//--- Diffusion model
   CNeuronDiffusion *diff = new CNeuronDiffusion();
   if(!diff ||
      !diff.Init(0, layer, OpenCL, 3, window_key, heads, units_count * centroids, 2, layers, optimization, iBatch) ||
      !cLayers.Add(diff))
     {
      DeleteObj(diff);
      ReturnFalse;
     }
   layer++;
//--- Pooling
   CNeuronMHAttentionPooling *pooling = new CNeuronMHAttentionPooling();
   if(!pooling ||
      !pooling.Init(0, layer, OpenCL, 3, units_count, centroids, optimization, iBatch) ||
      !cLayers.Add(pooling))
     {
      DeleteObj(pooling);
      ReturnFalse;
     }
   layer++;
//--- Resize to source size
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, layer, OpenCL, 3, 3, window, units_count, 1, optimization, iBatch) ||
      !cLayers.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//---
   if(!SetOutput(conv.getOutput(), true) ||
      !SetGradient(conv.getGradient(), true))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronPSformer layer and allocates its GPU      |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronPSformer::Init(uint numOutputs,
                           uint myIndex,
                           COpenCLMy * open_cl,
                           uint window,
                           uint units_count,
                           uint segments,
                           float rho,
                           ENUM_OPTIMIZATION optimization_type,
                           uint batch
                          )
  {
   if((window * units_count) % segments > 0)
      ReturnFalse;
   if(!CNeuronBaseSAMOCL::Init(numOutputs, myIndex, open_cl, window * units_count, rho, optimization_type, batch))
      ReturnFalse;
//---
   uint count = Neurons() / segments;
//---
   if(!acTranspose[0].Init(0, 0, OpenCL, segments, count, optimization, iBatch))
      ReturnFalse;
   acTranspose[0].SetActivationFunction(None);
//---
   if(!acPSBlocks[0].Init(0, 1, OpenCL, segments, segments, units_count / segments, 1, fRho, optimization, iBatch))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      if(!acAttention[i].Init(0, i + 2, OpenCL, segments, segments, units_count / segments, 2, optimization, iBatch))
         ReturnFalse;
      if(!acPSBlocks[i + 1].InitPS((CNeuronPSBlock*)acPSBlocks[0].AsObject()))
         ReturnFalse;
     }
//---
   if(!cResidual.Init(0, 4, OpenCL, acAttention[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(acAttention[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction((ENUM_ACTIVATION)acAttention[1].Activation());
//---
   if(!acTranspose[1].Init(0, 5, OpenCL, count, segments, optimization, iBatch))
      ReturnFalse;
   acTranspose[1].SetActivationFunction((ENUM_ACTIVATION)acPSBlocks[2].Activation());
//---
   if(!SetOutput(acTranspose[1].getOutput(), true) ||
      !SetGradient(acTranspose[1].getGradient(), true))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronPSformer feed-forward pass and writes the layer  |
//| output.                                                          |
//+------------------------------------------------------------------+
bool CNeuronPSformer::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Dimension Transformation
   if(!acTranspose[0].FeedForward(NeuronOCL))
      ReturnFalse;
//--- Segment Attention
   CObject* prev = acTranspose[0].AsObject();
   for(int i = 0; i < 2; i++)
     {
      if(!acPSBlocks[i].FeedForward(prev))
         ReturnFalse;
      if(!acAttention[i].FeedForward(acPSBlocks[i].AsObject()))
         ReturnFalse;
      prev = acAttention[i].AsObject();
     }
//--- Residual Add
   if(!SumAndNormalize(acTranspose[0].getOutput(), acAttention[1].getOutput(), cResidual.getOutput(), acAttention[1].GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//--- PS Block
   if(!acPSBlocks[2].FeedForward(cResidual.AsObject()))
      ReturnFalse;
//--- Inverse Transformation
   if(!acTranspose[1].FeedForward(acPSBlocks[2].AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronPSformer  |
//| layer.                                                           |
//+------------------------------------------------------------------+
bool CNeuronPSformer::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!acPSBlocks[2].CalcHiddenGradients(acTranspose[1].AsObject()))
      ReturnFalse;
//---
   if(!cResidual.CalcHiddenGradients(acPSBlocks[2].AsObject()))
      ReturnFalse;
//---
   if(!acPSBlocks[1].CalcHiddenGradients(acAttention[1].AsObject()))
      ReturnFalse;
   if(!acAttention[0].CalcHiddenGradients(acPSBlocks[1].AsObject()))
      ReturnFalse;
   if(!acPSBlocks[0].CalcHiddenGradients(acAttention[0].AsObject()))
      ReturnFalse;
//---
   if(!acTranspose[0].CalcHiddenGradients(acPSBlocks[0].AsObject()))
      ReturnFalse;
   if(acTranspose[0].Activation() == None)
     {
      if(!SumAndNormalize(acTranspose[0].getGradient(), cResidual.getGradient(), acTranspose[0].getGradient(), acAttention[1].GetWindow(), false, 0, 0, 0, 1))
         ReturnFalse;
     }
   else
     {
      if(!DeActivation(acTranspose[0].getOutput(), cResidual.getGradient(), acTranspose[0].getPrevOutput(), acTranspose[0].Activation()) ||
         !SumAndNormalize(acTranspose[0].getGradient(), acTranspose[0].getPrevOutput(), acTranspose[0].getGradient(), acAttention[1].GetWindow(), false, 0, 0, 0, 1))
         ReturnFalse;
     }
   if(!NeuronOCL.CalcHiddenGradients(acTranspose[0].AsObject()))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronPSformer trainable weights from the           |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronPSformer::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!acPSBlocks[2].UpdateInputWeights(cResidual.AsObject()))
      ReturnFalse;
//---
   CObject* prev = acAttention[0].AsObject();
   for(int i = 1; i >= 0; i--)
     {
      if(!acAttention[i].UpdateInputWeights(acPSBlocks[i].AsObject()))
         ReturnFalse;
      if(!acPSBlocks[i].UpdateInputWeights(prev))
         ReturnFalse;
      prev = acTranspose[0].AsObject();
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronPSformer GPU buffers to the given OpenCL      |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronPSformer::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseSAMOCL::SetOpenCL(obj);
   for(int i = 0; i < 2; i++)
     {
      acTranspose[i].SetOpenCL(OpenCL);
      acPSBlocks[i].SetOpenCL(OpenCL);
      acAttention[i].SetOpenCL(OpenCL);
     }
   acPSBlocks[2].SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronPSformer state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronPSformer::Save(const int file_handle)
  {
   if(!CNeuronBaseSAMOCL::Save(file_handle))
      ReturnFalse;
   if(!acPSBlocks[0].Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!acTranspose[i].Save(file_handle) ||
         !acAttention[i].Save(file_handle))
         ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronPSformer state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronPSformer::Load(const int file_handle)
  {
   if(!CNeuronBaseSAMOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, acPSBlocks[0].AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, acTranspose[i].AsObject()) ||
         !LoadInsideLayer(file_handle, acAttention[i].AsObject()))
         ReturnFalse;
//---
   for(int i = 1; i < 3; i++)
      if(!acPSBlocks[i].InitPS((CNeuronPSBlock*)acPSBlocks[0].AsObject()))
         ReturnFalse;
//---
   if(!cResidual.Init(0, 4, OpenCL, acAttention[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(acAttention[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction((ENUM_ACTIVATION)acAttention[1].Activation());
//---
   if(!SetOutput(acTranspose[1].getOutput(), true) ||
      !SetGradient(acTranspose[1].getGradient(), true))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronAdaBN layer and allocates its GPU         |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronAdaBN::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                        uint window, uint window_out, uint units_count,
                        uint & bottlenecks[], uint top_k, uint variables,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables, optimization_type, batch))
      ReturnFalse;
   int index = 0;
   if(!cGates.Init(0, index, OpenCL, window, units_count * variables, bottlenecks.Size(), top_k, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cXProj.Init(0, index, OpenCL, window, window, window, units_count, optimization, iBatch))
      ReturnFalse;
   cXProj.SetActivationFunction(None);
//---
   cExperts.Clear();
   cExperts.SetOpenCL(OpenCL);
   CNeuronConvOCL *conv = NULL;
   CNeuronMultiWindowsConvOCL *mwconv = NULL;
   CNeuronTransposeRCDOCL *transp = NULL;
//---
   uint bn_size = 0;
   for(uint i = 0; i < bottlenecks.Size(); i++)
      bn_size += bottlenecks[i];
//---
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, bn_size, units_count, variables, optimization, iBatch) ||
      !cExperts.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(LReLU);
//---
   index++;
   mwconv = new CNeuronMultiWindowsConvOCL();
   if(!mwconv ||
      !mwconv.Init(0, index, OpenCL, bottlenecks, window_out, units_count, variables, optimization, iBatch) ||
      !cExperts.Add(mwconv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   mwconv.SetActivationFunction(LReLU);
   transp = new CNeuronTransposeRCDOCL();
   index++;
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_count * variables, bottlenecks.Size(), window_out, optimization, iBatch) ||
      !cExperts.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_out, window_out, window, units_count * variables, bottlenecks.Size(), optimization, iBatch) ||
      !cExperts.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
   transp = new CNeuronTransposeRCDOCL();
   index++;
   if(!transp ||
      !transp.Init(0, index, OpenCL, bottlenecks.Size(), units_count * variables, window, optimization, iBatch) ||
      !cExperts.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronTempEmbedding layer and allocates its GPU |
//| buffers.                                                         |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint units, uint window,
                                uint embed_dim1, uint period1, uint frame1,
                                uint embed_dim2, uint period2, uint frame2,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl,
                            (window + embed_dim1 + embed_dim2)*units, optimization_type, batch))
      ReturnFalse;
//---
   if(!caEmbeddings[0].Init(0, 0, OpenCL, embed_dim1 * period1, optimization, iBatch))
      ReturnFalse;
   if(!caEmbeddings[1].Init(0, 1, OpenCL, embed_dim2 * period2, optimization, iBatch))
      ReturnFalse;
//---
   iWindow = window;
   iUnits = units;
   aiEmbeddingDim[0] = embed_dim1;
   aiEmbeddingDim[1] = embed_dim2;
   aiFrames[0] = MathMax(1, frame1);
   aiFrames[1] = MathMax(1, frame2);
   aiPeriod[0] = period1;
   aiPeriod[1] = period2;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronTempEmbedding concat by label operation.      |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::ConcatByLabel(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() < int(iUnits * iWindow) ||
      SecondInput.Total() < (int)iUnits)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iUnits,
                               MathMax(iWindow, aiEmbeddingDim[ArrayMaximum(aiEmbeddingDim)]),
                               3
                              };
   uint kernel = def_k_ConcatByLabel;
   setBuffer(kernel, def_k_cbl_data, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_cbl_label, SecondInput.GetIndex())
   setBuffer(kernel, def_k_cbl_embedding1, caEmbeddings[0].getOutputIndex())
   setBuffer(kernel, def_k_cbl_embedding2, caEmbeddings[1].getOutputIndex())
   setBuffer(kernel, def_k_cbl_output, getOutputIndex())
   setArgument(kernel, def_k_cbl_dimension_data, (int)iWindow)
   setArgument(kernel, def_k_cbl_dimension_emb1, (int)aiEmbeddingDim[0])
   setArgument(kernel, def_k_cbl_dimension_emb2, (int)aiEmbeddingDim[1])
   setArgument(kernel, def_k_cbl_period1, (int)aiPeriod[0])
   setArgument(kernel, def_k_cbl_period2, (int)aiPeriod[1])
   setArgument(kernel, def_k_cbl_frame1, (int)aiFrames[0])
   setArgument(kernel, def_k_cbl_frame2, (int)aiFrames[1])
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronTempEmbedding concat by label grad operation. |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::ConcatByLabelGrad(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() < int(iUnits * iWindow) ||
      SecondInput.Total() < (int)iUnits)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {MathMax(iUnits, aiPeriod[ArrayMaximum(aiPeriod)]),
                               MathMax(iWindow, aiEmbeddingDim[ArrayMaximum(aiEmbeddingDim)]),
                               3
                              };
   uint kernel = def_k_ConcatByLabelGrad;
   setBuffer(kernel, def_k_cbl_data, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_cbl_label, SecondInput.GetIndex())
   setBuffer(kernel, def_k_cbl_embedding1, caEmbeddings[0].getGradientIndex())
   setBuffer(kernel, def_k_cbl_embedding2, caEmbeddings[1].getGradientIndex())
   setBuffer(kernel, def_k_cbl_output, getGradientIndex())
   setArgument(kernel, def_k_cbl_dimension_data, (int)iWindow)
   setArgument(kernel, def_k_cbl_dimension_emb1, (int)aiEmbeddingDim[0])
   setArgument(kernel, def_k_cbl_dimension_emb2, (int)aiEmbeddingDim[1])
   setArgument(kernel, def_k_cbl_period1, (int)aiPeriod[0])
   setArgument(kernel, def_k_cbl_period2, (int)aiPeriod[1])
   setArgument(kernel, def_k_cbl_frame1, (int)aiFrames[0])
   setArgument(kernel, def_k_cbl_frame2, (int)aiFrames[1])
   setArgument(kernel, def_k_cbl_gr_units, (int)iUnits)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronTempEmbedding feed-forward pass and writes the   |
//| layer output.                                                    |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(bTrain)
      for(uint i = 0; i < caEmbeddings.Size(); i++)
         if(!caEmbeddings[i].FeedForward())
            ReturnFalse;
//---
   return(ConcatByLabel(NeuronOCL, SecondInput));
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronTempEmbedding layer.                                      |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!ConcatByLabelGrad(NeuronOCL, SecondInput))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronTempEmbedding trainable weights from the      |
//| accumulated gradients.                                           |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].UpdateInputWeights())
         ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronTempEmbedding weights toward the source      |
//| model with factor tau.                                           |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronTempEmbedding* Source = source;
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].WeightsUpdate(Source.caEmbeddings[i].AsObject(), tau))
         ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronTempEmbedding state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].Save(file_handle))
         ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iWindow) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iUnits) < INT_VALUE)
      ReturnFalse;
   for(uint i = 0; i < caEmbeddings.Size(); i++)
     {
      if(FileWriteInteger(file_handle, (int)aiEmbeddingDim[i]) < INT_VALUE)
         ReturnFalse;
      if(FileWriteInteger(file_handle, (int)aiFrames[i]) < INT_VALUE)
         ReturnFalse;
      if(FileWriteInteger(file_handle, (int)aiPeriod[i]) < INT_VALUE)
         ReturnFalse;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronTempEmbedding state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!LoadInsideLayer(file_handle, caEmbeddings[i].AsObject()))
         ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (uint)FileReadInteger(file_handle);
   for(uint i = 0; i < caEmbeddings.Size(); i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      aiEmbeddingDim[i] = (uint)FileReadInteger(file_handle);
      if(FileIsEnding(file_handle))
         ReturnFalse;
      aiFrames[i] = (uint)FileReadInteger(file_handle);
      if(FileIsEnding(file_handle))
         ReturnFalse;
      aiPeriod[i] = (uint)FileReadInteger(file_handle);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronTempEmbedding GPU buffers to the given OpenCL |
//| context.                                                         |
//+------------------------------------------------------------------+
void CNeuronTempEmbedding::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      caEmbeddings[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Initializes the CNeuronMultiScaleExcitation layer and allocates  |
//| its GPU buffers.                                                 |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint dimension, uint units, uint scales, uint bottleneck,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, dimension * units * scales, optimization_type, batch))
      ReturnFalse;
//---
   cExcitation.Clear();
   cExcitation.SetOpenCL(OpenCL);
   CNeuronSpikeConvBlock*  conv = NULL;
   CNeuronTransposeOCL*    transp = NULL;
   CNeuronSoftMaxOCL*      softmax = NULL;
   uint index = 0;
//--- [Scale, Unit, Dimension]
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, scales, units * dimension, optimization, iBatch) ||
      !cExcitation.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
//--- [Unit, Dimension, Scale]
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, scales, scales, 1, dimension * units, 1, optimization, iBatch) ||
      !cExcitation.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//--- [Unit, Dimension]
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dimension, dimension, bottleneck, units, 1, optimization, iBatch) ||
      !cExcitation.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//--- [Unit, Bottleneck]
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, bottleneck, bottleneck, scales * dimension, units, 1, optimization, iBatch) ||
      !cExcitation.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//--- [Unit, Dimension, Scale]
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, conv.Neurons(), optimization, iBatch) ||
      !cExcitation.Add(softmax))
     {
      DeleteObj(softmax)
      ReturnFalse;
     }
   softmax.SetHeads(dimension * units);
   index++;
//--- [Unit, Dimension, Scale]
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units * dimension, scales, optimization, iBatch) ||
      !cExcitation.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
//--- [Scale, Unit, Dimension]
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the CNeuronMultiScaleExcitation feed-forward pass and       |
//| writes the layer output.                                         |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cExcitation.Total(); i++)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!ElementMult(NeuronOCL.getOutput(), prev.getOutput(), Output))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Back-propagates the input gradients through the                  |
//| CNeuronMultiScaleExcitation layer.                               |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* next = cExcitation[-1];
   CNeuronBaseOCL* curr = NULL;
//---
   if(!NeuronOCL || !next)
      ReturnFalse;
   if(!ElementMultGrad(NeuronOCL.getOutput(), PrevOutput,
                       next.getOutput(), next.getGradient(),
                       Gradient, NeuronOCL.Activation(), next.Activation()))
      ReturnFalse;
//---
   for(int i = cExcitation.Total() - 2; i >= 0; i--)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next) ||
      !SumAndNormalize(NeuronOCL.getGradient(), PrevOutput,
                       NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the CNeuronMultiScaleExcitation trainable weights from   |
//| the accumulated gradients.                                       |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cExcitation.Total(); i++)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Averages the CNeuronMultiScaleExcitation weights toward the      |
//| source model with factor tau.                                    |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMultiScaleExcitation* Source = source;
   if(!cExcitation.WeightsUpdate(Source.cExcitation.AsObject(), tau))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMultiScaleExcitation state to the file handle.  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cExcitation.Save(file_handle))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMultiScaleExcitation state from the file     |
//| handle.                                                          |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cExcitation.Load(file_handle))
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMultiScaleExcitation execution state and its   |
//| GPU buffers.                                                     |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cExcitation.ClearStates())
      ReturnFalse;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronMultiScaleExcitation GPU buffers to the given |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cExcitation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronMultiScaleExcitation layer. |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cExcitation.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//| Sets the sample of the CNeuronBayesianSampler.                   |
//+------------------------------------------------------------------+
bool CNeuronBayesianSampler::SetSample(const vector<float> &sample)
  {
   if(!m_cRandom || m_cRandom.Total() != sample.Size() ||
      !m_cRandom.AssignArray(sample) || !m_cRandom.BufferWrite())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears one retained Bayesian draw without changing its contract. |
//+------------------------------------------------------------------+
bool CNeuronBayesianSampler::Clear(void)
  {
//--- Clear inherited output and gradient buffers.
   if(!CNeuronBaseOCL::Clear())
      ReturnFalseEx("sampler output");
//--- The local reparameterization noise is execution-only state.
   if(!m_cRandom || !m_cRandom.Fill(0.0f))
      ReturnFalseEx("sampler random");
//--- Finalize the sampler scratch reset.
   return(true);
  }
//+------------------------------------------------------------------+
//| Feed the existing VAE kernel from a caller-owned local draw.     |
//+------------------------------------------------------------------+
bool CNeuronBayesianSampler::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || !m_cRandom || NeuronOCL.Neurons() % 2 != 0 ||
      NeuronOCL.Neurons() / 2 != Neurons())
      ReturnFalse;
   return(VAEFeedForward(NeuronOCL, NULL));
  }
//+------------------------------------------------------------------+
//| Runs the existing Bayesian local draw with numeric checking.     |
//+------------------------------------------------------------------+
bool CNeuronBayesianSampler::FeedForwardChecked(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *numeric_flags,
      CBufferFloat *numeric_ones,
      CBufferFloat *numeric_scalar,
      bool &rejected)
  {
   rejected = false;
//--- Preserve the externally prepared local draw without regeneration.
   if(!ValidateCheckedScratch(NeuronOCL, numeric_flags, numeric_ones, numeric_scalar))
      return(false);
//--- Dispatch the caller-supplied draw and reduce its complete output prefix.
   if(!VAEFeedForward(NeuronOCL, numeric_flags))
      return(false);
//--- Finalize the checked Bayesian sampler dispatch.
   return(ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar,
                             (uint)Neurons(), rejected));
  }
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronBayesianLinearOCL layer      |
//| defaults.                                                        |
//+------------------------------------------------------------------+
CNeuronBayesianLinearOCL::CNeuronBayesianLinearOCL(void)
   : iVariables(0),
     iInputs(0),
     iOutputs(0),
     iSamples(0),
     iSampleState(1),
     m_fKLDMult(0.01f),
     bRegularizersEnabled(true),
     bSamplesGradientReady(false),
     bSamplesPrepared(false)
  {
   cRandomSamplers.FreeMode(true);
  }
//+------------------------------------------------------------------+
//| Release the runtime-only posterior samplers                      |
//| (CNeuronBayesianLinearOCL).                                      |
//+------------------------------------------------------------------+
CNeuronBayesianLinearOCL::~CNeuronBayesianLinearOCL(void)
  {
   cRandomSamplers.Clear();
  }
//+------------------------------------------------------------------+
//| Sets the kld mult of the CNeuronBayesianLinearOCL.               |
//+------------------------------------------------------------------+
void CNeuronBayesianLinearOCL::SetKLDMult(float value)
  {
   if(!MathIsValidNumber(value) || value < 0.0f)
      return;
   m_fKLDMult = value;
   cMeanSampler.SetKLDMult(bRegularizersEnabled ? m_fKLDMult : 0.0f);
  }
//+------------------------------------------------------------------+
//| Enables or suppresses the runtime posterior regularizers.        |
//+------------------------------------------------------------------+
void CNeuronBayesianLinearOCL::SetRegularizersEnabled(bool value)
  {
   if(bRegularizersEnabled == value)
      return;
//--- Keep the configured multiplier while changing only its effective runtime value.
   bRegularizersEnabled = value;
   cMeanSampler.SetKLDMult(bRegularizersEnabled ? m_fKLDMult : 0.0f);
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronBayesianLinearOCL create temporary operation. |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::CreateTemporary(void)
  {
//--- Allocate the deterministic posterior and posterior-sample work layers.
   uint w = iVariables * iInputs * iOutputs;
   uint b = iVariables * iOutputs;
   if(!OpenCL)
      ReturnFalseEx("Bayesian temporary: OpenCL context");
   if(!iSamples)
      ReturnFalseEx("Bayesian temporary: sample count");
   if(!cPosterior.Init(0, 0, OpenCL, 2 * (w + b), optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: posterior");
   if(!cSampleWeights.Init(0, 0, OpenCL, w, optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: sampled weights");
   if(!cSampleBias.Init(0, 0, OpenCL, b, optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: sampled bias");
   if(!cSampleOutput.Init(0, 0, OpenCL, b, optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: sampled output");
   if(!cInputGradient.Init(0, 0, OpenCL, iVariables * iInputs, optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: input gradient");
   if(!cPosteriorGradient.Init(0, 0, OpenCL, 2 * (w + b), optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: posterior gradient");
   if(!cMeanSampler.Init(0, 0, OpenCL, w + b, optimization, iBatch))
      ReturnFalseEx("Bayesian temporary: mean sampler");
//--- The production forward path always evaluates the posterior mean.
   cMeanSampler.SetSampleMode(VAE_SAMPLE_MEAN);
   cMeanSampler.SetKLDMult(bRegularizersEnabled ? m_fKLDMult : 0.0f);
   cRandomSamplers.Clear();
//--- Create the stochastic samplers retained exclusively for regularization.
   for(uint i = 0; i < iSamples; i++)
     {
      CNeuronBayesianSampler *s = new CNeuronBayesianSampler();
      if(!s)
         ReturnFalseEx("Bayesian temporary: random sampler allocation");
      if(!s.Init(0, 0, OpenCL, w + b, optimization, iBatch))
        {
         delete s;
         ReturnFalseEx("Bayesian temporary: random sampler initialization");
        }
      if(!cRandomSamplers.Add(s))
        {
         delete s;
         ReturnFalseEx("Bayesian temporary: random sampler registry");
        }
      s.SetKLDMult(0.0f);
     }
//--- Finalize the complete temporary state.
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears Bayesian execution scratch and retained stochastic draws. |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::Clear(void)
  {
//--- No stale sample or gradient cache may survive the public Clear boundary.
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
   if(!CNeuronBaseOCL::Clear())
      ReturnFalseEx("output");
//--- Keep cached posterior Outputs for inference while clearing only gradients.
   CBufferFloat *weights_gradient = cWeights.getGradient();
   if(!weights_gradient || !weights_gradient.Fill(0.0f))
      ReturnFalseEx("weight gradient");
   CBufferFloat *bias_gradient = cBias.getGradient();
   if(!bias_gradient || !bias_gradient.Fill(0.0f))
      ReturnFalseEx("bias gradient");
   if(!cPosterior.Clear())
      ReturnFalseEx("posterior");
   if(!cSampleWeights.Clear())
      ReturnFalseEx("sample weights");
   if(!cSampleBias.Clear())
      ReturnFalseEx("sample bias");
   if(!cSampleOutput.Clear())
      ReturnFalseEx("sample output");
   if(!cInputGradient.Clear())
      ReturnFalseEx("input gradient");
   if(!cPosteriorGradient.Clear())
      ReturnFalseEx("posterior gradient");
   if(!cMeanSampler.Clear())
      ReturnFalseEx("mean sampler");
//--- Random sampler overrides also clear their owned reparameterization noise.
   for(int i = 0; i < cRandomSamplers.Total(); i++)
     {
      CVAE *sampler = cRandomSamplers.At(i);
      if(!sampler || !sampler.Clear())
         ReturnFalseEx("random sampler");
     }
//--- Batch-local buffers are optional and remain allocated for reuse.
   if(cBatchSampleGradient.Total() > 0 && !cBatchSampleGradient.Fill(0.0f))
      ReturnFalseEx("batch sample gradient");
   if(cBatchInputGradient.Total() > 0 && !cBatchInputGradient.Fill(0.0f))
      ReturnFalseEx("batch input gradient");
   if(cBatchInputGradientAccum.Total() > 0 && !cBatchInputGradientAccum.Fill(0.0f))
      ReturnFalseEx("batch input accumulation");
//--- Finalize the retained Bayesian execution-state reset.
   return(true);
  }
//+------------------------------------------------------------------+
//| Create GPU-only accumulation buffers for an internal history.    |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::CreateBatchTemporary(uint rows_per_group)
  {
   if(!OpenCL || rows_per_group == 0)
      ReturnFalseEx("Bayesian batch temporary: arguments");
//--- Allocate GPU-only buffers used to accumulate the sample gradients.
   uint sample_total = iVariables * rows_per_group * iOutputs;
   uint input_total = iVariables * rows_per_group * iInputs;
   if(!cBatchSampleGradient.BufferInit(sample_total, 0.0f))
      ReturnFalseEx("Bayesian batch temporary: sample gradient host buffer");
   if(!cBatchSampleGradient.BufferCreate(OpenCL))
      ReturnFalseEx("Bayesian batch temporary: sample gradient GPU buffer");
   if(!cBatchInputGradient.BufferInit(input_total, 0.0f))
      ReturnFalseEx("Bayesian batch temporary: input gradient host buffer");
   if(!cBatchInputGradient.BufferCreate(OpenCL))
      ReturnFalseEx("Bayesian batch temporary: input gradient GPU buffer");
   if(!cBatchInputGradientAccum.BufferInit(input_total, 0.0f))
      ReturnFalseEx("Bayesian batch temporary: accumulated gradient host buffer");
   if(!cBatchInputGradientAccum.BufferCreate(OpenCL))
      ReturnFalseEx("Bayesian batch temporary: accumulated gradient GPU buffer");
//--- Finalize the batch-local accumulation state.
   return(true);
  }
//+------------------------------------------------------------------+
//| Assemble the posterior parameter tensor (BuildPosterior).        |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::BuildPosterior(void)
  {
//--- Refresh the trainable posterior parameters before concatenation.
   if(bTrain && !cWeights.FeedForward())
      ReturnFalseEx("Bayesian posterior: weights forward");
   if(bTrain && !cBias.FeedForward())
      ReturnFalseEx("Bayesian posterior: bias forward");
//--- Assemble [mu | logvar] for weights and bias.
   if(!Concat(cWeights.getOutput(), cBias.getOutput(), cPosterior.getOutput(),
              iVariables * iInputs * iOutputs, iVariables * iOutputs, 2))
      ReturnFalseEx("Bayesian posterior: parameter concatenation");
//--- Finalize the assembled posterior tensor.
   return(true);
  }
//+------------------------------------------------------------------+
//| Assemble posterior parameters and reduce each checked producer.  |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::BuildPosteriorChecked(CBufferFloat *numeric_flags,
      CBufferFloat *numeric_ones, CBufferFloat *numeric_scalar, bool &rejected)
  {
   rejected = false;
//--- Preserve cached inference parameters while checking each raw posterior slice.
   if(!bTrain)
     {
      const uint weights = iVariables * iInputs * iOutputs;
      const uint bias = iVariables * iOutputs;
      CBufferFloat *cached_weights = cWeights.getOutput();
      CBufferFloat *cached_bias = cBias.getOutput();
      CBufferFloat *posterior = cPosterior.getOutput();
      if(!cached_weights || !cached_bias || !posterior)
         return(false);
      if(!MatMul(numeric_ones, cached_weights, posterior, 1, 1, weights, 1, false,
                 0, 0, 0, numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, weights, rejected))
         return(false);
      if(rejected)
         return(true);
      if(!MatMul(numeric_ones, cached_bias, posterior, 1, 1, bias, 1, false,
                 0, 0, weights, numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, bias, rejected))
         return(false);
      if(rejected)
         return(true);
      if(!MatMul(numeric_ones, cached_weights, posterior, 1, 1, weights, 1, false,
                 0, weights, weights + bias, numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, weights, rejected))
         return(false);
      if(rejected)
         return(true);
      if(!MatMul(numeric_ones, cached_bias, posterior, 1, 1, bias, 1, false,
                 0, bias, 2 * weights + bias, numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, bias, rejected))
         return(false);
      return(true);
     }
//--- Refresh both trainable parameter groups before concatenation.
   if(!cWeights.FeedForwardChecked(numeric_flags, numeric_ones, numeric_scalar, rejected))
      return(false);
   if(rejected)
      return(true);
   if(!cBias.FeedForwardChecked(numeric_flags, numeric_ones, numeric_scalar, rejected))
      return(false);
   if(rejected)
      return(true);
//--- Concatenate the existing posterior layout without changing its ABI.
   if(!Concat(cWeights.getOutput(), cBias.getOutput(), cPosterior.getOutput(),
              iVariables * iInputs * iOutputs, iVariables * iOutputs, 2))
      return(false);
//--- Finalize the checked posterior assembly.
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates caller scratch before any Bayesian checked dispatch.   |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::ValidateCheckedScratch(CBufferFloat *source_buffer, CBufferFloat *output,
      uint required_count, CBufferFloat *numeric_flags, CBufferFloat *numeric_ones,
      CBufferFloat *numeric_scalar)
  {
//--- Validate external operands and the complete caller-owned scratch contract.
   if(!OpenCL || !source_buffer || !output || !numeric_flags || !numeric_ones || !numeric_scalar ||
      required_count == 0 || required_count > INT_MAX || source_buffer == output ||
      source_buffer.GetOpenCL() != OpenCL || output.GetOpenCL() != OpenCL ||
      source_buffer.GetIndex() < 0 || output.GetIndex() < 0 || source_buffer.GetIndex() == output.GetIndex() ||
      numeric_flags.GetOpenCL() != OpenCL ||
      numeric_ones.GetOpenCL() != OpenCL || numeric_scalar.GetOpenCL() != OpenCL ||
      numeric_flags.GetIndex() < 0 || numeric_ones.GetIndex() < 0 || numeric_scalar.GetIndex() < 0 ||
      numeric_flags.GetIndex() == numeric_ones.GetIndex() || numeric_flags.GetIndex() == numeric_scalar.GetIndex() ||
      numeric_ones.GetIndex() == numeric_scalar.GetIndex() || numeric_flags.Total() < int(required_count) ||
      numeric_ones.Total() < int(required_count) || numeric_scalar.Total() != 1)
      return(false);
//--- Reject aliases with every operand that the checked forward can read or overwrite.
   int blocked[] = {source_buffer.GetIndex(), output.GetIndex(), cWeights.getOutput().GetIndex(),
                    cWeights.getWeightsParams().GetIndex(), cBias.getOutput().GetIndex(),
                    cBias.getWeightsParams().GetIndex(), cPosterior.getOutput().GetIndex(),
                    cSampleWeights.getOutput().GetIndex(), cSampleBias.getOutput().GetIndex(),
                    cSampleOutput.getOutput().GetIndex(), cMeanSampler.getOutput().GetIndex()
                   };
   for(int i = 0; i < ArraySize(blocked); i++)
      if(blocked[i] < 0 || numeric_flags.GetIndex() == blocked[i] || numeric_ones.GetIndex() == blocked[i] ||
         numeric_scalar.GetIndex() == blocked[i])
         return(false);
//--- Reject aliases with every retained stochastic sampler output before mutation.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      CBufferFloat *sample_output = (sample ? sample.getOutput() : NULL);
      if(!sample_output || sample_output.GetOpenCL() != OpenCL || sample_output.GetIndex() < 0 ||
         numeric_flags.GetIndex() == sample_output.GetIndex() || numeric_ones.GetIndex() == sample_output.GetIndex() ||
         numeric_scalar.GetIndex() == sample_output.GetIndex())
         return(false);
     }
//--- Finalize the pre-dispatch alias validation.
   return(true);
  }
//+------------------------------------------------------------------+
//| Advance the deterministic sampler noise generator.               |
//+------------------------------------------------------------------+
uint CNeuronBayesianLinearOCL::NextSampleRandom(void)
  {
   iSampleState ^= iSampleState << 13;
   iSampleState ^= iSampleState >> 17;
   iSampleState ^= iSampleState << 5;
   return(iSampleState);
  }
//+------------------------------------------------------------------+
//| Set reparameterization noise for one posterior sampler.          |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::SetSample(CVAE *sampler)
  {
   if(!sampler)
      ReturnFalseEx("Bayesian sample: sampler");
//--- Fill the sampler's reparameterization noise with deterministic pseudo-random normals.
   vector<float> sample = vector<float>::Zeros(iVariables * (iInputs * iOutputs + iOutputs));
   for(uint i = 0; i < sample.Size(); i += 2)
     {
      double u1 = (double(NextSampleRandom()) + 1.0) / 4294967297.0;
      double u2 = (double(NextSampleRandom()) + 1.0) / 4294967297.0;
      double radius = MathSqrt(-2.0 * MathLog(u1));
      sample[i] = (float)(radius * MathCos(2.0 * M_PI * u2));
      if(i + 1 < sample.Size())
         sample[i + 1] = (float)(radius * MathSin(2.0 * M_PI * u2));
     }
   CNeuronBayesianSampler *random_sampler = sampler;
   if(!random_sampler.SetSample(sample))
      ReturnFalseEx("Bayesian sample: reparameterization noise");
//--- The sampler holds the noise for the next posterior draw.
   return(true);
  }
//+------------------------------------------------------------------+
//| Apply one grouped posterior affine transformation.               |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::Affine(CNeuronBaseOCL *n, CVAE *s, CNeuronBaseOCL *r)
  {
   if(!n || !s || !r)
      ReturnFalseEx("Bayesian affine: layer arguments");
//--- Split the sampled posterior into grouped affine weights and bias.
   uint w = iVariables * iInputs * iOutputs;
   uint b = iVariables * iOutputs;
   if(!DeConcat(cSampleWeights.getOutput(), cSampleBias.getOutput(), s.getOutput(), w, b, 1))
      ReturnFalseEx("Bayesian affine: sampled posterior split");
   if(!MatMul(n.getOutput(), cSampleWeights.getOutput(), r.getOutput(), 1, iInputs, iOutputs,
              iVariables, true))
      ReturnFalseEx("Bayesian affine: grouped matrix multiplication");
   if(!SumVecMatrix(cSampleBias.getOutput(), r.getOutput(), r.getOutput(), iOutputs, iVariables))
      ReturnFalseEx("Bayesian affine: bias addition");
//--- The sampled affine output is ready for the caller.
   return(true);
  }
//+------------------------------------------------------------------+
//| Apply one checked grouped posterior affine transformation.       |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::AffineChecked(CNeuronBaseOCL *n, CVAE *s, CNeuronBaseOCL *r,
      CBufferFloat *numeric_flags, CBufferFloat *numeric_ones, CBufferFloat *numeric_scalar,
      bool &rejected)
  {
   rejected = false;
   if(!n || !s || !r)
      return(false);
//--- Read the raw sampled posterior weights directly from its prefix.
   uint w = iVariables * iInputs * iOutputs;
   if(!MatMul(n.getOutput(), s.getOutput(), r.getOutput(), 1, iInputs, iOutputs,
              iVariables, true, 0, 0, 0, numeric_flags))
      return(false);
   if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, (uint)r.Neurons(), rejected))
      return(false);
   if(rejected)
      return(true);
   if(!SumVecMatrix(s.getOutput(), r.getOutput(), r.getOutput(), iOutputs, iVariables,
                    w, 0, 0, 1.0f, 0, numeric_flags))
      return(false);
   if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, (uint)r.Neurons(), rejected))
      return(false);
//--- Finalize the checked affine transformation.
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialize grouped posterior parameters and runtime samplers.    |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::Init(uint no, uint mi, COpenCLMy *oc, uint v, uint in, uint out, uint k,
                                    ENUM_OPTIMIZATION opt, uint batch)
  {
   if(!v || !in || !out || !k)
      ReturnFalseEx("Bayesian init: dimensions");
   if(!CNeuronBaseOCL::Init(no, mi, oc, v * out, opt, batch))
      ReturnFalseEx("Bayesian init: base layer");
//--- Store the grouped affine contract before allocating posterior parameters.
   iVariables = v;
   iInputs = in;
   iOutputs = out;
   iSamples = k;
   iSampleState = 1;
   bRegularizersEnabled = true;
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
   activation = None;
   uint w = v * in * out;
   uint b = v * out;
   if(!cWeights.Init(0, 0, OpenCL, 2 * w, opt, iBatch))
      ReturnFalseEx("Bayesian init: posterior weights");
   if(!cBias.Init(0, 0, OpenCL, 2 * b, opt, iBatch))
      ReturnFalseEx("Bayesian init: posterior bias");
   if(!cWeights.getWeightsParams().Fill(0.0f))
      ReturnFalseEx("Bayesian init: posterior weight initialization");
   if(!cBias.getWeightsParams().Fill(0.0f))
      ReturnFalseEx("Bayesian init: posterior bias initialization");
   if(!cWeights.FeedForward())
      ReturnFalseEx("Bayesian init: posterior weights forward");
   if(!cBias.FeedForward())
      ReturnFalseEx("Bayesian init: posterior bias forward");
   if(!CreateTemporary())
      ReturnFalseEx("Bayesian init: temporary state");
//--- Finalize the initialized posterior layer.
   return(true);
  }
//+------------------------------------------------------------------+
//| Generate deterministic output and optional posterior samples.    |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::feedForward(CNeuronBaseOCL *n)
  {
   if(!n || n.Neurons() != int(iVariables * iInputs))
      ReturnFalseEx("Bayesian forward: input shape");
   if(!BuildPosterior())
      ReturnFalseEx("Bayesian forward: posterior assembly");
//--- Every forward invalidates prior stochastic samples and their gradients.
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
//--- Generate the deterministic production output from the posterior mean.
   CVAE *mean = GetPointer(cMeanSampler);
   if(!mean)
      ReturnFalseEx("Bayesian forward: mean sampler");
   if(!mean.FeedForward(cPosterior.AsObject()))
      ReturnFalseEx("Bayesian forward: mean sample");
   if(!Affine(n, mean, GetPointer(this)))
      ReturnFalseEx("Bayesian forward: mean affine");
   if(!bTrain || !bRegularizersEnabled)
      return(true);
//--- Stochastic states are retained exclusively for regularizers.
//--- The CNet output remains the posterior mean.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      if(!sample)
         ReturnFalseEx("Bayesian forward: random sampler");
      if(!SetSample(sample))
         ReturnFalseEx("Bayesian forward: random sample");
      if(!sample.FeedForward(cPosterior.AsObject()))
         ReturnFalseEx("Bayesian forward: random sampler forward");
     }
//--- Mark the regularizer samples valid for the next backward pass.
   bSamplesPrepared = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Generates checked mean output and retained posterior samples.    |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::FeedForwardChecked(CNeuronBaseOCL *n,
      CBufferFloat *numeric_flags, CBufferFloat *numeric_ones, CBufferFloat *numeric_scalar,
      bool &rejected)
  {
   rejected = false;
   if(!n || !n.getOutput())
      return(false);
//--- Derive bounded checked spans before any sampling or kernel dispatch.
   const ulong source_total = ulong(iVariables) * iInputs;
   const ulong weight_total = ulong(iVariables) * iInputs * iOutputs;
   const ulong bias_total = ulong(iVariables) * iOutputs;
   const ulong posterior_total = 2 * (weight_total + bias_total);
   const ulong output_total = ulong(iVariables) * iOutputs;
   const ulong scratch_total = (posterior_total > output_total ? posterior_total : output_total);
   if(!n || !numeric_flags || !numeric_ones || !numeric_scalar || iVariables == 0 || iInputs == 0 ||
      iOutputs == 0 || iSamples == 0 || weight_total > INT_MAX || bias_total > INT_MAX ||
      posterior_total > INT_MAX || output_total > INT_MAX || scratch_total > INT_MAX || source_total > INT_MAX ||
      n.Neurons() != int(source_total) || n.getOutput().GetOpenCL() != OpenCL ||
      n.getOutput().GetIndex() < 0 || Output.GetIndex() < 0 || numeric_flags.GetOpenCL() != OpenCL ||
      numeric_ones.GetOpenCL() != OpenCL || numeric_scalar.GetOpenCL() != OpenCL ||
      numeric_flags.GetIndex() < 0 || numeric_ones.GetIndex() < 0 || numeric_scalar.GetIndex() < 0 ||
      numeric_flags.GetIndex() == numeric_ones.GetIndex() || numeric_flags.GetIndex() == numeric_scalar.GetIndex() ||
      numeric_ones.GetIndex() == numeric_scalar.GetIndex() || numeric_flags.Total() < int(scratch_total) ||
      numeric_ones.Total() < int(scratch_total) || numeric_scalar.Total() != 1 ||
      numeric_flags.GetIndex() == n.getOutput().GetIndex() || numeric_flags.GetIndex() == Output.GetIndex() ||
      numeric_ones.GetIndex() == n.getOutput().GetIndex() || numeric_ones.GetIndex() == Output.GetIndex() ||
      numeric_scalar.GetIndex() == n.getOutput().GetIndex() || numeric_scalar.GetIndex() == Output.GetIndex() ||
      numeric_ones.GetIndex() == cWeights.getOutput().GetIndex() ||
      numeric_ones.GetIndex() == cBias.getOutput().GetIndex() ||
      numeric_ones.GetIndex() == cPosterior.getOutput().GetIndex() ||
      numeric_ones.GetIndex() == cSampleWeights.getOutput().GetIndex() ||
      numeric_ones.GetIndex() == cSampleBias.getOutput().GetIndex() ||
      numeric_scalar.GetIndex() == cWeights.getOutput().GetIndex() ||
      numeric_scalar.GetIndex() == cBias.getOutput().GetIndex() ||
      numeric_scalar.GetIndex() == cPosterior.getOutput().GetIndex() ||
      numeric_scalar.GetIndex() == cSampleWeights.getOutput().GetIndex() ||
      numeric_scalar.GetIndex() == cSampleBias.getOutput().GetIndex())
      return(false);
   if(!ValidateCheckedScratch(n.getOutput(), Output, (uint)scratch_total, numeric_flags, numeric_ones,
                              numeric_scalar))
      return(false);
//--- Invalidate retained stochastic state until the complete checked forward succeeds.
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
   if(!BuildPosteriorChecked(numeric_flags, numeric_ones, numeric_scalar, rejected))
      return(false);
   if(rejected)
      return(true);
//--- Evaluate the deterministic mean through the checked VAE and affine paths.
   CVAE *mean = GetPointer(cMeanSampler);
   if(!mean || !mean.FeedForwardChecked(cPosterior.AsObject(), numeric_flags, numeric_ones,
                                        numeric_scalar, rejected))
      return(false);
   if(rejected)
      return(true);
   if(!AffineChecked(n, mean, GetPointer(this), numeric_flags, numeric_ones, numeric_scalar, rejected))
      return(false);
   if(rejected || !bTrain || !bRegularizersEnabled)
      return(true);
//--- Prepare each stochastic draw once and retain it for batch reference use.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      if(!sample)
         return(false);
      if(!SetSample(sample))
         return(false);
      if(!sample.FeedForwardChecked(cPosterior.AsObject(), numeric_flags, numeric_ones,
                                    numeric_scalar, rejected))
         return(false);
      if(rejected)
         return(true);
     }
//--- Mark only fully checked stochastic samples prepared for later regularizers.
   bSamplesPrepared = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Backpropagate the deterministic posterior-mean path.             |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::calcInputGradients(CNeuronBaseOCL *n)
  {
   if(!bTrain)
      return(true);
//--- Accumulate the mean affine path into the posterior gradient.
   uint w = iVariables * iInputs * iOutputs;
   uint b = iVariables * iOutputs;
   uint p = w + b;
   if(!n)
      ReturnFalseEx("Bayesian backward: input layer");
   if(!n.getGradient().Fill(0.0f))
      ReturnFalseEx("Bayesian backward: input gradient reset");
   if(!bSamplesGradientReady && !cPosteriorGradient.getGradient().Fill(0.0f))
      ReturnFalseEx("Bayesian backward: posterior gradient reset");
   CVAE *mean = GetPointer(cMeanSampler);
   if(!mean)
      ReturnFalseEx("Bayesian backward: mean sampler");
   if(!DeConcat(cSampleWeights.getOutput(), cSampleBias.getOutput(), mean.getOutput(), w, b, 1))
      ReturnFalseEx("Bayesian backward: mean posterior split");
   if(!CopyBufferRaw(Gradient, cSampleOutput.getGradient(), b))
      ReturnFalseEx("output gradient device copy");
   if(!SumVecMatrixGrad(cSampleBias.getGradient(), cSampleOutput.getGradient(),
                        cSampleOutput.getGradient(), iOutputs, iVariables))
      ReturnFalseEx("Bayesian backward: bias gradient");
   if(!MatMulGrad(n.getOutput(), cInputGradient.getGradient(), cSampleWeights.getOutput(),
                  cSampleWeights.getGradient(), cSampleOutput.getGradient(), 1, iInputs, iOutputs,
                  iVariables, true))
      ReturnFalseEx("Bayesian backward: grouped affine gradient");
   if(!SumAndNormalize(cInputGradient.getGradient(), n.getGradient(), n.getGradient(), iInputs,
                       false, 0, 0, 0, 1.0f))
      ReturnFalseEx("Bayesian backward: input gradient accumulation");
   if(!Concat(cSampleWeights.getGradient(), cSampleBias.getGradient(), mean.getGradient(), w, b, 1))
      ReturnFalseEx("Bayesian backward: mean gradient assembly");
   if(!cPosterior.CalcHiddenGradients(mean))
      ReturnFalseEx("Bayesian backward: mean sampler gradient");
   if(!SumAndNormalize(cPosterior.getGradient(), cPosteriorGradient.getGradient(),
                       cPosteriorGradient.getGradient(), p, false, 0, 0, 0, 1.0f))
      ReturnFalseEx("Bayesian backward: posterior gradient accumulation");
   if(!DeConcat(cWeights.getGradient(), cBias.getGradient(), cPosteriorGradient.getGradient(), w, b, 2))
      ReturnFalseEx("Bayesian backward: posterior gradient split");
//--- Invalidate the stochastic-gradient cache and pass the mean gradient upstream.
   bSamplesGradientReady = false;
   if(n.Activation() != None &&
      !DeActivation(n.getOutput(), n.getGradient(), n.getGradient(), n.Activation()))
      ReturnFalseEx("Bayesian backward: input deactivation");
   return(true);
  }
//+------------------------------------------------------------------+
//| Generate K posterior-affine samples for one internal history.    |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::ForwardSamplesBatch(CBufferFloat *batch_input,
      CBufferFloat *output,
      uint rows_per_group)
  {
//--- Validate the grouped history buffers before using their OpenCL offsets.
   uint w = iVariables * iInputs * iOutputs;
   uint b = iVariables * iOutputs;
   uint sample_total = iVariables * rows_per_group * iOutputs;
   if(!bTrain || !bRegularizersEnabled || !bSamplesPrepared || !batch_input || !output || rows_per_group == 0 ||
      batch_input.GetOpenCL() != OpenCL || output.GetOpenCL() != OpenCL ||
      batch_input.GetIndex() < 0 || output.GetIndex() < 0 ||
      batch_input.Total() != int(iVariables * rows_per_group * iInputs) ||
      output.Total() != int(iSamples * sample_total))
      ReturnFalseEx("Bayesian batch forward: input or output contract");
   if(!BuildPosterior())
      ReturnFalseEx("Bayesian batch forward: posterior assembly");
//--- Evaluate every posterior sample into its strided output segment.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      if(!sample)
         ReturnFalseEx("Bayesian batch forward: random sampler");
      if(!DeConcat(cSampleWeights.getOutput(), cSampleBias.getOutput(), sample.getOutput(), w, b, 1))
         ReturnFalseEx("Bayesian batch forward: sampled posterior split");
      if(!MatMul(batch_input, cSampleWeights.getOutput(), output, rows_per_group, iInputs, iOutputs,
                 iVariables, true, 0, 0, i * sample_total))
         ReturnFalseEx("Bayesian batch forward: grouped matrix multiplication");
      if(!SumVecMatrix(cSampleBias.getOutput(), output, output, iOutputs, iVariables,
                       0, i * sample_total, i * sample_total, 1.0f, rows_per_group))
         ReturnFalseEx("Bayesian batch forward: bias addition");
     }
//--- The next backward pass must derive a fresh stochastic-gradient cache.
   bSamplesGradientReady = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluates retained posterior samples with checked affine output. |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::ForwardSamplesBatchChecked(CBufferFloat *batch_input,
      CBufferFloat *output, uint rows_per_group, CBufferFloat *numeric_flags,
      CBufferFloat *numeric_ones, CBufferFloat *numeric_scalar, bool &rejected)
  {
   rejected = false;
//--- Derive all grouped spans before narrowing OpenCL offsets.
   const ulong input_total = ulong(iVariables) * rows_per_group * iInputs;
   const ulong sample_total = ulong(iVariables) * rows_per_group * iOutputs;
   const ulong output_total = ulong(iSamples) * sample_total;
   const ulong weight_total = ulong(iVariables) * iInputs * iOutputs;
   const ulong bias_total = ulong(iVariables) * iOutputs;
   const ulong parameter_total = 2 * (weight_total + bias_total);
   const ulong scratch_total = (sample_total > parameter_total ? sample_total : parameter_total);
   if(!bTrain || !bRegularizersEnabled || !bSamplesPrepared || !batch_input || !output ||
      !numeric_flags || !numeric_ones || !numeric_scalar || rows_per_group == 0 ||
      input_total > INT_MAX || sample_total > INT_MAX || output_total > INT_MAX ||
      weight_total > INT_MAX || bias_total > INT_MAX || parameter_total > INT_MAX || scratch_total > INT_MAX ||
      batch_input.GetOpenCL() != OpenCL ||
      output.GetOpenCL() != OpenCL || batch_input.GetIndex() < 0 || output.GetIndex() < 0 ||
      batch_input.Total() != int(input_total) || output.Total() != int(output_total) ||
      numeric_flags.GetOpenCL() != OpenCL || numeric_ones.GetOpenCL() != OpenCL ||
      numeric_scalar.GetOpenCL() != OpenCL || numeric_flags.GetIndex() < 0 ||
      numeric_ones.GetIndex() < 0 || numeric_scalar.GetIndex() < 0 ||
      numeric_flags.GetIndex() == numeric_ones.GetIndex() || numeric_flags.GetIndex() == numeric_scalar.GetIndex() ||
      numeric_ones.GetIndex() == numeric_scalar.GetIndex() || numeric_flags.Total() < int(scratch_total) ||
      numeric_ones.Total() < int(scratch_total) || numeric_scalar.Total() != 1 ||
      numeric_flags.GetIndex() == batch_input.GetIndex() || numeric_flags.GetIndex() == output.GetIndex() ||
      numeric_ones.GetIndex() == batch_input.GetIndex() || numeric_ones.GetIndex() == output.GetIndex() ||
      numeric_scalar.GetIndex() == batch_input.GetIndex() || numeric_scalar.GetIndex() == output.GetIndex())
      return(false);
   if(!ValidateCheckedScratch(batch_input, output, (uint)scratch_total, numeric_flags, numeric_ones,
                              numeric_scalar))
      return(false);
   if(!BuildPosteriorChecked(numeric_flags, numeric_ones, numeric_scalar, rejected))
      return(false);
   if(rejected)
     {
      bSamplesPrepared = false;
      return(true);
     }
//--- Reuse the retained draws without regenerating stochastic samples.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      if(!sample)
         return(false);
      if(!MatMul(batch_input, sample.getOutput(), output, int(rows_per_group), int(iInputs),
                 int(iOutputs), int(iVariables), true, 0, 0, (uint)(ulong(i) * sample_total), numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, (uint)sample_total, rejected))
         return(false);
      if(rejected)
        {
         bSamplesPrepared = false;
         return(true);
        }
      if(!SumVecMatrix(sample.getOutput(), output, output, iOutputs, iVariables,
                       (uint)weight_total, (uint)(ulong(i) * sample_total), (uint)(ulong(i) * sample_total),
                       1.0f, rows_per_group, numeric_flags))
         return(false);
      if(!ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar, (uint)sample_total, rejected))
         return(false);
      if(rejected)
        {
         bSamplesPrepared = false;
         return(true);
        }
     }
//--- Invalidate only the gradient cache after a successful checked batch.
   bSamplesGradientReady = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| Accumulate posterior sample gradients for one internal history.  |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::BackwardSamplesBatch(CBufferFloat *batch_input,
      CBufferFloat *sample_gradients,
      uint rows_per_group)
  {
//--- Validate and reset per-history buffers before accumulating all samples.
   uint w = iVariables * iInputs * iOutputs;
   uint b = iVariables * iOutputs;
   uint p = w + b;
   uint sample_total = iVariables * rows_per_group * iOutputs;
   if(!bTrain || !bRegularizersEnabled || !bSamplesPrepared || !batch_input || !sample_gradients || rows_per_group == 0 ||
      batch_input.GetOpenCL() != OpenCL || sample_gradients.GetOpenCL() != OpenCL ||
      batch_input.GetIndex() < 0 || sample_gradients.GetIndex() < 0 ||
      batch_input.Total() != int(iVariables * rows_per_group * iInputs) ||
      sample_gradients.Total() != int(iSamples * sample_total))
      ReturnFalseEx("Bayesian batch backward: input or gradient contract");
   if(!CreateBatchTemporary(rows_per_group))
      ReturnFalseEx("Bayesian batch backward: temporary buffers");
   if(!bSamplesGradientReady && !cPosteriorGradient.getGradient().Fill(0.0f))
      ReturnFalseEx("Bayesian batch backward: posterior gradient reset");
   if(!cBatchInputGradientAccum.Fill(0.0f))
      ReturnFalseEx("Bayesian batch backward: input gradient reset");
//--- Accumulate every random posterior sample into the common posterior gradient.
   for(uint i = 0; i < iSamples; i++)
     {
      CVAE *sample = cRandomSamplers.At(i);
      if(!sample)
         ReturnFalseEx("Bayesian batch backward: random sampler");
      if(!DeConcat(cSampleWeights.getOutput(), cSampleBias.getOutput(), sample.getOutput(), w, b, 1))
         ReturnFalseEx("Bayesian batch backward: sampled posterior split");
      if(!SumVecMatrixGrad(cSampleBias.getGradient(), &cBatchSampleGradient,
                           sample_gradients, iOutputs, iVariables, 0, 0, i * sample_total, 1.0f,
                           rows_per_group))
         ReturnFalseEx("Bayesian batch backward: bias gradient");
      //--- Restore the affine bias sum without scaling the per-row matrix gradient.
      if(rows_per_group > 1 &&
         !SumAndNormalize(cSampleBias.getGradient(), cSampleBias.getGradient(),
                          cSampleBias.getGradient(), iOutputs, false, 0, 0, 0,
                          0.5f * float(rows_per_group)))
         ReturnFalseEx("Bayesian batch backward: bias gradient reduction");
      if(!MatMulGrad(batch_input, &cBatchInputGradient, cSampleWeights.getOutput(),
                     cSampleWeights.getGradient(), &cBatchSampleGradient, rows_per_group,
                     iInputs, iOutputs, iVariables, true))
         ReturnFalseEx("Bayesian batch backward: grouped affine gradient");
      if(!SumAndNormalize(&cBatchInputGradient, &cBatchInputGradientAccum,
                          &cBatchInputGradientAccum, iInputs, false, 0, 0, 0, 1.0f))
         ReturnFalseEx("Bayesian batch backward: input gradient accumulation");
      if(!Concat(cSampleWeights.getGradient(), cSampleBias.getGradient(), sample.getGradient(), w, b, 1))
         ReturnFalseEx("Bayesian batch backward: sampled gradient assembly");
      if(!cPosterior.CalcHiddenGradients(sample))
         ReturnFalseEx("Bayesian batch backward: random sampler gradient");
      //--- Preserve exact affine mean gradients while retaining the VAE log-variance path.
      if(!DeConcat(cSampleWeights.getGradient(), cSampleBias.getGradient(),
                   cSampleWeights.getOutput(), cSampleBias.getOutput(),
                   cPosterior.getGradient(), w, b, w, b, 1))
         ReturnFalseEx("Bayesian batch backward: posterior gradient partition");
      if(!DeConcat(cSampleWeights.getGradient(), cSampleBias.getGradient(),
                   sample.getGradient(), w, b, 1))
         ReturnFalseEx("Bayesian batch backward: affine mean gradient restore");
      if(!Concat(cSampleWeights.getGradient(), cSampleBias.getGradient(),
                 cSampleWeights.getOutput(), cSampleBias.getOutput(),
                 cPosterior.getGradient(), w, b, w, b, 1))
         ReturnFalseEx("Bayesian batch backward: posterior gradient restore");
      if(!SumAndNormalize(cPosterior.getGradient(), cPosteriorGradient.getGradient(),
                          cPosteriorGradient.getGradient(), p, false, 0, 0, 0, 1.0f))
         ReturnFalseEx("Bayesian batch backward: posterior gradient accumulation");
     }
   if(!DeConcat(cWeights.getGradient(), cBias.getGradient(),
                cPosteriorGradient.getGradient(), w, b, 2))
      ReturnFalseEx("Bayesian batch backward: posterior gradient split");
//--- Mark the accumulated sample gradients ready for the parameter update.
   bSamplesGradientReady = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Update posterior parameters after gradient accumulation.         |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::updateInputWeights(CNeuronBaseOCL *n)
  {
   if(!bTrain)
      return(true);
//--- Apply the posterior updates once after all sample gradients are accumulated.
   if(!cWeights.UpdateInputWeights())
      ReturnFalseEx("Bayesian update: posterior weights");
   if(!cBias.UpdateInputWeights())
      ReturnFalseEx("Bayesian update: posterior bias");
//--- Refresh the deterministic caches before switching to inference.
   if(!cWeights.FeedForward())
      ReturnFalseEx("updated posterior weights cache");
   if(!cBias.FeedForward())
      ReturnFalseEx("updated posterior bias cache");
//--- Finalize the inherited update contract.
   return(true);
  }
//+------------------------------------------------------------------+
//| Save posterior parameters and deterministic configuration.       |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::Save(const int h)
  {
//--- Persist only posterior parameters and deterministic layer configuration.
   if(!CNeuronBaseOCL::Save(h))
      ReturnFalseEx("Bayesian save: base layer");
   if(FileWriteInteger(h, iVariables, INT_VALUE) < INT_VALUE)
      ReturnFalseEx("Bayesian save: variables");
   if(FileWriteInteger(h, iInputs, INT_VALUE) < INT_VALUE)
      ReturnFalseEx("Bayesian save: inputs");
   if(FileWriteInteger(h, iOutputs, INT_VALUE) < INT_VALUE)
      ReturnFalseEx("Bayesian save: outputs");
   if(FileWriteInteger(h, iSamples, INT_VALUE) < INT_VALUE)
      ReturnFalseEx("Bayesian save: samples");
   if(FileWriteFloat(h, m_fKLDMult) < sizeof(float))
      ReturnFalseEx("Bayesian save: KLD multiplier");
   if(!cWeights.Save(h))
      ReturnFalseEx("Bayesian save: posterior weights");
   if(!cBias.Save(h))
      ReturnFalseEx("Bayesian save: posterior bias");
//--- Finalize the posterior checkpoint.
   return(true);
  }
//+------------------------------------------------------------------+
//| Load posterior parameters and rebuild transient samplers.        |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::Load(const int h)
  {
   if(!CNeuronBaseOCL::Load(h))
      ReturnFalseEx("Bayesian load: base layer");
//--- Restore the deterministic dimensions and reset runtime-only sampling state.
   iVariables = (uint)FileReadInteger(h, INT_VALUE);
   iInputs = (uint)FileReadInteger(h, INT_VALUE);
   iOutputs = (uint)FileReadInteger(h, INT_VALUE);
   iSamples = (uint)FileReadInteger(h, INT_VALUE);
   m_fKLDMult = FileReadFloat(h);
   iSampleState = 1;
   bRegularizersEnabled = true;
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
   if(!iVariables || !iInputs || !iOutputs || !iSamples)
      ReturnFalseEx("Bayesian load: dimensions");
   if(!MathIsValidNumber(m_fKLDMult) || m_fKLDMult < 0.0f)
      ReturnFalseEx("KLD multiplier is nonfinite or negative");
   if(iSamples > INT_MAX)
      ReturnFalseEx("sample count");
//--- Derive every persisted posterior span with staged signed-buffer bounds.
   const ulong bias_total = (ulong)iVariables * (ulong)iOutputs;
   if(bias_total > ulong(INT_MAX) / 4)
      ReturnFalseEx("bias tensor size");
   const ulong weights_total = bias_total * (ulong)iInputs;
   if(weights_total > ulong(INT_MAX) / 4)
      ReturnFalseEx("weight tensor size");
   const int expected_output           = (int)bias_total;
   const int expected_weight_logical   = (int)(2 * weights_total);
   const int expected_weight_physical  = (int)(4 * weights_total);
   const int expected_bias_logical     = (int)(2 * bias_total);
   const int expected_bias_physical    = (int)(4 * bias_total);
//--- Validate the outer layer before accepting nested parameter owners.
   CBufferFloat *output_buffer   = getOutput();
   CBufferFloat *gradient_buffer = getGradient();
   if(!output_buffer || !gradient_buffer)
      ReturnFalseEx("outer buffers");
   if(output_buffer.Total() != expected_output || gradient_buffer.Total() != expected_output)
      ReturnFalseEx("outer buffer shape");
   if(!LoadInsideLayer(h, cWeights.AsObject()))
      ReturnFalseEx("Bayesian load: posterior weights");
//--- Confirm all serialized weight owners match the restored affine contract.
   CBufferFloat *weights_output   = cWeights.getOutput();
   CBufferFloat *weights_gradient = cWeights.getGradient();
   CBufferFloat *weights_physical = cWeights.getWeightsParams();
   if(!weights_output || !weights_gradient || !weights_physical)
      ReturnFalseEx("weight buffers");
   if(weights_output.Total() != expected_weight_logical ||
      weights_gradient.Total() != expected_weight_logical ||
      weights_physical.Total() != expected_weight_physical)
      ReturnFalseEx("weight buffer shape");
   if(!LoadInsideLayer(h, cBias.AsObject()))
      ReturnFalseEx("Bayesian load: posterior bias");
//--- Confirm all serialized bias owners match the restored affine contract.
   CBufferFloat *bias_output   = cBias.getOutput();
   CBufferFloat *bias_gradient = cBias.getGradient();
   CBufferFloat *bias_physical = cBias.getWeightsParams();
   if(!bias_output || !bias_gradient || !bias_physical)
      ReturnFalseEx("bias buffers");
   if(bias_output.Total() != expected_bias_logical ||
      bias_gradient.Total() != expected_bias_logical ||
      bias_physical.Total() != expected_bias_physical)
      ReturnFalseEx("bias buffer shape");
   if(!CreateTemporary())
      ReturnFalseEx("Bayesian load: temporary state");
//--- Recreate transient samplers; stochastic samples are not serialized.
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds the CNeuronBayesianLinearOCL GPU buffers to the given    |
//| OpenCL context.                                                  |
//+------------------------------------------------------------------+
void CNeuronBayesianLinearOCL::SetOpenCL(COpenCLMy *o)
  {
//--- Rebind every persistent buffer and sampler to the replacement device.
   CNeuronBaseOCL::SetOpenCL(o);
   cWeights.SetOpenCL(OpenCL);
   cBias.SetOpenCL(OpenCL);
   cPosterior.SetOpenCL(OpenCL);
   cSampleWeights.SetOpenCL(OpenCL);
   cSampleBias.SetOpenCL(OpenCL);
   cSampleOutput.SetOpenCL(OpenCL);
   cInputGradient.SetOpenCL(OpenCL);
   cPosteriorGradient.SetOpenCL(OpenCL);
   cMeanSampler.SetOpenCL(OpenCL);
   if(cBatchSampleGradient.Total() > 0)
      cBatchSampleGradient.BufferCreate(OpenCL);
   if(cBatchInputGradient.Total() > 0)
      cBatchInputGradient.BufferCreate(OpenCL);
   if(cBatchInputGradientAccum.Total() > 0)
      cBatchInputGradientAccum.BufferCreate(OpenCL);
   for(int i = 0; i < cRandomSamplers.Total(); i++)
     {
      CVAE *s = cRandomSamplers.At(i);
      if(s)
         s.SetOpenCL(OpenCL);
     }
//--- Existing stochastic gradients cannot cross device contexts.
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
  }
//+------------------------------------------------------------------+
//| Checks checked-transfer support of the complete Bayesian tree.   |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::SupportsOpenCLChecked(void) const
  {
//--- Validate the layer identity and complete stochastic sampler registry.
   if(Type() != defNeuronBayesianLinearOCL || iSamples == 0 ||
      (uint)cRandomSamplers.Total() != iSamples)
      return(false);
//--- Validate every embedded owner before a checked transfer can begin.
   if(!cWeights.SupportsOpenCLChecked() || !cBias.SupportsOpenCLChecked() ||
      !cPosterior.SupportsOpenCLChecked() || !cSampleWeights.SupportsOpenCLChecked() ||
      !cSampleBias.SupportsOpenCLChecked() || !cSampleOutput.SupportsOpenCLChecked() ||
      !cInputGradient.SupportsOpenCLChecked() || !cPosteriorGradient.SupportsOpenCLChecked() ||
      !cMeanSampler.SupportsOpenCLChecked())
      return(false);
   for(int i = 0; i < cRandomSamplers.Total(); i++)
     {
      CObject *object = cRandomSamplers.At(i);
      if(!object || object.Type() != defNeuronVAEOCL)
         return(false);
      CVAE *sampler = object;
      if(!sampler.SupportsOpenCLChecked())
         return(false);
     }
//--- Finalize only when the complete embedded hierarchy is transferable.
   return(true);
  }
//+------------------------------------------------------------------+
//| Transfers Bayesian children before inherited owners publish      |
//| last.                                                            |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::SetOpenCLChecked(COpenCLMy *obj)
  {
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      return(false);
//--- Valid same-context calls also discard stale sampled-gradient state.
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
//--- Transfer each embedded checked owner before the inherited base owner.
   if(!cWeights.SetOpenCLChecked(obj))
      ReturnFalseEx("posterior weights context");
   if(!cBias.SetOpenCLChecked(obj))
      ReturnFalseEx("posterior bias context");
   if(!cPosterior.SetOpenCLChecked(obj))
      ReturnFalseEx("posterior context");
   if(!cSampleWeights.SetOpenCLChecked(obj))
      ReturnFalseEx("sample weights context");
   if(!cSampleBias.SetOpenCLChecked(obj))
      ReturnFalseEx("sample bias context");
   if(!cSampleOutput.SetOpenCLChecked(obj))
      ReturnFalseEx("sample output context");
   if(!cInputGradient.SetOpenCLChecked(obj))
      ReturnFalseEx("input gradient context");
   if(!cPosteriorGradient.SetOpenCLChecked(obj))
      ReturnFalseEx("posterior gradient context");
   if(!cMeanSampler.SetOpenCLChecked(obj))
      ReturnFalseEx("mean sampler context");
//--- Synchronize direct batch owners independently for safe same-B retry.
   if(cBatchSampleGradient.Total() > 0 && cBatchSampleGradient.GetIndex() >= 0 &&
      cBatchSampleGradient.GetOpenCL() != obj && !cBatchSampleGradient.BufferRead())
      ReturnFalseEx("batch sample gradient read");
   if(cBatchSampleGradient.Total() > 0 &&
      (cBatchSampleGradient.GetOpenCL() != obj || cBatchSampleGradient.GetIndex() < 0) &&
      !cBatchSampleGradient.BufferCreate(obj))
      ReturnFalseEx("batch sample gradient create");
   if(cBatchInputGradient.Total() > 0 && cBatchInputGradient.GetIndex() >= 0 &&
      cBatchInputGradient.GetOpenCL() != obj && !cBatchInputGradient.BufferRead())
      ReturnFalseEx("batch input gradient read");
   if(cBatchInputGradient.Total() > 0 &&
      (cBatchInputGradient.GetOpenCL() != obj || cBatchInputGradient.GetIndex() < 0) &&
      !cBatchInputGradient.BufferCreate(obj))
      ReturnFalseEx("batch input gradient create");
   if(cBatchInputGradientAccum.Total() > 0 && cBatchInputGradientAccum.GetIndex() >= 0 &&
      cBatchInputGradientAccum.GetOpenCL() != obj && !cBatchInputGradientAccum.BufferRead())
      ReturnFalseEx("batch input accumulation read");
   if(cBatchInputGradientAccum.Total() > 0 &&
      (cBatchInputGradientAccum.GetOpenCL() != obj || cBatchInputGradientAccum.GetIndex() < 0) &&
      !cBatchInputGradientAccum.BufferCreate(obj))
      ReturnFalseEx("batch input accumulation create");
//--- Transfer every validated stochastic sampler before outer publication.
   for(int i = 0; i < cRandomSamplers.Total(); i++)
     {
      CObject *object = cRandomSamplers.At(i);
      if(!object || object.Type() != defNeuronVAEOCL)
         ReturnFalseEx("random sampler type");
      CVAE *sampler = object;
      if(!sampler.SetOpenCLChecked(obj))
         ReturnFalseEx("random sampler context");
     }
//--- Publish the inherited base owner only after every child succeeded.
   if(!CNeuronBaseOCL::SetOpenCLChecked(obj))
      ReturnFalseEx("inherited base context");
//--- Finalize the checked context adoption after all owners succeeded.
   return(true);
  }
//+------------------------------------------------------------------+
//| Switch posterior parameters between training and inference.      |
//+------------------------------------------------------------------+
void CNeuronBayesianLinearOCL::TrainMode(bool f)
  {
//--- Propagate the mode to posterior parameters and discard stochastic state.
   CNeuronBaseOCL::TrainMode(f);
   cWeights.TrainMode(bTrain);
   cBias.TrainMode(bTrain);
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
  }
//+------------------------------------------------------------------+
//| Copy posterior parameters from a compatible source layer.        |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalseEx("Bayesian weight copy: source layer type");
//--- Synchronize the posterior parameters and invalidate stochastic runtime state.
   CNeuronBayesianLinearOCL *Source = source;
   if(!cWeights.WeightsUpdate(Source.cWeights.AsObject(), tau))
      ReturnFalseEx("Bayesian weight copy: posterior weights");
   if(!cBias.WeightsUpdate(Source.cBias.AsObject(), tau))
      ReturnFalseEx("Bayesian weight copy: posterior bias");
   iSampleState = Source.iSampleState;
   bSamplesGradientReady = false;
   bSamplesPrepared = false;
//--- Finalize the inherited weight-copy contract.
   return(true);
  }
//+------------------------------------------------------------------+
//| Append posterior state to the parameter fingerprint.             |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::AppendParameterFingerprint(ulong &f)
  {
//--- Include the Bayesian type tag before the posterior parameter state.
   f = (f ^ ulong(Type())) * ulong(1099511628211);
   if(!cWeights.AppendParameterFingerprint(f))
      ReturnFalseEx("Bayesian fingerprint: posterior weights");
   if(!cBias.AppendParameterFingerprint(f))
      ReturnFalseEx("Bayesian fingerprint: posterior bias");
//--- Finalize the posterior fingerprint.
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the layer info of the CNeuronBayesianLinearOCL.          |
//+------------------------------------------------------------------+
CLayerDescription *CNeuronBayesianLinearOCL::GetLayerInfo(void)
  {
//--- Report the grouped affine and sampling contract to the graph description.
   CLayerDescription *r = CNeuronBaseOCL::GetLayerInfo();
   if(r)
     {
      r.count = iOutputs;
      r.window = iInputs;
      r.variables = iVariables;
      r.layers = iSamples;
     }
//--- Return the inherited description when its allocation succeeded.
   return(r);
  }
//+------------------------------------------------------------------+