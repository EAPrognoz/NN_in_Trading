#ifndef NEURONET_MISC_MQH
#ifdef NEURONET_BUILDING_FACADE
#define NEURONET_MISC_MQH
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh: Misc
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Init(uint numInputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, int models)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_HGMultiModels, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iModels) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   iModels = FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronConcatenate::CNeuronConcatenate(void) : i_SecondInputs(0)
  {
   ConcWeights = new CBufferFloat();
   ConcDeltaWeights = new CBufferFloat();
   ConcFirstMomentum = new CBufferFloat();
   ConcSecondMomentum = new CBufferFloat;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronConcatenate::~CNeuronConcatenate()
  {
   DeleteObj(ConcWeights);
   DeleteObj(ConcDeltaWeights);
   DeleteObj(ConcFirstMomentum);
   DeleteObj(ConcSecondMomentum);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, uint numInputs1, uint numInputs2, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint & windows[], uint variables, uint inputs_count, bool need_transpose, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, uint inside_block, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   if(!SumAndNormalize(cAddNoise.getOutput(), cMergeSplit[1].getOutput(), cResidual.getOutput(), 1, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!cRevIn.FeedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDiffusion::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronUShapeAttention::SetOpenCL(obj);
   cAddNoise.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cRevIn.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Save(const int file_handle)
  {
   if(!CNeuronUShapeAttention::Save(file_handle))
      ReturnFalse;
   if(!cAddNoise.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (window + 1)*units_count, optimization_type, batch))
      ReturnFalse;
   iWindow = window;
   iUnits = units_count;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint centroids, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMap(CNeuronBaseOCL* featers, CNeuronBaseOCL* centroids, CNeuronBaseOCL* curvatures, CNeuronBaseOCL* outputs)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMapGrad(CNeuronBaseOCL* featers, CNeuronBaseOCL* centroids, CNeuronBaseOCL* curvatures, CNeuronBaseOCL* outputs)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypDiff::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint units_count, uint heads, uint layers, uint centroids, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSformer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint segments, float rho, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(bTrain)
      for(uint i = 0; i < caEmbeddings.Size(); i++)
         if(!caEmbeddings[i].FeedForward())
            ReturnFalse;
//---
   return ConcatByLabel(NeuronOCL, SecondInput);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!ConcatByLabelGrad(NeuronOCL, SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].UpdateInputWeights())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTempEmbedding::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      caEmbeddings[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cExcitation.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cExcitation.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cExcitation.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cExcitation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cExcitation.TrainMode(bTrain);
  }
#else
#define NEURONET_BUILDING_FACADE
#ifndef NEURONET_DEFINITIONS_MQH
#include "NeuroNet_Definitions.mqh"
#endif // NEURONET_DEFINITIONS_MQH
#ifndef NEURONET_DECLARATIONS_MQH
#include "NeuroNet_Declarations.mqh"
#endif // NEURONET_DECLARATIONS_MQH
#include "NeuroNet_Core.mqh"
#include "NeuroNet_RecurrentState.mqh"
#include "NeuroNet_SpikeFlow.mqh"
#include "NeuroNet_Attention.mqh"
#include "NeuroNet_GraphSparse.mqh"
#include "NeuroNet_AgentsRL.mqh"
#include "NeuroNet_TimeSeries.mqh"
#define NEURONET_MISC_MQH
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh: Misc
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Init(uint numInputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, int models)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_HGMultiModels, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iModels) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiModel::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   iModels = FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronConcatenate::CNeuronConcatenate(void) : i_SecondInputs(0)
  {
   ConcWeights = new CBufferFloat();
   ConcDeltaWeights = new CBufferFloat();
   ConcFirstMomentum = new CBufferFloat();
   ConcSecondMomentum = new CBufferFloat;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronConcatenate::~CNeuronConcatenate()
  {
   DeleteObj(ConcWeights);
   DeleteObj(ConcDeltaWeights);
   DeleteObj(ConcFirstMomentum);
   DeleteObj(ConcSecondMomentum);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, uint numInputs1, uint numInputs2, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatenate::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCSCMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint & windows[], uint variables, uint inputs_count, bool need_transpose, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, uint inside_block, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   if(!SumAndNormalize(cAddNoise.getOutput(), cMergeSplit[1].getOutput(), cResidual.getOutput(), 1, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!cRevIn.FeedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDiffusion::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronUShapeAttention::SetOpenCL(obj);
   cAddNoise.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cRevIn.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDiffusion::Save(const int file_handle)
  {
   if(!CNeuronUShapeAttention::Save(file_handle))
      ReturnFalse;
   if(!cAddNoise.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperProjection::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (window + 1)*units_count, optimization_type, batch))
      ReturnFalse;
   iWindow = window;
   iUnits = units_count;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint centroids, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMap(CNeuronBaseOCL* featers, CNeuronBaseOCL* centroids, CNeuronBaseOCL* curvatures, CNeuronBaseOCL* outputs)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperboloids::LogMapGrad(CNeuronBaseOCL* featers, CNeuronBaseOCL* centroids, CNeuronBaseOCL* curvatures, CNeuronBaseOCL* outputs)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypDiff::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint units_count, uint heads, uint layers, uint centroids, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSformer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint segments, float rho, ENUM_OPTIMIZATION optimization_type, uint batch)
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(bTrain)
      for(uint i = 0; i < caEmbeddings.Size(); i++)
         if(!caEmbeddings[i].FeedForward())
            ReturnFalse;
//---
   return ConcatByLabel(NeuronOCL, SecondInput);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!ConcatByLabelGrad(NeuronOCL, SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTempEmbedding::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].UpdateInputWeights())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTempEmbedding::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      caEmbeddings[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
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
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cExcitation.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cExcitation.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleExcitation::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cExcitation.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cExcitation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleExcitation::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cExcitation.TrainMode(bTrain);
  }

#undef NEURONET_BUILDING_FACADE
#endif // NEURONET_BUILDING_FACADE
#endif // NEURONET_MISC_MQH
