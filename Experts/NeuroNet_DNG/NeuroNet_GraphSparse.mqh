//+------------------------------------------------------------------+
//|                                         NeuroNet_GraphSparse.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+----------------------------------------------------------------------------+
//| Logical method implementations generated from  NeuroNet.mqh: GraphSparse.  |
//+----------------------------------------------------------------------------+
//+-------------------------------------------------------------------------+
//| Initializes the CNeuronCGConvOCL layer and allocates its GPU  buffers.  |
//+-------------------------------------------------------------------------+
bool CNeuronCGConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   if(!cInputF.Init(numNeurons, 0, OpenCL, window, optimization, batch))
      ReturnFalse;
   if(!cInputS.Init(numNeurons, 1, OpenCL, window, optimization, batch))
      ReturnFalse;
   cInputF.SetActivationFunction(None);
   cInputS.SetActivationFunction(None);
//---
   if(!cF.Init(0, 2, OpenCL, numNeurons, optimization, batch))
      ReturnFalse;
   cF.SetActivationFunction(SIGMOID);
   if(!cS.Init(0, 3, OpenCL, numNeurons, optimization, batch))
      ReturnFalse;
   cS.SetActivationFunction(LReLU);
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Runs the CNeuronCGConvOCL feed-forward pass and writes the layer  output.  |
//+----------------------------------------------------------------------------+
bool CNeuronCGConvOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput() || NeuronOCL.getOutputIndex() < 0)
      ReturnFalse;
//---
   if(cInputF.getOutputIndex() != NeuronOCL.getOutputIndex())
     {
      if(!cInputF.getOutput().BufferSet(NeuronOCL.getOutputIndex()))
         ReturnFalse;
      cInputF.SetActivationFunction((ENUM_ACTIVATION)NeuronOCL.Activation());
     }
   if(cInputS.getOutputIndex() != NeuronOCL.getOutputIndex())
     {
      if(!cInputS.getOutput().BufferSet(NeuronOCL.getOutputIndex()))
         ReturnFalse;
      cInputS.SetActivationFunction((ENUM_ACTIVATION)NeuronOCL.Activation());
     }
//---
   if(!cF.FeedForward(GetPointer(cInputF)))
      ReturnFalse;
   if(!cS.FeedForward(GetPointer(cInputS)))
      ReturnFalse;
//---
   if(!ElementMult(cF.getOutput(), cS.getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronCGConvOCL  layer.  |
//+---------------------------------------------------------------------------+
bool CNeuronCGConvOCL::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer || !prevLayer.getGradient() || prevLayer.getGradientIndex() < 0)
      ReturnFalse;
//---
   if(!ElementMultGrad(cF.getOutput(), cF.getGradient(), cS.getOutput(), cS.getGradient(), Gradient, cF.Activation(), cS.Activation()))
      ReturnFalse;
//---
   if(!cInputF.CalcHiddenGradients((CObject *)GetPointer(cF)))
      ReturnFalse;
   if(!cInputS.CalcHiddenGradients((CObject *)GetPointer(cS)))
      ReturnFalse;
   if(!SumAndNormalize(cF.getOutput(), cS.getOutput(), prevLayer.getOutput(), 1, false))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Updates the CNeuronCGConvOCL trainable weights from the  accumulated gradients.  |
//+----------------------------------------------------------------------------------+
bool CNeuronCGConvOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cF.UpdateInputWeights(cInputF.AsObject()))
      ReturnFalse;
   if(!cS.UpdateInputWeights(cInputS.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronCGConvOCL state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronCGConvOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cInputF.Save(file_handle))
      ReturnFalse;
   if(!cInputS.Save(file_handle))
      ReturnFalse;
   if(!cF.Save(file_handle))
      ReturnFalse;
   if(!cS.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronCGConvOCL state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronCGConvOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cInputF.Type() || !cInputF.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cInputS.Type() || !cInputS.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cF.Type() || !cF.Load(file_handle))
      ReturnFalse;
   if(FileReadInteger(file_handle) != cS.Type() || !cS.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Averages the CNeuronCGConvOCL weights toward the source model  with factor tau.  |
//+----------------------------------------------------------------------------------+
bool CNeuronCGConvOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cInputF.WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cInputS.WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cF.WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cS.WeightsUpdate(source, tau))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Rebinds the CNeuronCGConvOCL GPU buffers to the given OpenCL  context.  |
//+-------------------------------------------------------------------------+
void CNeuronCGConvOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cInputF.SetOpenCL(OpenCL);
   cInputS.SetOpenCL(OpenCL);
   cF.SetOpenCL(OpenCL);
   cS.SetOpenCL(OpenCL);
  }
//+-------------------------------------------------------------------+
//| Initializes the CNeuronAGCN layer and allocates its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronAGCN::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint units_count, uint dimension,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl,
                            units_count * dimension, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cEa.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cEa.SetActivationFunction(None);
   index++;
   if(!cWx.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cWe.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cWconcat_ex.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cWconcat_ex.SetActivationFunction(None);
   index++;
   if(!cEn.Init(0, index, OpenCL, 2 * dimension, 2 * dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cEn.SetActivationFunction(None);
   index++;
   if(!cEnT.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(GELU);
   index++;
   if(!cAadapt.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cAadapt.SetHeads(units_count);
   index++;
   if(!cAadaptX.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cAadaptX.SetActivationFunction(None);
   index++;
   if(!cApreX.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cApreX.SetActivationFunction(None);
   index++;
   if(!cWadapt.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cWadapt.SetActivationFunction(None);
   index++;
   if(!cWpre.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cWpre.SetActivationFunction(None);
   SetActivationFunction(None);
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Runs the CNeuronAGCN feed-forward pass and writes the layer  output.  |
//+-----------------------------------------------------------------------+
bool CNeuronAGCN::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!SecondInput || SecondInput.Total() < cAadapt.Neurons())
      ReturnFalse;
//---
   if(bTrain)
     {
      if(!cEa.FeedForward())
         ReturnFalse;
      if(!cWe.FeedForward(cEa.AsObject()))
         ReturnFalse;
     }
//---
   if(!cWx.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!Concat(cWx.getOutput(), cWe.getOutput(), cWconcat_ex.getOutput(), cWx.GetWindowOut(), cWe.GetWindowOut(), cWx.GetUnits()))
      ReturnFalse;
   if(!cEn.FeedForward(cWconcat_ex.AsObject()))
      ReturnFalse;
   if(!cEnT.FeedForward(cEn.AsObject()))
      ReturnFalse;
   if(!MatMul(cEn.getOutput(), cEnT.getOutput(), cEnEnT.getOutput(), cEnT.GetCount(), cEnT.GetWindow(), cEnT.GetCount()))
      ReturnFalse;
   if(cEnEnT.Activation() != None)
      if(!Activation(cEnEnT.getOutput(), cEnEnT.getOutput(), cEnEnT.Activation()))
         ReturnFalse;
   if(!cAadapt.FeedForward(cEnEnT.AsObject()))
      ReturnFalse;
   if(!IdentSum(cAadapt.getOutput(), cAadapt.getOutput(), cAadapt.Heads()))
      ReturnFalse;
   if(!MatMul(SecondInput, NeuronOCL.getOutput(), cApreX.getOutput(),
              cWpre.GetUnits(), cWpre.GetUnits(), cWpre.GetWindow()))
      ReturnFalse;
   if(!MatMul(cAadapt.getOutput(), NeuronOCL.getOutput(), cAadaptX.getOutput(),
              cWadapt.GetUnits(), cWadapt.GetUnits(), cWadapt.GetWindow()))
      ReturnFalse;
   if(!cWpre.FeedForward(cApreX.AsObject()))
      ReturnFalse;
   if(!cWadapt.FeedForward(cAadaptX.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cWadapt.getOutput(), cWpre.getOutput(), Output, cWadapt.GetWindowOut(), true))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronAGCN  layer.  |
//+----------------------------------------------------------------------+
bool CNeuronAGCN::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                     CBufferFloat *SecondInput,
                                     CBufferFloat *SecondGradient,
                                     ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient ||
      SecondInput.Total() < cAadapt.Neurons() ||
      SecondGradient.Total() < cAadapt.Neurons())
      ReturnFalse;
//---
   if(!DeActivation(cWadapt.getOutput(), cWadapt.getGradient(), Gradient, cWadapt.Activation()))
      ReturnFalse;
   if(!DeActivation(cWpre.getOutput(), cWpre.getGradient(), Gradient, cWpre.Activation()))
      ReturnFalse;
   if(!cApreX.CalcHiddenGradients(cWpre.AsObject()))
      ReturnFalse;
   if(!cAadaptX.CalcHiddenGradients(cWadapt.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(SecondInput, SecondGradient,
                  NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                  cApreX.getGradient(), cWpre.GetUnits(),
                  cWpre.GetUnits(), cWpre.GetWindow()))
      ReturnFalse;
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient,
                       SecondGradient, SecondActivation))
         ReturnFalse;
//---
   if(!MatMulGrad(cAadapt.getOutput(), cAadapt.getGradient(),
                  NeuronOCL.getOutput(), PrevOutput,
                  cAadaptX.getGradient(),  cWadapt.GetUnits(),
                  cWadapt.GetUnits(), cWadapt.GetWindow()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), PrevOutput,
                       PrevOutput, cWx.GetWindow(), false))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), PrevOutput,
                       PrevOutput, NeuronOCL.Activation()))
         ReturnFalse;
   if(!cEnEnT.CalcHiddenGradients(cAadapt.AsObject()))
      ReturnFalse;
   if(cEnEnT.Activation() != None)
      if(!DeActivation(cEnEnT.getOutput(), cEnEnT.getGradient(),
                       cEnEnT.getGradient(), cEnEnT.Activation()))
         ReturnFalse;
   if(!MatMulGrad(cEn.getOutput(), cEn.getPrevOutput(),
                  cEnT.getOutput(), cEnT.getGradient(),
                  cEnEnT.getGradient(), cEnT.GetCount(),
                  cEnT.GetWindow(), cEnT.GetCount()))
      ReturnFalse;
   if(!cEn.CalcHiddenGradients(cEnT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cEn.getGradient(), cEn.getPrevOutput(),
                       cEn.getGradient(), cEnT.GetWindow(), false))
      ReturnFalse;
   if(cEn.Activation() != None)
      if(!DeActivation(cEn.getOutput(), cEn.getGradient(),
                       cEn.getGradient(), cEn.Activation()))
         ReturnFalse;
   if(!cWconcat_ex.CalcHiddenGradients(cEn.AsObject()))
      ReturnFalse;
   if(!DeConcat(cWx.getGradient(), cWe.getGradient(), cWconcat_ex.getGradient(),
                cWx.GetWindowOut(), cWe.GetWindowOut(), cWx.GetUnits()))
      ReturnFalse;
   if(cWx.Activation() != None)
      if(!DeActivation(cWx.getOutput(), cWx.getGradient(),
                       cWx.getGradient(), cWx.Activation()))
         ReturnFalse;
   if(cWe.Activation() != None)
      if(!DeActivation(cWe.getOutput(), cWe.getGradient(),
                       cWe.getGradient(), cWe.Activation()))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cWx.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), PrevOutput,
                       NeuronOCL.getGradient(), cWx.GetWindow(), false))
      ReturnFalse;
   if(!cEa.CalcHiddenGradients(cWe.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Updates the CNeuronAGCN trainable weights from the accumulated  gradients.  |
//+-----------------------------------------------------------------------------+
bool CNeuronAGCN::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cEa.UpdateInputWeights())
      ReturnFalse;
   if(!cWe.UpdateInputWeights(cEa.AsObject()))
      ReturnFalse;
   if(!cWx.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cEn.UpdateInputWeights(cWconcat_ex.AsObject()))
      ReturnFalse;
   if(!cWpre.UpdateInputWeights(cApreX.AsObject()))
      ReturnFalse;
   if(!cWadapt.UpdateInputWeights(cAadaptX.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Averages the CNeuronAGCN weights toward the source model with  factor tau.  |
//+-----------------------------------------------------------------------------+
bool CNeuronAGCN::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronAGCN* Source = source;
   if(!cEa.WeightsUpdate(Source.cEa.AsObject(), tau))
      ReturnFalse;
   if(!cWe.WeightsUpdate(Source.cWe.AsObject(), tau))
      ReturnFalse;
   if(!cWx.WeightsUpdate(Source.cWx.AsObject(), tau))
      ReturnFalse;
   if(!cEn.WeightsUpdate(Source.cEn.AsObject(), tau))
      ReturnFalse;
   if(!cWpre.WeightsUpdate(Source.cWpre.AsObject(), tau))
      ReturnFalse;
   if(!cWadapt.WeightsUpdate(Source.cWadapt.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronAGCN state to the file handle.                  |
//+------------------------------------------------------------------+
bool CNeuronAGCN::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cEa.Save(file_handle))
      ReturnFalse;
   if(!cWe.Save(file_handle))
      ReturnFalse;
   if(!cWx.Save(file_handle))
      ReturnFalse;
   if(!cEn.Save(file_handle))
      ReturnFalse;
   if(!cWpre.Save(file_handle))
      ReturnFalse;
   if(!cWadapt.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronAGCN state from the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronAGCN::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEa.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWe.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWx.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWpre.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWadapt.AsObject()))
      ReturnFalse;
//---
   uint units_count = GetUnits();
   uint dimension = GetWindow();
//---
   int index = 3;
   if(!cWconcat_ex.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cWconcat_ex.SetActivationFunction(None);
   index += 2;
   if(!cEnT.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(GELU);
   index++;
   if(!cAadapt.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cAadapt.SetHeads(units_count);
   index++;
   if(!cAadaptX.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cAadaptX.SetActivationFunction(None);
   index++;
   if(!cApreX.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cApreX.SetActivationFunction(None);
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Rebinds the CNeuronAGCN GPU buffers to the given OpenCL context.  |
//+-------------------------------------------------------------------+
void CNeuronAGCN::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cEa.SetOpenCL(OpenCL);
   cWe.SetOpenCL(OpenCL);
   cWx.SetOpenCL(OpenCL);
   cEn.SetOpenCL(OpenCL);
   cWpre.SetOpenCL(OpenCL);
   cWadapt.SetOpenCL(OpenCL);
   cWconcat_ex.SetOpenCL(OpenCL);
   cEnT.SetOpenCL(OpenCL);
   cEnEnT.SetOpenCL(OpenCL);
   cAadapt.SetOpenCL(OpenCL);
   cAadaptX.SetOpenCL(OpenCL);
   cApreX.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronAGCN layer.                 |
//+------------------------------------------------------------------+
void CNeuronAGCN::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cEa.TrainMode(bTrain);
   cWe.TrainMode(bTrain);
   cWx.TrainMode(bTrain);
   cEn.TrainMode(bTrain);
   cWpre.TrainMode(bTrain);
   cWadapt.TrainMode(bTrain);
   cWconcat_ex.TrainMode(bTrain);
   cEnT.TrainMode(bTrain);
   cEnEnT.TrainMode(bTrain);
   cAadapt.TrainMode(bTrain);
   cAadaptX.TrainMode(bTrain);
   cApreX.TrainMode(bTrain);
  }
//+-------------------------------------------------------------------------+
//| Initializes the CNeuronGinARCell layer and allocates its GPU  buffers.  |
//+-------------------------------------------------------------------------+
bool CNeuronGinARCell::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint units_count, uint dimension,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl,
                            units_count * dimension, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cX_IA.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cX_AGCN.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cForgetGate.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cResetGate.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cContext.Init(0, index, OpenCL, units_count * dimension, optimization, iBatch))
      ReturnFalse;
   if(!cContext.getPrevOutput().Fill(0))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   bTemp.BufferFree();
   if(!bTemp.BufferInit(units_count * units_count, 0) ||
      !bTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Runs the CNeuronGinARCell feed-forward pass and writes the layer  output.  |
//+----------------------------------------------------------------------------+
bool CNeuronGinARCell::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!cX_IA.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cX_AGCN.FeedForward(cX_IA.AsObject(), SecondInput))
      ReturnFalse;
   if(!cForgetGate.FeedForward(cX_IA.AsObject(), SecondInput))
      ReturnFalse;
   if(!cResetGate.FeedForward(cX_IA.AsObject(), SecondInput))
      ReturnFalse;
   if(!Activation(cForgetGate.getOutput(), cForgetGate.getOutput(), GELU))
      ReturnFalse;
   if(!Activation(cResetGate.getOutput(), cResetGate.getOutput(), GELU))
      ReturnFalse;
//--- Context
   if(!cContext.SwapOutputs())
      ReturnFalse;
   if(!GateElementMult(cContext.getPrevOutput(), cX_AGCN.getOutput(), cForgetGate.getOutput(), cContext.getOutput()))
      ReturnFalse;
//--- Output
   if(!Activation(cContext.getOutput(), cContext.getPrevOutput(), ELU))
      ReturnFalse;
   if(!GateElementMult(cContext.getPrevOutput(), cX_IA.getOutput(), cResetGate.getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronGinARCell  layer.  |
//+---------------------------------------------------------------------------+
bool CNeuronGinARCell::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
//--- Output
   if(!GateElementMultGrad(cContext.getPrevOutput(), cContext.getGradient(),
                           cX_IA.getOutput(), cX_IA.getPrevOutput(),
                           cResetGate.getOutput(), cResetGate.getGradient(),
                           Gradient, ELU, cX_IA.Activation(), GELU))
      ReturnFalse;
//--- Context
   if(!GateElementMultGrad(cContext.getOutput(), cContext.getPrevOutput(),
                           cX_AGCN.getOutput(), cX_AGCN.getGradient(),
                           cForgetGate.getOutput(), cForgetGate.getGradient(),
                           cContext.getGradient(), None,
                           cX_AGCN.Activation(), GELU))
      ReturnFalse;
//--- Gradient to Interposition Attention
   if(!cX_IA.CalcHiddenGradients(cX_AGCN.AsObject(), SecondInput, SecondGradient, SecondActivation) ||
      !SumAndNormalize(cX_IA.getGradient(), cX_IA.getPrevOutput(), cX_IA.getPrevOutput(), cForgetGate.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cX_IA.CalcHiddenGradients(cForgetGate.AsObject(), SecondInput, GetPointer(bTemp), SecondActivation) ||
      !SumAndNormalize(cX_IA.getGradient(), cX_IA.getPrevOutput(), cX_IA.getPrevOutput(), cForgetGate.GetWindow(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(SecondGradient, GetPointer(bTemp), SecondGradient, cForgetGate.GetUnits(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cX_IA.CalcHiddenGradients(cResetGate.AsObject(), SecondInput, GetPointer(bTemp), SecondActivation) ||
      !SumAndNormalize(cX_IA.getGradient(), cX_IA.getPrevOutput(), cX_IA.getPrevOutput(), cForgetGate.GetWindow(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(SecondGradient, GetPointer(bTemp), SecondGradient, cForgetGate.GetUnits(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cX_IA.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Updates the CNeuronGinARCell trainable weights from the  accumulated gradients.  |
//+----------------------------------------------------------------------------------+
bool CNeuronGinARCell::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cX_IA.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cX_AGCN.UpdateInputWeights(cX_IA.AsObject(), second))
      ReturnFalse;
   if(!cForgetGate.UpdateInputWeights(cX_IA.AsObject(), second))
      ReturnFalse;
   if(!cResetGate.UpdateInputWeights(cX_IA.AsObject(), second))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Averages the CNeuronGinARCell weights toward the source model  with factor tau.  |
//+----------------------------------------------------------------------------------+
bool CNeuronGinARCell::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronGinARCell* Source = source;
   if(!cX_IA.WeightsUpdate(Source.cX_IA.AsObject(), tau))
      ReturnFalse;
   if(!cX_AGCN.WeightsUpdate(Source.cX_AGCN.AsObject(), tau))
      ReturnFalse;
   if(!cForgetGate.WeightsUpdate(Source.cForgetGate.AsObject(), tau))
      ReturnFalse;
   if(!cResetGate.WeightsUpdate(Source.cResetGate.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronGinARCell state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronGinARCell::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cX_IA.Save(file_handle))
      ReturnFalse;
   if(!cX_AGCN.Save(file_handle))
      ReturnFalse;
   if(!cForgetGate.Save(file_handle))
      ReturnFalse;
   if(!cResetGate.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronGinARCell state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronGinARCell::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cX_IA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cX_AGCN.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cForgetGate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResetGate.AsObject()))
      ReturnFalse;
//---
   int index = 4;
   if(!cContext.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cContext.getPrevOutput().Fill(0))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   bTemp.BufferFree();
   if(!bTemp.BufferInit(uint(MathPow(cForgetGate.GetUnits(), 2)), 0) ||
      !bTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Rebinds the CNeuronGinARCell GPU buffers to the given OpenCL  context.  |
//+-------------------------------------------------------------------------+
void CNeuronGinARCell::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cX_IA.SetOpenCL(OpenCL);
   cX_AGCN.SetOpenCL(OpenCL);
   cForgetGate.SetOpenCL(OpenCL);
   cResetGate.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   bTemp.BufferFree();
   bTemp.BufferInit(uint(MathPow(cForgetGate.GetUnits(), 2)), 0);
   bTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronGinARCell layer.            |
//+------------------------------------------------------------------+
void CNeuronGinARCell::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cX_IA.TrainMode(bTrain);
   cX_AGCN.TrainMode(bTrain);
   cForgetGate.TrainMode(bTrain);
   cResetGate.SetOpenCL(OpenCL);
  }
//+-------------------------------------------------------------------+
//| Clears the CNeuronGinARCell execution state and its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronGinARCell::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cContext.getPrevOutput() ||
      !cContext.getPrevOutput().Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Initializes the CNeuronGinAR layer and allocates its GPU  buffers.  |
//+---------------------------------------------------------------------+
bool CNeuronGinAR::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint units_count, uint dimension,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSwiGLUOCL::Init(numOutputs, myIndex, open_cl, caCells.Size()*dimension,
                              caCells.Size()*dimension, dimension, units_count, 1,
                              optimization_type, batch))
      ReturnFalse;
   int index = 0;
   if(!cEa.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cEa.SetActivationFunction(None);
   index++;
   if(!cWx.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cWe.Init(0, index, OpenCL, dimension, dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   if(!cWconcat_ex.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cWconcat_ex.SetActivationFunction(None);
   index++;
   if(!cEn.Init(0, index, OpenCL, 2 * dimension, 2 * dimension, dimension, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cEn.SetActivationFunction(None);
   index++;
   if(!cEnT.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(GELU);
   index++;
   if(!cApre.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cApre.SetHeads(units_count);
   for(uint i = 0; i < caCells.Size(); i++)
     {
      index++;
      if(!caCells[i].Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
         ReturnFalse;
     }
   index++;
   if(!cConcat.Init(0, index, OpenCL, GetWindow()*units_count, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Runs the CNeuronGinAR feed-forward pass and writes the layer  output.  |
//+------------------------------------------------------------------------+
bool CNeuronGinAR::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Calculate Apre
   if(bTrain)
     {
      if(!cEa.FeedForward())
         ReturnFalse;
      if(!cWe.FeedForward(cEa.AsObject()))
         ReturnFalse;
     }
//---
   if(!cWx.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!Concat(cWx.getOutput(), cWe.getOutput(), cWconcat_ex.getOutput(), cWx.GetWindowOut(), cWe.GetWindowOut(), cWx.GetUnits()))
      ReturnFalse;
   if(!cEn.FeedForward(cWconcat_ex.AsObject()))
      ReturnFalse;
   if(!cEnT.FeedForward(cEn.AsObject()))
      ReturnFalse;
   if(!MatMul(cEn.getOutput(), cEnT.getOutput(), cEnEnT.getOutput(), cEnT.GetCount(), cEnT.GetWindow(), cEnT.GetCount()))
      ReturnFalse;
   if(cEnEnT.Activation() != None)
      if(!Activation(cEnEnT.getOutput(), cEnEnT.getOutput(), cEnEnT.Activation()))
         ReturnFalse;
   if(!cApre.FeedForward(cEnEnT.AsObject()))
      ReturnFalse;
   if(!IdentSum(cApre.getOutput(), cApre.getOutput(), cApre.Heads()))
      ReturnFalse;
//--- GimAR Cells
   CNeuronBaseOCL *temp = NeuronOCL;
   for(uint i = 0; i < caCells.Size(); i++)
     {
      if(!caCells[i].FeedForward(temp, cApre.getOutput()))
         ReturnFalse;
      temp = caCells[i].AsObject();
     }
//---
   if(!Concat(caCells[0].getOutput(), caCells[1].getOutput(), caCells[2].getOutput(), caCells[3].getOutput(),
              cConcat.getOutput(), GetWindow() / 4, GetWindow() / 4, GetWindow() / 4, GetWindow() / 4, GetUnits()))
      ReturnFalse;
//---
   return CNeuronSwiGLUOCL::feedForward(cConcat.AsObject());
  }
//+-----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronGinAR  layer.  |
//+-----------------------------------------------------------------------+
bool CNeuronGinAR::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSwiGLUOCL::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
   if(!DeConcat(caCells[0].getPrevOutput(), caCells[1].getPrevOutput(),
                caCells[2].getPrevOutput(), caCells[3].getGradient(),
                cConcat.getGradient(), GetWindow() / 4,
                GetWindow() / 4, GetWindow() / 4,
                GetWindow() / 4, GetUnits()))
      ReturnFalse;
//--- GimAR Cells
   cApre.getGradient().Fill(0);
   for(uint i = caCells.Size() - 1; i > 0; i--)
      if(!caCells[i - 1].CalcHiddenGradients(caCells[i].AsObject(), cApre.getOutput(),
                                             cApre.getPrevOutput(),
                                             (ENUM_ACTIVATION)cApre.Activation()) ||
         !SumAndNormalize(caCells[i - 1].getGradient(), caCells[i - 1].getPrevOutput(),
                          caCells[i - 1].getGradient(), cWx.GetWindow(), false, 0, 0, 0, 1) ||
         !SumAndNormalize(cApre.getGradient(), cApre.getPrevOutput(),
                          cApre.getGradient(), GetUnits(), false, 0, 0, 0, 1))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(caCells[0].AsObject(), cApre.getOutput(),
                                     cApre.getPrevOutput(), (ENUM_ACTIVATION)cApre.Activation()) ||
      !SumAndNormalize(cApre.getGradient(), cApre.getPrevOutput(),
                       cApre.getGradient(), GetUnits(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cEnEnT.CalcHiddenGradients(cApre.AsObject()))
      ReturnFalse;
   if(cEnEnT.Activation() != None)
      if(!DeActivation(cEnEnT.getOutput(), cEnEnT.getGradient(),
                       cEnEnT.getGradient(), cEnEnT.Activation()))
         ReturnFalse;
   if(!MatMulGrad(cEn.getOutput(), cEn.getPrevOutput(),
                  cEnT.getOutput(), cEnT.getGradient(),
                  cEnEnT.getGradient(), cEnT.GetCount(),
                  cEnT.GetWindow(), cEnT.GetCount()))
      ReturnFalse;
   if(!cEn.CalcHiddenGradients(cEnT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cEn.getGradient(), cEn.getPrevOutput(),
                       cEn.getGradient(), cEnT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(cEn.Activation() != None)
      if(!DeActivation(cEn.getOutput(), cEn.getGradient(),
                       cEn.getGradient(), cEn.Activation()))
         ReturnFalse;
   if(!cWconcat_ex.CalcHiddenGradients(cEn.AsObject()))
      ReturnFalse;
   if(!DeConcat(cWx.getGradient(), cWe.getGradient(), cWconcat_ex.getGradient(),
                cWx.GetWindowOut(), cWe.GetWindowOut(), cWx.GetUnits()))
      ReturnFalse;
   if(cWx.Activation() != None)
      if(!DeActivation(cWx.getOutput(), cWx.getGradient(),
                       cWx.getGradient(), cWx.Activation()))
         ReturnFalse;
   if(cWe.Activation() != None)
      if(!DeActivation(cWe.getOutput(), cWe.getGradient(),
                       cWe.getGradient(), cWe.Activation()))
         ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cWx.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), temp,
                       NeuronOCL.getGradient(), cWx.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
   if(!cEa.CalcHiddenGradients(cWe.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------+
//| Updates the CNeuronGinAR trainable weights from the accumulated  gradients.  |
//+------------------------------------------------------------------------------+
bool CNeuronGinAR::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//--- Apre
   if(!cEa.UpdateInputWeights())
      ReturnFalse;
   if(!cWe.UpdateInputWeights(cEa.AsObject()))
      ReturnFalse;
//---
   if(!cWx.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cEn.UpdateInputWeights(cWconcat_ex.AsObject()))
      ReturnFalse;
//--- GimAR Cells
   CNeuronBaseOCL *temp = NeuronOCL;
   for(uint i = 0; i < caCells.Size(); i++)
     {
      if(!caCells[i].UpdateInputWeights(temp, cApre.getOutput()))
         ReturnFalse;
      temp = caCells[i].AsObject();
     }
//---
   return CNeuronSwiGLUOCL::updateInputWeights(cConcat.AsObject());
  }
//+------------------------------------------------------------------------------+
//| Averages the CNeuronGinAR weights toward the source model with  factor tau.  |
//+------------------------------------------------------------------------------+
bool CNeuronGinAR::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSwiGLUOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronGinAR* Source = source;
//--- Apre
   if(!cEa.WeightsUpdate(Source.cEa.AsObject(), tau))
      ReturnFalse;
   if(!cWe.WeightsUpdate(Source.cWe.AsObject(), tau))
      ReturnFalse;
   if(!cWx.WeightsUpdate(Source.cWx.AsObject(), tau))
      ReturnFalse;
   if(!cEn.WeightsUpdate(Source.cEn.AsObject(), tau))
      ReturnFalse;
//--- GimAR Cells
   for(uint i = 0; i < caCells.Size(); i++)
      if(!caCells[i].WeightsUpdate(Source.caCells[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronGinAR state to the file handle.                 |
//+------------------------------------------------------------------+
bool CNeuronGinAR::Save(const int file_handle)
  {
   if(!CNeuronSwiGLUOCL::Save(file_handle))
      ReturnFalse;
//--- Apre
   if(!cEa.Save(file_handle))
      ReturnFalse;
   if(!cWe.Save(file_handle))
      ReturnFalse;
   if(!cWx.Save(file_handle))
      ReturnFalse;
   if(!cEn.Save(file_handle))
      ReturnFalse;
//--- GimAR Cells
   for(uint i = 0; i < caCells.Size(); i++)
      if(!caCells[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronGinAR state from the file handle.            |
//+------------------------------------------------------------------+
bool CNeuronGinAR::Load(const int file_handle)
  {
   if(!CNeuronSwiGLUOCL::Load(file_handle))
      ReturnFalse;
//--- Apre
   if(!LoadInsideLayer(file_handle, cEa.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWe.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWx.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEn.AsObject()))
      ReturnFalse;
//--- GimAR Cells
   for(uint i = 0; i < caCells.Size(); i++)
      if(!LoadInsideLayer(file_handle, caCells[i].AsObject()))
         ReturnFalse;
//---
   uint units_count = GetUnits();
   uint dimension = cWx.GetWindow();
   uint index = 3;
   if(!cWconcat_ex.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cWconcat_ex.SetActivationFunction(None);
   index += 2;
   if(!cEnT.Init(0, index, OpenCL, units_count, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(GELU);
   index++;
   if(!cApre.Init(0, index, OpenCL, units_count * units_count, optimization, iBatch))
      ReturnFalse;
   cApre.SetHeads(units_count);
   index += caCells.Size();
   if(!cConcat.Init(0, index, OpenCL, GetWindow()*units_count, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Rebinds the CNeuronGinAR GPU buffers to the given OpenCL  context.  |
//+---------------------------------------------------------------------+
void CNeuronGinAR::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSwiGLUOCL::SetOpenCL(obj);
//--- Apre
   cEa.SetOpenCL(OpenCL);
   cWe.SetOpenCL(OpenCL);
   cWx.SetOpenCL(OpenCL);
   cEn.SetOpenCL(OpenCL);
//--- GimAR Cells
   for(uint i = 0; i < caCells.Size(); i++)
      caCells[i].SetOpenCL(OpenCL);
//---
   cWconcat_ex.SetOpenCL(OpenCL);
   cEnT.SetOpenCL(OpenCL);
   cEnEnT.SetOpenCL(OpenCL);
   cApre.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronGinAR layer.                |
//+------------------------------------------------------------------+
void CNeuronGinAR::TrainMode(bool flag)
  {
   CNeuronSwiGLUOCL::TrainMode(flag);
//--- Apre
   cEa.TrainMode(bTrain);
   cWe.TrainMode(bTrain);
   cWx.TrainMode(bTrain);
   cEn.TrainMode(bTrain);
//--- GimAR Cells
   for(uint i = 0; i < caCells.Size(); i++)
      caCells[i].TrainMode(bTrain);
//---
   cWconcat_ex.TrainMode(bTrain);
   cEnT.TrainMode(bTrain);
   cEnEnT.TrainMode(bTrain);
   cApre.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronGinAR execution state and its GPU buffers.     |
//+------------------------------------------------------------------+
bool CNeuronGinAR::Clear(void)
  {
   if(!CNeuronSwiGLUOCL::Clear())
      ReturnFalse;
   for(uint i = 0; i < caCells.Size(); i++)
      if(!caCells[i].Clear())
         ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Initializes the CNeuronAdaptSpatialNorm layer and allocates its  GPU buffers.  |
//+--------------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint units_count, uint variables,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * variables, optimization_type, batch))
      ReturnFalse;
//---
   iVariables = variables;
   iCount = units_count;
//---
   uint dimension = (iVariables + 1) / 2;
   uint index = 0;
   if(!cEn.Init(0, index, OpenCL, iVariables * dimension, optimization, iBatch))
      ReturnFalse;
   cEn.SetActivationFunction(None);
   index++;
   if(!cEnT.Init(0, index, OpenCL, iVariables, dimension, optimization, iBatch))
      ReturnFalse;
   cEnT.SetActivationFunction(None);
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, iVariables * iVariables, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(None);
   index++;
   if(!cAttan.Init(0, index, OpenCL, iVariables * iVariables, optimization, iBatch))
      ReturnFalse;
   cAttan.SetHeads(iVariables);
//---
   index++;
   if(!cMeanSTDevs.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cMeanSTDevs.SetActivationFunction(None);
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Handles the CNeuronAdaptSpatialNorm adapt spatial norm  operation.  |
//+---------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::AdaptSpatialNorm(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
   uint global_work_offset[3] = { 0 };
   uint global_work_size[] = { iCount, MathMin(iVariables, uint(OpenCL.GetMaxLocalSize(1))), iVariables};
   uint local_work_size[] = { 1, global_work_size[1], 1};
//---
   uint kernel = def_k_AdaptSpatialNorm;
   setBuffer(kernel, def_k_asn_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_asn_mean_stdevs, cMeanSTDevs.getOutputIndex())
   setBuffer(kernel, def_k_asn_attention, cAttan.getOutputIndex())
   setBuffer(kernel, def_k_asn_outputs, getOutputIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Handles the CNeuronAdaptSpatialNorm adapt spatial norm grad  operation.  |
//+--------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::AdaptSpatialNormGrad(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
   uint global_work_offset[3] = { 0 };
   uint global_work_size[] = { MathMax(iVariables, iCount), MathMin(MathMax(iVariables, iCount), uint(OpenCL.GetMaxLocalSize(1))), iVariables};
   uint local_work_size[] = { 1, global_work_size[1], 1};
//---
   uint kernel = def_k_AdaptSpatialNormGrad;
   setBuffer(kernel, def_k_asng_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_asng_inputs_gr, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_asng_mean_stdevs, cMeanSTDevs.getOutputIndex())
   setBuffer(kernel, def_k_asng_mean_stdevs_gr, cMeanSTDevs.getGradientIndex())
   setBuffer(kernel, def_k_asng_attention, cAttan.getOutputIndex())
   setBuffer(kernel, def_k_asng_attention_gr, cAttan.getGradientIndex())
   setBuffer(kernel, def_k_asng_outputs_gr, getGradientIndex())
   setArgument(kernel, def_k_asng_total_inputs, iCount)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Runs the CNeuronAdaptSpatialNorm feed-forward pass and writes  the  layer output.  |
//+------------------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
     {
      if(!cEn.FeedForward())
         ReturnFalse;
      if(!cEnT.FeedForward(cEn.AsObject()))
         ReturnFalse;
      if(!MatMul(cEn.getOutput(), cEnT.getOutput(), cEnEnT.getOutput(),
                 iVariables, cEnT.GetWindow(), iVariables, 1, false))
         ReturnFalse;
      if(!cAttan.FeedForward(cEnEnT.AsObject()))
         ReturnFalse;
     }
//---
   return AdaptSpatialNorm(NeuronOCL);
  }
//+----------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronAdaptSpatialNorm layer.  |
//+----------------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!AdaptSpatialNormGrad(NeuronOCL))
      ReturnFalse;
   if(!cEnEnT.CalcHiddenGradients(cAttan.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(cEn.getOutput(), cEn.getPrevOutput(),
                  cEnT.getOutput(), cEnT.getGradient(),
                  cEnEnT.getGradient(), iVariables,
                  cEnT.GetWindow(), iVariables, 1, false))
      ReturnFalse;
   if(!cEn.CalcHiddenGradients(cEnT.AsObject()) ||
      !SumAndNormalize(cEn.getGradient(), cEn.getPrevOutput(), cEn.getGradient(),
                       cEnT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(cEn.Activation() != None)
      if(!DeActivation(cEn.getOutput(), cEn.getGradient(), cEn.getGradient(), cEn.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------------------+
//| Updates the CNeuronAdaptSpatialNorm trainable weights from the  accumulated gradients.  |
//+-----------------------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   return cEn.UpdateInputWeights();
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronAdaptSpatialNorm state to the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cEn.Save(file_handle))
      ReturnFalse;
   if(!cAttan.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Restores the CNeuronAdaptSpatialNorm state from the file handle.  |
//+-------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttan.AsObject()))
      ReturnFalse;
//---
   iVariables = cAttan.Heads();
   iCount = Neurons() / iVariables;
   uint dimension = cEn.Neurons() / iVariables;
//---
   uint index = 1;
   if(!cEnT.Init(0, index, OpenCL, iVariables, dimension, optimization, iBatch))
      ReturnFalse;
   cEnT.SetActivationFunction(None);
   index++;
   if(!cEnEnT.Init(0, index, OpenCL, iVariables * iVariables, optimization, iBatch))
      ReturnFalse;
   cEnEnT.SetActivationFunction(None);
   index += 2;
   if(!cMeanSTDevs.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cMeanSTDevs.SetActivationFunction(None);
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Rebinds the CNeuronAdaptSpatialNorm GPU buffers to the given  OpenCL context.  |
//+--------------------------------------------------------------------------------+
void CNeuronAdaptSpatialNorm::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cEn.SetOpenCL(OpenCL);
   cEnT.SetOpenCL(OpenCL);
   cEnEnT.SetOpenCL(OpenCL);
   cAttan.SetOpenCL(OpenCL);
   cMeanSTDevs.SetOpenCL(OpenCL);
  }
//+-----------------------------------------------------------------------------------------+
//| Averages the CNeuronAdaptSpatialNorm weights toward the source  model with factor tau.  |
//+-----------------------------------------------------------------------------------------+
bool CNeuronAdaptSpatialNorm::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronAdaptSpatialNorm* Source = source;
   if(!cEn.WeightsUpdate(Source.cEn.AsObject(), tau))
      ReturnFalse;
//---
   if(!cEn.FeedForward())
      ReturnFalse;
   if(!cEnT.FeedForward(cEn.AsObject()))
      ReturnFalse;
   if(!MatMul(cEn.getOutput(), cEnT.getOutput(), cEnEnT.getOutput(),
              iVariables, cEnT.GetWindow(), iVariables, 1, false))
      ReturnFalse;
   if(!cAttan.FeedForward(cEnEnT.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Initializes the CNeuronSCNNEncoder layer and allocates its GPU  buffers.  |
//+---------------------------------------------------------------------------+
bool CNeuronSCNNEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                              uint units_count, uint variables, uint forecast,
                              uint season_period, uint short_period,
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, units_count + forecast, variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   uint index = 0;
   if(!cLongNorm.Init(0, index, OpenCL, 1, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cLongNorm.SetActivationFunction(None);
   index++;
   if(!cSeasonTransp.Init(0, index, OpenCL, variables, units_count / season_period, season_period, optimization, iBatch))
      ReturnFalse;
   cSeasonTransp.SetActivationFunction(None);
   index++;
   if(!cSeasonNorm.Init(0, index, OpenCL, cSeasonTransp.GetCount(), season_period, variables, optimization, iBatch))
      ReturnFalse;
   cSeasonNorm.SetActivationFunction(None);
   index++;
   if(!cUnSeasonTransp.Init(0, index, OpenCL, variables, season_period, cSeasonTransp.GetCount(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cShortNorm.Init(0, index, OpenCL, units_count / short_period, short_period, variables, optimization, iBatch))
      ReturnFalse;
   cSeasonNorm.SetActivationFunction(None);
   index++;
   if(!cAdaptSpatNorm.Init(0, index, OpenCL, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cAdaptSpatNorm.SetActivationFunction(None);
   index++;
   uint concatSize = units_count * variables;                                      //inputs
   concatSize += cLongNorm.Neurons() + cLongNorm.GetMeanSTDevs().Neurons();         // long term
   concatSize += cSeasonNorm.Neurons() + cSeasonNorm.GetMeanSTDevs().Neurons();     // seasons
   concatSize += cShortNorm.Neurons() + cShortNorm.GetMeanSTDevs().Neurons();       // short term
   concatSize += cAdaptSpatNorm.Neurons() + cAdaptSpatNorm.GetMeanSTDevs().Neurons(); // spatial
   if(!cConcatenated.Init(0, index, OpenCL, concatSize, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
   index++;
   if(!cProjection.Init(0, index, OpenCL, concatSize / variables, concatSize / variables, units_count + forecast, 1, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose.Init(0, index, OpenCL, variables, units_count + forecast, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!caFusion[0].Init(0, index, OpenCL, variables, variables, variables, units_count + forecast, optimization, iBatch))
      ReturnFalse;
   caFusion[0].SetActivationFunction(TANH);
   index++;
   if(!caFusion[1].Init(0, index, OpenCL, variables, variables, variables, units_count + forecast, optimization, iBatch))
      ReturnFalse;
   caFusion[1].SetActivationFunction(SIGMOID);
   index++;
   if(!cFusionOut.Init(0, index, OpenCL, caFusion[0].Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------+
//| Runs the CNeuronSCNNEncoder feed-forward pass and writes the  layer output.  |
//+------------------------------------------------------------------------------+
bool CNeuronSCNNEncoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cLongNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSeasonTransp.FeedForward(cLongNorm.AsObject()))
      ReturnFalse;
   if(!cSeasonNorm.FeedForward(cSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cUnSeasonTransp.FeedForward(cSeasonNorm.AsObject()))
      ReturnFalse;
   if(!cShortNorm.FeedForward(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cAdaptSpatNorm.FeedForward(cShortNorm.AsObject()))
      ReturnFalse;
   uint windows[3] = {NeuronOCL.Neurons() / iWindow,
                      cLongNorm.GetPeriod()*cLongNorm.GetUnits(),
                      2 * cLongNorm.GetUnits()
                     };
   if(!Concat(NeuronOCL.getOutput(), cLongNorm.getOutput(), cLongNorm.GetMeanSTDevs().getOutput(),
              cConcatenated.getOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[0] = windows[0] + windows[1] + windows[2];
   windows[1] = cSeasonNorm.GetPeriod() * cSeasonNorm.GetUnits();
   windows[2] = 2 * cSeasonNorm.GetUnits();
   if(!cConcatenated.SwapOutputs() ||
      !Concat(cConcatenated.getPrevOutput(), cSeasonNorm.getOutput(), cSeasonNorm.GetMeanSTDevs().getOutput(),
              cConcatenated.getOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[0] = windows[0] + windows[1] + windows[2];
   windows[1] = cShortNorm.GetPeriod() * cShortNorm.GetUnits();
   windows[2] = 2 * cShortNorm.GetUnits();
   if(!cConcatenated.SwapOutputs() ||
      !Concat(cConcatenated.getPrevOutput(), cShortNorm.getOutput(), cShortNorm.GetMeanSTDevs().getOutput(),
              cConcatenated.getOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[0] = windows[0] + windows[1] + windows[2];
   windows[1] = cAdaptSpatNorm.GetUnits();
   windows[2] = 2 * cAdaptSpatNorm.GetUnits();
   if(!cConcatenated.SwapOutputs() ||
      !Concat(cConcatenated.getPrevOutput(), cAdaptSpatNorm.getOutput(), cAdaptSpatNorm.GetMeanSTDevs().getOutput(),
              cConcatenated.getOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   if(!cProjection.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   if(!cTranspose.FeedForward(cProjection.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caFusion.Size(); i++)
      if(!caFusion[i].FeedForward(cTranspose.AsObject()))
         ReturnFalse;
   if(!ElementMult(caFusion[0].getOutput(), caFusion[1].getOutput(), cFusionOut.getOutput()))
      ReturnFalse;
//---
   return CNeuronTransposeOCL::feedForward(cFusionOut.AsObject());
  }
//+-----------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronSCNNEncoder layer.  |
//+-----------------------------------------------------------------------------+
bool CNeuronSCNNEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::calcInputGradients(cFusionOut.AsObject()))
      ReturnFalse;
   if(!ElementMultGrad(caFusion[0].getOutput(), caFusion[0].getGradient(),
                       caFusion[1].getOutput(), caFusion[1].getGradient(),
                       cFusionOut.getGradient(), caFusion[0].Activation(), caFusion[1].Activation()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(caFusion[0].AsObject()))
      ReturnFalse;
   CBufferFloat* temp = cTranspose.getGradient();
   if(!cTranspose.SetGradient(cTranspose.getPrevOutput(), false) ||
      !cTranspose.CalcHiddenGradients(caFusion[1].AsObject()) ||
      !SumAndNormalize(temp, cTranspose.getGradient(), temp, cTranspose.GetCount(), false, 0, 0, 0, 1) ||
      !cTranspose.SetGradient(temp, false))
      ReturnFalse;
   if(!cProjection.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
//---
   uint windows[3] = {0};
   windows[1] = cAdaptSpatNorm.GetUnits();
   windows[2] = 2 * cAdaptSpatNorm.GetUnits();
   windows[0] = cConcatenated.Neurons() / iWindow - windows[1] - windows[2];
   if(!DeConcat(cConcatenated.getPrevOutput(), cAdaptSpatNorm.getGradient(),
                cAdaptSpatNorm.GetMeanSTDevs().getGradient(),
                cConcatenated.getGradient(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[1] = cShortNorm.GetPeriod() * cShortNorm.GetUnits();
   windows[2] = 2 * cShortNorm.GetUnits();
   windows[0] = windows[0] - windows[1] - windows[2];
   if(!DeConcat(cConcatenated.getGradient(), cShortNorm.getPrevOutput(),
                cShortNorm.GetMeanSTDevs().getGradient(),
                cConcatenated.getPrevOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[1] = cSeasonNorm.GetPeriod() * cSeasonNorm.GetUnits();
   windows[2] = 2 * cSeasonNorm.GetUnits();
   windows[0] = windows[0] - windows[1] - windows[2];
   if(!DeConcat(cConcatenated.getPrevOutput(), cSeasonNorm.getPrevOutput(),
                cSeasonNorm.GetMeanSTDevs().getGradient(),
                cConcatenated.getGradient(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
   windows[1] = cLongNorm.GetPeriod() * cLongNorm.GetUnits();
   windows[2] = 2 * cLongNorm.GetUnits();
   windows[0] = windows[0] - windows[1] - windows[2];
   if(!DeConcat(NeuronOCL.getPrevOutput(), cLongNorm.getPrevOutput(),
                cLongNorm.GetMeanSTDevs().getGradient(),
                cConcatenated.getPrevOutput(), windows[0], windows[1], windows[2], iWindow))
      ReturnFalse;
//---
   if(!cShortNorm.CalcHiddenGradients(cAdaptSpatNorm.AsObject()) ||
      !SumAndNormalize(cShortNorm.getGradient(), cShortNorm.getPrevOutput(),
                       cShortNorm.getGradient(), cShortNorm.GetPeriod(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cUnSeasonTransp.CalcHiddenGradients(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cSeasonNorm.CalcHiddenGradients(cUnSeasonTransp.AsObject()) ||
      !SumAndNormalize(cSeasonNorm.getGradient(), cSeasonNorm.getPrevOutput(),
                       cSeasonNorm.getGradient(), cSeasonNorm.GetPeriod(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cSeasonTransp.CalcHiddenGradients(cSeasonNorm.AsObject()))
      ReturnFalse;
   if(!cLongNorm.CalcHiddenGradients(cSeasonTransp.AsObject()) ||
      !SumAndNormalize(cLongNorm.getGradient(), cLongNorm.getPrevOutput(),
                       cLongNorm.getGradient(), cLongNorm.GetPeriod(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cLongNorm.AsObject()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getGradient(), cLongNorm.GetPeriod(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Updates the CNeuronSCNNEncoder trainable weights from the  accumulated gradients.  |
//+------------------------------------------------------------------------------------+
bool CNeuronSCNNEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cAdaptSpatNorm.UpdateInputWeights(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cProjection.UpdateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caFusion.Size(); i++)
      if(!caFusion[i].UpdateInputWeights(cTranspose.AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSCNNEncoder state to the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronSCNNEncoder::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cLongNorm.Save(file_handle))
      ReturnFalse;
   if(!cSeasonTransp.Save(file_handle))
      ReturnFalse;
   if(!cSeasonNorm.Save(file_handle))
      ReturnFalse;
   if(!cUnSeasonTransp.Save(file_handle))
      ReturnFalse;
   if(!cShortNorm.Save(file_handle))
      ReturnFalse;
   if(!cAdaptSpatNorm.Save(file_handle))
      ReturnFalse;
   if(!cConcatenated.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caFusion.Size(); i++)
      if(!caFusion[i].Save(file_handle))
         ReturnFalse;
   if(!cFusionOut.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSCNNEncoder state from the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronSCNNEncoder::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cLongNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonTransp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShortNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAdaptSpatNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenated.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caFusion.Size(); i++)
      if(!LoadInsideLayer(file_handle, caFusion[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFusionOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Rebinds the CNeuronSCNNEncoder GPU buffers to the given OpenCL  context.  |
//+---------------------------------------------------------------------------+
void CNeuronSCNNEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
//---
   cLongNorm.SetOpenCL(OpenCL);
   cSeasonTransp.SetOpenCL(OpenCL);
   cSeasonNorm.SetOpenCL(OpenCL);
   cUnSeasonTransp.SetOpenCL(OpenCL);
   cShortNorm.SetOpenCL(OpenCL);
   cAdaptSpatNorm.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
   for(uint i = 0; i < caFusion.Size(); i++)
      caFusion[i].SetOpenCL(OpenCL);
   cFusionOut.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronSCNNEncoder layer.          |
//+------------------------------------------------------------------+
void CNeuronSCNNEncoder::TrainMode(bool flag)
  {
   CNeuronTransposeOCL::TrainMode(flag);
//---
   cLongNorm.TrainMode(bTrain);
   cSeasonTransp.TrainMode(bTrain);
   cSeasonNorm.TrainMode(bTrain);
   cUnSeasonTransp.TrainMode(bTrain);
   cShortNorm.TrainMode(bTrain);
   cAdaptSpatNorm.TrainMode(bTrain);
   cConcatenated.TrainMode(bTrain);
   cProjection.TrainMode(bTrain);
   cTranspose.TrainMode(bTrain);
   for(uint i = 0; i < caFusion.Size(); i++)
      caFusion[i].TrainMode(bTrain);
   cFusionOut.TrainMode(bTrain);
  }
//+------------------------------------------------------------------------------------+
//| Averages the CNeuronSCNNEncoder weights toward the source model  with factor tau.  |
//+------------------------------------------------------------------------------------+
bool CNeuronSCNNEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronTransposeOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSCNNEncoder* Source = source;
   if(!cAdaptSpatNorm.WeightsUpdate(Source.cAdaptSpatNorm.AsObject(), tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   for(uint i = 0; i < caFusion.Size(); i++)
      if(!caFusion[i].WeightsUpdate(Source.caFusion[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Initializes the CNeuronSCNN layer and allocates its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronSCNN::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint units_count, uint variables, uint forecast,
                       uint season_period, uint short_period, uint layers,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (units_count + forecast)*variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
   CNeuronSCNNEncoder* encoder = NULL;
   CNeuronBaseOCL* residual = NULL;
   for(uint l = 0; l < layers; l++)
     {
      encoder = new CNeuronSCNNEncoder();
      if(!encoder)
         ReturnFalse;
      if(!encoder.Init(0, l, OpenCL, units_count, variables, forecast, season_period, short_period, optimization, iBatch) ||
         !cLayers.Add(encoder))
         ReturnFalse;
      encoder.SetActivationFunction(None);
      if((l + 1) == layers)
         break;
      //---
      residual = new CNeuronBaseOCL();
      if(!residual)
         ReturnFalse;
      if(!residual.Init(0, l, OpenCL, units_count * variables, optimization, iBatch) ||
         !cLayers.Add(residual))
         ReturnFalse;
      residual.SetActivationFunction(None);
     }
   if(!SetGradient(encoder.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Runs the CNeuronSCNN feed-forward pass and writes the layer  output.  |
//+-----------------------------------------------------------------------+
bool CNeuronSCNN::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!Output.Fill(0))
      ReturnFalse;
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronSCNNEncoder* current = NULL;
   CNeuronBaseOCL* residual = NULL;
   CNeuronBaseOCL* temp = NULL;
   int layers = cLayers.Total();
//---
   for(int l = 0; l < layers; l += 2)
     {
      current = cLayers[l];
      if(!current ||
         !current.FeedForward(inputs) ||
         !SumAndNormalize(Output, current.getOutput(), Output, current.GetCount(), false, 0, 0, 0, 1))
         ReturnFalse;
      if((l + 1) == layers)
         break;
      uint variables = current.GetWindow();
      uint dimension = inputs.Neurons() / variables;
      uint forecast = current.GetCount() - dimension;
      residual = cLayers[l + 1];
      if(!residual)
         ReturnFalse;
      if(!DeConcat(residual.getOutput(), current.getPrevOutput(), current.getOutput(),
                   dimension, forecast, variables) ||
         !SumAndNormalize(residual.getOutput(), inputs.getOutput(), residual.getOutput(),
                          dimension, true, 0, 0, 0, 1))
         ReturnFalse;
      inputs = residual;
     }
//---
   return true;
  }
//+----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronSCNN  layer.  |
//+----------------------------------------------------------------------+
bool CNeuronSCNN::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!PrevOutput.Fill(0))
      ReturnFalse;
//---
   CNeuronBaseOCL* inputs = NULL;
   CNeuronSCNNEncoder* current = cLayers[-1];
   CNeuronBaseOCL* residual = NULL;
   int layers = cLayers.Total() - 2;
//---
   for(int l = layers; l >= 0; l--)
      switch(cLayers[l].Type())
        {
         case defNeuronBaseOCL:
            inputs = cLayers[l];
            if(!inputs ||
               !inputs.CalcHiddenGradients(current))
               ReturnFalse;
            if(!!residual)
               if(!SumAndNormalize(inputs.getGradient(), residual.getGradient(), inputs.getGradient(),
                                   current.GetWindow(), false, 0, 0, 0, 1))
                  ReturnFalse;
            residual = inputs;
            break;
         case defNeuronSCNNEncoder:
            current = cLayers[l];
            if(!residual)
               break;
            inputs = cLayers[l + 2];
            if(!inputs)
               ReturnFalse;
            if(!Concat(residual.getGradient(), PrevOutput, current.getGradient(),
                       residual.Neurons() / current.GetWindow(),
                       current.GetCount() - residual.Neurons() / current.GetWindow(),
                       current.GetWindow()) ||
               !SumAndNormalize(current.getGradient(), inputs.getGradient(), current.getGradient(),
                                current.GetWindow(), false, 0, 0, 0, 1))
               ReturnFalse;
            break;
         default:
            ReturnFalse;
            break;
        }
//---
   if(!NeuronOCL.CalcHiddenGradients(current))
      ReturnFalse;
   if(!!residual)
      if(!SumAndNormalize(NeuronOCL.getGradient(), residual.getGradient(), NeuronOCL.getGradient(),
                          current.GetWindow(), false, 0, 0, 0, 1))
         ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Updates the CNeuronSCNN trainable weights from the accumulated  gradients.  |
//+-----------------------------------------------------------------------------+
bool CNeuronSCNN::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
//---
   for(int l = 0; l < cLayers.Total(); l++)
     {
      current = cLayers[l];
      if(!current)
         ReturnFalse;
      if(current.Type() == defNeuronSCNNEncoder ||
         current.Type() == defNeuronSSCNNEncoder)
         if(!current.UpdateInputWeights(inputs))
            ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Averages the CNeuronSCNN weights toward the source model with  factor tau.  |
//+-----------------------------------------------------------------------------+
bool CNeuronSCNN::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSCNN* Source = source;
   if(!cLayers.WeightsUpdate(Source.cLayers.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSCNN state to the file handle.                  |
//+------------------------------------------------------------------+
bool CNeuronSCNN::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cLayers.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSCNN state from the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronSCNN::Load(const int file_handle)
  {
   cLayers.Clear();
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   cLayers.SetOpenCL(OpenCL);
   if(!cLayers.Load(file_handle))
      ReturnFalse;
//---
   CNeuronBaseOCL* last = cLayers[-1];
   if(!SetGradient(last.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Rebinds the CNeuronSCNN GPU buffers to the given OpenCL context.  |
//+-------------------------------------------------------------------+
void CNeuronSCNN::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cLayers.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronSCNN layer.                 |
//+------------------------------------------------------------------+
void CNeuronSCNN::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cLayers.TrainMode(flag);
  }
//+------------------------------------------------------------------------------------+
//| Initializes the CNeuronPolynomialRegression layer and allocates  its GPU buffers.  |
//+------------------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint units_count, uint window, uint window_out,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * window_out, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   if(!cProjection.Init(0, 0, OpenCL, window, window, window_out, units_count, 1, optimization, iBatch))
      ReturnFalse;
   if(!cConvolution.Init(0, 1, OpenCL, window_out, window_out, window_out, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cConvolution.SetActivationFunction(None);
   if(!cResidual.Init(0, 2, OpenCL, window, window, window_out, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   if(!SetGradient(cConvolution.getGradient(), true))
      ReturnFalse;
   if(!cResidual.SetGradient(getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------------+
//| Runs the CNeuronPolynomialRegression feed-forward pass and  writes  the layer output.  |
//+----------------------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cConvolution.FeedForward(cProjection.AsObject()))
      ReturnFalse;
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!SumAndNormalize(cConvolution.getOutput(), cResidual.getOutput(),
                       Output, GetWindowOut(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronPolynomialRegression layer.  |
//+--------------------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cProjection.CalcHiddenGradients(cConvolution.AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cResidual.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp,
                       GetWindowIn(), false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------------------+
//| Updates the CNeuronPolynomialRegression trainable weights from  the accumulated gradients.  |
//+---------------------------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cConvolution.UpdateInputWeights(cProjection.AsObject()))
      ReturnFalse;
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------------------+
//| Averages the CNeuronPolynomialRegression weights toward the  source model with factor tau.  |
//+---------------------------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronPolynomialRegression* Source = source;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   if(!cConvolution.WeightsUpdate(Source.cConvolution.AsObject(), tau))
      ReturnFalse;
   if(!cResidual.WeightsUpdate(Source.cResidual.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronPolynomialRegression state to the file handle.  |
//+------------------------------------------------------------------+
bool CNeuronPolynomialRegression::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cConvolution.Save(file_handle))
      ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Restores the CNeuronPolynomialRegression state from the file  handle.  |
//+------------------------------------------------------------------------+
bool CNeuronPolynomialRegression::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConvolution.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Rebinds the CNeuronPolynomialRegression GPU buffers to the given  OpenCL context.  |
//+------------------------------------------------------------------------------------+
void CNeuronPolynomialRegression::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cProjection.SetOpenCL(OpenCL);
   cConvolution.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+----------------------------------------------------------------------------+
//| Initializes the CNeuronSSCNNEncoder layer and allocates its GPU  buffers.  |
//+----------------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint units_count, uint variables, uint forecast,
                               uint season_period, uint short_period,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, units_count + forecast,
                                 variables, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cLongNorm.Init(0, index, OpenCL, 1, units_count, iWindow, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLongExtrapolate.Init(0, index, OpenCL, units_count, units_count, iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cLongExtrapolate.SetActivationFunction(None);
   index++;
   if(!cLongMeanSTDevTransp.Init(0, index, OpenCL, iWindow, 2, optimization, iBatch))
      ReturnFalse;
   if(!cLongMeanSTDevTransp.getGradient().Fill(0))
      ReturnFalse;
   index++;
   if(!cLongMeanExtrapolate.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cLongExtrapolate.SetActivationFunction(None);
   if(!cLongMeanExtrapolate.getPrevOutput().Fill(1))
      ReturnFalse;
   index++;
   if(!cSeasonTransp.Init(0, index, OpenCL, iWindow, units_count / season_period, season_period, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSeasonNorm.Init(0, index, OpenCL, season_period, cSeasonTransp.GetCount(), iWindow, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cUnSeasonTransp.Init(0, index, OpenCL, iWindow, season_period, cSeasonTransp.GetCount(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSeasonExtrapolate.Init(0, index, OpenCL, units_count, units_count, iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cSeasonExtrapolate.SetActivationFunction(None);
   index++;
   if(!cSeasonMeanExtrapolate.Init(0, index, OpenCL, season_period, season_period, iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cSeasonMeanExtrapolate.SetActivationFunction(None);
   index++;
   if(!cShortNorm.Init(0, index, OpenCL, units_count / short_period, short_period, iWindow, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cShortExtrapolate.Init(0, index, OpenCL, units_count, units_count, iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cShortExtrapolate.SetActivationFunction(None);
   index++;
   if(!cShortMeanExtrapolate.Init(0, index, OpenCL, cShortNorm.GetUnits(), cShortNorm.GetUnits(), iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cShortMeanExtrapolate.SetActivationFunction(None);
   index++;
   if(!cSpatialNorm.Init(0, index, OpenCL, units_count, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSpatialExtrapolate.Init(0, index, OpenCL, units_count, units_count, iCount, iWindow, 1, optimization, iBatch))
      ReturnFalse;
   cSpatialExtrapolate.SetActivationFunction(None);
   index++;
   if(!cSpatialMeanExtrapolate.Init(0, index, OpenCL, units_count, units_count, iCount, 1, iWindow, optimization, iBatch))
      ReturnFalse;
   cSpatialMeanExtrapolate.SetActivationFunction(None);
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, 8 * Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose.Init(0, index, OpenCL, 8 * iWindow, iCount, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cFusion.Init(0, index, OpenCL, iCount, 8 * iWindow, iWindow, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Runs the CNeuronSSCNNEncoder feed-forward pass and writes the  layer output.  |
//+-------------------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Long
   if(!cLongNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cLongExtrapolate.FeedForward(cLongNorm.AsObject()))
      ReturnFalse;
   if(!cLongMeanSTDevTransp.FeedForward(cLongNorm.GetMeanSTDevs()))
      ReturnFalse;
   if(!MatMul(cLongMeanSTDevTransp.getOutput(), cLongMeanExtrapolate.getPrevOutput(),
              cLongMeanExtrapolate.getOutput(), 1, 1, iCount, iWindow, true))
      ReturnFalse;
//--- Season
   if(!cSeasonTransp.FeedForward(cLongNorm.AsObject()))
      ReturnFalse;
   if(!cSeasonNorm.FeedForward(cSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cUnSeasonTransp.FeedForward(cSeasonNorm.AsObject()))
      ReturnFalse;
   if(!cSeasonExtrapolate.FeedForward(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cSeasonMeanExtrapolate.FeedForward(cSeasonNorm.GetMeans()))
      ReturnFalse;
//--- Short
   if(!cShortNorm.FeedForward(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cShortExtrapolate.FeedForward(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cShortMeanExtrapolate.FeedForward(cShortNorm.GetMeans()))
      ReturnFalse;
//--- Spatial
   if(!cSpatialNorm.FeedForward(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cSpatialExtrapolate.FeedForward(cSpatialNorm.AsObject()))
      ReturnFalse;
   if(!cSpatialMeanExtrapolate.FeedForward(cSpatialNorm.GetMeans()))
      ReturnFalse;
//--- Concat
   if(!Concat(cLongExtrapolate.getOutput(), cLongMeanExtrapolate.getOutput(),
              cSeasonExtrapolate.getOutput(), cSeasonMeanExtrapolate.getOutput(),
              cConcatenated.getOutput(), iCount, iCount, iCount, iCount, iWindow))
      ReturnFalse;
   if(!Concat(cConcatenated.getOutput(), cShortExtrapolate.getOutput(),
              cShortMeanExtrapolate.getOutput(), cConcatenated.getPrevOutput(),
              4 * iCount, iCount, iCount, iWindow))
      ReturnFalse;
   if(!Concat(cConcatenated.getPrevOutput(), cSpatialExtrapolate.getOutput(),
              cSpatialMeanExtrapolate.getOutput(), cConcatenated.getOutput(),
              6 * iCount, iCount, iCount, iWindow))
      ReturnFalse;
//--- Fusion
   if(!cTranspose.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   if(!cFusion.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
//---
   return CNeuronTransposeOCL::feedForward(cFusion.AsObject());
  }
//+------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronSSCNNEncoder layer.  |
//+------------------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Fusion
   if(!CNeuronTransposeOCL::calcInputGradients(cFusion.AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cFusion.AsObject()))
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//--- DeConcat
   if(!DeConcat(cConcatenated.getPrevOutput(), cSpatialExtrapolate.getGradient(),
                cSpatialMeanExtrapolate.getGradient(), cConcatenated.getGradient(),
                6 * iCount, iCount, iCount, iWindow))
      ReturnFalse;
   if(!DeConcat(cConcatenated.getGradient(), cShortExtrapolate.getGradient(),
                cShortMeanExtrapolate.getGradient(), cConcatenated.getPrevOutput(),
                4 * iCount, iCount, iCount, iWindow))
      ReturnFalse;
   if(!DeConcat(cLongExtrapolate.getGradient(), cLongMeanExtrapolate.getGradient(),
                cSeasonExtrapolate.getGradient(), cSeasonMeanExtrapolate.getGradient(),
                cConcatenated.getGradient(), iCount, iCount, iCount, iCount, iWindow))
      ReturnFalse;
//--- Spatial
   if(!cSpatialNorm.GetMeans().CalcHiddenGradients(cSpatialMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!cSpatialNorm.CalcHiddenGradients(cSpatialExtrapolate.AsObject()))
      ReturnFalse;
//--- Short
   if(!cShortNorm.CalcHiddenGradients(cSpatialNorm.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = cShortNorm.getGradient();
   if(!cShortNorm.SetGradient(cShortNorm.getPrevOutput(), false) ||
      !cShortNorm.CalcHiddenGradients(cShortExtrapolate.AsObject()) ||
      !SumAndNormalize(temp, cShortNorm.getGradient(), temp, iWindow, false, 0, 0, 0, 1) ||
      !cShortNorm.SetGradient(temp, false))
      ReturnFalse;
   if(!cShortNorm.GetMeans().CalcHiddenGradients(cShortMeanExtrapolate.AsObject()))
      ReturnFalse;
//--- Season
   if(!cUnSeasonTransp.CalcHiddenGradients(cShortNorm.AsObject()))
      ReturnFalse;
   temp = cUnSeasonTransp.getGradient();
   if(!cUnSeasonTransp.SetGradient(cUnSeasonTransp.getPrevOutput(), false) ||
      !cUnSeasonTransp.CalcHiddenGradients(cSeasonExtrapolate.AsObject()) ||
      !SumAndNormalize(temp, cUnSeasonTransp.getGradient(), temp, iWindow, false, 0, 0, 0, 1) ||
      !cUnSeasonTransp.SetGradient(temp, false))
      ReturnFalse;
   if(!cSeasonNorm.GetMeans().CalcHiddenGradients(cSeasonMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!cSeasonNorm.CalcHiddenGradients(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cSeasonTransp.CalcHiddenGradients(cSeasonNorm.AsObject()))
      ReturnFalse;
//--- Long
   if(!MatMulGrad(cLongMeanSTDevTransp.getOutput(), cLongMeanSTDevTransp.getGradient(),
                  cLongMeanExtrapolate.getPrevOutput(), cLongMeanExtrapolate.getGradient(),
                  cLongMeanExtrapolate.getGradient(), 1, 1, iCount, iWindow, true))
      ReturnFalse;
   if(!cLongNorm.GetMeanSTDevs().CalcHiddenGradients(cLongMeanSTDevTransp.AsObject()))
      ReturnFalse;
   if(!cLongNorm.CalcHiddenGradients(cSeasonTransp.AsObject()))
      ReturnFalse;
   temp = cLongNorm.getGradient();
   if(!cLongNorm.SetGradient(cLongNorm.getPrevOutput(), false) ||
      !cLongNorm.CalcHiddenGradients(cLongExtrapolate.AsObject()) ||
      !SumAndNormalize(temp, cLongNorm.getGradient(), temp, iWindow, false, 0, 0, 0, 1) ||
      !cLongNorm.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cLongNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------------+
//| Updates the CNeuronSSCNNEncoder trainable weights from the  accumulated gradients.  |
//+-------------------------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//--- Long
   if(!cLongNorm.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cLongExtrapolate.UpdateInputWeights(cLongNorm.AsObject()))
      ReturnFalse;
//--- Season
   if(!cSeasonNorm.UpdateInputWeights(cSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cSeasonExtrapolate.UpdateInputWeights(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cSeasonMeanExtrapolate.UpdateInputWeights(cSeasonNorm.GetMeans()))
      ReturnFalse;
//--- Short
   if(!cShortNorm.UpdateInputWeights(cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!cShortExtrapolate.UpdateInputWeights(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cShortMeanExtrapolate.UpdateInputWeights(cShortNorm.GetMeans()))
      ReturnFalse;
//--- Spatial
   if(!cSpatialNorm.UpdateInputWeights(cShortNorm.AsObject()))
      ReturnFalse;
   if(!cSpatialExtrapolate.UpdateInputWeights(cSpatialNorm.AsObject()))
      ReturnFalse;
   if(!cSpatialMeanExtrapolate.UpdateInputWeights(cSpatialNorm.GetMeans()))
      ReturnFalse;
//--- Fusion
   if(!cFusion.UpdateInputWeights(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------------+
//| Averages the CNeuronSSCNNEncoder weights toward the source model  with factor tau.  |
//+-------------------------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronTransposeOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSSCNNEncoder* Source = source;
//--- Long
   if(!cLongNorm.WeightsUpdate(Source.cLongNorm.AsObject(), tau))
      ReturnFalse;
   if(!cLongExtrapolate.WeightsUpdate(Source.cLongExtrapolate.AsObject(), tau))
      ReturnFalse;
//--- Season
   if(!cSeasonNorm.WeightsUpdate(Source.cSeasonNorm.AsObject(), tau))
      ReturnFalse;
   if(!cSeasonExtrapolate.WeightsUpdate(Source.cSeasonExtrapolate.AsObject(), tau))
      ReturnFalse;
   if(!cSeasonMeanExtrapolate.WeightsUpdate(Source.cSeasonMeanExtrapolate.AsObject(), tau))
      ReturnFalse;
//--- Short
   if(!cShortNorm.WeightsUpdate(Source.cShortNorm.AsObject(), tau))
      ReturnFalse;
   if(!cShortExtrapolate.WeightsUpdate(Source.cShortExtrapolate.AsObject(), tau))
      ReturnFalse;
   if(!cShortMeanExtrapolate.WeightsUpdate(Source.cShortMeanExtrapolate.AsObject(), tau))
      ReturnFalse;
//--- Spatial
   if(!cSpatialNorm.WeightsUpdate(Source.cSpatialNorm.AsObject(), tau))
      ReturnFalse;
   if(!cSpatialExtrapolate.WeightsUpdate(Source.cSpatialExtrapolate.AsObject(), tau))
      ReturnFalse;
   if(!cSpatialMeanExtrapolate.WeightsUpdate(Source.cSpatialMeanExtrapolate.AsObject(), tau))
      ReturnFalse;
//--- Fusion
   if(!cFusion.WeightsUpdate(Source.cFusion.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSSCNNEncoder state to the file handle.          |
//+------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cLongNorm.Save(file_handle))
      ReturnFalse;
   if(!cLongExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cLongMeanSTDevTransp.Save(file_handle))
      ReturnFalse;
   if(!cLongMeanExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cSeasonTransp.Save(file_handle))
      ReturnFalse;
   if(!cSeasonNorm.Save(file_handle))
      ReturnFalse;
   if(!cUnSeasonTransp.Save(file_handle))
      ReturnFalse;
   if(!cSeasonExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cSeasonMeanExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cShortNorm.Save(file_handle))
      ReturnFalse;
   if(!cShortExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cShortMeanExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cSpatialNorm.Save(file_handle))
      ReturnFalse;
   if(!cSpatialExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cSpatialMeanExtrapolate.Save(file_handle))
      ReturnFalse;
   if(!cConcatenated.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cFusion.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSSCNNEncoder state from the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronSSCNNEncoder::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cLongNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLongExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLongMeanSTDevTransp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLongMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonTransp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cUnSeasonTransp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSeasonMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShortNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShortExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cShortMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpatialNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpatialExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpatialMeanExtrapolate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenated.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFusion.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Rebinds the CNeuronSSCNNEncoder GPU buffers to the given OpenCL  context.  |
//+----------------------------------------------------------------------------+
void CNeuronSSCNNEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
//---
   cLongNorm.SetOpenCL(OpenCL);
   cLongExtrapolate.SetOpenCL(OpenCL);
   cLongMeanSTDevTransp.SetOpenCL(OpenCL);
   cLongMeanExtrapolate.SetOpenCL(OpenCL);
   cSeasonTransp.SetOpenCL(OpenCL);
   cSeasonNorm.SetOpenCL(OpenCL);
   cUnSeasonTransp.SetOpenCL(OpenCL);
   cSeasonExtrapolate.SetOpenCL(OpenCL);
   cSeasonMeanExtrapolate.SetOpenCL(OpenCL);
   cShortNorm.SetOpenCL(OpenCL);
   cShortExtrapolate.SetOpenCL(OpenCL);
   cShortMeanExtrapolate.SetOpenCL(OpenCL);
   cSpatialNorm.SetOpenCL(OpenCL);
   cSpatialExtrapolate.SetOpenCL(OpenCL);
   cSpatialMeanExtrapolate.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
   cFusion.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronSSCNNEncoder layer.         |
//+------------------------------------------------------------------+
void CNeuronSSCNNEncoder::TrainMode(bool flag)
  {
   CNeuronTransposeOCL::TrainMode(flag);
//---
   cLongNorm.TrainMode(bTrain);
   cLongExtrapolate.TrainMode(bTrain);
   cLongMeanSTDevTransp.TrainMode(bTrain);
   cLongMeanExtrapolate.TrainMode(bTrain);
   cSeasonTransp.TrainMode(bTrain);
   cSeasonNorm.TrainMode(bTrain);
   cUnSeasonTransp.TrainMode(bTrain);
   cSeasonExtrapolate.TrainMode(bTrain);
   cSeasonMeanExtrapolate.TrainMode(bTrain);
   cShortNorm.TrainMode(bTrain);
   cShortExtrapolate.TrainMode(bTrain);
   cShortMeanExtrapolate.TrainMode(bTrain);
   cSpatialNorm.TrainMode(bTrain);
   cSpatialExtrapolate.TrainMode(bTrain);
   cSpatialMeanExtrapolate.TrainMode(bTrain);
   cConcatenated.TrainMode(bTrain);
   cTranspose.TrainMode(bTrain);
   cFusion.TrainMode(bTrain);
  }
//+---------------------------------------------------------------------+
//| Initializes the CNeuronSSCNN layer and allocates its GPU  buffers.  |
//+---------------------------------------------------------------------+
bool CNeuronSSCNN::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint units_count, uint variables, uint forecast,
                        uint season_period, uint short_period, uint layers,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (units_count + forecast)*variables,
                            optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
   CNeuronSSCNNEncoder* encoder = NULL;
   CNeuronBaseOCL* residual = NULL;
   for(uint l = 0; l < layers; l++)
     {
      encoder = new CNeuronSSCNNEncoder();
      if(!encoder)
         ReturnFalse;
      if(!encoder.Init(0, 2 * l, OpenCL, units_count, variables, forecast, season_period,
                       short_period, optimization, iBatch) ||
         !cLayers.Add(encoder))
         ReturnFalse;
      encoder.SetActivationFunction(None);
      if((l + 1) == layers)
         break;
      //---
      residual = new CNeuronBaseOCL();
      if(!residual)
         ReturnFalse;
      if(!residual.Init(0, 2 * l + 1, OpenCL, units_count * variables, optimization, iBatch) ||
         !cLayers.Add(residual))
         ReturnFalse;
      residual.SetActivationFunction(None);
     }
   if(!SetGradient(encoder.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Runs the CNeuronSSCNN feed-forward pass and writes the layer  output.  |
//+------------------------------------------------------------------------+
bool CNeuronSSCNN::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!Output.Fill(0))
      ReturnFalse;
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronSSCNNEncoder* current = NULL;
   CNeuronBaseOCL* residual = NULL;
   CNeuronBaseOCL* temp = NULL;
   int layers = cLayers.Total();
//---
   for(int l = 0; l < layers; l += 2)
     {
      current = cLayers[l];
      if(!current ||
         !current.FeedForward(inputs) ||
         !SumAndNormalize(Output, current.getOutput(), Output, current.GetCount(), false, 0, 0, 0, 1))
         ReturnFalse;
      if((l + 1) == layers)
         break;
      uint variables = current.GetWindow();
      uint dimension = inputs.Neurons() / variables;
      uint forecast = current.GetCount() - dimension;
      residual = cLayers[l + 1];
      if(!residual)
         ReturnFalse;
      if(!DeConcat(residual.getOutput(), current.getPrevOutput(), current.getOutput(),
                   dimension, forecast, variables) ||
         !SumAndNormalize(residual.getOutput(), inputs.getOutput(), residual.getOutput(),
                          dimension, true, 0, 0, 0, 1))
         ReturnFalse;
      inputs = residual;
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronSSCNN  layer.  |
//+-----------------------------------------------------------------------+
bool CNeuronSSCNN::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!PrevOutput.Fill(0))
      ReturnFalse;
//---
   CNeuronBaseOCL* inputs = NULL;
   CNeuronSSCNNEncoder* current = cLayers[-1];
   CNeuronBaseOCL* residual = NULL;
   int layers = cLayers.Total() - 2;
//---
   for(int l = layers; l >= 0; l--)
      switch(cLayers[l].Type())
        {
         case defNeuronBaseOCL:
            inputs = cLayers[l];
            if(!inputs ||
               !inputs.CalcHiddenGradients(current))
               ReturnFalse;
            if(!!residual)
               if(!SumAndNormalize(inputs.getGradient(), residual.getGradient(), inputs.getGradient(),
                                   current.GetWindow(), false, 0, 0, 0, 1))
                  ReturnFalse;
            residual = inputs;
            break;
         case defNeuronSSCNNEncoder:
            current = cLayers[l];
            if(!residual)
               break;
            inputs = cLayers[l + 2];
            if(!inputs)
               ReturnFalse;
            if(!Concat(residual.getGradient(), PrevOutput, current.getGradient(),
                       residual.Neurons() / current.GetWindow(),
                       current.GetCount() - residual.Neurons() / current.GetWindow(),
                       current.GetWindow()) ||
               !SumAndNormalize(current.getGradient(), inputs.getGradient(), current.getGradient(),
                                current.GetWindow(), false, 0, 0, 0, 1))
               ReturnFalse;
            break;
         default:
            ReturnFalse;
            break;
        }
//---
   if(!NeuronOCL.CalcHiddenGradients(current))
      ReturnFalse;
   if(!!residual)
      if(!SumAndNormalize(NeuronOCL.getGradient(), residual.getGradient(), NeuronOCL.getGradient(),
                          current.GetWindow(), false, 0, 0, 0, 1))
         ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Initializes the CChebPolinom layer and allocates its GPU  buffers.  |
//+---------------------------------------------------------------------+
bool CChebPolinom::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint dimension,
                        uint steps, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, dimension * dimension * steps, optimization_type, batch))
      ReturnFalse;
//---
   activation = None;
   iDimension = dimension;
   iSteps = steps;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Runs the CChebPolinom feed-forward pass and writes the layer  output.  |
//+------------------------------------------------------------------------+
bool CChebPolinom::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() != (iDimension * iDimension))
      ReturnFalse;
//---
   uint global_work_offset[3] = { 0 };
   uint global_work_size[3] = { MathMin(iDimension, uint(OpenCL.GetMaxLocalSize(0))),
                                iDimension, iDimension
                              };
   uint local_work_size[3] = { global_work_size[0], 1, 1 };
//---
   uint kernel = def_k_ChebStep;
   setBuffer(kernel, def_k_cheb_support, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_cheb_outputs, getOutputIndex())
   for(int step = MathMin(2, MathMax(int(iSteps) - 1, 0)); step < int(iSteps); step++)
     {
      setArgument(kernel, def_k_cheb_step, step + 1)
      kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!Output.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CChebPolinom  layer.  |
//+-----------------------------------------------------------------------+
bool CChebPolinom::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() != (iDimension * iDimension))
      ReturnFalse;
//---
   uint global_work_offset[3] = { 0 };
   uint global_work_size[3] = { MathMin(iDimension, uint(OpenCL.GetMaxLocalSize(0))),
                                iDimension, iDimension
                              };
   uint local_work_size[3] = { global_work_size[0], 1, 1 };
//---
   uint kernel = def_k_ChebStepGrad;
   setBuffer(kernel, def_k_chebgr_support, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_chebgr_support_g, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_chebgr_outputs, getOutputIndex())
   setBuffer(kernel, def_k_chebgr_outputs_g, getGradientIndex())
   for(int step = (int(iSteps) - 1); step >= MathMax(MathMin(2, int(iSteps) - 1), 1); step--)
     {
      setArgument(kernel, def_k_chebgr_step, step)
      kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!NeuronOCL.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CChebPolinom state to the file handle.                 |
//+------------------------------------------------------------------+
bool CChebPolinom::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iDimension)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iSteps)) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CChebPolinom state from the file handle.            |
//+------------------------------------------------------------------+
bool CChebPolinom::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimension = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iSteps = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetGrapConv layer and allocates its  GPU  buffers.  |
//+-------------------------------------------------------------------------------+
bool CNeuronHimNetGrapConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint count, uint window, uint cheb_k,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeRCDOCL::Init(numOutputs, myIndex, open_cl, cheb_k,
                                    count, window, optimization_type, batch))
      ReturnFalse;
   activation = None;
   if(!cX_G.Init(0, 0, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cX_G.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronHimNetGrapConv feed forward operation.        |
//+------------------------------------------------------------------+
bool CNeuronHimNetGrapConv::FeedForward(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support)
  {
   if(!NeuronOCL || !Support)
      ReturnFalse;
   if(Support.GetDimension() != iWindow ||
      Support.GetSteps() != iCount)
      ReturnFalse;
   if(NeuronOCL.Neurons() != (Neurons() / iCount))
      ReturnFalse;
//---
   if(!MatMul(Support.getOutput(), NeuronOCL.getOutput(), cX_G.getOutput(),
              iWindow, iWindow, GetDimension(), iCount, false))
      ReturnFalse;
//---
   return CNeuronTransposeRCDOCL::feedForward(cX_G.AsObject());
  }
//+--------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetGrapConv layer.  |
//+--------------------------------------------------------------------------------+
bool CNeuronHimNetGrapConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CChebPolinom *Support)
  {
   if(!NeuronOCL || !Support)
      ReturnFalse;
   if(Support.GetDimension() != iWindow ||
      Support.GetSteps() != iCount)
      ReturnFalse;
   if(NeuronOCL.Neurons() != (Neurons() / iCount))
      ReturnFalse;
//---
   if(!CNeuronTransposeRCDOCL::calcInputGradients(cX_G.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(Support.getOutput(), Support.getGradient(),
                  NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                  cX_G.getGradient(), iWindow, iWindow,
                  GetDimension(), iCount, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHimNetGrapConv state from the file handle.   |
//+------------------------------------------------------------------+
bool CNeuronHimNetGrapConv::Load(const int file_handle)
  {
   if(!CNeuronTransposeRCDOCL::Load(file_handle))
      ReturnFalse;
   if(!cX_G.Init(0, 0, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetGrapConv GPU buffers to the given  OpenCL  context.  |
//+-------------------------------------------------------------------------------+
void CNeuronHimNetGrapConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronTransposeRCDOCL::SetOpenCL(obj);
   cX_G.SetOpenCL(OpenCL);
  }
//+--------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetGCRU layer and allocates its GPU  buffers.  |
//+--------------------------------------------------------------------------+
bool CNeuronHimNetGCRU::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint units, uint window, uint window_out,
                             uint cheb_k, uint embed_dim,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * window_out, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cInpAndHidden.Init(0, index, OpenCL, (window + window_out)*units, optimization, iBatch))
      ReturnFalse;
   cInpAndHidden.SetActivationFunction(None);
   index++;
   if(!cZ_R.Init(0, index, OpenCL, units, window + window_out, cheb_k, optimization, iBatch))
      ReturnFalse;
   cZ_R.SetActivationFunction(None);
   index++;
   if(!cZ_R_emb.Init(0, index, OpenCL, embed_dim, embed_dim,
                     2 * cheb_k * (window + window_out) * window_out,
                     units, 1, optimization, iBatch))
      ReturnFalse;
   cZ_R_emb.SetActivationFunction(None);
   index++;
   if(!cZe_Re.Init(0, index, OpenCL, 2 * window_out * units, optimization, iBatch))
      ReturnFalse;
   cZe_Re.SetActivationFunction(None);
   index++;
   if(!cZ.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction(None);
   index++;
   if(!cR.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cR.SetActivationFunction(None);
   index++;
   if(!cCandidate.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   cCandidate.SetActivationFunction(None);
   index++;
   if(!cHC.Init(0, index, OpenCL, units, window + window_out, cheb_k, optimization, iBatch))
      ReturnFalse;
   cHC.SetActivationFunction(None);
   index++;
   if(!cHC_emb.Init(0, index, OpenCL, embed_dim, embed_dim,
                    (window + window_out) * window_out * cheb_k,
                    units, 1, optimization, iBatch))
      ReturnFalse;
   cHC_emb.SetActivationFunction(None);
   index++;
   if(!cHCe.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cHCe.SetActivationFunction(None);
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!Output.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------+
//| Runs the CNeuronHimNetGCRU feed-forward pass and writes the  layer  output.  |
//+------------------------------------------------------------------------------+
bool CNeuronHimNetGCRU::feedForward(CNeuronBaseOCL *NeuronOCL,
                                    CChebPolinom *Support,
                                    CNeuronBaseOCL *Embedding)
  {
   if(!NeuronOCL || !Support || !Embedding)
      ReturnFalse;
   if(!SwapOutputs())
      ReturnFalse;
//---
   uint cheb_k = cZ_R.GetChebK();
   uint units = cZ_R.GetCount();
   uint window_out = Neurons() / units;
   uint window = cZ_R.GetWindow() - window_out;
//---
   if(!Concat(NeuronOCL.getOutput(), PrevOutput, cInpAndHidden.getOutput(), window, window_out, units))
      ReturnFalse;
   if(!cZ_R.FeedForward(cInpAndHidden.AsObject(), Support))
      ReturnFalse;
   if(!cZ_R_emb.FeedForward(Embedding))
      ReturnFalse;
   if(!MatMul(cZ_R.getOutput(), cZ_R_emb.getOutput(), cZe_Re.getOutput(), 1, (window + window_out)*cheb_k,
              2 * window_out, units, true))
      ReturnFalse;
   if(!Activation(cZe_Re.getOutput(), cZe_Re.getOutput(), SIGMOID))
      ReturnFalse;
   if(!DeConcat(cZ.getOutput(), cR.getOutput(), cZe_Re.getOutput(), window_out, window_out, units))
      ReturnFalse;
   if(!ElementMult(cZ.getOutput(), PrevOutput, cZ.getPrevOutput()))
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), cZ.getPrevOutput(), cCandidate.getOutput(), window, window_out, units))
      ReturnFalse;
   if(!cHC.FeedForward(cCandidate.AsObject(), Support))
      ReturnFalse;
   if(!cHC_emb.FeedForward(Embedding))
      ReturnFalse;
   if(!MatMul(cHC.getOutput(), cHC_emb.getOutput(), cHCe.getOutput(), 1, (window + window_out)*cheb_k,
              window_out, units, true))
      ReturnFalse;
   if(!Activation(cHCe.getOutput(), cHCe.getOutput(), TANH))
      ReturnFalse;
   if(!GateElementMult(PrevOutput, cHCe.getOutput(), cR.getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetGCRU  layer.  |
//+-----------------------------------------------------------------------------+
bool CNeuronHimNetGCRU::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CChebPolinom *Support,
      CNeuronBaseOCL *Embedding)
  {
   if(!NeuronOCL || !Support || !Embedding)
      ReturnFalse;
//---
   uint cheb_k = cZ_R.GetChebK();
   uint units = cZ_R.GetCount();
   uint window_out = Neurons() / units;
   uint window = cZ_R.GetWindow() - window_out;
//---
   if(!GateElementMultGrad(PrevOutput, cR.getPrevOutput(),
                           cHCe.getOutput(), cHCe.getGradient(),
                           cR.getOutput(), cR.getGradient(),
                           Gradient, None, TANH, cR.Activation()))
      ReturnFalse;
   if(!MatMulGrad(cHC.getOutput(), cHC.getGradient(),
                  cHC_emb.getOutput(), cHC_emb.getGradient(),
                  cHCe.getGradient(), 1, (window + window_out)*cheb_k,
                  window_out, units, true))
      ReturnFalse;
   if(!cHC.calcInputGradients(cCandidate.AsObject(), Support))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), cZ.getPrevOutput(), cCandidate.getGradient(),
                window, window_out, units))
      ReturnFalse;
   if(!ElementMultGrad(cZ.getOutput(), cZ.getGradient(),
                       PrevOutput, cCandidate.getPrevOutput(),
                       cZ.getPrevOutput(), None, None))
      ReturnFalse;
   if(!Concat(cZ.getGradient(), cR.getGradient(), cZe_Re.getGradient(),
              window_out, window_out, units))
      ReturnFalse;
   if(!DeActivation(cZe_Re.getOutput(), cZe_Re.getGradient(), cZe_Re.getGradient(), SIGMOID))
      ReturnFalse;
   if(!MatMulGrad(cZ_R.getOutput(), cZ_R.getGradient(),
                  cZ_R_emb.getOutput(), cZ_R_emb.getGradient(),
                  cZe_Re.getGradient(), 1, (window + window_out)*cheb_k,
                  2 * window_out, units, true))
      ReturnFalse;
   CBufferFloat* temp = Support.getGradient();
   if(!Support.SetGradient(GetPointer(bSupportAccum), false) ||
      !cZ_R.calcInputGradients(cInpAndHidden.AsObject(), Support) ||
      !SumAndNormalize(temp, Support.getGradient(), temp, Support.GetDimension(), false, 0, 0, 0, 1) ||
      !Support.SetGradient(temp, false))
      ReturnFalse;
   if(!DeConcat(cInpAndHidden.getPrevOutput(), cCandidate.getPrevOutput(), cInpAndHidden.getGradient(), window, window_out, units))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cInpAndHidden.getPrevOutput(), NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   if(!Embedding.CalcHiddenGradients(cHC_emb.AsObject()))
      ReturnFalse;
   temp = Embedding.getGradient();
   if(!Embedding.SetGradient(Embedding.getPrevOutput(), false) ||
      !Embedding.CalcHiddenGradients(cZ_R_emb.AsObject()) ||
      !SumAndNormalize(temp, Embedding.getGradient(), temp, cZ_R_emb.GetWindow(), false, 0, 0, 0, 1) ||
      !Embedding.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------------+
//| Updates the CNeuronHimNetGCRU trainable weights from the  accumulated gradients.  |
//+-----------------------------------------------------------------------------------+
bool CNeuronHimNetGCRU::updateInputWeights(CNeuronBaseOCL *NeuronOCL,
      CChebPolinom *Support,
      CNeuronBaseOCL *Embedding)
  {
   if(!Embedding)
      ReturnFalse;
//---
   if(!cZ_R_emb.UpdateInputWeights(Embedding))
      ReturnFalse;
   if(!cHC_emb.UpdateInputWeights(Embedding))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------------+
//| Averages the CNeuronHimNetGCRU weights toward the source model  with factor tau.  |
//+-----------------------------------------------------------------------------------+
bool CNeuronHimNetGCRU::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronHimNetGCRU* Source = source;
   if(!cZ_R_emb.WeightsUpdate(Source.cZ_R_emb.AsObject(), tau))
      ReturnFalse;
   if(!cHC_emb.WeightsUpdate(Source.cHC_emb.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHimNetGCRU state to the file handle.            |
//+------------------------------------------------------------------+
bool CNeuronHimNetGCRU::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cZ_R.Save(file_handle))
      ReturnFalse;
   if(!cZ_R_emb.Save(file_handle))
      ReturnFalse;
   if(!cHC.Save(file_handle))
      ReturnFalse;
   if(!cHC_emb.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHimNetGCRU state from the file handle.       |
//+------------------------------------------------------------------+
bool CNeuronHimNetGCRU::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cZ_R.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cZ_R_emb.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHC.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHC_emb.AsObject()))
      ReturnFalse;
//---
   uint cheb_k = cZ_R.GetChebK();
   uint units = cZ_R.GetCount();
   uint window_out = Neurons() / units;
   uint window = cZ_R.GetWindow() - window_out;
//---
   int index = 0;
   if(!cInpAndHidden.Init(0, index, OpenCL, (window + window_out)*units, optimization, iBatch))
      ReturnFalse;
   cInpAndHidden.SetActivationFunction(None);
   index += 3;
   if(!cZe_Re.Init(0, index, OpenCL, 2 * window_out * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cZ.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction(None);
   index++;
   if(!cR.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cR.SetActivationFunction(None);
   index++;
   if(!cCandidate.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   cCandidate.SetActivationFunction(None);
   index += 3;
   if(!cHCe.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   cHCe.SetActivationFunction(None);
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!Output.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetGCRU GPU buffers to the given OpenCL  context.  |
//+--------------------------------------------------------------------------+
void CNeuronHimNetGCRU::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cZ_R.SetOpenCL(OpenCL);
   cZ_R_emb.SetOpenCL(OpenCL);
   cHC.SetOpenCL(OpenCL);
   cHC_emb.SetOpenCL(OpenCL);
   cInpAndHidden.SetOpenCL(OpenCL);
   cZe_Re.SetOpenCL(OpenCL);
   cZ.SetOpenCL(OpenCL);
   cR.SetOpenCL(OpenCL);
   cCandidate.SetOpenCL(OpenCL);
   cHCe.SetOpenCL(OpenCL);
//---
   bSupportAccum.BufferCreate(OpenCL);
  }
//+---------------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetTempEncoder layer and allocates its  GPU buffers.  |
//+---------------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint units, uint window, uint window_out, uint cheb_k,
                                    uint layers, uint embed_dim, uint period1,
                                    uint timeframe1, uint period2, uint timeframe2,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(layers <= 0)
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!caEmbeddings[0].Init(0, index, OpenCL, embed_dim, period1, optimization, iBatch))
      ReturnFalse;
   aTimeframes[0] = MathMax(1, timeframe1);
   index++;
   if(!caEmbeddings[1].Init(0, index, OpenCL, embed_dim, period2, optimization, iBatch))
      ReturnFalse;
   aTimeframes[1] = MathMax(1, timeframe2);
   if(!cConcatEmbeddings.Init(numOutputs, myIndex, open_cl, 2 * embed_dim, optimization_type, batch))
      ReturnFalse;
//---
   cGRCUs.Clear();
   cGRCUs.SetOpenCL(OpenCL);
   index++;
   CNeuronHimNetGCRU *temp = new CNeuronHimNetGCRU();
   if(!temp ||
      !temp.Init(0, index, OpenCL, units, window, window_out, cheb_k, 2 * embed_dim, optimization, iBatch) ||
      !cGRCUs.Add(temp))
      ReturnFalse;
   for(uint i = 1; i < layers; i++)
     {
      index++;
      temp = new CNeuronHimNetGCRU();
      if(!temp ||
         !temp.Init(0, index, OpenCL, units, window_out, window_out, cheb_k, 2 * embed_dim, optimization, iBatch) ||
         !cGRCUs.Add(temp))
         ReturnFalse;
     }
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Runs the CNeuronHimNetTempEncoder feed-forward pass and writes  the layer output.  |
//+------------------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::feedForward(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support, int label)
  {
   for(uint i = 0; i < caEmbeddings.Size(); i++)
     {
      int position = (label / int(aTimeframes[i])) % caEmbeddings[i].GetPeriod();
      if(!caEmbeddings[i].SetPosition(position) ||
         !caEmbeddings[i].FeedForward())
         ReturnFalse;
     }
   if(!Concat(caEmbeddings[0].getOutput(), caEmbeddings[1].getOutput(),
              cConcatEmbeddings.getOutput(), 1, 1, caEmbeddings[0].Neurons()))
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = NeuronOCL;
   CNeuronHimNetGCRU *current = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current ||
         !current.feedForward(inputs, Support, cConcatEmbeddings.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetTempEncoder layer.  |
//+-----------------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support)
  {
   if(!NeuronOCL || !Support)
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = (cGRCUs.Total() > 1 ? cGRCUs[-2] : NeuronOCL);
   CNeuronHimNetGCRU* current = cGRCUs[-1];
   if(!current ||
      !current.calcInputGradients(inputs, Support, cConcatEmbeddings.AsObject()))
      ReturnFalse;
   if(!DeConcat(caEmbeddings[0].getGradient(), caEmbeddings[1].getGradient(),
                cConcatEmbeddings.getGradient(), 1, 1, caEmbeddings[0].Neurons()))
      ReturnFalse;
   if(cGRCUs.Total() > 1)
     {
      CBufferFloat *temp = Support.getGradient();
      if(!Support.SetGradient(GetPointer(bSupportAccum), false))
         ReturnFalse;
      for(int i = cGRCUs.Total() - 2; i >= 0; i--)
        {
         current = cGRCUs[i];
         inputs = (i > 0 ? cGRCUs[i - 1] : NeuronOCL);
         if(!current ||
            !current.calcInputGradients(inputs, Support, cConcatEmbeddings.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(temp, Support.getGradient(), temp, 1, false, 0, 0, 0, 1))
            ReturnFalse;
         if(!DeConcat(caEmbeddings[0].getPrevOutput(), caEmbeddings[1].getPrevOutput(),
                      cConcatEmbeddings.getGradient(), 1, 1, caEmbeddings[0].Neurons()))
            ReturnFalse;
         for(uint i = 0; i < caEmbeddings.Size(); i++)
            if(!SumAndNormalize(caEmbeddings[i].getGradient(), caEmbeddings[i].getPrevOutput(),
                                caEmbeddings[i].getGradient(), 1, false, 0, 0, 0, 1))
               ReturnFalse;
        }
      if(!Support.SetGradient(temp, false))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------------------------------+
//| Updates the CNeuronHimNetTempEncoder trainable weights from the  accumulated gradients.  |
//+------------------------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support)
  {
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].UpdateInputWeights())
         ReturnFalse;
//---
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronHimNetGCRU* current = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current.updateInputWeights(inputs, Support, cConcatEmbeddings.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+------------------------------------------------------------------------------------------+
//| Averages the CNeuronHimNetTempEncoder weights toward the source  model with factor tau.  |
//+------------------------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronHimNetTempEncoder* Source = source;
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      if(!caEmbeddings[i].WeightsUpdate(Source.caEmbeddings[i].AsObject(), tau))
         ReturnFalse;
   if(!cGRCUs.WeightsUpdate(Source.cGRCUs.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHimNetTempEncoder state to the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
     {
      if(!caEmbeddings[i].Save(file_handle))
         ReturnFalse;
      if(FileWriteInteger(file_handle, int(aTimeframes[i])) < INT_VALUE)
         ReturnFalse;
     }
   if(!cGRCUs.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Restores the CNeuronHimNetTempEncoder state from the file  handle.  |
//+---------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caEmbeddings.Size(); i++)
     {
      if(!LoadInsideLayer(file_handle, caEmbeddings[i].AsObject()))
         ReturnFalse;
      if(FileIsEnding(file_handle))
         ReturnFalse;
      aTimeframes[i] = (uint)FileReadInteger(file_handle);
     }
   if(!cGRCUs.Load(file_handle))
      ReturnFalse;
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   CNeuronHimNetGCRU* temp = cGRCUs[-1];
   uint units = temp.GetCount();
   uint cheb_k = temp.GetChebK();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetTempEncoder GPU buffers to the given  OpenCL context.  |
//+---------------------------------------------------------------------------------+
void CNeuronHimNetTempEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(uint i = 0; i < caEmbeddings.Size(); i++)
      caEmbeddings[i].SetOpenCL(OpenCL);
   cGRCUs.SetOpenCL(OpenCL);
//---
   bSupportAccum.BufferFree();
   bSupportAccum.BufferCreate(OpenCL);
  }
//+----------------------------------------------------------------------------+
//| Clears the CNeuronHimNetTempEncoder execution state and its GPU  buffers.  |
//+----------------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::Clear(void)
  {
   CNeuronBaseOCL* temp = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      temp = cGRCUs[i];
      if(!temp ||
         !temp.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Returns the count of the CNeuronHimNetTempEncoder.               |
//+------------------------------------------------------------------+
uint CNeuronHimNetTempEncoder::GetCount(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetCount();
  }
//+------------------------------------------------------------------+
//| Returns the cheb k of the CNeuronHimNetTempEncoder.              |
//+------------------------------------------------------------------+
uint CNeuronHimNetTempEncoder::GetChebK(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetChebK();
  }
//+------------------------------------------------------------------+
//| Returns the window in of the CNeuronHimNetTempEncoder.           |
//+------------------------------------------------------------------+
uint CNeuronHimNetTempEncoder::GetWindowIn(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetWindowIn();
  }
//+------------------------------------------------------------------+
//| Returns the window out of the CNeuronHimNetTempEncoder.          |
//+------------------------------------------------------------------+
uint CNeuronHimNetTempEncoder::GetWindowOut(void) const
  {
   uint count = GetCount();
   if(count <= 0)
      return 0;
//---
   return Neurons() / count;
  }
//+---------------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetSpatEncoder layer and allocates its  GPU buffers.  |
//+---------------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint units, uint window, uint window_out, uint cheb_k,
                                    uint layers, uint embed_dim,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(layers <= 0)
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cEmbedding.Init(0, index, OpenCL, embed_dim, optimization, iBatch))
      ReturnFalse;
//---
   cGRCUs.Clear();
   cGRCUs.SetOpenCL(OpenCL);
   index++;
   CNeuronHimNetGCRU *temp = new CNeuronHimNetGCRU();
   if(!temp ||
      !temp.Init(0, index, OpenCL, units, window, window_out, cheb_k, embed_dim, optimization, iBatch) ||
      !cGRCUs.Add(temp))
     {
      DeleteObj(temp);
      ReturnFalse;
     }
   for(uint i = 1; i < layers; i++)
     {
      index++;
      temp = new CNeuronHimNetGCRU();
      if(!temp ||
         !temp.Init(0, index, OpenCL, units, window_out, window_out, cheb_k, 2 * embed_dim, optimization, iBatch) ||
         !cGRCUs.Add(temp))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
     }
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.Clear();
   if(!bEmbeddingAccum.BufferInit(cEmbedding.Neurons(), 0) ||
      !bEmbeddingAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Runs the CNeuronHimNetSpatEncoder feed-forward pass and writes  the layer output.  |
//+------------------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::feedForward(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support)
  {
   if(bTrain)
      if(!cEmbedding.FeedForward())
         ReturnFalse;
//---
   CNeuronBaseOCL *inputs = NeuronOCL;
   CNeuronHimNetGCRU *current = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current ||
         !current.feedForward(inputs, Support, cEmbedding.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetSpatEncoder layer.  |
//+-----------------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support)
  {
   if(!NeuronOCL || !Support)
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = (cGRCUs.Total() > 1 ? cGRCUs[-2] : NeuronOCL);
   CNeuronHimNetGCRU* current = cGRCUs[-1];
   if(!current ||
      !current.calcInputGradients(inputs, Support, cEmbedding.AsObject()))
      ReturnFalse;
   if(cGRCUs.Total() > 1)
     {
      CBufferFloat *sup = Support.getGradient();
      if(!Support.SetGradient(GetPointer(bSupportAccum), false))
         ReturnFalse;
      CBufferFloat *emb = cEmbedding.getGradient();
      if(!cEmbedding.SetGradient(GetPointer(bEmbeddingAccum), false))
         ReturnFalse;
      for(int i = cGRCUs.Total() - 2; i >= 0; i--)
        {
         current = cGRCUs[i];
         inputs = (i > 0 ? cGRCUs[i - 1] : NeuronOCL);
         if(!current ||
            !current.calcInputGradients(inputs, Support, cEmbedding.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(sup, Support.getGradient(), sup, 1, false, 0, 0, 0, 1))
            ReturnFalse;
         if(!SumAndNormalize(cEmbedding.getGradient(), emb, emb, 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(!Support.SetGradient(sup, false))
         ReturnFalse;
      if(!cEmbedding.SetGradient(emb, false))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------------------------------+
//| Updates the CNeuronHimNetSpatEncoder trainable weights from the  accumulated gradients.  |
//+------------------------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support)
  {
   if(!cEmbedding.UpdateInputWeights())
      ReturnFalse;
//---
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronHimNetGCRU* current = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current.updateInputWeights(inputs, Support, cEmbedding.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+------------------------------------------------------------------------------------------+
//| Averages the CNeuronHimNetSpatEncoder weights toward the source  model with factor tau.  |
//+------------------------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronHimNetSpatEncoder* Source = source;
   if(!cEmbedding.WeightsUpdate(Source.cEmbedding.AsObject(), tau))
      ReturnFalse;
   if(!cGRCUs.WeightsUpdate(Source.cGRCUs.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHimNetSpatEncoder state to the file handle.     |
//+------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cEmbedding.Save(file_handle))
      ReturnFalse;
   if(!cGRCUs.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Restores the CNeuronHimNetSpatEncoder state from the file  handle.  |
//+---------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cEmbedding.AsObject()))
      ReturnFalse;
   if(!cGRCUs.Load(file_handle))
      ReturnFalse;
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   CNeuronHimNetGCRU* temp = cGRCUs[-1];
   uint units = temp.GetCount();
   uint cheb_k = temp.GetChebK();
   if(!bSupportAccum.BufferInit(units * units * cheb_k, 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.Clear();
   if(!bEmbeddingAccum.BufferInit(cEmbedding.Neurons(), 0) ||
      !bEmbeddingAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetSpatEncoder GPU buffers to the given  OpenCL context.  |
//+---------------------------------------------------------------------------------+
void CNeuronHimNetSpatEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cEmbedding.SetOpenCL(OpenCL);
   cGRCUs.SetOpenCL(OpenCL);
//---
   bSupportAccum.BufferFree();
   bSupportAccum.BufferCreate(OpenCL);
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.BufferCreate(OpenCL);
  }
//+----------------------------------------------------------------------------+
//| Clears the CNeuronHimNetSpatEncoder execution state and its GPU  buffers.  |
//+----------------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::Clear(void)
  {
   CNeuronBaseOCL* temp = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      temp = cGRCUs[i];
      if(!temp ||
         !temp.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Returns the count of the CNeuronHimNetSpatEncoder.               |
//+------------------------------------------------------------------+
uint CNeuronHimNetSpatEncoder::GetCount(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetCount();
  }
//+------------------------------------------------------------------+
//| Returns the cheb k of the CNeuronHimNetSpatEncoder.              |
//+------------------------------------------------------------------+
uint CNeuronHimNetSpatEncoder::GetChebK(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetChebK();
  }
//+------------------------------------------------------------------+
//| Returns the window in of the CNeuronHimNetSpatEncoder.           |
//+------------------------------------------------------------------+
uint CNeuronHimNetSpatEncoder::GetWindowIn(void) const
  {
   if(cGRCUs.Total() <= 0)
      return 0;
   CNeuronHimNetGCRU* temp = cGRCUs[0];
   if(!temp)
      return 0;
//---
   return temp.GetWindowIn();
  }
//+------------------------------------------------------------------+
//| Returns the window out of the CNeuronHimNetSpatEncoder.          |
//+------------------------------------------------------------------+
uint CNeuronHimNetSpatEncoder::GetWindowOut(void) const
  {
   uint count = GetCount();
   if(count <= 0)
      return 0;
//---
   return Neurons() / count;
  }
//+------------------------------------------------------------------+
//| Sets the gradient of the CNeuronHimNetSpatEncoder.               |
//+------------------------------------------------------------------+
bool CNeuronHimNetSpatEncoder::SetGradient(CBufferFloat *buffer, bool delete_prev = true)
  {
   if(!CNeuronBaseOCL::SetGradient(buffer, delete_prev))
      ReturnFalse;
//---
   CNeuronBaseOCL* temp = cGRCUs[-1];
   if(!temp ||
      !temp.SetGradient(Gradient, delete_prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Sets the gradient of the CNeuronHimNetTempEncoder.               |
//+------------------------------------------------------------------+
bool CNeuronHimNetTempEncoder::SetGradient(CBufferFloat *buffer, bool delete_prev = true)
  {
   if(!CNeuronBaseOCL::SetGradient(buffer, delete_prev))
      ReturnFalse;
//---
   CNeuronBaseOCL* temp = cGRCUs[-1];
   if(!temp ||
      !temp.SetGradient(Gradient, delete_prev))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetEncoder layer and allocates its GPU  buffers.  |
//+-----------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint units, uint window, uint window_out, uint cheb_k,
                                uint layers, uint embed_dim, uint period1,
                                uint timeframe1, uint period2, uint timeframe2,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cEmbedding.Init(0, index, OpenCL, units * embed_dim, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(TANH);
   index++;
   if(!cEmbeddingT.Init(0, index, OpenCL, units, embed_dim, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSupport.Init(0, index, OpenCL, units * units, optimization, iBatch))
      ReturnFalse;
   cSupport.SetActivationFunction(None);
   index++;
   if(!cNormSupport.Init(0, index, OpenCL, cSupport.Neurons(), optimization, iBatch))
      ReturnFalse;
   cNormSupport.SetHeads(units);
   cNormSupport.SetActivationFunction(None);
   index++;
   if(!cPolinomSupport.Init(0, index, OpenCL, units, cheb_k, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTempEncoder.Init(0, index, OpenCL, units, window, window_out, cheb_k, layers, (embed_dim + 1) / 2, period1, timeframe1, period2, timeframe2, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSpatEncoder.Init(0, index, OpenCL, units, window, window_out, cheb_k, layers, embed_dim, optimization, iBatch))
      ReturnFalse;
//---
   if(!SetGradient(cTempEncoder.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Runs the CNeuronHimNetEncoder feed-forward pass and writes the  layer output.  |
//+--------------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!SecondInput)
      ReturnFalse;
//---
   if(bTrain)
     {
      if(!cEmbedding.FeedForward())
         ReturnFalse;
      if(!cEmbeddingT.FeedForward(cEmbedding.AsObject()))
         ReturnFalse;
      if(!MatMul(cEmbedding.getOutput(), cEmbeddingT.getOutput(), cSupport.getOutput(),
                 cEmbeddingT.GetCount(), cEmbeddingT.GetWindow(), cEmbeddingT.GetCount(), 1, false))
         ReturnFalse;
      if(!cNormSupport.FeedForward(cSupport.AsObject()))
         ReturnFalse;
      if(!cPolinomSupport.FeedForward(cNormSupport.AsObject()))
         ReturnFalse;
     }
//---
   if(!cTempEncoder.feedForward(NeuronOCL, cPolinomSupport.AsObject(), int(SecondInput[0])))
      ReturnFalse;
   if(!cSpatEncoder.feedForward(NeuronOCL, cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cTempEncoder.getOutput(), cSpatEncoder.getOutput(), Output,
                       cTempEncoder.GetWindowOut(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetEncoder layer.  |
//+-------------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::calcInputGradients(CNeuronBaseOCL *prevLayer)
  {
   if(!cTempEncoder.calcInputGradients(prevLayer, cPolinomSupport.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = cPolinomSupport.getGradient();
   CBufferFloat* prev = prevLayer.getGradient();
   if(!cPolinomSupport.SetGradient(cPolinomSupport.getPrevOutput(), false) ||
      !prevLayer.SetGradient(prevLayer.getPrevOutput(), false) ||
      !cSpatEncoder.calcInputGradients(prevLayer, cPolinomSupport.AsObject()) ||
      !SumAndNormalize(temp, cPolinomSupport.getGradient(), temp,
                       cSpatEncoder.GetCount(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(prev, prevLayer.getGradient(), prev,
                       cSpatEncoder.GetCount(), false, 0, 0, 0, 1) ||
      !cPolinomSupport.SetGradient(temp, false) ||
      !prevLayer.SetGradient(prev, false))
      ReturnFalse;
   if(prevLayer.Activation() != None)
      if(!DeActivation(prevLayer.getOutput(), prev, prev, prevLayer.Activation()))
         ReturnFalse;
//---
   if(!cNormSupport.CalcHiddenGradients(cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!cSupport.CalcHiddenGradients(cNormSupport.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(cEmbedding.getOutput(), cEmbedding.getPrevOutput(),
                  cEmbeddingT.getOutput(), cEmbeddingT.getGradient(),
                  cSupport.getGradient(),
                  cEmbeddingT.GetCount(), cEmbeddingT.GetWindow(), cEmbeddingT.GetCount(), 1, false))
      ReturnFalse;
   if(!cEmbedding.CalcHiddenGradients(cEmbeddingT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cEmbedding.getGradient(), cEmbedding.getPrevOutput(), cEmbedding.getGradient(),
                       cEmbeddingT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(cEmbedding.Activation() != None)
      if(!DeActivation(cEmbedding.getOutput(), cEmbedding.getGradient(),
                       cEmbedding.getGradient(), cEmbedding.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------------+
//| Updates the CNeuronHimNetEncoder trainable weights from the  accumulated gradients.  |
//+--------------------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEmbedding.UpdateInputWeights())
      ReturnFalse;
   if(!cTempEncoder.updateInputWeights(NeuronOCL, cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!cSpatEncoder.updateInputWeights(NeuronOCL, cPolinomSupport.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------------+
//| Averages the CNeuronHimNetEncoder weights toward the source  model  with factor tau.  |
//+---------------------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronHimNetEncoder* Source = source;
   if(!cEmbedding.WeightsUpdate(Source.cEmbedding.AsObject(), tau))
      ReturnFalse;
   if(!cTempEncoder.WeightsUpdate(Source.cTempEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cSpatEncoder.WeightsUpdate(Source.cSpatEncoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHimNetEncoder state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronHimNetEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cEmbedding.Save(file_handle))
      ReturnFalse;
   if(!cEmbeddingT.Save(file_handle))
      ReturnFalse;
   if(!cSupport.Save(file_handle))
      ReturnFalse;
   if(!cNormSupport.Save(file_handle))
      ReturnFalse;
   if(!cPolinomSupport.Save(file_handle))
      ReturnFalse;
   if(!cTempEncoder.Save(file_handle))
      ReturnFalse;
   if(!cSpatEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHimNetEncoder state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronHimNetEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cEmbedding.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEmbeddingT.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSupport.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormSupport.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTempEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpatEncoder.AsObject()))
      ReturnFalse;
//---
   if(!SetGradient(cTempEncoder.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetEncoder GPU buffers to the given OpenCL  context.  |
//+-----------------------------------------------------------------------------+
void CNeuronHimNetEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cEmbedding.SetOpenCL(OpenCL);
   cEmbeddingT.SetOpenCL(OpenCL);
   cSupport.SetOpenCL(OpenCL);
   cNormSupport.SetOpenCL(OpenCL);
   cPolinomSupport.SetOpenCL(OpenCL);
   cTempEncoder.SetOpenCL(OpenCL);
   cSpatEncoder.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronHimNetEncoder layer.        |
//+------------------------------------------------------------------+
void CNeuronHimNetEncoder::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cEmbedding.TrainMode(bTrain);
   cEmbeddingT.TrainMode(bTrain);
   cSupport.TrainMode(bTrain);
   cNormSupport.TrainMode(bTrain);
   cPolinomSupport.TrainMode(bTrain);
   cTempEncoder.TrainMode(bTrain);
   cSpatEncoder.TrainMode(bTrain);
  }
//+------------------------------------------------------------------------+
//| Clears the CNeuronHimNetEncoder execution state and its GPU  buffers.  |
//+------------------------------------------------------------------------+
bool CNeuronHimNetEncoder::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cEmbedding.Clear())
      ReturnFalse;
   if(!cEmbeddingT.Clear())
      ReturnFalse;
   if(!cSupport.Clear())
      ReturnFalse;
   if(!cNormSupport.Clear())
      ReturnFalse;
   if(!cPolinomSupport.Clear())
      ReturnFalse;
   if(!cTempEncoder.Clear())
      ReturnFalse;
   if(!cSpatEncoder.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Sets the gradient of the CNeuronHimNetEncoder.                   |
//+------------------------------------------------------------------+
bool CNeuronHimNetEncoder::SetGradient(CBufferFloat *buffer, bool delete_prev = true)
  {
   if(!CNeuronBaseOCL::SetGradient(buffer, delete_prev))
      ReturnFalse;
//---
   if(!cTempEncoder.SetGradient(Gradient, delete_prev))
      ReturnFalse;
   if(!cSpatEncoder.SetGradient(Gradient, delete_prev))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Initializes the CNeuronHimNetDecoder layer and allocates its GPU  buffers.  |
//+-----------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint units, uint window, uint window_out,
                                uint cheb_k, uint layers, uint embed_dim,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cProjection.Init(0, index, OpenCL, window, window, embed_dim, units, 1, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(TANH);
   index++;
   if(!cProjectionT.Init(0, index, OpenCL, units, embed_dim, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEmbedding.Init(0, index, OpenCL, units, units, 1, 1, embed_dim, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(SIGMOID);
   index++;
   if(!cSupport.Init(0, index, OpenCL, units * units, optimization, iBatch))
      ReturnFalse;
   cSupport.SetActivationFunction(None);
   index++;
   if(!cNormSupport.Init(0, index, OpenCL, cSupport.Neurons(), optimization, iBatch))
      ReturnFalse;
   cNormSupport.SetHeads(units);
   cNormSupport.SetActivationFunction(None);
   index++;
   if(!cPolinomSupport.Init(0, index, OpenCL, units, cheb_k, optimization, iBatch))
      ReturnFalse;
//---
   cGRCUs.Clear();
   cGRCUs.SetOpenCL(OpenCL);
   index++;
   CNeuronHimNetGCRU *temp = new CNeuronHimNetGCRU();
   if(!temp ||
      !temp.Init(0, index, OpenCL, units, window, window_out, cheb_k, embed_dim, optimization, iBatch) ||
      !cGRCUs.Add(temp))
     {
      DeleteObj(temp);
      ReturnFalse;
     }
   for(uint i = 1; i < layers; i++)
     {
      index++;
      temp = new CNeuronHimNetGCRU();
      if(!temp ||
         !temp.Init(0, index, OpenCL, units, window_out, window_out, cheb_k, 2 * embed_dim, optimization, iBatch) ||
         !cGRCUs.Add(temp))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
     }
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   if(!bSupportAccum.BufferInit(cPolinomSupport.Neurons(), 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.Clear();
   if(!bEmbeddingAccum.BufferInit(cEmbedding.Neurons(), 0) ||
      !bEmbeddingAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Runs the CNeuronHimNetDecoder feed-forward pass and writes the  layer output.  |
//+--------------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cProjectionT.FeedForward(cProjection.AsObject()))
      ReturnFalse;
   if(!MatMul(cProjection.getOutput(), cProjectionT.getOutput(), cSupport.getOutput(),
              cProjectionT.GetCount(), cProjectionT.GetWindow(), cProjectionT.GetCount(), 1, false))
      ReturnFalse;
   if(!cNormSupport.FeedForward(cSupport.AsObject()))
      ReturnFalse;
   if(!cPolinomSupport.FeedForward(cNormSupport.AsObject()))
      ReturnFalse;
//---
   if(!cEmbedding.FeedForward(cProjectionT.AsObject()))
      ReturnFalse;
//---
   CNeuronHimNetGCRU* current = NULL;
   CNeuronBaseOCL* inputs = NeuronOCL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current ||
         !current.feedForward(NeuronOCL, cPolinomSupport.AsObject(), cEmbedding.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronHimNetDecoder layer.  |
//+-------------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::calcInputGradients(CNeuronBaseOCL *prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = (cGRCUs.Total() > 1 ? cGRCUs[-2] : prevLayer);
   CNeuronHimNetGCRU* current = cGRCUs[-1];
   if(!current ||
      !current.calcInputGradients(inputs, cPolinomSupport.AsObject(), cEmbedding.AsObject()))
      ReturnFalse;
   if(cGRCUs.Total() > 1)
     {
      CBufferFloat *sup = cPolinomSupport.getGradient();
      if(!cPolinomSupport.SetGradient(GetPointer(bSupportAccum), false))
         ReturnFalse;
      CBufferFloat *emb = cEmbedding.getGradient();
      if(!cEmbedding.SetGradient(GetPointer(bEmbeddingAccum), false))
         ReturnFalse;
      for(int i = cGRCUs.Total() - 2; i >= 0; i--)
        {
         current = cGRCUs[i];
         inputs = (i > 0 ? cGRCUs[i - 1] : prevLayer);
         if(!current ||
            !current.calcInputGradients(inputs, cPolinomSupport.AsObject(), cEmbedding.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(cPolinomSupport.getGradient(), sup, sup, 1, false, 0, 0, 0, 1))
            ReturnFalse;
         if(!SumAndNormalize(cEmbedding.getGradient(), emb, emb, 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(!cPolinomSupport.SetGradient(sup, false))
         ReturnFalse;
      if(!cEmbedding.SetGradient(emb, false))
         ReturnFalse;
     }
   if(!cNormSupport.CalcHiddenGradients(cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!cSupport.CalcHiddenGradients(cNormSupport.AsObject()))
      ReturnFalse;
   if(!MatMulGrad(cProjection.getOutput(), cProjection.getPrevOutput(),
                  cProjectionT.getOutput(), cProjectionT.getPrevOutput(),
                  cSupport.getGradient(), cProjectionT.GetCount(),
                  cProjectionT.GetWindow(), cProjectionT.GetCount(), 1, false))
      ReturnFalse;
   if(!cProjectionT.CalcHiddenGradients(cEmbedding.AsObject()))
      ReturnFalse;
   if(cProjectionT.Activation() != None)
      if(!DeActivation(cProjectionT.getOutput(), cProjectionT.getPrevOutput(),
                       cProjectionT.getPrevOutput(), cProjectionT.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cProjectionT.getGradient(), cProjectionT.getPrevOutput(),
                       cProjectionT.getGradient(), cProjectionT.GetCount(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cProjection.CalcHiddenGradients(cProjectionT.AsObject()))
      ReturnFalse;
   if(cProjection.Activation() != None)
      if(!DeActivation(cProjection.getOutput(), cProjection.getPrevOutput(),
                       cProjection.getPrevOutput(), cProjection.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cProjection.getGradient(), cProjection.getPrevOutput(),
                       cProjection.getGradient(), cProjectionT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   CBufferFloat* temp = prevLayer.getGradient();
   if(!prevLayer.SetGradient(prevLayer.getPrevOutput(), false) ||
      !prevLayer.CalcHiddenGradients(cProjection.AsObject()) ||
      !SumAndNormalize(temp, prevLayer.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !prevLayer.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------------+
//| Updates the CNeuronHimNetDecoder trainable weights from the  accumulated gradients.  |
//+--------------------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cEmbedding.UpdateInputWeights(cProjectionT.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* inputs = NeuronOCL;
   CNeuronHimNetGCRU* current = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      current = cGRCUs[i];
      if(!current.updateInputWeights(inputs, cPolinomSupport.AsObject(), cEmbedding.AsObject()))
         ReturnFalse;
      inputs = current;
     }
//---
   return true;
  }
//+---------------------------------------------------------------------------------------+
//| Averages the CNeuronHimNetDecoder weights toward the source  model  with factor tau.  |
//+---------------------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronHimNetDecoder* Source = source;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   if(!cEmbedding.WeightsUpdate(Source.cEmbedding.AsObject(), tau))
      ReturnFalse;
   if(!cGRCUs.WeightsUpdate(Source.cGRCUs.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronHimNetDecoder state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronHimNetDecoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cProjectionT.Save(file_handle))
      ReturnFalse;
   if(!cEmbedding.Save(file_handle))
      ReturnFalse;
   if(!cSupport.Save(file_handle))
      ReturnFalse;
   if(!cNormSupport.Save(file_handle))
      ReturnFalse;
   if(!cPolinomSupport.Save(file_handle))
      ReturnFalse;
   if(!cGRCUs.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronHimNetDecoder state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronHimNetDecoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjectionT.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEmbedding.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSupport.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormSupport.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPolinomSupport.AsObject()))
      ReturnFalse;
   if(!cGRCUs.Load(file_handle))
      ReturnFalse;
//---
   bSupportAccum.BufferFree();
   bSupportAccum.Clear();
   CNeuronHimNetGCRU* temp = cGRCUs[-1];
   if(!bSupportAccum.BufferInit(cPolinomSupport.Neurons(), 0) ||
      !bSupportAccum.BufferCreate(OpenCL))
      ReturnFalse;
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.Clear();
   if(!bEmbeddingAccum.BufferInit(cEmbedding.Neurons(), 0) ||
      !bEmbeddingAccum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   SetActivationFunction((ENUM_ACTIVATION)temp.Activation());
   if(!SetOutput(temp.getOutput(), true) ||
      !SetGradient(temp.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Rebinds the CNeuronHimNetDecoder GPU buffers to the given OpenCL  context.  |
//+-----------------------------------------------------------------------------+
void CNeuronHimNetDecoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cProjection.SetOpenCL(OpenCL);
   cProjectionT.SetOpenCL(OpenCL);
   cEmbedding.SetOpenCL(OpenCL);
   cSupport.SetOpenCL(OpenCL);
   cNormSupport.SetOpenCL(OpenCL);
   cPolinomSupport.SetOpenCL(OpenCL);
   cGRCUs.SetOpenCL(OpenCL);
//---
   bSupportAccum.BufferFree();
   bSupportAccum.BufferCreate(OpenCL);
   bEmbeddingAccum.BufferFree();
   bEmbeddingAccum.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------------+
//| Clears the CNeuronHimNetDecoder execution state and its GPU  buffers.  |
//+------------------------------------------------------------------------+
bool CNeuronHimNetDecoder::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cProjection.Clear())
      ReturnFalse;
   if(!cProjectionT.Clear())
      ReturnFalse;
   if(!cEmbedding.Clear())
      ReturnFalse;
   if(!cSupport.Clear())
      ReturnFalse;
   if(!cNormSupport.Clear())
      ReturnFalse;
   if(!cPolinomSupport.Clear())
      ReturnFalse;
   CNeuronBaseOCL* temp = NULL;
   for(int i = 0; i < cGRCUs.Total(); i++)
     {
      temp = cGRCUs[i];
      if(!temp || !temp.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Initializes the CNeuronFastGConv layer and allocates its GPU  buffers.  |
//+-------------------------------------------------------------------------+
bool CNeuronFastGConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint units, uint window, uint sparse_dimension,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * window, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cInpAndHidden.Init(0, index, OpenCL, 2 * units * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormAttention.Init(0, index, OpenCL, units * sparse_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cInvDiag.Init(0, index, OpenCL, units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAXplusX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormAXplusX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cZ_R.Init(0, index, OpenCL, 2 * window, 2 * window, 2 * window, units, 1, optimization, iBatch))
      ReturnFalse;
   cZ_R.SetActivationFunction(SIGMOID);
   index++;
   if(!cZ.Init(0, index, OpenCL, window * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cR.Init(0, index, OpenCL, window * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cCandidate.Init(0, index, OpenCL, 2 * window * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHC.Init(0, index, OpenCL, 2 * window, 2 * window, window, units, 1, optimization, iBatch))
      ReturnFalse;
   cHC.SetActivationFunction(TANH);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronFastGConv random walk operation.              |
//+------------------------------------------------------------------+
bool CNeuronFastGConv::RandomWalk(CBufferFloat *data,
                                  CBufferFloat *normal,
                                  CBufferFloat *inv_diag,
                                  const int rows,
                                  const int cols)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {(uint)rows, MathMin((uint)cols, (uint)OpenCL.GetMaxLocalSize(1))};
   uint local_work_size[2] = {1, global_work_size[1]};
   uint kernel = def_k_RandomWalk;
   setBuffer(kernel, def_k_rw_data, data.GetIndex())
   setBuffer(kernel, def_k_rw_norm, normal.GetIndex())
   setBuffer(kernel, def_k_rw_inv_diag, inv_diag.GetIndex())
   setArgument(kernel, def_k_rw_total_cols, cols)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!inv_diag.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronFastGConv feed forward operation.             |
//+------------------------------------------------------------------+
bool CNeuronFastGConv::FeedForward(CNeuronBaseOCL *SourceData, CNeuronSNSMHAttention *SparseAttent)
  {
   if(!SourceData || !SparseAttent)
      ReturnFalse;
//---
   const uint units = GetCount();
   const uint window = GetWindow();
   const uint sparse = GetSparseDimension();
//---
   if(!RandomWalk(SparseAttent.getOutput(), cNormAttention.getOutput(), cInvDiag.getOutput(), window, sparse))
      ReturnFalse;
   if(!SwapOutputs())
      ReturnFalse;
   if(!Concat(SourceData.getOutput(), PrevOutput, cInpAndHidden.getOutput(), window, window, units))
      ReturnFalse;
//---
   if(!SparseMatMul(SparseAttent.GetIndexes(), cNormAttention.getOutput(), cInpAndHidden.getOutput(),
                    cAX.getOutput(), units, sparse, units, 2 * window))
      ReturnFalse;
   if(!SumAndNormalize(cAX.getOutput(), cInpAndHidden.getOutput(), cAXplusX.getOutput(), 2 * window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!DiagMatMul(cInvDiag.getOutput(), cAXplusX.getOutput(), cNormAXplusX.getOutput(), units, 2 * window, 1, None))
      ReturnFalse;
   if(!cZ_R.FeedForward(cNormAXplusX.AsObject()))
      ReturnFalse;
   if(!DeConcat(cZ.getOutput(), cR.getOutput(), cZ_R.getOutput(), window, window, units))
      ReturnFalse;
   if(!ElementMult(cR.getOutput(), PrevOutput, cR.getPrevOutput()))
      ReturnFalse;
   if(!Concat(SourceData.getOutput(), cR.getPrevOutput(), cCandidate.getOutput(), window, window, units))
      ReturnFalse;
   if(!cHC.FeedForward(cCandidate.AsObject()))
      ReturnFalse;
   if(!GateElementMult(PrevOutput, cHC.getOutput(), cZ.getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronFastGConv calc input gradients operation.     |
//+------------------------------------------------------------------+
bool CNeuronFastGConv::CalcInputGradients(CNeuronBaseOCL *SourceData, CNeuronSNSMHAttention *SparseAttent)
  {
   if(!SourceData || !SparseAttent)
      ReturnFalse;
//---
   const uint units = GetCount();
   const uint window = GetWindow();
   const uint sparse = GetSparseDimension();
//---
   if(!GateElementMultGrad(PrevOutput, cInpAndHidden.getGradient(),
                           cHC.getOutput(), cHC.getGradient(),
                           cZ.getOutput(), cZ.getGradient(),
                           Gradient, None, cHC.Activation(), cZ_R.Activation()))
      ReturnFalse;
   if(!cCandidate.CalcHiddenGradients(cHC.AsObject()))
      ReturnFalse;
   if(!DeConcat(SourceData.getGradient(), cR.getPrevOutput(),
                cCandidate.getGradient(), window, window, units))
      ReturnFalse;
   if(!ElementMultGrad(cR.getOutput(), cR.getGradient(),
                       PrevOutput, cInpAndHidden.getGradient(),
                       cR.getPrevOutput(), cZ_R.Activation(), None))
      ReturnFalse;
   if(!Concat(cZ.getGradient(), cR.getGradient(),
              cZ_R.getGradient(), window, window, units))
      ReturnFalse;
   if(!cNormAXplusX.CalcHiddenGradients(cZ_R.AsObject()))
      ReturnFalse;
   if(!DiagMatMulGrad(cInvDiag.getOutput(), cInvDiag.getGradient(),
                      cAXplusX.getOutput(), cAX.getGradient(),
                      cNormAXplusX.getGradient(), units, 2 * window, 1))
      ReturnFalse;
   if(!SparseMatMulGrad(SparseAttent.GetIndexes(), cNormAttention.getOutput(),
                        cNormAttention.getGradient(), cInpAndHidden.getOutput(),
                        cInpAndHidden.getGradient(), cAX.getGradient(),
                        units, sparse, units, 2 * window))
      ReturnFalse;
   if(!SumAndNormalize(cAX.getGradient(), cInpAndHidden.getGradient(),
                       cInpAndHidden.getGradient(), 2 * window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!DeConcat(cAXplusX.getGradient(), cAXplusX.getPrevOutput(),
                cInpAndHidden.getGradient(), window, window, units))
      ReturnFalse;
   if(!SumAndNormalize(SourceData.getGradient(), cAXplusX.getGradient(),
                       SourceData.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(SourceData.Activation() != None)
      if(!DeActivation(SourceData.getOutput(), SourceData.getGradient(),
                       SourceData.getGradient(), SourceData.Activation()))
         ReturnFalse;
   if(!DiagMatMul(cInvDiag.getOutput(), cNormAttention.getGradient(),
                  SparseAttent.getGradient(), units, sparse, 1, None))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Updates the CNeuronFastGConv trainable weights from the  accumulated gradients.  |
//+----------------------------------------------------------------------------------+
bool CNeuronFastGConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//---
   if(!cZ_R.UpdateInputWeights(cNormAXplusX.AsObject()))
      ReturnFalse;
   if(!cHC.UpdateInputWeights(cCandidate.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Averages the CNeuronFastGConv weights toward the source model  with factor tau.  |
//+----------------------------------------------------------------------------------+
bool CNeuronFastGConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronFastGConv *Source = source;
   if(!cZ_R.WeightsUpdate(Source.cZ_R.AsObject(), tau))
      ReturnFalse;
   if(!cHC.WeightsUpdate(Source.cHC.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronFastGConv state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronFastGConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cZ_R.Save(file_handle))
      ReturnFalse;
   if(!cHC.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)GetSparseDimension()) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronFastGConv state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronFastGConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cZ_R.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHC.AsObject()))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   uint sparse_dimension = FileReadInteger(file_handle);
   uint units = cZ_R.GetUnits();
   uint window_out = Neurons() / units;
   uint window = cZ_R.GetWindow() - window_out;
//---
   int index = 0;
   if(!cInpAndHidden.Init(0, index, OpenCL, units * (window + window_out), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormAttention.Init(0, index, OpenCL, units * sparse_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cInvDiag.Init(0, index, OpenCL, units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAXplusX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormAXplusX.Init(0, index, OpenCL, cInpAndHidden.Neurons(), optimization, iBatch))
      ReturnFalse;
   index += 2;
   if(!cZ.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cR.Init(0, index, OpenCL, window_out * units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cCandidate.Init(0, index, OpenCL, (window + window_out) * units, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Rebinds the CNeuronFastGConv GPU buffers to the given OpenCL  context.  |
//+-------------------------------------------------------------------------+
void CNeuronFastGConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cZ_R.SetOpenCL(OpenCL);
   cHC.SetOpenCL(OpenCL);
   cInpAndHidden.SetOpenCL(OpenCL);
   cNormAttention.SetOpenCL(OpenCL);
   cInvDiag.SetOpenCL(OpenCL);
   cAX.SetOpenCL(OpenCL);
   cAXplusX.SetOpenCL(OpenCL);
   cNormAXplusX.SetOpenCL(OpenCL);
   cZ.SetOpenCL(OpenCL);
   cR.SetOpenCL(OpenCL);
   cCandidate.SetOpenCL(OpenCL);
  }
//+----------------------------------------------------------------------+
//| Initializes the CNeuronSAGDFN layer and allocates its GPU  buffers.  |
//+----------------------------------------------------------------------+
bool CNeuronSAGDFN::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint time_steps, uint variables, uint embedding_dim,
                         uint emb_layers, uint sparse_dimension, uint heads,
                         float sparse, uint gcru_layers,
                         uint forecast, uint forec_layers,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(emb_layers <= 0 || gcru_layers <= 0 || forec_layers <= 0)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, variables, forecast, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cTranspose.Init(0, index, OpenCL, time_steps, variables, optimization, iBatch))
      ReturnFalse;
//--- Embedding
   cEmbedding.Clear();
   cEmbedding.SetOpenCL(OpenCL);
   index++;
   CNeuronConvOCL *conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, time_steps, time_steps, embedding_dim, variables, 1, optimization, iBatch) ||
      !cEmbedding.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   for(uint i = 1; i < emb_layers; i++)
     {
      index++;
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, embedding_dim, embedding_dim,
                    embedding_dim, variables, 1, optimization, iBatch) ||
         !cEmbedding.Add(conv))
        {
         DeleteObj(conv);
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
     }
   CNeuronBatchNormOCL *norm = new CNeuronBatchNormOCL();
   index++;
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cEmbedding.Add(norm))
     {
      DeleteObj(norm);
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
//--- GCRUs
   cGCRU.Clear();
   cGCRU.SetOpenCL(OpenCL);
   index++;
   if(!cAttention.Init(0, index, OpenCL, variables, embedding_dim, heads,
                       sparse_dimension, sparse, optimization, iBatch))
      ReturnFalse;
   CNeuronFastGConv *gcru = NULL;
   for(uint i = 0; i < gcru_layers; i++)
     {
      index++;
      gcru = new CNeuronFastGConv();
      if(!gcru ||
         !gcru.Init(0, index, OpenCL, variables, embedding_dim, sparse_dimension, optimization, iBatch) ||
         !cGCRU.Add(gcru))
        {
         DeleteObj(gcru);
         ReturnFalse;
        }
     }
//--- Forecast
   cProjection.Clear();
   cProjection.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, embedding_dim, embedding_dim,
                 forecast, variables, 1, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   for(uint i = 1; i < forec_layers; i++)
     {
      index++;
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, forecast, forecast,
                    forecast, variables, 1, optimization, iBatch) ||
         !cProjection.Add(conv))
        {
         DeleteObj(conv);
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
     }
   norm = new CNeuronBatchNormOCL();
   index++;
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cProjection.Add(norm))
     {
      DeleteObj(norm);
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Runs the CNeuronSAGDFN feed-forward pass and writes the layer  output.  |
//+-------------------------------------------------------------------------+
bool CNeuronSAGDFN::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL *inputs = cTranspose.AsObject();
   CNeuronBaseOCL *current = NULL;
//--- Embedding
   for(int i = 0; i < cEmbedding.Total(); i++)
     {
      current = cEmbedding[i];
      if(!current ||
         !current.FeedForward(inputs))
         ReturnFalse;
      inputs = current;
     }
//--- GCRUs
   if(!cAttention.FeedForward(inputs))
      ReturnFalse;
   CNeuronFastGConv *gcru = NULL;
   for(int i = 0; i < cGCRU.Total(); i++)
     {
      gcru = cGCRU[i];
      if(!gcru ||
         !gcru.FeedForward(inputs, cAttention.AsObject()))
         ReturnFalse;
      inputs = gcru;
     }
//--- Forecast
   for(int i = 0; i < cProjection.Total(); i++)
     {
      current = cProjection[i];
      if(!current ||
         !current.FeedForward(inputs))
         ReturnFalse;
      inputs = current;
     }
//--- result
   return CNeuronTransposeOCL::feedForward(inputs);
  }
//+------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronSAGDFN  layer.  |
//+------------------------------------------------------------------------+
bool CNeuronSAGDFN::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::calcInputGradients(cProjection[-1]))
      ReturnFalse;
//--- Forecast
   CNeuronBaseOCL *inputs = NULL;
   for(int i = cProjection.Total() - 1; i >= 0; i--)
     {
      inputs = (i > 0 ? cProjection[i - 1] : cGCRU[-1]);
      if(!inputs ||
         !inputs.CalcHiddenGradients(cProjection[i]))
         ReturnFalse;
     }
//--- GCRU
   CNeuronFastGConv *gcru = cGCRU[-1];
   int layers = cGCRU.Total();
   inputs = (layers > 1 ? cGCRU[-2] : cEmbedding[-1]);
   if(!gcru ||
      !gcru.CalcInputGradients(inputs, cAttention.AsObject()))
      ReturnFalse;
   if(layers > 1)
     {
      CBufferFloat *temp = cAttention.getGradient();
      if(!cAttention.SetGradient(cAttention.getPrevOutput(), false))
         ReturnFalse;
      for(int i = layers - 2; i >= 0; i--)
        {
         inputs = (i > 0 ? cGCRU[i - 1] : cEmbedding[-1]);
         gcru = cGCRU[i];
         if(!gcru ||
            !gcru.CalcInputGradients(inputs, cAttention.AsObject()) ||
            !SumAndNormalize(temp, cAttention.getGradient(), temp, 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(!cAttention.SetGradient(temp, false))
         ReturnFalse;
     }
   CBufferFloat *temp = inputs.getGradient();
   if(!inputs.SetGradient(inputs.getPrevOutput(), false) ||
      !inputs.CalcHiddenGradients(cAttention.AsObject()) ||
      !SumAndNormalize(temp, inputs.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !inputs.SetGradient(temp, false))
      ReturnFalse;
//--- Embedding
   for(int i = cEmbedding.Total() - 2; i >= 0; i--)
     {
      inputs = cEmbedding[i];
      if(!inputs ||
         !inputs.CalcHiddenGradients(cEmbedding[i + 1]))
         ReturnFalse;
     }
//---
   if(!cTranspose.CalcHiddenGradients(inputs))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Updates the CNeuronSAGDFN trainable weights from the accumulated  gradients.  |
//+-------------------------------------------------------------------------------+
bool CNeuronSAGDFN::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *inputs = cTranspose.AsObject();
   CNeuronBaseOCL *current = NULL;
//--- Embedding
   for(int i = 0; i < cEmbedding.Total(); i++)
     {
      current = cEmbedding[i];
      if(!current ||
         !current.UpdateInputWeights(inputs))
         ReturnFalse;
      inputs = current;
     }
//--- GCRUs
   if(!cAttention.UpdateInputWeights(inputs))
      ReturnFalse;
   for(int i = 0; i < cGCRU.Total(); i++)
     {
      current = cGCRU[i];
      if(!current ||
         !current.UpdateInputWeights(inputs))
         ReturnFalse;
      inputs = current;
     }
//--- Forecast
   for(int i = 0; i < cProjection.Total(); i++)
     {
      current = cProjection[i];
      if(!current ||
         !current.UpdateInputWeights(inputs))
         ReturnFalse;
      inputs = current;
     }
//--- result
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Averages the CNeuronSAGDFN weights toward the source model with  factor tau.  |
//+-------------------------------------------------------------------------------+
bool CNeuronSAGDFN::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronTransposeOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSAGDFN *Source = source;
//--- Embedding
   if(!cEmbedding.WeightsUpdate(Source.cEmbedding.AsObject(), tau))
      ReturnFalse;
//--- GCRUs
   if(!cAttention.WeightsUpdate(Source.cAttention.AsObject(), tau))
      ReturnFalse;
   if(!cGCRU.WeightsUpdate(Source.cGCRU.AsObject(), tau))
      ReturnFalse;
//--- Forecast
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSAGDFN state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronSAGDFN::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
//--- Embedding
   if(!cEmbedding.Save(file_handle))
      ReturnFalse;
//--- GCRUs
   if(!cGCRU.Save(file_handle) ||
      !cAttention.Save(file_handle))
      ReturnFalse;
//--- Forecast
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSAGDFN state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronSAGDFN::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
//--- Embedding
   if(!cEmbedding.Load(file_handle))
      ReturnFalse;
//--- GCRUs
   if(!cGCRU.Load(file_handle) ||
      !LoadInsideLayer(file_handle, cAttention.AsObject()))
      ReturnFalse;
//--- Forecast
   if(!cProjection.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------+
//| Rebinds the CNeuronSAGDFN GPU buffers to the given OpenCL  context.  |
//+----------------------------------------------------------------------+
void CNeuronSAGDFN::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
//---
   cTranspose.SetOpenCL(OpenCL);
   cEmbedding.SetOpenCL(OpenCL);
   cGCRU.SetOpenCL(OpenCL);
   cAttention.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
  }
//+--------------------------------------------------------------------------------+
//| Initializes the CNeuronSpatialEmbedding layer and allocates its  GPU buffers.  |
//+--------------------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint units, uint window, uint embed_dim,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (window + embed_dim)*units, optimization_type, batch))
      ReturnFalse;
   activation = None;
   if(!cEmbedding.Init(0, 0, OpenCL, embed_dim * units, optimization, iBatch))
      ReturnFalse;
   cEmbedding.SetActivationFunction(TANH);
//---
   iUnits = units;
   iWindow = window;
   iEmbeddingDim = embed_dim;
//---
   return true;
  }
//+------------------------------------------------------------------------------------+
//| Runs the CNeuronSpatialEmbedding feed-forward pass and writes  the  layer output.  |
//+------------------------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(bTrain)
      if(!cEmbedding.FeedForward())
         ReturnFalse;
//---
   if(!Concat(NeuronOCL.getOutput(), cEmbedding.getOutput(), Output, iWindow, iEmbeddingDim, iUnits))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronSpatialEmbedding layer.  |
//+----------------------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), cEmbedding.getGradient(), Gradient, iWindow, iEmbeddingDim, iUnits))
      ReturnFalse;
   Deactivation(NeuronOCL)
   Deactivation(cEmbedding)
//---
   return true;
  }
//+-----------------------------------------------------------------------------------------+
//| Updates the CNeuronSpatialEmbedding trainable weights from the  accumulated gradients.  |
//+-----------------------------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEmbedding.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------------------+
//| Averages the CNeuronSpatialEmbedding weights toward the source  model with factor tau.  |
//+-----------------------------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpatialEmbedding *Source = source;
   if(!cEmbedding.WeightsUpdate(Source.cEmbedding.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSpatialEmbedding state to the file handle.      |
//+------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cEmbedding.Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iUnits) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iWindow) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iEmbeddingDim) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Restores the CNeuronSpatialEmbedding state from the file handle.  |
//+-------------------------------------------------------------------+
bool CNeuronSpatialEmbedding::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cEmbedding.AsObject()))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iEmbeddingDim = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Rebinds the CNeuronSpatialEmbedding GPU buffers to the given  OpenCL context.  |
//+--------------------------------------------------------------------------------+
void CNeuronSpatialEmbedding::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cEmbedding.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------------+
//| Initializes the CNeuronGraphons layer and allocates its GPU  buffers.  |
//+------------------------------------------------------------------------+
bool CNeuronGraphons::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint units, uint window, uint emb_dimension,
                           uint experts, float dropout,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * units, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cGraphs.Init(0, index, OpenCL, Neurons()*experts, optimization, iBatch))
      ReturnFalse;
   cGraphs.SetActivationFunction(SIGMOID);
//---
   CNeuronBaseOCL       *neuron  = NULL;
   CNeuronConvOCL       *conv    = NULL;
   CNeuronTransposeOCL  *transp  = NULL;
   CNeuronDropoutOCL    *dout    = NULL;
   CNeuronSoftMaxOCL    *softmax = NULL;
//--- Probability
   acProbability.Clear();
   acProbability.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(experts, index, OpenCL, window, window, experts, units, 1, optimization, iBatch) ||
      !acProbability.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, experts, optimization, iBatch) ||
      !acProbability.Add(neuron))
     {
      DeleteObj(neuron);
      ReturnFalse;
     }
   index++;
   dout = new CNeuronDropoutOCL();
   if(!dout ||
      !dout.Init(0, index, OpenCL, experts, dropout, optimization, iBatch) ||
      !acProbability.Add(dout))
     {
      DeleteObj(dout);
      ReturnFalse;
     }
   index++;
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, experts, optimization, iBatch) ||
      !acProbability.Add(softmax))
     {
      DeleteObj(softmax);
      ReturnFalse;
     }
   softmax.SetHeads(1);
//--- Data Embedding
   acDataEmb.Clear();
   acDataEmb.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, 2 * emb_dimension, units, 1, optimization, iBatch) ||
      !acDataEmb.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * emb_dimension, 2 * emb_dimension, emb_dimension, units, 1, optimization, iBatch) ||
      !acDataEmb.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units, emb_dimension, optimization, iBatch) ||
      !acDataEmb.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
//---
   index++;
   if(!cExpertsEmb.Init(0, index, OpenCL, units * emb_dimension * experts, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Runs the CNeuronGraphons feed-forward pass and writes the layer  output.  |
//+---------------------------------------------------------------------------+
bool CNeuronGraphons::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
//--- Probability
   for(int i = 0; i < acProbability.Total(); i++)
     {
      current = acProbability[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//--- Data Embedding
   prev = NeuronOCL;
   for(int i = 0; i < acDataEmb.Total(); i++)
     {
      current = acDataEmb[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//--- Experts
   if(bTrain)
      if(!cExpertsEmb.FeedForward())
         ReturnFalse;
//--- Graphs
   uint units = (uint)MathSqrt((double)Neurons());
   uint emb_dim = acDataEmb[-1].Neurons() / units;
   uint experts = cExpertsEmb.Neurons() / acDataEmb[-1].Neurons();
   if(!MatMul(cExpertsEmb.getOutput(), current.getOutput(), cGraphs.getOutput(),
              units, emb_dim, units, experts, false))
      ReturnFalse;
   if(cGraphs.Activation() != None)
      if(!Activation(cGraphs.getOutput(), cGraphs.getOutput(), cGraphs.Activation()))
         ReturnFalse;
//--- Result
   if(!MatMul(acProbability[-1].getOutput(), cGraphs.getOutput(), Output, 1, experts, units * units, 1, false))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronGraphons  layer.  |
//+--------------------------------------------------------------------------+
bool CNeuronGraphons::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!acDataEmb[-1] || !acProbability[-1])
      ReturnFalse;
//---
   uint units = (uint)MathSqrt((double)Neurons());
   uint emb_dim = acDataEmb[-1].Neurons() / units;
   uint experts = cExpertsEmb.Neurons() / acDataEmb[-1].Neurons();
//---
   if(!MatMulGrad(acProbability[-1].getOutput(), acProbability[-1].getGradient(),
                  cGraphs.getOutput(), cGraphs.getGradient(), Gradient,
                  1, experts, units * units, 1, false))
      ReturnFalse;
//--- Graphs
   if(cGraphs.Activation() != None)
      if(!DeActivation(cGraphs.getOutput(), cGraphs.getGradient(),
                       cGraphs.getGradient(), cGraphs.Activation()))
         ReturnFalse;
   if(!MatMulGrad(cExpertsEmb.getOutput(), cExpertsEmb.getGradient(),
                  acDataEmb[-1].getOutput(), acDataEmb[-1].getGradient(),
                  cGraphs.getGradient(), units, emb_dim, units, experts, false))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = NULL;
   CNeuronBaseOCL* current = NULL;
//--- Probability
   for(int i = acProbability.Total() - 1; i >= 0; i--)
     {
      next = acProbability[i];
      current = (i == 0 ? NeuronOCL : acProbability[i - 1]);
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
     }
//--- Data Embedding
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false))
      ReturnFalse;
   for(int i = acDataEmb.Total() - 1; i >= 0; i--)
     {
      next = acDataEmb[i];
      current = (i == 0 ? NeuronOCL : acDataEmb[i - 1]);
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
     }
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Updates the CNeuronGraphons trainable weights from the  accumulated gradients.  |
//+---------------------------------------------------------------------------------+
bool CNeuronGraphons::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
//--- Probability
   for(int i = 0; i < acProbability.Total(); i++)
     {
      current = acProbability[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//--- Data Embedding
   prev = NeuronOCL;
   for(int i = 0; i < acDataEmb.Total(); i++)
     {
      current = acDataEmb[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//--- Experts
   if(!cExpertsEmb.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Averages the CNeuronGraphons weights toward the source model  with  factor tau.  |
//+----------------------------------------------------------------------------------+
bool CNeuronGraphons::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronGraphons *Source = source;
   if(!acProbability.WeightsUpdate(Source.acProbability.AsObject(), tau))
      ReturnFalse;
   if(!acDataEmb.WeightsUpdate(Source.acDataEmb.AsObject(), tau))
      ReturnFalse;
   if(!cExpertsEmb.WeightsUpdate(Source.cExpertsEmb.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronGraphons state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronGraphons::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!acProbability.Save(file_handle))
      ReturnFalse;
   if(!acDataEmb.Save(file_handle))
      ReturnFalse;
   if(!cExpertsEmb.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronGraphons state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronGraphons::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!acProbability.Load(file_handle))
      ReturnFalse;
   if(!acDataEmb.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cExpertsEmb.AsObject()))
      ReturnFalse;
//---
   uint experts = cExpertsEmb.Neurons() / acDataEmb[-1].Neurons();
   if(!cGraphs.Init(0, 0, OpenCL, Neurons()*experts, optimization, iBatch))
      ReturnFalse;
   cGraphs.SetActivationFunction(SIGMOID);
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Rebinds the CNeuronGraphons GPU buffers to the given OpenCL  context.  |
//+------------------------------------------------------------------------+
void CNeuronGraphons::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   acProbability.SetOpenCL(OpenCL);
   acDataEmb.SetOpenCL(OpenCL);
   cExpertsEmb.SetOpenCL(OpenCL);
   cGraphs.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronGraphons layer.             |
//+------------------------------------------------------------------+
void CNeuronGraphons::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   acProbability.TrainMode(bTrain);
   acDataEmb.TrainMode(bTrain);
   cExpertsEmb.TrainMode(bTrain);
  }
//+-----------------------------------------------------------------------------+
//| Initializes the CNeuronSparseSoftMax layer and allocates its GPU  buffers.  |
//+-----------------------------------------------------------------------------+
bool CNeuronSparseSoftMax::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint units, uint dimension_in, uint dimension_out,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimension_in < dimension_out)
      ReturnFalse;
   if(!CNeuronSoftMaxOCL::Init(numOutputs, myIndex, open_cl, units * dimension_out, optimization_type, batch))
      ReturnFalse;
//---
   SetHeads(units);
   iDimensionIn = dimension_in;
   if(!cIndexes.BufferInit(Neurons(), -1) ||
      !cIndexes.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Runs the CNeuronSparseSoftMax feed-forward pass and writes the  layer output.  |
//+--------------------------------------------------------------------------------+
bool CNeuronSparseSoftMax::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL ||
      NeuronOCL.Neurons() < int(iHeads * iDimensionIn))
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = { iHeads, iDimensionIn };
   uint local_work_size[2] = { 1, global_work_size[1] };
   uint kernel = def_k_SparseSoftMax;
   setBuffer(kernel, def_k_ssoftmax_data, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ssoftmax_outputs, getOutputIndex())
   setBuffer(kernel, def_k_ssoftmax_indexes, cIndexes.GetIndex())
   setArgument(kernel, def_k_ssoftmax_out_dimension, int(DimensionOut()))
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronSparseSoftMax layer.  |
//+-------------------------------------------------------------------------------+
bool CNeuronSparseSoftMax::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL ||
      NeuronOCL.Neurons() < int(iHeads * iDimensionIn))
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = { iHeads, iDimensionIn };
   uint local_work_size[2] = { 1, global_work_size[1] };
   uint kernel = def_k_SparseSoftMaxGrad;
   setBuffer(kernel, def_k_ssoftmax_gr_data_gr, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_ssoftmax_gr_outputs, getOutputIndex())
   setBuffer(kernel, def_k_ssoftmax_gr_outputs_gr, getGradientIndex())
   setBuffer(kernel, def_k_ssoftmax_gr_indexes, cIndexes.GetIndex())
   setArgument(kernel, def_k_ssoftmax_gr_out_dimension, int(DimensionOut()))
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSparseSoftMax state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronSparseSoftMax::Save(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iDimensionIn) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSparseSoftMax state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronSparseSoftMax::Load(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimensionIn = (uint)FileReadInteger(file_handle);
//---
   cIndexes.BufferFree();
   if(!cIndexes.BufferInit(Neurons(), -1) ||
      !cIndexes.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Rebinds the CNeuronSparseSoftMax GPU buffers to the given OpenCL  context.  |
//+-----------------------------------------------------------------------------+
void CNeuronSparseSoftMax::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSoftMaxOCL::SetOpenCL(obj);
   cIndexes.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//| Reports checked-transfer support for SparseSoftMax.              |
//+------------------------------------------------------------------+
bool CNeuronSparseSoftMax::SupportsOpenCLChecked(void) const
  {
//--- Reject subtypes with an unverified ownership tree.
   return(Type() == defNeuronSparseSoftMax);
  }
//+------------------------------------------------------------------+
//| Transfers sparse indexes before publishing the base context.     |
//+------------------------------------------------------------------+
bool CNeuronSparseSoftMax::SetOpenCLChecked(COpenCLMy *obj)
  {
//--- Validate the target and exact ownership contract before mutation.
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      ReturnFalseEx("checked context target or sparse ownership");
//--- Preserve allocated index data from its current device owner.
   if(cIndexes.Total() > 0 && cIndexes.GetIndex() >= 0 &&
      cIndexes.GetOpenCL() != obj && !cIndexes.BufferRead())
      ReturnFalseEx("sparse indexes read");
//--- Do not overwrite already transferred indexes from a stale host cache.
   if(cIndexes.Total() > 0 &&
      (cIndexes.GetOpenCL() != obj || cIndexes.GetIndex() < 0) &&
      !cIndexes.BufferCreate(obj))
      ReturnFalseEx("sparse indexes create");
//--- Publish the inherited context only after index transfer succeeds.
   if(!CNeuronBaseOCL::SetOpenCLChecked(obj))
      ReturnFalseEx("base context transfer");
//--- Finalize without resetting the sparse configuration.
   return(true);
  }
//+-------------------------------------------------------------------------------+
//| Initializes the CNeuronGlobLocGraphAtt layer and allocates its  GPU buffers.  |
//+-------------------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                  uint units, uint window, uint experts, float dropout,
                                  uint emb_dimension, uint sparse_dimension,
                                  ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMHAttentionPooling::Init(numOutputs, myIndex, open_cl, window, units,
                                       2, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cGlobal.Init(0, index, OpenCL, iUnits, iWindow, emb_dimension, experts, dropout,
                    optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLocal.Init(0, index, OpenCL, iUnits, iWindow, experts, dropout, emb_dimension,
                   sparse_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcat.Init(0, index, OpenCL, 2 * iWindow * iUnits, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Runs the CNeuronGlobLocGraphAtt feed-forward pass and writes the  layer output.  |
//+----------------------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cGlobal.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cLocal.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!Concat(cGlobal.getOutput(), cLocal.getOutput(), cConcat.getOutput(),
              iWindow, iWindow, iUnits))
      ReturnFalse;
//---
   return CNeuronMHAttentionPooling::feedForward(cConcat.AsObject());
  }
//+---------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronGlobLocGraphAtt layer.  |
//+---------------------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronMHAttentionPooling::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
   if(!DeConcat(cGlobal.getGradient(), cLocal.getGradient(), cConcat.getGradient(),
                iWindow, iWindow, iUnits))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cGlobal.AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(PrevOutput, false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cLocal.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, PrevOutput, temp, iWindow, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------------+
//| Updates the CNeuronGlobLocGraphAtt trainable weights from the  accumulated gradients.  |
//+----------------------------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cGlobal.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cLocal.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronMHAttentionPooling::updateInputWeights(cConcat.AsObject());
  }
//+----------------------------------------------------------------------------------------+
//| Averages the CNeuronGlobLocGraphAtt weights toward the source  model with factor tau.  |
//+----------------------------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMHAttentionPooling::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronGlobLocGraphAtt* Source = source;
   if(!cGlobal.WeightsUpdate(Source.cGlobal.AsObject(), tau))
      ReturnFalse;
   if(!cLocal.WeightsUpdate(Source.cLocal.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronGlobLocGraphAtt state to the file handle.       |
//+------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::Save(const int file_handle)
  {
   if(!CNeuronMHAttentionPooling::Save(file_handle))
      ReturnFalse;
//---
   if(!cGlobal.Save(file_handle))
      ReturnFalse;
   if(!cLocal.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronGlobLocGraphAtt state from the file handle.  |
//+------------------------------------------------------------------+
bool CNeuronGlobLocGraphAtt::Load(const int file_handle)
  {
   if(!CNeuronMHAttentionPooling::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cGlobal.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLocal.AsObject()))
      ReturnFalse;
//---
   if(!cConcat.Init(0, 2, OpenCL, 2 * iWindow * iUnits, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Rebinds the CNeuronGlobLocGraphAtt GPU buffers to the given  OpenCL context.  |
//+-------------------------------------------------------------------------------+
void CNeuronGlobLocGraphAtt::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMHAttentionPooling::SetOpenCL(obj);
//---
   cGlobal.SetOpenCL(OpenCL);
   cLocal.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Sets the training flag of the CNeuronGlobLocGraphAtt layer.      |
//+------------------------------------------------------------------+
void CNeuronGlobLocGraphAtt::TrainMode(bool flag)
  {
   CNeuronMHAttentionPooling::TrainMode(flag);
//---
   cGlobal.TrainMode(flag);
   cLocal.TrainMode(flag);
   cConcat.TrainMode(flag);
  }
//+---+