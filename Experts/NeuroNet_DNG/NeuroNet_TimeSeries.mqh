//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh: TimeSeries
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint history, uint forecast, uint variables, uint features, uint & encoder_decoder[], ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, forecast * variables, optimization_type, batch))
      ReturnFalse;
//---
   iHistory = MathMax(history, 1);
   iForecast = forecast;
   iVariables = variables;
   iFeatures = MathMax(features, 1);
//---
   if(!acFeatureProjection[0].Init(0, 0, OpenCL, iFeatures, iHistory * iVariables, 1, optimization, iBatch))
      ReturnFalse;
   if(!acFeatureProjection[1].Init(0, 1, OpenCL, iFeatures, iForecast * iVariables, 1, optimization, iBatch))
      ReturnFalse;
//---
   int total = ArraySize(encoder_decoder);
   if(ArrayResize(acEncoderDecoder, total + 1) < total + 1)
      ReturnFalse;
   if(total == 0)
     {
      if(!acEncoderDecoder[0].Init(0, 2, OpenCL, 2 * iHistory, iForecast, iVariables, optimization, iBatch))
         ReturnFalse;
     }
   else
     {
      if(!acEncoderDecoder[0].Init(0, 2, OpenCL, 2 * iHistory, encoder_decoder[0], iVariables, optimization, iBatch))
         ReturnFalse;
      for(int i = 1; i < total; i++)
         if(!acEncoderDecoder[i].Init(0, i + 2, OpenCL, encoder_decoder[i - 1], encoder_decoder[i], iVariables, optimization, iBatch))
            ReturnFalse;
      if(!acEncoderDecoder[total].Init(0, total + 2, OpenCL, encoder_decoder[total - 1], iForecast, iVariables, optimization, iBatch))
         ReturnFalse;
     }
//---
   if(!cGlobalResidual.Init(0, total + 3, OpenCL, iHistory, iHistory, iForecast, iVariables, optimization, iBatch))
      ReturnFalse;
   cGlobalResidual.SetActivationFunction(TANH);
   if(!cTemporalDecoder.Init(0, total + 4, OpenCL, 2 * iForecast, iForecast, iVariables, optimization, iBatch))
      ReturnFalse;
//---
   if(!cHistoryInput.Init(0, total + 5, OpenCL, iHistory * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cFeatureInput.Init(0, total + 6, OpenCL, iFeatures, optimization, iBatch))
      ReturnFalse;
   if(!cEncoderInput.Init(0, total + 7, OpenCL, 2 * iHistory * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTemporalDecoderInput.Init(0, total + 8, OpenCL, 2 * iForecast * iVariables, optimization, iBatch))
      ReturnFalse;
//---
   if(cGlobalResidual.getGradient() != Gradient)
      if(!cGlobalResidual.SetGradient(Gradient))
         ReturnFalse;
   if(cTemporalDecoder.getGradient() != getGradient())
      if(!cTemporalDecoder.SetGradient(Gradient))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(cHistoryInput.getOutputIndex() != NeuronOCL.getOutputIndex())
     {
      CBufferFloat *temp = cHistoryInput.getOutput();
      if(!temp.BufferSet(NeuronOCL.getOutputIndex()))
         ReturnFalse;
     }
   if(cFeatureInput.getOutputIndex() != SecondInput.GetIndex())
     {
      CBufferFloat *temp = cFeatureInput.getOutput();
      if(!temp.BufferSet(SecondInput.GetIndex()))
         ReturnFalse;
     }
//---
   if(!cGlobalResidual.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!acFeatureProjection[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!acFeatureProjection[1].FeedForward(cFeatureInput.AsObject()))
      ReturnFalse;
//---
   if(!Concat(NeuronOCL.getOutput(), acFeatureProjection[0].getOutput(), cEncoderInput.getOutput(), iHistory, iHistory, iVariables))
      ReturnFalse;
   uint total = acEncoderDecoder.Size();
   CNeuronBaseOCL *prev = cEncoderInput.AsObject();
   for(uint i = 0; i < total; i++)
     {
      if(!acEncoderDecoder[i].FeedForward(prev))
         ReturnFalse;
      prev = acEncoderDecoder[i].AsObject();
     }
//---
   if(!Concat(prev.getOutput(), acFeatureProjection[1].getOutput(), cTemporalDecoderInput.getOutput(), iForecast, iForecast, iVariables))
      ReturnFalse;
   if(!cTemporalDecoder.FeedForward(cTemporalDecoderInput.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cGlobalResidual.getOutput(), cTemporalDecoder.getOutput(), Output, iForecast, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!cTemporalDecoderInput.CalcHiddenGradients(cTemporalDecoder.AsObject()))
      ReturnFalse;
//---
   int total = (int)acEncoderDecoder.Size();
   if(!DeConcat(acEncoderDecoder[total - 1].getGradient(), acFeatureProjection[1].getGradient(), cTemporalDecoderInput.getGradient(), iForecast, iForecast, iVariables))
      ReturnFalse;
   for(int i = total - 2; i >= 0; i--)
      if(!acEncoderDecoder[i].CalcHiddenGradients(acEncoderDecoder[i + 1].AsObject()))
         ReturnFalse;
   if(!cEncoderInput.CalcHiddenGradients(acEncoderDecoder[0].AsObject()))
      ReturnFalse;
//---
   if(!DeConcat(cHistoryInput.getGradient(), acFeatureProjection[0].getGradient(), cEncoderInput.getGradient(), iHistory, iHistory, iVariables))
      ReturnFalse;
//---
   if(cGlobalResidual.Activation() != None)
     {
      if(!DeActivation(cGlobalResidual.getOutput(), cGlobalResidual.getGradient(), cGlobalResidual.getGradient(), cGlobalResidual.Activation()))
         ReturnFalse;
     }
   if(!NeuronOCL.CalcHiddenGradients(cGlobalResidual.AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(cHistoryInput.getOutput(), cHistoryInput.getGradient(), cHistoryInput.getGradient(), SecondActivation))
         ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cHistoryInput.getGradient(), NeuronOCL.getGradient(), iHistory, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cFeatureInput.CalcHiddenGradients(acFeatureProjection[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cFeatureInput.getGradient(), cFeatureInput.getGradient(), SecondGradient, iFeatures, false, 0, 0, 0, 0.5f))
      ReturnFalse;
   if(!cFeatureInput.CalcHiddenGradients(acFeatureProjection[1].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(SecondGradient, cFeatureInput.getGradient(), SecondGradient, iFeatures, false, 0, 0, 0, 1.0f))
      ReturnFalse;
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
//---
   if(!cGlobalResidual.UpdateInputWeights(cHistoryInput.AsObject()))
      ReturnFalse;
   if(!acFeatureProjection[0].UpdateInputWeights(cHistoryInput.AsObject()))
      ReturnFalse;
   if(!acFeatureProjection[1].UpdateInputWeights(cFeatureInput.AsObject()))
      ReturnFalse;
//---
   uint total = acEncoderDecoder.Size();
   CNeuronBaseOCL *prev = cEncoderInput.AsObject();
   for(uint i = 0; i < total; i++)
     {
      if(!acEncoderDecoder[i].UpdateInputWeights(prev))
         ReturnFalse;
      prev = acEncoderDecoder[i].AsObject();
     }
//---
   if(!cTemporalDecoder.UpdateInputWeights(cTemporalDecoderInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iHistory, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iForecast, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iVariables, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iFeatures, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   uint total = acEncoderDecoder.Size();
   if(FileWriteInteger(file_handle, (int)total, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   for(uint i = 0; i < total; i++)
      if(!acEncoderDecoder[i].Save(file_handle))
         ReturnFalse;
   if(!cGlobalResidual.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!acFeatureProjection[i].Save(file_handle))
         ReturnFalse;
   if(!cTemporalDecoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHistory = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iForecast = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFeatures = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
//---
   int total = FileReadInteger(file_handle);
   int prev_size = (int)acEncoderDecoder.Size();
   if(prev_size != total)
      if(ArrayResize(acEncoderDecoder, total) < total)
         ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      if(i >= prev_size)
         if(!acEncoderDecoder[i].Init(0, i + 2, OpenCL, 1, 1, 1, ADAM, 1))
            ReturnFalse;
      if(!LoadInsideLayer(file_handle, acEncoderDecoder[i].AsObject()))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, cGlobalResidual.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, acFeatureProjection[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTemporalDecoder.AsObject()))
      ReturnFalse;
//---
   if(!cHistoryInput.Init(0, total + 5, OpenCL, iHistory * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cFeatureInput.Init(0, total + 6, OpenCL, iFeatures, optimization, iBatch))
      ReturnFalse;
   if(!cEncoderInput.Init(0, total + 7, OpenCL, 2 * iHistory * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTemporalDecoderInput.Init(0, total + 8, OpenCL, 2 * iForecast * iVariables, optimization, iBatch))
      ReturnFalse;
//---
   if(cGlobalResidual.getGradient() != Gradient)
      if(!cGlobalResidual.SetGradient(Gradient))
         ReturnFalse;
   if(cTemporalDecoder.getGradient() != getGradient())
      if(!cTemporalDecoder.SetGradient(Gradient))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTiDEOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   uint total = acEncoderDecoder.Size();
   for(uint i = 0; i < total; i++)
      acEncoderDecoder[i].SetOpenCL(OpenCL);
   cGlobalResidual.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      acFeatureProjection[i].SetOpenCL(OpenCL);
   cTemporalDecoder.SetOpenCL(OpenCL);
//---
   cHistoryInput.SetOpenCL(OpenCL);
   cFeatureInput.SetOpenCL(OpenCL);
   cEncoderInput.SetOpenCL(OpenCL);
   cTemporalDecoderInput.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTiDEOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
//---
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronTiDEOCL *Source = source;
   if(!cGlobalResidual.WeightsUpdate(Source.cGlobalResidual.AsObject(), tau))
      ReturnFalse;
   if(!acFeatureProjection[0].WeightsUpdate(Source.acFeatureProjection[0].AsObject(), tau))
      ReturnFalse;
   if(!acFeatureProjection[1].WeightsUpdate(Source.acFeatureProjection[1].AsObject(), tau))
      ReturnFalse;
//---
   uint total = acEncoderDecoder.Size();
   for(uint i = 0; i < total; i++)
      if(!acEncoderDecoder[i].WeightsUpdate(Source.acEncoderDecoder[i].AsObject(), tau))
         ReturnFalse;
//---
   if(!cTemporalDecoder.WeightsUpdate(Source.cTemporalDecoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint count, uint filters, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * count, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = window;
   iCount = count;
   iFilters = MathMax(filters, 1);
//---
   if(!cWavlets.Init(0, 0, OpenCL, iWindow, iWindow, iCount, iFilters, optimization, iBatch))
      ReturnFalse;
   cWavlets.SetActivationFunction(None);
   if(!cNorm.Init(0, 1, OpenCL, iFilters * iCount, 1000, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   if(!cSoftMax.Init(0, 1, OpenCL, iFilters * iCount, optimization, iBatch))
      ReturnFalse;
   cSoftMax.SetHeads(iCount);
   cSoftMax.SetActivationFunction(None);
   if(!cReconstruct.Init(0, 2, OpenCL, iWindow, optimization, iBatch))
      ReturnFalse;
   cReconstruct.SetActivationFunction(None);
   if(!cFF[0].Init(0, 3, OpenCL, iWindow, iWindow, 4 * iWindow, iCount, optimization, iBatch))
      ReturnFalse;
   cFF[0].SetActivationFunction(LReLU);
   if(!cFF[1].Init(0, 4, OpenCL, 4 * iWindow, 4 * iWindow, iWindow, iCount, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   if(Gradient != cFF[1].getGradient())
      SetGradient(cFF[1].getGradient());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cWavlets.FeedForward(NeuronOCL.AsObject()))
      ReturnFalse;
   if(!cNorm.FeedForward(cWavlets.AsObject()))
      ReturnFalse;
   if(!cSoftMax.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!Reconsruct(cReconstruct.getOutput(), cSoftMax.getOutput()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), cReconstruct.getOutput(), cReconstruct.getOutput(), iWindow, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!cFF[0].FeedForward(cReconstruct.AsObject()))
      ReturnFalse;
   if(!cFF[1].FeedForward(cFF[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cFF[1].getOutput(), cReconstruct.getOutput(), getOutput(), iWindow, true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::Reconsruct(CBufferFloat* sequence, CBufferFloat* probability)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = sequence.Total();
   global_work_size[1] = 1;
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_w, cWavlets.GetWeightsConv().GetIndex())
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_g, probability.GetIndex())
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_o, probability.GetIndex())
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_ig, sequence.GetIndex())
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_outputs, probability.Total())
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_step, (int)iWindow)
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_window_in, (int)iWindow)
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_window_out, (int)iFilters)
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_activation, (int)None)
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_shift_out, (int)0)
   kernelExecute(def_k_CalcHiddenGradientConv, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cWavlets.GetWeightsConv().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!cFF[0].CalcHiddenGradients(cFF[1].AsObject()))
      ReturnFalse;
   if(!cReconstruct.CalcHiddenGradients(cFF[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(Gradient, cReconstruct.getGradient(), cReconstruct.getGradient(), iWindow, false))
      ReturnFalse;
//---
   CBufferFloat *temp_r = cReconstruct.getOutput();
   if(!cReconstruct.SetOutput(cReconstruct.getGradient(), false))
      ReturnFalse;
   CBufferFloat *temp_w = cWavlets.getOutput();
   if(!cWavlets.SetOutput(cSoftMax.getGradient(), false))
      ReturnFalse;
//---
   if(!cWavlets.FeedForward(cReconstruct.AsObject()))
      ReturnFalse;
//---
   if(!cWavlets.SetOutput(temp_w, false))
      ReturnFalse;
   if(!cReconstruct.SetOutput(temp_r, false))
      ReturnFalse;
//---
   if(!cNorm.CalcHiddenGradients(cSoftMax.AsObject()))
      ReturnFalse;
   if(!cWavlets.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cWavlets.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cReconstruct.getGradient(), NeuronOCL.getGradient(), iWindow, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cFF[0].UpdateInputWeights(cReconstruct.AsObject()))
      ReturnFalse;
   if(!cFF[1].UpdateInputWeights(cFF[0].AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cWavlets.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFEDW::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cWavlets.SetOpenCL(OpenCL);
   cSoftMax.SetOpenCL(OpenCL);
   cFF[0].SetOpenCL(OpenCL);
   cFF[1].SetOpenCL(OpenCL);
   cReconstruct.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cWavlets.Save(file_handle))
      ReturnFalse;
   if(!cSoftMax.Save(file_handle))
      ReturnFalse;
   if(!cFF[0].Save(file_handle))
      ReturnFalse;
   if(!cFF[1].Save(file_handle))
      ReturnFalse;
   if(!cReconstruct.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iWindow) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iCount) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iFilters) < INT_VALUE)
      ReturnFalse;//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWavlets.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSoftMax.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFF[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFF[1].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cReconstruct.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   iWindow = (uint)FileReadInteger(file_handle);
   iCount = (uint)FileReadInteger(file_handle);
   iFilters = (uint)FileReadInteger(file_handle);
//---
   if(Gradient != cFF[1].getGradient())
      SetGradient(cFF[1].getGradient());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFEDW::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronFEDW *Source = source;
   for(int i = 0; i < 2; i++)
     {
      if(!cFF[i].WeightsUpdate(Source.cFF[i].AsObject(), tau))
         ReturnFalse;
     }
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_out, uint count, float dropout, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(window <= 0)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * count, optimization_type, batch))
      ReturnFalse;
//--- Save constants
   iWindow = window;
   iWindowOut = window_out;
   iCount = count;
   activation = None;
//--- Calculate FFT and iFFT size
   int power = int(MathLog(iWindow) / M_LN2);
   if(MathPow(2, power) != iWindow)
      power++;
   iFFTin = uint(MathPow(2, power));
   power = int(MathLog(iWindowOut) / M_LN2);
   if(MathPow(2, power) != iWindowOut)
      power++;
   iIFFTin = uint(MathPow(2, power));
//--- Init objects
   if(!cInputsRe.Init(0, 0, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cInputsIm.Init(0, 1, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cFFTRe.Init(0, 2, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cFFTIm.Init(0, 3, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cDropRe.Init(0, 4, OpenCL, iFFTin * iCount, dropout, optimization, iBatch))
      ReturnFalse;
   if(!cDropIm.Init(0, 5, OpenCL, iFFTin * iCount, dropout, optimization, iBatch))
      ReturnFalse;
   if(!cInsideRe1.Init(0, 6, OpenCL, iFFTin, iFFTin, 4 * iIFFTin, iCount, optimization, iBatch))
      ReturnFalse;
   cInsideRe1.SetActivationFunction(TANH);
   if(!cInsideIm1.Init(0, 7, OpenCL, iFFTin, iFFTin, 4 * iIFFTin, iCount, optimization, iBatch))
      ReturnFalse;
   cInsideIm1.SetActivationFunction(TANH);
   if(!cInsideRe2.Init(0, 8, OpenCL, 4 * iIFFTin, 4 * iIFFTin, iIFFTin, iCount, optimization, iBatch))
      ReturnFalse;
   cInsideRe2.SetActivationFunction(None);
   if(!cInsideIm2.Init(0, 9, OpenCL, 4 * iIFFTin, 4 * iIFFTin, iIFFTin, iCount, optimization, iBatch))
      ReturnFalse;
   cInsideIm2.SetActivationFunction(None);
   if(!cComplexRe.Init(0, 10, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cComplexIm.Init(0, 11, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cIFFTRe.Init(0, 12, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cIFFTIm.Init(0, 13, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cClear.BufferInit(MathMax(iFFTin, iIFFTin)*iCount, 0))
      ReturnFalse;
   cClear.BufferCreate(OpenCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iCount};
   setBuffer(def_k_FFT, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(def_k_FFT, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(def_k_FFT, def_k_fft_input_window, (int)(inp_re.Total() / iCount))
   setArgument(def_k_FFT, def_k_fft_input_complex, int(!!inp_im))
   setArgument(def_k_FFT, def_k_fft_output_window, (int)(out_re.Total() / iCount))
   setArgument(def_k_FFT, def_k_fft_reverse, int(reverse))
   kernelExecute(def_k_FFT, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::ComplexLayerOut(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im)
  {
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { iCount, inp_re.Total() / iCount };
   setBuffer(def_k_ComplexLayer, def_k_cl_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_ComplexLayer, def_k_cl_inputs_im, inp_im.GetIndex())
   setBuffer(def_k_ComplexLayer, def_k_cl_outputs_re, out_re.GetIndex())
   setBuffer(def_k_ComplexLayer, def_k_cl_outputs_im, out_im.GetIndex())
   kernelExecute(def_k_ComplexLayer, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::ComplexLayerGradient(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im)
  {
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { iCount, inp_re.Total() / iCount };
   setBuffer(def_k_ComplexLayerGradient, def_k_cl_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_ComplexLayerGradient, def_k_cl_inputs_im, inp_im.GetIndex())
   setBuffer(def_k_ComplexLayerGradient, def_k_cl_outputs_re, out_re.GetIndex())
   setBuffer(def_k_ComplexLayerGradient, def_k_cl_outputs_im, out_im.GetIndex())
   kernelExecute(def_k_ComplexLayerGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- FFT
   if(!FFT(NeuronOCL.getOutput(), NULL, cFFTRe.getOutput(), cFFTIm.getOutput(), false))
      ReturnFalse;
//--- DropOut
   if(!cDropRe.FeedForward(cFFTRe.AsObject()))
      ReturnFalse;
   if(!cDropIm.FeedForward(cFFTIm.AsObject()))
      ReturnFalse;
//--- Complex Layer
   if(!cInsideRe1.FeedForward(cDropRe.AsObject()))
      ReturnFalse;
   if(!cInsideRe2.FeedForward(cInsideRe1.AsObject()))
      ReturnFalse;
   if(!cInsideIm1.FeedForward(cDropIm.AsObject()))
      ReturnFalse;
   if(!cInsideIm2.FeedForward(cInsideIm1.AsObject()))
      ReturnFalse;
   if(!ComplexLayerOut(cInsideRe2.getOutput(), cInsideIm2.getOutput(), cComplexRe.getOutput(), cComplexIm.getOutput()))
      ReturnFalse;
//--- iFFT
   if(!FFT(cComplexRe.getOutput(), cComplexIm.getOutput(), cIFFTRe.getOutput(), cIFFTIm.getOutput(), true))
      ReturnFalse;
//--- To Output
   if(!DeConcat(Output, cIFFTRe.getGradient(), cIFFTRe.getOutput(), iWindowOut, iIFFTin - iWindowOut, iCount))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Copy Gradients
   if(!SumAndNormalize(cIFFTIm.getOutput(), GetPointer(cClear), cIFFTIm.getGradient(), 1, false, 0, 0, 0, -1))
      ReturnFalse;
   if(!Concat(Gradient, GetPointer(cClear), cIFFTRe.getGradient(), iWindowOut, iIFFTin - iWindowOut, iCount))
      ReturnFalse;
//--- FFT
   if(!FFT(cIFFTRe.getGradient(), cIFFTIm.getGradient(), cComplexRe.getGradient(), cComplexIm.getGradient(), false))
      ReturnFalse;
//--- Complex Layer
   if(!ComplexLayerGradient(cInsideRe2.getGradient(), cInsideIm2.getGradient(), cComplexRe.getGradient(), cComplexIm.getGradient()))
      ReturnFalse;
   if(!cInsideRe1.CalcHiddenGradients(cInsideRe2.AsObject()))
      ReturnFalse;
   if(!cInsideIm1.CalcHiddenGradients(cInsideIm2.AsObject()))
      ReturnFalse;
   if(!cDropRe.CalcHiddenGradients(cInsideRe1.AsObject()))
      ReturnFalse;
   if(!cDropIm.CalcHiddenGradients(cInsideIm1.AsObject()))
      ReturnFalse;
//--- Dropout
   if(!cFFTRe.CalcHiddenGradients(cDropRe.AsObject()))
      ReturnFalse;
   if(!cFFTIm.CalcHiddenGradients(cDropIm.AsObject()))
      ReturnFalse;
//--- IFFT
   if(!FFT(cFFTRe.getGradient(), cFFTIm.getGradient(), cInputsRe.getGradient(), cInputsIm.getGradient(), true))
      ReturnFalse;
//--- To Input Layer
   if(!DeConcat(NeuronOCL.getGradient(), cFFTIm.getGradient(), cFFTRe.getGradient(), iWindow, iFFTin - iWindow, iCount))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cInsideRe1.UpdateInputWeights(cDropRe.AsObject()))
      ReturnFalse;
   if(!cInsideIm1.UpdateInputWeights(cDropIm.AsObject()))
      ReturnFalse;
   if(!cInsideRe2.UpdateInputWeights(cInsideRe1.AsObject()))
      ReturnFalse;
   if(!cInsideIm2.UpdateInputWeights(cInsideIm1.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//--- Save constants
   if(FileWriteInteger(file_handle, int(iWindow)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iWindowOut)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iCount)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iFFTin)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iIFFTin)) < INT_VALUE)
      ReturnFalse;
//--- Save objects
   if(!cInsideRe1.Save(file_handle))
      ReturnFalse;
   if(!cInsideIm1.Save(file_handle))
      ReturnFalse;
   if(!cInsideRe2.Save(file_handle))
      ReturnFalse;
   if(!cInsideIm2.Save(file_handle))
      ReturnFalse;
   if(!cDropRe.Save(file_handle))
      ReturnFalse;
   if(!cDropIm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//--- Load constants
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindowOut = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCount = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFFTin = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iIFFTin = uint(FileReadInteger(file_handle));
   activation = None;
//--- Load objects
   if(!LoadInsideLayer(file_handle, cInsideRe1.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInsideIm1.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInsideRe2.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInsideIm2.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDropRe.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDropIm.AsObject()))
      ReturnFalse;
//--- Init objects
   if(!cInputsRe.Init(0, 0, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cInputsIm.Init(0, 1, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cFFTRe.Init(0, 2, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cFFTIm.Init(0, 3, OpenCL, iFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cComplexRe.Init(0, 8, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cComplexIm.Init(0, 9, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cIFFTRe.Init(0, 10, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cIFFTIm.Init(0, 11, OpenCL, iIFFTin * iCount, optimization, iBatch))
      ReturnFalse;
   if(!cClear.BufferInit(MathMax(iFFTin, iIFFTin)*iCount, 0))
      ReturnFalse;
   cClear.BufferCreate(OpenCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFITSOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cInputsRe.SetOpenCL(OpenCL);
   cInputsIm.SetOpenCL(OpenCL);
   cFFTRe.SetOpenCL(OpenCL);
   cFFTIm.SetOpenCL(OpenCL);
   cDropRe.SetOpenCL(OpenCL);
   cDropIm.SetOpenCL(OpenCL);
   cInsideRe1.SetOpenCL(OpenCL);
   cInsideIm1.SetOpenCL(OpenCL);
   cInsideRe2.SetOpenCL(OpenCL);
   cInsideIm2.SetOpenCL(OpenCL);
   cComplexRe.SetOpenCL(OpenCL);
   cComplexIm.SetOpenCL(OpenCL);
   cIFFTRe.SetOpenCL(OpenCL);
   cIFFTIm.SetOpenCL(OpenCL);
   cClear.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFITSOCL::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cInputsRe.TrainMode(bTrain);
   cInputsIm.TrainMode(bTrain);
   cFFTRe.TrainMode(bTrain);
   cFFTIm.TrainMode(bTrain);
   cDropRe.TrainMode(bTrain);
   cDropIm.TrainMode(bTrain);
   cInsideRe1.TrainMode(bTrain);
   cInsideIm1.TrainMode(bTrain);
   cInsideRe2.TrainMode(bTrain);
   cInsideIm2.TrainMode(bTrain);
   cComplexRe.TrainMode(bTrain);
   cComplexIm.TrainMode(bTrain);
   cIFFTRe.TrainMode(bTrain);
   cIFFTIm.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFITSOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronFITSOCL *Source = source;
   if(cInsideRe1.WeightsUpdate(Source.cInsideRe1.AsObject(), tau))
      ReturnFalse;
   if(cInsideIm1.WeightsUpdate(Source.cInsideIm1.AsObject(), tau))
      ReturnFalse;
   if(cInsideRe2.WeightsUpdate(Source.cInsideRe2.AsObject(), tau))
      ReturnFalse;
   if(cInsideIm2.WeightsUpdate(Source.cInsideIm2.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint count, float Alpha, bool need_transpose = true, ENUM_OPTIMIZATION optimization_type = ADAM, uint batch = 1)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * count, optimization_type, batch))
      ReturnFalse;
//---
   bTranspose = need_transpose;
   iWindow = window;
   iCount = count;
   fAlpha = MathMax(0, MathMin(Alpha, 1));
   activation = None;
//--- Calculate FFTsize
   uint size = (bTranspose ? count : window);
   int power = int(MathLog(size) / M_LN2);
   if(MathPow(2, power) != size)
      power++;
   iFFTin = uint(MathPow(2, power));
//---
   uint n = (bTranspose ? iWindow : iCount);
   if(!cForecastFreRe.BufferInit(iFFTin * n, 0) || !cForecastFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cForecastFreIm.BufferInit(iFFTin * n, 0) || !cForecastFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cTargetFreRe.BufferInit(iFFTin * n, 0) || !cTargetFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cTargetFreIm.BufferInit(iFFTin * n, 0) || !cTargetFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLossFreRe.BufferInit(iFFTin * n, 0) || !cLossFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLossFreIm.BufferInit(iFFTin * n, 0) || !cLossFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGradientFreRe.BufferInit(iFFTin * n, 0) || !cGradientFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGradientFreIm.BufferInit(iFFTin * n, 0) || !cGradientFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(bTranspose)
      if(!cTranspose.BufferInit(iWindow * iCount, 0) || !cTranspose.BufferCreate(OpenCL))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {(bTranspose ? iWindow : iCount)};
   setBuffer(def_k_FFT, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(def_k_FFT, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(def_k_FFT, def_k_fft_input_window, (int)(inp_re.Total() / (!bTranspose ? iWindow : iCount)))
   setArgument(def_k_FFT, def_k_fft_input_complex, int(!!inp_im))
   setArgument(def_k_FFT, def_k_fft_output_window, (int)(out_re.Total() / (!bTranspose ? iWindow : iCount)))
   setArgument(def_k_FFT, def_k_fft_reverse, int(reverse))
   kernelExecute(def_k_FFT, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
   if(NeuronOCL.getOutput() != Output)
     {
      Output.BufferFree();
      DeleteObj(Output);
      Output = NeuronOCL.getOutput();
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::Transpose(CBufferFloat* inputs, CBufferFloat* outputs, uint rows, uint cols)
  {
   if(!inputs || !outputs)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {rows, cols};
   setBuffer(def_k_Transpose, def_k_tr_matrix_in, inputs.GetIndex())
   setBuffer(def_k_Transpose, def_k_tr_matrix_out, outputs.GetIndex())
   kernelExecute(def_k_Transpose, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!outputs.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::FreqMSA(CBufferFloat* target, CBufferFloat* forecast, CBufferFloat* gradient)
  {
   if(!target || !forecast || !gradient)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {gradient.Total()};
   setBuffer(def_k_GradientMSA, def_k_gmsa_target, target.GetIndex())
   setBuffer(def_k_GradientMSA, def_k_gmsa_forecast, forecast.GetIndex())
   setBuffer(def_k_GradientMSA, def_k_gmsa_gradient, gradient.GetIndex())
   kernelExecute(def_k_GradientMSA, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::CumulativeGradient(CBufferFloat* gradient1, CBufferFloat* gradient2, CBufferFloat* cummulative, float Alpha)
  {
   if(!gradient1 || !gradient2 || !cummulative)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {cummulative.Total()};
   setBuffer(def_k_CumulativeGradient, def_k_cg_gradient1, gradient1.GetIndex())
   setBuffer(def_k_CumulativeGradient, def_k_cg_gradient2, gradient2.GetIndex())
   setBuffer(def_k_CumulativeGradient, def_k_cg_gradient_out, cummulative.GetIndex())
   setArgument(def_k_CumulativeGradient, def_k_cg_alpha, float(Alpha))
   kernelExecute(def_k_CumulativeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cummulative.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::calcOutputGradients(CArrayFloat* Target, float & error)
  {
   if(!Target)
      ReturnFalse;
   if(Target.Total() < Output.Total())
      ReturnFalse;
//---
   if(Target.Total() == Output.Total())
     {
      if(!Gradient.AssignArray(Target))
         ReturnFalse;
     }
   else
     {
      for(int i = 0; i < Output.Total(); i++)
        {
         if(!Gradient.Update(i, Target.At(i)))
            ReturnFalse;
        }
     }
   if(!Gradient.BufferWrite())
      ReturnFalse;
//---
   if(bTranspose)
     {
      if(!Transpose(Output, GetPointer(cTranspose), iWindow, iCount))
         ReturnFalse;
      if(!FFT(GetPointer(cTranspose), NULL, GetPointer(cForecastFreRe), GetPointer(cForecastFreIm), false))
         ReturnFalse;
      if(!Transpose(Gradient, GetPointer(cTranspose), iWindow, iCount))
         ReturnFalse;
      if(!FFT(GetPointer(cTranspose), NULL, GetPointer(cTargetFreRe), GetPointer(cTargetFreIm), false))
         ReturnFalse;
     }
   else
     {
      if(!FFT(Output, NULL, GetPointer(cForecastFreRe), GetPointer(cForecastFreIm), false))
         ReturnFalse;
      if(!FFT(Gradient, NULL, GetPointer(cTargetFreRe), GetPointer(cTargetFreIm), false))
         ReturnFalse;
     }
//---
   if(!FreqMSA(GetPointer(cTargetFreRe), GetPointer(cForecastFreRe), GetPointer(cLossFreRe)))
      ReturnFalse;
   if(!FreqMSA(GetPointer(cTargetFreIm), GetPointer(cForecastFreIm), GetPointer(cLossFreIm)))
      ReturnFalse;
   if(!FreqMSA(Gradient, Output, Gradient))
      ReturnFalse;
//---
   if(!FFT(GetPointer(cLossFreRe), GetPointer(cLossFreIm), GetPointer(cGradientFreRe), GetPointer(cGradientFreIm), true))
      ReturnFalse;
   if(bTranspose)
     {
      if(!Transpose(GetPointer(cGradientFreRe), GetPointer(cTranspose), iCount, iWindow))
         ReturnFalse;
      if(!CumulativeGradient(GetPointer(cTranspose), Gradient, Gradient, fAlpha))
         ReturnFalse;
     }
   else
      if(!CumulativeGradient(GetPointer(cGradientFreRe), Gradient, Gradient, fAlpha))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   return DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), Gradient, NeuronOCL.Activation());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iWindow)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iCount)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iFFTin)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(bTranspose)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteFloat(file_handle, fAlpha) < sizeof(float))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFreDFOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCount = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFFTin = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   bTranspose = bool(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   fAlpha = FileReadFloat(file_handle);
//---
   uint n = (bTranspose ? iWindow : iCount);
   if(!cForecastFreRe.BufferInit(iFFTin * n, 0) || !cForecastFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cForecastFreIm.BufferInit(iFFTin * n, 0) || !cForecastFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cTargetFreRe.BufferInit(iFFTin * n, 0) || !cTargetFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cTargetFreIm.BufferInit(iFFTin * n, 0) || !cTargetFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLossFreRe.BufferInit(iFFTin * n, 0) || !cLossFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLossFreIm.BufferInit(iFFTin * n, 0) || !cLossFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGradientFreRe.BufferInit(iFFTin * n, 0) || !cGradientFreRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGradientFreIm.BufferInit(iFFTin * n, 0) || !cGradientFreIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(bTranspose)
     {
      if(!cTranspose.BufferInit(iWindow * iCount, 0) || !cTranspose.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      cTranspose.BufferFree();
      cTranspose.Clear();
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFreDFOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cForecastFreRe.BufferCreate(OpenCL);
   cForecastFreIm.BufferCreate(OpenCL);
   cTargetFreRe.BufferCreate(OpenCL);
   cTargetFreIm.BufferCreate(OpenCL);
   cLossFreRe.BufferCreate(OpenCL);
   cLossFreIm.BufferCreate(OpenCL);
   cGradientFreRe.BufferCreate(OpenCL);
   cGradientFreIm.BufferCreate(OpenCL);
   if(bTranspose)
      cTranspose.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint history, uint forecast, uint variables, uint heads, uint layers, uint & patch[], ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, forecast * variables, optimization_type, batch))
      ReturnFalse;
//---
   iHistory = MathMax(history, 1);
   iForecast = forecast;
   iVariables = variables;
//--- Calculate FFTsize
   uint size = iHistory + iForecast;
   int power = int(MathLog(size) / M_LN2);
   if(MathPow(2, power) < size)
      power++;
   iFFT = uint(MathPow(2, power));
//--- T-Block
   if(!cNorm.Init(0, 0, OpenCL, iHistory * iVariables, batch, optimization))
      ReturnFalse;
   if(!cPositionEncoder.Init(0, 1, OpenCL, iVariables, iHistory, optimization, batch))
      ReturnFalse;
   if(!cTranspose.Init(0, 2, OpenCL, iHistory, iVariables, optimization, batch))
      ReturnFalse;
   cTranspose.SetActivationFunction(None);
   uint count = (iHistory - patch[0] + 2 * patch[1] - 1) / patch[1];
   if(!cPatching.Init(0, 3, OpenCL, patch[0], patch[1], patch[2], count, iVariables, optimization, batch))
      ReturnFalse;
   caAttention.SetOpenCL(OpenCL);
   for(uint l = 0; l < layers; l++)
     {
      CNeuronConformer *temp = new CNeuronConformer();
      if(!temp)
         ReturnFalse;
      if(!temp.Init(0, 4 + l, OpenCL, patch[2], 32, heads, iVariables, count, optimization, batch))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
      if(!caAttention.Add(temp))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
     }
   int total = 3;
   caProjection.SetOpenCL(OpenCL);
   uint window = patch[2] * count;
   for(int l = 0; l < total; l++)
     {
      CNeuronConvOCL *temp = new CNeuronConvOCL();
      if(!temp)
         ReturnFalse;
      if(!temp.Init(0, 4 + layers + l, OpenCL, window, window, (total - l)*iForecast, iVariables, optimization, batch))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
      temp.SetActivationFunction(TANH);
      if(!caProjection.Add(temp))
        {
         DeleteObj(temp);
         ReturnFalse;
        }
      window = (total - l) * iForecast;
     }
   if(!cRevIN.Init(0, 4 + layers + total, OpenCL, iForecast * iVariables, 1, cNorm.AsObject()))
      ReturnFalse;
//--- F-Block
   if(!cInputFreqRe.BufferInit(iFFT * iVariables, 0) || !cInputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqIm.BufferInit(iFFT * iVariables, 0) || !cInputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqComplex.Init(0, 0, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cMainFreqWeights.BufferInit(iVariables, 0) || !cMainFreqWeights.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cNormFreqComplex.Init(0, 1, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cMeans.BufferInit(iVariables, 0) || !cMeans.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cVariances.BufferInit(iVariables, 0) || !cVariances.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cFreqAtteention.Init(0, 2, OpenCL, iFFT, 32, heads, iVariables, layers, optimization, batch))
      ReturnFalse;
   if(!cUnNormFreqComplex.Init(0, 1, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cOutputFreqRe.BufferInit(iFFT * iVariables, 0) || !cOutputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputFreqIm.BufferInit(iFFT * iVariables, 0) || !cOutputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasRe.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasIm.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasReGrad.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasReGrad.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cReconstructInput.BufferInit(iHistory * iVariables, 0) || !cReconstructInput.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cForecast.BufferInit(iForecast * iVariables, 0) || !cForecast.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cReconstructInputGrad.BufferInit(iHistory * iVariables, 0) || !cReconstructInputGrad.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cForecastGrad.BufferInit(iForecast * iVariables, 0) || !cForecastGrad.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cZero.BufferInit(iFFT * iVariables, 0) || !cZero.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_FFT, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(def_k_FFT, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(def_k_FFT, def_k_fft_input_window, (int)(inp_re.Total() / iVariables))
   setArgument(def_k_FFT, def_k_fft_input_complex, int(!!inp_im))
   setArgument(def_k_FFT, def_k_fft_output_window, (int)(out_re.Total() / iVariables))
   setArgument(def_k_FFT, def_k_fft_reverse, int(reverse))
   kernelExecute(def_k_FFT, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::ComplexNormalize(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexNormalize, def_k_cn_inputs, cInputFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_outputs, cNormFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_means, cMeans.GetIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalize, def_k_cn_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexNormalize, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::ComplexUnNormalize(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_inputs, cFreqAtteention.getOutputIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_outputs, cUnNormFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_means, cMeans.GetIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexUnNormalize, def_k_cn_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexUnNormalize, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFreqAtteention.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::ComplexNormalizeGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_inputs_gr, cInputFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_outputs_gr, cNormFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalizeGradient, def_k_cng_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexNormalizeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::ComplexUnNormalizeGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_inputs_gr, cFreqAtteention.getGradientIndex())
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_outputs_gr, cUnNormFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexUnNormalizeGradient, def_k_cng_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexUnNormalizeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFreqAtteention.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::MainFreqWeights(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_MainFreqWeight, def_k_mfw_freq, cInputFreqComplex.getOutputIndex())
   setBuffer(def_k_MainFreqWeight, def_k_mfw_weight, cMainFreqWeights.GetIndex())
   setArgument(def_k_MainFreqWeight, def_k_mfw_dimension, (int)(iFFT))
   kernelExecute(def_k_MainFreqWeight, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::WeightedSum(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_WeightedSum, def_k_ws_inputs1, cForecast.GetIndex())
   setBuffer(def_k_WeightedSum, def_k_ws_inputs2, cRevIN.getOutputIndex())
   setBuffer(def_k_WeightedSum, def_k_ws_outputs, getOutputIndex())
   setBuffer(def_k_WeightedSum, def_k_ws_weight, cMainFreqWeights.GetIndex())
   setArgument(def_k_WeightedSum, def_k_ws_dimension, (int)(iForecast))
   kernelExecute(def_k_WeightedSum, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cMainFreqWeights.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::WeightedSumGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_WeightedSumGradient, def_k_ws_inputs1, cForecastGrad.GetIndex())
   setBuffer(def_k_WeightedSumGradient, def_k_ws_inputs2, cRevIN.getGradientIndex())
   setBuffer(def_k_WeightedSumGradient, def_k_ws_outputs, getGradientIndex())
   setBuffer(def_k_WeightedSumGradient, def_k_ws_weight, cMainFreqWeights.GetIndex())
   setArgument(def_k_WeightedSumGradient, def_k_ws_dimension, (int)(iForecast))
   kernelExecute(def_k_WeightedSumGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cMainFreqWeights.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::calcReconstructGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = iHistory * iVariables;
   setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_t, cInputs.GetIndex())
   setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_o, cReconstructInput.GetIndex())
   setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_ig, cReconstructInputGrad.GetIndex())
   setArgument(def_k_CalcOutputGradient, def_k_cog_activation, (int)None)
   setArgument(def_k_CalcOutputGradient, def_k_cog_error, 1)
   ResetLastError();
   kernelExecute(def_k_CalcOutputGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cReconstructInputGrad.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
   if(cInputs != NeuronOCL.getOutput())
      cInputs = NeuronOCL.getOutput();
//--- T-Block
   if(!cNorm.FeedForward(NeuronOCL))
      ReturnFalse;;
   if(!cPositionEncoder.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!cTranspose.FeedForward(cPositionEncoder.AsObject()))
      ReturnFalse;
   if(!cPatching.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   int total = caAttention.Total();
   CNeuronBaseOCL* prev = cPatching.AsObject();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL* att = caAttention.At(i);
      if(!att.FeedForward(prev))
         ReturnFalse;
      prev = att;
     }
   total = caProjection.Total();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL* proj = caProjection.At(i);
      if(!proj.FeedForward(prev))
         ReturnFalse;
      prev = proj;
     }
   if(!cRevIN.FeedForward(prev))
      ReturnFalse;
//--- F-Block
   if(!FFT(cInputs, cInputs, GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), false))
      ReturnFalse;
   if(!Concat(GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), cInputFreqComplex.getOutput(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!ComplexNormalize())
      ReturnFalse;
   if(!MainFreqWeights())
      ReturnFalse;
   if(!cFreqAtteention.FeedForward(cNormFreqComplex.AsObject()))
      ReturnFalse;
   if(!ComplexUnNormalize())
      ReturnFalse;
   if(!DeConcat(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), cUnNormFreqComplex.getOutput(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!FFT(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), GetPointer(cOutputTimeSeriasRe), GetPointer(cOutputTimeSeriasIm), true))
      ReturnFalse;
   if(!DeConcat(GetPointer(cReconstructInput), GetPointer(cForecast), GetPointer(cOutputTimeSeriasReGrad), GetPointer(cOutputTimeSeriasRe), iHistory, iForecast, iFFT - iHistory - iForecast, iVariables))
      ReturnFalse;
//--- Output
   if(!WeightedSum())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getGradient() || !cInputs)
      ReturnFalse;
//--- Output
   if(!WeightedSumGradient())
      ReturnFalse;
//--- T-Block
   if(cRevIN.Activation() != None && !DeActivation(cRevIN.getOutput(), cRevIN.getGradient(), cRevIN.getGradient(), cRevIN.Activation()))
      ReturnFalse;
   CNeuronBaseOCL *next = cRevIN.AsObject();
   for(int i = caProjection.Total() - 1; i >= 0; i--)
     {
      CNeuronBaseOCL *proj = caProjection.At(i);
      if(!proj || !proj.CalcHiddenGradients((CObject *)next))
         ReturnFalse;
      next = proj;
     }
   for(int i = caAttention.Total() - 1; i >= 0; i--)
     {
      CNeuronBaseOCL *att = caAttention.At(i);
      if(!att || !att.CalcHiddenGradients((CObject *)next))
         ReturnFalse;
      next = att;
     }
   if(!cPatching.CalcHiddenGradients((CObject*)next))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cPatching.AsObject()))
      ReturnFalse;
   if(!cPositionEncoder.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!cNorm.CalcHiddenGradients(cPositionEncoder.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
//--- F-Block
   if(!CNeuronBaseOCL::SumAndNormalize(GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputTimeSeriasIm), iFFT * iVariables, false, 0, 0, 0, -0.5))
      ReturnFalse;
   if(!calcReconstructGradient())
      ReturnFalse;
   if(!Concat(GetPointer(cReconstructInputGrad), GetPointer(cForecastGrad), GetPointer(cZero), GetPointer(cOutputTimeSeriasReGrad), iHistory, iForecast, iFFT - iHistory - iForecast, iVariables))
      ReturnFalse;
   if(!FFT(GetPointer(cOutputTimeSeriasReGrad), GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), false))
      ReturnFalse;
   if(!Concat(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), cUnNormFreqComplex.getGradient(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!ComplexUnNormalizeGradient())
      ReturnFalse;
   if(!cNormFreqComplex.CalcHiddenGradients(cFreqAtteention.AsObject()))
      ReturnFalse;
   if(!ComplexNormalizeGradient())
      ReturnFalse;
   if(!DeConcat(GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), cInputFreqComplex.getGradient(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!FFT(GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), GetPointer(cOutputTimeSeriasRe), GetPointer(cOutputTimeSeriasIm), false))
      ReturnFalse;
   if(!DeConcat(GetPointer(cInputFreqRe), GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputTimeSeriasRe), iHistory, iFFT - iHistory, iVariables))
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::SumAndNormalize(NeuronOCL.getGradient(), GetPointer(cInputFreqRe), NeuronOCL.getGradient(), iHistory * iVariables, false, 0, 0, 0, 0.5))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- F-Block
   if(!cFreqAtteention.UpdateInputWeights(cNormFreqComplex.AsObject()))
      ReturnFalse;
//--- T-Block
   if(!cPatching.UpdateInputWeights(cPositionEncoder.AsObject()))
      ReturnFalse;
   int total = caAttention.Total();
   CNeuronBaseOCL *prev = cPatching.AsObject();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL *att = caAttention.At(i);
      if(!att.UpdateInputWeights(prev))
         ReturnFalse;
      prev = att;
     }
   total = caProjection.Total();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL *proj = caProjection.At(i);
      if(!proj.UpdateInputWeights(prev))
         ReturnFalse;
      prev = proj;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iHistory) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iForecast) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iVariables) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iFFT) < INT_VALUE)
      ReturnFalse;
//--- T-Block
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(!cPositionEncoder.Save(file_handle))
      ReturnFalse;
   if(!cPatching.Save(file_handle))
      ReturnFalse;
   if(!caAttention.Save(file_handle))
      ReturnFalse;
   if(!caProjection.Save(file_handle))
      ReturnFalse;
//--- F-Block
   if(!cFreqAtteention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronATFNetOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHistory = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iForecast = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFFT = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
//--- T-Block
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPositionEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatching.AsObject()))
      ReturnFalse;
   if(!caAttention.Load(file_handle))
      ReturnFalse;
   if(!caProjection.Load(file_handle))
      ReturnFalse;
   if(!cTranspose.Init(0, 2, OpenCL, iHistory, iVariables, optimization, iBatch))
      ReturnFalse;
   cTranspose.SetActivationFunction(None);
   if(!cRevIN.Init(0, 4 + caAttention.Total() + caProjection.Total(), OpenCL, iForecast * iVariables, 1, cNorm.AsObject()))
      ReturnFalse;
//--- F-Block
   if(!LoadInsideLayer(file_handle, cFreqAtteention.AsObject()))
      ReturnFalse;
   cInputFreqRe.BufferFree();
   if(!cInputFreqRe.BufferInit(iFFT * iVariables, 0) || !cInputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   cInputFreqIm.BufferFree();
   if(!cInputFreqIm.BufferInit(iFFT * iVariables, 0) || !cInputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqComplex.Init(0, 0, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   cMainFreqWeights.BufferFree();
   if(!cMainFreqWeights.BufferInit(iVariables, 0) || !cMainFreqWeights.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cNormFreqComplex.Init(0, 1, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   cMeans.BufferFree();
   if(!cMeans.BufferInit(iVariables, 0) || !cMeans.BufferCreate(OpenCL))
      ReturnFalse;
   cVariances.BufferFree();
   if(!cVariances.BufferInit(iVariables, 0) || !cVariances.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cUnNormFreqComplex.Init(0, 1, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   cOutputFreqRe.BufferFree();
   if(!cOutputFreqRe.BufferInit(iFFT * iVariables, 0) || !cOutputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   cOutputFreqIm.BufferFree();
   if(!cOutputFreqIm.BufferInit(iFFT * iVariables, 0) || !cOutputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasRe.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasRe.BufferCreate(OpenCL))
      ReturnFalse;
   cOutputTimeSeriasIm.BufferFree();
   if(!cOutputTimeSeriasIm.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasIm.BufferCreate(OpenCL))
      ReturnFalse;
   cOutputTimeSeriasReGrad.BufferFree();
   if(!cOutputTimeSeriasReGrad.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasReGrad.BufferCreate(OpenCL))
      ReturnFalse;
   cReconstructInput.BufferFree();
   if(!cReconstructInput.BufferInit(iHistory * iVariables, 0) || !cReconstructInput.BufferCreate(OpenCL))
      ReturnFalse;
   cForecast.BufferFree();
   if(!cForecast.BufferInit(iForecast * iVariables, 0) || !cForecast.BufferCreate(OpenCL))
      ReturnFalse;
   cReconstructInputGrad.BufferFree();
   if(!cReconstructInputGrad.BufferInit(iHistory * iVariables, 0) || !cReconstructInputGrad.BufferCreate(OpenCL))
      ReturnFalse;
   cForecastGrad.BufferFree();
   if(!cForecastGrad.BufferInit(iForecast * iVariables, 0) || !cForecastGrad.BufferCreate(OpenCL))
      ReturnFalse;
   cZero.BufferFree();
   if(!cZero.BufferInit(iFFT * iVariables, 0) || !cZero.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronATFNetOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//--- T-Block
   cNorm.SetOpenCL(OpenCL);
   cPositionEncoder.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
   cPatching.SetOpenCL(OpenCL);
   for(int l = 0; l < caAttention.Total(); l++)
     {
      CNeuronBaseOCL *temp = caAttention.At(l);
      if(!temp)
         continue;
      temp.SetOpenCL(OpenCL);
     }
   for(int l = 0; l < caProjection.Total(); l++)
     {
      CNeuronBaseOCL *temp = caProjection.At(l);
      if(!temp)
         continue;
      temp.SetOpenCL(OpenCL);
     }
   cRevIN.SetOpenCL(OpenCL);
//--- F-Block
   cInputFreqRe.BufferCreate(OpenCL);
   cInputFreqIm.BufferCreate(OpenCL);
   cInputFreqComplex.SetOpenCL(OpenCL);
   cMainFreqWeights.BufferCreate(OpenCL);
   cNormFreqComplex.SetOpenCL(OpenCL);
   cMeans.BufferCreate(OpenCL);
   cVariances.BufferCreate(OpenCL);
   cFreqAtteention.SetOpenCL(OpenCL);
   cUnNormFreqComplex.SetOpenCL(OpenCL);
   cOutputFreqRe.BufferCreate(OpenCL);
   cOutputFreqIm.BufferCreate(OpenCL);
   cOutputTimeSeriasRe.BufferCreate(OpenCL);
   cOutputTimeSeriasIm.BufferCreate(OpenCL);
   cOutputTimeSeriasReGrad.BufferCreate(OpenCL);
   cReconstructInput.BufferCreate(OpenCL);
   cForecast.BufferCreate(OpenCL);
   cReconstructInputGrad.BufferCreate(OpenCL);
   cForecastGrad.BufferCreate(OpenCL);
   cZero.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronATFNetOCL::getWeights(void)
  {
   CNeuronConvOCL *conv = caProjection.At(caProjection.Total() - 1);
   return conv.GetWeightsConv();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = MathMax(window, 1);
   iSegments = (numNeurons + window - 1) / window;
//---
   if(!cOne.Init(0, 0, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   CBufferFloat *buffer = cOne.getOutput();
   if(!buffer || !buffer.BufferInit(buffer.Total(), 1))
      ReturnFalse;
   if(!buffer.BufferWrite())
      ReturnFalse;
//---
   if(!cShufle.Init(0, 1, OpenCL, 1, 1, iSegments, 1, optimization, iBatch))
      ReturnFalse;
   cShufle.SetActivationFunction(None);
   if(!cProbability.Init(0, 2, OpenCL, iSegments, optimization, iBatch))
      ReturnFalse;
   cProbability.SetActivationFunction(None);
   cProbability.SetHeads(1);
   if(!cWeights.Init(0, 3, OpenCL, 1, 1, 2, 1, optimization, iBatch))
      ReturnFalse;
   cWeights.SetActivationFunction(SIGMOID);
   if(!cPositions.BufferInit(iSegments, 0) || !cPositions.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::feedForwardS3(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iSegments};
   setBuffer(def_k_FeedForwardS3, def_k_s3_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_FeedForwardS3, def_k_s3_outputs, getOutputIndex())
   setBuffer(def_k_FeedForwardS3, def_k_s3_positions, cPositions.GetIndex())
   setBuffer(def_k_FeedForwardS3, def_k_s3_probability, cProbability.getOutputIndex())
   setBuffer(def_k_FeedForwardS3, def_k_s3_weights, cWeights.getOutputIndex())
   setArgument(def_k_FeedForwardS3, def_k_s3_window, (int)(iWindow))
   setArgument(def_k_FeedForwardS3, def_k_s3_total, (int)(Neurons()))
   kernelExecute(def_k_FeedForwardS3, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cWeights.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::calcInputGradientsS3(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput())
      ReturnFalse;
//---
     {
      uint global_work_offset[1] = {0};
      uint global_work_size[1] = {iSegments};
      setBuffer(def_k_InsideGradientS3, def_k_s3g_inputs, NeuronOCL.getOutputIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_inputs_gr, NeuronOCL.getGradientIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_outputs_gr, getGradientIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_positions, cPositions.GetIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_probability, cProbability.getOutputIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_probability_gr, cProbability.getGradientIndex())
      setBuffer(def_k_InsideGradientS3, def_k_s3g_weights, cWeights.getOutputIndex())
      setArgument(def_k_InsideGradientS3, def_k_s3g_window, (int)(iWindow))
      setArgument(def_k_InsideGradientS3, def_k_s3g_total, (int)(Neurons()))
      kernelExecute(def_k_InsideGradientS3, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!cWeights.getOutput().BufferRead())
         ReturnFalse;
#endif
     }
//---
     {
      uint global_work_offset[2] = {0, 0};
      uint dimension = MathMin(Neurons(), 64);
      uint global_work_size[2] = {dimension, 2};
      uint local_work_size[2] = {dimension, 1};
      setBuffer(def_k_WeightGradientS3, def_k_s3w_inputs, NeuronOCL.getOutputIndex())
      setBuffer(def_k_WeightGradientS3, def_k_s3w_outputs_gr, getGradientIndex())
      setBuffer(def_k_WeightGradientS3, def_k_s3w_positions, cPositions.GetIndex())
      setBuffer(def_k_WeightGradientS3, def_k_s3w_weights_gr, cWeights.getGradientIndex())
      setArgument(def_k_WeightGradientS3, def_k_s3w_window, (int)(iWindow))
      setArgument(def_k_WeightGradientS3, def_k_s3w_total, (int)(Neurons()))
      kernelExecuteLoc(def_k_WeightGradientS3, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
      if(!cWeights.getGradient().BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(bTrain)
     {
      if(!cWeights.FeedForward(cOne.AsObject()))
         ReturnFalse;
      if(!cShufle.FeedForward(cOne.AsObject()))
         ReturnFalse;
      if(!cProbability.FeedForward(cShufle.AsObject()))
         ReturnFalse;
     }
   if(!feedForwardS3(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!calcInputGradientsS3(NeuronOCL))
      ReturnFalse;
   if(!cShufle.CalcHiddenGradients(cProbability.AsObject()))
      ReturnFalse;
   if(cWeights.Activation() != None)
      if(!DeActivation(cWeights.getOutput(), cWeights.getGradient(), cWeights.getGradient(), cWeights.Activation()))
         ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cWeights.UpdateInputWeights(cOne.AsObject()))
      ReturnFalse;
   if(!cShufle.UpdateInputWeights(cOne.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cShufle.Save(file_handle))
      ReturnFalse;
   if(!cWeights.Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iWindow), INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iSegments), INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cShufle.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWeights.AsObject()))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iSegments = uint(FileReadInteger(file_handle));
//---
   if(cProbability.Neurons() != iSegments)
     {
      if(!cProbability.Init(0, 2, OpenCL, iSegments, optimization, iBatch))
         ReturnFalse;
      cProbability.SetActivationFunction(None);
     }
   if(!cProbability.FeedForward(cShufle.AsObject()))
      ReturnFalse;
   if(cPositions.Total() != iSegments)
     {
      cPositions.BufferFree();
      if(!cPositions.BufferInit(iSegments, 0) || !cPositions.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronS3::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cOne.SetOpenCL(OpenCL);
   cShufle.SetOpenCL(OpenCL);
   cProbability.SetOpenCL(OpenCL);
   cWeights.SetOpenCL(OpenCL);
   cPositions.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronS3::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronS3 *Source = source;
   if(!cWeights.WeightsUpdate(Source.cWeights.AsObject(), tau))
      ReturnFalse;
   if(!cShufle.WeightsUpdate(Source.cShufle.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint variables, uint lenth, uint hidden_size, bool ts_in_row, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronLSTMOCL::Init(numOutputs, myIndex, open_cl, hidden_size, optimization_type, batch))
      ReturnFalse;
   if(!SetInputs(variables * lenth))
      ReturnFalse;
//---
   uint windows[] = {variables, 6, 5, 4};
   if(!cFeatureExtraction.Init(0, 0, OpenCL, windows, lenth, variables, ts_in_row, optimization, batch))
      ReturnFalse;
//---
   if(!cMemAndHidden.Init(0, 1, OpenCL, hidden_size * 2, optimization, batch))
      ReturnFalse;
   if(!cConcatenated.Init(0, 2, OpenCL, variables * lenth, variables * lenth, hidden_size * 2, optimization, batch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(TANH);
   if(!cSoftMax.Init(0, 3, OpenCL, variables * lenth, optimization, batch))
      ReturnFalse;
   cSoftMax.SetHeads(variables);
   if(!cAttentionOut.Init(0, 4, OpenCL, variables * lenth, optimization, batch))
      ReturnFalse;
//---
   bTSinRow = ts_in_row;
   if(!bTSinRow)
     {
      if(!cTranspose.Init(0, 5, OpenCL, variables, lenth, optimization, iBatch))
         ReturnFalse;
     }
//---
   if(!cTemp.BufferInit(variables * lenth, 0) || !cTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- FEATURE EXTRACTION
   if(!cFeatureExtraction.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Memory and Hidden
   if(!Concat(m_iHiddenState, m_iMemory, m_iHiddenState, m_iMemory, cMemAndHidden.getOutputIndex(), 1, 1, 0, 0, Neurons()))
      ReturnFalse;
//--- Attention
   if(!cConcatenated.FeedForward(cFeatureExtraction.AsObject(), cMemAndHidden.getOutput()))
      ReturnFalse;
   if(!cSoftMax.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   CBufferFloat *map = cSoftMax.getOutput();
   if(!bTSinRow)
     {
      if(!cTranspose.FeedForward(cSoftMax.AsObject()))
         ReturnFalse;
      map = cTranspose.getOutput();
     }
//---
   if(!ElementMult(cFeatureExtraction.getOutput(), map, cAttentionOut.getOutput()))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      ReturnFalse;
     }
//--- LSTM
   if(!CNeuronLSTMOCL::feedForward(cAttentionOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronLSTMOCL::calcInputGradients(cAttentionOut.AsObject()))
      ReturnFalse;
//---
   if(!ElementMultGrad(cFeatureExtraction.getOutput(), GetPointer(cTemp), (bTSinRow ? cSoftMax.getOutput() : cTranspose.getOutput()), (bTSinRow ? cSoftMax.getGradient() : cTranspose.getGradient()), cAttentionOut.getGradient(), NeuronOCL.Activation(), int(None)))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      ReturnFalse;
     }
//---
   if(bTSinRow)
     {
      if(!cSoftMax.CalcHiddenGradients(cTranspose.AsObject()))
         ReturnFalse;
     }
//---
   if(!cConcatenated.CalcHiddenGradients((CObject*)cSoftMax.AsObject(), NULL) ||
      !DeActivation(cConcatenated.getOutput(), cConcatenated.getGradient(), cConcatenated.getGradient(), cConcatenated.Activation()))
      ReturnFalse;
   if(!cFeatureExtraction.CalcHiddenGradients(cConcatenated.AsObject(), cMemAndHidden.getOutput(), cMemAndHidden.getGradient()))
      ReturnFalse;
   if(!DeActivation(cFeatureExtraction.getOutput(), GetPointer(cTemp), GetPointer(cTemp), NeuronOCL.Activation()) ||
      !SumAndNormalize(cFeatureExtraction.getGradient(), GetPointer(cTemp), cFeatureExtraction.getGradient(), 1, false))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cFeatureExtraction.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronLSTMOCL::updateInputWeights(cAttentionOut.AsObject()))
      ReturnFalse;
   if(!cFeatureExtraction.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cConcatenated.UpdateInputWeights(cFeatureExtraction.AsObject(), cMemAndHidden.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTPMEncoder::SetOpenCL(COpenCLMy * obj)
  {
   if(!OpenCL)
      cTemp.BufferFree();
//---
   CNeuronLSTMOCL::SetOpenCL(obj);
   cFeatureExtraction.SetOpenCL(OpenCL);
   cMemAndHidden.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cSoftMax.SetOpenCL(OpenCL);
   cAttentionOut.SetOpenCL(OpenCL);
   if(!bTSinRow)
      cTranspose.SetOpenCL(OpenCL);
   cTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::Save(const int file_handle)
  {
   if(!CNeuronLSTMOCL::Save(file_handle))
      ReturnFalse;
   if(!cFeatureExtraction.Save(file_handle))
      ReturnFalse;
   if(!cConcatenated.Save(file_handle))
      ReturnFalse;
   if(!cSoftMax.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(bTSinRow)) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPMEncoder::Load(const int file_handle)
  {
   if(!CNeuronLSTMOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenated.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSoftMax.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   bTSinRow = (bool)FileReadInteger(file_handle);
//---
   if(!cMemAndHidden.Init(0, 1, OpenCL, Neurons() * 2, optimization, iBatch))
      ReturnFalse;
   if(!cAttentionOut.Init(0, 3, OpenCL, cSoftMax.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   cTemp.BufferFree();
   if(!cTemp.BufferInit(cSoftMax.Neurons(), 0) || !cTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint variables, uint lenth, uint hidden_size, bool ts_in_row, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronLSTMOCL::Init(numOutputs, myIndex, open_cl, hidden_size, optimization_type, batch))
      ReturnFalse;
   if(!SetInputs(hidden_size))
      ReturnFalse;
//---
   if(!cEncoder.Init(0, 0, OpenCL, variables, lenth, hidden_size, ts_in_row, optimization, iBatch))
      ReturnFalse;
//---
   if(!cFeatureExtraction.Init(0, 1, OpenCL, variables, lenth, !ts_in_row, optimization, iBatch))
      ReturnFalse;
//---
   if(!cMemAndHidden.Init(0, 2, OpenCL, hidden_size * 2, optimization, iBatch))
      ReturnFalse;
   if(!cConcatenated.Init(0, 3, OpenCL, hidden_size, hidden_size, hidden_size * 2, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(TANH);
   if(!cSoftMax.Init(0, 4, OpenCL, hidden_size, optimization, iBatch))
      ReturnFalse;
   cSoftMax.SetHeads(1);
   if(!cAttentionOut.Init(0, 5, OpenCL, hidden_size, optimization, iBatch))
      ReturnFalse;
   if(!cAttAndFeature.Init(0, 6, OpenCL, hidden_size, hidden_size, variables * lenth, optimization, iBatch))
      ReturnFalse;
//---
   if(!cTemp.BufferInit(variables * lenth, 0) || !cTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Encoder
   if(!cEncoder.FeedForward(NeuronOCL))
      ReturnFalse;
//--- FEATURE EXTRACTION
   if(!cFeatureExtraction.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Memory and Hidden
   if(!Concat(m_iHiddenState, m_iMemory, m_iHiddenState, m_iMemory, cMemAndHidden.getOutputIndex(), 1, 1, 0, 0, Neurons()))
      ReturnFalse;
//--- Attention
   if(!cConcatenated.FeedForward(cEncoder.AsObject(), cMemAndHidden.getOutput()))
      ReturnFalse;
   if(!cSoftMax.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
//---
   if(!ElementMult(cEncoder.getOutput(), cSoftMax.getOutput(), cAttentionOut.getOutput()))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      ReturnFalse;
     }
//--- Attention and Features
   if(!cAttAndFeature.FeedForward(cAttentionOut.AsObject(), cFeatureExtraction.getOutput()))
      ReturnFalse;
//--- LSTM
   if(!CNeuronLSTMOCL::feedForward(cAttAndFeature.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronLSTMOCL::calcInputGradients(cAttAndFeature.AsObject()))
      ReturnFalse;
//---
   if(!cAttentionOut.CalcHiddenGradients(cAttAndFeature.AsObject(),
                                         cFeatureExtraction.getOutput(),
                                         cFeatureExtraction.getGradient(),
                                         (ENUM_ACTIVATION)cFeatureExtraction.Activation()
                                        ))
      ReturnFalse;
//---
   if(!ElementMultGrad(cEncoder.getOutput(), cEncoder.getGradient(),
                       cSoftMax.getOutput(), cSoftMax.getGradient(),
                       cAttentionOut.getGradient(), cEncoder.Activation(), int(None)))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      ReturnFalse;
     }
//---
   if(!cConcatenated.CalcHiddenGradients(cSoftMax.AsObject(), NULL) ||
      !DeActivation(cConcatenated.getOutput(), cConcatenated.getGradient(), cConcatenated.getGradient(), cConcatenated.Activation()))
      ReturnFalse;
   if(!cFeatureExtraction.CalcHiddenGradients(cConcatenated.AsObject(), cMemAndHidden.getOutput(), cMemAndHidden.getGradient()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getGradient(), GetPointer(cTemp),  1, false, 0, 0, 0, 0.5f))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cEncoder.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), GetPointer(cTemp), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 0.5f))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronLSTMOCL::updateInputWeights(cAttAndFeature.AsObject()))
      ReturnFalse;
   if(!cEncoder.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cFeatureExtraction.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cConcatenated.UpdateInputWeights(cEncoder.AsObject(), cMemAndHidden.getOutput()))
      ReturnFalse;
   if(!cAttAndFeature.UpdateInputWeights(cAttentionOut.AsObject(), cFeatureExtraction.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTPM::SetOpenCL(COpenCLMy * obj)
  {
   if(!OpenCL)
      cTemp.BufferFree();
//---
   CNeuronLSTMOCL::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
   cFeatureExtraction.SetOpenCL(OpenCL);
   cMemAndHidden.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cSoftMax.SetOpenCL(OpenCL);
   cAttentionOut.SetOpenCL(OpenCL);
   cAttAndFeature.SetOpenCL(OpenCL);
   cTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::Save(const int file_handle)
  {
   if(!CNeuronLSTMOCL::Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cFeatureExtraction.Save(file_handle))
      ReturnFalse;
   if(!cConcatenated.Save(file_handle))
      ReturnFalse;
   if(!cAttAndFeature.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTPM::Load(const int file_handle)
  {
   if(!CNeuronLSTMOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenated.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttAndFeature.AsObject()))
      ReturnFalse;
//---
   if(!cMemAndHidden.Init(0, 2, OpenCL, Neurons() * 2, optimization, iBatch))
      ReturnFalse;
   if(!cSoftMax.Init(0, 4, OpenCL, cConcatenated.Neurons(), optimization, iBatch))
      ReturnFalse;
   cSoftMax.SetHeads(1);
   if(!cAttentionOut.Init(0, 5, OpenCL, cSoftMax.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   cTemp.BufferFree();
   if(!cTemp.BufferInit(cFeatureExtraction.Neurons(), 0) || !cTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNEncoder::AttentionOut(CBufferFloat* q, CBufferFloat* kv, CBufferFloat* scores, CBufferFloat* out)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits/*Q units*/, iUnits/*K units*/, iHeads};
   uint local_work_size[3] = {1, iUnits, 1};
   ResetLastError();
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_q, q.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_kv, kv.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_score, scores.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_out, out.GetIndex())
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_dimension, (int)iWindowKey)
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_heads_kv, (int)iHeadsKV)
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_mask, 1)
   kernelExecuteLoc(def_k_MH2AttentionOut, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   CBufferFloat *kv = NULL;
   for(uint i = 0; (i < iLayers && !IsStopped()); i++)
     {
      //--- Feed Forward
      CBufferFloat *inputs = (i == 0 ? NeuronOCL.getOutput() : FF_Tensors.At(6 * i - 4));
      CBufferFloat *temp = FF_Tensors.At(i * 6 + 1);
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 1), inputs, temp, iWindow, 4 * iWindow, LReLU))
         ReturnFalse;
      inputs = FF_Tensors.At(i * 6);
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 2), temp, inputs, 4 * iWindow, iWindow, None))
         ReturnFalse;
      //--- Calculate Queries, Keys, Values
      CBufferFloat *q = QKV_Tensors.At(i * 2);
      if(IsStopped() || !ConvolutionForward(QKV_Weights.At(i * (optimization == SGD ? 2 : 3)), inputs, q, iWindow, iWindowKey * iHeads, None))
         ReturnFalse;
      if((i % iLayersToOneKV) == 0)
        {
         uint i_kv = i / iLayersToOneKV;
         kv = KV_Tensors.At(i_kv * 2);
         if(IsStopped() || !ConvolutionForward(KV_Weights.At(i_kv * (optimization == SGD ? 2 : 3)), inputs, kv, iWindow, 2 * iWindowKey * iHeadsKV, None))
            ReturnFalse;
        }
      //--- Score calculation and Multi-heads attention calculation
      temp = S_Tensors.At(i * 2);
      CBufferFloat *out = AO_Tensors.At(i * 2);
      if(IsStopped() || !AttentionOut(q, kv, temp, out))
         ReturnFalse;
      //--- Attention out calculation
      temp = FF_Tensors.At(i * 6 + 2);
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9)), out, temp, iWindowKey * iHeads, iWindow, None))
         ReturnFalse;
      //--- Sum and normilize attention
      if(IsStopped() || !SumAndNormalize(temp, inputs, temp, iWindow, true))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNEncoder::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   CBufferFloat *out_grad = Gradient;
   CBufferFloat *kv_g = KV_Tensors.At(KV_Tensors.Total() - 1);
//---
   for(int i = int(iLayers - 1); (i >= 0 && !IsStopped()); i--)
     {
      if(i == int(iLayers - 1) || (i + 1) % iLayersToOneKV == 0)
         kv_g = KV_Tensors.At((i / iLayersToOneKV) * 2 + 1);
      //--- Split gradient to multi-heads
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9)), out_grad, AO_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1), iWindowKey * iHeads, iWindow, None))
         ReturnFalse;
      //--- Passing gradient to query, key and value
      if(i == int(iLayers - 1) || (i + 1) % iLayersToOneKV == 0)
        {
         if(IsStopped() || !AttentionInsideGradients(QKV_Tensors.At(i * 2), QKV_Tensors.At(i * 2 + 1), KV_Tensors.At((i / iLayersToOneKV) * 2), kv_g, S_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1)))
            ReturnFalse;
        }
      else
        {
         if(IsStopped() || !AttentionInsideGradients(QKV_Tensors.At(i * 2), QKV_Tensors.At(i * 2 + 1), KV_Tensors.At((i / iLayersToOneKV) * 2), GetPointer(Temp), S_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1)))
            ReturnFalse;
         if(IsStopped() || !SumAndNormalize(kv_g, GetPointer(Temp), kv_g, iWindowKey, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      //---
      CBufferFloat *inp = FF_Tensors.At(i * 6);
      CBufferFloat *temp = FF_Tensors.At(i * 6 + 3);
      if(IsStopped() || !ConvolutionInputGradients(QKV_Weights.At(i * (optimization == SGD ? 2 : 3)), QKV_Tensors.At(i * 2 + 1), inp, temp, iWindow, iWindowKey * iHeads, None))
         ReturnFalse;
      //--- Sum and normilize gradients
      if(IsStopped() || !SumAndNormalize(out_grad, temp, temp, iWindow, false, 0, 0, 0, 1))
         ReturnFalse;
      //---
      if((i % iLayersToOneKV) == 0)
        {
         if(IsStopped() || !ConvolutionInputGradients(KV_Weights.At(i / iLayersToOneKV * (optimization == SGD ? 2 : 3)), kv_g, inp, GetPointer(Temp), iWindow, 2 * iWindowKey * iHeadsKV, None))
            ReturnFalse;
         if(IsStopped() || !SumAndNormalize(GetPointer(Temp), temp, temp, iWindow, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      //--- Passing gradient through feed forward layers
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 2), out_grad, FF_Tensors.At(i * 6 + 1), FF_Tensors.At(i * 6 + 4), 4 * iWindow, iWindow, None))
         ReturnFalse;
      inp = (i > 0 ? FF_Tensors.At(i * 6 - 4) : prevLayer.getOutput());
      temp = (i > 0 ? FF_Tensors.At(i * 6 - 1) : prevLayer.getGradient());
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 1), FF_Tensors.At(i * 6 + 4), inp, temp, iWindow, 4 * iWindow, LReLU))
         ReturnFalse;
      out_grad = temp;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNEncoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *inputs = NeuronOCL.getOutput();
   for(uint l = 0; l < iLayers; l++)
     {
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9) + 1), FF_Tensors.At(l * 6 + 4), inputs, (optimization == SGD ? FF_Weights.At(l * 6 + 4) : FF_Weights.At(l * 9 + 4)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 7)), iWindow, 4 * iWindow))
         ReturnFalse;
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9) + 2), FF_Tensors.At(l * 6 + 3), FF_Tensors.At(l * 6 + 1), (optimization == SGD ? FF_Weights.At(l * 6 + 5) : FF_Weights.At(l * 9 + 5)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 8)), 4 * iWindow, iWindow))
         ReturnFalse;
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(QKV_Weights.At(l * (optimization == SGD ? 2 : 3)), QKV_Tensors.At(l * 2 + 1), FF_Tensors.At(l * 6), (optimization == SGD ? QKV_Weights.At(l * 2 + 1) : QKV_Weights.At(l * 3 + 1)), (optimization == SGD ? NULL : QKV_Weights.At(l * 3 + 2)), iWindow, iWindowKey * iHeads))
         ReturnFalse;
      if(l % iLayersToOneKV == 0)
        {
         uint l_kv = l / iLayersToOneKV;
         if(IsStopped() || !ConvolutuionUpdateWeights(KV_Weights.At(l_kv * (optimization == SGD ? 2 : 3)), KV_Tensors.At(l_kv * 2 + 1), FF_Tensors.At(l * 6), (optimization == SGD ? KV_Weights.At(l_kv * 2 + 1) : KV_Weights.At(l_kv * 3 + 1)), (optimization == SGD ? NULL : KV_Weights.At(l_kv * 3 + 2)), iWindow, 2 * iWindowKey * iHeadsKV))
            ReturnFalse;
        }
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9)), FF_Tensors.At(l * 6 + 5), AO_Tensors.At(l * 2), (optimization == SGD ? FF_Weights.At(l * 6 + 3) : FF_Weights.At(l * 9 + 3)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 6)), iWindowKey * iHeads, iWindow))
         ReturnFalse;
      //---
      inputs = FF_Tensors.At(l * 6 + 2);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint window_kv, uint heads_kv, uint units_count, uint units_count_kv, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!cEncoder.Init(0, 0, open_cl, window, window_key, heads, heads_kv, units_count, layers, layers_to_one_kv, optimization_type, batch))
      ReturnFalse;
   if(!CNeuronMLCrossAttentionMLKV::Init(numOutputs, myIndex, open_cl, window, window_key, heads, window_kv, heads_kv, units_count, units_count_kv, layers, layers_to_one_kv, optimization_type, batch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::AttentionOut(CBufferFloat* q, CBufferFloat* kv, CBufferFloat* scores, CBufferFloat* out)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits/*Q units*/, iUnitsKV/*K units*/, iHeads};
   uint local_work_size[3] = {1, iUnitsKV, 1};
   ResetLastError();
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_q, q.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_kv, kv.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_score, scores.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_out, out.GetIndex())
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_dimension, (int)iWindowKey)
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_heads_kv, (int)iHeadsKV)
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_mask, 1)
   kernelExecuteLoc(def_k_MH2AttentionOut, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* Context)
  {
   if(!cEncoder.FeedForward(NeuronOCL, Context))
      ReturnFalse;
   if(!CNeuronMLCrossAttentionMLKV::feedForward(cEncoder.AsObject(), Context))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!CNeuronMLCrossAttentionMLKV::calcInputGradients(cEncoder.AsObject(), SecondInput, SecondGradient, SecondActivation))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cEncoder.AsObject(), SecondInput, SecondGradient, SecondActivation))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* Context)
  {
   if(!cEncoder.UpdateInputWeights(NeuronOCL, Context))
      ReturnFalse;
   if(!CNeuronMLCrossAttentionMLKV::updateInputWeights(cEncoder.AsObject(), Context))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::Save(const int file_handle)
  {
   if(!CNeuronMLCrossAttentionMLKV::Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::Load(const int file_handle)
  {
   if(!CNeuronMLCrossAttentionMLKV::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEncoder.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTNNDecoder::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronMLCrossAttentionMLKV::WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cEncoder.WeightsUpdate(((CNeuronSTNNDecoder*)source).cEncoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTNNDecoder::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMLCrossAttentionMLKV::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                            uint sequence, uint variables,
                            uint period, uint forecast,
                            ENUM_OPTIMIZATION optimization_type, uint batch
                           )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, forecast * variables, optimization_type, batch))
      ReturnFalse;
   if(!acTranspose[0].Init(0, 0, OpenCL, sequence, variables, optimization, iBatch))
      ReturnFalse;
   if(!cConvolution.Init(0, 1, OpenCL, 2 * period, period, period, sequence / period, variables, optimization, iBatch))
      ReturnFalse;
   cConvolution.SetActivationFunction(None);
   if(!acTranspose[1].Init(0, 2, OpenCL, variables * sequence / period, period, optimization, iBatch))
      ReturnFalse;
   if(!cForecast.Init(0, 3, OpenCL, sequence / period, sequence / period, forecast / period, variables, period, optimization, iBatch))
      ReturnFalse;
   cForecast.SetActivationFunction(TANH);
   if(!acTranspose[2].Init(0, 4, OpenCL, period, variables * forecast / period, optimization, iBatch))
      ReturnFalse;
   if(!acTranspose[3].Init(0, 5, OpenCL, variables, forecast, optimization, iBatch))
      ReturnFalse;
//---
   if(!SetOutput(acTranspose[3].getOutput()) ||
      !SetGradient(acTranspose[3].getGradient())
     )
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!acTranspose[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cConvolution.FeedForward(acTranspose[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cConvolution.getOutput(), acTranspose[0].getOutput(), cConvolution.getOutput(), 1, false))
      ReturnFalse;
   if(!acTranspose[1].FeedForward(cConvolution.AsObject()))
      ReturnFalse;
   if(!cForecast.FeedForward(acTranspose[1].AsObject()))
      ReturnFalse;
   if(!acTranspose[2].FeedForward(cForecast.AsObject()))
      ReturnFalse;
   if(!acTranspose[3].FeedForward(acTranspose[2].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!acTranspose[2].CalcHiddenGradients(acTranspose[3].AsObject()))
      ReturnFalse;
   if(!cForecast.CalcHiddenGradients(acTranspose[2].AsObject()))
      ReturnFalse;
   if(cForecast.Activation() != None &&
      !DeActivation(cForecast.getOutput(), cForecast.getGradient(), cForecast.getGradient(), cForecast.Activation()))
      ReturnFalse;
   if(!acTranspose[1].CalcHiddenGradients(cForecast.AsObject()))
      ReturnFalse;
   if(!cConvolution.CalcHiddenGradients(acTranspose[1].AsObject()))
      ReturnFalse;
   if(!acTranspose[0].CalcHiddenGradients(cConvolution.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cConvolution.getGradient(), acTranspose[0].getGradient(), acTranspose[0].getGradient(), 1, false))
      ReturnFalse;
   if(!NeuronOCL || !NeuronOCL.CalcHiddenGradients(acTranspose[0].AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() != None &&
      !DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cConvolution.UpdateInputWeights(acTranspose[0].AsObject()))
      ReturnFalse;
   if(!cForecast.UpdateInputWeights(acTranspose[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSparseTSF::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(int i = 0; i < 4; i++)
      acTranspose[i].SetOpenCL(OpenCL);
   cConvolution.SetOpenCL(OpenCL);
   cForecast.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSparseTSF *Source = source;
   if(!cConvolution.WeightsUpdate(Source.cConvolution.AsObject(), tau))
      ReturnFalse;
   if(!cForecast.WeightsUpdate(Source.cForecast.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 4; i++)
      if(!acTranspose[i].Save(file_handle))
         ReturnFalse;;
   if(!cConvolution.Save(file_handle))
      ReturnFalse;
   if(!cForecast.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseTSF::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   for(int i = 0; i < 4; i++)
      if(!LoadInsideLayer(file_handle, acTranspose[i].AsObject()))
         ReturnFalse;;
   if(!LoadInsideLayer(file_handle, cConvolution.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cForecast.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint sequence, uint variables, uint forecast, uint heads, uint layers, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- base
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, forecast * variables, optimization_type, batch))
      ReturnFalse;
//--- constants
   iVariables = variables;
   iForecast = forecast;
   iSequence = MathMax(sequence, 1);
//--- Calculate FFTsize
   uint size = iSequence;
   int power = int(MathLog(size) / M_LN2);
   if(MathPow(2, power) < size)
      power++;
   iFFT = uint(MathPow(2, power));
//--- trend
   if(!cPLR.Init(0, 0, OpenCL, iVariables, iSequence, true, optimization, iBatch))
      ReturnFalse;
   if(!cTrend.Init(0, 1, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
//--- seasons
   if(!cInputSeasons.Init(0, 2, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTranspose[0].Init(0, 3, OpenCL, iSequence, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTranspose[1].Init(0, 4, OpenCL, iVariables, iSequence, optimization, iBatch))
      ReturnFalse;
   if(!cInputFreqRe.BufferInit(iFFT * iVariables, 0) || !cInputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqIm.BufferInit(iFFT * iVariables, 0) || !cInputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqComplex.Init(0, 5, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cNormFreqComplex.Init(0, 6, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cMeans.BufferInit(iVariables, 0) || !cMeans.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cVariances.BufferInit(iVariables, 0) || !cVariances.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cFreqAtteention.Init(0, 7, OpenCL, iFFT, 32, heads, iVariables, layers, optimization, batch))
      ReturnFalse;
   if(!cUnNormFreqComplex.Init(0, 8, OpenCL, iFFT * iVariables * 2, optimization, batch))
      ReturnFalse;
   if(!cOutputFreqRe.BufferInit(iFFT * iVariables, 0) || !cOutputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputFreqIm.BufferInit(iFFT * iVariables, 0) || !cOutputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasRe.Init(0, 9, OpenCL, iFFT * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cOutputTimeSeriasIm.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cZero.BufferInit(iFFT * iVariables, 0) || !cZero.BufferCreate(OpenCL))
      ReturnFalse;
//--- Noise
   if(!cResidual.Init(0, 10, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
//--- Forecast
   if(!cConcatInput.Init(0, 11, OpenCL, 3 * iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cNormalize.Init(0, 12, OpenCL, 3 * iSequence * iVariables, iBatch, optimization))
      ReturnFalse;
   int window = MathMin(5, (int)iSequence - 1);
   int patches = (int)iSequence - window + 1;
   if(!cPatching.Init(0, 13, OpenCL, window, 1, 8, patches, 3 * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!acPE[0].Init(0, 14, OpenCL, patches, 3 * 8 * iVariables, optimization, iBatch))
      ReturnFalse;
   int plr = cPLR.Neurons();
   if(!cNormalizePLR.Init(0, 15, OpenCL, plr, iBatch, optimization))
      ReturnFalse;
   plr = MathMax(plr / (3 * (int)iVariables), 1);
   if(!cPatchingPLR.Init(0, 16, OpenCL, 3, 3, 8, plr, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!acPE[1].Init(0, 17, OpenCL, plr, 8 * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cAttention.Init(0, 18, OpenCL, 3 * 8 * iVariables, 3 * iVariables, MathMax(heads, 1), 8 * iVariables, MathMax(heads / 2, 1), patches, plr, MathMax(layers, 1), 2, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeAtt.Init(0, 19, OpenCL, patches, 3 * 8 * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!acForecast[0].Init(0, 20, OpenCL, patches, patches, iForecast, 3 * 8 * iVariables, optimization, iBatch))
      ReturnFalse;
   acForecast[0].SetActivationFunction(LReLU);
   if(!acForecast[1].Init(0, 21, OpenCL, 8 * iForecast, 8 * iForecast, iForecast, 3 * iVariables, optimization, iBatch))
      ReturnFalse;
   acForecast[1].SetActivationFunction(TANH);
   if(!cTransposeFrc.Init(0, 22, OpenCL, 3 * iVariables, iForecast, optimization, iBatch))
      ReturnFalse;
   if(!cRevIn.Init(0, 23, OpenCL, 3 * iVariables * iForecast, 11, GetPointer(cNormalize)))
      ReturnFalse;
   if(!cSum.Init(0, 24, OpenCL, 3, 3, 1, iVariables, iForecast, optimization, iBatch))
      ReturnFalse;
   cSum.SetActivationFunction(None);
//---
   SetActivationFunction(None);
   SetOutput(cSum.getOutput(), true);
   SetGradient(cSum.getGradient(), true);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_FFT, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(def_k_FFT, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(def_k_FFT, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(def_k_FFT, def_k_fft_input_window, (int)(inp_re.Total() / iVariables))
   setArgument(def_k_FFT, def_k_fft_input_complex, int(!!inp_im))
   setArgument(def_k_FFT, def_k_fft_output_window, (int)(out_re.Total() / iVariables))
   setArgument(def_k_FFT, def_k_fft_reverse, int(reverse))
   kernelExecute(def_k_FFT, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::ComplexNormalize(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexNormalize, def_k_cn_inputs, cInputFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_outputs, cNormFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_means, cMeans.GetIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalize, def_k_cn_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexNormalize, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::ComplexUnNormalize(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_inputs, cFreqAtteention.getOutputIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_outputs, cUnNormFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_means, cMeans.GetIndex())
   setBuffer(def_k_ComplexUnNormalize, def_k_cn_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexUnNormalize, def_k_cn_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexUnNormalize, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFreqAtteention.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::ComplexNormalizeGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_inputs_gr, cInputFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_outputs_gr, cNormFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalizeGradient, def_k_cng_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexNormalizeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::ComplexUnNormalizeGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iVariables};
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_inputs_gr, cFreqAtteention.getGradientIndex())
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_outputs_gr, cUnNormFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexUnNormalizeGradient, def_k_cng_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexUnNormalizeGradient, def_k_cng_dimension, (int)(iFFT))
   kernelExecute(def_k_ComplexUnNormalizeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFreqAtteention.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::CutTrendAndOther(CBufferFloat* inputs)
  {
   if(!inputs)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iSequence, iVariables};
   setBuffer(def_k_CutTrendAndOther, def_k_ct_inputs, inputs.GetIndex())
   setBuffer(def_k_CutTrendAndOther, def_k_ct_plr, cPLR.getOutputIndex())
   setBuffer(def_k_CutTrendAndOther, def_k_ct_trend, cTrend.getOutputIndex())
   setBuffer(def_k_CutTrendAndOther, def_k_ct_other, cInputSeasons.getOutputIndex())
   kernelExecute(def_k_CutTrendAndOther, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputSeasons.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::CutTrendAndOtherGradient(CBufferFloat* inputs_gr)
  {
   if(!inputs_gr)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iSequence, iVariables};
   setBuffer(def_k_CutTrendAndOtherGradient, def_k_ctg_inputs_gr, inputs_gr.GetIndex())
   setBuffer(def_k_CutTrendAndOtherGradient, def_k_ctg_plr, cPLR.getOutputIndex())
   setBuffer(def_k_CutTrendAndOtherGradient, def_k_ctg_plr_gr, cPLR.getGradientIndex())
   setBuffer(def_k_CutTrendAndOtherGradient, def_k_ctg_trend_gr, cTrend.getGradientIndex())
   setBuffer(def_k_CutTrendAndOtherGradient, def_k_ctg_other_gr, cInputSeasons.getGradientIndex())
   kernelExecute(def_k_CutTrendAndOtherGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cPLR.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::CutOneFromAnother(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iSequence * iVariables};
   setBuffer(def_k_CutOneFromAnother, def_k_ctofa_inputs, cInputSeasons.getOutputIndex())
   setBuffer(def_k_CutOneFromAnother, def_k_ctofa_cut, cTranspose[1].getOutputIndex())
   setBuffer(def_k_CutOneFromAnother, def_k_ctofa_other, cResidual.getOutputIndex())
   kernelExecute(def_k_CutOneFromAnother, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cResidual.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::CutOneFromAnotherGradient(void)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iSequence * iVariables};
   setBuffer(def_k_CutOneFromAnotherGradient, def_k_ctofa_inputs, cInputSeasons.getGradientIndex())
   setBuffer(def_k_CutOneFromAnotherGradient, def_k_ctofa_cut, cOutputTimeSeriasRe.getGradientIndex())
   setBuffer(def_k_CutOneFromAnotherGradient, def_k_ctofa_other, cResidual.getGradientIndex())
   kernelExecute(def_k_CutOneFromAnother, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cResidual.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- trend
   if(!cPLR.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CutTrendAndOther(NeuronOCL.getOutput()))
      ReturnFalse;
//--- seasons
   if(!cTranspose[0].FeedForward(cInputSeasons.AsObject()))
      ReturnFalse;
   if(!FFT(cTranspose[0].getOutput(), NULL, GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), false))
      ReturnFalse;
   if(!Concat(GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), cInputFreqComplex.getOutput(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!ComplexNormalize())
      ReturnFalse;
   if(!cFreqAtteention.FeedForward(cNormFreqComplex.AsObject()))
      ReturnFalse;
   if(!ComplexUnNormalize())
      ReturnFalse;
   if(!DeConcat(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), cUnNormFreqComplex.getOutput(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!FFT(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), GetPointer(cInputFreqRe), GetPointer(cOutputTimeSeriasIm), true))
      ReturnFalse;
   if(!DeConcat(cOutputTimeSeriasRe.getOutput(), cOutputTimeSeriasRe.getGradient(), GetPointer(cInputFreqRe), iSequence, iFFT - iSequence, iVariables))
      ReturnFalse;
   if(!cTranspose[1].FeedForward(cOutputTimeSeriasRe.AsObject()))
      ReturnFalse;
//--- Noise
   if(!CutOneFromAnother())
      ReturnFalse;
//--- Forecast
   if(!Concat(cTrend.getOutput(), cTranspose[1].getOutput(), cResidual.getOutput(), cConcatInput.getOutput(), 1, 1, 1, 3 * iSequence * iVariables))
      ReturnFalse;
   if(!cNormalize.FeedForward(cConcatInput.AsObject()))
      ReturnFalse;
   if(!cPatching.FeedForward(cNormalize.AsObject()))
      ReturnFalse;
   if(!acPE[0].FeedForward(cPatching.AsObject()))
      ReturnFalse;
   if(!cNormalizePLR.FeedForward(cPLR.AsObject()))
      ReturnFalse;
   if(!cPatchingPLR.FeedForward(cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!acPE[1].FeedForward(cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!cAttention.FeedForward(acPE[0].AsObject(), acPE[1].getOutput()))
      ReturnFalse;
   if(!cTransposeAtt.FeedForward(cAttention.AsObject()))
      ReturnFalse;
   if(!acForecast[0].FeedForward(cTransposeAtt.AsObject()))
      ReturnFalse;
   if(!acForecast[1].FeedForward(acForecast[0].AsObject()))
      ReturnFalse;
   if(!cTransposeFrc.FeedForward(acForecast[1].AsObject()))
      ReturnFalse;
   if(!cRevIn.FeedForward(cTransposeFrc.AsObject()))
      ReturnFalse;
   if(!cSum.FeedForward(cRevIn.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Devide to Trend, Seasons and Noise
   if(!cRevIn.CalcHiddenGradients(cSum.AsObject()))
      ReturnFalse;
//--- Forecast gradient
   if(!cTransposeFrc.CalcHiddenGradients(cRevIn.AsObject()))
      ReturnFalse;
   if(!acForecast[1].CalcHiddenGradients(cTransposeFrc.AsObject()))
      ReturnFalse;
   if(acForecast[1].Activation() != None &&
      !DeActivation(acForecast[1].getOutput(), acForecast[1].getGradient(), acForecast[1].getGradient(), acForecast[1].Activation())
     )
      ReturnFalse;
   if(!acForecast[0].CalcHiddenGradients(acForecast[1].AsObject()))
      ReturnFalse;
//--- Attention gradient
   if(!cTransposeAtt.CalcHiddenGradients(acForecast[0].AsObject()))
      ReturnFalse;
   if(!cAttention.CalcHiddenGradients(cTransposeAtt.AsObject()))
      ReturnFalse;
   if(!acPE[0].CalcHiddenGradients(cAttention.AsObject(), acPE[1].getOutput(), acPE[1].getGradient(), (ENUM_ACTIVATION)acPE[1].Activation()))
      ReturnFalse;
//--- Gradient to PLR
   if(!cPatchingPLR.CalcHiddenGradients(acPE[1].AsObject()))
      ReturnFalse;
   if(!cNormalizePLR.CalcHiddenGradients(cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!cPLR.CalcHiddenGradients(cNormalizePLR.AsObject()))
      ReturnFalse;
//--- Gradient to Concatenate buffer of Trend, Season and Noise
   if(!cPatching.CalcHiddenGradients(acPE[0].AsObject()))
      ReturnFalse;
   if(!cNormalize.CalcHiddenGradients(cPatching.AsObject()))
      ReturnFalse;
   if(!cConcatInput.CalcHiddenGradients(cNormalize.AsObject()))
      ReturnFalse;
//--- DeConcatenate
   if(!DeConcat(cTrend.getGradient(), cOutputTimeSeriasRe.getGradient(), cResidual.getGradient(), cConcatInput.getGradient(), 1, 1, 1, 3 * iSequence * iVariables))
      ReturnFalse;
//--- Seasons
   if(!CutOneFromAnotherGradient())
      ReturnFalse;
   if(!SumAndNormalize(cOutputTimeSeriasRe.getGradient(), cTranspose[1].getGradient(), cTranspose[1].getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cOutputTimeSeriasRe.CalcHiddenGradients(cTranspose[1].AsObject()))
      ReturnFalse;
   if(!Concat(cOutputTimeSeriasRe.getGradient(), GetPointer(cZero), GetPointer(cInputFreqRe), iSequence, iFFT - iSequence, iVariables))
      ReturnFalse;
   if(!SumAndNormalize(GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputTimeSeriasIm), 1, false, 0, 0, 0, -0.5f))
      ReturnFalse;
   if(!FFT(GetPointer(cInputFreqRe), GetPointer(cOutputTimeSeriasIm), GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), false))
      ReturnFalse;
   if(!Concat(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), cUnNormFreqComplex.getGradient(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!ComplexUnNormalizeGradient())
      ReturnFalse;
   if(!cNormFreqComplex.CalcHiddenGradients(cFreqAtteention.AsObject()))
      ReturnFalse;
   if(!ComplexNormalizeGradient())
      ReturnFalse;
   if(!DeConcat(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), cInputFreqComplex.getGradient(), 1, 1, iFFT * iVariables))
      ReturnFalse;
   if(!FFT(GetPointer(cOutputFreqRe), GetPointer(cOutputFreqIm), GetPointer(cInputFreqRe), GetPointer(cInputFreqIm), true))
      ReturnFalse;
   if(!DeConcat(cTranspose[0].getGradient(), GetPointer(cInputFreqIm), GetPointer(cInputFreqRe), iSequence, iFFT - iSequence, iVariables))
      ReturnFalse;
   if(!cInputSeasons.CalcHiddenGradients(cTranspose[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cInputSeasons.getGradient(), cResidual.getGradient(), cInputSeasons.getGradient(), 1, 1, false, 0, 0, 1))
      ReturnFalse;
//--- trend
   if(!CutTrendAndOtherGradient(NeuronOCL.getGradient()))
      ReturnFalse;
//---  input gradient
   if(!NeuronOCL.CalcHiddenGradients(cPLR.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cInputSeasons.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- trend
   if(!cPLR.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//--- seasons
   if(!cFreqAtteention.UpdateInputWeights(cNormFreqComplex.AsObject()))
      ReturnFalse;
//--- Forecast
   if(!cNormalize.UpdateInputWeights(cConcatInput.AsObject()))
      ReturnFalse;
   if(!cPatching.UpdateInputWeights(cNormalize.AsObject()))
      ReturnFalse;
   if(!acPE[0].UpdateInputWeights(cPatching.AsObject()))
      ReturnFalse;
   if(!cNormalizePLR.UpdateInputWeights(cPLR.AsObject()))
      ReturnFalse;
   if(!cPatchingPLR.UpdateInputWeights(cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!acPE[1].UpdateInputWeights(cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!cAttention.UpdateInputWeights(acPE[0].AsObject(), acPE[1].getOutput()))
      ReturnFalse;
   if(!acForecast[0].UpdateInputWeights(cTransposeAtt.AsObject()))
      ReturnFalse;
   if(!acForecast[1].UpdateInputWeights(acForecast[0].AsObject()))
      ReturnFalse;
   if(!cSum.UpdateInputWeights(cRevIn.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//--- constants
   if(FileWriteInteger(file_handle, int(iVariables)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iSequence)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iForecast)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iFFT)) < INT_VALUE)
      ReturnFalse;
//--- trend
   if(!cPLR.Save(file_handle))
      ReturnFalse;
//--- seasons
   if(!cFreqAtteention.Save(file_handle))
      ReturnFalse;
//--- Forecast
   if(!cNormalize.Save(file_handle))
      ReturnFalse;
   if(!cPatching.Save(file_handle))
      ReturnFalse;
   if(!acPE[0].Save(file_handle))
      ReturnFalse;
   if(!cNormalizePLR.Save(file_handle))
      ReturnFalse;
   if(!cPatchingPLR.Save(file_handle))
      ReturnFalse;
   if(!acPE[1].Save(file_handle))
      ReturnFalse;
   if(!cAttention.Save(file_handle))
      ReturnFalse;
   if(!acForecast[0].Save(file_handle))
      ReturnFalse;
   if(!acForecast[1].Save(file_handle))
      ReturnFalse;
   if(!cSum.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTEMPOOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//--- constants
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iSequence = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iForecast = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFFT = (uint)FileReadInteger(file_handle);
//--- trend
   if(!LoadInsideLayer(file_handle, cPLR.AsObject()))
      ReturnFalse;
//--- seasons
   if(!LoadInsideLayer(file_handle, cFreqAtteention.AsObject()))
      ReturnFalse;
//--- Forecast
   if(!LoadInsideLayer(file_handle, cNormalize.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatching.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, acPE[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormalizePLR.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatchingPLR.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, acPE[1].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, acForecast[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, acForecast[1].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSum.AsObject()))
      ReturnFalse;
//--- ReInit Objects
//--- trend
   if(!cTrend.Init(0, 1, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
//--- seasons
   if(!cInputSeasons.Init(0, 2, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTranspose[0].Init(0, 3, OpenCL, iSequence, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTranspose[1].Init(0, 4, OpenCL, iVariables, iSequence, optimization, iBatch))
      ReturnFalse;
   cInputFreqRe.BufferFree();
   if(!cInputFreqRe.BufferInit(iFFT * iVariables, 0) || !cInputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   cInputFreqIm.BufferFree();
   if(!cInputFreqIm.BufferInit(iFFT * iVariables, 0) || !cInputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cInputFreqComplex.Init(0, 5, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   if(!cNormFreqComplex.Init(0, 6, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   cMeans.BufferFree();
   if(!cMeans.BufferInit(iVariables, 0) || !cMeans.BufferCreate(OpenCL))
      ReturnFalse;
   cVariances.BufferFree();
   if(!cVariances.BufferInit(iVariables, 0) || !cVariances.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cUnNormFreqComplex.Init(0, 8, OpenCL, iFFT * iVariables * 2, optimization, iBatch))
      ReturnFalse;
   cOutputFreqRe.BufferFree();
   if(!cOutputFreqRe.BufferInit(iFFT * iVariables, 0) || !cOutputFreqRe.BufferCreate(OpenCL))
      ReturnFalse;
   cOutputFreqIm.BufferFree();
   if(!cOutputFreqIm.BufferInit(iFFT * iVariables, 0) || !cOutputFreqIm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cOutputTimeSeriasRe.Init(0, 9, OpenCL, iFFT * iVariables, optimization, iBatch))
      ReturnFalse;
   cOutputTimeSeriasIm.BufferFree();
   if(!cOutputTimeSeriasIm.BufferInit(iFFT * iVariables, 0) || !cOutputTimeSeriasIm.BufferCreate(OpenCL))
      ReturnFalse;
   cZero.BufferFree();
   if(!cZero.BufferInit(iFFT * iVariables, 0) || !cZero.BufferCreate(OpenCL))
      ReturnFalse;
//--- Noise
   if(!cResidual.Init(0, 10, OpenCL, iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
//--- Forecast
   if(!cConcatInput.Init(0, 11, OpenCL, 3 * iSequence * iVariables, optimization, iBatch))
      ReturnFalse;
   uint patches = cAttention.Neurons() / (3 * 8 * iVariables);
   if(!cTransposeAtt.Init(0, 19, OpenCL, patches, 3 * 8 * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeFrc.Init(0, 22, OpenCL, 3 * iVariables, iForecast, optimization, iBatch))
      ReturnFalse;
   if(!cRevIn.Init(0, 23, OpenCL, 3 * iVariables * iForecast, 11, GetPointer(cNormalize)))
      ReturnFalse;
//---
   SetOutput(cSum.getOutput(), true);
   SetGradient(cSum.getGradient(), true);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTEMPOOCL::SetOpenCL(COpenCLMy * obj)
  {
//--- base
   CNeuronBaseOCL::SetOpenCL(obj);
//--- trend
   cPLR.SetOpenCL(OpenCL);
   cTrend.SetOpenCL(OpenCL);
//--- seasons
   cInputSeasons.SetOpenCL(OpenCL);
   cTranspose[0].SetOpenCL(OpenCL);
   cTranspose[1].SetOpenCL(OpenCL);
   cInputFreqRe.BufferCreate(OpenCL);
   cInputFreqIm.BufferCreate(OpenCL);
   cInputFreqComplex.SetOpenCL(OpenCL);
   cNormFreqComplex.SetOpenCL(OpenCL);
   cMeans.BufferCreate(OpenCL);
   cVariances.BufferCreate(OpenCL);
   cFreqAtteention.SetOpenCL(OpenCL);
   cUnNormFreqComplex.SetOpenCL(OpenCL);
   cOutputFreqRe.BufferCreate(OpenCL);
   cOutputFreqIm.BufferCreate(OpenCL);
   cOutputTimeSeriasRe.SetOpenCL(OpenCL);
   cOutputTimeSeriasIm.BufferCreate(OpenCL);
   cZero.BufferCreate(OpenCL);
//--- Noise
   cResidual.SetOpenCL(OpenCL);
//--- Forecast
   cConcatInput.SetOpenCL(OpenCL);
   cNormalize.SetOpenCL(OpenCL);
   cPatching.SetOpenCL(OpenCL);
   acPE[0].SetOpenCL(OpenCL);
   cNormalizePLR.SetOpenCL(OpenCL);
   cPatchingPLR.SetOpenCL(OpenCL);
   acPE[1].SetOpenCL(OpenCL);
   cAttention.SetOpenCL(OpenCL);
   cTransposeAtt.SetOpenCL(OpenCL);
   acForecast[0].SetOpenCL(OpenCL);
   acForecast[1].SetOpenCL(OpenCL);
   cTransposeFrc.SetOpenCL(OpenCL);
   cRevIn.SetOpenCL(OpenCL);
   cSum.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronTEMPOOCL::getWeights(void)
  {
   CBufferFloat *result = CNeuronBaseOCL::getWeights();
   if(!result)
      result = cSum.GetWeightsConv();
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * forecast, optimization_type, batch))
      ReturnFalse;
//--- State Encoder
   if(!cStateRNN.Init(0, 0, OpenCL, window_key, units_count, optimization, iBatch) ||
      !cStateRNN.SetInputs(window))
      ReturnFalse;
   if(!cStateMLP[0].Init(0, 1, OpenCL, window_key, window_key, 4 * window_key, units_count, optimization, iBatch))
      ReturnFalse;
   cStateMLP[0].SetActivationFunction(LReLU);
   if(!cStateMLP[1].Init(0, 2, OpenCL, 4 * window_key, 4 * window_key, window_key, units_count, optimization, iBatch))
      ReturnFalse;
//--- Variables Encoder
   if(!cTranspose.Init(0, 3, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
   if(!cVariablesRNN.Init(0, 4, OpenCL, window_key, window, optimization, iBatch) ||
      !cVariablesRNN.SetInputs(units_count))
      ReturnFalse;
   if(!cVariablesMLP[0].Init(0, 5, OpenCL, window_key, window_key, 4 * window_key, window, optimization, iBatch))
      ReturnFalse;
   cVariablesMLP[0].SetActivationFunction(LReLU);
   if(!cVariablesMLP[1].Init(0, 6, OpenCL, 4 * window_key, 4 * window_key, window_key, window, optimization, iBatch))
      ReturnFalse;
//--- Position Encoder
   if(!cStatePE.Init(0, 7, OpenCL, cStateMLP[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cVariablesPE.Init(0, 8, OpenCL, cVariablesMLP[1].Neurons(), optimization, iBatch))
      ReturnFalse;
//--- Context
   if(!cStateToState.Init(0, 9, OpenCL, window_key, window_key, heads, heads / 2, units_count, 2, 1, optimization, iBatch))
      ReturnFalse;
   if(!cStateToVariable.Init(0, 10, OpenCL, window_key, window_key, heads, window_key, heads / 2, units_count, window, 2, 1, optimization, iBatch))
      ReturnFalse;
   if(!cVariableToState.Init(0, 11, OpenCL, window_key, window_key, heads, window_key, heads / 2, window, units_count, 2, 1, optimization, iBatch))
      ReturnFalse;
   if(!cContext.Init(0, 12, OpenCL, window_key * (units_count + window), optimization, iBatch))
      ReturnFalse;
   if(!cContextMLP[0].Init(0, 13, OpenCL, window_key, window_key, 4 * window_key, window + units_count, optimization, iBatch))
      ReturnFalse;
   cContextMLP[0].SetActivationFunction(LReLU);
   if(!cContextMLP[1].Init(0, 14, OpenCL, 4 * window_key, 4 * window_key, window_key, window + units_count, optimization, iBatch))
      ReturnFalse;
//---
   if(!cHighLevelInteraction.Init(0, 15, OpenCL, window_key, window_key, heads, heads / 2, window + units_count, 4, 2, optimization, iBatch))
      ReturnFalse;
   for(int i = 0; i < int(caMamba.Size()); i++)
     {
      if(!caMamba[i].Init(0, 16 + i, OpenCL, window_key, 2 * window_key, window + units_count, optimization, iBatch))
         ReturnFalse;
     }
   if(!cLaneAware.Init(0, 19, OpenCL, window_key, window_key, heads, window_key, heads / 2, window, window + units_count, 2, 1, optimization, iBatch))
      ReturnFalse;
//---
   if(!caForecastMLP[0].Init(0, 20, OpenCL, window_key, window_key, 4 * forecast, window, optimization, iBatch))
      ReturnFalse;
   caForecastMLP[0].SetActivationFunction(LReLU);
   if(!caForecastMLP[1].Init(0, 21, OpenCL, 4 * forecast, 4 * forecast, forecast, window, optimization, iBatch))
      ReturnFalse;
   caForecastMLP[1].SetActivationFunction(TANH);
   if(!cTransposeOut.Init(0, 22, OpenCL, window, forecast, optimization, iBatch))
      ReturnFalse;
//---
   SetOutput(cTransposeOut.getOutput(), true);
   SetGradient(cTransposeOut.getGradient(), true);
   SetActivationFunction((ENUM_ACTIVATION)caForecastMLP[1].Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- State Encoder
   if(!cStateRNN.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cStateMLP[0].FeedForward(cStateRNN.AsObject()))
      ReturnFalse;
   if(!cStateMLP[1].FeedForward(cStateMLP[0].AsObject()))
      ReturnFalse;
//--- Variables Encoder
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cVariablesRNN.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cVariablesMLP[0].FeedForward(cVariablesRNN.AsObject()))
      ReturnFalse;
   if(!cVariablesMLP[1].FeedForward(cVariablesMLP[0].AsObject()))
      ReturnFalse;
//--- Position Encoder
   if(!cStatePE.FeedForward(cStateMLP[1].AsObject()))
      ReturnFalse;
   if(!cVariablesPE.FeedForward(cVariablesMLP[1].AsObject()))
      ReturnFalse;
//--- Context
   if(!cStateToState.FeedForward(cStatePE.AsObject()))
      ReturnFalse;
   if(!cStateToVariable.FeedForward(cStateToState.AsObject(), cVariablesPE.getOutput()))
      ReturnFalse;
   if(!cVariableToState.FeedForward(cVariablesPE.AsObject(), cStateToVariable.getOutput()))
      ReturnFalse;
   if(!Concat(cStateToVariable.getOutput(), cVariableToState.getOutput(), cContext.getOutput(), cStateToVariable.Neurons(), cVariableToState.Neurons(), 1))
      ReturnFalse;
   if(!cContextMLP[0].FeedForward(cContext.AsObject()))
      ReturnFalse;
   if(!cContextMLP[1].FeedForward(cContextMLP[0].AsObject()))
      ReturnFalse;
//--- Lane aware
   if(!cHighLevelInteraction.FeedForward(cContextMLP[1].AsObject()))
      ReturnFalse;
   if(!caMamba[0].FeedForward(cHighLevelInteraction.AsObject()))
      ReturnFalse;
   for(int i = 1; i < int(caMamba.Size()); i++)
     {
      if(!caMamba[i].FeedForward(caMamba[i - 1].AsObject()))
         ReturnFalse;
     }
   if(!cLaneAware.FeedForward(cVariablesPE.AsObject(), caMamba[caMamba.Size() - 1].getOutput()))
      ReturnFalse;
//--- Forecast
   if(!caForecastMLP[0].FeedForward(cLaneAware.AsObject()))
      ReturnFalse;
   if(!caForecastMLP[1].FeedForward(caForecastMLP[0].AsObject()))
      ReturnFalse;
   if(!cTransposeOut.FeedForward(caForecastMLP[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Forecast
   if(!caForecastMLP[1].CalcHiddenGradients(cTransposeOut.AsObject()))
      ReturnFalse;
   if(!caForecastMLP[0].CalcHiddenGradients(caForecastMLP[1].AsObject()))
      ReturnFalse;
//--- Lane aware
   if(!cLaneAware.CalcHiddenGradients(caForecastMLP[0].AsObject()))
      ReturnFalse;
   if(!cVariablesPE.CalcHiddenGradients(cLaneAware.AsObject(), caMamba[caMamba.Size() - 1].getOutput(), caMamba[caMamba.Size() - 1].getGradient(), (ENUM_ACTIVATION)caMamba[caMamba.Size() - 1].Activation()))
      ReturnFalse;
//---
   for(int i = int(caMamba.Size()) - 2; i >= 0; i--)
      if(!caMamba[i].CalcHiddenGradients(caMamba[i + 1].AsObject()))
         ReturnFalse;
   if(!cHighLevelInteraction.CalcHiddenGradients(caMamba[0].AsObject()))
      ReturnFalse;
//--- Context
   if(!cContextMLP[1].CalcHiddenGradients(cHighLevelInteraction.AsObject()))
      ReturnFalse;
   if(!cContextMLP[0].CalcHiddenGradients(cContextMLP[1].AsObject()))
      ReturnFalse;
   if(!cContext.CalcHiddenGradients(cContextMLP[0].AsObject()))
      ReturnFalse;
   if(!DeConcat(cStatePE.getGradient(), cVariableToState.getGradient(), cContext.getGradient(), cStateToVariable.Neurons(), cVariableToState.Neurons(), 1))
      ReturnFalse;
   if(!SumAndNormalize(cVariablesPE.getGradient(), cVariablesPE.getGradient(), cVariablesMLP[1].getGradient(), 1, false))
      ReturnFalse;
   if(!cVariablesPE.CalcHiddenGradients(cVariableToState.AsObject(), cStateToVariable.getOutput(), cStateToVariable.getGradient(), (ENUM_ACTIVATION)cStateToVariable.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(cVariablesPE.getGradient(), cVariablesMLP[1].getGradient(), cVariablesMLP[1].getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cStateToVariable.getGradient(), cStatePE.getGradient(), cStateToVariable.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cStateToState.CalcHiddenGradients(cStateToVariable.AsObject(), cVariablesPE.getOutput(), cVariablesPE.getGradient(), (ENUM_ACTIVATION)cVariablesPE.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(cVariablesPE.getGradient(), cVariablesMLP[1].getGradient(), cVariablesPE.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cStatePE.CalcHiddenGradients(cStateToState.AsObject()))
      ReturnFalse;
//--- Position Encoder
   if(!cStateMLP[1].CalcHiddenGradients(cStatePE.AsObject()))
      ReturnFalse;
   if(!cVariablesMLP[1].CalcHiddenGradients(cVariablesPE.AsObject()))
      ReturnFalse;
//--- Variables Encoder
   if(!cVariablesMLP[0].CalcHiddenGradients(cVariablesMLP[1].AsObject()))
      ReturnFalse;
   if(!cVariablesRNN.CalcHiddenGradients(cVariablesMLP[0].AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cVariablesRNN.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getGradient(), cTranspose.getGradient(), 1, false))
      ReturnFalse;
//--- State Encoder
   if(!cStateMLP[0].CalcHiddenGradients(cStateMLP[1].AsObject()))
      ReturnFalse;
   if(!cStateRNN.CalcHiddenGradients(cStateMLP[0].AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cStateRNN.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cTranspose.getGradient(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- State Encoder
   if(!cStateRNN.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cStateMLP[0].UpdateInputWeights(cStateRNN.AsObject()))
      ReturnFalse;
   if(!cStateMLP[1].UpdateInputWeights(cStateMLP[0].AsObject()))
      ReturnFalse;
//--- Variables Encoder
   if(!cVariablesRNN.UpdateInputWeights(cTranspose.AsObject()))
      ReturnFalse;
   if(!cVariablesMLP[0].UpdateInputWeights(cVariablesRNN.AsObject()))
      ReturnFalse;
   if(!cVariablesMLP[1].UpdateInputWeights(cVariablesMLP[0].AsObject()))
      ReturnFalse;
//--- Position Encoder
   if(!cStatePE.UpdateInputWeights(cStateMLP[1].AsObject()))
      ReturnFalse;
   if(!cVariablesPE.UpdateInputWeights(cVariablesMLP[1].AsObject()))
      ReturnFalse;
//--- Context
   if(!cStateToState.UpdateInputWeights(cStatePE.AsObject()))
      ReturnFalse;
   if(!cStateToVariable.UpdateInputWeights(cStateToState.AsObject(), cVariablesPE.getOutput()))
      ReturnFalse;
   if(!cVariableToState.UpdateInputWeights(cVariablesPE.AsObject(), cStateToVariable.getOutput()))
      ReturnFalse;
   if(!cContextMLP[0].UpdateInputWeights(cContext.AsObject()))
      ReturnFalse;
   if(!cContextMLP[1].UpdateInputWeights(cContextMLP[0].AsObject()))
      ReturnFalse;
//--- Lane aware
   if(!cHighLevelInteraction.UpdateInputWeights(cContextMLP[1].AsObject()))
      ReturnFalse;
   if(!caMamba[0].UpdateInputWeights(cHighLevelInteraction.AsObject()))
      ReturnFalse;
   for(int i = 1; i < int(caMamba.Size()); i++)
     {
      if(!caMamba[i].UpdateInputWeights(caMamba[i - 1].AsObject()))
         ReturnFalse;
     }
   if(!cLaneAware.UpdateInputWeights(cVariablesPE.AsObject(), caMamba[caMamba.Size() - 1].getOutput()))
      ReturnFalse;
//--- Forecast
   if(!caForecastMLP[0].UpdateInputWeights(cLaneAware.AsObject()))
      ReturnFalse;
   if(!caForecastMLP[1].UpdateInputWeights(caForecastMLP[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//--- State Encoder
   if(!cStateRNN.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cStateMLP[i].Save(file_handle))
         ReturnFalse;
//--- Variables Encoder
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cVariablesRNN.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cVariablesMLP[i].Save(file_handle))
         ReturnFalse;
//--- Context Encoder
   if(!cStatePE.Save(file_handle))
      ReturnFalse;
   if(!cVariablesPE.Save(file_handle))
      ReturnFalse;
   if(!cStateToState.Save(file_handle))
      ReturnFalse;
   if(!cVariableToState.Save(file_handle))
      ReturnFalse;
   if(!cStateToVariable.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cContextMLP[i].Save(file_handle))
         ReturnFalse;
//---
   if(!cHighLevelInteraction.Save(file_handle))
      ReturnFalse;
   int total = int(caMamba.Size());
   if(FileWriteInteger(file_handle, total) < INT_VALUE)
      ReturnFalse;
   for(int i = 0; i < total; i++)
      if(!caMamba[i].Save(file_handle))
         ReturnFalse;
   if(!cLaneAware.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!caForecastMLP[i].Save(file_handle))
         ReturnFalse;
   if(!cTransposeOut.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTrajLLMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//--- State Encoder
   if(!LoadInsideLayer(file_handle, cStateRNN.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cStateMLP[i].AsObject()))
         ReturnFalse;
//--- Variables Encoder
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cVariablesRNN.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cVariablesMLP[i].AsObject()))
         ReturnFalse;
//--- Context Encoder
   if(!LoadInsideLayer(file_handle, cStatePE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cVariablesPE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStateToState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cVariableToState.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStateToVariable.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cContext.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cContextMLP[i].AsObject()))
         ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cHighLevelInteraction.AsObject()))
      ReturnFalse;
   int total = FileReadInteger(file_handle);
   if(total != int(caMamba.Size()))
     {
      ReturnFalse;
     }
   for(int i = 0; i < total; i++)
      if(!LoadInsideLayer(file_handle, caMamba[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLaneAware.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, caForecastMLP[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTrajLLMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//--- State Encoder
   cStateRNN.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      cStateMLP[i].SetOpenCL(OpenCL);
//--- Variables Encoder
   cTranspose.SetOpenCL(OpenCL);
   cVariablesRNN.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      cVariablesMLP[i].SetOpenCL(OpenCL);
//--- Context Encoder
   cStatePE.SetOpenCL(OpenCL);
   cVariablesPE.SetOpenCL(OpenCL);
   cStateToState.SetOpenCL(OpenCL);
   cVariableToState.SetOpenCL(OpenCL);
   cStateToVariable.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      cContextMLP[i].SetOpenCL(OpenCL);
//---
   cHighLevelInteraction.SetOpenCL(OpenCL);
   int total = int(caMamba.Size());
   for(int i = 0; i < total; i++)
      caMamba[i].SetOpenCL(OpenCL);
   cLaneAware.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      caForecastMLP[i].SetOpenCL(OpenCL);
   cTransposeOut.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, float dropout, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * (units_count + forecast), optimization_type, batch))
      ReturnFalse;
//---
   iVariables = window;
   fDropout = MathMax(MathMin(dropout, 1), 0);
//---
   if(!cHistoryMask.BufferInit(iVariables * units_count, 1) ||
      !cHistoryMask.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cFutureMask.BufferInit(iVariables * forecast, 1) ||
      !cFutureMask.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cData.Init(0, 0, OpenCL, 3 * iVariables * (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   if(!cPE.Init(0, 1, OpenCL, cData.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cEncoder.Init(0, 2, OpenCL, 3, window_key, heads, (heads + 1) / 2, iVariables, 1, 1, (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   if(!cDForw.Init(0, 3, OpenCL, iVariables * (units_count + forecast), optimization, iBatch))
      ReturnFalse;;
   if(!cDBakw.Init(0, 4, OpenCL, iVariables * (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   if(!cProjDForw.Init(0, 5, OpenCL, 1, 1, 3, iVariables, (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   cProjDForw.SetActivationFunction(SIGMOID);
   if(!cProjDBakw.Init(0, 6, OpenCL, 1, 1, 3, iVariables, (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   cProjDBakw.SetActivationFunction(SIGMOID);
   if(!cDataDForw.Init(0, 7, OpenCL, cData.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cDataDBakw.Init(0, 8, OpenCL, cData.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cConcatDataDForwBakw.Init(0, 9, OpenCL, 2 * cData.Neurons(), optimization, iBatch))
      ReturnFalse;
   for(uint i = 0; i < cSSM.Size(); i++)
     {
      if(!cSSM[i].Init(0, 10 + i, OpenCL, 6 * iVariables, 12 * iVariables, (units_count + forecast), optimization, iBatch))
         ReturnFalse;
     }
   uint id = 10 + cSSM.Size();
   if(!cStat.Init(0, id, OpenCL, 6, 6, 12, iVariables * (units_count + forecast), optimization, iBatch))
      ReturnFalse;
   id++;
   if(!cTranspStat.Init(0, id, OpenCL, iVariables * (units_count + forecast), 12, optimization, iBatch))
      ReturnFalse;
   id++;
   if(!cVAE.Init(0, id, OpenCL, cTranspStat.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   id++;
   if(!cTranspVAE.Init(0, id, OpenCL, cVAE.Neurons() / iVariables, iVariables, optimization, iBatch))
      ReturnFalse;
   id++;
   uint w = cTranspVAE.Neurons() / iVariables;
   if(!cDecoder[0].Init(0, id, OpenCL, w, w, 2 * (units_count + forecast), iVariables, optimization, iBatch))
      ReturnFalse;
   cDecoder[0].SetActivationFunction(LReLU);
   id++;
   if(!cDecoder[1].Init(0, id, OpenCL, 2 * (units_count + forecast), 2 * (units_count + forecast), (units_count + forecast), iVariables, optimization, iBatch))
      ReturnFalse;
   cDecoder[1].SetActivationFunction(TANH);
   id++;
   if(!cTranspResult.Init(0, id, OpenCL, iVariables, (units_count + forecast), optimization, iBatch))
      ReturnFalse;
//---
   if(!SetOutput(cTranspResult.getOutput(), true) ||
      !SetGradient(cTranspResult.getGradient(), true))
      ReturnFalse;
   SetActivationFunction((ENUM_ACTIVATION)cDecoder[1].Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::Prepare(const CBufferFloat* history, const CBufferFloat* future)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {MathMax(cHistoryMask.Total(), cFutureMask.Total()) / iVariables, iVariables};
   ResetLastError();
   setBuffer(def_k_UniTrajPrepare, def_k_utp_h_mask, cHistoryMask.GetIndex())
   setBuffer(def_k_UniTrajPrepare, def_k_utp_history, history.GetIndex())
   setBuffer(def_k_UniTrajPrepare, def_k_utp_f_mask, cFutureMask.GetIndex())
   setBuffer(def_k_UniTrajPrepare, def_k_utp_future, (!!future ? future.GetIndex() : cFutureMask.GetIndex()))
   setBuffer(def_k_UniTrajPrepare, def_k_utp_output, cData.getOutputIndex())
   setArgument(def_k_UniTrajPrepare, def_k_utp_h_total, cHistoryMask.Total())
   setArgument(def_k_UniTrajPrepare, def_k_utp_f_total, cFutureMask.Total())
   kernelExecute(def_k_UniTrajPrepare, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cData.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::PrepareGrad(CBufferFloat* history_gr, CBufferFloat* future_gr)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {MathMax(cHistoryMask.Total(), cFutureMask.Total()) / iVariables, iVariables};
   ResetLastError();
   setBuffer(def_k_UniTrajPrepareGrad, def_k_utpg_history_gr, history_gr.GetIndex())
   setBuffer(def_k_UniTrajPrepareGrad, def_k_utpg_future_gr, (!!future_gr ? future_gr.GetIndex() : cFutureMask.GetIndex()))
   setBuffer(def_k_UniTrajPrepareGrad, def_k_utpg_output, cData.getOutputIndex())
   setBuffer(def_k_UniTrajPrepareGrad, def_k_utpg_output_gr, cData.getGradientIndex())
   setArgument(def_k_UniTrajPrepareGrad, def_k_utpg_h_total, cHistoryMask.Total())
   setArgument(def_k_UniTrajPrepareGrad, def_k_utpg_f_total, cFutureMask.Total())
   kernelExecute(def_k_UniTrajPrepareGrad, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!history_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::BTS(void)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {2, iVariables};
   ResetLastError();
   setBuffer(def_k_UniTrajBTS, def_k_utbts_concat_inp, cData.getOutputIndex())
   setBuffer(def_k_UniTrajBTS, def_k_utbts_d_forw, cDForw.getOutputIndex())
   setBuffer(def_k_UniTrajBTS, def_k_utbts_d_bakw, cDBakw.getOutputIndex())
   setArgument(def_k_UniTrajBTS, def_k_utbts_total, int(cDForw.Neurons() / iVariables))
   kernelExecute(def_k_UniTrajBTS, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cDBakw.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Create History Mask
   int total = cHistoryMask.Total();
   if(!cHistoryMask.BufferInit(total, 1))
      ReturnFalse;
   if(bTrain)
     {
      for(int i = 0; i < int(total * fDropout); i++)
         cHistoryMask.Update(RND(total), 0);
     }
   if(!cHistoryMask.BufferWrite())
      ReturnFalse;
//--- Create Future Mask
   total = cFutureMask.Total();
   if(!cFutureMask.BufferInit(total, (!SecondInput ? 0 : 1)))
      ReturnFalse;
   if(bTrain && !!SecondInput)
     {
      for(int i = 0; i < int(total * fDropout); i++)
         cFutureMask.Update(RND(total), 0);
     }
   if(!cFutureMask.BufferWrite())
      ReturnFalse;
//--- Prepare Data
   if(!Prepare(NeuronOCL.getOutput(), SecondInput))
      ReturnFalse;
//--- Encoder
   if(!cPE.FeedForward(cData.AsObject()))
      ReturnFalse;
   if(!cEncoder.FeedForward(cPE.AsObject()))
      ReturnFalse;
//--- BTS
   if(!BTS())
      ReturnFalse;
   if(!cProjDForw.FeedForward(cDForw.AsObject()))
      ReturnFalse;
   if(!cProjDBakw.FeedForward(cDBakw.AsObject()))
      ReturnFalse;
   if(!ElementMult(cEncoder.getOutput(), cProjDForw.getOutput(), cDataDForw.getOutput()))
      ReturnFalse;
   if(!ElementMult(cEncoder.getOutput(), cProjDBakw.getOutput(), cDataDBakw.getOutput()))
      ReturnFalse;
   if(!Concat(cDataDForw.getOutput(), cDataDBakw.getOutput(), cConcatDataDForwBakw.getOutput(),
              3, 3, cData.Neurons() / 3))
      ReturnFalse;
//--- SSM
   if(!cSSM[0].FeedForward(cConcatDataDForwBakw.AsObject()))
      ReturnFalse;
   for(uint i = 1; i < cSSM.Size(); i++)
      if(!cSSM[i].FeedForward(cSSM[i - 1].AsObject()))
         ReturnFalse;
//--- VAE
   if(!cStat.FeedForward(cSSM[cSSM.Size() - 1].AsObject()))
      ReturnFalse;
   if(!cTranspStat.FeedForward(cStat.AsObject()))
      ReturnFalse;
   if(!cVAE.FeedForward(cTranspStat.AsObject()))
      ReturnFalse;
//--- Decoder
   if(!cTranspVAE.FeedForward(cVAE.AsObject()))
      ReturnFalse;
   if(!cDecoder[0].FeedForward(cTranspVAE.AsObject()))
      ReturnFalse;
   if(!cDecoder[1].FeedForward(cDecoder[0].AsObject()))
      ReturnFalse;
   if(!cTranspResult.FeedForward(cDecoder[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::calcInputGradients(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput, CBufferFloat* SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
//--- Decoder
   if(!cDecoder[1].CalcHiddenGradients(cTranspResult.AsObject()))
      ReturnFalse;
   if(!cDecoder[0].CalcHiddenGradients(cDecoder[1].AsObject()))
      ReturnFalse;
   if(!cTranspVAE.CalcHiddenGradients(cDecoder[0].AsObject()))
      ReturnFalse;
   if(!cVAE.CalcHiddenGradients(cTranspVAE.AsObject()))
      ReturnFalse;
//--- VAE
   if(!cTranspStat.CalcHiddenGradients(cVAE.AsObject()))
      ReturnFalse;
   if(!cStat.CalcHiddenGradients(cTranspStat.AsObject()))
      ReturnFalse;
   if(!cSSM[cSSM.Size() - 1].CalcHiddenGradients(cStat.AsObject()))
      ReturnFalse;
//--- SSM
   for(int i = (int)cSSM.Size() - 2; i >= 0; i--)
      if(!cSSM[i].CalcHiddenGradients(cSSM[i + 1].AsObject()))
         ReturnFalse;
   if(!cConcatDataDForwBakw.CalcHiddenGradients(cSSM[0].AsObject()))
      ReturnFalse;
//--- BTS
   if(!DeConcat(cDataDForw.getGradient(), cDataDBakw.getGradient(), cConcatDataDForwBakw.getGradient(),
                3, 3, cData.Neurons() / 3))
      ReturnFalse;
   if(!ElementMultGrad(cEncoder.getOutput(), cEncoder.getGradient(),
                       cProjDForw.getOutput(), cProjDForw.getGradient(),
                       cDataDForw.getGradient(), cEncoder.Activation(), cProjDForw.Activation()))
      ReturnFalse;
   if(!ElementMultGrad(cEncoder.getOutput(), cPE.getGradient(),
                       cProjDBakw.getOutput(), cProjDBakw.getGradient(),
                       cDataDBakw.getGradient(), cEncoder.Activation(), cProjDBakw.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(cEncoder.getGradient(), cPE.getGradient(), cEncoder.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cDForw.CalcHiddenGradients(cProjDForw.AsObject()))
      ReturnFalse;
   if(!cDBakw.CalcHiddenGradients(cProjDBakw.AsObject()))
      ReturnFalse;
//--- Encoder
   if(!cPE.CalcHiddenGradients(cEncoder.AsObject()))
      ReturnFalse;
   if(!cData.CalcHiddenGradients(cPE.AsObject()))
      ReturnFalse;
//--- Prepare Data
   if(!PrepareGrad(NeuronOCL.getGradient(), SecondGradient))
      ReturnFalse;
//--- Deactivation Gradient
   if(NeuronOCL.Activation() != None)
     {
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
     }
   if(SecondActivation != None && !!SecondInput && !!SecondGradient)
     {
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::updateInputWeights(CNeuronBaseOCL* NeuronOCL, CBufferFloat* second)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Encoder
   if(!cPE.UpdateInputWeights(cData.AsObject()))
      ReturnFalse;
   if(!cEncoder.UpdateInputWeights(cPE.AsObject()))
      ReturnFalse;
//--- BTS
   if(!cProjDForw.UpdateInputWeights(cDForw.AsObject()))
      ReturnFalse;
   if(!cProjDBakw.UpdateInputWeights(cDBakw.AsObject()))
      ReturnFalse;
//--- SSM
   if(!cSSM[0].UpdateInputWeights(cConcatDataDForwBakw.AsObject()))
      ReturnFalse;
   for(uint i = 1; i < cSSM.Size(); i++)
      if(!cSSM[i].UpdateInputWeights(cSSM[i - 1].AsObject()))
         ReturnFalse;
//--- VAE
   if(!cStat.UpdateInputWeights(cSSM[cSSM.Size() - 1].AsObject()))
      ReturnFalse;
//--- Decoder
   if(!cDecoder[0].UpdateInputWeights(cTranspVAE.AsObject()))
      ReturnFalse;
   if(!cDecoder[1].UpdateInputWeights(cDecoder[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iVariables) < INT_VALUE)
      ReturnFalse;
   if(FileWriteFloat(file_handle, fDropout) < sizeof(float))
      ReturnFalse;
//---
   if(!cHistoryMask.Save(file_handle))
      ReturnFalse;
   if(!cFutureMask.Save(file_handle))
      ReturnFalse;
   if(!cData.Save(file_handle))
      ReturnFalse;
   if(!cPE.Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cDForw.Save(file_handle))
      ReturnFalse;
   if(!cDBakw.Save(file_handle))
      ReturnFalse;
   if(!cProjDForw.Save(file_handle))
      ReturnFalse;
   if(!cProjDBakw.Save(file_handle))
      ReturnFalse;
   if(!cDataDForw.Save(file_handle))
      ReturnFalse;
   if(!cDataDBakw.Save(file_handle))
      ReturnFalse;
   if(!cConcatDataDForwBakw.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cSSM.Size(); i++)
      if(!cSSM[i].Save(file_handle))
         ReturnFalse;
   if(!cStat.Save(file_handle))
      ReturnFalse;
   if(!cTranspStat.Save(file_handle))
      ReturnFalse;
   if(!cVAE.Save(file_handle))
      ReturnFalse;
   if(!cTranspVAE.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cDecoder[i].Save(file_handle))
         ReturnFalse;
   if(!cTranspResult.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniTraj::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   fDropout = FileReadFloat(file_handle);
//---
   if(!cHistoryMask.Load(file_handle))
      ReturnFalse;
   if(!cFutureMask.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cData.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDForw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDBakw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjDForw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjDBakw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDataDForw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDataDBakw.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatDataDForwBakw.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cSSM.Size(); i++)
      if(!LoadInsideLayer(file_handle, cSSM[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStat.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspStat.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cVAE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspVAE.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cDecoder[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspResult.AsObject()))
      ReturnFalse;
//---
   if(!SetOutput(cTranspResult.getOutput(), true) ||
      !SetGradient(cTranspResult.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniTraj::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cHistoryMask.BufferCreate(OpenCL);
   cFutureMask.BufferCreate(OpenCL);
   cData.SetOpenCL(OpenCL);
   cPE.SetOpenCL(OpenCL);
   cEncoder.SetOpenCL(OpenCL);
   cDForw.SetOpenCL(OpenCL);
   cDBakw.SetOpenCL(OpenCL);
   cProjDForw.SetOpenCL(OpenCL);
   cProjDBakw.SetOpenCL(OpenCL);
   cDataDForw.SetOpenCL(OpenCL);
   cDataDBakw.SetOpenCL(OpenCL);
   cConcatDataDForwBakw.SetOpenCL(OpenCL);
   for(uint i = 0; i < cSSM.Size(); i++)
      cSSM[i].SetOpenCL(OpenCL);
   cStat.SetOpenCL(OpenCL);
   cTranspStat.SetOpenCL(OpenCL);
   cVAE.SetOpenCL(OpenCL);
   cTranspVAE.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      cDecoder[i].SetOpenCL(OpenCL);
   cTranspResult.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTAAEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = fmax(window, 1);
   iWindowKey = fmax(window_key, 1);
   iUnits = fmax(units_count, 1);
   iHeads = fmax(heads, 1);
   iLayers = fmax(layers, 1);
   iHeadsKV = fmax(heads_kv, 1);
   iLayersToOneKV = fmax(layers_to_one_kv, 1);
   iVariables = variables;
//---
   uint num_q = iWindowKey * iHeads * iUnits * iVariables;              //Size of Q tensor
   uint num_kv = iWindowKey * iHeadsKV * iUnits * iVariables;           //Size of KV tensor
   uint q_weights = (iWindow * iHeads + 1) * iWindowKey;                //Size of weights' matrix of Q tenzor
   uint kv_weights = (iWindow * iHeadsKV + 1) * iWindowKey;             //Size of weights' matrix of Q tenzor
   uint scores = iUnits * iUnits * iHeads * iVariables;                 //Size of Score tensor
   uint mh_out = iWindowKey * iHeads * iUnits * iVariables;             //Size of multi-heads self-attention
   uint out = iWindow * iUnits * iVariables;                            //Size of out tensore
   uint w0 = (iWindowKey * iHeads + 1) * iWindow;                       //Size W0 tensor
   uint gate = (2 * iWindow + 1) * iWindow;                             //Size of weights' matrix 1-st feed forward layer
   uint self = (iWindow + 1) * iWindow;                             //Size of weights' matrix 2-nd feed forward layer
//---
   for(uint i = 0; i < iLayers; i++)
     {
      CBufferFloat *temp = NULL;
      for(int d = 0; d < 2; d++)
        {
         //--- Initilize Q tensor
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(num_q, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!QKV_Tensors.Add(temp))
            ReturnFalse;
         //--- Initilize KV tensor
         if(i % iLayersToOneKV == 0)
           {
            temp = new CBufferFloat();
            if(CheckPointer(temp) == POINTER_INVALID)
               ReturnFalse;
            if(!temp.BufferInit(num_kv, 0))
               ReturnFalse;
            if(!temp.BufferCreate(OpenCL))
               ReturnFalse;
            if(!K_Tensors.Add(temp))
               ReturnFalse;
            temp = new CBufferFloat();
            if(CheckPointer(temp) == POINTER_INVALID)
               ReturnFalse;
            if(!temp.BufferInit(num_kv, 0))
               ReturnFalse;
            if(!temp.BufferCreate(OpenCL))
               ReturnFalse;
            if(!V_Tensors.Add(temp))
               ReturnFalse;
            temp = new CBufferFloat();
            if(CheckPointer(temp) == POINTER_INVALID)
               ReturnFalse;
            if(!temp.BufferInit(2 * num_kv, 0))
               ReturnFalse;
            if(!temp.BufferCreate(OpenCL))
               ReturnFalse;
            if(!KV_Tensors.Add(temp))
               ReturnFalse;
           }
         //--- Initialize scores
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(scores, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!S_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize multi-heads attention out
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(mh_out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!AO_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize attention out
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize Concatenate
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(2 * out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize Gate
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize Self
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Tensors.Add(temp))
            ReturnFalse;
         //--- Initialize Out
         if(i == iLayers - 1)
           {
            if(!FF_Tensors.Add(d == 0 ? Output : Gradient))
               ReturnFalse;
            continue;
           }
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit(out, 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Tensors.Add(temp))
            ReturnFalse;
        }
      //--- Initilize Q weights
      temp = new CBufferFloat();
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      if(!temp.Reserve(q_weights))
         ReturnFalse;
      float k = (float)(1 / sqrt(iWindow + 1));
      for(uint w = 0; w < q_weights; w++)
        {
         if(!temp.Add(GenerateWeight() * 2 * k - k))
            ReturnFalse;
        }
      if(!temp.BufferCreate(OpenCL))
         ReturnFalse;
      if(!QKV_Weights.Add(temp))
         ReturnFalse;
      //--- Initilize K weights
      if(i % iLayersToOneKV == 0)
        {
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.Reserve(kv_weights))
            ReturnFalse;
         for(uint w = 0; w < kv_weights; w++)
           {
            if(!temp.Add(GenerateWeight() * 2 * k - k))
               ReturnFalse;
           }
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!K_Weights.Add(temp))
            ReturnFalse;
         //---
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.Reserve(kv_weights))
            ReturnFalse;
         for(uint w = 0; w < kv_weights; w++)
           {
            if(!temp.Add(GenerateWeight() * 2 * k - k))
               ReturnFalse;
           }
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!V_Weights.Add(temp))
            ReturnFalse;
        }
      //--- Initilize Weights0
      temp = new CBufferFloat();
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      if(!temp.Reserve(w0))
         ReturnFalse;
      for(uint w = 0; w < w0; w++)
        {
         if(!temp.Add(GenerateWeight() * 2 * k - k))
            ReturnFalse;
        }
      if(!temp.BufferCreate(OpenCL))
         ReturnFalse;
      if(!FF_Weights.Add(temp))
         ReturnFalse;
      //--- Initilize Gate Weights
      temp = new CBufferFloat();
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      if(!temp.Reserve(gate))
         ReturnFalse;
      k = (float)(1 / sqrt(2 * iWindow + 1));
      for(uint w = 0; w < gate; w++)
        {
         if(!temp.Add(GenerateWeight() * 2 * k - k))
            ReturnFalse;
        }
      if(!temp.BufferCreate(OpenCL))
         ReturnFalse;
      if(!FF_Weights.Add(temp))
         ReturnFalse;
      //--- Self
      temp = new CBufferFloat();
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      if(!temp.Reserve(self))
         ReturnFalse;
      k = (float)(1 / sqrt(iWindow + 1));
      for(uint w = 0; w < self; w++)
        {
         if(!temp.Add(GenerateWeight() * 2 * k - k))
            ReturnFalse;
        }
      if(!temp.BufferCreate(OpenCL))
         ReturnFalse;
      if(!FF_Weights.Add(temp))
         ReturnFalse;
      //---
      for(int d = 0; d < (optimization == SGD ? 1 : 2); d++)
        {
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit((d == 0 || optimization == ADAM ? q_weights : iWindowKey * iHeads), 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!QKV_Weights.Add(temp))
            ReturnFalse;
         if(i % iLayersToOneKV == 0)
           {
            temp = new CBufferFloat();
            if(CheckPointer(temp) == POINTER_INVALID)
               ReturnFalse;
            if(!temp.BufferInit((d == 0 || optimization == ADAM ? kv_weights : iWindowKey * iHeadsKV), 0))
               ReturnFalse;
            if(!temp.BufferCreate(OpenCL))
               ReturnFalse;
            if(!K_Weights.Add(temp))
               ReturnFalse;
            //---
            temp = new CBufferFloat();
            if(CheckPointer(temp) == POINTER_INVALID)
               ReturnFalse;
            if(!temp.BufferInit((d == 0 || optimization == ADAM ? kv_weights : iWindowKey * iHeadsKV), 0))
               ReturnFalse;
            if(!temp.BufferCreate(OpenCL))
               ReturnFalse;
            if(!V_Weights.Add(temp))
               ReturnFalse;
           }
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit((d == 0 || optimization == ADAM ? w0 : iWindow), 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Weights.Add(temp))
            ReturnFalse;
         //--- Initilize Gate Momentum
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit((d == 0 || optimization == ADAM ? gate : 2 * iWindow), 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Weights.Add(temp))
            ReturnFalse;
         //--- Initilize Self Momentum
         temp = new CBufferFloat();
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!temp.BufferInit((d == 0 || optimization == ADAM ? self : iWindow), 0))
            ReturnFalse;
         if(!temp.BufferCreate(OpenCL))
            ReturnFalse;
         if(!FF_Weights.Add(temp))
            ReturnFalse;
        }
     }
//---
   if(!Temp.BufferInit(MathMax(2 * num_kv, out), 0))
      ReturnFalse;
   if(!Temp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTAAEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   CBufferFloat *kv = NULL;
   for(uint i = 0; (i < iLayers && !IsStopped()); i++)
     {
      //--- Calculate Queries, Keys, Values
      CBufferFloat *inputs = (i == 0 ? NeuronOCL.getOutput() : FF_Tensors.At(10 * i - 6));
      CBufferFloat *q = QKV_Tensors.At(i * 2);
      if(IsStopped() || !ConvolutionForward(QKV_Weights.At(i * (optimization == SGD ? 2 : 3)), inputs, q, iWindow, iWindowKey * iHeads, None))
         ReturnFalse;
      if((i % iLayersToOneKV) == 0)
        {
         uint i_kv = i / iLayersToOneKV;
         kv = KV_Tensors.At(i_kv * 2);
         CBufferFloat *k = K_Tensors.At(i_kv * 2);
         CBufferFloat *v = V_Tensors.At(i_kv * 2);
         if(IsStopped() || !ConvolutionForward(K_Weights.At(i_kv * (optimization == SGD ? 2 : 3)), inputs, k, iWindow, iWindowKey * iHeadsKV, None))
            ReturnFalse;
         if(IsStopped() || !ConvolutionForward(V_Weights.At(i_kv * (optimization == SGD ? 2 : 3)), inputs, v, iWindow, iWindowKey * iHeadsKV, None))
            ReturnFalse;
         if(IsStopped() || !Concat(k, v, kv, iWindowKey * iHeadsKV * iVariables, iWindowKey * iHeadsKV * iVariables, iUnits))
            ReturnFalse;
        }
      //--- Score calculation and Multi-heads attention calculation
      CBufferFloat *temp = S_Tensors.At(i * 2);
      CBufferFloat *out = AO_Tensors.At(i * 2);
      if(IsStopped() || !AttentionOut(q, kv, temp, out))
         ReturnFalse;
      //--- Attention out calculation
      temp = FF_Tensors.At(i * 10);
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9)), out, temp, iWindowKey * iHeads, iWindow, None))
         ReturnFalse;
      //--- Concat
      out = FF_Tensors.At(i * 10 + 1);
      if(IsStopped() || !Concat(temp, inputs, out, iWindow, iWindow, iUnits))
         ReturnFalse;
      //--- Gate
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 1), out, FF_Tensors.At(i * 10 + 2), 2 * iWindow, iWindow, SIGMOID))
         ReturnFalse;
      //--- Self
      if(IsStopped() || !ConvolutionForward(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 2), inputs, FF_Tensors.At(i * 10 + 3), iWindow, iWindow, None))
         ReturnFalse;
      //--- Out
      if(IsStopped() || !GateElementMult(FF_Tensors.At(i * 10 + 3), temp, FF_Tensors.At(i * 10 + 2), FF_Tensors.At(i * 10 + 4)))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTAAEncoder::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   CBufferFloat *out_grad = Gradient;
   CBufferFloat *kv_g = KV_Tensors.At(KV_Tensors.Total() - 1);
//---
   for(int i = int(iLayers - 1); (i >= 0 && !IsStopped()); i--)
     {
      if(i == int(iLayers - 1) || (i + 1) % iLayersToOneKV == 0)
         kv_g = KV_Tensors.At((i / iLayersToOneKV) * 2 + 1);
      //--- Gate out
      if(IsStopped() || !GateElementMultGrad(FF_Tensors.At(i * 10 + 3), FF_Tensors.At(i * 10 + 8), FF_Tensors.At(i * 10), FF_Tensors.At(i * 10 + 5), FF_Tensors.At(i * 10 + 2), FF_Tensors.At(i * 10 + 7), out_grad, None, None, SIGMOID))
         ReturnFalse;
      //--- Passing gradient through Self
      CBufferFloat *inp = NULL;
      CBufferFloat *inp_gr = NULL;
      if(i == 0)
        {
         inp = prevLayer.getOutput();
         inp_gr = prevLayer.getGradient();
        }
      else
        {
         inp = FF_Tensors.At(i * 10 - 6);
         inp_gr = FF_Tensors.At(i * 10 - 1);
        }
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 2), FF_Tensors.At(i * 10 + 8), inp, inp_gr, iWindow, iWindow, None))
         ReturnFalse;
      //--- Passing gradient through Gate
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9) + 1), FF_Tensors.At(i * 10 + 7), FF_Tensors.At(i * 10 + 1), FF_Tensors.At(i * 10 + 6), 2 * iWindow, iWindow, None))
         ReturnFalse;
      //--- Concat
      if(IsStopped() || !DeConcat(FF_Tensors.At(i * 10 + 5), GetPointer(Temp), FF_Tensors.At(i * 10 + 6), iWindow, iWindow, iUnits))
         ReturnFalse;
      if(IsStopped() || !SumAndNormalize(GetPointer(Temp), inp_gr, inp_gr, iWindow, false, 0, 0, 0, 1))
         ReturnFalse;
      //--- Split gradient to multi-heads
      if(IsStopped() || !ConvolutionInputGradients(FF_Weights.At(i * (optimization == SGD ? 6 : 9)), FF_Tensors.At(i * 10 + 5), AO_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1), iWindowKey * iHeads, iWindow, None))
         ReturnFalse;
      //--- Passing gradient to query, key and value
      if(i == int(iLayers - 1) || (i + 1) % iLayersToOneKV == 0)
        {
         if(IsStopped() || !AttentionInsideGradients(QKV_Tensors.At(i * 2), QKV_Tensors.At(i * 2 + 1), KV_Tensors.At((i / iLayersToOneKV) * 2), kv_g, S_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1)))
            ReturnFalse;
        }
      else
        {
         if(IsStopped() || !AttentionInsideGradients(QKV_Tensors.At(i * 2), QKV_Tensors.At(i * 2 + 1), KV_Tensors.At((i / iLayersToOneKV) * 2), GetPointer(Temp), S_Tensors.At(i * 2), AO_Tensors.At(i * 2 + 1)))
            ReturnFalse;
         if(IsStopped() || !SumAndNormalize(kv_g, GetPointer(Temp), kv_g, iWindowKey, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      //---
      if(IsStopped() || !ConvolutionInputGradients(QKV_Weights.At(i * (optimization == SGD ? 2 : 3)), QKV_Tensors.At(i * 2 + 1), inp, GetPointer(Temp), iWindow, iWindowKey * iHeads, None))
         ReturnFalse;
      if(IsStopped() || !SumAndNormalize(GetPointer(Temp), inp_gr, inp_gr, iWindow, false, 0, 0, 0, 1))
         ReturnFalse;
      //---
      if((i % iLayersToOneKV) == 0)
        {
         CBufferFloat *k_g = K_Tensors.At((i / iLayersToOneKV) * 2 + 1);
         CBufferFloat *v_g = V_Tensors.At((i / iLayersToOneKV) * 2 + 1);
         if(IsStopped() || !DeConcat(k_g, v_g, kv_g, iWindowKey * iHeadsKV * iVariables, iWindowKey * iHeadsKV * iVariables, iUnits))
            ReturnFalse;
         if(IsStopped() || !ConvolutionInputGradients(K_Weights.At(i / iLayersToOneKV * (optimization == SGD ? 2 : 3)), k_g, inp, GetPointer(Temp), iWindow, iWindowKey * iHeadsKV, None))
            ReturnFalse;
         if(IsStopped() || !SumAndNormalize(GetPointer(Temp), inp_gr, inp_gr, iWindow, false, 0, 0, 0, 1))
            ReturnFalse;
         if(IsStopped() || !ConvolutionInputGradients(V_Weights.At(i / iLayersToOneKV * (optimization == SGD ? 2 : 3)), v_g, inp, GetPointer(Temp), iWindow, iWindowKey * iHeadsKV, None))
            ReturnFalse;
         if(IsStopped() || !SumAndNormalize(GetPointer(Temp), inp_gr, inp_gr, iWindow, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(i > 0)
         out_grad = inp_gr;
     }
//---
   if(prevLayer.Activation() != None)
     {
      if(!DeActivation(prevLayer.getOutput(), prevLayer.getGradient(), prevLayer.getGradient(), prevLayer.Activation()))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTAAEncoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *inputs = NeuronOCL.getOutput();
   for(uint l = 0; l < iLayers; l++)
     {
      if(IsStopped() || !ConvolutuionUpdateWeights(QKV_Weights.At(l * (optimization == SGD ? 2 : 3)), QKV_Tensors.At(l * 2 + 1), inputs, (optimization == SGD ? QKV_Weights.At(l * 2 + 1) : QKV_Weights.At(l * 3 + 1)), (optimization == SGD ? NULL : QKV_Weights.At(l * 3 + 2)), iWindow, iWindowKey * iHeads))
         ReturnFalse;
      if(l % iLayersToOneKV == 0)
        {
         uint l_kv = l / iLayersToOneKV;
         if(IsStopped() || !ConvolutuionUpdateWeights(K_Weights.At(l_kv * (optimization == SGD ? 2 : 3)), K_Tensors.At(l_kv * 2 + 1), inputs, (optimization == SGD ? K_Weights.At(l_kv * 2 + 1) : K_Weights.At(l_kv * 3 + 1)), (optimization == SGD ? NULL : K_Weights.At(l_kv * 3 + 2)), iWindow, iWindowKey * iHeadsKV, 0, iHeadsKV))
            ReturnFalse;
         if(IsStopped() || !ConvolutuionUpdateWeights(V_Weights.At(l_kv * (optimization == SGD ? 2 : 3)), V_Tensors.At(l_kv * 2 + 1), inputs, (optimization == SGD ? V_Weights.At(l_kv * 2 + 1) : V_Weights.At(l_kv * 3 + 1)), (optimization == SGD ? NULL : V_Weights.At(l_kv * 3 + 2)), iWindow, iWindowKey * iHeadsKV, 0, iHeadsKV))
            ReturnFalse;
        }
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9)), FF_Tensors.At(l * 10 + 5), AO_Tensors.At(l * 2), (optimization == SGD ? FF_Weights.At(l * 6 + 3) : FF_Weights.At(l * 9 + 3)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 6)), iWindowKey * iHeads, iWindow, 0, 1))
         ReturnFalse;
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9) + 1), FF_Tensors.At(l * 10 + 7), FF_Tensors.At(l * 10 + 1), (optimization == SGD ? FF_Weights.At(l * 6 + 4) : FF_Weights.At(l * 9 + 4)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 7)), 2 * iWindow, iWindow, 0, 1))
         ReturnFalse;
      //---
      if(IsStopped() || !ConvolutuionUpdateWeights(FF_Weights.At(l * (optimization == SGD ? 6 : 9) + 2), FF_Tensors.At(l * 10 + 8), inputs, (optimization == SGD ? FF_Weights.At(l * 6 + 5) : FF_Weights.At(l * 9 + 5)), (optimization == SGD ? NULL : FF_Weights.At(l * 9 + 8)), iWindow, iWindow, 0, 1))
         ReturnFalse;
      inputs = FF_Tensors.At(l * 10 + 4);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, uint num_traj, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(units_count < 2 ||
      !CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * forecast, optimization_type, batch))
      ReturnFalse;
//---
   iVariables = window;
   iHistory = units_count - 1;
   iForecast = forecast;
   iNumTraj = MathMax(num_traj, 1);
//---
   if(!cDataTAD.Init(0, 0, OpenCL, 2 * iVariables * iVariables * iHistory, optimization, iBatch))
      ReturnFalse;
   if(!cEmbeddingTAD.Init(0, 1, OpenCL, 2 * iVariables, 2 * iVariables, window_key, iVariables * iHistory, 1, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeATD.Init(0, 2, OpenCL, iHistory, iVariables, window_key, optimization, iBatch))
      ReturnFalse;
   if(!cAAEncoder.Init(0, 3, OpenCL, window_key, window_key, heads, (heads + 1) / 2, iVariables, 2, 1, iHistory, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeTAD.Init(0, 4, OpenCL, iVariables, iHistory, window_key, optimization, iBatch))
      ReturnFalse;
   if(!cPosEmbeddingTAD.Init(0, 5, OpenCL, iVariables * iHistory * window_key, optimization, iBatch))
      ReturnFalse;
   if(!cTemporalEncoder.Init(0, 6, OpenCL, window_key, window_key, heads, (heads + 1) / 2, iHistory, 2, 1, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cLineEmbeddibg.Init(0, 7, OpenCL, 3, 1, 8, iHistory - 1, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cPosLineEmbeddingTAD.Init(0, 8, OpenCL, cLineEmbeddibg.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cALEncoder.Init(0, 9, OpenCL, window_key, window_key, heads, 8, (heads + 1) / 2, iHistory, iHistory - 1, 2, 1, iVariables, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cGlobalEncoder.Init(0, 10, OpenCL, window_key * iVariables, window_key * iVariables, heads, (heads + 1) / 2, iHistory, 4, 2, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeADT.Init(0, 11, OpenCL, iHistory, window_key * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cDecoder[0].Init(0, 12, OpenCL, iHistory, iHistory, iForecast, window_key * iVariables, optimization, iBatch))
      ReturnFalse;
   cDecoder[0].SetActivationFunction(SIGMOID);
   if(!cDecoder[1].Init(0, 13, OpenCL, iForecast * window_key, iForecast * window_key, iForecast * window_key, iVariables, optimization, iBatch))
      ReturnFalse;
   cDecoder[1].SetActivationFunction(LReLU);
   if(!cDecoder[2].Init(0, 14, OpenCL, iForecast * window_key, iForecast * window_key, iForecast * iNumTraj, iVariables, optimization, iBatch))
      ReturnFalse;
   cDecoder[2].SetActivationFunction(TANH);
   if(!cProbProj.Init(0, 15, OpenCL, iForecast * iNumTraj, iForecast * iNumTraj, iNumTraj, iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cProbability.Init(0, 16, OpenCL, iForecast * iNumTraj * iVariables, optimization, iBatch))
      ReturnFalse;
   cProbability.SetHeads(iVariables); // Agent* Traj
   if(!cForecast.Init(0, 17, OpenCL, iForecast * iVariables, optimization, iBatch))
      ReturnFalse;
   if(!cTransposeTA.Init(0, 18, OpenCL, iVariables, iForecast, optimization, iBatch))
      ReturnFalse;
//---
   SetOutput(cTransposeTA.getOutput(), true);
   SetGradient(cTransposeTA.getGradient(), true);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::Prepare(const CNeuronBaseOCL* history)
  {
   if(!history)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iHistory, iVariables};
   setBuffer(def_k_HiVTPrepare, def_k_hivtp_data, history.getOutputIndex())
   setBuffer(def_k_HiVTPrepare, def_k_hivtp_output, cDataTAD.getOutputIndex())
   kernelExecute(def_k_HiVTPrepare, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cDataTAD.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!Prepare(NeuronOCL))
      ReturnFalse;
   if(!cEmbeddingTAD.FeedForward(cDataTAD.AsObject()))
      ReturnFalse;
   if(!cTransposeATD.FeedForward(cEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!cAAEncoder.FeedForward(cTransposeATD.AsObject()))
      ReturnFalse;
   if(!cTransposeTAD.FeedForward(cAAEncoder.AsObject()))
      ReturnFalse;
   if(!cPosEmbeddingTAD.FeedForward(cTransposeTAD.AsObject()))
      ReturnFalse;
   if(!cTemporalEncoder.FeedForward(cPosEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!cLineEmbeddibg.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cPosLineEmbeddingTAD.FeedForward(cLineEmbeddibg.AsObject()))
      ReturnFalse;
   if(!cALEncoder.FeedForward(cTemporalEncoder.AsObject(), cPosLineEmbeddingTAD.getOutput()))
      ReturnFalse;
   if(!cGlobalEncoder.FeedForward(cALEncoder.AsObject()))
      ReturnFalse;
   if(!cTransposeADT.FeedForward(cGlobalEncoder.AsObject()))
      ReturnFalse;
   if(!cDecoder[0].FeedForward(cTransposeADT.AsObject()))
      ReturnFalse;
   if(!cDecoder[1].FeedForward(cDecoder[0].AsObject()))
      ReturnFalse;
   if(!cDecoder[2].FeedForward(cDecoder[1].AsObject()))
      ReturnFalse;
   if(!cProbProj.FeedForward(cDecoder[2].AsObject()))
      ReturnFalse;
   if(!cProbability.FeedForward(cProbProj.AsObject()))
      ReturnFalse;
   if(IsStopped() ||
      !MatMul(cDecoder[2].getOutput(), cProbability.getOutput(), cForecast.getOutput(), iForecast, iNumTraj, 1, iVariables))
      ReturnFalse;
   if(!cTransposeTA.FeedForward(cForecast.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!cForecast.CalcHiddenGradients(cTransposeTA.AsObject()))
      ReturnFalse;
   if(IsStopped() ||
      !MatMulGrad(cDecoder[2].getOutput(), cDecoder[2].getGradient(), cProbability.getOutput(), cProbability.getGradient(), cForecast.getGradient(), iForecast, iNumTraj, 1, iVariables))
      ReturnFalse;
   if(!cProbProj.CalcHiddenGradients(cProbability.AsObject()))
      ReturnFalse;
   if(!cDecoder[1].CalcHiddenGradients(cDecoder[2].AsObject()))
      ReturnFalse;
   if(!cDecoder[0].CalcHiddenGradients(cDecoder[1].AsObject()))
      ReturnFalse;
   if(!cTransposeADT.CalcHiddenGradients(cDecoder[0].AsObject()))
      ReturnFalse;
   if(!cGlobalEncoder.CalcHiddenGradients(cTransposeADT.AsObject()))
      ReturnFalse;
   if(!cALEncoder.CalcHiddenGradients(cGlobalEncoder.AsObject()))
      ReturnFalse;
   if(!cTemporalEncoder.CalcHiddenGradients(cALEncoder.AsObject(), cPosLineEmbeddingTAD.getOutput(), cPosLineEmbeddingTAD.getGradient(), (ENUM_ACTIVATION)cPosLineEmbeddingTAD.Activation()))
      ReturnFalse;
   if(!cLineEmbeddibg.CalcHiddenGradients(cPosLineEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cLineEmbeddibg.AsObject()))
      ReturnFalse;
   if(!cPosEmbeddingTAD.CalcHiddenGradients(cTemporalEncoder.AsObject()))
      ReturnFalse;
   if(!cTransposeTAD.CalcHiddenGradients(cPosEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!cAAEncoder.CalcHiddenGradients(cTransposeTAD.AsObject()))
      ReturnFalse;
   if(!cTransposeATD.CalcHiddenGradients(cAAEncoder.AsObject()))
      ReturnFalse;
   if(!cEmbeddingTAD.CalcHiddenGradients(cTransposeATD.AsObject()))
      ReturnFalse;
   if(!cDataTAD.CalcHiddenGradients(cEmbeddingTAD.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cEmbeddingTAD.UpdateInputWeights(cDataTAD.AsObject()))
      ReturnFalse;
   if(!cAAEncoder.UpdateInputWeights(cTransposeATD.AsObject()))
      ReturnFalse;
   if(!cPosEmbeddingTAD.UpdateInputWeights(cTransposeTAD.AsObject()))
      ReturnFalse;
   if(!cTemporalEncoder.UpdateInputWeights(cPosEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!cLineEmbeddibg.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cPosLineEmbeddingTAD.UpdateInputWeights(cLineEmbeddibg.AsObject()))
      ReturnFalse;
   if(!cALEncoder.UpdateInputWeights(cTemporalEncoder.AsObject(), cPosLineEmbeddingTAD.getOutput()))
      ReturnFalse;
   if(!cGlobalEncoder.UpdateInputWeights(cALEncoder.AsObject()))
      ReturnFalse;
   if(!cDecoder[0].UpdateInputWeights(cTransposeADT.AsObject()))
      ReturnFalse;
   if(!cDecoder[1].UpdateInputWeights(cDecoder[0].AsObject()))
      ReturnFalse;
   if(!cDecoder[2].UpdateInputWeights(cDecoder[1].AsObject()))
      ReturnFalse;
   if(!cProbProj.UpdateInputWeights(cDecoder[2].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iVariables)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iHistory)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iForecast)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iNumTraj)) < INT_VALUE)
      ReturnFalse;
//---
   if(!cDataTAD.Save(file_handle))
      ReturnFalse;
   if(!cEmbeddingTAD.Save(file_handle))
      ReturnFalse;
   if(!cTransposeATD.Save(file_handle))
      ReturnFalse;
   if(!cAAEncoder.Save(file_handle))
      ReturnFalse;
   if(!cTransposeTAD.Save(file_handle))
      ReturnFalse;
   if(!cPosEmbeddingTAD.Save(file_handle))
      ReturnFalse;
   if(!cTemporalEncoder.Save(file_handle))
      ReturnFalse;
   if(!cLineEmbeddibg.Save(file_handle))
      ReturnFalse;
   if(!cPosLineEmbeddingTAD.Save(file_handle))
      ReturnFalse;
   if(!cALEncoder.Save(file_handle))
      ReturnFalse;
   if(!cGlobalEncoder.Save(file_handle))
      ReturnFalse;
   if(!cTransposeADT.Save(file_handle))
      ReturnFalse;
   if(!cDecoder[0].Save(file_handle))
      ReturnFalse;
   if(!cDecoder[1].Save(file_handle))
      ReturnFalse;
   if(!cDecoder[2].Save(file_handle))
      ReturnFalse;
   if(!cProbProj.Save(file_handle))
      ReturnFalse;
   if(!cProbability.Save(file_handle))
      ReturnFalse;
   if(!cForecast.Save(file_handle))
      ReturnFalse;
   if(!cTransposeTA.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHiVTOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHistory = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iForecast = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iNumTraj = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cDataTAD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeATD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAAEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeTAD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPosEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTemporalEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLineEmbeddibg.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPosLineEmbeddingTAD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cALEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGlobalEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeADT.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDecoder[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDecoder[1].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDecoder[2].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProbProj.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProbability.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cForecast.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeTA.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronHiVTOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cDataTAD.SetOpenCL(OpenCL);
   cEmbeddingTAD.SetOpenCL(OpenCL);
   cTransposeATD.SetOpenCL(OpenCL);
   cAAEncoder.SetOpenCL(OpenCL);
   cTransposeTAD.SetOpenCL(OpenCL);
   cPosEmbeddingTAD.SetOpenCL(OpenCL);
   cTemporalEncoder.SetOpenCL(OpenCL);
   cLineEmbeddibg.SetOpenCL(OpenCL);
   cPosLineEmbeddingTAD.SetOpenCL(OpenCL);
   cALEncoder.SetOpenCL(OpenCL);
   cGlobalEncoder.SetOpenCL(OpenCL);
   cTransposeADT.SetOpenCL(OpenCL);
   cDecoder[0].SetOpenCL(OpenCL);
   cDecoder[1].SetOpenCL(OpenCL);
   cDecoder[2].SetOpenCL(OpenCL);
   cProbProj.SetOpenCL(OpenCL);
   cProbability.SetOpenCL(OpenCL);
   cForecast.SetOpenCL(OpenCL);
   cTransposeTA.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronPointNetOCL::~CNeuronPointNetOCL(void)
  {
   DeleteObj(cTNet1);
   DeleteObj(cTNet2);
   DeleteObj(cTurned1);
   DeleteObj(cTurned2);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint output, bool use_tnets, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, output, optimization_type, batch))
      ReturnFalse;
//--- Init T-Nets
   if(use_tnets)
     {
      if(!cTNet1)
        {
         cTNet1 = new CNeuronPointNetOCL();
         if(!cTNet1)
            ReturnFalse;
        }
      if(!cTNet1.Init(0, 0, OpenCL, window, units_count, window * window, false, optimization, iBatch))
         ReturnFalse;
      if(!cTurned1)
        {
         cTurned1 = new CNeuronBaseOCL();
         if(!cTurned1)
            ReturnFalse;
        }
      if(!cTurned1.Init(0, 1, OpenCL, window * units_count, optimization, iBatch))
         ReturnFalse;
      if(!cTNet2)
        {
         cTNet2 = new CNeuronPointNetOCL();
         if(!cTNet2)
            ReturnFalse;
        }
      if(!cTNet2.Init(0, 2, OpenCL, 64, units_count, 64 * 64, false, optimization, iBatch))
         ReturnFalse;
      if(!cTurned2)
        {
         cTurned2 = new CNeuronBaseOCL();
         if(!cTurned2)
            ReturnFalse;
        }
      if(!cTurned2.Init(0, 3, OpenCL, 64 * units_count, optimization, iBatch))
         ReturnFalse;
     }
//--- Init PreNet
   if(!cPreNet[0].Init(0, 0, OpenCL, window, window, 64, units_count, optimization, iBatch))
      ReturnFalse;
   cPreNet[0].SetActivationFunction(None);
   if(!cPreNetNorm[0].Init(0, 1, OpenCL, 64 * units_count, iBatch, optimization))
      ReturnFalse;
   cPreNetNorm[0].SetActivationFunction(LReLU);
   if(!cPreNet[1].Init(0, 2, OpenCL, 64, 64, 64, units_count, optimization, iBatch))
      ReturnFalse;
   cPreNet[1].SetActivationFunction(None);
   if(!cPreNetNorm[1].Init(0, 3, OpenCL, 64 * units_count, iBatch, optimization))
      ReturnFalse;
   cPreNetNorm[1].SetActivationFunction(None);
//--- Init Feature Net
   if(!cFeatureNet[0].Init(0, 4, OpenCL, 64, 64, 64, units_count, optimization, iBatch))
      ReturnFalse;
   cFeatureNet[0].SetActivationFunction(None);
   if(!cFeatureNetNorm[0].Init(0, 5, OpenCL, 64 * units_count, iBatch, optimization))
      ReturnFalse;
   cFeatureNet[0].SetActivationFunction(LReLU);
   if(!cFeatureNet[1].Init(0, 6, OpenCL, 64, 64, 128, units_count, optimization, iBatch))
      ReturnFalse;
   cFeatureNet[1].SetActivationFunction(None);
   if(!cFeatureNetNorm[1].Init(0, 7, OpenCL, 128 * units_count, iBatch, optimization))
      ReturnFalse;
   cFeatureNetNorm[1].SetActivationFunction(LReLU);
   if(!cFeatureNet[2].Init(0, 8, OpenCL, 128, 128, 512, units_count, optimization, iBatch))
      ReturnFalse;
   cFeatureNet[2].SetActivationFunction(None);
   if(!cFeatureNetNorm[2].Init(0, 9, OpenCL, 512 * units_count, iBatch, optimization))
      ReturnFalse;
   cFeatureNetNorm[2].SetActivationFunction(None);
   if(!cTranspose.Init(0, 10, OpenCL, units_count, 512, optimization, iBatch))
      ReturnFalse;
   if(!cMaxPool.Init(512, 11, OpenCL, units_count, units_count, 512, optimization, iBatch))
      ReturnFalse;
//--- Init Final MLP
   if(!cFinalMLP[0].Init(256, 12, OpenCL, 512, optimization, iBatch))
      ReturnFalse;
   cFinalMLP[0].SetActivationFunction(LReLU);
   if(!cFinalMLP[1].Init(output, 13, OpenCL, 256, optimization, iBatch))
      ReturnFalse;
   cFinalMLP[1].SetActivationFunction(LReLU);
//---
   SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- PreNet
   if(!cTNet1)
     {
      if(!cPreNet[0].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   else
     {
      if(!cTurned1)
         ReturnFalse;
      if(!cTNet1.FeedForward(NeuronOCL))
         ReturnFalse;
      int window = (int)MathSqrt(cTNet1.Neurons());
      if(IsStopped() ||
         !MatMul(NeuronOCL.getOutput(), cTNet1.getOutput(), cTurned1.getOutput(), NeuronOCL.Neurons() / window, window, window))
         ReturnFalse;
      if(!cPreNet[0].FeedForward(cTurned1.AsObject()))
         ReturnFalse;
     }
   if(!cPreNetNorm[0].FeedForward(cPreNet[0].AsObject()))
      ReturnFalse;
   if(!cPreNet[1].FeedForward(cPreNetNorm[0].AsObject()))
      ReturnFalse;
   if(!cPreNetNorm[1].FeedForward(cPreNet[1].AsObject()))
      ReturnFalse;
//--- Feature Net
   if(!cTNet2)
     {
      if(!cFeatureNet[0].FeedForward(cPreNetNorm[1].AsObject()))
         ReturnFalse;
     }
   else
     {
      if(!cTurned2)
         ReturnFalse;
      if(!cTNet2.FeedForward(cPreNetNorm[1].AsObject()))
         ReturnFalse;
      int window = (int)MathSqrt(cTNet2.Neurons());
      if(IsStopped() ||
         !MatMul(cPreNetNorm[1].getOutput(), cTNet2.getOutput(), cTurned2.getOutput(), cPreNetNorm[1].Neurons() / window, window, window))
         ReturnFalse;
      if(!cFeatureNet[0].FeedForward(cTurned2.AsObject()))
         ReturnFalse;
     }
   if(!cFeatureNetNorm[0].FeedForward(cFeatureNet[0].AsObject()))
      ReturnFalse;
   uint total = cFeatureNet.Size();
   for(uint i = 1; i < total; i++)
     {
      if(!cFeatureNet[i].FeedForward(cFeatureNetNorm[i - 1].AsObject()))
         ReturnFalse;
      if(!cFeatureNetNorm[i].FeedForward(cFeatureNet[i].AsObject()))
         ReturnFalse;
     }
   if(!cTranspose.FeedForward(cFeatureNetNorm[total - 1].AsObject()))
      ReturnFalse;
   if(!cMaxPool.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cFinalMLP[0].FeedForward(cMaxPool.AsObject()))
      ReturnFalse;
   if(!cFinalMLP[1].FeedForward(cFinalMLP[0].AsObject()))
      ReturnFalse;
   if(!CNeuronBaseOCL::feedForward(cFinalMLP[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::OrthoganalLoss(CNeuronBaseOCL* NeuronOCL, bool add = false)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint dimension = (uint)MathSqrt(NeuronOCL.Neurons());
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {dimension, dimension};
   uint local_work_size[2] = {1, dimension};
   int kernel = def_k_OrthoganalLoss;
   setBuffer(kernel, def_k_ol_data, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ol_grad, NeuronOCL.getGradientIndex())
   setArgument(kernel, def_k_ol_add, (int)add)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
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
bool CNeuronPointNetOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronBaseOCL::calcInputGradients(cFinalMLP[1].AsObject()))
      ReturnFalse;
   if(!cFinalMLP[0].CalcHiddenGradients(cFinalMLP[1].AsObject()))
      ReturnFalse;
//--- Feature Net
   if(!cMaxPool.CalcHiddenGradients(cFinalMLP[0].AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cMaxPool.AsObject()))
      ReturnFalse;
   uint total = cFeatureNet.Size();
   for(uint i = total - 1; i > 0; i--)
     {
      if(!cFeatureNet[i].CalcHiddenGradients(cFeatureNetNorm[i].AsObject()))
         ReturnFalse;
      if(!cFeatureNetNorm[i - 1].CalcHiddenGradients(cFeatureNet[i].AsObject()))
         ReturnFalse;
     }
   if(!cFeatureNet[0].CalcHiddenGradients(cFeatureNetNorm[0].AsObject()))
      ReturnFalse;
   if(!cTNet2)
     {
      if(!cPreNetNorm[1].CalcHiddenGradients(cFeatureNet[0].AsObject()))
         ReturnFalse;
     }
   else
     {
      if(!cTurned2)
         ReturnFalse;
      if(!cTurned2.CalcHiddenGradients(cFeatureNet[0].AsObject()))
         ReturnFalse;
      int window = (int)MathSqrt(cTNet2.Neurons());
      if(IsStopped() ||
         !MatMulGrad(cPreNetNorm[1].getOutput(), cPreNet[1].getGradient(), cTNet2.getOutput(), cTNet2.getGradient(), cTurned2.getGradient(), cPreNetNorm[1].Neurons() / window, window, window))
         ReturnFalse;
      if(!OrthoganalLoss(cTNet2.AsObject(), true))
         ReturnFalse;
      if(!cPreNetNorm[1].CalcHiddenGradients((CObject*)cTNet2))
         ReturnFalse;
      if(!SumAndNormalize(cPreNetNorm[1].getGradient(), cPreNet[1].getGradient(), cPreNetNorm[1].getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//--- PreNet
   if(!cPreNet[1].CalcHiddenGradients(cPreNetNorm[1].AsObject()))
      ReturnFalse;
   if(!cPreNetNorm[0].CalcHiddenGradients(cPreNet[1].AsObject()))
      ReturnFalse;
   if(!cPreNet[0].CalcHiddenGradients(cPreNetNorm[0].AsObject()))
      ReturnFalse;
   if(!cTNet1)
     {
      if(!NeuronOCL.CalcHiddenGradients(cPreNet[0].AsObject()))
         ReturnFalse;
     }
   else
     {
      if(!cTurned1)
         ReturnFalse;
      if(!cTurned1.CalcHiddenGradients(cPreNet[0].AsObject()))
         ReturnFalse;
      int window = (int)MathSqrt(cTNet1.Neurons());
      if(IsStopped() ||
         !MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(), cTNet1.getOutput(), cTNet1.getGradient(), cTurned1.getGradient(), NeuronOCL.Neurons() / window, window, window))
         ReturnFalse;
      if(!OrthoganalLoss(cTNet1, true))
         ReturnFalse;
      //---
      CBufferFloat *temp = NeuronOCL.getGradient();
      NeuronOCL.SetGradient(cTurned1.getGradient(), false);
      cTurned1.SetGradient(temp, false);
      //---
      if(!NeuronOCL.CalcHiddenGradients(cTNet1.AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(NeuronOCL.getGradient(), cTurned1.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- PreNet
   if(!cTNet1)
     {
      if(!cPreNet[0].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
     }
   else
     {
      if(!cTNet1.UpdateInputWeights(NeuronOCL))
         ReturnFalse;
      if(!cPreNet[0].UpdateInputWeights(cTurned1.AsObject()))
         ReturnFalse;
     }
   if(!cPreNetNorm[0].UpdateInputWeights(cPreNet[0].AsObject()))
      ReturnFalse;
   if(!cPreNet[1].UpdateInputWeights(cPreNetNorm[0].AsObject()))
      ReturnFalse;
   if(!cPreNetNorm[1].UpdateInputWeights(cPreNet[1].AsObject()))
      ReturnFalse;
//--- Feature Net
   if(!cTNet2)
     {
      if(!cFeatureNet[0].UpdateInputWeights(cPreNetNorm[1].AsObject()))
         ReturnFalse;
     }
   else
     {
      if(!cTurned2)
         ReturnFalse;
      if(!cTNet2.UpdateInputWeights(cPreNetNorm[1].AsObject()))
         ReturnFalse;
      if(!cFeatureNet[0].UpdateInputWeights(cTurned2.AsObject()))
         ReturnFalse;
     }
   if(!cFeatureNetNorm[0].UpdateInputWeights(cFeatureNet[0].AsObject()))
      ReturnFalse;
   uint total = cFeatureNet.Size();
   for(uint i = 1; i < total; i++)
     {
      if(!cFeatureNet[i].UpdateInputWeights(cFeatureNetNorm[i - 1].AsObject()))
         ReturnFalse;
      if(!cFeatureNetNorm[i].UpdateInputWeights(cFeatureNet[i].AsObject()))
         ReturnFalse;
     }
   if(!cFinalMLP[0].UpdateInputWeights(cMaxPool.AsObject()))
      ReturnFalse;
   if(!cFinalMLP[1].UpdateInputWeights(cFinalMLP[0].AsObject()))
      ReturnFalse;
   if(!CNeuronBaseOCL::updateInputWeights(cFinalMLP[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!(!cTNet1 || !cTNet2))
     {
      if(FileWriteInteger(file_handle, 1) < INT_VALUE)
         ReturnFalse;
      int units = int(cTurned1.Neurons() / MathSqrt(cTNet1.Neurons()));
      if(FileWriteInteger(file_handle, units) < INT_VALUE)
         ReturnFalse;
      if(!cTNet1.Save(file_handle))
         ReturnFalse;
      if(!cTNet2.Save(file_handle))
         ReturnFalse;
     }
   else
      if(FileWriteInteger(file_handle, 0) < INT_VALUE)
         ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
     {
      if(!cPreNet[i].Save(file_handle))
         ReturnFalse;
      if(!cPreNetNorm[i].Save(file_handle))
         ReturnFalse;
     }
   for(int i = 0; i < 3; i++)
     {
      if(!cFeatureNet[i].Save(file_handle))
         ReturnFalse;
      if(!cFeatureNetNorm[i].Save(file_handle))
         ReturnFalse;
     }
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cMaxPool.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cFinalMLP[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNetOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   bool use_tnets = bool(FileReadInteger(file_handle));
   if(use_tnets)
     {
      int units = FileReadInteger(file_handle);
      if(!cTNet1)
        {
         cTNet1 = new CNeuronPointNetOCL();
         if(!cTNet1)
            ReturnFalse;
        }
      if(!cTNet1.Init(0, 0, OpenCL, 1, 1, 1, false, optimization, iBatch))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cTNet1))
         ReturnFalse;
      if(!cTurned1)
        {
         cTurned1 = new CNeuronBaseOCL();
         if(!cTurned1)
            ReturnFalse;
        }
      int dimension = (int)MathSqrt(cTNet1.Neurons());
      if(!cTurned1.Init(0, 1, OpenCL, dimension * units, optimization, iBatch))
         ReturnFalse;
      //---
      if(!cTNet2)
        {
         cTNet2 = new CNeuronPointNetOCL();
         if(!cTNet2)
            ReturnFalse;
        }
      if(!cTNet2.Init(0, 2, OpenCL, 1, 1, 1, false, optimization, iBatch))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cTNet2))
         ReturnFalse;
      if(!cTurned2)
        {
         cTurned2 = new CNeuronBaseOCL();
         if(!cTurned2)
            ReturnFalse;
        }
      dimension = (int)MathSqrt(cTNet2.Neurons());
      if(!cTurned2.Init(0, 3, OpenCL, dimension * units, optimization, iBatch))
         ReturnFalse;
     }
//---
   for(int i = 0; i < 2; i++)
     {
      if(!LoadInsideLayer(file_handle, cPreNet[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cPreNetNorm[i].AsObject()))
         ReturnFalse;
     }
   for(int i = 0; i < 3; i++)
     {
      if(!LoadInsideLayer(file_handle, cFeatureNet[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cFeatureNetNorm[i].AsObject()))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMaxPool.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cFinalMLP[i].AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPointNetOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!cTNet1)
      cTNet1.SetOpenCL(OpenCL);
   if(!!cTurned1)
      cTurned1.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
     {
      cPreNet[i].SetOpenCL(OpenCL);
      cPreNetNorm[i].SetOpenCL(OpenCL);
     }
   if(!!cTNet2)
      cTNet2.SetOpenCL(OpenCL);
   if(!!cTurned2)
      cTurned2.SetOpenCL(OpenCL);
   for(int i = 0; i < 3; i++)
     {
      cFeatureNet[i].SetOpenCL(OpenCL);
      cFeatureNetNorm[i].SetOpenCL(OpenCL);
     }
   cTranspose.SetOpenCL(OpenCL);
   cMaxPool.SetOpenCL(OpenCL);
   for(int i = 0; i < 2; i++)
      cFinalMLP[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint window_out, float radius, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, 128, 128, window_out, units_count, 1, optimization_type, batch))
      ReturnFalse;
//---
   fRadius = MathMax(0.1f, radius);
   iUnits = units_count;
//---
   cDistance.BufferFree();
   if(!cDistance.BufferInit(iUnits * iUnits, 0) ||
      !cDistance.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!cFeatureNet[0].Init(0, 0, OpenCL, window, window, 64, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   if(!cFeatureNetNorm[0].Init(0, 1, OpenCL, 64 * iUnits, iBatch, optimization))
      ReturnFalse;
   cFeatureNetNorm[0].SetActivationFunction(LReLU);
   if(!cFeatureNet[1].Init(0, 2, OpenCL, 64, 64, 128, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   if(!cFeatureNetNorm[1].Init(0, 3, OpenCL, 128 * iUnits, iBatch, optimization))
      ReturnFalse;
   cFeatureNetNorm[1].SetActivationFunction(LReLU);
   if(!cFeatureNet[2].Init(0, 4, OpenCL, 128, 128, 256, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   if(!cFeatureNetNorm[2].Init(0, 5, OpenCL, 256 * iUnits, iBatch, optimization))
      cFeatureNetNorm[2].SetActivationFunction(None);
//---
   if(!cLocalMaxPool.Init(0, 6, OpenCL, cFeatureNetNorm[2].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cFinalMLP.Init(0, 7, OpenCL, 256, 256, 128, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   cFinalMLP.SetActivationFunction(LReLU);
//---
   SetActivationFunction(None);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::CalcDistance(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnits, iUnits};
   uint local_work_size[2] = {1, iUnits};
   int kernel = def_k_CalcDistance;
   setBuffer(kernel, def_k_cd_data, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_cd_distance, cDistance.GetIndex())
   setArgument(kernel, def_k_cd_dimension, (int)iWindow)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cDistance.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::LocalMaxPool(void)
  {
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnits, iWindow};
   int kernel = def_k_FeedForwardLocalMax;
   setBuffer(kernel, def_k_fflm_matrix_i, cFeatureNetNorm[cFeatureNetNorm.Size() - 1].getOutputIndex())
   setBuffer(kernel, def_k_fflm_distance, cDistance.GetIndex())
   setBuffer(kernel, def_k_fflm_matrix_o, cLocalMaxPool.getOutputIndex())
   setArgument(kernel, def_k_fflm_radius, fRadius)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cLocalMaxPool.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::LocalMaxPoolGrad(void)
  {
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnits, iWindow};
   int kernel = def_k_CalcInputGradientLocalMax;
   setBuffer(kernel, def_k_cglm_matrix_i, cFeatureNetNorm[cFeatureNetNorm.Size() - 1].getOutputIndex())
   setBuffer(kernel, def_k_cglm_matrix_ig, cFeatureNetNorm[cFeatureNetNorm.Size() - 1].getGradientIndex())
   setBuffer(kernel, def_k_cglm_distance, cDistance.GetIndex())
   setBuffer(kernel, def_k_cglm_matrix_o, cLocalMaxPool.getOutputIndex())
   setBuffer(kernel, def_k_cglm_matrix_g, cLocalMaxPool.getGradientIndex())
   setArgument(kernel, def_k_cglm_radius, fRadius)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFeatureNetNorm[cFeatureNetNorm.Size() - 1].getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CalcDistance(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL *temp = NeuronOCL;
   uint total = cFeatureNet.Size();
   for(uint i = 0; i < total; i++)
     {
      if(!cFeatureNet[i].FeedForward(temp))
         ReturnFalse;
      if(!cFeatureNetNorm[i].FeedForward(cFeatureNet[i].AsObject()))
         ReturnFalse;
      temp = cFeatureNetNorm[i].AsObject();
     }
   if(!LocalMaxPool())
      ReturnFalse;
   if(!cFinalMLP.FeedForward(cLocalMaxPool.AsObject()))
      ReturnFalse;
   if(!CNeuronConvOCL::feedForward(cFinalMLP.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronConvOCL::calcInputGradients(cFinalMLP.AsObject()))
      ReturnFalse;
   if(!cLocalMaxPool.CalcHiddenGradients(cFinalMLP.AsObject()))
      ReturnFalse;
   if(!LocalMaxPoolGrad())
      ReturnFalse;
   int total = (int)cFeatureNet.Size() - 1;
   for(int i = total; i > 0; i--)
     {
      if(!cFeatureNet[i].CalcHiddenGradients(cFeatureNetNorm[i].AsObject()))
         ReturnFalse;
      if(!cFeatureNetNorm[i - 1].CalcHiddenGradients(cFeatureNet[i].AsObject()))
         ReturnFalse;
     }
   if(!cFeatureNet[0].CalcHiddenGradients(cFeatureNetNorm[0].AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cFeatureNet[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *temp = NeuronOCL;
   uint total = cFeatureNet.Size();
   for(uint i = 0; i < total; i++)
     {
      if(!cFeatureNet[i].UpdateInputWeights(temp))
         ReturnFalse;
      if(!cFeatureNetNorm[i].UpdateInputWeights(cFeatureNet[i].AsObject()))
         ReturnFalse;
      temp = cFeatureNetNorm[i].AsObject();
     }
   if(!cFinalMLP.UpdateInputWeights(cLocalMaxPool.AsObject()))
      ReturnFalse;
   if(!CNeuronConvOCL::updateInputWeights(cFinalMLP.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
//--- Save constants
   if(FileWriteFloat(file_handle, fRadius) < sizeof(float))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iUnits) < INT_VALUE)
      ReturnFalse;
//--- Save objects
   uint total = cFeatureNet.Size();
   for(uint i = 0; i < total; i++)
     {
      if(!cFeatureNet[i].Save(file_handle))
         ReturnFalse;
      if(!cFeatureNetNorm[i].Save(file_handle))
         ReturnFalse;
     }
   if(!cFinalMLP.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2Local::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
//--- Load constants
   fRadius = FileReadFloat(file_handle);
   iUnits = FileReadInteger(file_handle);
//--- Load objects
   uint total = cFeatureNet.Size();
   for(uint i = 0; i < total; i++)
     {
      if(!LoadInsideLayer(file_handle, cFeatureNet[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cFeatureNetNorm[i].AsObject()))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, cFinalMLP.AsObject()))
      ReturnFalse;
//---
   cDistance.BufferFree();
   if(!cDistance.BufferInit(iUnits * iUnits, 0) ||
      !cDistance.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!cLocalMaxPool.Init(0, 6, OpenCL, cFeatureNetNorm[2].Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPointNet2Local::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   uint total = cFeatureNet.Size();
   for(uint i = 0; i < total; i++)
     {
      cFeatureNet[i].SetOpenCL(OpenCL);
      cFeatureNetNorm[i].SetOpenCL(OpenCL);
     }
   cFinalMLP.SetOpenCL(OpenCL);
   cDistance.BufferCreate(OpenCL);
   cLocalMaxPool.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronPointNet2OCL::~CNeuronPointNet2OCL(void)
  {
   DeleteObj(cTNetG);
   DeleteObj(cTurnedG);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint output, bool use_tnets, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronPointNetOCL::Init(numOutputs, myIndex, open_cl, 64, units_count, output, use_tnets, optimization_type, batch))
      ReturnFalse;
//--- Init T-Nets
   if(use_tnets)
     {
      if(!cTNetG)
        {
         cTNetG = new CNeuronPointNetOCL();
         if(!cTNetG)
            ReturnFalse;
        }
      if(!cTNetG.Init(0, 0, OpenCL, window, units_count, window * window, false, optimization, iBatch))
         ReturnFalse;
      if(!cTurnedG)
        {
         cTurnedG = new CNeuronBaseOCL();
         if(!cTurned1)
            ReturnFalse;
        }
      if(!cTurnedG.Init(0, 1, OpenCL, window * units_count, optimization, iBatch))
         ReturnFalse;
     }
   else
     {
      DeleteObj(cTNetG);
      DeleteObj(cTurnedG);
     }
//---
   if(!caLocalPointNet[0].Init(0, 0, OpenCL, window, units_count, 64, 0.1f, optimization, iBatch))
      ReturnFalse;
   if(!caLocalPointNet[1].Init(0, 0, OpenCL, 64, units_count, 64, 0.3f, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   if(!cTNetG)
     {
      if(!caLocalPointNet[0].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   else
     {
      if(!cTurnedG)
         ReturnFalse;
      if(!cTNetG.FeedForward(NeuronOCL))
         ReturnFalse;
      int window = (int)MathSqrt(cTNetG.Neurons());
      if(IsStopped() ||
         !MatMul(NeuronOCL.getOutput(), cTNetG.getOutput(), cTurnedG.getOutput(), NeuronOCL.Neurons() / window, window, window))
         ReturnFalse;
      if(!caLocalPointNet[0].FeedForward(cTurnedG.AsObject()))
         ReturnFalse;
     }
   if(!caLocalPointNet[1].FeedForward(caLocalPointNet[0].AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::feedForward(caLocalPointNet[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronPointNetOCL::calcInputGradients(caLocalPointNet[1].AsObject()))
      ReturnFalse;
//--- Local Net
   if(!caLocalPointNet[0].CalcHiddenGradients(caLocalPointNet[1].AsObject()))
      ReturnFalse;
   if(!cTNetG)
     {
      if(!NeuronOCL.CalcHiddenGradients(caLocalPointNet[0].AsObject()))
         ReturnFalse;
     }
   else
     {
      if(!cTurnedG)
         ReturnFalse;
      if(!cTurnedG.CalcHiddenGradients(caLocalPointNet[0].AsObject()))
         ReturnFalse;
      int window = (int)MathSqrt(cTNetG.Neurons());
      if(IsStopped() ||
         !MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(), cTNetG.getOutput(), cTNetG.getGradient(), cTurnedG.getGradient(), NeuronOCL.Neurons() / window, window, window))
         ReturnFalse;
      if(!OrthoganalLoss(cTNetG, true))
         ReturnFalse;
      //---
      CBufferFloat *temp = NeuronOCL.getGradient();
      NeuronOCL.SetGradient(cTurnedG.getGradient(), false);
      cTurnedG.SetGradient(temp, false);
      //---
      if(!NeuronOCL.CalcHiddenGradients(cTNetG.AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(NeuronOCL.getGradient(), cTurnedG.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   if(!cTNetG)
     {
      if(!caLocalPointNet[0].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
     }
   else
     {
      if(!cTurnedG)
         ReturnFalse;
      if(!cTNetG.UpdateInputWeights(NeuronOCL))
         ReturnFalse;
      if(!caLocalPointNet[0].UpdateInputWeights(cTurnedG.AsObject()))
         ReturnFalse;
     }
   if(!caLocalPointNet[1].UpdateInputWeights(caLocalPointNet[0].AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::updateInputWeights(caLocalPointNet[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPointNet2OCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronPointNetOCL::SetOpenCL(obj);
   if(!!cTNetG)
     {
      cTNetG.SetOpenCL(OpenCL);
      cTurnedG.SetOpenCL(OpenCL);
     }
   caLocalPointNet[0].SetOpenCL(OpenCL);
   caLocalPointNet[1].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::Save(const int file_handle)
  {
   if(!CNeuronPointNetOCL::Save(file_handle))
      ReturnFalse;
   if(!caLocalPointNet[0].Save(file_handle))
      ReturnFalse;
   if(!caLocalPointNet[1].Save(file_handle))
      ReturnFalse;
   bool use_tnet = bool(!!cTNetG);
   if(FileWriteInteger(file_handle, int(use_tnet)) < INT_VALUE)
      ReturnFalse;
   if(use_tnet)
     {
      if(!cTNetG.Save(file_handle))
         ReturnFalse;
      if(!cTurnedG.Save(file_handle))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointNet2OCL::Load(const int file_handle)
  {
   if(!CNeuronPointNetOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, caLocalPointNet[0].AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, caLocalPointNet[1].AsObject()))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   bool use_tnet = bool(FileReadInteger(file_handle));
   if(use_tnet)
     {
      if(!cTNetG)
        {
         cTNetG = new CNeuronPointNetOCL();
         if(!cTNetG)
            ReturnFalse;
        }
      if(!cTNetG.Init(0, 0, OpenCL, 1, 1, 1, false, optimization, iBatch))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cTNetG))
         ReturnFalse;
      if(!cTurnedG)
        {
         cTurnedG = new CNeuronBaseOCL();
         if(!cTurnedG)
            ReturnFalse;
        }
      if(!cTurnedG.Init(0, 1, OpenCL, 1, optimization, iBatch))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, cTurnedG))
         ReturnFalse;
     }
   else
     {
      DeleteObj(cTNetG);
      DeleteObj(cTurnedG);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint output, bool use_tnets, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronPointNet2OCL::Init(numOutputs, myIndex, open_cl, window, units_count, output, use_tnets, optimization_type, batch))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      if(!caLocalAttention[i].Init(0, i * 5, OpenCL, 64, 16, 4, units_count, 2, optimization, iBatch))
         ReturnFalse;
      if(!caLocalGlobalAttention[i].Init(0, i * 5 + 1, OpenCL, 64, 16, 4, 64, 2, units_count, units_count, 2, 2, optimization, iBatch))
         ReturnFalse;
      if(!caGlobalAttention[i].Init(0, i * 5 + 2, OpenCL, 64, 16, 4, 2, units_count, 2, 2, optimization, iBatch))
         ReturnFalse;
      if(!caLocalPE[i].Init(0, i * 5 + 3, OpenCL, 64 * units_count, optimization, iBatch))
         ReturnFalse;
      if(!caGlobalPE[i].Init(0, i * 5 + 4, OpenCL, 64 * units_count, optimization, iBatch))
         ReturnFalse;
     }
   caLocalAttention[0].Sparse(0.1f);
   caLocalAttention[1].Sparse(0.3f);
//---
   if(!cConcatenate.Init(0, 10, OpenCL, 128 * units_count, optimization, iBatch))
      ReturnFalse;
   if(!cScale.Init(0, 11, OpenCL, 128, 128, 64, units_count, 1, optimization, iBatch))
      ReturnFalse;
//---
   DeleteObj(cbTemp);
   cbTemp = new CBufferFloat();
   if(!cbTemp ||
      !cbTemp.BufferInit(caGlobalAttention[0].Neurons(), 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   CNeuronBaseOCL *inputs = NeuronOCL;
   for(int i = 0; i < 2; i++)
     {
      if(!cTNetG || i > 0)
        {
         if(!caLocalPointNet[i].FeedForward(inputs))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.FeedForward(inputs))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMul(NeuronOCL.getOutput(), cTNetG.getOutput(), cTurnedG.getOutput(), NeuronOCL.Neurons() / window, window, window))
            ReturnFalse;
         if(!caLocalPointNet[i].FeedForward(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Local Attention
      if(!caLocalAttention[i].FeedForward(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalPE[i].FeedForward(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!caGlobalPE[i].FeedForward(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalGlobalAttention[i].FeedForward(caLocalPE[i].AsObject(), caGlobalPE[i].getOutput()))
         ReturnFalse;
      //--- Global Attention
      if(!caGlobalAttention[i].FeedForward(caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      inputs = caGlobalAttention[i].AsObject();
     }
//---
   if(!Concat(caGlobalAttention[0].getOutput(), caGlobalAttention[1].getOutput(), cConcatenate.getOutput(), 64, 64, cConcatenate.Neurons() / 128))
      ReturnFalse;
   if(!cScale.FeedForward(cConcatenate.AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::feedForward(cScale.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPointNetOCL::calcInputGradients(cScale.AsObject()))
      ReturnFalse;
   if(!cConcatenate.CalcHiddenGradients(cScale.AsObject()))
      ReturnFalse;
   if(!DeConcat(caGlobalAttention[0].getGradient(), caGlobalAttention[1].getGradient(), cConcatenate.getGradient(), 64, 64, cConcatenate.Neurons() / 128))
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = caGlobalAttention[0].AsObject();
   for(int i = 1; i >= 0; i--)
     {
      //--- Global Attention
      if(!caLocalGlobalAttention[i].CalcHiddenGradients(caGlobalAttention[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalPE[i].CalcHiddenGradients(caLocalGlobalAttention[i].AsObject(), caGlobalPE[i].getOutput(),
                                           caGlobalPE[i].getGradient(),
                                           (ENUM_ACTIVATION)caGlobalPE[i].Activation()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalAttention[i].CalcHiddenGradients(caLocalPE[i].AsObject()))
         ReturnFalse;
      if(!caLocalPointNet[i].CalcHiddenGradients(caGlobalPE[i].AsObject()))
         ReturnFalse;
      //--- Local Attention
      CBufferFloat *temp = caLocalPointNet[i].getGradient();
      caLocalPointNet[i].SetGradient(cbTemp, false);
      cbTemp = temp;
      if(!caLocalPointNet[i].CalcHiddenGradients(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(caLocalPointNet[i].getGradient(), cbTemp, caLocalPointNet[i].getGradient(), 64, false, 0, 0, 0, 1))
         ReturnFalse;
      if(i > 0)
        {
         temp = inputs.getGradient();
         inputs.SetGradient(cbTemp, false);
         cbTemp = temp;
        }
      //--- Local Net
      if(!cTNetG || i > 0)
        {
         if(!inputs.CalcHiddenGradients(caLocalPointNet[i].AsObject()))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTurnedG.CalcHiddenGradients(caLocalPointNet[i].AsObject()))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMulGrad(inputs.getOutput(), inputs.getGradient(), cTNetG.getOutput(), cTNetG.getGradient(), cTurnedG.getGradient(), inputs.Neurons() / window, window, window))
            ReturnFalse;
         if(!OrthoganalLoss(cTNetG, true))
            ReturnFalse;
         //---
         CBufferFloat *temp = inputs.getGradient();
         inputs.SetGradient(cTurnedG.getGradient(), false);
         cTurnedG.SetGradient(temp, false);
         //---
         if(!inputs.CalcHiddenGradients(cTNetG.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(inputs.getGradient(), cTurnedG.getGradient(), inputs.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(i > 0)
        {
         if(!SumAndNormalize(inputs.getGradient(), cbTemp, inputs.getGradient(), 64, false, 0, 0, 0, 1))
            ReturnFalse;
         inputs = caGlobalAttention[i - 1].AsObject();
        }
      else
         inputs = NeuronOCL;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   CNeuronBaseOCL *inputs = NeuronOCL;
   for(int i = 0; i < 2; i++)
     {
      if(!cTNetG || i > 0)
        {
         if(!caLocalPointNet[i].UpdateInputWeights(inputs))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.UpdateInputWeights(inputs))
            ReturnFalse;
         if(!caLocalPointNet[i].UpdateInputWeights(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Local Attention
      if(!caLocalAttention[i].UpdateInputWeights(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalPE[i].UpdateInputWeights(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!caGlobalPE[i].UpdateInputWeights(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalGlobalAttention[i].UpdateInputWeights(caLocalPE[i].AsObject(), caGlobalPE[i].getOutput()))
         ReturnFalse;
      //--- Global Attention
      if(!caGlobalAttention[i].UpdateInputWeights(caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      inputs = caGlobalAttention[i].AsObject();
     }
//---
   if(!cScale.UpdateInputWeights(cConcatenate.AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::updateInputWeights(cScale.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::Save(const int file_handle)
  {
   if(!CNeuronPointNet2OCL::Save(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
     {
      if(!caLocalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caLocalGlobalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caGlobalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caLocalPE[i].Save(file_handle))
         ReturnFalse;
      if(!caGlobalPE[i].Save(file_handle))
         ReturnFalse;
     }
   if(!cConcatenate.Save(file_handle))
      ReturnFalse;
   if(!cScale.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPointFormer::Load(const int file_handle)
  {
   DeleteObj(cbTemp);
//---
   if(!CNeuronPointNet2OCL::Load(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
     {
      if(!LoadInsideLayer(file_handle, caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caGlobalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caLocalPE[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caGlobalPE[i].AsObject()))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, cConcatenate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScale.AsObject()))
      ReturnFalse;
//---
   cbTemp = new CBufferFloat();
   if(!cbTemp ||
      !cbTemp.BufferInit(caGlobalAttention[0].Neurons(), 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPointFormer::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronPointNet2OCL::SetOpenCL(obj);
//---
   for(int i = 0; i < 2; i++)
     {
      caLocalAttention[i].SetOpenCL(OpenCL);
      caLocalGlobalAttention[i].SetOpenCL(OpenCL);
      caGlobalAttention[i].SetOpenCL(OpenCL);
      caLocalPE[i].SetOpenCL(OpenCL);
      caGlobalPE[i].SetOpenCL(OpenCL);
     }
   cConcatenate.SetOpenCL(OpenCL);
   cScale.SetOpenCL(OpenCL);
//---
   cbTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint output, bool use_tnets, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronPointNet2OCL::Init(numOutputs, myIndex, open_cl, window, units_count, output, use_tnets, optimization_type, batch))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      if(!caLocalAttention[i].Init(0, i * 5, OpenCL, 64, 16, 4, units_count, 2, optimization, iBatch))
         ReturnFalse;
      if(!caLocalGlobalAttention[i].Init(0, i * 5 + 1, OpenCL, 64, 16, 4, 64, 2, units_count, units_count, 2, 2, optimization, iBatch))
         ReturnFalse;
      if(!caGlobalAttention[i].Init(0, i * 5 + 2, OpenCL, 64, 16, 4, 2, units_count, 2, 2, optimization, iBatch))
         ReturnFalse;
      if(!caLocalPE[i].Init(0, i * 5 + 3, OpenCL, 64 * units_count, optimization, iBatch))
         ReturnFalse;
      if(!caGlobalPE[i].Init(0, i * 5 + 4, OpenCL, 64 * units_count, optimization, iBatch))
         ReturnFalse;
     }
   caLocalAttention[0].Sparse(0.1f);
   caLocalAttention[1].Sparse(0.3f);
//---
   if(!cConcatenate.Init(0, 10, OpenCL, 128 * units_count, optimization, iBatch))
      ReturnFalse;
   if(!cScale.Init(0, 11, OpenCL, 128, 128, 64, units_count, 1, optimization, iBatch))
      ReturnFalse;
//---
   DeleteObj(cbTemp);
   cbTemp = new CBufferFloat();
   if(!cbTemp ||
      !cbTemp.BufferInit(caGlobalAttention[0].Neurons(), 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   CNeuronBaseOCL *inputs = NeuronOCL;
   for(int i = 0; i < 2; i++)
     {
      if(!cTNetG || i > 0)
        {
         if(!caLocalPointNet[i].FeedForward(inputs))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.FeedForward(inputs))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMul(NeuronOCL.getOutput(), cTNetG.getOutput(), cTurnedG.getOutput(), NeuronOCL.Neurons() / window, window, window))
            ReturnFalse;
         if(!caLocalPointNet[i].FeedForward(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Local Attention
      if(!caLocalAttention[i].FeedForward(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalPE[i].FeedForward(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!caGlobalPE[i].FeedForward(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalGlobalAttention[i].FeedForward(caLocalPE[i].AsObject(), caGlobalPE[i].getOutput()))
         ReturnFalse;
      //--- Global Attention
      if(!caGlobalAttention[i].FeedForward(caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      inputs = caGlobalAttention[i].AsObject();
     }
//---
   if(!Concat(caGlobalAttention[0].getOutput(), caGlobalAttention[1].getOutput(), cConcatenate.getOutput(), 64, 64, cConcatenate.Neurons() / 128))
      ReturnFalse;
   if(!cScale.FeedForward(cConcatenate.AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::feedForward(cScale.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPointNetOCL::calcInputGradients(cScale.AsObject()))
      ReturnFalse;
   if(!cConcatenate.CalcHiddenGradients(cScale.AsObject()))
      ReturnFalse;
   if(!DeConcat(caGlobalAttention[0].getGradient(), caGlobalAttention[1].getGradient(), cConcatenate.getGradient(), 64, 64, cConcatenate.Neurons() / 128))
      ReturnFalse;
//---
   CNeuronBaseOCL *inputs = caGlobalAttention[0].AsObject();
   for(int i = 1; i >= 0; i--)
     {
      //--- Global Attention
      if(!caLocalGlobalAttention[i].CalcHiddenGradients(caGlobalAttention[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalPE[i].CalcHiddenGradients(caLocalGlobalAttention[i].AsObject(), caGlobalPE[i].getOutput(),
                                           caGlobalPE[i].getGradient(),
                                           (ENUM_ACTIVATION)caGlobalPE[i].Activation()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalAttention[i].CalcHiddenGradients(caLocalPE[i].AsObject()))
         ReturnFalse;
      if(!caLocalPointNet[i].CalcHiddenGradients(caGlobalPE[i].AsObject()))
         ReturnFalse;
      //--- Local Attention
      CBufferFloat *temp = caLocalPointNet[i].getGradient();
      caLocalPointNet[i].SetGradient(cbTemp, false);
      cbTemp = temp;
      if(!caLocalPointNet[i].CalcHiddenGradients(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(caLocalPointNet[i].getGradient(), cbTemp, caLocalPointNet[i].getGradient(), 64, false, 0, 0, 0, 1))
         ReturnFalse;
      if(i > 0)
        {
         temp = inputs.getGradient();
         inputs.SetGradient(cbTemp, false);
         cbTemp = temp;
        }
      //--- Local Net
      if(!cTNetG || i > 0)
        {
         if(!inputs.CalcHiddenGradients(caLocalPointNet[i].AsObject()))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTurnedG.CalcHiddenGradients(caLocalPointNet[i].AsObject()))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMulGrad(inputs.getOutput(), inputs.getGradient(), cTNetG.getOutput(), cTNetG.getGradient(), cTurnedG.getGradient(), inputs.Neurons() / window, window, window))
            ReturnFalse;
         if(!OrthoganalLoss(cTNetG, true))
            ReturnFalse;
         //---
         CBufferFloat *temp = inputs.getGradient();
         inputs.SetGradient(cTurnedG.getGradient(), false);
         cTurnedG.SetGradient(temp, false);
         //---
         if(!inputs.CalcHiddenGradients(cTNetG.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(inputs.getGradient(), cTurnedG.getGradient(), inputs.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      if(i > 0)
         if(!SumAndNormalize(inputs.getGradient(), cbTemp, inputs.getGradient(), 64, false, 0, 0, 0, 1))
            ReturnFalse;
      inputs = NeuronOCL;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- LocalNet
   CNeuronBaseOCL *inputs = NeuronOCL;
   for(int i = 0; i < 2; i++)
     {
      if(!cTNetG || i > 0)
        {
         if(!caLocalPointNet[i].UpdateInputWeights(inputs))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.UpdateInputWeights(inputs))
            ReturnFalse;
         if(!caLocalPointNet[i].UpdateInputWeights(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Local Attention
      if(!caLocalAttention[i].UpdateInputWeights(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Position Encoder
      if(!caLocalPE[i].UpdateInputWeights(caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!caGlobalPE[i].UpdateInputWeights(caLocalPointNet[i].AsObject()))
         ReturnFalse;
      //--- Local to Global Attention
      if(!caLocalGlobalAttention[i].UpdateInputWeights(caLocalPE[i].AsObject(), caGlobalPE[i].getOutput()))
         ReturnFalse;
      //--- Global Attention
      if(!caGlobalAttention[i].UpdateInputWeights(caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      inputs = caGlobalAttention[i].AsObject();
     }
//---
   if(!cScale.UpdateInputWeights(cConcatenate.AsObject()))
      ReturnFalse;
   if(!CNeuronPointNetOCL::updateInputWeights(cScale.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::Save(const int file_handle)
  {
   if(!CNeuronPointNet2OCL::Save(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
     {
      if(!caLocalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caLocalGlobalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caGlobalAttention[i].Save(file_handle))
         ReturnFalse;
      if(!caLocalPE[i].Save(file_handle))
         ReturnFalse;
      if(!caGlobalPE[i].Save(file_handle))
         ReturnFalse;
     }
   if(!cConcatenate.Save(file_handle))
      ReturnFalse;
   if(!cScale.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHyperDet::Load(const int file_handle)
  {
   DeleteObj(cbTemp);
//---
   if(!CNeuronPointNet2OCL::Load(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
     {
      if(!LoadInsideLayer(file_handle, caLocalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caLocalGlobalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caGlobalAttention[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caLocalPE[i].AsObject()))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, caGlobalPE[i].AsObject()))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, cConcatenate.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScale.AsObject()))
      ReturnFalse;
//---
   cbTemp = new CBufferFloat();
   if(!cbTemp ||
      !cbTemp.BufferInit(caGlobalAttention[0].Neurons(), 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronHyperDet::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronPointNet2OCL::SetOpenCL(obj);
//---
   for(int i = 0; i < 2; i++)
     {
      caLocalAttention[i].SetOpenCL(OpenCL);
      caLocalGlobalAttention[i].SetOpenCL(OpenCL);
      caGlobalAttention[i].SetOpenCL(OpenCL);
      caLocalPE[i].SetOpenCL(OpenCL);
      caGlobalPE[i].SetOpenCL(OpenCL);
     }
   cConcatenate.SetOpenCL(OpenCL);
   cScale.SetOpenCL(OpenCL);
//---
   cbTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                           uint window, uint units_count, uint output, bool use_tnets,
                           uint center_points, uint center_window,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronPointNet2OCL::Init(numOutputs, myIndex, open_cl, window, units_count, output, use_tnets,
                                 optimization_type, batch))
      ReturnFalse;
//---
   iUnits = units_count;
   iPoints = MathMax(center_points, 9);
//---
   cQuery.SetOpenCL(OpenCL);
   cKey.SetOpenCL(OpenCL);
   cValue.SetOpenCL(OpenCL);
   cKeyValue.SetOpenCL(OpenCL);
   cMHAttentionOut.SetOpenCL(OpenCL);
   cAttentionOut.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cFeedForward.SetOpenCL(OpenCL);
   cCenterPoints.SetOpenCL(OpenCL);
   cFinalAttention.SetOpenCL(OpenCL);
//--- Init center points
   CNeuronBaseOCL *base = new CNeuronBaseOCL();
   if(!base)
      ReturnFalse;
   if(!base.Init(iPoints * center_window * 2, 0, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   CBufferFloat *buf = base.getOutput();
   if(!buf || !buf.BufferInit(1, 1) || !buf.BufferWrite())
      ReturnFalse;
   if(!cCenterPoints.Add(base))
      ReturnFalse;
   base = new CNeuronBaseOCL();
   if(!base.Init(0, 1, OpenCL, iPoints * center_window * 2, optimization, iBatch))
      ReturnFalse;
   if(!cCenterPoints.Add(base))
      ReturnFalse;
//--- Inside layers
   for(int i = 0; i < 2; i++)
     {
      //--- Interpolation
      CNeuronMVCrossAttentionMLKV *cross = new CNeuronMVCrossAttentionMLKV();
      if(!cross ||
         !cross.Init(0, i * 12 + 2, OpenCL, center_window, 32, 4, 64, 2, iPoints, iUnits, 2, 2, 2, 1, optimization, iBatch))
         ReturnFalse;
      if(!cCenterPoints.Add(cross))
         ReturnFalse;
      //--- Query
      CNeuronConvOCL *conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 3, OpenCL, 64, 64, 64, iUnits, optimization, iBatch))
         ReturnFalse;
      if(!cQuery.Add(conv))
         ReturnFalse;
      //--- Key
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 4, OpenCL, center_window, center_window, 32, iPoints, 2, optimization, iBatch))
         ReturnFalse;
      if(!cKey.Add(conv))
         ReturnFalse;
      //--- Value
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 5, OpenCL, center_window, center_window, 32, 1, iPoints * 2, optimization, iBatch))
         ReturnFalse;
      if(!cValue.Add(conv))
         ReturnFalse;
      //--- Key-Value
      base = new CNeuronBaseOCL();
      if(!base ||
         !base.Init(0, i * 12 + 6, OpenCL, iPoints * 2 * 32 * 2, optimization, iBatch))
         ReturnFalse;
      if(!cKeyValue.Add(base))
         ReturnFalse;
      //--- Score
      int s = int(iUnits * iPoints * 4);
      s = OpenCL.AddBuffer(sizeof(float) * s, CL_MEM_READ_WRITE);
      if(s < 0 || !cScores.Add(s))
         ReturnFalse;
      //--- MH Attention Out
      base = new CNeuronBaseOCL();
      if(!base ||
         !base.Init(0, i * 12 + 7, OpenCL, iUnits * 64, optimization, iBatch))
         ReturnFalse;
      if(!cMHAttentionOut.Add(base))
         ReturnFalse;
      //--- Attention Out
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 8, OpenCL, 64, 64, 64, iUnits, 1, optimization, iBatch))
         ReturnFalse;
      if(!cAttentionOut.Add(conv))
         ReturnFalse;
      //--- Residual
      base = new CNeuronBaseOCL();
      if(!base ||
         !base.Init(0, i * 12 + 9, OpenCL, iUnits * 64, optimization, iBatch))
         ReturnFalse;
      if(!cResidual.Add(base))
         ReturnFalse;
      //--- Feed Forward
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 10, OpenCL, 64, 64, 256, iUnits, 1, optimization, iBatch))
         ReturnFalse;
      conv.SetActivationFunction(LReLU);
      if(!cFeedForward.Add(conv))
         ReturnFalse;
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, i * 12 + 11, OpenCL, 256, 64, 64, iUnits, 1, optimization, iBatch))
         ReturnFalse;
      if(!cFeedForward.Add(conv))
         ReturnFalse;
      //--- Residual
      base = new CNeuronBaseOCL();
      if(!base ||
         !base.Init(0, i * 12 + 12, OpenCL, iUnits * 64, optimization, iBatch))
         ReturnFalse;
      if(!base.SetGradient(conv.getGradient(), true))
         ReturnFalse;
      if(!cResidual.Add(base))
         ReturnFalse;
      //--- Final Attention
      CNeuronMLMHSceneConditionAttention *att = new CNeuronMLMHSceneConditionAttention();
      if(!att ||
         !att.Init(0, i * 12 + 13, OpenCL, 64, 16, 4, 2, iUnits, 2, 1, optimization, iBatch))
         ReturnFalse;
      if(!cFinalAttention.Add(att))
         ReturnFalse;
     }
//---
   if(!SEOut.Init(0, 26, OpenCL, 64, 64, 4, 16, 4, iUnits, iUnits, 4, 1, optimization, iBatch))
      ReturnFalse;
//---
   uint buf_size = MathMax(MathMax(64, window) * iUnits, center_window * iPoints);
   if(!cbTemp.BufferInit(buf_size, 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::AttentionOut(CBufferFloat* q, CBufferFloat* kv, int scores, CBufferFloat* out)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits/*Q units*/, iPoints, 4};
   uint local_work_size[3] = {1, iPoints, 1};
   ResetLastError();
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_q, q.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_kv, kv.GetIndex())
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_score, scores)
   setBuffer(def_k_MH2AttentionOut, def_k_mh2ao_out, out.GetIndex())
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_dimension, 16)
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_heads_kv, (2 * 2))
   setArgument(def_k_MH2AttentionOut, def_k_mh2ao_mask, 0)
   kernelExecuteLoc(def_k_MH2AttentionOut, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::AttentionInsideGradients(CBufferFloat* q, CBufferFloat* qg,
      CBufferFloat* kv, CBufferFloat* kvg,
      int scores, CBufferFloat* outg
                                              )
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits, 16, 4};
   ResetLastError();
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_q, q.GetIndex())
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_qg, qg.GetIndex())
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kv, kv.GetIndex())
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kvg, kvg.GetIndex())
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_score, scores)
   setBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_outg, outg.GetIndex())
   setArgument(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kunits, (int)iPoints)
   setArgument(def_k_MH2AttentionInsideGradients, def_k_mh2aig_heads_kv, (2 * 2))
   kernelExecute(def_k_MH2AttentionInsideGradients, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!outg.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//---
   CNeuronBaseOCL *neuron = NULL, *q = NULL, *k = NULL, *v = NULL, *kv = NULL;
//--- Init Points
   if(bTrain)
     {
      neuron = cCenterPoints[1];
      if(!neuron ||
         !neuron.FeedForward(cCenterPoints[0]))
         ReturnFalse;
     }
//--- Inside Layers
   for(int l = 0; l < 2; l++)
     {
      //--- Segmentation Inputs
      if(l > 0 || !cTNetG)
        {
         if(!caLocalPointNet[l].FeedForward((l == 0 ? NeuronOCL : GetPointer(caLocalPointNet[l - 1]))))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.FeedForward(NeuronOCL))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMul(NeuronOCL.getOutput(), cTNetG.getOutput(), cTurnedG.getOutput(), NeuronOCL.Neurons() / window, window, window))
            ReturnFalse;
         if(!caLocalPointNet[0].FeedForward(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Interpolate center points
      neuron = cCenterPoints[l + 2];
      if(!neuron ||
         !neuron.FeedForward(cCenterPoints[l + 1], caLocalPointNet[l].getOutput()))
         ReturnFalse;
      //--- Structure-Embedding Attention
      q = cQuery[l];
      k = cKey[l];
      v = cValue[l];
      kv = cKeyValue[l];
      //--- Query
      if(!q || !q.FeedForward(GetPointer(caLocalPointNet[l])))
         ReturnFalse;
      //--- Key
      if(!k || !k.FeedForward(cCenterPoints[l + 2]))
         ReturnFalse;
      //--- Value
      if(!v || !v.FeedForward(cCenterPoints[l + 2]))
         ReturnFalse;
      if(!kv ||
         !Concat(k.getOutput(), v.getOutput(), kv.getOutput(), 32 * 2, 32 * 2, iPoints))
         ReturnFalse;
      //--- Multi-Head Attention
      neuron = cMHAttentionOut[l];
      if(!neuron ||
         !AttentionOut(q.getOutput(), kv.getOutput(), cScores[l], neuron.getOutput()))
         ReturnFalse;
      //--- Scale
      neuron = cAttentionOut[l];
      if(!neuron || !neuron.FeedForward(cMHAttentionOut[l]))
         ReturnFalse;
      //--- Residual
      q = cResidual[l * 2];
      if(!q ||
         !SumAndNormalize(caLocalPointNet[l].getOutput(), neuron.getOutput(), q.getOutput(), 64, true, 0, 0, 0, 1))
         ReturnFalse;
      //--- Feed Forward
      neuron = cFeedForward[l * 2];
      if(!neuron || !neuron.FeedForward(q))
         ReturnFalse;
      neuron = cFeedForward[l * 2 + 1];
      if(!neuron || !neuron.FeedForward(cFeedForward[l * 2]))
         ReturnFalse;
      //--- Residual
      k = cResidual[l * 2 + 1];
      if(!k ||
         !SumAndNormalize(q.getOutput(), neuron.getOutput(), k.getOutput(), 64, true, 0, 0, 0, 1))
         ReturnFalse;
      //--- Final Attention
      neuron = cFinalAttention[l];
      if(!neuron || !neuron.FeedForward(k))
         ReturnFalse;
     }
//--- Cross scale attention
   if(!SEOut.FeedForward(cFinalAttention[0], neuron.getOutput()))
      ReturnFalse;
//--- Global Point Cloud Embedding
   if(!CNeuronPointNetOCL::feedForward(SEOut.AsObject()))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL *neuron = NULL, *q = NULL, *k = NULL, *v = NULL, *kv = NULL;
   CBufferFloat *buf = NULL;
//--- Global Point Cloud Embedding
   if(!CNeuronPointNetOCL::calcInputGradients(SEOut.AsObject()))
      ReturnFalse;
//--- Cross scale attention
   neuron = cFinalAttention[0];
   q = cFinalAttention[1];
   if(!neuron.CalcHiddenGradients(SEOut.AsObject(), q.getOutput(), q.getGradient(), (ENUM_ACTIVATION)q.Activation()))
      ReturnFalse;
//--- Inside Layers
   for(int l = 1; l >= 0; l--)
     {
      //--- Final Attention
      neuron = cResidual[l * 2 + 1];
      if(!neuron || !neuron.CalcHiddenGradients(cFinalAttention[l]))
         ReturnFalse;
      //--- Feed Forward
      neuron = cFeedForward[l * 2];
      if(!neuron || !neuron.CalcHiddenGradients(cFeedForward[l * 2 + 1]))
         ReturnFalse;
      neuron = cResidual[l * 2];
      if(!neuron || !neuron.CalcHiddenGradients(cFeedForward[l * 2]))
         ReturnFalse;
      //--- Residual
      q = cResidual[l * 2 + 1];
      k = neuron;
      neuron = cAttentionOut[l];
      if(!neuron ||
         !SumAndNormalize(q.getGradient(), k.getGradient(), neuron.getGradient(), 64, false, 0, 0, 0, 1))
         ReturnFalse;
      //--- Scale
      neuron = cMHAttentionOut[l];
      if(!neuron || !neuron.CalcHiddenGradients(cAttentionOut[l]))
         ReturnFalse;
      //--- MH Attention
      q = cQuery[l];
      kv = cKeyValue[l];
      k = cKey[l];
      v = cValue[l];
      if(!AttentionInsideGradients(q.getOutput(), q.getGradient(), kv.getOutput(), kv.getGradient(), cScores[l], neuron.getGradient()))
         ReturnFalse;
      if(!DeConcat(k.getGradient(), v.getGradient(), kv.getGradient(), 32 * 2, 32 * 2, iPoints))
         ReturnFalse;
      if(l == 0)
        {
         buf = caLocalPointNet[l].getGradient();
         if(!caLocalPointNet[l].SetGradient(GetPointer(cbTemp), false))
            ReturnFalse;
        }
      if(!caLocalPointNet[l].CalcHiddenGradients(q, NULL))
         ReturnFalse;
      if(l == 0)
        {
         if(!SumAndNormalize(buf, GetPointer(cbTemp), buf, 64, false, 0, 0, 0, 1))
            ReturnFalse;
         if(!caLocalPointNet[l].SetGradient(buf, false))
            ReturnFalse;
        }
      neuron = cAttentionOut[l];
      //--- Residual
      if(!SumAndNormalize(caLocalPointNet[l].getGradient(), neuron.getGradient(), caLocalPointNet[l].getGradient(), 64, false, 0, 0, 0, 1))
         ReturnFalse;
      //--- Interpolate Center points
      neuron = cCenterPoints[l + 2];
      if(!neuron)
         ReturnFalse;
      buf = neuron.getGradient();
      if(!neuron.SetGradient(GetPointer(cbTemp), false))
         ReturnFalse;
      if(!neuron.CalcHiddenGradients(k, NULL))
         ReturnFalse;
      if(l == 0)
        {
         if(!SumAndNormalize(buf, GetPointer(cbTemp), buf, 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      else
        {
         if(!SumAndNormalize(GetPointer(cbTemp), GetPointer(cbTemp), buf, 1, false, 0, 0, 0, 0.5f))
            ReturnFalse;
        }
      if(!neuron.CalcHiddenGradients(v, NULL))
         ReturnFalse;
      if(!SumAndNormalize(buf, GetPointer(cbTemp), buf, 1, false, 0, 0, 0, 1))
         ReturnFalse;
      if(!neuron.SetGradient(buf, false))
         ReturnFalse;
      neuron = cCenterPoints[l + 1];
      if(!neuron.CalcHiddenGradients(cCenterPoints[l + 2], caLocalPointNet[l].getOutput(), GetPointer(cbTemp), (ENUM_ACTIVATION)caLocalPointNet[l].Activation()))
         ReturnFalse;
      if(!SumAndNormalize(caLocalPointNet[l].getGradient(), GetPointer(cbTemp), caLocalPointNet[l].getGradient(), 64, false, 0, 0, 0, 1))
         ReturnFalse;
      //--- Local Net
      neuron = (l > 0 ? GetPointer(caLocalPointNet[l - 1]) : NeuronOCL);
      if(l > 0 || !cTNetG)
        {
         if(!neuron.CalcHiddenGradients(caLocalPointNet[l].AsObject()))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTurnedG.CalcHiddenGradients(caLocalPointNet[l].AsObject()))
            ReturnFalse;
         int window = (int)MathSqrt(cTNetG.Neurons());
         if(IsStopped() ||
            !MatMulGrad(neuron.getOutput(), neuron.getGradient(), cTNetG.getOutput(), cTNetG.getGradient(), cTurnedG.getGradient(), neuron.Neurons() / window, window, window))
            ReturnFalse;
         if(!OrthoganalLoss(cTNetG, true))
            ReturnFalse;
         //---
         CBufferFloat *temp = neuron.getGradient();
         neuron.SetGradient(cTurnedG.getGradient(), false);
         cTurnedG.SetGradient(temp, false);
         //---
         if(!neuron.CalcHiddenGradients(cTNetG.AsObject()))
            ReturnFalse;
         if(!SumAndNormalize(neuron.getGradient(), cTurnedG.getGradient(), neuron.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//---
   CNeuronBaseOCL *q = NULL, *k = NULL, *v = NULL, *kv = NULL;
//--- Init Points
   CNeuronBaseOCL *neuron = cCenterPoints[1];
   if(!neuron ||
      !neuron.UpdateInputWeights(cCenterPoints[0]))
      ReturnFalse;
//--- Inside Layers
   for(int l = 0; l < 2; l++)
     {
      //--- Segmentation Inputs
      if(l > 0 || !cTNetG)
        {
         if(!caLocalPointNet[l].UpdateInputWeights((l == 0 ? NeuronOCL : GetPointer(caLocalPointNet[l - 1]))))
            ReturnFalse;
        }
      else
        {
         if(!cTurnedG)
            ReturnFalse;
         if(!cTNetG.UpdateInputWeights(NeuronOCL))
            ReturnFalse;
         if(!caLocalPointNet[0].UpdateInputWeights(cTurnedG.AsObject()))
            ReturnFalse;
        }
      //--- Interpolate center points
      neuron = cCenterPoints[l + 2];
      if(!neuron ||
         !neuron.UpdateInputWeights(cCenterPoints[l + 1], caLocalPointNet[l].getOutput()))
         ReturnFalse;
      //--- Structure-Embedding Attention
      q = cQuery[l];
      k = cKey[l];
      v = cValue[l];
      //--- Query
      if(!q || !q.UpdateInputWeights(GetPointer(caLocalPointNet[l])))
         ReturnFalse;
      //--- Key
      if(!k || !k.UpdateInputWeights(cCenterPoints[l + 2]))
         ReturnFalse;
      //--- Value
      if(!v || !v.UpdateInputWeights(cCenterPoints[l + 2]))
         ReturnFalse;
      //--- Scale
      neuron = cAttentionOut[l];
      if(!neuron || !neuron.UpdateInputWeights(cMHAttentionOut[l]))
         ReturnFalse;
      //--- Feed Forward
      neuron = cFeedForward[l * 2];
      if(!neuron || !neuron.UpdateInputWeights(cResidual[l * 2]))
         ReturnFalse;
      neuron = cFeedForward[l * 2 + 1];
      if(!neuron || !neuron.UpdateInputWeights(cFeedForward[l * 2]))
         ReturnFalse;
      //--- Final Attention
      neuron = cFinalAttention[l];
      if(!neuron || !neuron.UpdateInputWeights(cResidual[l * 2 + 1]))
         ReturnFalse;
     }
//--- Cross scale attention
   if(!SEOut.UpdateInputWeights(cFinalAttention[0], neuron.getOutput()))
      ReturnFalse;
//--- Global Point Cloud Embedding
   if(!CNeuronPointNetOCL::updateInputWeights(SEOut.AsObject()))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::Save(const int file_handle)
  {
   if(!CNeuronPointNet2OCL::Save(file_handle))
      ReturnFalse;
//--- Save constants
   if(FileWriteInteger(file_handle, (int)iUnits) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iPoints) < INT_VALUE)
      ReturnFalse;
//--- Save objects
   if(!cQuery.Save(file_handle))
      ReturnFalse;
   if(!cKey.Save(file_handle))
      ReturnFalse;
   if(!cValue.Save(file_handle))
      ReturnFalse;
   if(!cKeyValue.Save(file_handle))
      ReturnFalse;
   if(!cMHAttentionOut.Save(file_handle))
      ReturnFalse;
   if(!cAttentionOut.Save(file_handle))
      ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
   if(!cFeedForward.Save(file_handle))
      ReturnFalse;
   if(!cCenterPoints.Save(file_handle))
      ReturnFalse;
   if(!cFinalAttention.Save(file_handle))
      ReturnFalse;
   if(!SEOut.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSEFormer::Load(const int file_handle)
  {
   if(!CNeuronPointNet2OCL::Load(file_handle))
      ReturnFalse;
//--- Save constants
   iUnits = (uint)FileReadInteger(file_handle);
   iPoints = (uint)FileReadInteger(file_handle);
//--- Save objects
   if(!cQuery.Load(file_handle))
      ReturnFalse;
   if(!cKey.Load(file_handle))
      ReturnFalse;
   if(!cValue.Load(file_handle))
      ReturnFalse;
   if(!cKeyValue.Load(file_handle))
      ReturnFalse;
   if(!cMHAttentionOut.Load(file_handle))
      ReturnFalse;
   if(!cAttentionOut.Load(file_handle))
      ReturnFalse;
   if(!cResidual.Load(file_handle))
      ReturnFalse;
   if(!cFeedForward.Load(file_handle))
      ReturnFalse;
   if(!cCenterPoints.Load(file_handle))
      ReturnFalse;
   if(!cFinalAttention.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, SEOut.AsObject()))
      ReturnFalse;
//--- Score
   int s = int(iUnits * iPoints * 4);
   for(int i = 0; i < 2; i++)
     {
      OpenCL.BufferFree(cScores[i]);
      if(!cScores.Update(i, OpenCL.AddBuffer(sizeof(float) * s, CL_MEM_READ_WRITE)) ||
         cScores[i] < 0)
         ReturnFalse;
     }
//---
   CNeuronBaseOCL *ff = NULL, *residual = NULL;
   for(int i = 1; i < cFeedForward.Total(); i += 2)
     {
      ff = cFeedForward[i];
      residual = cResidual[i];
      if(!ff || !residual ||
         !residual.SetGradient(ff.getGradient()))
         ReturnFalse;
     }
//---
   int buf_size = MathMax(((CNeuronBaseOCL*)cCenterPoints[2]).Neurons(), caLocalPointNet[0].Neurons());
   cbTemp.BufferFree();
   if(!cbTemp.BufferInit(buf_size, 0) ||
      !cbTemp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSEFormer::SetOpenCL(COpenCLMy * obj)
  {
   bool create_score = false;
   if(!!OpenCL && obj != OpenCL)
     {
      for(int i = 0; i < cScores.Total(); i++)
         OpenCL.BufferFree(cScores[i]);
      create_score = true;
     }
//---
   CNeuronPointNet2OCL::SetOpenCL(obj);
   cQuery.SetOpenCL(OpenCL);
   cKey.SetOpenCL(OpenCL);
   cValue.SetOpenCL(OpenCL);
   cKeyValue.SetOpenCL(OpenCL);
   cMHAttentionOut.SetOpenCL(OpenCL);
   cAttentionOut.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cFeedForward.SetOpenCL(OpenCL);
   cCenterPoints.SetOpenCL(OpenCL);
   cFinalAttention.SetOpenCL(OpenCL);
   SEOut.SetOpenCL(OpenCL);
//--- Score
   int s = int(iUnits * iPoints * 4);
   for(int i = 0; i < 2 && create_score; i++)
      cScores.Update(i, OpenCL.AddBuffer(sizeof(float) * s, CL_MEM_READ_WRITE));
//---
   cbTemp.BufferFree();
   cbTemp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                       uint window, uint window_key, uint units_count,
                                       uint heads, uint layers, uint neurons_out, uint filters,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, neurons_out, optimization_type, batch))
      ReturnFalse;
//--- Decoupling Flow
   uint index = 0;
   uint wave_window = MathMin(24, units_count);
   if(!cDecouplingFlow.Init(0, index, OpenCL, wave_window, 2, units_count, filters, window, optimization, iBatch))
      ReturnFalse;
   cDecouplingFlow.SetActivationFunction(None);
//--- Dual-Frequency Spatiotemporal Encoder
   uint wave_units_out = cDecouplingFlow.GetUnits();
   index++;
   if(!cLowFreqSignal.Init(0, index, OpenCL, cDecouplingFlow.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cLowFreqSignal.SetActivationFunction(None);
   index++;
   if(!cHighFreqSignal.Init(0, index, OpenCL, cDecouplingFlow.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cHighFreqSignal.SetActivationFunction(None);
   index++;
   if(!cTemporalAttention.Init(0, index, OpenCL, filters, window_key, wave_units_out * window, heads, layers, optimization, iBatch))
      ReturnFalse;
   cTemporalAttention.SetActivationFunction(None);
   index++;
   if(!cDilatedCasualConvolution.Init(0, index, OpenCL, 2, 2, filters, wave_units_out, window, layers, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowFreqPE.Init(0, index, OpenCL, cTemporalAttention.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHighFreqPE.Init(0, index, OpenCL, cDilatedCasualConvolution.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowFreqGraphAttention.Init(0, index, OpenCL, filters, 3, wave_units_out * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHighFreqGraphAttention.Init(0, index, OpenCL, filters, 3, cDilatedCasualConvolution.Neurons() / filters, optimization, iBatch))
      ReturnFalse;
   index++;
//--- Dual-Frequency Fusion Decoder
   if(!cLowFreqFusionDecoder.Init(0, index, OpenCL, filters, window_key, wave_units_out * window, heads, layers, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowHighFreqFusionDecoder.Init(0, index, OpenCL, filters, window_key, wave_units_out * window, filters, cDilatedCasualConvolution.Neurons() / filters, heads, layers, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLowHigh.Init(0, index, OpenCL, cLowFreqFusionDecoder.Neurons(), optimization, iBatch))
      ReturnFalse;
   CBufferFloat *grad = cLowFreqFusionDecoder.getGradient();
   if(!grad ||
      !cLowHigh.SetGradient(grad, true) ||
      !cLowHighFreqFusionDecoder.SetGradient(grad, true))
      ReturnFalse;
   index++;
   if(!cProjection.Init(Neurons(), index, OpenCL, filters, filters, 3, wave_units_out, window, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Decoupling Flow
   if(!cDecouplingFlow.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!DeConcat(cLowFreqSignal.getOutput(), cHighFreqSignal.getOutput(), cDecouplingFlow.getOutput(), cDecouplingFlow.GetFilters(), cDecouplingFlow.GetFilters(), cDecouplingFlow.GetUnits()*cDecouplingFlow.GetVariables()))
      ReturnFalse;
//--- Dual-Frequency Spatiotemporal Encoder
//--- Low Frequency Encoder
   if(!cTemporalAttention.FeedForward(cLowFreqSignal.AsObject()))
      ReturnFalse;
   if(!cLowFreqPE.FeedForward(cTemporalAttention.AsObject()))
      ReturnFalse;
   if(!cLowFreqGraphAttention.FeedForward(cLowFreqPE.AsObject()))
      ReturnFalse;
//--- High Frequency Encoder
   if(!cDilatedCasualConvolution.FeedForward(cHighFreqSignal.AsObject()))
      ReturnFalse;
   if(!cHighFreqPE.FeedForward(cDilatedCasualConvolution.AsObject()))
      ReturnFalse;
   if(!cHighFreqGraphAttention.FeedForward(cHighFreqPE.AsObject()))
      ReturnFalse;
//--- Dual-Frequency Fusion Decoder
   if(!cLowFreqFusionDecoder.FeedForward(cLowFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!cLowHighFreqFusionDecoder.FeedForward(cLowFreqGraphAttention.AsObject(), cHighFreqGraphAttention.getOutput()))
      ReturnFalse;
   if(!SumAndNormalize(cLowFreqFusionDecoder.getOutput(), cLowHighFreqFusionDecoder.getOutput(), cLowHigh.getOutput(), cLowFreqFusionDecoder.GetWindow(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cProjection.FeedForward(cLowHigh.AsObject()))
      ReturnFalse;
//---
   return CNeuronBaseOCL::feedForward(cProjection.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::calcInputGradients(cProjection.AsObject()))
      ReturnFalse;
   if(!cLowHigh.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
//--- Dual-Frequency Fusion Decoder
   if(!cLowFreqGraphAttention.CalcHiddenGradients(cLowFreqFusionDecoder.AsObject()))
      ReturnFalse;
   CBufferFloat *grad = cLowFreqGraphAttention.getGradient();
   if(!cLowFreqGraphAttention.SetGradient(cLowFreqGraphAttention.getPrevOutput(), false) ||
      !cLowFreqGraphAttention.CalcHiddenGradients(cLowHighFreqFusionDecoder.AsObject(),
            cHighFreqGraphAttention.getOutput(),
            cHighFreqGraphAttention.getGradient(),
            (ENUM_ACTIVATION)cHighFreqGraphAttention.Activation()) ||
      !SumAndNormalize(grad, cLowFreqGraphAttention.getGradient(), grad, 1, false, 0, 0, 0, 1) ||
      !cLowFreqGraphAttention.SetGradient(grad, false))
      ReturnFalse;
//--- Dual-Frequency Spatiotemporal Encoder
//--- Low Frequency Encoder
   if(!cLowFreqPE.CalcHiddenGradients(cLowFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!cTemporalAttention.CalcHiddenGradients(cLowFreqPE.AsObject()))
      ReturnFalse;
   if(!cLowFreqSignal.CalcHiddenGradients(cTemporalAttention.AsObject()))
      ReturnFalse;
//--- High Frequency Encoder
   if(!cHighFreqPE.CalcHiddenGradients(cHighFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!cDilatedCasualConvolution.CalcHiddenGradients(cHighFreqPE.AsObject()))
      ReturnFalse;
   if(!cHighFreqSignal.CalcHiddenGradients(cDilatedCasualConvolution.AsObject()))
      ReturnFalse;
//--- Decoupling Flow
   if(!Concat(cLowFreqSignal.getGradient(), cHighFreqSignal.getGradient(),
              cDecouplingFlow.getGradient(), cDecouplingFlow.GetFilters(),
              cDecouplingFlow.GetFilters(), cDecouplingFlow.GetUnits()*cDecouplingFlow.GetVariables()))
      ReturnFalse;
   if(!prevLayer.CalcHiddenGradients(cDecouplingFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//---
   if(!CNeuronBaseOCL::updateInputWeights(cProjection.AsObject()))
      ReturnFalse;
   if(!cProjection.UpdateInputWeights(cLowHigh.AsObject()))
      ReturnFalse;
//--- Dual-Frequency Fusion Decoder
   if(!cLowFreqFusionDecoder.UpdateInputWeights(cLowFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!cLowHighFreqFusionDecoder.UpdateInputWeights(cLowFreqGraphAttention.AsObject(),
         cHighFreqGraphAttention.getOutput()))
      ReturnFalse;
//--- Dual-Frequency Spatiotemporal Encoder
//--- Low Frequency Encoder
   if(!cLowFreqPE.UpdateInputWeights(cTemporalAttention.AsObject()))
      ReturnFalse;
   if(!cTemporalAttention.UpdateInputWeights(cLowFreqSignal.AsObject()))
      ReturnFalse;
//--- High Frequency Encoder
   if(!cHighFreqPE.UpdateInputWeights(cDilatedCasualConvolution.AsObject()))
      ReturnFalse;
   if(!cDilatedCasualConvolution.UpdateInputWeights(cHighFreqSignal.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cDecouplingFlow.Save(file_handle))
      ReturnFalse;
   if(!cLowFreqSignal.Save(file_handle))
      ReturnFalse;
   if(!cHighFreqSignal.Save(file_handle))
      ReturnFalse;
   if(!cTemporalAttention.Save(file_handle))
      ReturnFalse;
   if(!cDilatedCasualConvolution.Save(file_handle))
      ReturnFalse;
   if(!cLowFreqPE.Save(file_handle))
      ReturnFalse;
   if(!cHighFreqPE.Save(file_handle))
      ReturnFalse;
   if(!cLowFreqGraphAttention.Save(file_handle))
      ReturnFalse;
   if(!cHighFreqGraphAttention.Save(file_handle))
      ReturnFalse;
   if(!cLowFreqFusionDecoder.Save(file_handle))
      ReturnFalse;
   if(!cLowHighFreqFusionDecoder.Save(file_handle))
      ReturnFalse;
   if(!cLowHigh.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultitaskStockformer::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cDecouplingFlow.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowFreqSignal.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHighFreqSignal.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTemporalAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDilatedCasualConvolution.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowFreqPE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHighFreqPE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHighFreqGraphAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowFreqFusionDecoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowHighFreqFusionDecoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLowHigh.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
//---
   CBufferFloat *grad = cLowFreqFusionDecoder.getGradient();
   if(!grad ||
      !cLowHigh.SetGradient(grad, true) ||
      !cLowHighFreqFusionDecoder.SetGradient(grad, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultitaskStockformer::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cDecouplingFlow.SetOpenCL(OpenCL);
   cLowFreqSignal.SetOpenCL(OpenCL);
   cHighFreqSignal.SetOpenCL(OpenCL);
   cTemporalAttention.SetOpenCL(OpenCL);
   cDilatedCasualConvolution.SetOpenCL(OpenCL);
   cLowFreqPE.SetOpenCL(OpenCL);
   cHighFreqPE.SetOpenCL(OpenCL);
   cLowFreqGraphAttention.SetOpenCL(OpenCL);
   cHighFreqGraphAttention.SetOpenCL(OpenCL);
   cLowFreqFusionDecoder.SetOpenCL(OpenCL);
   cLowHighFreqFusionDecoder.SetOpenCL(OpenCL);
   cLowHigh.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
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
   uint half = (caTSAgents.Size() + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      index++;
      if(!caTSAgents[i].Init(0, index, OpenCL, window, window_key, units_count, heads,
                             stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   for(uint i = half; i < caTSAgents.Size(); i++)
     {
      index++;
      if(!caTSAgents[i].Init(0, index, OpenCL, units_count, window_key, window, heads,
                             stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   half = (caFreqAgents.Size() + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      index++;
      if(!caFreqAgents[i].Init(0, index, OpenCL, window, window_key, units_count, heads,
                               stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   for(uint i = half; i < caFreqAgents.Size(); i++)
     {
      index++;
      if(!caFreqAgents[i].Init(0, index, OpenCL, units_count, window_key, window, heads,
                               stack_size, nactions, optimization, iBatch))
         ReturnFalse;
     }
   index++;
   if(!cHyperAgent.Init(0, index, OpenCL, window, window_key, units_count, heads, layers,
                        caTSAgents.Size() + caFreqAgents.Size(), stack_size, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, (caTSAgents.Size() + caFreqAgents.Size())*nactions, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint total_ts = caTSAgents.Size();
   uint half = (total_ts + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      if(!caTSAgents[i].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   for(uint i = half; i < total_ts; i++)
     {
      if(!caTSAgents[i].FeedForward(cTranspose.AsObject()))
         ReturnFalse;
     }
//---
   uint total_fr = caFreqAgents.Size();
   half = (total_fr + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      if(!caFreqAgents[i].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   for(uint i = half; i < total_fr; i++)
     {
      if(!caFreqAgents[i].FeedForward(cTranspose.AsObject()))
         ReturnFalse;
     }
   if(!cHyperAgent.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!Concat(caTSAgents[0].getOutput(), caTSAgents[1].getOutput(), caTSAgents[2].getOutput(),
              caTSAgents[3].getOutput(), cConcatenated.getPrevOutput(), Neurons(), Neurons(),
              Neurons(), Neurons(), 1) ||
      !Concat(cConcatenated.getPrevOutput(), caFreqAgents[0].getOutput(), caFreqAgents[1].getOutput(),
              cConcatenated.getOutput(), 4 * Neurons(), Neurons(), Neurons(), 1))
      ReturnFalse;
   if(!MatMul(cHyperAgent.getOutput(), cConcatenated.getOutput(), Output, 1, total_fr + total_ts, Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint total_ts = caTSAgents.Size();
   uint total_fr = caFreqAgents.Size();
   if(!MatMulGrad(cHyperAgent.getOutput(), cHyperAgent.getGradient(), cConcatenated.getOutput(), cConcatenated.getGradient(), Gradient, 1, total_ts + total_fr, Neurons(), 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cHyperAgent.AsObject()))
      ReturnFalse;
//---
   if(!DeConcat(cConcatenated.getPrevOutput(), caFreqAgents[0].getGradient(), caFreqAgents[1].getGradient(),
                cConcatenated.getGradient(), 4 * Neurons(), Neurons(), Neurons(), 1) ||
      !DeConcat(caTSAgents[0].getGradient(), caTSAgents[1].getGradient(), caTSAgents[2].getGradient(),
                caTSAgents[3].getGradient(), cConcatenated.getPrevOutput(), Neurons(), Neurons(),
                Neurons(), Neurons(), 1))
      ReturnFalse;
//---
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!temp ||
      !NeuronOCL.SetGradient(cTranspose.getPrevOutput(), false))
      ReturnFalse;
   uint half = (total_ts + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      if(!NeuronOCL.CalcHiddenGradients(caTSAgents[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   for(uint i = half; i < total_ts; i++)
     {
      if(!cTranspose.CalcHiddenGradients(caTSAgents[i].AsObject()) ||
         !NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   half = (total_fr + 1) / 2;
   for(uint i = 0; i < half; i++)
     {
      if(!NeuronOCL.CalcHiddenGradients(caFreqAgents[i].AsObject()))
         ReturnFalse;
      if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   for(uint i = half; i < total_fr; i++)
     {
      if(!cTranspose.CalcHiddenGradients(caFreqAgents[i].AsObject()) ||
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cHyperAgent.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   uint total = caTSAgents.Size();
   uint half = (total + 1) / 2;
   for(uint i = 0; i < half; i++)
      if(!caTSAgents[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
   for(uint i = half; i < total; i++)
      if(!caTSAgents[i].UpdateInputWeights(cTranspose.AsObject()))
         ReturnFalse;
//---
   total = caFreqAgents.Size();
   half = (total + 1) / 2;
   for(uint i = 0; i < half; i++)
      if(!caFreqAgents[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
   for(uint i = half; i < total; i++)
      if(!caFreqAgents[i].UpdateInputWeights(cTranspose.AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caTSAgents.Size(); i++)
      if(!caTSAgents[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caFreqAgents.Size(); i++)
      if(!caFreqAgents[i].Save(file_handle))
         ReturnFalse;
   if(!cHyperAgent.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caTSAgents.Size(); i++)
      if(!LoadInsideLayer(file_handle, caTSAgents[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caFreqAgents.Size(); i++)
      if(!LoadInsideLayer(file_handle, caFreqAgents[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHyperAgent.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.Init(0, 8, OpenCL, (caTSAgents.Size() + caFreqAgents.Size())*Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronHidformer::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cTranspose.SetOpenCL(obj);
   for(uint i = 0; i < caTSAgents.Size(); i++)
      caTSAgents[i].SetOpenCL(obj);
   for(uint i = 0; i < caFreqAgents.Size(); i++)
      caFreqAgents[i].SetOpenCL(obj);
   cHyperAgent.SetOpenCL(obj);
   cConcatenated.SetOpenCL(obj);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHidformer::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cTranspose.Clear())
      ReturnFalse;
   for(uint i = 0; i < caTSAgents.Size(); i++)
      if(!caTSAgents[i].Clear())
         ReturnFalse;
   for(uint i = 0; i < caFreqAgents.Size(); i++)
      if(!caFreqAgents[i].Clear())
         ReturnFalse;
   if(!cHyperAgent.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                          uint window_in, uint window_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_out * window_out, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   int index = 0;
   for(int i = 0; i < 2; i++)
     {
      if(!caSSM[i].Init(0, index, OpenCL, window_in, (i + 1)*window_out, units_in, units_out, optimization, iBatch))
         ReturnFalse;
      index++;
     }
   if(!cDiscretization.Init(0, index, OpenCL, 2 * window_out, 2 * window_out, window_out, units_out, 1, optimization, iBatch))
      ReturnFalse;
   cDiscretization.SetActivationFunction(None);
//--- Residual
   cResidual.Clear();
   cResidual.SetOpenCL(OpenCL);
   CNeuronConvOCL *conv = NULL;
   CNeuronTransposeOCL *transp = NULL;
//---
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, window_in, optimization, iBatch) ||
      !cResidual.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, units_out, window_in, 1, optimization, iBatch) ||
      !cResidual.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
//---
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, window_in, units_out, optimization, iBatch) ||
      !cResidual.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_in, window_in, window_out, units_out, 1, optimization, iBatch) ||
      !cResidual.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
//---
   if(!SetGradient(caSSM[0].getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   for(uint i = 0; i < caSSM.Size(); i++)
     {
      if(!caSSM[i].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   if(!cDiscretization.FeedForward(caSSM[1].AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < cResidual.Total(); i++)
     {
      current = cResidual[i];
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
      inp = current;
     }
//---
   if(!SumAndNormalize(caSSM[0].getOutput(), cDiscretization.getOutput(), Output, 1, false, 0, 0, 0, 1) ||
      !SumAndNormalize(Output, current.getOutput(), Output, cDiscretization.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!DeActivation(cDiscretization.getOutput(), cDiscretization.getGradient(), Gradient, cDiscretization.Activation()))
      ReturnFalse;
   if(!caSSM[1].CalcHiddenGradients(cDiscretization.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL *residual = cResidual[-1];
   if(!residual)
      ReturnFalse;
   if(!DeActivation(residual.getOutput(), residual.getGradient(), Gradient, residual.Activation()))
      ReturnFalse;
   for(int i = cResidual.Total() - 2; i >= 0; i--)
     {
      residual = cResidual[i];
      if(!residual ||
         !residual.CalcHiddenGradients(cResidual[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(caSSM[0].AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(residual.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(caSSM[1].AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients((CObject*)residual) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false)
     )
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cDiscretization.UpdateInputWeights(caSSM[1].AsObject()))
      ReturnFalse;
//---
   for(uint i = 0; i < caSSM.Size(); i++)
     {
      if(!caSSM[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
     }
//---
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < cResidual.Total(); i++)
     {
      current = cResidual[i];
      if(!current ||
         !current.UpdateInputWeights(inp))
         ReturnFalse;
      inp = current;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caSSM.Size(); i++)
      if(!caSSM[i].Save(file_handle))
         ReturnFalse;
   if(!cDiscretization.Save(file_handle))
      ReturnFalse;
//---
   if(!cResidual.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caSSM.Size(); i++)
      if(!LoadInsideLayer(file_handle, caSSM[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDiscretization.AsObject()))
      ReturnFalse;
//---
   if(!cResidual.Load(file_handle))
      ReturnFalse;
//---
   if(!SetGradient(caSSM[0].getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronChimera::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(uint i = 0; i < caSSM.Size(); i++)
      caSSM[i].SetOpenCL(OpenCL);
   cDiscretization.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimera::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   for(uint i = 0; i < caSSM.Size(); i++)
      if(!caSSM[i].Clear())
         ReturnFalse;
   if(!cDiscretization.Clear())
      ReturnFalse;
//---
   for(int i = 0; i < cResidual.Total(); i++)
      if(!((CNeuronBaseOCL*)cResidual[i]).Clear())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                      uint window, uint units_count,
                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMHAttentionPooling::Init(numOutputs, myIndex, open_cl, window, units_count, 4, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cNodesTokenizer.Init(0, index, OpenCL, iWindow, iWindow, iWindow, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   cNodesTokenizer.SetActivationFunction(SoftPlus);
//---
   index++;
   if(!cEdgesTokenizer.Init(0, index, OpenCL, 2 * iWindow, iWindow, iWindow, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   cEdgesTokenizer.SetActivationFunction(SoftPlus);
//---
   index++;
   if(!cSubGraphsTokenizer.Init(0, index, OpenCL, 3 * iWindow, iWindow, iWindow, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   cSubGraphsTokenizer.SetActivationFunction(SoftPlus);
//---
   cUnitarSubGraphsTokenizer.Clear();
   cUnitarSubGraphsTokenizer.SetOpenCL(OpenCL);
   CNeuronConvOCL *conv = NULL;
   CNeuronTransposeOCL *transp = NULL;
//---
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, iUnits, iWindow, optimization, iBatch) ||
      !cUnitarSubGraphsTokenizer.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 3, 1, 1, iUnits, iWindow, optimization, iBatch) ||
      !cUnitarSubGraphsTokenizer.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
//---
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, iWindow, iUnits, optimization, iBatch) ||
      !cUnitarSubGraphsTokenizer.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
//---
   index++;
   if(!cConcatenate.Init(0, index, OpenCL, 4 * iWindow * iUnits, optimization, iBatch))
      ReturnFalse;
   cConcatenate.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cNodesTokenizer.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cEdgesTokenizer.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSubGraphsTokenizer.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL *prev = NeuronOCL, *current = NULL;
   for(int i = 0; i < cUnitarSubGraphsTokenizer.Total(); i++)
     {
      current = cUnitarSubGraphsTokenizer[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!Concat(cNodesTokenizer.getOutputIndex(), cEdgesTokenizer.getOutputIndex(),
              cSubGraphsTokenizer.getOutputIndex(), current.getOutputIndex(),
              cConcatenate.getOutputIndex(), iWindow, iWindow, iWindow, iWindow, iUnits))
      ReturnFalse;
//---
   return CNeuronMHAttentionPooling::feedForward(cConcatenate.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronMHAttentionPooling::calcInputGradients(cConcatenate.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL *current = cUnitarSubGraphsTokenizer[-1];
   if(!current ||
      !DeConcat(cNodesTokenizer.getGradient(), cEdgesTokenizer.getGradient(),
                cSubGraphsTokenizer.getGradient(), current.getGradient(),
                cConcatenate.getGradient(), iWindow, iWindow, iWindow, iWindow, iUnits))
      ReturnFalse;
//---
   if(current.Activation() != None &&
      !DeActivation(current.getOutput(), current.getGradient(),
                    current.getGradient(), current.Activation()))
      ReturnFalse;
   for(int i = cUnitarSubGraphsTokenizer.Total() - 2; i >= 0; i--)
     {
      current = cUnitarSubGraphsTokenizer[i];
      if(!current ||
         !current.CalcHiddenGradients(cUnitarSubGraphsTokenizer[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(current.AsObject()))
      ReturnFalse;
//---
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(current.getPrevOutput(), false))
      ReturnFalse;
//---
   if(cNodesTokenizer.Activation() != None &&
      !DeActivation(cNodesTokenizer.getOutput(), cNodesTokenizer.getGradient(),
                    cNodesTokenizer.getGradient(), cNodesTokenizer.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cNodesTokenizer.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(cEdgesTokenizer.Activation() != None &&
      !DeActivation(cEdgesTokenizer.getOutput(), cEdgesTokenizer.getGradient(),
                    cEdgesTokenizer.getGradient(), cEdgesTokenizer.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cEdgesTokenizer.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(cSubGraphsTokenizer.Activation() != None &&
      !DeActivation(cSubGraphsTokenizer.getOutput(), cSubGraphsTokenizer.getGradient(),
                    cSubGraphsTokenizer.getGradient(), cSubGraphsTokenizer.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSubGraphsTokenizer.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronMHAttentionPooling::updateInputWeights(cConcatenate.AsObject()))
      ReturnFalse;
//---
   if(!cNodesTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cEdgesTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSubGraphsTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL *prev = NeuronOCL, *current = NULL;
   for(int i = 0; i < cUnitarSubGraphsTokenizer.Total(); i++)
     {
      current = cUnitarSubGraphsTokenizer[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::Save(const int file_handle)
  {
   if(!CNeuronMHAttentionPooling::Save(file_handle))
      ReturnFalse;
//---
   if(!cNodesTokenizer.Save(file_handle))
      ReturnFalse;
   if(!cEdgesTokenizer.Save(file_handle))
      ReturnFalse;
   if(!cSubGraphsTokenizer.Save(file_handle))
      ReturnFalse;
   if(!cUnitarSubGraphsTokenizer.Save(file_handle))
      ReturnFalse;
   if(!cConcatenate.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoT::Load(const int file_handle)
  {
   if(!CNeuronMHAttentionPooling::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cNodesTokenizer.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEdgesTokenizer.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSubGraphsTokenizer.AsObject()))
      ReturnFalse;
   if(!cUnitarSubGraphsTokenizer.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConcatenate.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMoT::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronMHAttentionPooling::SetOpenCL(obj);
//---
   cNodesTokenizer.SetOpenCL(OpenCL);
   cEdgesTokenizer.SetOpenCL(OpenCL);
   cSubGraphsTokenizer.SetOpenCL(OpenCL);
   cUnitarSubGraphsTokenizer.SetOpenCL(OpenCL);
   cConcatenate.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                              uint window_in, uint window_out, uint units_in, uint units_out,
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronChimera::Init(numOutputs, myIndex, open_cl, window_in, window_out, units_in, units_out, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cSSMPlus.Init(0, index, OpenCL, window_in, window_out, units_in, 2 * units_out, optimization, iBatch))
      ReturnFalse;
//---
   CNeuronTransposeOCL *transp = NULL;
   CNeuronConvOCL      *conv = NULL;
   cDiscretizationPlus.Clear();
   cDiscretizationPlus.SetOpenCL(OpenCL);
//---
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, 2 * units_out, window_out, optimization, iBatch) ||
      !cDiscretizationPlus.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * units_out, 2 * units_out, units_out, window_out, 1, optimization, iBatch) ||
      !cDiscretizationPlus.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, window_out, units_out, optimization, iBatch) ||
      !cDiscretizationPlus.Add(transp))
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
bool CNeuronChimeraPlus::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   for(uint i = 0; i < caSSM.Size(); i++)
     {
      if(!caSSM[i].FeedForward(NeuronOCL))
         ReturnFalse;
     }
   if(!cSSMPlus.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!cDiscretization.FeedForward(caSSM[1].AsObject()))
      ReturnFalse;
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < cDiscretizationPlus.Total(); i++)
     {
      current = cDiscretizationPlus[i];
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
      inp = current;
     }
//---
   inp = NeuronOCL;
   for(int i = 0; i < cResidual.Total(); i++)
     {
      current = cResidual[i];
      if(!current ||
         !current.FeedForward(inp))
         ReturnFalse;
      inp = current;
     }
//---
   inp = cDiscretizationPlus[-1];
   if(!SumAndNormalize(caSSM[0].getOutput(), cDiscretization.getOutput(), Output, 1, false, 0, 0, 0, 1) ||
      !SumAndNormalize(Output, inp.getOutput(), Output, 1, false, 0, 0, 0, 1) ||
      !SumAndNormalize(Output, current.getOutput(), Output, cDiscretization.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronChimera::calcInputGradients(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL *current = cDiscretizationPlus[-1];
   if(!current ||
      !DeActivation(current.getOutput(), current.getGradient(), Gradient, current.Activation()))
      ReturnFalse;
   for(int i = cDiscretizationPlus.Total() - 2; i >= 0; i--)
     {
      current = cDiscretizationPlus[i];
      if(!current ||
         !current.CalcHiddenGradients(cDiscretizationPlus[i + 1]))
         ReturnFalse;
     }
   if(!cSSMPlus.CalcHiddenGradients(current.AsObject()))
      ReturnFalse;
//---
   current = cResidual[0];
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(current.getGradient(), false) ||
      !NeuronOCL.CalcHiddenGradients(cSSMPlus.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronChimera::updateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSSMPlus.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL *current = NULL, *inp = cSSMPlus.AsObject();
   for(int i = 0; i < cDiscretizationPlus.Total(); i++)
     {
      current = cDiscretizationPlus[i];
      if(!current ||
         !current.UpdateInputWeights(inp.AsObject()))
         ReturnFalse;
      inp = current;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::Save(const int file_handle)
  {
   if(!CNeuronChimera::Save(file_handle))
      ReturnFalse;
   if(!cSSMPlus.Save(file_handle))
      ReturnFalse;
   if(!cDiscretizationPlus.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::Load(const int file_handle)
  {
   if(!CNeuronChimera::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSSMPlus.AsObject()))
      ReturnFalse;
   if(!cDiscretizationPlus.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronChimeraPlus::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronChimera::SetOpenCL(obj);
   cSSMPlus.SetOpenCL(OpenCL);
   cDiscretizationPlus.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChimeraPlus::Clear(void)
  {
   if(!CNeuronChimera::Clear())
      ReturnFalse;
   if(!cSSMPlus.Clear())
      ReturnFalse;
   CNeuronBaseOCL *current = NULL;
   for(int i = 0; i < cDiscretizationPlus.Total(); i++)
     {
      current = cDiscretizationPlus[i];
      if(!current ||
         !current.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                uint window, uint window_key, uint units_count,
                                uint heads, uint layers, uint stack_size, uint nactions,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronHidformer::Init(numOutputs, myIndex, open_cl, window_key, window_key, nactions,
                              heads, layers, stack_size, nactions, optimization_type, batch))
      ReturnFalse;
   if(!cChimera.Init(0, 0, OpenCL, window, window_key, units_count, nactions, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cChimera.FeedForward(NeuronOCL))
      ReturnFalse;
   return CNeuronHidformer::feedForward(cChimera.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronHidformer::calcInputGradients(cChimera.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cChimera.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronHidformer::updateInputWeights(cChimera.AsObject()))
      ReturnFalse;
//---
   if(!cChimera.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::Save(const int file_handle)
  {
   if(!CNeuronHidformer::Save(file_handle))
      ReturnFalse;
   if(!cChimera.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::Load(const int file_handle)
  {
   if(!CNeuronHidformer::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cChimera.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronHypridDecoder::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronHidformer::SetOpenCL(obj);
   cChimera.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronHypridDecoder::Clear(void)
  {
   if(!CNeuronHidformer::Clear())
      ReturnFalse;
   if(!cChimera.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                          uint window, uint window_key, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   int index = 0;
   if(!cOne.Init(0, index, OpenCL, units_count, optimization, iBatch))
      ReturnFalse;
   if(!cOne.getOutput().Fill(1))
      ReturnFalse;
   cOne.SetActivationFunction(None);
//---
   index++;
   if(!cA.Init(0, index, OpenCL, 1, 1, window * window_key, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cA.SetActivationFunction(MinusSoftPlus);
   CBufferFloat *w = cA.GetWeightsConv();
   if(!w || !w.Fill(0))
      ReturnFalse;
//---
   index++;
   if(!cD.Init(0, index, OpenCL, 1, 1, window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cD.SetActivationFunction(None);
   w = cD.GetWeightsConv();
   if(!w || !w.Fill(1))
      ReturnFalse;
//---
   index++;
   if(!cX_norm.Init(0, index, OpenCL, window * units_count, optimization, iBatch))
      ReturnFalse;
   cX_norm.SetActivationFunction(None);
   index++;
   if(!cX_proj.Init(0, index, OpenCL, window, window, 4 * window_key, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cX_proj.SetActivationFunction(None);
//---
   index++;
   if(!cDelta.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDelta.SetActivationFunction(None);
   index++;
   if(!cB.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cB.SetActivationFunction(None);
   index++;
   if(!cC.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cC.SetActivationFunction(None);
   index++;
   if(!cH.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cH.SetActivationFunction(None);
   index++;
   if(!cDelta_proj.Init(0, index, OpenCL, window_key, window_key, window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cDelta_proj.SetActivationFunction(SoftPlus);
//---
   index++;
   if(!cDeltaA.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaA.SetActivationFunction(None);
   index++;
   if(!cDeltaB.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaB.SetActivationFunction(None);
   index++;
   if(!cDeltaBX.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaBX.SetActivationFunction(None);
   index++;
   if(!cDeltaH.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaH.SetActivationFunction(None);
   index++;
   if(!cHS.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cHS.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::PScan(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint window = cX_proj.GetWindow();
   uint window_key = cX_proj.GetFilters() / 4;
   uint units = cD.GetUnits();
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {units, window * window_key};
   uint local_work_size[2] = {units, 1};
//---
   uint kernel = def_k_PScan;
   setBuffer(kernel, def_k_psc_A, cDeltaA.getOutputIndex())
   setBuffer(kernel, def_k_psc_X, cDeltaBX.getOutputIndex())
   setBuffer(kernel, def_k_psc_H, cDeltaH.getOutputIndex())
   setBuffer(kernel, def_k_psc_X_out, cHS.getOutputIndex())
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cHS.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::PScanCalcGradient(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint window = cX_proj.GetWindow();
   uint window_key = cX_proj.GetFilters() / 4;
   uint units = cD.GetUnits();
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {units, window * window_key};
   uint local_work_size[2] = {units, 1};
//---
   uint kernel = def_k_PScan_CalcHiddenGradient;
   setBuffer(kernel, def_k_pscgr_A, cDeltaA.getOutputIndex())
   setBuffer(kernel, def_k_pscgr_grad_A, cDeltaA.getGradientIndex())
   setBuffer(kernel, def_k_pscgr_X, cDeltaBX.getOutputIndex())
   setBuffer(kernel, def_k_pscgr_grad_X, cDeltaBX.getGradientIndex())
   setBuffer(kernel, def_k_pscgr_H, cDeltaH.getOutputIndex())
   setBuffer(kernel, def_k_pscgr_grad_H, cDeltaH.getGradientIndex())
   setBuffer(kernel, def_k_pscgr_grad_X_out, cHS.getGradientIndex())
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cHS.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//---
   uint window = cX_proj.GetWindow();
   uint window_key = cX_proj.GetFilters() / 4;
   uint units = cD.GetUnits();
//---
   if(!cA.FeedForward(cOne.AsObject()))   // (Units, Window, WindowKey)
      ReturnFalse;
   if(!cD.FeedForward(cOne.AsObject()))   // (Units, Window))
      ReturnFalse;
//---
   if(!NeuronOCL ||
      !SumAndNormalize(NeuronOCL.getOutput(), NeuronOCL.getOutput(), cX_norm.getOutput(), window, true, 0, 0, 0, 0.5f))
      ReturnFalse;
   if(!cX_proj.FeedForward(cX_norm.AsObject()))    // (Units, 4*WindowKey)
      ReturnFalse;
//---
   if(!DeConcat(cDelta.getOutput(), cB.getOutput(), cC.getOutput(), cH.getOutput(), cX_proj.getOutput(),
                window_key, window_key, window_key, window_key, units))   // 4*(Units, WindowKey)
      ReturnFalse;
   if(!cDelta_proj.FeedForward(cDelta.AsObject()))       // (Units, Window)
      ReturnFalse;
   if(!DiagMatMul(cDelta_proj.getOutput(), cA.getOutput(), cDeltaA.getOutput(),
                  window, window_key, units, SoftPlus))  // (Units, Window, WindowKey)
      ReturnFalse;
   if(!MatMul(cDelta_proj.getOutput(), cB.getOutput(), cDeltaB.getOutput(),
              window, 1, window_key, units))             // (Units, Window, WindowKey)
      ReturnFalse;
   if(!DiagMatMul(cX_norm.getOutput(), cDeltaB.getOutput(), cDeltaBX.getOutput(),
                  window, window_key, units, None))      // (Units, Window, WindowKey)
      ReturnFalse;
   if(!MatMul(cDelta_proj.getOutput(), cH.getOutput(), cDeltaH.getOutput(),
              window, 1, window_key, units))             // (Units, Window, WindowKey)
      ReturnFalse;
   if(!PScan())
      ReturnFalse;
   if(!MatMul(cHS.getOutput(), cC.getOutput(), Output, window, window_key, 1, units)) // (Units, Window, 1)
      ReturnFalse;
   if(!ElementMult(cD.getOutput(), cX_norm.getOutput(), PrevOutput))           // (Units, Window))
      ReturnFalse;
   if(!SumAndNormalize(Output, PrevOutput, Output, window, false, 0, 0, 0, 1))   // (Units, Window))
      ReturnFalse;
   if(!SumAndNormalize(Output, NeuronOCL.getOutput(), Output, window, false, 0, 0, 0, 1))   // (Units, Window))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint window = cX_proj.GetWindow();
   uint window_key = cX_proj.GetFilters() / 4;
   uint units = cD.GetUnits();
//---
   if(!ElementMultGrad(cD.getOutput(), cD.getGradient(), cX_norm.getOutput(), cX_norm.getPrevOutput(),
                       Gradient, cD.Activation(), None))          // (Units, Window))
      ReturnFalse;
   if(!MatMulGrad(cHS.getOutput(), cHS.getGradient(), cC.getOutput(), cC.getGradient(), Gradient,
                  window, window_key, 1, units)) // (Units, Window, 1)
      ReturnFalse;
   if(cHS.Activation() != None)
     {
      if(!DeActivation(cHS.getOutput(), cHS.getGradient(), cHS.getGradient(), cHS.Activation()))
         ReturnFalse;
     }
   if(cC.Activation() != None)
     {
      if(!DeActivation(cC.getOutput(), cC.getGradient(), cC.getGradient(), cC.Activation()))
         ReturnFalse;
     }
//---
   if(!PScanCalcGradient())
      ReturnFalse;
//---
   if(!MatMulGrad(cDelta_proj.getOutput(), cDelta_proj.getGradient(), cH.getOutput(), cH.getGradient(),
                  cDeltaH.getGradient(), window, 1, window_key, units))             // (Units, Window, WindowKey)
      ReturnFalse;
   if(!DiagMatMulGrad(cX_norm.getOutput(), cX_norm.getGradient(), cDeltaB.getOutput(), cDeltaB.getGradient(),
                      cDeltaBX.getGradient(), window, window_key, units))      // (Units, Window, WindowKey)
      ReturnFalse;
   if(!SumAndNormalize(cX_norm.getGradient(), cX_norm.getPrevOutput(), cX_norm.getPrevOutput(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!MatMulGrad(cDelta_proj.getOutput(), cDelta_proj.getPrevOutput(), cB.getOutput(), cB.getGradient(),
                  cDeltaB.getGradient(), window, 1, window_key, units))             // (Units, Window, WindowKey)
      ReturnFalse;
   if(!SumAndNormalize(cDelta_proj.getGradient(), cDelta_proj.getPrevOutput(), cDelta_proj.getGradient(),
                       window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!DeActivation(cDeltaA.getOutput(), cDeltaA.getGradient(), cDeltaA.getGradient(), SoftPlus))
      ReturnFalse;
   if(!DiagMatMulGrad(cDelta_proj.getOutput(), cDelta_proj.getPrevOutput(), cA.getOutput(), cA.getGradient(),
                      cDeltaA.getGradient(), window, window_key, units))  // (Units, Window, WindowKey)
      ReturnFalse;
   if(!SumAndNormalize(cDelta_proj.getGradient(), cDelta_proj.getPrevOutput(), cDelta_proj.getGradient(),
                       window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(cDelta_proj.Activation() != None)
     {
      if(!DeActivation(cDelta_proj.getOutput(), cDelta_proj.getGradient(), cDelta_proj.getGradient(), cDelta_proj.Activation()))
         ReturnFalse;
     }
//---
   if(!cDelta.CalcHiddenGradients(cDelta_proj.AsObject()))
      ReturnFalse;
//---
   if(!Concat(cDelta.getGradient(), cB.getGradient(), cC.getGradient(), cH.getGradient(), cX_proj.getGradient(),
              window_key, window_key, window_key, window_key, units))   // 4*(Units, WindowKey)
      ReturnFalse;
//---
   if(!cX_norm.CalcHiddenGradients(cX_proj.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cX_norm.getGradient(), cX_norm.getPrevOutput(), cX_norm.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cX_norm.getGradient(), Gradient, cX_norm.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), cX_norm.getGradient(), NeuronOCL.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//---
   if(!cA.UpdateInputWeights(cOne.AsObject()))   // (Units, Window, WindowKey)
      ReturnFalse;
   if(!cD.UpdateInputWeights(cOne.AsObject()))   // (Units, Window))
      ReturnFalse;
   if(!cX_proj.UpdateInputWeights(NeuronOCL))    // (Units, 4*WindowKey)
      ReturnFalse;
   if(!cDelta_proj.UpdateInputWeights(cDelta.AsObject()))       // (Units, Window)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cA.Save(file_handle))
      ReturnFalse;
   if(!cD.Save(file_handle))
      ReturnFalse;
   if(!cX_proj.Save(file_handle))
      ReturnFalse;
   if(!cDelta_proj.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAttraos::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cX_proj.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDelta_proj.AsObject()))
      ReturnFalse;
//---
   uint window = cX_proj.GetWindow();
   uint window_key = cX_proj.GetFilters() / 4;
   uint units_count = cD.GetUnits();
//---
   if(!cOne.Init(0, 0, OpenCL, units_count, optimization, iBatch))
      ReturnFalse;
   if(!cOne.getOutput().Fill(1))
      ReturnFalse;
   cOne.SetActivationFunction(None);
   int index = 3;
   if(!cX_norm.Init(0, index, OpenCL, window * units_count, optimization, iBatch))
      ReturnFalse;
   cX_norm.SetActivationFunction(None);
   index += 2;
   if(!cDelta.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDelta.SetActivationFunction(None);
   index++;
   if(!cB.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cB.SetActivationFunction(None);
   index++;
   if(!cC.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cC.SetActivationFunction(None);
   index++;
   if(!cH.Init(0, index, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cH.SetActivationFunction(None);
   index += 2;
   if(!cDeltaA.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaA.SetActivationFunction(None);
   index++;
   if(!cDeltaB.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaB.SetActivationFunction(None);
   index++;
   if(!cDeltaBX.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaBX.SetActivationFunction(None);
   index++;
   if(!cDeltaH.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cDeltaH.SetActivationFunction(None);
   index++;
   if(!cHS.Init(0, index, OpenCL, window * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cHS.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronAttraos::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cOne.SetOpenCL(OpenCL);
   cA.SetOpenCL(OpenCL);
   cX_proj.SetOpenCL(OpenCL);
   cDelta.SetOpenCL(OpenCL);
   cB.SetOpenCL(OpenCL);
   cC.SetOpenCL(OpenCL);
   cD.SetOpenCL(OpenCL);
   cH.SetOpenCL(OpenCL);
   cDelta_proj.SetOpenCL(OpenCL);
   cDeltaA.SetOpenCL(OpenCL);
   cDeltaB.SetOpenCL(OpenCL);
   cDeltaBX.SetOpenCL(OpenCL);
   cDeltaH.SetOpenCL(OpenCL);
   cHS.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoE::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                      uint window, uint window_out, uint units_count,
                      uint experts, uint top_k,
                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units_count, optimization_type, batch))
      ReturnFalse;
   int index = 0;
   if(!cGates.Init(0, index, OpenCL, window, units_count, experts, top_k, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cXProj.Init(0, index, OpenCL, window, window, window_out, units_count, optimization, iBatch))
      ReturnFalse;
   cXProj.SetActivationFunction(None);
//---
   cExperts.Clear();
   cExperts.SetOpenCL(OpenCL);
   CNeuronConvOCL *conv = NULL;
   CNeuronTransposeRCDOCL *transp = NULL;
//---
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, window_out * experts, units_count, 1, optimization, iBatch) ||
      !cExperts.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   transp = new CNeuronTransposeRCDOCL();
   index++;
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_count, experts, window_out, optimization, iBatch) ||
      !cExperts.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   transp.SetActivationFunction((ENUM_ACTIVATION)conv.Activation());
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_out, window_out, window_out, units_count, experts, optimization, iBatch) ||
      !cExperts.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
   transp = new CNeuronTransposeRCDOCL();
   index++;
   if(!transp ||
      !transp.Init(0, index, OpenCL, experts, units_count, window_out, optimization, iBatch) ||
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
bool CNeuronMoE::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cGates.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL *prev = NeuronOCL;
   int total = cExperts.Total();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL *neuron = cExperts[i];
      if(!neuron ||
         !neuron.FeedForward(prev))
         ReturnFalse;
      prev = neuron;
     }
   if(!MatMul(cGates.getOutput(), prev.getOutput(), getOutput(),
              1, cGates.GetGates(), Neurons() / cGates.GetUnits(), cGates.GetUnits()))
      ReturnFalse;
//---
   if(!cXProj.FeedForward(NeuronOCL) ||
      !SumAndNormalize(cXProj.getOutput(), Output, Output, cXProj.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoE::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   CNeuronBaseOCL *neuron = cExperts[-1];
   if(!MatMulGrad(cGates.getOutput(), cGates.getGradient(), neuron.getOutput(),
                  neuron.getGradient(), getGradient(), 1, cGates.GetGates(),
                  Neurons() / cGates.GetUnits(), cGates.GetUnits()))
      ReturnFalse;
//---
   if(cGates.Activation() != None)
      if(!DeActivation(cGates.getOutput(), cGates.getGradient(),
                       cGates.getGradient(), cGates.Activation()))
         ReturnFalse;
   if(neuron.Activation() != None)
      if(!DeActivation(neuron.getOutput(), neuron.getGradient(),
                       neuron.getGradient(), neuron.Activation()))
         ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cGates.AsObject()))
      ReturnFalse;
//---
   for(int i = cExperts.Total() - 2; i >= 0; i--)
     {
      neuron = cExperts[i];
      if(!neuron ||
         !neuron.CalcHiddenGradients(cExperts[i + 1]))
         ReturnFalse;
     }
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(neuron.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!DeActivation(cXProj.getOutput(), cXProj.getGradient(), Gradient, cXProj.Activation()) ||
      !NeuronOCL.CalcHiddenGradients(cXProj.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoE::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cGates.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   if(!cXProj.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL *prev = NeuronOCL;
   int total = cExperts.Total();
   for(int i = 0; i < total; i++)
     {
      CNeuronBaseOCL *neuron = cExperts[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(prev))
         ReturnFalse;
      prev = neuron;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoE::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cGates.Save(file_handle))
      ReturnFalse;
   if(!cXProj.Save(file_handle))
      ReturnFalse;
   if(!cExperts.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoE::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGates.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cXProj.AsObject()))
      ReturnFalse;
   if(!cExperts.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMoE::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
   cGates.SetOpenCL(OpenCL);
   cXProj.SetOpenCL(OpenCL);
   cExperts.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                             uint window, uint units_count,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(window <= 0)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * units_count, optimization_type, batch))
      ReturnFalse;
//--- Save constants
   iUnits = units_count;
   activation = None;
//--- Calculate FFT dimension
   int power = int(MathLog(window) / M_LN2);
   if(MathPow(2, power) != window)
      power++;
   iFFTdimension = uint(MathPow(2, power));
//---
   if(!cbFFTReal.BufferInit(iFFTdimension * iUnits, 0) ||
      !cbFFTReal.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cbFFTImag.BufferInit(iFFTdimension * iUnits, 0) ||
      !cbFFTImag.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iUnits};
   int kernel = def_k_FFT;
   setBuffer(kernel, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(kernel, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(kernel, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(kernel, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(kernel, def_k_fft_input_window, (int)(inp_re.Total() / iUnits))
   setArgument(kernel, def_k_fft_input_complex, int(!!inp_im))
   setArgument(kernel, def_k_fft_output_window, (int)(out_re.Total() / iUnits))
   setArgument(kernel, def_k_fft_reverse, int(reverse))
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::Mask(void)
  {
   uint global_work_offset[2] = {0, 0 };
   uint global_work_size[2] = {iUnits, iUnits};
   uint local_work_size[2] = {1, iUnits};
//---
   int kernel = def_k_MaskByDistance;
   setBuffer(kernel, def_k_maskDist_buf_real, cbFFTReal.GetIndex())
   setBuffer(kernel, def_k_maskDist_buf_imag, cbFFTImag.GetIndex())
   setBuffer(kernel, def_k_maskDist_mask, getOutputIndex())
   setArgument(kernel, def_k_maskDist_dimension, int(iFFTdimension))
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!FFT(NeuronOCL.getOutput(), NULL, GetPointer(cbFFTReal), GetPointer(cbFFTImag), false))
      ReturnFalse;
//---
   return Mask();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iUnits)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iFFTdimension)) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronChanelMask::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iFFTdimension = (uint)FileReadInteger(file_handle);
//---
   SetOpenCL(OpenCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronChanelMask::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cbFFTReal.BufferFree();
   cbFFTImag.BufferFree();
   cbFFTReal.BufferInit(iFFTdimension * iUnits, 0);
   cbFFTImag.BufferInit(iFFTdimension * iUnits, 0);
   cbFFTReal.BufferCreate(OpenCL);
   cbFFTImag.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                       uint window, uint window_key, uint units_count,
                       uint heads, uint units_out, uint experts, uint top_k,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, window, units_out, optimization_type, batch))
      ReturnFalse;
//---
   iWindowKey = MathMax(window_key, 1);
   iHeads = MathMax(heads, 1);
//---
   int index = 0;
   if(!cTranspose.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cExperts.Init(0, index, OpenCL, units_count, units_out, window, experts, top_k, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMask.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cQKV.Init(0, index, OpenCL, units_out, units_out, iHeads * iWindowKey * 3, window, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cQ.Init(0, index, OpenCL, cQKV.Neurons() / 3, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKV.Init(0, index, OpenCL, cQ.Neurons() * 2, optimization, iBatch))
      ReturnFalse;
   if(!cbScores.BufferInit(cMask.Neurons()*iHeads, 0) ||
      !cbScores.BufferCreate(OpenCL))
      ReturnFalse;
   index++;
   if(!cMHAttentionOut.Init(0, index, OpenCL, cQ.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPooling.Init(0, index, OpenCL, iWindowKey * iHeads, iWindowKey * iHeads, units_out, window, 1, optimization, iBatch))
      ReturnFalse;
   cPooling.SetActivationFunction(None);
   index++;
   if(!cResidual.Init(0, index, OpenCL, cPooling.Neurons(), optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   index++;
   if(!cFeedForward.Init(0, index, OpenCL, units_out, 4 * units_out, window, 1, heads, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::AttentionOut(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iCount/*Q units*/, iCount/*K units*/, iHeads};
   uint local_work_size[3] = {1, iCount, 1};
   ResetLastError();
   int kernel = def_k_MaskAttention;
   setBuffer(kernel, def_k_maskatt_q, cQ.getOutputIndex())
   setBuffer(kernel, def_k_maskatt_kv, cKV.getOutputIndex())
   setBuffer(kernel, def_k_maskatt_scores, cbScores.GetIndex())
   setBuffer(kernel, def_k_maskatt_out, cMHAttentionOut.getOutputIndex())
   setBuffer(kernel, def_k_maskatt_masks, cMask.getOutputIndex())
   setArgument(kernel, def_k_maskatt_dimension, (int)iWindowKey)
   setArgument(kernel, def_k_maskatt_heads_kv, (int)iHeads)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cMHAttentionOut.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::AttentionInsideGradients(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iCount, iWindowKey, iHeads};
   ResetLastError();
   int kernel = def_k_MaskAttentionGradients;
   setBuffer(kernel, def_k_maskattgr_q, cQ.getOutputIndex())
   setBuffer(kernel, def_k_maskattgr_q_g, cQ.getGradientIndex())
   setBuffer(kernel, def_k_maskattgr_kv, cKV.getOutputIndex())
   setBuffer(kernel, def_k_maskattgr_kv_g, cKV.getGradientIndex())
   setBuffer(kernel, def_k_maskattgr_scores, cbScores.GetIndex())
   setBuffer(kernel, def_k_maskattgr_gradient, cMHAttentionOut.getGradientIndex())
   setArgument(kernel, def_k_maskattgr_kunits, (int)iCount)
   setArgument(kernel, def_k_maskattgr_heads_kv, (int)iHeads)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cKV.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cExperts.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cMask.FeedForward(cTranspose.AsObject()))
      ReturnFalse;
//---
   if(!cQKV.FeedForward(cExperts.AsObject()))
      ReturnFalse;
   if(!DeConcat(cQ.getOutput(), cKV.getOutput(), cQKV.getOutput(), iWindowKey, 2 * iWindowKey, cQKV.GetUnits()))
      ReturnFalse;
   if(!AttentionOut())
      ReturnFalse;
   if(!cPooling.FeedForward(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cExperts.getOutput(), cPooling.getOutput(), cResidual.getOutput(), iWindow, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!cFeedForward.FeedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return CNeuronTransposeOCL::feedForward(cFeedForward.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::calcInputGradients(cFeedForward.AsObject()))
      ReturnFalse;
   if(!cPooling.CalcHiddenGradients(cFeedForward.AsObject()))
      ReturnFalse;
   if(!cMHAttentionOut.CalcHiddenGradients(cPooling.AsObject()))
      ReturnFalse;
   if(!AttentionInsideGradients())
      ReturnFalse;
   if(!Concat(cQ.getGradient(), cKV.getGradient(), cQKV.getGradient(), iWindowKey, 2 * iWindowKey, iCount))
      ReturnFalse;
   if(cQKV.Activation() != None)
      if(!DeActivation(cQKV.getOutput(), cQKV.getGradient(), cQKV.getGradient(), cQKV.Activation()))
         ReturnFalse;
   if(!cExperts.CalcHiddenGradients(cQKV.AsObject()) ||
      !DeActivation(cExperts.getOutput(), cExperts.getPrevOutput(), cPooling.getGradient(), cExperts.Activation()) ||
      !SumAndNormalize(cExperts.getGradient(), cExperts.getPrevOutput(), cExperts.getGradient(), iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cExperts.AsObject()))
      ReturnFalse;
//---
   return prevLayer.CalcHiddenGradients(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cExperts.UpdateInputWeights(cTranspose.AsObject()))
      ReturnFalse;
//---
   if(!cQKV.UpdateInputWeights(cExperts.AsObject()))
      ReturnFalse;
   if(!cPooling.UpdateInputWeights(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!cFeedForward.UpdateInputWeights(cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, iWindowKey) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, iHeads) < INT_VALUE)
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cExperts.Save(file_handle))
      ReturnFalse;
   if(!cQKV.Save(file_handle))
      ReturnFalse;
   if(!cMask.Save(file_handle))
      ReturnFalse;
   if(!cPooling.Save(file_handle))
      ReturnFalse;
   if(!cFeedForward.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDUET::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindowKey = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHeads = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cExperts.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cQKV.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMask.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPooling.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFeedForward.AsObject()))
      ReturnFalse;
//---
   int index = 4;
   if(!cQ.Init(0, index, OpenCL, cQKV.Neurons() / 3, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKV.Init(0, index, OpenCL, cQ.Neurons() * 2, optimization, iBatch))
      ReturnFalse;
   if(!cbScores.BufferInit(cMask.Neurons()*iHeads, 0) ||
      !cbScores.BufferCreate(OpenCL))
      ReturnFalse;
   index++;
   if(!cMHAttentionOut.Init(0, index, OpenCL, cQ.Neurons(), optimization, iBatch))
      ReturnFalse;
   index += 2;
   if(!cResidual.Init(0, index, OpenCL, cPooling.Neurons(), optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDUET::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
//---
   cTranspose.SetOpenCL(OpenCL);
   cExperts.SetOpenCL(OpenCL);
   cQKV.SetOpenCL(OpenCL);
   cMask.SetOpenCL(OpenCL);
   cPooling.SetOpenCL(OpenCL);
   cFeedForward.SetOpenCL(OpenCL);
   cQ.SetOpenCL(OpenCL);
   cKV.SetOpenCL(OpenCL);
   cbScores.BufferCreate(OpenCL);
   cMHAttentionOut.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                      uint &windows[], uint window_out, uint units_count,
                                      uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(windows.Size() <= 0 || ArrayCopy(aiWindows, windows, 0, 0, windows.Size()) < int(windows.Size()))
      ReturnFalse;
   int window = 0;
   for(uint i = 0; i < aiWindows.Size(); i++)
     {
      window += aiWindows[i] + 1;
      if(aiWindows[i] <= 0)
         ReturnFalse;
     }
   window--;
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window, window, window_out, units_count * aiWindows.Size()*variables, 1, ADAM, batch))
      ReturnFalse;
//---
   iVariables = variables;
   int temp = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
   if(temp < 0)
      ReturnFalse;
   iWindow = (uint)temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {Neurons() / iVariables, iVariables};
   ResetLastError();
#ifdef _DEBUG
   if(NeuronOCL.Neurons() != OpenCL.GetBufferSize(NeuronOCL.getOutputIndex()) / sizeof(float))
      ReturnFalse;
   if(Neurons() != OpenCL.GetBufferSize(getOutputIndex()) / sizeof(float))
      ReturnFalse;
   if(WeightsConv.Total() != OpenCL.GetBufferSize(WeightsConv.GetIndex()) / sizeof(float))
      ReturnFalse;
   if(aiWindows.Size() != OpenCL.GetBufferSize(iWindow) / sizeof(float))
      ReturnFalse;
   if(!Output.BufferRead())
      ReturnFalse;
#endif
   int kernel = def_k_FeedForwardMultWinConv;
   setBuffer(kernel, def_k_ffmwc_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ffmwc_matrix_o, getOutputIndex())
   setBuffer(kernel, def_k_ffmwc_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_ffmwc_windows_in, iWindow)
   setArgument(kernel, def_k_ffmwc_inputs, NeuronOCL.Neurons() / iVariables)
   setArgument(kernel, def_k_ffmwc_window_out, iWindowOut)
   setArgument(kernel, def_k_ffmwc_windows_total, (int)aiWindows.Size())
   setArgument(kernel, def_k_ffmwc_activation, (int)activation)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {NeuronOCL.Neurons() / iVariables, iVariables};
   ResetLastError();
   int kernel = def_k_CalcHiddenGradientMultWinConv;
   setBuffer(kernel, def_k_chgmwc_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_chgmwc_matrix_ig, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_chgmwc_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_chgmwc_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_chgmwc_windows_in, iWindow)
   setArgument(kernel, def_k_chgmwc_outputs, Neurons() / iVariables)
   setArgument(kernel, def_k_chgmwc_window_out, iWindowOut)
   setArgument(kernel, def_k_chgmwc_windows_total, (int)aiWindows.Size())
   setArgument(kernel, def_k_chgmwc_activation, NeuronOCL.Activation())
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
bool CNeuronMultiWindowsConvOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {WeightsConv.Total(), iVariables};
   uint local_work_size[2] = {1, iVariables};
   ResetLastError();
   int kernel = def_k_UpdateWeightsMultWinConvAdam;
   setBuffer(kernel, def_k_uwmwc_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_uwmwc_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_uwmwc_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_uwmwc_windows_in, iWindow)
   setBuffer(kernel, def_k_uwmwc_matrix_m, FirstMomentumConv.GetIndex())
   setBuffer(kernel, def_k_uwmwc_matrix_v, SecondMomentumConv.GetIndex())
   setArgument(kernel, def_k_uwmwc_inputs, NeuronOCL.Neurons() / iVariables)
   setArgument(kernel, def_k_uwmwc_outputs, Neurons() / iVariables)
   setArgument(kernel, def_k_uwmwc_window_out, iWindowOut)
   setArgument(kernel, def_k_uwmwc_windows_total, (int)aiWindows.Size())
   setArgument(kernel, def_k_uwmwc_l, lr)
   setArgument(kernel, def_k_uwmwc_b1, b1)
   setArgument(kernel, def_k_uwmwc_b2, b2)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!SecondMomentumConv.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvOCL::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   uint total = aiWindows.Size();
   if(FileWriteInteger(file_handle, (int)total, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   for(int i = 0; i < (int)total; i++)
      if(FileWriteInteger(file_handle, aiWindows[i]) < INT_VALUE)
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvOCL::Load(const int file_handle)
  {
   if(!!OpenCL && iWindow >= 0)
      OpenCL.BufferFree(iWindow);
//---
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
//---
   ArrayFree(aiWindows);
   int total = FileReadInteger(file_handle);
   if(ArrayResize(aiWindows, total) != total)
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      aiWindows[i] = FileReadInteger(file_handle);
      if(aiWindows[i] <= 0)
         ReturnFalse;
     }
   int sum = 0;
   for(int i = 0; i < total; i++)
      sum += aiWindows[i] + 1;
   if(iStep != (sum - 1))
      ReturnFalse;
   int temp = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
   if(temp < 0)
      ReturnFalse;
   iWindow = (uint)temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiWindowsConvOCL::SetOpenCL(COpenCLMy * obj)
  {
   if(!!OpenCL && iWindow >= 0)
      OpenCL.BufferFree(iWindow);
   CNeuronConvOCL::SetOpenCL(obj);
   iWindow = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronComplexConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint step, uint window_out, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, 2 * units_count * window_out * variables, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = (int)window;
   iStep = MathMax(step, 1);
   activation = None;
   iWindowOut = window_out;
   iVariables = variables;
//---
   if(CheckPointer(WeightsConv) == POINTER_INVALID)
     {
      WeightsConv = new CBufferFloat();
      if(CheckPointer(WeightsConv) == POINTER_INVALID)
         ReturnFalse;
     }
   int count = (int)(2 * (iWindow + 1) * iWindowOut * iVariables);
   if(!WeightsConv.Reserve(count))
      ReturnFalse;
   float k = (float)(1 / sqrt(iWindow + 1));
   for(int i = 0; i < count; i++)
     {
      if(!WeightsConv.Add((GenerateWeight() * 2 * k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!WeightsConv.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(optimization == SGD)
     {
      if(CheckPointer(DeltaWeightsConv) == POINTER_INVALID)
        {
         DeltaWeightsConv = new CBufferFloat();
         if(CheckPointer(DeltaWeightsConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!DeltaWeightsConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!DeltaWeightsConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      if(CheckPointer(FirstMomentumConv) == POINTER_INVALID)
        {
         FirstMomentumConv = new CBufferFloat();
         if(CheckPointer(FirstMomentumConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!FirstMomentumConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!FirstMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(CheckPointer(SecondMomentumConv) == POINTER_INVALID)
        {
         SecondMomentumConv = new CBufferFloat();
         if(CheckPointer(SecondMomentumConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!SecondMomentumConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!SecondMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronComplexConvOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3];
   global_work_size[0] = Output.Total() / (2 * iWindowOut * iVariables);
   global_work_size[1] = iWindowOut;
   global_work_size[2] = iVariables;
   int kernel = def_k_FeedForwardComplexConv;
   setBuffer(kernel, def_k_ffc_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_ffc_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ffc_matrix_o, Output.GetIndex())
   setArgument(kernel, def_k_ffc_inputs, NeuronOCL.Neurons() / (2 * iVariables))
   setArgument(kernel, def_k_ffc_step, (int)iStep)
   setArgument(kernel, def_k_ffc_window_in, (int)iWindow)
   setArgument(kernel, def_k_ffc_activation - 1, (int)activation)
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
bool CNeuronComplexConvOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = NeuronOCL.Neurons() / (2 * iVariables);
   global_work_size[1] = iVariables;
   int kernel = def_k_CalcHiddenGradientComplexConv;
   setBuffer(kernel, def_k_chgc_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_chgc_matrix_g, Gradient.GetIndex())
   setBuffer(kernel, def_k_chgc_matrix_o, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_chgc_matrix_ig, NeuronOCL.getGradientIndex())
   setArgument(kernel, def_k_chgc_outputs, Neurons() / (2 * iVariables))
   setArgument(kernel, def_k_chgc_step, (int)iStep)
   setArgument(kernel, def_k_chgc_window_in, (int)iWindow)
   setArgument(kernel, def_k_chgc_window_out, (int)iWindowOut)
   setArgument(kernel, def_k_chgc_activation, (int)NeuronOCL.Activation())
   setArgument(kernel, def_k_chgc_shift_out, (int)0)
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
bool CNeuronComplexConvOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = { 0, 0 };
   uint global_work_size[2] = { WeightsConv.Total() / (2 * iVariables), iVariables };
   float lt = 0;
   int kernel = -1;
   switch(optimization)
     {
      case SGD:
         kernel = def_k_UpdateWeightsComplexConvMomentum;
         setBuffer(kernel, def_k_uwcm_matrix_w, WeightsConv.GetIndex())
         setBuffer(kernel, def_k_uwcm_matrix_g, getGradientIndex())
         setBuffer(kernel, def_k_uwcm_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(kernel, def_k_uwcm_matrix_dw, DeltaWeightsConv.GetIndex())
         setBuffer(kernel, def_k_uwcm_inputs, NeuronOCL.Neurons() / (2 * iVariables))
         setArgument(kernel, def_k_uwcm_learning_rates, lr)
         setArgument(kernel, def_k_uwcm_momentum, alpha)
         setBuffer(kernel, def_k_uwcm_window_in, (int)iWindow)
         setBuffer(kernel, def_k_uwcm_window_out, (int)iWindowOut)
         setBuffer(kernel, def_k_uwcm_step, (int)iStep)
         kernelExecute(kernel, global_work_offset, global_work_size)
         break;
      case ADAM:
         kernel = def_k_UpdateWeightsComplexConvAdam;
         setBuffer(kernel, def_k_uwca_matrix_w, WeightsConv.GetIndex())
         setBuffer(kernel, def_k_uwca_matrix_g, getGradientIndex())
         setBuffer(kernel, def_k_uwca_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(kernel, def_k_uwca_matrix_m, FirstMomentumConv.GetIndex())
         setBuffer(kernel, def_k_uwca_matrix_v, SecondMomentumConv.GetIndex())
         lt = (float)(lr * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t)));
         setArgument(kernel, def_k_uwca_inputs, NeuronOCL.Neurons() / (2 * iVariables))
         setArgument(kernel, def_k_uwca_l, lt)
         setArgument(kernel, def_k_uwca_b1, b1)
         setArgument(kernel, def_k_uwca_b2, b2)
         setArgument(kernel, def_k_uwca_window_in, (int)iWindow)
         setArgument(kernel, def_k_uwca_window_out, (int)iWindowOut)
         setArgument(kernel, def_k_uwca_step, (int)iStep)
         //---
         kernelExecute(kernel, global_work_offset, global_work_size)
         t++;
         break;
      default:
         ReturnFalse;
         break;
     }
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im,
                       CBufferFloat* out_re, CBufferFloat* out_im,
                       uint variables, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {variables};
   int kernel = def_k_FFT;
   setBuffer(kernel, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(kernel, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(kernel, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(kernel, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(kernel, def_k_fft_input_window, (int)(inp_re.Total() / variables))
   setArgument(kernel, def_k_fft_input_complex, int(!!inp_im))
   setArgument(kernel, def_k_fft_output_window, (int)(out_re.Total() / variables))
   setArgument(kernel, def_k_fft_reverse, int(reverse))
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_re.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                        uint time_step, uint variables,
                        uint window, uint step, uint window_key, uint heads,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, variables, time_step, optimization_type, batch))
      ReturnFalse;
//--- Calculate FFT size
   int power = int(MathLog(time_step) / M_LN2);
   if(power <= 0)
      ReturnFalse;
   if(MathPow(2, power) != time_step)
      power++;
   uint FreqUinits = uint(MathPow(2, power));
//---
   if(window <= 0 || step <= 0)
      ReturnFalse;
   int Segments = int((FreqUinits - int(window) + step - 1) / step);
   if(Segments <= 0)
      ReturnFalse;
//---
   uint index = 0;
   if(!cTranspose.Init(0, index, OpenCL, time_step, variables, optimization, iBatch))
      ReturnFalse;
   for(uint i = 0; i < caFreqIn.Size(); i++)
     {
      index++;
      if(!caFreqIn[i].Init(0, index, OpenCL, FreqUinits * variables, optimization, iBatch))
         ReturnFalse;
      caFreqIn[i].SetActivationFunction(None);
     }
   index++;
   if(!cFreqConcat.Init(0, index, OpenCL, 2 * FreqUinits * variables, optimization, iBatch))
      ReturnFalse;
   cFreqConcat.SetActivationFunction(None);
//---
   index++;
   if(!caProjection[0].Init(0, index, OpenCL, window, step, 2 * window, Segments, variables, optimization, iBatch))
      ReturnFalse;
   caProjection[0].SetActivationFunction(LReLU);
   index++;
   if(!caProjection[1].Init(0, index, OpenCL, 2 * window, 2 * window, window_key, Segments, variables, optimization, iBatch))
      ReturnFalse;
   caProjection[1].SetActivationFunction(TANH);
//---
   index++;
   if(!caChannelFusion[0].Init(0, index, OpenCL, window_key, window_key, heads, Segments, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(Segments % 2 == 0)
     {
      if(!caChannelFusion[1].Init(0, index, OpenCL, 2 * window_key, window_key, heads, Segments / 2, variables, optimization, iBatch))
         ReturnFalse;
     }
   else
      if(!caChannelFusion[1].Init(0, index, OpenCL, window_key, window_key, heads, Segments, variables, optimization, iBatch))
         ReturnFalse;
//---
   index++;
   if(!caLinearHead[0].Init(0, index, OpenCL, window_key, window_key, window, Segments, variables, optimization, iBatch))
      ReturnFalse;
   caLinearHead[0].SetActivationFunction(LReLU);
   index++;
   if(!caLinearHead[1].Init(0, index, OpenCL, window * Segments, window * Segments, FreqUinits, variables, 1, optimization, iBatch))
      ReturnFalse;
   caLinearHead[1].SetActivationFunction(None);
//---
   for(uint i = 0; i < caFreqOut.Size(); i++)
     {
      index++;
      if(!caFreqOut[i].Init(0, index, OpenCL, FreqUinits * variables, optimization, iBatch))
         ReturnFalse;
      caFreqOut[i].SetActivationFunction(None);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!FFT(cTranspose.getOutput(), NULL, caFreqIn[0].getOutput(), caFreqIn[1].getOutput(), iCount, false))
      ReturnFalse;
   if(!Concat(caFreqIn[0].getOutput(), caFreqIn[1].getOutput(), cFreqConcat.getOutput(), 1, 1, caFreqIn[0].Neurons()))
      ReturnFalse;
   if(!ComplexNormalize(cFreqConcat.AsObject(), cFreqConcat.AsObject(), cFreqConcat.getPrevOutput(), cFreqConcat.getPrevOutput(), caFreqIn[0].Neurons() / iCount, iCount))
      ReturnFalse;
//---
   CNeuronBaseOCL *neuron = cFreqConcat.AsObject();
//neuron.getOutput().BufferRead();
   for(uint i = 0; i < caProjection.Size(); i++)
     {
      if(!caProjection[i].FeedForward(neuron))
         ReturnFalse;
      neuron = caProjection[i].AsObject();
      //neuron.getOutput().BufferRead();
     }
//---
   for(uint i = 0; i < caChannelFusion.Size(); i++)
     {
      if(!caChannelFusion[i].FeedForward(neuron))
         ReturnFalse;
      neuron = caChannelFusion[i].AsObject();
      //neuron.getOutput().BufferRead();
     }
//---
   for(uint i = 0; i < caLinearHead.Size(); i++)
     {
      if(!caLinearHead[i].FeedForward(neuron))
         ReturnFalse;
      neuron = caLinearHead[i].AsObject();
      //neuron.getOutput().BufferRead();
     }
//---
   if(!DeConcat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), neuron.getOutput(), 1, 1, caFreqOut[0].Neurons()))
      ReturnFalse;
   if(!FFT(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caFreqOut[0].getPrevOutput(), caFreqOut[1].getPrevOutput(), iCount, true))
      ReturnFalse;
//---
   if(!DeConcat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caFreqOut[0].getPrevOutput(), iWindow, caFreqOut[0].Neurons() / iCount - iWindow, iCount))
      ReturnFalse;
//---
   if(!SumAndNormalize(caFreqOut[0].getOutput(), cTranspose.getOutput(), caFreqOut[0].getOutput(), iWindow, true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return CNeuronTransposeOCL::feedForward(caFreqOut[0].AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::calcInputGradients(caFreqOut[1].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(caFreqOut[1].getGradient(), caFreqOut[1].getGradient(), cTranspose.getPrevOutput(), iWindow, false, 0, 0, 0, 0.5f))
      ReturnFalse;
   if((caFreqOut[0].Neurons() - iWindow) > 0)
      if(!SumAndNormalize(caFreqOut[1].getOutput(), caFreqOut[1].getOutput(), caFreqOut[1].getOutput(), 1, false, 0, 0, 0, -0.5f))
         ReturnFalse;
   if(!Concat(caFreqOut[1].getGradient(), caFreqOut[1].getOutput(), caFreqOut[0].getGradient(), iWindow, caFreqOut[0].Neurons() - iWindow, iCount))
      ReturnFalse;
   if(!SumAndNormalize(caFreqOut[1].getPrevOutput(), caFreqOut[1].getPrevOutput(), caFreqOut[1].getGradient(), 1, false, 0, 0, 0, -0.5f))
      ReturnFalse;
//---
   if(!FFT(caFreqOut[0].getGradient(), caFreqOut[1].getGradient(), caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), iCount, false))
      ReturnFalse;
   if(!Concat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caLinearHead[1].getGradient(), 1, 1, caFreqOut[0].Neurons()))
      ReturnFalse;
//---
   if(!caLinearHead[0].CalcHiddenGradients(caLinearHead[1].AsObject()))
      ReturnFalse;
//---
   CObject *neuron = caLinearHead[0].AsObject();
   for(int i = int(caChannelFusion.Size()) - 1; i >= 0; i--)
     {
      if(!caChannelFusion[i].CalcHiddenGradients(neuron))
         ReturnFalse;
      neuron = caChannelFusion[i].AsObject();
     }
//---
   for(int i = int(caProjection.Size()) - 1; i >= 0; i--)
     {
      if(!caProjection[i].CalcHiddenGradients(neuron))
         ReturnFalse;
      neuron = caProjection[i].AsObject();
     }
//---
   if(!cFreqConcat.CalcHiddenGradients(neuron))
      ReturnFalse;
//---
   if(!DeConcat(caFreqIn[0].getGradient(), caFreqIn[1].getGradient(), cFreqConcat.getGradient(), 1, 1, caFreqIn[0].Neurons()))
      ReturnFalse;
   if(!FFT(caFreqIn[0].getGradient(), caFreqIn[1].getGradient(), caFreqIn[0].getPrevOutput(), caFreqIn[1].getPrevOutput(), iCount, false))
      ReturnFalse;
   if(!DeConcat(cTranspose.getGradient(), caFreqIn[0].getGradient(), caFreqIn[0].getPrevOutput(), iWindow, caFreqIn[0].Neurons() / iCount - iWindow, iCount))
      ReturnFalse;
//---
   if(!SumAndNormalize(cTranspose.getGradient(), cTranspose.getPrevOutput(), cTranspose.getGradient(), iWindow, false, 0, 0, 0, 1.0f))
      ReturnFalse;
//---
   if(!prevLayer.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(prevLayer.Activation() != None)
     {
      if(!DeActivation(prevLayer.getOutput(), prevLayer.getGradient(), prevLayer.getGradient(), prevLayer.Activation()))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *neuron = cFreqConcat.AsObject();
   for(uint i = 0; i < caProjection.Size(); i++)
     {
      if(!caProjection[i].UpdateInputWeights(neuron))
         ReturnFalse;
      neuron = caProjection[i].AsObject();
     }
//---
   for(uint i = 0; i < caChannelFusion.Size(); i++)
     {
      if(!caChannelFusion[i].UpdateInputWeights(neuron))
         ReturnFalse;
      neuron = caChannelFusion[i].AsObject();
     }
//---
   for(uint i = 0; i < caLinearHead.Size(); i++)
     {
      if(!caLinearHead[i].UpdateInputWeights(neuron))
         ReturnFalse;
      neuron = caLinearHead[i].AsObject();
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronCATCH *Source = source;
   for(uint i = 0; i < caProjection.Size(); i++)
      if(!caProjection[i].WeightsUpdate(GetPointer(Source.caProjection[i]), tau))
         ReturnFalse;
//---
   for(uint i = 0; i < caChannelFusion.Size(); i++)
      if(!caChannelFusion[i].WeightsUpdate(GetPointer(Source.caChannelFusion[i]), tau))
         ReturnFalse;
//---
   for(uint i = 0; i < caLinearHead.Size(); i++)
      if(!caLinearHead[i].WeightsUpdate(GetPointer(Source.caLinearHead[i]), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caFreqIn.Size(); i++)
      if(!caFreqIn[i].Save(file_handle))
         ReturnFalse;
   if(!cFreqConcat.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caProjection.Size(); i++)
      if(!caProjection[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caChannelFusion.Size(); i++)
      if(!caChannelFusion[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caLinearHead.Size(); i++)
      if(!caLinearHead[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caFreqOut.Size(); i++)
      if(!caFreqOut[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caFreqIn.Size(); i++)
      if(!LoadInsideLayer(file_handle, caFreqIn[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFreqConcat.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < caProjection.Size(); i++)
      if(!LoadInsideLayer(file_handle, caProjection[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caChannelFusion.Size(); i++)
      if(!LoadInsideLayer(file_handle, caChannelFusion[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caLinearHead.Size(); i++)
      if(!LoadInsideLayer(file_handle, caLinearHead[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caFreqOut.Size(); i++)
      if(!LoadInsideLayer(file_handle, caFreqOut[i].AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCATCH::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
   cTranspose.SetOpenCL(OpenCL);
   for(uint i = 0; i < caFreqIn.Size(); i++)
      caFreqIn[i].SetOpenCL(OpenCL);
   cFreqConcat.SetOpenCL(OpenCL);
   for(uint i = 0; i < caProjection.Size(); i++)
      caProjection[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < caChannelFusion.Size(); i++)
      caChannelFusion[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < caLinearHead.Size(); i++)
      caLinearHead[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < caFreqOut.Size(); i++)
      caFreqOut[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::ComplexNormalize(CNeuronBaseOCL* cInputFreqComplex,
                                    CNeuronBaseOCL* cNormFreqComplex,
                                    CBufferFloat   * cMeans,
                                    CBufferFloat   * cVariances,
                                    int dimension, int variables)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {variables};
   setBuffer(def_k_ComplexNormalize, def_k_cn_inputs, cInputFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_outputs, cNormFreqComplex.getOutputIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_means, cMeans.GetIndex())
   setBuffer(def_k_ComplexNormalize, def_k_cn_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalize, def_k_cn_dimension, (int)(dimension))
   kernelExecute(def_k_ComplexNormalize, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cNormFreqComplex.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCATCH::ComplexNormalizeGradient(CNeuronBaseOCL* cInputFreqComplex,
      CNeuronBaseOCL* cNormFreqComplex,
      CBufferFloat   * cVariances,
      int dimension, int variables)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {variables};
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_inputs_gr, cInputFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_outputs_gr, cNormFreqComplex.getGradientIndex())
   setBuffer(def_k_ComplexNormalizeGradient, def_k_cng_vars, cVariances.GetIndex())
   setArgument(def_k_ComplexNormalizeGradient, def_k_cng_dimension, (int)(dimension))
   kernelExecute(def_k_ComplexNormalizeGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cInputFreqComplex.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                uint time_step, uint variables, uint skills,
                                uint window, uint step, uint window_key, uint heads,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSoftMaxOCL::Init(numOutputs, myIndex, open_cl, variables * skills,
                               optimization_type, batch))
      ReturnFalse;
   SetHeads(variables);
//---
   int index = 0;
   if(!cCrossObservAttention.Init(0, index, OpenCL, time_step, variables, window,
                                  step, window_key, heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose.Init(0, index, OpenCL, time_step, variables, optimization, iBatch))
      ReturnFalse;
//---
   uint count = (time_step - (window - step) + (step - 1)) / step;
   if(count <= 1)
     {
      window = time_step;
      count = 1;
     }
//---
   index++;
   if(!cSkillsProjection[0].Init(0, index, OpenCL, window, step, (window_key + 1) / 2, count,
                                 variables, optimization, iBatch))
      ReturnFalse;
   cSkillsProjection[0].SetActivationFunction(SoftPlus);
   index++;
   if(!cSkillsProjection[1].Init(0, index, OpenCL, (window_key + 1) / 2 * count, (window_key + 1) / 2 * count,
                                 skills, 1, variables, optimization, iBatch))
      ReturnFalse;
   cSkillsProjection[1].SetActivationFunction(None);
//---
   index++;
   if(!cPrevSkillsConcat.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cPrevSkillsConcat.SetActivationFunction((ENUM_ACTIVATION)cSkillsProjection[1].Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cCrossObservAttention.FeedForward(NeuronOCL))
      ReturnFalse;
//cCrossObservAttention.getOutput().BufferRead();
   if(!cTranspose.FeedForward(cCrossObservAttention.AsObject()))
      ReturnFalse;
   if(!cSkillsProjection[0].FeedForward(cTranspose.AsObject()))
      ReturnFalse;
   if(!cSkillsProjection[1].SwapOutputs() ||
      !cSkillsProjection[1].FeedForward(cSkillsProjection[0].AsObject()))
      ReturnFalse;
//---
   return CNeuronSoftMaxOCL::feedForward(cSkillsProjection[1].AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   if(!CNeuronSoftMaxOCL::calcInputGradients(cSkillsProjection[1].AsObject()))
      ReturnFalse;
//---
   if(!Concat(cSkillsProjection[1].getOutput(), cSkillsProjection[1].getPrevOutput(),
              cPrevSkillsConcat.getOutput(), cSkillsProjection[1].GetFilters(),
              cSkillsProjection[1].GetFilters(), iHeads))
      ReturnFalse;
   if(!DiversityLoss(cPrevSkillsConcat.AsObject(), 2 * iHeads, cSkillsProjection[1].GetFilters(), false))
      ReturnFalse;
   if(!DeConcat(cPrevSkillsConcat.getOutput(), cPrevSkillsConcat.getPrevOutput(),
                cPrevSkillsConcat.getGradient(), cSkillsProjection[1].GetFilters(),
                cSkillsProjection[1].GetFilters(), iHeads))
      ReturnFalse;
   if(!SumAndNormalize(cSkillsProjection[1].getGradient(), cPrevSkillsConcat.getOutput(),
                       cSkillsProjection[1].getGradient(), cSkillsProjection[1].GetFilters(),
                       false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cSkillsProjection[0].CalcHiddenGradients(cSkillsProjection[1].AsObject()))
      ReturnFalse;
   if(!cTranspose.CalcHiddenGradients(cSkillsProjection[0].AsObject()))
      ReturnFalse;
   if(!cCrossObservAttention.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return prevLayer.CalcHiddenGradients(cCrossObservAttention.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cCrossObservAttention.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSkillsProjection[0].UpdateInputWeights(cTranspose.AsObject()))
      ReturnFalse;
   if(!cSkillsProjection[1].UpdateInputWeights(cSkillsProjection[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronSkillsEncoder *Source = source;
   if(!cCrossObservAttention.WeightsUpdate(GetPointer(Source.cCrossObservAttention), tau))
      ReturnFalse;
   for(uint i = 0; i < cSkillsProjection.Size(); i++)
      if(!cSkillsProjection[i].WeightsUpdate(GetPointer(Source.cSkillsProjection[i]), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::Save(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Save(file_handle))
      ReturnFalse;
   if(!cCrossObservAttention.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cSkillsProjection.Size(); i++)
      if(!cSkillsProjection[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSkillsEncoder::Load(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossObservAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cSkillsProjection.Size(); i++)
      if(!LoadInsideLayer(file_handle, cSkillsProjection[i].AsObject()))
         ReturnFalse;
//---
   if(!cPrevSkillsConcat.Init(0, 4, OpenCL, 2 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cPrevSkillsConcat.SetActivationFunction((ENUM_ACTIVATION)cSkillsProjection[1].Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSkillsEncoder::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronSoftMaxOCL::SetOpenCL(obj);
   cCrossObservAttention.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
   for(uint i = 0; i < cSkillsProjection.Size(); i++)
      cSkillsProjection[i].SetOpenCL(OpenCL);
   cPrevSkillsConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                    uint window, uint units_count, uint & periods[], uint freqs,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, 2 * units_count * freqs * periods.Size(), optimization_type, batch))
      ReturnFalse;
//---
   cPeriods.BufferFree();
   if(!cPeriods.AssignArray(periods) ||
      !cPeriods.BufferCreate(OpenCL))
      ReturnFalse;
//---
   int index = 0;
   if(!cProjection.Init(0, index, OpenCL, window, window, 2 * freqs * cPeriods.Total(), units_count, 1, optimization, iBatch))
      ReturnFalse;
   cProjection.SetActivationFunction(TANH);
   index++;
   if(!cNorm.Init(0, index, OpenCL, cProjection.Neurons(), iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   SetActivationFunction(None);
   if(!SetGradient(cNorm.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::AddPE(CBufferFloat* time)
  {
   if(!time)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {GetUnits(), GetWindowOut() / (2 * cPeriods.Total()), cPeriods.Total()};
   uint kernel = def_k_TSPositonEncoder;
   setBuffer(kernel, def_k_tspe_data, cNorm.getOutputIndex())
   setBuffer(kernel, def_k_tspe_time, time.GetIndex())
   setBuffer(kernel, def_k_tspe_output, getOutputIndex())
   setBuffer(kernel, def_k_tspe_period, cPeriods.GetIndex())
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
bool CNeuronTSPositionEncoder::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNorm.FeedForward(cProjection.AsObject()))
      ReturnFalse;
//---
   return AddPE(SecondInput);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cProjection.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
//---
   return NeuronOCL.CalcHiddenGradients(cProjection.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronTSPositionEncoder* Source = source;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cPeriods.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTSPositionEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cPeriods.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!SetGradient(cNorm.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTSPositionEncoder::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cProjection.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
   cPeriods.BufferFree();
   cPeriods.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvWPadOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint & windows[],
      uint step, uint window_out, uint units_count, uint variables,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(windows.Size() <= 0 || ArrayCopy(aiWindows, windows) < int(windows.Size()))
      ReturnFalse;
   int window = 0;
   for(uint i = 0; i < aiWindows.Size(); i++)
      window += aiWindows[i] + 1;
   window--;
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window, step, window_out, units_count * aiWindows.Size() * variables, 1, ADAM, batch))
      ReturnFalse;
//---
   iVariables = variables;
   int temp = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
   if(temp < 0)
      ReturnFalse;
   iWindow = (uint)temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvWPadOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = { GetUnits(), aiWindows.Size(), iVariables};
//---
   uint inputs = NeuronOCL.Neurons() / iVariables;
   for(uint id = 0; id < global_work_size[0]; id++)
     {
      for(uint id_w = 0; id_w < global_work_size[1]; id_w++)
        {
         for(uint v = 0; id < global_work_size[2]; id++)
           {
            //---
            int window_in = aiWindows[id_w];
            int mid_win = window_in / 2;
            int shift_in = int(id * iStep) - mid_win;
            uint shift_in_var = v * inputs;
            uint shift_weight = 0;
            //---
            for(uint w = 0; w < id_w; w++)
               shift_weight += (aiWindows[w] + 1) * iWindowOut;
            //---
            for(uint w_out = 0; w_out < iWindowOut; w_out++)
              {
               float sum = WeightsConv[shift_weight + window_in];
               //---
               for(int w = 0; w < window_in; w++)
                  if((shift_in + w) >= 0 && (shift_in + w) < int(inputs))
                     sum += NeuronOCL.getOutput()[shift_in_var + shift_in + w] * WeightsConv[shift_weight + w];
               //---
               uint shift_out = (v * global_work_size[0] + id) * iWindowOut + w_out;
               Output.Update(shift_out, sum);
               shift_weight += window_in + 1;
              }
           }
        }
     }
//---
   ResetLastError();
   int kernel = def_k_FeedForwardMultWinConvWPad;
   setBuffer(kernel, def_k_ffmwconvwpad_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ffmwconvwpad_matrix_o, getOutputIndex())
   setBuffer(kernel, def_k_ffmwconvwpad_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_ffmwconvwpad_windows_in, iWindow)
   setArgument(kernel, def_k_ffmwconvwpad_inputs, NeuronOCL.Neurons() / iVariables)
   setArgument(kernel, def_k_ffmwconvwpad_window_out, iWindowOut)
   setArgument(kernel, def_k_ffmwconvwpad_step, (int)iStep)
   setArgument(kernel, def_k_ffmwconvwpad_activation, (int)activation)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvWPadOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint filters = aiWindows.Size() * iWindowOut;
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {NeuronOCL.Neurons() / iVariables,
                               (uint)MathMin(filters, OpenCL.GetMaxLocalSize(1)),
                               iVariables
                              };
   uint local_work_size[3] = {1, global_work_size[1], 1};
   ResetLastError();
   int kernel = def_k_CalcHiddenGradientMultWinConvWPad;
   setBuffer(kernel, def_k_hgmwconvwpad_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_hgmwconvwpad_matrix_ig, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_hgmwconvwpad_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_hgmwconvwpad_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_hgmwconvwpad_windows_in, iWindow)
   setArgument(kernel, def_k_hgmwconvwpad_outputs, GetUnits())
   setArgument(kernel, def_k_hgmwconvwpad_window_out, (int)iWindowOut)
   setArgument(kernel, def_k_hgmwconvwpad_step, (int)iStep)
   setArgument(kernel, def_k_hgmwconvwpad_filters, (int)filters)
   setArgument(kernel, def_k_hgmwconvwpad_activation, NeuronOCL.Activation())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
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
bool CNeuronMultiWindowsConvWPadOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {WeightsConv.Total(), iVariables};
   uint local_work_size[2] = {1, iVariables};
   ResetLastError();
   int kernel = def_k_UpdateWeightsMultWinConvAdamWPad;
   setBuffer(kernel, def_k_uwmwconvwpad_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_uwmwconvwpad_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_uwmwconvwpad_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_uwmwconvwpad_windows_in, iWindow)
   setBuffer(kernel, def_k_uwmwconvwpad_matrix_m, FirstMomentumConv.GetIndex())
   setBuffer(kernel, def_k_uwmwconvwpad_matrix_v, SecondMomentumConv.GetIndex())
   setArgument(kernel, def_k_uwmwconvwpad_inputs, NeuronOCL.Neurons() / iVariables)
   setArgument(kernel, def_k_uwmwconvwpad_outputs, GetUnits())
   setArgument(kernel, def_k_uwmwconvwpad_window_out, iWindowOut)
   setArgument(kernel, def_k_uwmwconvwpad_windows_total, (int)aiWindows.Size())
   setArgument(kernel, def_k_uwmwconvwpad_l, lr)
   setArgument(kernel, def_k_uwmwconvwpad_b1, b1)
   setArgument(kernel, def_k_uwmwconvwpad_b2, b2)
   setArgument(kernel, def_k_uwmwconvwpad_step, (int)iStep)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvWPadOCL::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)aiWindows.Size()) < INT_VALUE)
      ReturnFalse;
   for(int i = 0; i < (int)aiWindows.Size(); i++)
      if(FileWriteInteger(file_handle, (int)aiWindows[i]) < INT_VALUE)
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiWindowsConvWPadOCL::Load(const int file_handle)
  {
   ArrayFree(aiWindows);
   if(iWindow >= 0)
      OpenCL.BufferFree(iWindow);
//---
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   iWindow = INVALID_HANDLE;
//---
   int total = FileReadInteger(file_handle);
   if(ArrayResize(aiWindows, total) != total)
      ReturnFalse;
   for(int i = 0; i < total; i++)
      aiWindows[i] = FileReadInteger(file_handle);
   int temp = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
   if(temp < 0)
      ReturnFalse;
   iWindow = (uint)temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiWindowsConvWPadOCL::SetOpenCL(COpenCLMy * obj)
  {
   if(!!OpenCL && iWindow >= 0)
      OpenCL.BufferFree(iWindow);
   CNeuronConvOCL::SetOpenCL(obj);
   iWindow = OpenCL.AddBufferFromArray(aiWindows, 0, aiWindows.Size(), CL_MEM_READ_ONLY);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatDiff::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                             uint units_count, uint step, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, 2 * units_count * variables, optimization_type, batch))
      ReturnFalse;
   if(step <= 0 || step >= units_count)
      ReturnFalse;
//---
   iUnits = units_count;
   iVariables = variables;
   iStep = step;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatDiff::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || !Output)
      ReturnFalse;
   if(getOutputIndex() < 0)
      ReturnFalse;
//---
     {
      uint global_work_offset[2] = {0};
      uint global_work_size[2] = {iUnits, iVariables};
      const int kernel = def_k_ConcatDiff;
      setBuffer(kernel, def_k_concdiff_data, NeuronOCL.getOutputIndex())
      setBuffer(kernel, def_k_concdiff_output, getOutputIndex())
      setArgument(kernel, def_k_concdiff_step, iStep)
      //---
      kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!Output.BufferRead())
         ReturnFalse;
#endif
     }
//---
     {
      uint global_work_offset[1] = {0};
      uint global_work_size[1] = {2 * iUnits};
      const int kernel = def_k_Normalize;
      setBuffer(kernel, def_k_norm_buffer, getOutputIndex())
      setArgument(kernel, def_k_norm_dimension, iVariables)
      kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!Output.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatDiff::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), getPrevOutput(), getGradient(), iVariables, iVariables, iUnits))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatDiff::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!FileWriteInteger(file_handle, int(iUnits)))
      ReturnFalse;
   if(!FileWriteInteger(file_handle, int(iVariables)))
      ReturnFalse;
   if(!FileWriteInteger(file_handle, int(iStep)))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConcatDiff::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iStep = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                 uint count, uint patchs, uint variables,
                                 uint embedding_size, uint patch_filters,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, variables * embedding_size, patchs, optimization_type, batch))
      ReturnFalse;
   int index = 0;
   if(!cToVarSeq.Init(0, index, OpenCL, count, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cProjecting.Init(0, index, OpenCL, 3, 1, embedding_size, count - 2, variables, optimization, iBatch))
      ReturnFalse;
   cProjecting.SetActivationFunction(SoftPlus);
   index++;
   if(!cToVarProjSeq.Init(0, index, OpenCL, variables, count - 2, embedding_size, optimization, iBatch))
      ReturnFalse;
   index++;
   int patch_size = (int(count + patchs) - 3) / int(patchs);
   if(!cPatchingProj.Init(0, index, OpenCL, patch_size, patch_size, patch_filters, patchs, variables * embedding_size, optimization, iBatch))
      ReturnFalse;
   cPatchingProj.SetActivationFunction(SoftPlus);
   index++;
   if(!cProof.Init(0, index, OpenCL, patch_filters, patch_filters, patchs * variables * embedding_size, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cToVarSeq.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cProjecting.FeedForward(cToVarSeq.AsObject()))
      ReturnFalse;
   if(!cToVarProjSeq.FeedForward(cProjecting.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.FeedForward(cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!cProof.FeedForward(cPatchingProj.AsObject()))
      ReturnFalse;
   if(!CNeuronTransposeOCL::feedForward(cProof.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronTransposeOCL::calcInputGradients(cProof.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.CalcHiddenGradients(cProof.AsObject()))
      ReturnFalse;
   if(!cToVarProjSeq.CalcHiddenGradients(cPatchingProj.AsObject()))
      ReturnFalse;
   if(!cProjecting.CalcHiddenGradients(cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!cToVarSeq.CalcHiddenGradients(cProjecting.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cToVarSeq.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cProjecting.UpdateInputWeights(cToVarSeq.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.UpdateInputWeights(cToVarProjSeq.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::Save(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Save(file_handle))
      ReturnFalse;
   if(!cToVarSeq.Save(file_handle))
      ReturnFalse;
   if(!cProjecting.Save(file_handle))
      ReturnFalse;
   if(!cToVarProjSeq.Save(file_handle))
      ReturnFalse;
   if(!cPatchingProj.Save(file_handle))
      ReturnFalse;
   if(!cProof.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::Load(const int file_handle)
  {
   if(!CNeuronTransposeOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cToVarSeq.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjecting.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatchingProj.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProof.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMantisPatching::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronTransposeOCL::SetOpenCL(obj);
   cToVarSeq.SetOpenCL(obj);
   cProjecting.SetOpenCL(obj);
   cToVarProjSeq.SetOpenCL(obj);
   cPatchingProj.SetOpenCL(obj);
   cProof.SetOpenCL(obj);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisPatching::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronTransposeOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronMantisPatching *Source = source;
   if(!cProjecting.WeightsUpdate(Source.cProjecting.AsObject(), tau))
      ReturnFalse;
   if(!cPatchingProj.WeightsUpdate(Source.cPatchingProj.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                      uint token_size, uint window, uint window_key,
                                      uint heads, uint units_count, uint layers, uint variables,
                                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSoftMaxOCL::Init(numOutputs, myIndex, open_cl, token_size, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cClassToken[0].Init(token_size, index, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   if(!cClassToken[0].getOutput().Fill(1))
      ReturnFalse;
   index++;
   if(!cClassToken[1].Init(0, index, OpenCL, token_size, optimization, iBatch))
      ReturnFalse;
   cClassToken[1].SetActivationFunction(SIGMOID);
//---
   index++;
   if(!cAttention.Init(0, index, OpenCL, token_size, window_key, heads * variables, window,
                       heads, 1, units_count, layers, 1, 1, variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(bTrain)
     {
      if(!cClassToken[1].FeedForward(cClassToken[0].AsObject()))
         ReturnFalse;
     }
   if(!cAttention.FeedForward(cClassToken[1].AsObject(), NeuronOCL.getOutput()))
      ReturnFalse;
   if(!CNeuronSoftMaxOCL::feedForward(cAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSoftMaxOCL::calcInputGradients(cAttention.AsObject()))
      ReturnFalse;
   if(!cClassToken[1].CalcHiddenGradients(cAttention.AsObject(), NeuronOCL.getOutput(),
                                          NeuronOCL.getGradient(),
                                          (ENUM_ACTIVATION)NeuronOCL.Activation()))
      ReturnFalse;
   if(!cClassToken[0].CalcHiddenGradients(cClassToken[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cClassToken[1].UpdateInputWeights(cClassToken[0].AsObject()))
      ReturnFalse;
   if(!cAttention.UpdateInputWeights(cClassToken[1].AsObject(), NeuronOCL.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::Save(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cClassToken.Size(); i++)
      if(!cClassToken[i].Save(file_handle))
         ReturnFalse;
   if(!cAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::Load(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Load(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cClassToken.Size(); i++)
      if(!LoadInsideLayer(file_handle, cClassToken[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMantisAttentionUnit::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronSoftMaxOCL::SetOpenCL(obj);
   for(uint i = 0; i < cClassToken.Size(); i++)
      cClassToken[i].SetOpenCL(OpenCL);
   cAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMantisAttentionUnit::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronSoftMaxOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMantisAttentionUnit *Source = source;
   for(uint i = 0; i < cClassToken.Size(); i++)
      if(!cClassToken[i].WeightsUpdate(Source.cClassToken[i].AsObject(), tau))
         ReturnFalse;
   if(!cAttention.WeightsUpdate(Source.cAttention.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                    uint count, uint patchs, uint variables,
                                    uint embedding_size, uint patch_filters,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint inside_emb = MathMax((embedding_size + 2) / 6, 8);
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, 3 * inside_emb, 3 * inside_emb,
                            embedding_size, patchs * variables, 1, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(SoftPlus);
   int index = 0;
   if(!cToVarSeq.Init(0, index, OpenCL, count, variables, optimization, iBatch))
      ReturnFalse;
   index++;
     {
      uint windows[] = {3, 5, 7};
      if(!cProjecting.Init(0, index, OpenCL, windows, 1, inside_emb, count, variables,
                           optimization, iBatch))
         ReturnFalse;
      cProjecting.SetActivationFunction(SoftPlus);
     }
   index++;
   if(!cToVarProjSeq.Init(0, index, OpenCL, variables, count, 3 * inside_emb, optimization, iBatch))
      ReturnFalse;
   index++;
     {
      uint step = (count + patchs) / (patchs + 1);
      uint windows[] = {step, 3 * step / 2, 2 * step};
      if(!cPatchingProj.Init(0, index, OpenCL, windows, step, patch_filters, patchs,
                             3 * inside_emb * variables, optimization, iBatch))
         ReturnFalse;
      cPatchingProj.SetActivationFunction(SoftPlus);
     }
   index++;
   if(!cPatchsFeedForward.Init(0, index, OpenCL, 3 * patch_filters, 6 * patch_filters, patchs,
                               3 * inside_emb * variables, 3, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPatchingAgr.Init(0, index, OpenCL, 3 * patch_filters, 3 * patch_filters, patch_filters,
                         patchs, 3 * inside_emb * variables, optimization, iBatch))
      ReturnFalse;
   cPatchingAgr.SetActivationFunction(SoftPlus);
   index++;
   if(!cProof.Init(0, index, OpenCL, patch_filters, patch_filters, 3 * patchs * inside_emb * variables,
                   optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cToPatchVarEmb.Init(0, index, OpenCL, variables * 3 * inside_emb, patchs, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cToVarSeq.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cProjecting.FeedForward(cToVarSeq.AsObject()))
      ReturnFalse;
   if(!cToVarProjSeq.FeedForward(cProjecting.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.FeedForward(cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!cPatchsFeedForward.FeedForward(cPatchingProj.AsObject()))
      ReturnFalse;
   if(!cPatchingAgr.FeedForward(cPatchsFeedForward.AsObject()))
      ReturnFalse;
   if(!cProof.FeedForward(cPatchingAgr.AsObject()))
      ReturnFalse;
   if(!cToPatchVarEmb.FeedForward(cProof.AsObject()))
      ReturnFalse;
//---
   return CNeuronConvOCL::feedForward(cToPatchVarEmb.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronConvOCL::calcInputGradients(cToPatchVarEmb.AsObject()))
      ReturnFalse;
   if(!cProof.CalcHiddenGradients(cToPatchVarEmb.AsObject()))
      ReturnFalse;
   if(cPatchingAgr.Activation() != None)
      if(!DeActivation(cPatchingAgr.getOutput(), cPatchingAgr.getGradient(), cPatchingAgr.getGradient(), cPatchingAgr.Activation()))
         ReturnFalse;
   if(!cPatchingAgr.CalcHiddenGradients(cProof.AsObject()))
      ReturnFalse;
   if(!cPatchsFeedForward.CalcHiddenGradients(cPatchingAgr.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.CalcHiddenGradients(cPatchsFeedForward.AsObject()))
      ReturnFalse;
   if(!cToVarProjSeq.CalcHiddenGradients(cPatchingProj.AsObject()))
      ReturnFalse;
   if(!cProjecting.CalcHiddenGradients(cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!cToVarSeq.CalcHiddenGradients(cProjecting.AsObject()))
      ReturnFalse;
//---
   return NeuronOCL.CalcHiddenGradients(cToVarSeq.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cProjecting.UpdateInputWeights(cToVarSeq.AsObject()))
      ReturnFalse;
   if(!cPatchingProj.UpdateInputWeights(cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!cPatchsFeedForward.UpdateInputWeights(cPatchingProj.AsObject()))
      ReturnFalse;
   if(!cPatchingAgr.UpdateInputWeights(cPatchsFeedForward.AsObject()))
      ReturnFalse;
//---
   return CNeuronConvOCL::updateInputWeights(cToPatchVarEmb.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronConvOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronTimeFoundPatching *Source = source;
   if(!cProjecting.WeightsUpdate(Source.cProjecting.AsObject(), tau))
      ReturnFalse;
   if(!cPatchingProj.WeightsUpdate(Source.cPatchingProj.AsObject(), tau))
      ReturnFalse;
   if(!cPatchsFeedForward.WeightsUpdate(Source.cPatchsFeedForward.AsObject(), tau))
      ReturnFalse;
   if(!cPatchingAgr.WeightsUpdate(Source.cPatchingAgr.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(!cToVarSeq.Save(file_handle))
      ReturnFalse;
   if(!cProjecting.Save(file_handle))
      ReturnFalse;
   if(!cToVarProjSeq.Save(file_handle))
      ReturnFalse;
   if(!cPatchingProj.Save(file_handle))
      ReturnFalse;
   if(!cPatchsFeedForward.Save(file_handle))
      ReturnFalse;
   if(!cPatchingAgr.Save(file_handle))
      ReturnFalse;
   if(!cProof.Save(file_handle))
      ReturnFalse;
   if(!cToPatchVarEmb.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundPatching::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cToVarSeq.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjecting.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cToVarProjSeq.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatchingProj.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatchsFeedForward.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPatchingAgr.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProof.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cToPatchVarEmb.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTimeFoundPatching::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   cToVarSeq.SetOpenCL(OpenCL);
   cProjecting.SetOpenCL(OpenCL);
   cToVarProjSeq.SetOpenCL(OpenCL);
   cPatchingProj.SetOpenCL(OpenCL);
   cPatchsFeedForward.SetOpenCL(OpenCL);
   cPatchingAgr.SetOpenCL(OpenCL);
   cProof.SetOpenCL(OpenCL);
   cToPatchVarEmb.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window,
      uint window_key, uint heads, uint heads_kv, uint units_count,
      uint layers, uint layers_to_one_kv, uint variables,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronCrossDMHAttention::Init(numOutputs, myIndex, open_cl, window, window_key, variables, window,
                                      units_count * variables, heads, layers, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
   if(!cSelfAttention.Init(0, 0, OpenCL, window, window_key, heads, heads_kv, units_count, layers, layers_to_one_kv,
                           variables, optimization, iBatch))
      ReturnFalse;
   cSelfAttention.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cSelfAttention.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronCrossDMHAttention::feedForward(NeuronOCL, cSelfAttention.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   CBufferFloat *grad = prevLayer.getGradient();
   if(!grad || !grad.Fill(0))
      ReturnFalse;
//---
   if(!CNeuronCrossDMHAttention::calcInputGradients(prevLayer, cSelfAttention.getOutput(),
         cSelfAttention.getPrevOutput(),
         (ENUM_ACTIVATION)cSelfAttention.Activation()))
      ReturnFalse;
   if(!prevLayer.SetGradient(cSelfAttention.getPrevOutput(), false) ||
      !prevLayer.CalcHiddenGradients(cSelfAttention.AsObject()) ||
      !SumAndNormalize(grad, prevLayer.getGradient(), grad, GetWindow(), false, 0, 0, 0, 1) ||
      !prevLayer.SetGradient(grad, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cSelfAttention.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronCrossDMHAttention::updateInputWeights(NeuronOCL, cSelfAttention.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronCrossDMHAttention::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronTimeFoundTransformerUnit *Source = source;
   if(!cSelfAttention.WeightsUpdate(Source.cSelfAttention.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::Save(const int file_handle)
  {
   if(!CNeuronCrossDMHAttention::Save(file_handle))
      ReturnFalse;
   if(!cSelfAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeFoundTransformerUnit::Load(const int file_handle)
  {
   if(!CNeuronCrossDMHAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSelfAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTimeFoundTransformerUnit::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronCrossDMHAttention::SetOpenCL(obj);
   cSelfAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window,
                                       uint window_out, uint units_count, uint variables, uint experts,
                                       uint topK, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   int index = 0;
   if(!cExpertsIn.Init(0, index, OpenCL, window, window, window_out * experts, units_count, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSharedIn.Init(0, index, OpenCL, window, window, window_out, units_count, variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cExpertsOut.Init(0, index, OpenCL, window_out, experts, window, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cExpertsOut.SetActivationFunction(None);
   index++;
   if(!cSharedOut.Init(0, index, OpenCL, window_out, window_out, window, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cSharedOut.SetActivationFunction(None);
   index++;
   if(!cMasks.Init(0, index, OpenCL, window, units_count, experts, topK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSharedGates.Init(0, index, OpenCL, window, window, window, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cSharedGates.SetActivationFunction(SIGMOID);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cExpertsIn.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSharedIn.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSharedGates.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMasks.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cExpertsOut.FeedForward(cExpertsIn.AsObject(), cMasks.getOutput()))
      ReturnFalse;
   if(!cSharedOut.FeedForward(cSharedIn.AsObject()))
      ReturnFalse;
   if(!ElementMult(cSharedOut.getOutput(), cSharedGates.getOutput(), cSharedOut.getPrevOutput()))
      ReturnFalse;
   const int window = (int)cSharedGates.GetWindow();
   if(!SumAndNormalize(cExpertsOut.getOutput(), cSharedOut.getPrevOutput(), PrevOutput,
                       window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), PrevOutput, Output,
                       window, true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!DeActivation(cExpertsOut.getOutput(), cExpertsOut.getGradient(), Gradient, cExpertsOut.Activation()))
      ReturnFalse;
   if(!ElementMultGrad(cSharedOut.getOutput(), cSharedOut.getGradient(),
                       cSharedGates.getOutput(), cSharedGates.getGradient(),
                       Gradient, cSharedOut.Activation(), cSharedGates.Activation()))
      ReturnFalse;
//---
   if(!cExpertsIn.CalcHiddenGradients(cExpertsOut.AsObject(), cMasks.getOutput(),
                                      cMasks.getGradient(), (ENUM_ACTIVATION)cMasks.Activation()))
      ReturnFalse;
   if(!cSharedIn.CalcHiddenGradients(cSharedOut.AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cExpertsIn.AsObject()))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), PrevOutput, Gradient, NeuronOCL.Activation()))
      ReturnFalse;
   const int window = (int)cSharedGates.GetWindow();
   if(!SumAndNormalize(PrevOutput, NeuronOCL.getGradient(), PrevOutput, window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSharedIn.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(PrevOutput, NeuronOCL.getGradient(), PrevOutput, window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMasks.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(PrevOutput, NeuronOCL.getGradient(), PrevOutput, window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSharedGates.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(PrevOutput, NeuronOCL.getGradient(), NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cExpertsIn.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSharedIn.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSharedGates.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMasks.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cExpertsOut.UpdateInputWeights(cExpertsIn.AsObject(), cMasks.getOutput()))
      ReturnFalse;
   if(!cSharedOut.UpdateInputWeights(cSharedIn.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronTimeMoESparseExperts *Source = source;
   if(!cExpertsIn.WeightsUpdate(Source.cExpertsIn.AsObject(), tau))
      ReturnFalse;
   if(!cSharedIn.WeightsUpdate(Source.cSharedIn.AsObject(), tau))
      ReturnFalse;
   if(!cSharedGates.WeightsUpdate(Source.cSharedGates.AsObject(), tau))
      ReturnFalse;
   if(!cMasks.WeightsUpdate(Source.cMasks.AsObject(), tau))
      ReturnFalse;
   if(!cExpertsOut.WeightsUpdate(Source.cExpertsOut.AsObject(), tau))
      ReturnFalse;
   if(!cSharedOut.WeightsUpdate(Source.cSharedOut.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cExpertsIn.Save(file_handle))
      ReturnFalse;
   if(!cSharedIn.Save(file_handle))
      ReturnFalse;
   if(!cExpertsOut.Save(file_handle))
      ReturnFalse;
   if(!cSharedOut.Save(file_handle))
      ReturnFalse;
   if(!cMasks.Save(file_handle))
      ReturnFalse;
   if(!cSharedGates.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoESparseExperts::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cExpertsIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSharedIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cExpertsOut.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSharedOut.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMasks.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSharedGates.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTimeMoESparseExperts::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cExpertsIn.SetOpenCL(OpenCL);
   cSharedIn.SetOpenCL(OpenCL);
   cExpertsOut.SetOpenCL(OpenCL);
   cSharedOut.SetOpenCL(OpenCL);
   cMasks.SetOpenCL(OpenCL);
   cSharedGates.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTimeMoESparseExperts::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cExpertsIn.TrainMode(bTrain);
   cSharedIn.TrainMode(bTrain);
   cExpertsOut.TrainMode(bTrain);
   cSharedOut.TrainMode(bTrain);
   cMasks.TrainMode(bTrain);
   cSharedGates.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoEAttention::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                   uint window, uint window_key, uint units_count,
                                   uint window_cross, uint units_cross, uint heads,
                                   uint layers, uint experts, uint experts_dimension,
                                   uint topK, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
//---
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
   CNeuronRelativeSelfAttention *attention = NULL;
   CNeuronRelativeCrossAttention *cross = NULL;
   CNeuronTimeMoESparseExperts *MoE = NULL;
   bool use_self = units_count > 0;
   int layer = 0;
   for(uint i = 0; i < layers; i++)
     {
      if(use_self)
        {
         attention = new CNeuronRelativeSelfAttention();
         if(!attention ||
            !attention.Init(0, layer, OpenCL, window, window_key, units_count, heads, optimization, iBatch) ||
            !cLayers.Add(attention)
           )
           {
            DeleteObj(attention);
            ReturnFalse;
           }
         layer++;
        }
      cross = new CNeuronRelativeCrossAttention();
      if(!cross ||
         !cross.Init(0, layer, OpenCL, window, window_key, units_count, heads, window_cross, units_cross, optimization, iBatch) ||
         !cLayers.Add(cross)
        )
        {
         DeleteObj(cross);
         ReturnFalse;
        }
      layer++;
      MoE = new CNeuronTimeMoESparseExperts();
      if(!MoE ||
         !MoE.Init(0, layer, OpenCL, window, experts_dimension, units_count, 1, experts, topK, optimization, iBatch) ||
         !cLayers.Add(MoE)
        )
        {
         DeleteObj(MoE);
         ReturnFalse;
        }
      layer++;
     }
//---
   SetOutput(MoE.getOutput(), true);
   SetGradient(MoE.getGradient(), true);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoEAttention::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   return CNeuronCrossDMHAttention::feedForward(NeuronOCL, NeuronOCL.getOutput());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoEAttention::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!prevLayer)
      ReturnFalse;
   CBufferFloat *temp = prevLayer.getGradient();
   if(!temp || !temp.Fill(0))
      ReturnFalse;
   if(!CNeuronCrossDMHAttention::calcInputGradients(prevLayer, prevLayer.getOutput(),
         prevLayer.getPrevOutput(), (ENUM_ACTIVATION) prevLayer.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(temp, prevLayer.getPrevOutput(), temp, 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTimeMoEAttention::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   return CNeuronCrossDMHAttention::updateInputWeights(NeuronOCL, NeuronOCL.getOutput());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                            uint window, uint window_out, uint units_count, uint variables,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window, window, window_out, units_count, variables, optimization_type, batch))
      ReturnFalse;
//---
   bMainFreq.BufferFree();
   if(!bMainFreq.BufferInit(iVariables, 1) ||
      !bMainFreq.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::FFT(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* out_re, CBufferFloat* out_im, uint variables, bool reverse = false)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {variables};
   uint kernel = def_k_FFT;
   setBuffer(kernel, def_k_fft_inputs_re, inp_re.GetIndex())
   setBuffer(kernel, def_k_fft_inputs_im, (!!inp_im ? inp_im.GetIndex() : inp_re.GetIndex()))
   setBuffer(kernel, def_k_fft_outputs_re, out_re.GetIndex())
   setBuffer(kernel, def_k_fft_outputs_im, out_im.GetIndex())
   setArgument(kernel, def_k_fft_input_window, (int)(inp_re.Total() / variables))
   setArgument(kernel, def_k_fft_input_complex, int(!!inp_im))
   setArgument(kernel, def_k_fft_output_window, (int)(out_re.Total() / variables))
   setArgument(kernel, def_k_fft_reverse, int(reverse))
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out_im.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::PeriodsFinding(CBufferFloat* inp_re, CBufferFloat* inp_im, CBufferFloat* windows, uint variables)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {variables};
   uint kernel = def_k_MainFreq;
   setBuffer(kernel, def_k_mf_freq_r, inp_re.GetIndex())
   setBuffer(kernel, def_k_mf_freq_im, inp_im.GetIndex())
   setBuffer(kernel, def_k_mf_main_freq, windows.GetIndex())
   setArgument(kernel, def_k_mf_dimension, (int)(inp_re.Total() / variables))
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!windows.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::AdaptiveConvolution(CNeuronBaseOCL* NeuronOCL, CBufferFloat* main_freq)
  {
   uint global_work_offset[] = {0, 0, 0};
   uint global_work_size[] = {Output.Total() / (iWindowOut * iVariables), iWindowOut, iVariables};
   uint kernel = def_k_FeedForwardAdaptConv;
   setBuffer(kernel, def_k_ffac_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ffac_matrix_o, getOutputIndex())
   setBuffer(kernel, def_k_ffac_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_ffac_main_freq, main_freq.GetIndex())
   setArgument(kernel, def_k_ffac_window_in, iWindow)
   setArgument(kernel, def_k_ffac_inputs, (int)(NeuronOCL.Neurons()))
   setArgument(kernel, def_k_ffac_activation, (int)(activation))
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
bool CNeuronAdaptConv::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!FFT(NeuronOCL.getOutput(), NULL, Output, PrevOutput, iVariables, false))
      ReturnFalse;
   if(!PeriodsFinding(Output, PrevOutput, GetPointer(bMainFreq), iVariables))
      ReturnFalse;
//---
   return AdaptiveConvolution(NeuronOCL, GetPointer(bMainFreq));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL ||
      !OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {NeuronOCL.Neurons() / iVariables, iVariables};
   uint kernel = def_k_CalcHiddenGradientAdaptConv;
   setBuffer(kernel, def_k_chgac_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_chgac_matrix_ig, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_chgac_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_chgac_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_chgac_main_freq, bMainFreq.GetIndex())
   setArgument(kernel, def_k_chgac_window_in, iWindow)
   setArgument(kernel, def_k_chgac_window_out, iWindowOut)
   setArgument(kernel, def_k_chgac_activation, (int)(NeuronOCL.Activation()))
   setArgument(kernel, def_k_chgac_outputs, (int)(Neurons()))
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
bool CNeuronAdaptConv::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL ||
      !OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0, 0};
   uint global_work_size[] = {iWindow + 1, iWindowOut, iVariables};
   uint kernel = def_k_UpdateWeightsAdaptConvAdam;
   setBuffer(kernel, def_k_uwac_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_uwac_matrix_og, getGradientIndex())
   setBuffer(kernel, def_k_uwac_matrix_w, WeightsConv.GetIndex())
   setBuffer(kernel, def_k_uwac_matrix_m, FirstMomentumConv.GetIndex())
   setBuffer(kernel, def_k_uwac_matrix_v, SecondMomentumConv.GetIndex())
   setBuffer(kernel, def_k_uwac_main_freq, bMainFreq.GetIndex())
   setArgument(kernel, def_k_uwac_outputs, (int)Neurons())
   setArgument(kernel, def_k_uwac_inputs, int(NeuronOCL.Neurons()))
   setArgument(kernel, def_k_uwac_l, lr)
   setArgument(kernel, def_k_uwac_b1, b1)
   setArgument(kernel, def_k_uwac_b2, b2)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(!bMainFreq.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAdaptConv::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   if(!bMainFreq.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronAdaptConv::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   bMainFreq.BufferFree();
   bMainFreq.BufferFree();
   bMainFreq.BufferInit(iVariables, 1);
   bMainFreq.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                               uint window, uint window_key, uint units_cross,
                               uint heads, uint layers, uint scenarios,
                               uint experts, uint experts_dimension, uint topK,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * scenarios, optimization_type, batch))
      ReturnFalse;
//--- Koopman
   int index = 0;
   if(!cKoopman.Init(0, index, OpenCL, window, 2 * window, units_cross, 1, experts, topK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKoopmanPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKoopmanRest.Init(0, index, OpenCL, window * (units_cross - 1), optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cAuxiliaryNet.Init(0, index, OpenCL, window, window_key, 1, window, units_cross - 1,
                          heads, layers, experts, experts_dimension, topK, optimization, iBatch))
      ReturnFalse;
//--- Kalman Filter
   index++;
   if(!cB.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cB.Identity(window, window))
      ReturnFalse;
   index++;
   if(!cF.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cF.Identity(window, window))
      ReturnFalse;
   index++;
   if(!cH.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cH.Identity(window, window))
      ReturnFalse;
   index++;
   if(!cQ.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cQ.Identity(window, window))
      ReturnFalse;
   index++;
   if(!cR.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cR.Identity(window, window))
      ReturnFalse;
   index++;
   if(!cP.Init(0, index, OpenCL, window * window, optimization, iBatch) ||
      !cP.getOutput().Fill(matrix<float>::Identity(window, window)))
      ReturnFalse;
//---
   index++;
   if(!cFT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cQT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cRT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cQ_QT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cR_RT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cXPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cF_P.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPPred.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cP_HT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cH_P_HT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   mS = mSGrad = matrix<double>::Zeros(window, window);
//---
   index++;
   if(!cSInv.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cK.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cYPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cDeltY.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cX.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cK_H.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cIdifK_H.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cIdifK_HT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   mKL = mP = mPGrad = mS;
   mNoise = mGrad = matrix<double>::Zeros(scenarios, window);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Koopman
   if(!cKoopman.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Pred / Rest
   if(!DeConcat(cKoopmanPred.getOutput(), cKoopmanRest.getPrevOutput(), cKoopman.getOutput(),
                cKoopman.GetWindow(), cKoopman.Neurons() - cKoopman.GetWindow(), 1))
      ReturnFalse;
   if(!Different(NeuronOCL.getOutput(), cKoopmanRest.getPrevOutput(),
                 cKoopmanRest.getOutput(), cKoopman.GetWindow()))
      ReturnFalse;
//--- Rest Attention
   if(!cAuxiliaryNet.FeedForward(cKoopmanRest.AsObject()))
      ReturnFalse;
//--- Kalman Filter
   if(bTrain)
     {
      if(!cB.FeedForward())
         ReturnFalse;
      if(!cF.FeedForward())
         ReturnFalse;
      if(!cH.FeedForward())
         ReturnFalse;
      if(!cQ.FeedForward())
         ReturnFalse;
      if(!cR.FeedForward())
         ReturnFalse;
      if(!cFT.FeedForward(cF.AsObject()))
         ReturnFalse;
      if(!cHT.FeedForward(cH.AsObject()))
         ReturnFalse;
      if(!cQT.FeedForward(cQ.AsObject()))
         ReturnFalse;
      if(!cRT.FeedForward(cR.AsObject()))
         ReturnFalse;
      if(!MatMul(cQ.getOutput(), cQT.getOutput(), cQ_QT.getOutput(),
                 cQT.GetCount(), cQT.GetWindow(), cQT.GetCount(), 1, true))
         ReturnFalse;
      if(!MatMul(cR.getOutput(), cRT.getOutput(), cR_RT.getOutput(),
                 cRT.GetCount(), cRT.GetWindow(), cRT.GetCount(), 1, true))
         ReturnFalse;
     }
//--- Prediction step
   if(!MatMul(NeuronOCL.getOutput(), cFT.getOutput(), cXPred.getGradient(),
              1, cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMul(cAuxiliaryNet.getOutput(), cB.getOutput(), cXPred.getPrevOutput(),
              1, cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cXPred.getGradient(), cXPred.getPrevOutput(), cXPred.getOutput(),
                       cFT.GetCount(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   cP.SwapOutputs();
   if(!MatMul(cF.getOutput(), cP.getPrevOutput(), cF_P.getOutput(),
              cFT.GetCount(), cFT.GetWindow(), cP.Neurons() / cFT.GetWindow(), 1, true))
      ReturnFalse;
   if(!MatMul(cF_P.getOutput(), cFT.getOutput(), cPPred.getOutput(),
              cFT.GetCount(), cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cPPred.getOutput(), cQ_QT.getOutput(), cPPred.getOutput(),
                       cHT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//--- Update step
   if(!MatMul(cPPred.getOutput(), cHT.getOutput(), cP_HT.getOutput(),
              cPPred.Neurons() / cHT.GetWindow(), cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMul(cH.getOutput(), cP_HT.getOutput(), cH_P_HT.getOutput(),
              cHT.GetCount(), cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cH_P_HT.getOutput(), cR_RT.getOutput(), cR_RT.getPrevOutput(),
                       cRT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(cR_RT.getPrevOutput().GetData(mSGrad) <= 0)
      ReturnFalse;
   mS = mSGrad.Inv();
   if(mS.Rows() == 0)
     {
      double max = MathAbs(mSGrad).Max();
      if(max > 1)
         mSGrad = mSGrad / max;
      mSGrad = mSGrad + matrix<double>::Identity(mSGrad.Rows(), mSGrad.Cols());
      mS = mSGrad.Inv();
      if(mS.Rows() == 0)
        {
         mSGrad = mSGrad + mSGrad.Transpose();
         vector<double> eigvals;
         matrix<double> eigvecs;
         if(!mSGrad.Eig(eigvecs, eigvals))
            ReturnFalse;
         if(eigvals.Size() > 0)
           {
            if(!eigvals.Clip(1e-6, DBL_MAX))
               ReturnFalse;
            mSGrad = matrix<double>::Zeros(eigvals.Size(), eigvals.Size());
            mSGrad.Diag(eigvals);
            mSGrad = eigvecs.MatMul(mSGrad.MatMul(eigvecs.Transpose()));
            mSGrad = mSGrad + mSGrad.Transpose();
            mS = mSGrad.Inv();
           }
         if(mS.Rows() == 0)
           {
            mSGrad.Identity();
            mS = mSGrad;
           }
        }
     }
   cSInv.getOutput().Fill(mS);
   if(!MatMul(cP_HT.getOutput(), cSInv.getOutput(), cK.getOutput(),
              cHT.GetCount(), (int)mS.Rows(), (int)mS.Cols(), 1, true))
      ReturnFalse;
//--- Measurement update
   if(!MatMul(cXPred.getOutput(), cHT.getOutput(), cYPred.getOutput(),
              1, cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!Different(cKoopmanPred.getOutput(), cYPred.getOutput(), cDeltY.getOutput(),
                 1, 0, 0, 0, 1))
      ReturnFalse;
   if(!MatMul(cDeltY.getOutput(), cK.getOutput(), cX.getPrevOutput(), 1,
              cDeltY.Neurons(), cX.Neurons(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cX.getPrevOutput(), cXPred.getOutput(), cX.getOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Joseph stabilized form for P
   if(!MatMul(cK.getOutput(), cH.getOutput(), cK_H.getOutput(), cKT.GetCount(),
              cKT.GetWindow(), cHT.GetWindow(), 1, true))
      ReturnFalse;
   if(!IdentDifferent(cK_H.getOutput(), cIdifK_H.getOutput(), cHT.GetWindow(), 0, 0, 1))
      ReturnFalse;
   if(!cIdifK_HT.FeedForward(cIdifK_H.AsObject()))
      ReturnFalse;
   if(!cKT.FeedForward(cK.AsObject()))
      ReturnFalse;
   if(!MatMul(cIdifK_H.getOutput(), cPPred.getOutput(), cIdifK_H.getPrevOutput(),
              cIdifK_HT.GetCount(), cIdifK_HT.GetWindow(), cIdifK_HT.GetWindow(), 1, true))
      ReturnFalse;
   if(!MatMul(cIdifK_H.getPrevOutput(), cIdifK_HT.getOutput(), cP.getOutput(),
              cIdifK_HT.GetCount(), cIdifK_HT.GetWindow(), cIdifK_HT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMul(cK.getOutput(), cR_RT.getOutput(), cK.getPrevOutput(),
              cKT.GetCount(), cRT.GetCount(), cRT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMul(cK.getPrevOutput(), cKT.getOutput(), cKT.getPrevOutput(),
              cKT.GetCount(), cKT.GetWindow(), cKT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cP.getOutput(), cKT.getPrevOutput(), cP.getOutput(), cPT.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cPT.FeedForward(cP.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cP.getOutput(), cPT.getOutput(), cP.getOutput(), 1, false, 0, 0, 0, 0.5f))
      ReturnFalse;
   if(!SumAndNormalize(cX.getOutput(), cAuxiliaryNet.getOutput(), cX.getOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Sample Output
   if(!cP.getOutput().GetData(mPGrad))
      ReturnFalse;
   if(mPGrad.HasNan() > 0)
     {
      mPGrad.Identity();
      if(!cP.getOutput().Fill(mPGrad))
         ReturnFalse;
     }
   mP = mPGrad.Inv();
   if(mP.Rows() == 0)
     {
      double max = MathAbs(mPGrad).Max();
      if(max > 1)
         mPGrad = mPGrad / max;
      mPGrad = mPGrad + matrix<double>::Identity(mPGrad.Rows(), mPGrad.Cols());
      mP = mPGrad.Inv();
      if(mP.Rows() == 0)
        {
         mPGrad = mPGrad + mPGrad.Transpose();
         vector<double> eigvals;
         matrix<double> eigvecs;
         if(!mPGrad.Eig(eigvecs, eigvals))
            ReturnFalse;
         if(eigvals.Size() > 0)
           {
            if(!eigvals.Clip(1e-6, DBL_MAX))
               ReturnFalse;
            mPGrad = matrix<double>::Zeros(eigvals.Size(), eigvals.Size());
            mPGrad.Diag(eigvals);
            mPGrad = eigvecs.MatMul(mPGrad.MatMul(eigvecs.Transpose()));
            mPGrad = mPGrad + mPGrad.Transpose();
            mP = mPGrad.Inv();
           }
         if(mP.Rows() == 0)
           {
            mPGrad.Identity();
            if(!cP.getOutput().Fill(mPGrad))
               ReturnFalse;
            mP = mPGrad.Inv();
           }
         else
            if(!cP.getOutput().Fill(mPGrad))
               ReturnFalse;
        }
      else
         if(!cP.getOutput().Fill(mPGrad))
            ReturnFalse;
     }
   mNoise.Random(-1, 1);
   matrix<double> temp = mNoise.MatMul(mP);
   if(!PrevOutput.Fill(temp))
      ReturnFalse;
   if(!SumVecMatrix(cX.getOutput(), PrevOutput, Output, (int)mNoise.Cols(), 1, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- From Output
   if(!SumVecMatrixGrad(cX.getGradient(), PrevOutput, Gradient, (int)mNoise.Cols(), 1, 0, 0, 0, 1))
      ReturnFalse;
   if(!PrevOutput.GetData(mGrad))
      ReturnFalse;
   mPGrad = mNoise.Transpose().MatMul(mGrad);
   mKL = mP.Transpose();
   mKL = (mKL * (-1)).MatMul(mPGrad.MatMul(mKL));
//--- Joseph stabilized form for P
   if(!cPT.getGradient().Fill(mKL))
      ReturnFalse;
   if(!cP.CalcHiddenGradients(cPT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cP.getGradient(), cPT.getGradient(), cP.getGradient(), 1, false, 0, 0, 0, 0.5f))
      ReturnFalse;
//---
   if(!MatMulGrad(cK.getPrevOutput(), cKT.getPrevOutput(),
                  cKT.getOutput(), cKT.getGradient(),
                  cP.getGradient(), cKT.GetCount(),
                  cKT.GetWindow(), cKT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(cK.getOutput(), cK.getPrevOutput(),
                  cR_RT.getOutput(), cR_RT.getGradient(),
                  cKT.getPrevOutput(), cKT.GetCount(),
                  cRT.GetCount(), cRT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(cIdifK_H.getPrevOutput(), cIdifK_HT.getPrevOutput(),
                  cIdifK_HT.getOutput(), cIdifK_HT.getGradient(),
                  cP.getGradient(), cIdifK_HT.GetCount(),
                  cIdifK_HT.GetWindow(), cIdifK_HT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(cIdifK_H.getOutput(), cIdifK_H.getPrevOutput(),
                  cPPred.getOutput(), cPPred.getGradient(),
                  cIdifK_HT.getPrevOutput(), cIdifK_HT.GetCount(),
                  cIdifK_HT.GetWindow(), cIdifK_HT.GetWindow(), 1, true))
      ReturnFalse;
//---
   if(!cK.CalcHiddenGradients(cKT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cK.getGradient(), cK.getPrevOutput(), cK.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Measurement update
   if(!cIdifK_H.CalcHiddenGradients(cIdifK_HT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cIdifK_H.getGradient(), cIdifK_H.getPrevOutput(), cIdifK_H.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!IdentDifferentGrad(cK_H.getGradient(), cIdifK_H.getGradient(), cHT.GetWindow(), 0, 0, 1))
      ReturnFalse;
   if(!MatMulGrad(cK.getOutput(), cK.getPrevOutput(),
                  cH.getOutput(), cH.getGradient(),
                  cK_H.getGradient(), cKT.GetCount(),
                  cKT.GetWindow(), cHT.GetWindow(), 1, true))
      ReturnFalse;
//---
   if(!MatMulGrad(cDeltY.getOutput(), cDeltY.getGradient(),
                  cK.getOutput(), cK.getPrevOutput(),
                  cX.getGradient(), 1,
                  cDeltY.Neurons(), cX.Neurons(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cK.getGradient(), cK.getPrevOutput(), cK.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!DifferentGrad(cKoopmanPred.getGradient(), cYPred.getGradient(), cDeltY.getGradient(),
                     1, 0, 0, 0, 1))
      ReturnFalse;
   if(!MatMulGrad(cXPred.getOutput(), cXPred.getGradient(),
                  cHT.getOutput(), cHT.getGradient(),
                  cYPred.getGradient(),
                  1, cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cXPred.getGradient(), cX.getGradient(), cXPred.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Update step
   if(!MatMulGrad(cP_HT.getOutput(), cP_HT.getGradient(),
                  cSInv.getOutput(), cSInv.getGradient(),
                  cK.getGradient(), cHT.GetCount(),
                  (int)mS.Rows(), (int)mS.Cols(), 1, true))
      ReturnFalse;
   if(cSInv.getGradient().GetData(mSGrad) <= 0)
      ReturnFalse;
   mKL = mS.Transpose();
   mKL = (mKL * (-1)).MatMul(mSGrad.MatMul(mKL));
   if(cH_P_HT.getGradient().Fill(mKL) <= 0)
      ReturnFalse;
   if(!MatMulGrad(cH.getOutput(), cH.getPrevOutput(),
                  cP_HT.getOutput(), cP_HT.getPrevOutput(),
                  cH_P_HT.getGradient(), cHT.GetCount(),
                  cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cH_P_HT.getGradient(), cR_RT.getGradient(),
                       cR_RT.getGradient(), int(mS.Cols()), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cH.getGradient(), cH.getPrevOutput(),
                       cH.getPrevOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cP_HT.getGradient(), cP_HT.getPrevOutput(),
                       cP_HT.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!MatMulGrad(cPPred.getOutput(), cPPred.getPrevOutput(),
                  cHT.getOutput(), cHT.getPrevOutput(),
                  cP_HT.getGradient(), cPPred.Neurons() / cHT.GetWindow(),
                  cHT.GetWindow(), cHT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cPPred.getGradient(), cPPred.getPrevOutput(),
                       cQ_QT.getGradient(), int(cQT.GetWindow()), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cHT.getGradient(), cHT.getPrevOutput(), cHT.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Prediction step
   if(!MatMulGrad(cF_P.getOutput(), cF_P.getGradient(),
                  cFT.getOutput(), cFT.getGradient(),
                  cQ_QT.getGradient(), cFT.GetCount(),
                  cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(cF.getOutput(), cF.getPrevOutput(),
                  cP.getPrevOutput(), cP.getGradient(),
                  cF_P.getGradient(),
                  cFT.GetCount(), cFT.GetWindow(), cP.Neurons() / cFT.GetWindow(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(cAuxiliaryNet.getOutput(), cAuxiliaryNet.getGradient(),
                  cB.getOutput(), cB.getGradient(),
                  cXPred.getGradient(), 1,
                  cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                  cFT.getOutput(), cFT.getPrevOutput(),
                  cXPred.getGradient(), 1,
                  cFT.GetWindow(), cFT.GetCount(), 1, true))
      ReturnFalse;
   if(!SumAndNormalize(cFT.getGradient(), cFT.getPrevOutput(),
                       cFT.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!MatMulGrad(cR.getOutput(), cR.getPrevOutput(),
                  cRT.getOutput(), cRT.getGradient(),
                  cR_RT.getGradient(), cRT.GetCount(),
                  cRT.GetWindow(), cRT.GetCount(), 1, false))
      ReturnFalse;
   if(!MatMulGrad(cQ.getOutput(), cQ.getPrevOutput(),
                  cQT.getOutput(), cQT.getGradient(),
                  cQ_QT.getGradient(), cQT.GetCount(),
                  cQT.GetWindow(), cQT.GetCount(), 1, false))
      ReturnFalse;
   if(!cR.CalcHiddenGradients((CObject*)cRT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cR.getGradient(), cR.getPrevOutput(),
                       cR.getGradient(), cRT.GetWindow(), false, 0, 0, 0, 1e-4f))
      ReturnFalse;
   cR.getGradient().BufferRead();
   if(!cQ.CalcHiddenGradients(cQT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cQ.getGradient(), cQ.getPrevOutput(),
                       cQ.getGradient(), cQT.GetWindow(), false, 0, 0, 0, 1e-4f))
      ReturnFalse;
   cQ.getGradient().BufferRead();
   if(!cH.CalcHiddenGradients(cHT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cH.getGradient(), cH.getPrevOutput(),
                       cH.getGradient(), cHT.GetWindow(), false, 0, 0, 0, 0.01f))
      ReturnFalse;
   if(!cF.CalcHiddenGradients(cFT.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cF.getGradient(), cF.getPrevOutput(),
                       cF.getGradient(), cFT.GetWindow(), false, 0, 0, 0, 0.01f))
      ReturnFalse;
//--- Rest Attention
   if(!cKoopmanRest.CalcHiddenGradients(cAuxiliaryNet.AsObject()))
      ReturnFalse;
//--- Pred / Rest
   if(!NeuronOCL.getPrevOutput().Fill(0))
      ReturnFalse;
   if(!DifferentGrad(NeuronOCL.getPrevOutput(), cKoopmanRest.getPrevOutput(),
                     cKoopmanRest.getGradient(), cKoopman.GetWindow()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getPrevOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
     {
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
     }
   if(!Concat(cKoopmanPred.getGradient(), cKoopmanRest.getPrevOutput(), cKoopman.getGradient(),
              cKoopman.GetWindow(), cKoopman.Neurons() - cKoopman.GetWindow(), 1))
      ReturnFalse;
//--- Koopman
   if(!NeuronOCL.CalcHiddenGradients(cKoopman.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//--- Koopman
   if(!cKoopman.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cAuxiliaryNet.UpdateInputWeights(cKoopmanRest.AsObject()))
      ReturnFalse;
//--- Kalman Filter
   if(!cB.UpdateInputWeights())
      ReturnFalse;
   if(!cF.UpdateInputWeights())
      ReturnFalse;
   if(!cH.UpdateInputWeights())
      ReturnFalse;
   if(!cQ.UpdateInputWeights())
      ReturnFalse;
   if(!cR.UpdateInputWeights())
      ReturnFalse;
//---
   if(!SumAndNormalize(cQ.getWeightsParams(), cQ.getWeightsParams(), cQ.getWeightsParams(), 2 * cQT.GetWindow(), true))
      ReturnFalse;
   if(!SumAndNormalize(cR.getWeightsParams(), cR.getWeightsParams(), cR.getWeightsParams(), 2 * cRT.GetWindow(), true))
      ReturnFalse;
//---
   if(!cQ.FeedForward())
      ReturnFalse;
   mKL.Random(-1, 1);
   mKL = mKL.TriL();
   mKL = mKL.MatMul(mKL.Transpose());
   float error = 0;
   if(!cQ.getGradient().Fill(mKL) ||
      !cQ.calcOutputGradients(cQ.getGradient(), error) ||
      !cQ.UpdateInputWeights())
      ReturnFalse;
//---
   if(!cR.FeedForward())
      ReturnFalse;
   mKL.Random(-1, 1);
   mKL = mKL.TriL();
   mKL = mKL.MatMul(mKL.Transpose());
   error = 0;
   if(!cR.getGradient().Fill(mKL) ||
      !cR.calcOutputGradients(cR.getGradient(), error) ||
      !cR.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronK2VAEEncoder *Source = source;
//--- Koopman
   if(!cKoopman.WeightsUpdate(Source.cKoopman.AsObject(), tau))
      ReturnFalse;
   if(!cAuxiliaryNet.WeightsUpdate(Source.cAuxiliaryNet.AsObject(), tau))
      ReturnFalse;
//--- Kalman Filter
   if(!cB.WeightsUpdate(Source.cB.AsObject(), tau))
      ReturnFalse;
   if(!cF.WeightsUpdate(Source.cF.AsObject(), tau))
      ReturnFalse;
   if(!cH.WeightsUpdate(Source.cH.AsObject(), tau))
      ReturnFalse;
   if(!cQ.WeightsUpdate(Source.cQ.AsObject(), tau))
      ReturnFalse;
   if(!cR.WeightsUpdate(Source.cR.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//--- Koopman
   if(!cKoopman.Save(file_handle))
      ReturnFalse;
   if(!cAuxiliaryNet.Save(file_handle))
      ReturnFalse;
//--- Kalman Filter
   if(!cB.Save(file_handle))
      ReturnFalse;
   if(!cF.Save(file_handle))
      ReturnFalse;
   if(!cH.Save(file_handle))
      ReturnFalse;
   if(!cQ.Save(file_handle))
      ReturnFalse;
   if(!cR.Save(file_handle))
      ReturnFalse;
   if(!cP.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronK2VAEEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//--- Koopman
   if(!LoadInsideLayer(file_handle, cKoopman.AsObject()))
      ReturnFalse;
//---
   uint window = cKoopman.GetWindow();
   uint units_cross = cKoopman.GetUnits();
   int index = 1;
   if(!cKoopmanPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKoopmanRest.Init(0, index, OpenCL, window * (units_cross - 1), optimization, iBatch))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cAuxiliaryNet.AsObject()))
      ReturnFalse;
//--- Kalman Filter
   if(!LoadInsideLayer(file_handle, cB.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cF.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cH.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cQ.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cR.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cP.AsObject()))
      ReturnFalse;
//---
   index += 7;
   if(!cFT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cHT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cQT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cRT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cQ_QT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cR_RT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cXPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cF_P.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cPPred.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cP_HT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cH_P_HT.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   mS = mSGrad = matrix<double>::Zeros(window, window);
//---
   index++;
   if(!cSInv.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cK.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cYPred.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cDeltY.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cX.Init(0, index, OpenCL, window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cK_H.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cIdifK_H.Init(0, index, OpenCL, window * window, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cIdifK_HT.Init(0, index, OpenCL, window, window, optimization, iBatch))
      ReturnFalse;
//---
   mKL = mP = mPGrad = mS;
   mNoise = mGrad = matrix<double>::Zeros(Neurons() / window, window);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronK2VAEEncoder::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cKoopman.TrainMode(bTrain);
   cAuxiliaryNet.TrainMode(bTrain);
   cB.TrainMode(bTrain);
   cF.TrainMode(bTrain);
   cH.TrainMode(bTrain);
   cQ.TrainMode(bTrain);
   cR.TrainMode(bTrain);
   cP.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronK2VAEEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//--- Koopman
   cKoopman.SetOpenCL(OpenCL);
   cKoopmanPred.SetOpenCL(OpenCL);
   cKoopmanRest.SetOpenCL(OpenCL);
//---
   cAuxiliaryNet.SetOpenCL(OpenCL);
//--- Kalman Filter
   cB.SetOpenCL(OpenCL);
   cF.SetOpenCL(OpenCL);
   cH.SetOpenCL(OpenCL);
   cQ.SetOpenCL(OpenCL);
   cR.SetOpenCL(OpenCL);
   cP.SetOpenCL(OpenCL);
   cFT.SetOpenCL(OpenCL);
   cHT.SetOpenCL(OpenCL);
   cQT.SetOpenCL(OpenCL);
   cRT.SetOpenCL(OpenCL);
   cPT.SetOpenCL(OpenCL);
   cQ_QT.SetOpenCL(OpenCL);
   cR_RT.SetOpenCL(OpenCL);
   cXPred.SetOpenCL(OpenCL);
   cF_P.SetOpenCL(OpenCL);
   cPPred.SetOpenCL(OpenCL);
   cP_HT.SetOpenCL(OpenCL);
   cH_P_HT.SetOpenCL(OpenCL);
   cSInv.SetOpenCL(OpenCL);
   cK.SetOpenCL(OpenCL);
   cKT.SetOpenCL(OpenCL);
   cYPred.SetOpenCL(OpenCL);
   cDeltY.SetOpenCL(OpenCL);
   cX.SetOpenCL(OpenCL);
   cK_H.SetOpenCL(OpenCL);
   cIdifK_H.SetOpenCL(OpenCL);
   cIdifK_HT.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint units_count, uint dimension,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * dimension, optimization_type, batch))
      ReturnFalse;
//---
   iCount = units_count;
   iDimension = dimension;
//---
   int index = 0;
   if(!cW.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cA.Init(0, index, OpenCL, 2 * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cGL.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cH.Init(0, index, OpenCL, iCount * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAdj.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAttention.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::InterpolationAttention(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL ||
      NeuronOCL.Neurons() < int(iCount * iDimension))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {iCount, iCount};
   uint local_work_size[2] = {1, iCount};
   const int kernel = def_k_InterpolationAttention;
   setBuffer(kernel, def_k_ia_matrix_in, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ia_A, cA.getOutputIndex())
   setBuffer(kernel, def_k_ia_Adj, cAdj.getOutputIndex())
   setBuffer(kernel, def_k_ia_Atten, cAttention.getOutputIndex())
   setBuffer(kernel, def_k_ia_GL, cGL.getOutputIndex())
   setBuffer(kernel, def_k_ia_H, cH.getOutputIndex())
   setBuffer(kernel, def_k_ia_W, cW.getOutputIndex())
   setBuffer(kernel, def_k_ia_matrix_out, getOutputIndex())
   setArgument(kernel, def_k_ia_dimension, (int)iDimension)
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::InterpolationAttentionGrad(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL ||
      NeuronOCL.Neurons() < int(iCount * iDimension))
      ReturnFalse;
//---
   CBufferFloat* temp = cA.getGradient();
   if(!temp || !temp.Fill(0))
      ReturnFalse;
   temp = cGL.getGradient();
   if(!temp || !temp.Fill(0))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {iCount, iCount};
   uint local_work_size[2] = {1, iCount};
   const int kernel = def_k_InterpolationAttentionGrad;
   setBuffer(kernel, def_k_iag_matrix_in, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_iag_matrix_in_gr, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_iag_A, cA.getOutputIndex())
   setBuffer(kernel, def_k_iag_A_gr, cA.getGradientIndex())
   setBuffer(kernel, def_k_iag_Adj, cAdj.getOutputIndex())
   setBuffer(kernel, def_k_iag_Atten, cAttention.getOutputIndex())
   setBuffer(kernel, def_k_iag_GL, cGL.getOutputIndex())
   setBuffer(kernel, def_k_iag_GL_gr, cGL.getGradientIndex())
   setBuffer(kernel, def_k_iag_H, cH.getOutputIndex())
   setBuffer(kernel, def_k_iag_H_gr, cH.getGradientIndex())
   setBuffer(kernel, def_k_iag_W, cW.getOutputIndex())
   setBuffer(kernel, def_k_iag_W_gr, cW.getGradientIndex())
   setBuffer(kernel, def_k_iag_matrix_out_gr, getGradientIndex())
   setArgument(kernel, def_k_iag_dimension, (int)iDimension)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
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
bool CNeuronInterpolationAttention::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
     {
      if(!cW.FeedForward())
         ReturnFalse;
      if(!cA.FeedForward())
         ReturnFalse;
      if(!cGL.FeedForward())
         ReturnFalse;
     }
   if(!InterpolationAttention(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!InterpolationAttentionGrad(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cW.UpdateInputWeights())
      ReturnFalse;
   if(!cA.UpdateInputWeights())
      ReturnFalse;
   if(!cGL.UpdateInputWeights() ||
      !Normalize(cGL.getWeightsParams(), 2 * iDimension))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronInterpolationAttention *Source = source;
   if(!cW.WeightsUpdate(Source.cW.AsObject(), tau))
      ReturnFalse;
   if(!cA.WeightsUpdate(Source.cA.AsObject(), tau))
      ReturnFalse;
   if(!cGL.WeightsUpdate(Source.cGL.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iCount) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iDimension) < INT_VALUE)
      ReturnFalse;
//---
   if(!cW.Save(file_handle))
      ReturnFalse;
   if(!cA.Save(file_handle))
      ReturnFalse;
   if(!cGL.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInterpolationAttention::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCount = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimension = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cW.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGL.AsObject()))
      ReturnFalse;
//---
   int index = 3;
   if(!cH.Init(0, index, OpenCL, iCount * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAdj.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAttention.Init(0, index, OpenCL, iDimension * iDimension, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronInterpolationAttention::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cW.SetOpenCL(OpenCL);
   cA.SetOpenCL(OpenCL);
   cGL.SetOpenCL(OpenCL);
   cH.SetOpenCL(OpenCL);
   cAdj.SetOpenCL(OpenCL);
   cAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronInterpolationAttention::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cW.TrainMode(bTrain);
   cA.TrainMode(bTrain);
   cGL.TrainMode(bTrain);
   cH.TrainMode(bTrain);
   cAdj.TrainMode(bTrain);
   cAttention.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPeriodNorm::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint units_count, uint period, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * period * variables,
                            optimization_type, batch))
      ReturnFalse;
//---
   iPeriod = period;
   iVariables = variables;
   iCount = units_count;
//---
   if(!cMeanSTDevs.Init(0, 0, OpenCL, iCount * iVariables * 2, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPeriodNorm::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < Neurons())
      ReturnFalse;
//---
   uint global_work_offset[3] = { 0 };
   uint global_work_size[3] = { iCount, iPeriod, iVariables };
   uint local_work_size[3] = { 1, iPeriod, 1 };
//---
   uint kernel = def_k_PeriodNorm;
   setBuffer(kernel, def_k_pn_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_pn_mean_stdevs, cMeanSTDevs.getOutputIndex())
   setBuffer(kernel, def_k_pn_outputs, getOutputIndex())
   setArgument(kernel, def_k_pn_total_inputs, (int)MathMin(NeuronOCL.Neurons() / (iVariables * iPeriod), iCount))
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
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
bool CNeuronPeriodNorm::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < Neurons())
      ReturnFalse;
//---
   uint global_work_offset[3] = { 0 };
   uint global_work_size[3] = { iCount, iPeriod, iVariables };
   uint local_work_size[3] = { 1, iPeriod, 1 };
//---
   uint kernel = def_k_PeriodNormGrad;
   setBuffer(kernel, def_k_png_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_png_inputs_gr, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_png_mean_stdevs, cMeanSTDevs.getOutputIndex())
   setBuffer(kernel, def_k_png_mean_stdevs_gr, cMeanSTDevs.getGradientIndex())
   setBuffer(kernel, def_k_png_outputs, getOutputIndex())
   setBuffer(kernel, def_k_png_outputs_gr, getGradientIndex())
   setArgument(kernel, def_k_png_total_inputs, (int)MathMin(NeuronOCL.Neurons() / (iVariables * iPeriod), iCount))
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
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
bool CNeuronPeriodNorm::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cMeanSTDevs.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iCount)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iPeriod)) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iVariables)) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPeriodNorm::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMeanSTDevs.AsObject()))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iCount = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iPeriod = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPeriodNorm::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cMeanSTDevs.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out,
                          uint experts, float dropout, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * window_out, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
//---
   CNeuronConvOCL       *conv    = NULL;
   CNeuronDropoutOCL    *dout    = NULL;
   CNeuronSparseSoftMax *softmax = NULL;
//--- Probability
   acProbability.Clear();
   acProbability.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(experts, index, OpenCL, window, window, 2 * experts, units, 1, optimization, iBatch) ||
      !acProbability.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(experts, index, OpenCL, 2 * experts, 2 * experts, experts, units, 1, optimization, iBatch) ||
      !acProbability.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   dout = new CNeuronDropoutOCL();
   if(!dout ||
      !dout.Init(0, index, OpenCL, conv.Neurons(), dropout, optimization, iBatch) ||
      !acProbability.Add(dout))
     {
      DeleteObj(dout);
      ReturnFalse;
     }
   index++;
   softmax = new CNeuronSparseSoftMax;
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, units, experts, topK, optimization, iBatch) ||
      !acProbability.Add(softmax))
     {
      DeleteObj(softmax);
      ReturnFalse;
     }
   softmax.SetHeads(units);
//---
   index++;
   if(!cExperts.Init(0, index, OpenCL, window * window_out * experts, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cWeightsConv.Init(0, index, OpenCL, window * window_out * units, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
   CNeuronConvOCL *conv = NULL;
   CNeuronSparseSoftMax *softmax = NULL;
//--- Probability
   for(int i = 0; i < acProbability.Total(); i++)
     {
      current = acProbability[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
   conv = acProbability[0];
   softmax = prev;
//--- Experts
   if(bTrain)
      if(!cExperts.FeedForward())
         ReturnFalse;
//---
   uint units = conv.GetUnits();
   uint window_in = conv.GetWindow();
   uint window_out = Neurons() / units;
   uint topK = softmax.DimensionOut();
   uint experts = cExperts.Neurons() / (window_in * window_out);
//--- Generate Weights
   if(!SparseMatMul(softmax.GetIndexes(), softmax.getOutput(), cExperts.getOutput(),
                    cWeightsConv.getOutput(), units, topK, window_in * window_out, experts))
      ReturnFalse;
//--- Result
   if(!MatMul(NeuronOCL.getOutput(), cWeightsConv.getOutput(), Output, 1, window_in, window_out, units, true))
      ReturnFalse;
   if(activation != None)
      if(!Activation(Output, Output, activation))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!acProbability[0] || !acProbability[-1])
      ReturnFalse;
//---
   CNeuronConvOCL* conv = acProbability[0];
   CNeuronSparseSoftMax* softmax = acProbability[-1];
   uint units = conv.GetUnits();
   uint window_in = conv.GetWindow();
   uint window_out = Neurons() / units;
   uint topK = softmax.DimensionOut();
   uint experts = cExperts.Neurons() / (window_in * window_out);
//---
   if(!MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(),
                  cWeightsConv.getOutput(), cWeightsConv.getGradient(),
                  Gradient, 1, window_in, window_out, units, true))
      ReturnFalse;
   if(!SparseMatMulGrad(softmax.GetIndexes(), softmax.getOutput(), softmax.getGradient(),
                        cExperts.getOutput(), cExperts.getGradient(), cWeightsConv.getGradient(),
                        units, topK, window_in * window_out, experts))
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
//---
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getPrevOutput(), NeuronOCL.getGradient(),
                       NeuronOCL.getGradient(), window_in, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
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
//--- Experts
   if(!cExperts.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMoEConv *Source = source;
   if(!acProbability.WeightsUpdate(Source.acProbability.AsObject(), tau))
      ReturnFalse;
   if(!cExperts.WeightsUpdate(Source.cExperts.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!acProbability.Save(file_handle))
      ReturnFalse;
   if(!cExperts.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMoEConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!acProbability.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cExperts.AsObject()))
      ReturnFalse;
//---
   CNeuronConvOCL* conv = acProbability[0];
   uint window_in = conv.GetWindow();
   int index = acProbability.Total() + 1;
   if(!cWeightsConv.Init(0, index, OpenCL, window_in * Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMoEConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   acProbability.SetOpenCL(OpenCL);
   cExperts.SetOpenCL(OpenCL);
   cWeightsConv.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMoEConv::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   acProbability.TrainMode(bTrain);
   cExperts.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint dimension, uint units, uint bottleneck, uint &shifts[],
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, bottleneck, bottleneck,
                                   dimension, units, 1, optimization_type, batch))
      ReturnFalse;
//---
   int scales = int(shifts.Size());
   if(scales == 0 ||
      ArrayResize(iaShifts, scales) < scales ||
      ArrayCopy(iaShifts, shifts, 0, 0, scales) < scales)
      ReturnFalse;
   ibShifts = OpenCL.AddBufferFromArray(iaShifts, 0, scales, CL_MEM_READ_ONLY);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
//---
   cBottleneck.Clear();
   cExcitation.Clear();
   cProjection.Clear();
   cBottleneck.SetOpenCL(OpenCL);
   cExcitation.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
//---
   CNeuronMultiWindowsConvWPadOCL*  padconv = NULL;
   CNeuronSpikeDepthWiseConv*       dwconv = NULL;
   CNeuronSpikeConvBlock*           conv = NULL;
   CNeuronBaseOCL*                  neuron = NULL;
   CNeuronMultiScaleExcitation*     excitation = NULL;
   CNeuronTransposeOCL*             transpose = NULL;
   uint index = 0;
//--- Bottleneck
   uint units_out = (units + 1) / 2;
   dwconv = new CNeuronSpikeDepthWiseConv();
   if(!dwconv ||
      !dwconv.Init(0, index, OpenCL, dimension, bottleneck, 5, 2, units_out, 1, optimization, iBatch) ||
      !cBottleneck.Add(dwconv))
     {
      DeleteObj(dwconv);
      ReturnFalse;
     }
   index++;
   uint windows[] = { 3 * bottleneck};
   padconv = new CNeuronMultiWindowsConvWPadOCL();
   if(!padconv ||
      !padconv.Init(0, index, OpenCL, windows, bottleneck, bottleneck, units_out, 1, optimization, iBatch) ||
      !cBottleneck.Add(padconv))
     {
      DeleteObj(padconv)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, bottleneck, bottleneck, bottleneck, units_out, 1, optimization, iBatch) ||
      !cBottleneck.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//--- Excitation
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, conv.Neurons()*scales, optimization, iBatch) ||
      !cExcitation.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
   excitation = new CNeuronMultiScaleExcitation();
   if(!excitation ||
      !excitation.Init(0, index, OpenCL, bottleneck, units_out, scales, (bottleneck + 1) / 2, optimization, iBatch) ||
      !cExcitation.Add(excitation))
     {
      DeleteObj(excitation)
      ReturnFalse;
     }
   index++;
   transpose = new CNeuronTransposeOCL();
   if(!transpose ||
      !transpose.Init(0, index, OpenCL, bottleneck * units_out, scales, optimization, iBatch) ||
      !cExcitation.Add(transpose))
     {
      DeleteObj(transpose)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, scales, scales, 1, transpose.GetCount(), 1, optimization, iBatch) ||
      !cExcitation.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//--- Projection
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, conv.Neurons(), optimization, iBatch) ||
      !cProjection.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   if(!neuron.SetGradient(conv.getGradient(), true))
      ReturnFalse;
   index++;
   dwconv = new CNeuronSpikeDepthWiseConv();
   if(!dwconv ||
      !dwconv.Init(0, index, OpenCL, bottleneck, 2 * bottleneck, 5, 1, units_out, 1, optimization, iBatch) ||
      !cProjection.Add(dwconv))
     {
      DeleteObj(dwconv);
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Bottleneck
   for(int i = 0; i < cBottleneck.Total(); i++)
     {
      curr = cBottleneck[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Excitation
   curr = cExcitation[0];
   if(!curr)
      ReturnFalse;
   uint dimension = cConv.GetWindow();
   uint units = prev.Neurons() / dimension;
   if(!DilatedDifference(prev, curr, ibShifts, dimension, units))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cExcitation.Total(); i++)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Projection
   curr = cProjection[0];
   if(!curr ||
      !SumAndNormalize(cBottleneck[-1].getOutput(), prev.getOutput(), curr.getOutput(), dimension, true, 0, 0, 0, 1))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cProjection.Total(); i++)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), Output, Output, cConv.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Projection
   CNeuronBaseOCL* next = cProjection[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!CNeuronSpikeConvBlock::calcInputGradients(next))
      ReturnFalse;
   for(int i = cProjection.Total() - 2; i >= 0; i--)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Excitation
   next = cExcitation[-1];
   if(!next)
      ReturnFalse;
   Deactivation(next)
   for(int i = cExcitation.Total() - 2; i >= 0; i--)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Bottleneck
   curr = cBottleneck[-1];
   if(!curr)
      ReturnFalse;
   uint dimension = cConv.GetWindow();
   uint units = curr.Neurons() / dimension;
   if(!DilatedDifferenceGrad(curr, next, ibShifts, dimension, units))
      ReturnFalse;
   Deactivation(curr)
   next = curr;
   for(int i = cBottleneck.Total() - 2; i >= 0; i--)
     {
      curr = cBottleneck[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Bottleneck
   for(int i = 0; i < cBottleneck.Total(); i++)
     {
      curr = cBottleneck[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Excitation
   prev = cExcitation[0];
   for(int i = 1; i < cExcitation.Total(); i++)
     {
      curr = cExcitation[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Projection
   prev = cProjection[0];
   for(int i = 1; i < cProjection.Total(); i++)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronSpikeConvBlock::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   uint size = iaShifts.Size();
   if(FileWriteInteger(file_handle, size, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   for(uint i = 0; i < size; i++)
     {
      if(FileWriteInteger(file_handle, iaShifts[i], INT_VALUE) < INT_VALUE)
         ReturnFalse;
     }
//---
   if(!cBottleneck.Save(file_handle))
      ReturnFalse;
   if(!cExcitation.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   uint size = FileReadInteger(file_handle);
   if(ArrayResize(iaShifts, size) < (int)size)
      ReturnFalse;
   for(uint i = 0; i < size; i++)
     {
      if(FileIsEnding(file_handle))
         ReturnFalse;
      iaShifts[i] = FileReadInteger(file_handle);
     }
   OpenCL.BufferFree(ibShifts);
   ibShifts = OpenCL.AddBufferFromArray(iaShifts, 0, size, CL_MEM_READ_ONLY);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
//---
   if(!cBottleneck.Load(file_handle))
      ReturnFalse;
   if(!cExcitation.Load(file_handle))
      ReturnFalse;
   if(!cProjection.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMultiScaleDifference* Source = source;
   if(!cBottleneck.WeightsUpdate(Source.cBottleneck.AsObject(), tau))
      ReturnFalse;
   if(!cExcitation.WeightsUpdate(Source.cExcitation.AsObject(), tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleDifference::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cBottleneck.ClearStates())
      ReturnFalse;
   if(!cExcitation.ClearStates())
      ReturnFalse;
   if(!cProjection.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleDifference::SetOpenCL(COpenCLMy *obj)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibShifts);
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   ibShifts = OpenCL.AddBufferFromArray(iaShifts, 0, iaShifts.Size(), CL_MEM_READ_ONLY);
   cBottleneck.SetOpenCL(OpenCL);
   cExcitation.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleDifference::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cBottleneck.TrainMode(bTrain);
   cExcitation.TrainMode(bTrain);
   cProjection.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint dimension, uint units, uint forecast, uint experts,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, dimension * forecast, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   cConfidenceGating.Clear();
   cScore.Clear();
   cCGScore.Clear();
   cConfidenceGating.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
   cCGScore.SetOpenCL(OpenCL);
//---
   uint index = 0;
   if(!cTailAwareMoE.Init(0, index, OpenCL, dimension * forecast * 4 * experts, optimization, iBatch))
      ReturnFalse;
   cTailAwareMoE.SetActivationFunction(None);
//---
   uint windows[] = {7, 5, 3};
   uint steps[] = {2, 2, 2};
   CNeuronSpikeSuperKernelBlock* conv = NULL;
   CNeuronBaseOCL* neuron = NULL;
   CNeuronBatchNormOCL* norm = NULL;
   CNeuronSoftMaxOCL* softmax = NULL;
   CNeuronSpikeActivation* spike = NULL;
//--- Score
   index++;
   conv = new CNeuronSpikeSuperKernelBlock();
   if(!conv ||
      !conv.Init(experts, index, OpenCL, dimension, (dimension + 1) / 2, windows, steps, units, 1, optimization, iBatch) ||
      !cScore.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, experts, optimization, iBatch) ||
      !cScore.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(SoftPlus);
   index++;
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, neuron.Neurons(), optimization, iBatch) ||
      !cScore.Add(softmax))
      DeleteObjAndFalse(softmax);
   softmax.SetHeads(1);
   index++;
   if(!cScoreStack.Init(0, index, OpenCL, 50, neuron.Neurons(), 1, optimization, iBatch))
      ReturnFalse;
//--- Confidence-Gating
   index++;
   conv = new CNeuronSpikeSuperKernelBlock();
   if(!conv ||
      !conv.Init(experts, index, OpenCL, dimension, (dimension + 1) / 2, windows, steps, units, 1, optimization, iBatch) ||
      !cConfidenceGating.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, experts, optimization, iBatch) ||
      !cConfidenceGating.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, neuron.Neurons(), iBatch, optimization) ||
      !cConfidenceGating.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(SoftPlus);
   index++;
   spike = new CNeuronSpikeActivation();
   if(!spike ||
      !spike.Init(0, index, OpenCL, neuron.Neurons(), optimization, iBatch) ||
      !cConfidenceGating.Add(spike))
      DeleteObjAndFalse(spike);
   index++;
   if(!cCGStack.Init(0, index, OpenCL, 50, neuron.Neurons(), 1, optimization, iBatch))
      ReturnFalse;
//--- Confidence-Gating Score
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, experts, optimization, iBatch) ||
      !cCGScore.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
//---
   index++;
   if(!cDensity.Init(0, index, OpenCL, 4 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cDensity.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::MixExpertsPredict(void)
  {
   if(!OpenCL || !Output || !cDensity.getOutput())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[] = { Neurons() };
   uint kernel = def_k_MixExpertsPredict;
   setBuffer(kernel, def_k_moepred_experts, cDensity.getOutputIndex())
   setBuffer(kernel, def_k_moepred_outputs, getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::MixExpertsPredictGrad(void)
  {
   if(!OpenCL || !Gradient || !cDensity.getOutput())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[] = { Neurons() };
   uint kernel = def_k_MixExpertsPredictGrad;
   setBuffer(kernel, def_k_moepredgr_experts, cDensity.getOutputIndex())
   setBuffer(kernel, def_k_moepredgr_experts_gr, cDensity.getGradientIndex())
   setBuffer(kernel, def_k_moepredgr_outputs_gr, getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cDensity.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Score
   for(int i = 0; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Confidence-Gating
   prev = NeuronOCL;
   for(int i = 0; i < cConfidenceGating.Total(); i++)
     {
      curr = cConfidenceGating[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Confidence-Gating Score
   prev = cCGScore[0];
   if(!prev ||
      !ElementMult(cConfidenceGating[-1].getOutput(), cScore[-1].getOutput(), prev.getOutput()))
      ReturnFalse;
   for(int i = 1; i < cCGScore.Total(); i++)
     {
      curr = cCGScore[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Density
   if(bTrain)
      if(!cTailAwareMoE.FeedForward())
         ReturnFalse;
   if(!MatMul(prev.getOutput(), cTailAwareMoE.getOutput(), cDensity.getOutput(),
              1, prev.Neurons(), cDensity.Neurons(), 1, false))
      ReturnFalse;
   if(!MixExpertsPredict())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!MixExpertsPredictGrad())
      ReturnFalse;
//--- Confidence-Gating Score
   CNeuronBaseOCL* next = NULL;
   CNeuronBaseOCL* curr = cCGScore[-1];
   if(!curr ||
      !MatMulGrad(curr.getOutput(), curr.getGradient(),
                  cTailAwareMoE.getOutput(), cTailAwareMoE.getGradient(),
                  cDensity.getGradient(), 1, curr.Neurons(), cDensity.Neurons(), 1, false))
      ReturnFalse;
   next = curr;
   for(int i = cCGScore.Total() - 2; i >= 0; i--)
     {
      curr = cCGScore[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   next = cConfidenceGating[-1];
   curr = cScore[-1];
   if(!next || !curr ||
      !ElementMultGrad(next.getOutput(), next.getGradient(),
                       curr.getOutput(), curr.getGradient(),
                       cCGScore[0].getGradient(), next.Activation(), curr.Activation()))
      ReturnFalse;
//--- Confidence-Gating
   if(!cCGStack.FeedForward(next) ||
      !DiversityLoss(cCGStack.AsObject(), cCGStack.GetStackSize(), next.Neurons(), false))
      ReturnFalse;
   if(next.Activation() != None)
      if(!DeActivation(next.getOutput(), cCGStack.getGradient(), cCGStack.getGradient(), next.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(next.getGradient(), cCGStack.getGradient(), next.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   for(int i = cConfidenceGating.Total() - 2; i >= 0; i--)
     {
      curr = cConfidenceGating[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Score
   next = cScore[-1];
   if(!cScoreStack.FeedForward(next) ||
      !DiversityLoss(cScoreStack.AsObject(), cScoreStack.GetStackSize(), next.Neurons(), false))
      ReturnFalse;
   if(next.Activation() != None)
      if(!DeActivation(next.getOutput(), cScoreStack.getGradient(), cScoreStack.getGradient(), next.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(next.getGradient(), cScoreStack.getGradient(), next.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   for(int i = cScore.Total() - 2; i >= 0; i--)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cConfidenceGating[0]) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Score
   for(int i = 0; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Confidence-Gating
   prev = NeuronOCL;
   for(int i = 0; i < cConfidenceGating.Total(); i++)
     {
      curr = cConfidenceGating[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Confidence-Gating Score
   prev = cCGScore[0];
   for(int i = 1; i < cCGScore.Total(); i++)
     {
      curr = cCGScore[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Density
   if(!cTailAwareMoE.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronConfGateTailAwareMoE* Source = source;
   dWeightsUpdate(cScore, Source, tau);
   dWeightsUpdate(cConfidenceGating, Source, tau);
   dWeightsUpdate(cCGScore, Source, tau);
   dWeightsUpdate(cTailAwareMoE, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cConfidenceGating.Save(file_handle))
      ReturnFalse;
   if(!cScore.Save(file_handle))
      ReturnFalse;
   if(!cCGScore.Save(file_handle))
      ReturnFalse;
   if(!cTailAwareMoE.Save(file_handle))
      ReturnFalse;
   if(!cScoreStack.Save(file_handle))
      ReturnFalse;
   if(!cCGStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!cConfidenceGating.Load(file_handle))
      ReturnFalse;
   if(!cScore.Load(file_handle))
      ReturnFalse;
   if(!cCGScore.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTailAwareMoE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScoreStack.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCGStack.AsObject()))
      ReturnFalse;
//---
   uint index = 3 + cScore.Total() + cConfidenceGating.Total() + cCGScore.Total();
   if(!cDensity.Init(0, index, OpenCL, 4 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cDensity.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConfGateTailAwareMoE::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cConfidenceGating.ClearStates())
      ReturnFalse;
   if(!cScore.ClearStates())
      ReturnFalse;
   if(!cCGScore.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronConfGateTailAwareMoE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cConfidenceGating.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
   cCGScore.SetOpenCL(OpenCL);
   cTailAwareMoE.SetOpenCL(OpenCL);
   cScoreStack.SetOpenCL(OpenCL);
   cCGStack.SetOpenCL(OpenCL);
   cDensity.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronConfGateTailAwareMoE::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cConfidenceGating.TrainMode(bTrain);
   cScore.TrainMode(bTrain);
   cCGScore.TrainMode(bTrain);
   cTailAwareMoE.TrainMode(bTrain);
   cScoreStack.TrainMode(bTrain);
   cCGStack.TrainMode(bTrain);
   cDensity.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint window, uint filters, uint units,
                                uint candidates, uint topK,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, filters * units, optimization_type, batch))
      ReturnFalse;
   if(!cParams.Init(0, 0, OpenCL, window, units, window * filters, candidates, topK, optimization, iBatch))
      ReturnFalse;
   cParams.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cParams.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint units = cParams.GetFields();
   uint filters = Neurons() / units;
   uint window = cParams.Neurons() / (filters * units);
   if(!MatMul(NeuronOCL.getOutput(), cParams.getOutput(), Output, 1, window, filters, units, true))
      ReturnFalse;
   dActivation(this);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint units = cParams.GetFields();
   uint filters = Neurons() / units;
   uint window = cParams.Neurons() / (filters * units);
   if(NeuronOCL.Neurons() < int(window * units))
      ReturnFalse;
   if(!MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(),
                  cParams.getOutput(), cParams.getGradient(),
                  Gradient, 1, window, filters, units, true))
      ReturnFalse;
   Deactivation(cParams);
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cParams.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getPrevOutput(),
                       NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cParams.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cParams.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cParams.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSparseMoEConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSparseMoEConv* Source = source;
   dWeightsUpdate(cParams, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSparseMoEConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cParams.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSparseMoEConv::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cParams.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint &dimensions[],
                      uint units_s, uint heads, uint scenarios, uint stack_size,
                      uint layers, uint embed_size, uint candidates, uint topK,
                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() <= units_s || layers < 1)
      ReturnFalse;
   uint units_ns = dimensions.Size() - units_s;
//---
   if(!CNeuronPerTokenFFN::Init(numOutputs, myIndex, open_cl, embed_size, units_ns,
                                (embed_size + 1) / 2, candidates, topK, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cPrepare.Init(0, index, OpenCL, dimensions, units_s + units_ns, embed_size, candidates, topK, optimization, iBatch))
      ReturnFalse;
   if(!cLastNonSequence.Init(0, index, OpenCL, embed_size * units_ns, optimization, iBatch))
      ReturnFalse;
   cLastNonSequence.SetActivationFunction(None);
   index++;
   if(!cScenarios.Init(0, index, OpenCL, embed_size, units_s, embed_size,
                       scenarios, (embed_size + 1) / 2, candidates, topK, optimization, iBatch))
      ReturnFalse;
   cScenarios.SetActivationFunction(None);
//---
   cFeatures.Clear();
   cFlow.Clear();
   cFeatures.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
//---
   index++;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, embed_size * units_s, optimization, iBatch) ||
      !cFeatures.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, embed_size * (units_ns + scenarios), optimization, iBatch) ||
      !cFlow.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
//---
   CNeuronFeatureSelfIteration* feature = NULL;
   CNeuronDomainAwareAttention* domain = NULL;
   CNeuronDomainFused* fused = NULL;
   for(uint i = 0; i < layers; i++)
     {
      index++;
      feature = new CNeuronFeatureSelfIteration();
      if(!feature ||
         !feature.Init(0, index, OpenCL, embed_size, units_s, heads, stack_size, (embed_size + 1) / 2,
                       candidates, topK, optimization, iBatch) ||
         !cFeatures.Add(feature))
         DeleteObjAndFalse(feature);
      index++;
      domain = new CNeuronDomainAwareAttention();
      if(!domain ||
         !domain.Init(0, index, OpenCL, embed_size, units_ns + scenarios, heads, embed_size, stack_size * units_s,
                      (embed_size + 1) / 2, candidates, topK, optimization, iBatch) ||
         !cFlow.Add(domain))
         DeleteObjAndFalse(domain);
      index++;
      fused = new CNeuronDomainFused();
      if(!fused ||
         !fused.Init(0, index, OpenCL, embed_size, units_ns, scenarios, (embed_size + 1) / 2,
                     candidates, topK, optimization, iBatch) ||
         !cFlow.Add(fused))
         DeleteObjAndFalse(fused);
     }
   index++;
   CNeuronAddToStack* stack = new CNeuronAddToStack();
   if(!stack ||
      !stack.Init(0, index, OpenCL, stack_size, embed_size, units_s, optimization, iBatch) ||
      !cFeatures.Add(stack))
      DeleteObjAndFalse(stack);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint layers = cFeatures.Total() - 2;
//--- Features
   CNeuronBaseOCL* feature = cFeatures[0];
   if(!feature ||
      !DeConcat(feature.getOutput(), cLastNonSequence.getOutput(), cPrepare.getOutput(),
                feature.Neurons(), cLastNonSequence.Neurons(), 1))
      ReturnFalse;
   for(int i = 1; i < cFeatures.Total(); i++)
     {
      feature = cFeatures[i];
      if(!feature ||
         !feature.FeedForward(cFeatures[i - 1]))
         ReturnFalse;
     }
//--- Scenarios
   if(!cScenarios.FeedForward(cFeatures[0]))
      ReturnFalse;
   CNeuronBaseOCL* flow = cFlow[0];
   if(!flow ||
      !Concat(cLastNonSequence.getOutput(), cScenarios.getOutput(), flow.getOutput(),
              cLastNonSequence.Neurons(), cScenarios.Neurons(), 1))
      ReturnFalse;
//--- Flow
   for(uint i = 0; i < layers; i++)
     {
      uint shift_domain = i * 2 + 1;
      uint shift_feature = i + 2;
      if(!cFeatures[shift_feature])
         ReturnFalse;
      if(cFeatures[shift_feature].Type() == defNeuronFeatureSelfIteration)
         feature = ((CNeuronFeatureSelfIteration*)cFeatures[shift_feature]).GetStack();
      else
         feature = cFeatures[shift_feature];
      flow = cFlow[shift_domain];
      if(!flow ||
         !flow.FeedForward(cFlow[shift_domain - 1], feature.getOutput()))
         ReturnFalse;
      flow = cFlow[shift_domain + 1];
      if(!flow ||
         !flow.FeedForward(cFlow[shift_domain]))
         ReturnFalse;
     }
//---
   if(!CNeuronPerTokenFFN::feedForward(flow))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPerTokenFFN::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//---Flow
   int layers = int(cFeatures.Total()) - 2;
   CNeuronBaseOCL* flow = cFlow[-2];
   if(!flow ||
      !flow.CalcHiddenGradients(cFlow[-1]))
      ReturnFalse;
   flow = cFlow[-3];
   if(!flow || !cFeatures[-1] ||
      !flow.CalcHiddenGradients(cFlow[-2], cFeatures[-1].getOutput(),
                                cFeatures[-1].getGradient(), (ENUM_ACTIVATION)cFeatures[-1].Activation()))
      ReturnFalse;
   CNeuronFeatureSelfIteration* feature = NULL;
   CNeuronBaseOCL* stack = NULL;
   for(int i = cFlow.Total() - 4; i >= 0; i--)
     {
      flow = cFlow[i];
      if(!flow)
         ReturnFalse;
      feature = cFeatures[(i + 2) / 2];
      if(!feature || !feature.GetStack())
         ReturnFalse;
      stack = feature.GetStack();
      if(!flow.CalcHiddenGradients(cFlow[i + 1], stack.getOutput(), stack.getGradient(),
                                   (ENUM_ACTIVATION)stack.Activation()))
         ReturnFalse;
     }
//--- Features
   for(int i = cFeatures.Total() - 2; i >= 0; i--)
     {
      stack = cFeatures[i];
      if(!stack ||
         !stack.CalcHiddenGradients(cFeatures[i + 1]))
         ReturnFalse;
     }
//--- Scenarios
   if(!DeConcat(cLastNonSequence.getGradient(), cScenarios.getGradient(), flow.getGradient(),
                cLastNonSequence.Neurons(), cScenarios.Neurons(), 1))
      ReturnFalse;
   CBufferFloat *temp = stack.getGradient();
   if(!stack.SetGradient(stack.getPrevOutput(), false) ||
      !stack.CalcHiddenGradients(cScenarios.AsObject()) ||
      !SumAndNormalize(temp, stack.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !stack.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!Concat(stack.getGradient(), cLastNonSequence.getGradient(), cPrepare.getGradient(),
              stack.Neurons(), cLastNonSequence.Neurons(), 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cPrepare.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   uint layers = cFeatures.Total() - 2;
//--- Features
   CNeuronBaseOCL *feature = NULL;
   for(int i = 1; i < cFeatures.Total(); i++)
     {
      feature = cFeatures[i];
      if(!feature ||
         !feature.UpdateInputWeights(cFeatures[i - 1]))
         ReturnFalse;
     }
//--- Scenarios
   if(!cScenarios.UpdateInputWeights(cFeatures[0]))
      ReturnFalse;
   CNeuronBaseOCL* flow = cFlow[0];
//--- Flow
   for(uint i = 0; i < layers; i++)
     {
      uint shift_domain = i * 2 + 1;
      uint shift_feature = i + 2;
      if(!cFeatures[shift_feature])
         ReturnFalse;
      if(cFeatures[shift_feature].Type() == defNeuronFeatureSelfIteration)
         feature = ((CNeuronFeatureSelfIteration*)cFeatures[shift_feature]).GetStack();
      else
         feature = cFeatures[shift_feature];
      flow = cFlow[shift_domain];
      if(!flow ||
         !flow.UpdateInputWeights(cFlow[shift_domain - 1], feature.getOutput()))
         ReturnFalse;
      flow = cFlow[shift_domain + 1];
      if(!flow ||
         !flow.UpdateInputWeights(cFlow[shift_domain]))
         ReturnFalse;
     }
//---
   if(!CNeuronPerTokenFFN::updateInputWeights(flow))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::Save(const int file_handle)
  {
   if(!CNeuronPerTokenFFN::Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cLastNonSequence.Save(file_handle))
      ReturnFalse;
   if(!cScenarios.Save(file_handle))
      ReturnFalse;
   if(!cFeatures.Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::Load(const int file_handle)
  {
   if(!CNeuronPerTokenFFN::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cPrepare.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLastNonSequence.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScenarios.AsObject()))
      ReturnFalse;
   if(!cFeatures.Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::Clear(void)
  {
   if(!CNeuronPerTokenFFN::Clear())
      ReturnFalse;
//---
   if(!cPrepare.Clear())
      ReturnFalse;
   if(!cLastNonSequence.Clear())
      ReturnFalse;
   if(!cScenarios.Clear())
      ReturnFalse;
   if(!cFeatures.ClearStates())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronPerTokenFFN::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMDL* Source = source;
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cLastNonSequence, Source, tau);
   dWeightsUpdate(cScenarios, Source, tau);
   dWeightsUpdate(cFeatures, Source, tau);
   dWeightsUpdate(cFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMDL::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronPerTokenFFN::SetOpenCL(obj);
//---
   cPrepare.SetOpenCL(OpenCL);
   cLastNonSequence.SetOpenCL(OpenCL);
   cScenarios.SetOpenCL(OpenCL);
   cFeatures.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMDL::TrainMode(bool flag)
  {
   CNeuronPerTokenFFN::TrainMode(flag);
//---
   cPrepare.TrainMode(bTrain);
   cLastNonSequence.TrainMode(bTrain);
   cScenarios.TrainMode(bTrain);
   cFeatures.TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint &dimensions[], uint units, uint embed_size,
                                    uint blocks, uint candidates, uint topK,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(blocks <= 0 || embed_size <= 0 || units <= 0 ||
      (embed_size * units) % blocks != 0)
      ReturnFalse;
   uint block_size = (embed_size * units) / blocks;
   if(!CNeuronFieldAwareConv::Init(numOutputs, myIndex, open_cl, block_size, block_size,
                                   blocks, (block_size + 3) / 4, candidates, topK, optimization_type, batch))
      ReturnFalse;
   if(!cTokenizer.Init(0, 0, OpenCL, dimensions, units, embed_size, candidates, topK,
                       optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cTokenizer.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronFieldAwareConv::feedForward(cTokenizer.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronFieldAwareConv::calcInputGradients(cTokenizer.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTokenizer.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronFieldAwareConv::updateInputWeights(cTokenizer.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::Save(const int file_handle)
  {
   if(!CNeuronFieldAwareConv::Save(file_handle))
      ReturnFalse;
   if(!cTokenizer.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::Load(const int file_handle)
  {
   if(!CNeuronFieldAwareConv::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTokenizer.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerTokenizer::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronFieldAwareConv::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronUniMixerTokenizer* Source = source;
   dWeightsUpdate(cTokenizer, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixerTokenizer::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronFieldAwareConv::SetOpenCL(obj);
   cTokenizer.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixerTokenizer::TrainMode(bool flag)
  {
   CNeuronFieldAwareConv::TrainMode(flag);
   cTokenizer.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint units, uint blocks,
                           uint embed_size, uint candidates, uint topK,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(units <= blocks || units % blocks != 0)
      ReturnFalse;
   if(!CNeuronPerTokenSwiGLU::Init(numOutputs, myIndex, open_cl, window, units, embed_size,
                                   candidates, topK, optimization_type, batch))
      ReturnFalse;
//---
   iBlocks = blocks;
   uint block_size = units / iBlocks;
//---
   uint index = 0;
   if(!cWb.Init(0, index, OpenCL, block_size * block_size, iBlocks, block_size, candidates, topK, optimization, iBatch))
      ReturnFalse;
   cWb.SetActivationFunction(None);
   index++;
   if(!cLocalMixing.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cLocalMixing.SetActivationFunction(None);
   for(uint i = 0; i < cG.Size(); i++)
     {
      index++;
      if(!cG[i].Init(0, index, OpenCL, blocks * (blocks + 3) / 4, optimization, iBatch))
         ReturnFalse;
      cG[i].SetActivationFunction(None);
     }
   index++;
   if(!cGlobalMixing[0].Init(0, index, OpenCL, block_size * window * (blocks + 3) / 4, optimization, iBatch))
      ReturnFalse;
   cGlobalMixing[0].SetActivationFunction(None);
   index++;
   if(!cGlobalMixing[1].Init(0, index, OpenCL, window * units, optimization, iBatch))
      ReturnFalse;
   cGlobalMixing[1].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(bTrain)
     {
      if(!cWb.FeedForward())
         ReturnFalse;
      for(uint i = 0; i < cG.Size(); i++)
         if(!cG[i].FeedForward())
            ReturnFalse;
     }
//---
   uint dimension = GetWindow();
   uint units = GetFields();
   uint block_size = units / iBlocks;
//---
   if(!MatMul(cWb.getOutput(), NeuronOCL.getOutput(), cLocalMixing.getOutput(),
              block_size, block_size, dimension, iBlocks, true))
      ReturnFalse;
   if(!MatMul(cG[0].getOutput(), cLocalMixing.getOutput(), cGlobalMixing[0].getOutput(),
              cG[0].Neurons() / iBlocks, iBlocks, block_size * dimension, 1, true))
      ReturnFalse;
   if(!MatMul(cG[1].getOutput(), cGlobalMixing[0].getOutput(), cGlobalMixing[1].getOutput(),
              iBlocks, cG[1].Neurons() / iBlocks, block_size * dimension, 1, true))
      ReturnFalse;
   if(!Normalize(cGlobalMixing[1].getOutput(), dimension))
      ReturnFalse;
   if(!SumAndNormalize(cGlobalMixing[1].getOutput(), NeuronOCL.getOutput(),
                       cGlobalMixing[1].getOutput(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronPerTokenSwiGLU::feedForward(cGlobalMixing[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPerTokenSwiGLU::calcInputGradients(cGlobalMixing[1].AsObject()))
      ReturnFalse;
//---
   uint dimension = GetWindow();
   uint units = GetFields();
   uint block_size = units / iBlocks;
//---
   if(!MatMulGrad(cG[1].getOutput(), cG[1].getGradient(), cGlobalMixing[0].getOutput(),
                  cGlobalMixing[0].getGradient(), cGlobalMixing[1].getGradient(),
                  iBlocks, cG[1].Neurons() / iBlocks, block_size * dimension, 1, true))
      ReturnFalse;
   if(!MatMulGrad(cG[0].getOutput(), cG[0].getGradient(), cLocalMixing.getOutput(),
                  cLocalMixing.getGradient(), cGlobalMixing[0].getGradient(),
                  cG[0].Neurons() / iBlocks, iBlocks, block_size * dimension, 1, true))
      ReturnFalse;
   if(!MatMulGrad(cWb.getOutput(), cWb.getGradient(), NeuronOCL.getOutput(),
                  NeuronOCL.getGradient(), cLocalMixing.getGradient(),
                  block_size, block_size, dimension, iBlocks, true))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cGlobalMixing[1].getGradient(),
                       NeuronOCL.getGradient(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cWb.UpdateInputWeights())
      ReturnFalse;
   for(uint i = 0; i < cG.Size(); i++)
      if(!cG[i].UpdateInputWeights())
         ReturnFalse;
   if(!CNeuronPerTokenSwiGLU::updateInputWeights(cGlobalMixing[1].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronPerTokenSwiGLU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUniMixer* Source = source;
   dWeightsUpdate(cWb, Source, tau);
   for(uint i = 0; i < cG.Size(); i++)
      dWeightsUpdate(cG[i], Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::Save(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Save(file_handle))
      ReturnFalse;
   if(!cWb.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cG.Size(); i++)
      if(!cG[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixer::Load(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cWb.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cG.Size(); i++)
      if(!LoadInsideLayer(file_handle, cG[i].AsObject()))
         ReturnFalse;
//---
   iBlocks = cWb.GetFields();
   uint dimension = GetWindow();
   uint units = GetFields();
   uint block_size = units / iBlocks;
//---
   uint index = 1;
   if(!cLocalMixing.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cLocalMixing.SetActivationFunction(None);
   index += cG.Size() + 1;
   if(!cGlobalMixing[0].Init(0, index, OpenCL, block_size * dimension * (iBlocks + 3) / 4, optimization, iBatch))
      ReturnFalse;
   cGlobalMixing[0].SetActivationFunction(None);
   index++;
   if(!cGlobalMixing[1].Init(0, index, OpenCL, dimension * units, optimization, iBatch))
      ReturnFalse;
   cGlobalMixing[1].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixer::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronPerTokenSwiGLU::SetOpenCL(obj);
//---
   cWb.SetOpenCL(OpenCL);
   for(uint i = 0; i < cG.Size(); i++)
     {
      cG[i].SetOpenCL(OpenCL);
      cGlobalMixing[i].SetOpenCL(OpenCL);
     }
   cLocalMixing.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixer::TrainMode(bool flag)
  {
   CNeuronPerTokenSwiGLU::TrainMode(flag);
//---
   cWb.TrainMode(bTrain);
   for(uint i = 0; i < cG.Size(); i++)
      cG[i].TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint &dimensions[], uint units_s, uint blocks, uint scenarios,
                                uint stack_size, uint layers, uint embed_size, uint candidates,
                                uint topK, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() <= units_s || layers < 1)
      ReturnFalse;
   uint units_ns = dimensions.Size() - units_s;
   uint inside_embed_size = (embed_size + 1) / 2;
//---
   if(!CNeuronPerTokenSwiGLU::Init(numOutputs, myIndex, open_cl, embed_size, blocks,
                                   inside_embed_size, candidates, topK, optimization_type, batch))
      ReturnFalse;
//--- Prepare
   uint index = 0;
   if(!cPrepare.Init(0, index, OpenCL, dimensions, units_s + units_ns, embed_size, blocks, candidates, topK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLastNonSequence.Init(0, index, OpenCL, embed_size * units_ns, optimization, iBatch))
      ReturnFalse;
   cLastNonSequence.SetActivationFunction(None);
   index++;
   if(!cScenarios.Init(0, index, OpenCL, embed_size, units_s, embed_size,
                       scenarios, inside_embed_size, candidates, topK, optimization, iBatch))
      ReturnFalse;
   cScenarios.SetActivationFunction(None);
//---
   cFeatures.Clear();
   cFlow.Clear();
   cOutputProj.Clear();
   cFeatures.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
   cOutputProj.SetOpenCL(OpenCL);
//--- Features
   index++;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, embed_size * units_s, optimization, iBatch) ||
      !cFeatures.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   CNeuronAddToStack* stack = new CNeuronAddToStack();
   if(!stack ||
      !stack.Init(0, index, OpenCL, stack_size, embed_size, units_s, optimization, iBatch) ||
      !cFeatures.Add(stack))
      DeleteObjAndFalse(stack);
   index++;
   CNeuronSiameseNorm* mixing = new CNeuronSiameseNorm();
   if(!mixing ||
      !mixing.Init(0, index, OpenCL, embed_size, units_s * stack_size, blocks, inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cFeatures.Add(mixing))
      DeleteObjAndFalse(mixing);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, mixing.Neurons(), optimization, iBatch) ||
      !cFeatures.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
//--- Flow
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, mixing.Neurons() + cScenarios.Neurons() + cLastNonSequence.Neurons(), optimization, iBatch) ||
      !cFlow.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   for(uint i = 0; i < layers; i++)
     {
      index++;
      mixing = new CNeuronSiameseNorm();
      if(!mixing ||
         !mixing.Init(0, index, OpenCL, embed_size, units_s * stack_size + units_ns + scenarios, blocks, inside_embed_size,
                      candidates, topK, optimization, iBatch) ||
         !cFlow.Add(mixing))
         DeleteObjAndFalse(mixing);
     }
//--- Ouptput Projection
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, mixing.Neurons(), optimization, iBatch) ||
      !cOutputProj.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   uint window_in = neuron.Neurons() / blocks;
   CNeuronFieldAwareConv* proj = new CNeuronFieldAwareConv();
   if(!proj ||
      !proj.Init(0, index, OpenCL, window_in, embed_size, blocks, inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cOutputProj.Add(proj))
      DeleteObjAndFalse(proj);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Features
   CNeuronBaseOCL* feature = cFeatures[0];
   if(!feature ||
      !DeConcat(feature.getOutput(), cLastNonSequence.getOutput(), cPrepare.getOutput(),
                feature.Neurons(), cLastNonSequence.Neurons(), 1))
      ReturnFalse;
   for(int i = 1; i < cFeatures.Total() - 1; i++)
     {
      feature = cFeatures[i];
      if(!feature ||
         !feature.FeedForward(cFeatures[i - 1]))
         ReturnFalse;
     }
   if(feature.Type() == defNeuronSiameseNorm)
     {
      CNeuronSiameseNorm* mixing = feature;
      CNeuronBaseOCL*     Y = mixing.GetYOut();
      uint dimension = mixing.GetWindow();
      uint units = mixing.GetFields();
      feature = cFeatures[-1];
      if(!feature)
         ReturnFalse;
      if(!Concat(Y.getOutput(), Y.getOutput(), feature.getOutput(), dimension, 0, units))
         ReturnFalse;
      if(!Normalize(feature.getOutput(), dimension))
         ReturnFalse;
      if(!SumAndNormalize(mixing.getOutput(), feature.getOutput(), feature.getOutput(), dimension, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   else
     {
      feature = cFeatures[-1];
      if(!feature ||
         !feature.FeedForward(cFeatures[-2]))
         ReturnFalse;
     }
//--- Scenarios
   if(!cScenarios.FeedForward(cFeatures[0]))
      ReturnFalse;
//--- Flow
   uint blocks = CNeuronPerTokenSwiGLU::GetFields();
   CNeuronBaseOCL* flow = cFlow[0];
   if(!flow ||
      !Concat(cLastNonSequence.getOutput(), cScenarios.getOutput(), feature.getOutput(), flow.getOutput(),
              cLastNonSequence.Neurons() / blocks, cScenarios.Neurons() / blocks, feature.Neurons() / blocks, blocks))
      ReturnFalse;
   flow = cFlow[1];
   if(!flow ||
      !flow.FeedForward(cFlow[0], cFlow[0].getOutput()))
      ReturnFalse;
   for(int i = 2; i < cFlow.Total(); i++)
     {
      flow = cFlow[i];
      if(!flow ||
         !flow.FeedForward(cFlow[i - 1]))
         ReturnFalse;
     }
//--- Output Projection
   if(flow.Type() != defNeuronSiameseNorm)
      ReturnFalse;
   CNeuronSiameseNorm* mixing = flow;
   CNeuronBaseOCL*     Y = mixing.GetYOut();
   uint dimension = mixing.GetWindow();
   uint units = mixing.GetFields();
   CNeuronBaseOCL* proj = cOutputProj[0];
   if(!proj)
      ReturnFalse;
   if(!Concat(Y.getOutput(), Y.getOutput(), proj.getOutput(), dimension, 0, units))
      ReturnFalse;
   if(!Normalize(proj.getOutput(), dimension))
      ReturnFalse;
   if(!SumAndNormalize(mixing.getOutput(), proj.getOutput(), proj.getOutput(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   for(int i = 1; i < cOutputProj.Total(); i++)
     {
      proj = cOutputProj[i];
      if(!proj ||
         !proj.FeedForward(cOutputProj[i - 1]))
         ReturnFalse;
     }
   if(!CNeuronPerTokenSwiGLU::feedForward(proj))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPerTokenSwiGLU::calcInputGradients(cOutputProj[-1]))
      ReturnFalse;
//--- Output Projection
   CNeuronBaseOCL* proj = NULL;
   for(int i = cOutputProj.Total() - 2; i >= 0; i--)
     {
      proj = cOutputProj[i];
      if(!proj ||
         !proj.CalcHiddenGradients(cOutputProj[i + 1]))
         ReturnFalse;
     }
//--- Flow
   CNeuronBaseOCL* flow = cFlow[-1];
   if(flow.Type() != defNeuronSiameseNorm)
      ReturnFalse;
   CNeuronSiameseNorm* mixing = flow;
   CNeuronBaseOCL* Y = mixing.GetYOut();
   if(!DeActivation(mixing.getOutput(), mixing.getGradient(), proj.getGradient(), mixing.Activation()))
      ReturnFalse;
   if(!DeActivation(Y.getOutput(), Y.getGradient(), proj.getGradient(), Y.Activation()))
      ReturnFalse;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      flow = cFlow[i];
      if(!flow ||
         !flow.CalcHiddenGradients(cFlow[i + 1]))
         ReturnFalse;
     }
//--- Features
   CNeuronBaseOCL* feature = cFeatures[-1];
   uint blocks = CNeuronPerTokenSwiGLU::GetFields();
   if(!feature ||
      !DeConcat(cLastNonSequence.getGradient(), cScenarios.getGradient(), feature.getGradient(), flow.getGradient(),
                cLastNonSequence.Neurons() / blocks, cScenarios.Neurons() / blocks, feature.Neurons() / blocks, blocks))
      ReturnFalse;
   Deactivation(feature);
   feature = cFeatures[-2];
   if(!feature)
      ReturnFalse;
   if(feature.Type() == defNeuronSiameseNorm)
     {
      mixing = feature;
      Y = mixing.GetYOut();
      if(!DeActivation(feature.getOutput(), feature.getGradient(), cFeatures[-1].getGradient(), feature.Activation()))
         ReturnFalse;
      if(!DeActivation(Y.getOutput(), Y.getGradient(), cFeatures[-1].getGradient(), Y.Activation()))
         ReturnFalse;
     }
   else
      if(!feature.CalcHiddenGradients(cFeatures[-1]))
         ReturnFalse;
   for(int i = cFeatures.Total() - 3; i >= 0; i--)
     {
      feature = cFeatures[i];
      if(!feature ||
         !feature.CalcHiddenGradients(cFeatures[i + 1]))
         ReturnFalse;
     }
//--- Scenarios
   Deactivation(cScenarios);
   CBufferFloat *temp = feature.getGradient();
   if(!feature.SetGradient(feature.getPrevOutput(), false) ||
      !feature.CalcHiddenGradients(cScenarios.AsObject()) ||
      !SumAndNormalize(temp, feature.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !feature.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!Concat(feature.getGradient(), cLastNonSequence.getGradient(), cPrepare.getGradient(),
              feature.Neurons(), cLastNonSequence.Neurons(), 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cPrepare.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//--- Features
   CNeuronBaseOCL* feature = NULL;
   for(int i = 1; i < cFeatures.Total() - 1; i++)
     {
      feature = cFeatures[i];
      if(!feature ||
         !feature.UpdateInputWeights(cFeatures[i - 1]))
         ReturnFalse;
     }
   if(feature.Type() != defNeuronSiameseNorm)
     {
      feature = cFeatures[-1];
      if(!feature ||
         !feature.UpdateInputWeights(cFeatures[-2]))
         ReturnFalse;
     }
//--- Scenarios
   if(!cScenarios.UpdateInputWeights(cFeatures[0]))
      ReturnFalse;
//--- Flow
   CNeuronBaseOCL* flow = NULL;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      flow = cFlow[i];
      if(!flow ||
         !flow.UpdateInputWeights(cFlow[i - 1]))
         ReturnFalse;
     }
//--- Output Projection
   CNeuronBaseOCL* proj = NULL;
   for(int i = 1; i < cOutputProj.Total(); i++)
     {
      proj = cOutputProj[i];
      if(!proj ||
         !proj.UpdateInputWeights(cOutputProj[i - 1]))
         ReturnFalse;
     }
   if(!CNeuronPerTokenSwiGLU::updateInputWeights(proj))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::Save(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cLastNonSequence.Save(file_handle))
      ReturnFalse;
   if(!cScenarios.Save(file_handle))
      ReturnFalse;
   if(!cFeatures.Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
   if(!cOutputProj.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::Load(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cPrepare.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLastNonSequence.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScenarios.AsObject()))
      ReturnFalse;
   if(!cFeatures.Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
   if(!cOutputProj.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::Clear(void)
  {
   if(!CNeuronPerTokenSwiGLU::Clear())
      ReturnFalse;
//---
   if(!cPrepare.Clear())
      ReturnFalse;
   if(!cLastNonSequence.Clear())
      ReturnFalse;
   if(!cScenarios.Clear())
      ReturnFalse;
   if(!cFeatures.ClearStates())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
   if(!cOutputProj.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUniMixerBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronPerTokenSwiGLU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUniMixerBlock* Source = source;
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cLastNonSequence, Source, tau);
   dWeightsUpdate(cScenarios, Source, tau);
   dWeightsUpdate(cFeatures, Source, tau);
   dWeightsUpdate(cFlow, Source, tau);
   dWeightsUpdate(cOutputProj, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixerBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronPerTokenSwiGLU::SetOpenCL(obj);
//---
   cPrepare.SetOpenCL(OpenCL);
   cLastNonSequence.SetOpenCL(OpenCL);
   cScenarios.SetOpenCL(OpenCL);
   cFeatures.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
   cOutputProj.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUniMixerBlock::TrainMode(bool flag)
  {
   CNeuronPerTokenSwiGLU::TrainMode(flag);
//---
   cPrepare.TrainMode(bTrain);
   cLastNonSequence.TrainMode(bTrain);
   cScenarios.TrainMode(bTrain);
   cFeatures.TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
   cOutputProj.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                  uint window_in, uint &fields[], uint window_out,
                                  uint embed_size, uint candidates, uint topK,
                                  ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- Fields require [Sequence, Context, Global]
   if(fields.Size() < 3)
      ReturnFalse;
   uint fields_total = 0;
   for(int i = 0; i < 3; i++)
     {
      if(fields[i] <= 0)
         ReturnFalse;
      fields_total += fields[i];
     }
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, fields_total * window_out, optimization_type, batch))
      ReturnFalse;
//---
   cSequence.Clear();
   cContext.Clear();
   cGlobal.Clear();
   cSequence.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cGlobal.SetOpenCL(OpenCL);
   uint index = 0;
//--- Sequence
     {
      CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, window_in * fields[0], optimization, iBatch) ||
         !cSequence.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
      index++;
      CNeuronScenariosToken* scen = new CNeuronScenariosToken();
      if(!scen ||
         !scen.Init(0, index, OpenCL, window_in, fields[0], window_out, fields[0], embed_size, candidates, topK, optimization, iBatch) ||
         !cSequence.Add(scen))
         DeleteObjAndFalse(scen);
      scen.SetActivationFunction(None);
     }
//--- Context
     {
      index++;
      CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, window_in * fields[1], optimization, iBatch) ||
         !cContext.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
      index++;
      CNeuronFieldPatternEmbedding* scen = new CNeuronFieldPatternEmbedding();
      if(!scen ||
         !scen.Init(fields[1]*window_out, index, OpenCL, window_in, fields[1], (window_out + 1) / 2, candidates, topK, optimization, iBatch) ||
         !cContext.Add(scen))
         DeleteObjAndFalse(scen);
      scen.SetActivationFunction(None);
      CNeuronBaseOCL* mix = new CNeuronBaseOCL();
      if(!mix ||
         !mix.Init(0, index, OpenCL, window_out * fields[1], optimization, iBatch) ||
         !cContext.Add(mix))
         DeleteObjAndFalse(mix);
      mix.SetActivationFunction(None);
     }
//--- Global
     {
      index++;
      CFieldAwareParams* scen = new CFieldAwareParams();
      if(!scen ||
         !scen.Init(0, index, OpenCL, window_out, fields[2], embed_size, candidates, topK, optimization, iBatch) ||
         !cGlobal.Add(scen))
         DeleteObjAndFalse(scen);
      scen.SetActivationFunction(None);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSequence[0];
   CNeuronBaseOCL* context = cContext[0];
   CNeuronBaseOCL* global = NULL;
//---
   if(!sequence || !context ||
      !DeConcat(sequence.getOutput(), context.getOutput(), NeuronOCL.getOutput(),
                sequence.Neurons(), context.Neurons(), 1))
      ReturnFalse;
//--- Sequence
   for(int i = 1; i < cSequence.Total(); i++)
     {
      sequence = cSequence[i];
      if(!sequence ||
         !sequence.FeedForward(cSequence[i - 1]))
         ReturnFalse;
     }
//--- Context
   for(int i = 1; i < cContext.Total(); i++)
     {
      context = cContext[i];
      if(!context ||
         !context.FeedForward(cContext[i - 1]))
         ReturnFalse;
     }
//--- Global
   if(bTrain)
     {
      for(int i = 0; i < cGlobal.Total(); i++)
        {
         global = cGlobal[i];
         if(!global ||
            !global.FeedForward((i == 0 ? NeuronOCL : cGlobal[i - 1])))
            ReturnFalse;
        }
     }
   else
     {
      global = cGlobal[-1];
      if(!global)
         ReturnFalse;
     }
//--- Concatenate
   if(!Concat(sequence.getOutput(), context.getOutput(), global.getOutput(), Output,
              sequence.Neurons(), context.Neurons(), global.Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSequence[-1];
   CNeuronBaseOCL* context = cContext[-1];
   CNeuronBaseOCL* global = cGlobal[-1];
//--- Concatenating
   if(!sequence || !context || !global ||
      !DeConcat(sequence.getGradient(), context.getGradient(), global.getGradient(), Gradient,
                sequence.Neurons(), context.Neurons(), global.Neurons(), 1))
      ReturnFalse;
   Deactivation(sequence);
   Deactivation(context);
   Deactivation(global);
//--- Global
   for(int i = cGlobal.Total() - 2; i >= 0; i--)
     {
      global = cGlobal[i];
      if(!global ||
         !global.CalcHiddenGradients(cGlobal[i + 1]))
         ReturnFalse;
     }
//--- Context
   for(int i = cContext.Total() - 2; i >= 0; i--)
     {
      context = cContext[i];
      if(!context ||
         !context.CalcHiddenGradients(cContext[i + 1]))
         ReturnFalse;
     }
//--- Sequence
   for(int i = cSequence.Total() - 2; i >= 0; i--)
     {
      sequence = cSequence[i];
      if(!sequence ||
         !sequence.CalcHiddenGradients(cSequence[i + 1]))
         ReturnFalse;
     }
//---
   if(!Concat(sequence.getGradient(), context.getGradient(), NeuronOCL.getGradient(),
              sequence.Neurons(), context.Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* neuron = NULL;
//--- Sequence
   for(int i = 1; i < cSequence.Total(); i++)
     {
      neuron = cSequence[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(cSequence[i - 1]))
         ReturnFalse;
     }
//--- Context
   for(int i = 1; i < cContext.Total(); i++)
     {
      neuron = cContext[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(cContext[i - 1]))
         ReturnFalse;
     }
//--- Global
   for(int i = 0; i < cGlobal.Total(); i++)
     {
      neuron = cGlobal[i];
      if(!neuron ||
         !neuron.UpdateInputWeights((i == 0 ? NeuronOCL : cGlobal[i - 1])))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cSequence.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cGlobal.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!cSequence.Load(file_handle))
      ReturnFalse;
   if(!cContext.Load(file_handle))
      ReturnFalse;
   if(!cGlobal.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronINFNetScenarios* Source = source;
   dWeightsUpdate(cSequence, Source, tau);
   dWeightsUpdate(cContext, Source, tau);
   dWeightsUpdate(cGlobal, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetScenarios::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cSequence.ClearStates())
      ReturnFalse;
   if(!cContext.ClearStates())
      ReturnFalse;
   if(!cGlobal.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetScenarios::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cSequence.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cGlobal.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetScenarios::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cSequence.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cGlobal.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint window, uint units_count, uint hub_size, uint variables,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint hidden_dim = (window + 3) / 4;
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, hidden_dim * units_count, hidden_dim * units_count, hub_size, 1, variables, optimization_type, batch))
      ReturnFalse;
   if(!cProjection.Init(0, 0, OpenCL, window, window, hidden_dim, units_count, variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::calcInputGradients(cProjection.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronINFNetHUBSequence* Source = source;
   dWeightsUpdate(cProjection, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetHUBSequence::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cProjection.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetHUBSequence::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cProjection.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetHUBSequence::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cProjection.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                              uint &dimensions[], uint units_s, uint heads, uint stack_size,
                              uint layers, uint embed_size, uint candidates, uint topK,
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() <= units_s || layers < 1 || units_s < 1)
      ReturnFalse;
   iSequenceVar = units_s;
   iSequenceUnits = stack_size;
   iContextUnits = dimensions.Size() - units_s;
   iDimension = embed_size;
   iScenariosUnits = iSequenceVar + iContextUnits + 1;
   uint inside_embed_size = (embed_size + 1) / 2;
//---
   if(!CNeuronPerTokenSwiGLU::Init(numOutputs, myIndex, open_cl, iDimension, iScenariosUnits + iSequenceVar + 1,
                                   inside_embed_size, candidates, topK, optimization_type, batch))
      ReturnFalse;
//--- Prepare
   uint index = 0;
   if(!cPrepare.Init(0, index, OpenCL, dimensions, iSequenceVar + iContextUnits, iDimension,
                     iSequenceVar + iContextUnits, candidates, topK, optimization, iBatch))
      ReturnFalse;
//---
   cSequence.Clear();
   cContext.Clear();
   cScenarios.Clear();
   cHUBs.Clear();
   cSequence.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cScenarios.SetOpenCL(OpenCL);
   cHUBs.SetOpenCL(OpenCL);
//--- Sequence
   index++;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iDimension * iSequenceVar, optimization, iBatch) ||
      !cSequence.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   CNeuronAddToStack* stack = new CNeuronAddToStack();
   if(!stack ||
      !stack.Init(0, index, OpenCL, iSequenceUnits, iDimension, iSequenceVar, optimization, iBatch) ||
      !cSequence.Add(stack))
      DeleteObjAndFalse(stack);
   for(uint i = 0; i < layers; i++)
     {
      index++;
      neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, stack.Neurons(), optimization, iBatch) ||
         !cSequence.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
     }
//--- Context
   for(uint i = 0; i <= layers; i++)
     {
      index++;
      neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, iDimension * iContextUnits, optimization, iBatch) ||
         !cContext.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
     }
//--- Scenarios
   index++;
   CNeuronINFNetScenarios* scen = new CNeuronINFNetScenarios();
   uint fields[] = {iSequenceVar, iContextUnits, 1};
   if(!scen ||
      !scen.Init(0, index, OpenCL, iDimension, fields, iDimension, inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cScenarios.Add(scen))
      DeleteObjAndFalse(scen);
   for(uint i = 0; i < layers; i++)
     {
      index++;
      neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, scen.Neurons(), optimization, iBatch) ||
         !cScenarios.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
     }
//--- HUBs
   index++;
   CNeuronINFNetHUBSequence* hub = new CNeuronINFNetHUBSequence();
   if(!hub ||
      !hub.Init(0, index, OpenCL, iDimension, iSequenceUnits, iDimension, iSequenceVar, optimization, iBatch) ||
      !cHUBs.Add(hub))
      DeleteObjAndFalse(hub);
   hub = new CNeuronINFNetHUBSequence();
   if(!hub ||
      !hub.Init(0, index, OpenCL, iDimension, iContextUnits, iDimension, 1, optimization, iBatch) ||
      !cHUBs.Add(hub))
      DeleteObjAndFalse(hub);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, scen.Neurons() + iDimension * (iSequenceVar + 1), optimization, iBatch) ||
      !cHUBs.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   CNeuronDomainAwareAttention* att = NULL;
   CNeuronFieldAwareConv* conv = NULL;
   for(uint i = 0; i < layers; i++)
     {
      index++;
      att = new CNeuronDomainAwareAttention();
      if(!att ||
         !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iSequenceUnits * iSequenceVar,
                   inside_embed_size, candidates, topK, optimization, iBatch) ||
         !cHUBs.Add(att))
         DeleteObjAndFalse(att);
      index++;
      att = new CNeuronDomainAwareAttention();
      if(!att ||
         !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iContextUnits,
                   inside_embed_size, candidates, topK, optimization, iBatch) ||
         !cHUBs.Add(att))
         DeleteObjAndFalse(att);
      index++;
      att = new CNeuronDomainAwareAttention();
      if(!att ||
         !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iScenariosUnits,
                   inside_embed_size, candidates, topK, optimization, iBatch) ||
         !cHUBs.Add(att))
         DeleteObjAndFalse(att);
      index++;
      neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, 3 * att.Neurons(), optimization, iBatch) ||
         !cHUBs.Add(neuron))
         DeleteObjAndFalse(neuron);
      neuron.SetActivationFunction(None);
      index++;
      conv = new CNeuronFieldAwareConv();
      if(!conv ||
         !conv.Init(0, index, OpenCL, 3 * iDimension, 2 * iDimension, iSequenceVar + iScenariosUnits + 1, inside_embed_size,
                    candidates, topK, optimization, iBatch) ||
         !cHUBs.Add(conv))
         DeleteObjAndFalse(conv);
      index++;
      conv = new CNeuronFieldAwareConv();
      if(!conv ||
         !conv.Init(0, index, OpenCL, 3 * iDimension, iDimension, iSequenceVar + iScenariosUnits + 1, inside_embed_size,
                    candidates, topK, optimization, iBatch) ||
         !cHUBs.Add(conv))
         DeleteObjAndFalse(conv);
     }
   index++;
   att = new CNeuronDomainAwareAttention();
   if(!att ||
      !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iSequenceUnits * iSequenceVar,
                inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cHUBs.Add(att))
      DeleteObjAndFalse(att);
   index++;
   att = new CNeuronDomainAwareAttention();
   if(!att ||
      !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iContextUnits,
                inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cHUBs.Add(att))
      DeleteObjAndFalse(att);
   index++;
   att = new CNeuronDomainAwareAttention();
   if(!att ||
      !att.Init(0, index, OpenCL, iDimension, iScenariosUnits + iSequenceVar + 1, heads, iDimension, iScenariosUnits,
                inside_embed_size, candidates, topK, optimization, iBatch) ||
      !cHUBs.Add(att))
      DeleteObjAndFalse(att);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 3 * att.Neurons(), optimization, iBatch) ||
      !cHUBs.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   conv = new CNeuronFieldAwareConv();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 3 * iDimension, iDimension, iSequenceVar + iScenariosUnits + 1, inside_embed_size,
                 candidates, topK, optimization, iBatch) ||
      !cHUBs.Add(conv))
      DeleteObjAndFalse(conv);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSequence[0];
   CNeuronBaseOCL* context = cContext[0];
   if(!sequence || !context ||
      !DeConcat(sequence.getOutput(), context.getOutput(), cPrepare.getOutput(),
                sequence.Neurons(), context.Neurons(), 1))
      ReturnFalse;
   sequence = cSequence[1];
   if(!sequence ||
      !sequence.FeedForward(cSequence[0]))
      ReturnFalse;
//---
   CNeuronBaseOCL* scen = cScenarios[0];
   if(!scen ||
      !scen.FeedForward(cPrepare.AsObject()))
      ReturnFalse;
//--- HUBs
   CNeuronBaseOCL* hub = cHUBs[0];
   if(!hub ||
      !hub.FeedForward(sequence))
      ReturnFalse;
   hub = cHUBs[1];
   if(!hub ||
      !hub.FeedForward(context))
      ReturnFalse;
   hub = cHUBs[2];
   if(!hub ||
      !Concat(cHUBs[0].getOutput(), cHUBs[1].getOutput(), scen.getOutput(), hub.getOutput()
              , cHUBs[0].Neurons(), cHUBs[1].Neurons(), scen.Neurons(), 1))
      ReturnFalse;
//---
   int layers = cContext.Total() - 1;
   for(int i = 0; i < layers; i++)
     {
      CNeuronBaseOCL* att_seq = cHUBs[i * 6 + 3];
      if(!att_seq ||
         !att_seq.FeedForward(hub, sequence.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* att_cont = cHUBs[i * 6 + 4];
      if(!att_cont ||
         !att_cont.FeedForward(hub, context.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* att_scen = cHUBs[i * 6 + 5];
      if(!att_scen ||
         !att_scen.FeedForward(hub, scen.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* concat = cHUBs[i * 6 + 6];
      if(!concat ||
         !Concat(att_seq.getOutput(), att_cont.getOutput(), att_scen.getOutput(),
                 concat.getOutput(), iDimension, iDimension, iDimension, iSequenceVar + iScenariosUnits + 1))
         ReturnFalse;
      CNeuronBaseOCL* scal_and_shift = cHUBs[i * 6 + 7];
      if(!scal_and_shift ||
         !scal_and_shift.FeedForward(concat))
         ReturnFalse;
      sequence = cSequence[i + 2];
      context = cContext[i + 1];
      scen = cScenarios[i + 1];
      if(!sequence || !context || !scen ||
         !INFNetBGU(scal_and_shift, cSequence[i + 1], sequence, cContext[i], context, cScenarios[i], scen))
         ReturnFalse;
      hub = cHUBs[i * 6 + 8];
      if(!hub.FeedForward(concat))
         ReturnFalse;
     }
   CNeuronBaseOCL* att_seq = cHUBs[layers * 6 + 3];
   if(!att_seq ||
      !att_seq.FeedForward(hub, sequence.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* att_cont = cHUBs[layers * 6 + 4];
   if(!att_cont ||
      !att_cont.FeedForward(hub, context.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* att_scen = cHUBs[layers * 6 + 5];
   if(!att_scen ||
      !att_scen.FeedForward(hub, scen.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* concat = cHUBs[layers * 6 + 6];
   if(!concat ||
      !Concat(att_seq.getOutput(), att_cont.getOutput(), att_scen.getOutput(),
              concat.getOutput(), iDimension, iDimension, iDimension, iSequenceVar + iScenariosUnits + 1))
      ReturnFalse;
   hub = cHUBs[layers * 6 + 7];
   if(!hub.FeedForward(concat))
      ReturnFalse;
//---
   if(!CNeuronPerTokenSwiGLU::feedForward(hub))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::INFNetBGU(CNeuronBaseOCL *scale_and_shift, CNeuronBaseOCL *sequence_in,
                                   CNeuronBaseOCL *sequence_out, CNeuronBaseOCL *context_in,
                                   CNeuronBaseOCL *context_out, CNeuronBaseOCL *scenarios_in,
                                   CNeuronBaseOCL *scenarios_out)
  {
   if(!OpenCL || !scale_and_shift || !sequence_in || !sequence_out ||
      !context_in || !context_out ||
      !scenarios_in || !scenarios_out)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { (uint)MathMin(
                                  MathMax(
                                     MathMax(iScenariosUnits, iContextUnits),
                                     iSequenceUnits),
                                  OpenCL.GetMaxLocalSize(1)),
                               iDimension,
                               iSequenceVar + 2
                             };
   uint local_work_size[] = { global_work_size[0], 1, 1};
   uint kernel = def_k_INFNetBGU;
   setBuffer(kernel, def_k_infnbgu_scal_and_shift, scale_and_shift.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_context, context_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_context_out, context_out.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_sequence, sequence_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_sequence_out, sequence_out.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_scenarios, scenarios_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgu_scenarios_out, scenarios_out.getOutputIndex())
   setArgument(kernel, def_k_infnbgu_context_units, iContextUnits)
   setArgument(kernel, def_k_infnbgu_scenarios_units, iScenariosUnits)
   setArgument(kernel, def_k_infnbgu_sequence_units, iSequenceUnits)
   setArgument(kernel, def_k_infnbgu_sequence_vars, iSequenceVar)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!scenarios_out.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   if(!Normalize(sequence_out.getOutput(), iDimension) ||
      !Normalize(context_out.getOutput(), iDimension) ||
      !Normalize(scenarios_out.getOutput(), iDimension))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::INFNetBGUGrad(CNeuronBaseOCL *scale_and_shift, CNeuronBaseOCL *sequence_in,
                                       CNeuronBaseOCL *sequence_out, CNeuronBaseOCL *context_in,
                                       CNeuronBaseOCL *context_out, CNeuronBaseOCL *scenarios_in,
                                       CNeuronBaseOCL *scenarios_out)
  {
   if(!OpenCL || !scale_and_shift || !sequence_in || !sequence_out ||
      !context_in || !context_out ||
      !scenarios_in || !scenarios_out)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { (uint)MathMin(
                                  MathMax(
                                     MathMax(iScenariosUnits, iContextUnits),
                                     iSequenceUnits),
                                  OpenCL.GetMaxLocalSize(1)),
                               iDimension,
                               iSequenceVar + 2
                             };
   uint local_work_size[] = { global_work_size[0], 1, 1};
   uint kernel = def_k_INFNetBGUGrad;
   setBuffer(kernel, def_k_infnbgugr_scal_and_shift, scale_and_shift.getOutputIndex())
   setBuffer(kernel, def_k_infnbgugr_scal_and_shift_grad, scale_and_shift.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_context, context_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgugr_context_grad, context_in.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_context_out_grad, context_out.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_sequence, sequence_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgugr_sequence_grad, sequence_in.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_sequence_out_grad, sequence_out.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_scenarios, scenarios_in.getOutputIndex())
   setBuffer(kernel, def_k_infnbgugr_scenarios_grad, scenarios_in.getGradientIndex())
   setBuffer(kernel, def_k_infnbgugr_scenarios_out_grad, scenarios_out.getGradientIndex())
   setArgument(kernel, def_k_infnbgugr_context_units, iContextUnits)
   setArgument(kernel, def_k_infnbgugr_scenarios_units, iScenariosUnits)
   setArgument(kernel, def_k_infnbgugr_sequence_units, iSequenceUnits)
   setArgument(kernel, def_k_infnbgugr_sequence_vars, iSequenceVar)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!scale_and_shift.getGradient().BufferRead())
      ReturnFalse;
#endif
   Deactivation(scenarios_in);
   Deactivation(sequence_in);
   Deactivation(context_in);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronPerTokenSwiGLU::calcInputGradients(cHUBs[-1]))
      ReturnFalse;
//---
   int layers = cContext.Total() - 1;
   CNeuronBaseOCL* concat = cHUBs[-2];
   if(!concat ||
      !concat.CalcHiddenGradients(cHUBs[-1]))
      ReturnFalse;
   CNeuronBaseOCL* att_seq = cHUBs[-5];
   CNeuronBaseOCL* att_cont = cHUBs[-4];
   CNeuronBaseOCL* att_scen = cHUBs[-3];
   if(!att_seq || !att_cont || !att_scen ||
      !DeConcat(att_seq.getGradient(), att_cont.getGradient(), att_scen.getGradient(),
                concat.getGradient(), iDimension, iDimension, iDimension, iSequenceVar + iScenariosUnits + 1))
      ReturnFalse;
   CNeuronBaseOCL* hub = cHUBs[-6];
   CNeuronBaseOCL* sequence = cSequence[-1];
   CNeuronBaseOCL* context = cContext[-1];
   CNeuronBaseOCL* scen = cScenarios[-1];
   CNeuronBaseOCL* scal_and_shift = NULL;
//---
   if(!hub || !sequence ||
      !hub.CalcHiddenGradients(att_seq, sequence.getOutput(), sequence.getGradient(),
                               (ENUM_ACTIVATION)sequence.Activation()))
      ReturnFalse;
   CBufferFloat* temp = hub.getGradient();
   if(!hub.SetGradient(hub.getPrevOutput(), false) || !context ||
      !hub.CalcHiddenGradients(att_cont, context.getOutput(), context.getGradient(),
                               (ENUM_ACTIVATION)context.Activation()) ||
      !SumAndNormalize(temp, hub.getGradient(), temp, iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!scen ||
      !hub.CalcHiddenGradients(att_scen, scen.getOutput(), scen.getGradient(),
                               (ENUM_ACTIVATION)scen.Activation()) ||
      !SumAndNormalize(temp, hub.getGradient(), temp, iDimension, false, 0, 0, 0, 1) ||
      !hub.SetGradient(temp, false))
      ReturnFalse;
//---
   for(int i = layers - 1; i >= 0; i--)
     {
      att_seq = cHUBs[i * 6 + 3];
      att_cont = cHUBs[i * 6 + 4];
      att_scen = cHUBs[i * 6 + 5];
      concat = cHUBs[i * 6 + 6];
      scal_and_shift = cHUBs[i * 6 + 7];
      sequence = cSequence[i + 1];
      context = cContext[i];
      scen = cScenarios[i];
      if(!sequence || !context || !scen ||
         !INFNetBGUGrad(scal_and_shift, sequence, cSequence[i + 2], context, cContext[i + 1], scen, cScenarios[i + 1]))
         ReturnFalse;
      if(!concat ||
         !concat.CalcHiddenGradients(scal_and_shift))
         ReturnFalse;
      temp = concat.getGradient();
      if(!concat.SetGradient(concat.getPrevOutput(), false) ||
         !concat.CalcHiddenGradients(hub) ||
         !SumAndNormalize(temp, concat.getGradient(), temp, iDimension, false, 0, 0, 0, 1) ||
         !concat.SetGradient(temp, false))
         ReturnFalse;
      hub = cHUBs[i * 6 + 2];
      if(!att_seq || !att_cont || !att_scen ||
         !DeConcat(att_seq.getGradient(), att_cont.getGradient(), att_scen.getGradient(),
                   concat.getGradient(), iDimension, iDimension, iDimension, iSequenceVar + iScenariosUnits + 1))
         ReturnFalse;
      //---
      if(!hub || !sequence ||
         !hub.CalcHiddenGradients(att_seq, sequence.getOutput(), sequence.getGradient(),
                                  (ENUM_ACTIVATION)sequence.Activation()))
         ReturnFalse;
      temp = hub.getGradient();
      if(!hub.SetGradient(hub.getPrevOutput(), false) || !context ||
         !hub.CalcHiddenGradients(att_cont, context.getOutput(), context.getGradient(),
                                  (ENUM_ACTIVATION)context.Activation()) ||
         !SumAndNormalize(temp, hub.getGradient(), temp, iDimension, false, 0, 0, 0, 1))
         ReturnFalse;
      if(!scen ||
         !hub.CalcHiddenGradients(att_scen, scen.getOutput(), scen.getGradient(),
                                  (ENUM_ACTIVATION)scen.Activation()) ||
         !SumAndNormalize(temp, hub.getGradient(), temp, iDimension, false, 0, 0, 0, 1) ||
         !hub.SetGradient(temp, false))
         ReturnFalse;
     }
   if(!cHUBs[0] || !cHUBs[1] ||
      !DeConcat(cHUBs[0].getGradient(), cHUBs[1].getGradient(), scen.getPrevOutput(), hub.getGradient(),
                cHUBs[0].Neurons(), cHUBs[1].Neurons(), scen.Neurons(), 1))
      ReturnFalse;
   if(!SumAndNormalize(scen.getGradient(), scen.getPrevOutput(), scen.getGradient(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
   temp = context.getGradient();
   if(!context.SetGradient(context.getPrevOutput(), false) ||
      !context.CalcHiddenGradients(cHUBs[1]) ||
      !SumAndNormalize(temp, context.getGradient(), temp, iDimension, false, 0, 0, 0, 1) ||
      !context.SetGradient(temp, false))
      ReturnFalse;
   temp = sequence.getGradient();
   if(!sequence.SetGradient(sequence.getPrevOutput(), false) ||
      !sequence.CalcHiddenGradients(cHUBs[0]) ||
      !SumAndNormalize(temp, sequence.getGradient(), temp, iDimension, false, 0, 0, 0, 1) ||
      !sequence.SetGradient(temp, false))
      ReturnFalse;
   sequence = cSequence[0];
   if(!sequence ||
      !sequence.CalcHiddenGradients(cSequence[1]))
      ReturnFalse;
//---
   if(!cPrepare.CalcHiddenGradients(scen))
      ReturnFalse;
   if(!Concat(sequence.getGradient(), context.getGradient(), cPrepare.getPrevOutput(),
              sequence.Neurons(), context.Neurons(), 1))
      ReturnFalse;
   if(cPrepare.Activation() != None)
      if(!DeActivation(cPrepare.getOutput(), cPrepare.getPrevOutput(), cPrepare.getPrevOutput(), cPrepare.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cPrepare.getGradient(), cPrepare.getPrevOutput(), cPrepare.getGradient(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cPrepare.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSequence[1];
   CNeuronBaseOCL* context = cContext[0];
   if(!sequence ||
      !sequence.UpdateInputWeights(cSequence[0]))
      ReturnFalse;
//---
   CNeuronBaseOCL* scen = cScenarios[0];
   if(!scen ||
      !scen.UpdateInputWeights(cPrepare.AsObject()))
      ReturnFalse;
//--- HUBs
   CNeuronBaseOCL* hub = cHUBs[0];
   if(!hub ||
      !hub.UpdateInputWeights(sequence))
      ReturnFalse;
   hub = cHUBs[1];
   if(!hub ||
      !hub.UpdateInputWeights(context))
      ReturnFalse;
   hub = cHUBs[2];
//---
   int layers = cContext.Total() - 1;
   for(int i = 0; i < layers; i++)
     {
      CNeuronBaseOCL* att_seq = cHUBs[i * 6 + 3];
      if(!att_seq ||
         !att_seq.UpdateInputWeights(hub, sequence.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* att_cont = cHUBs[i * 6 + 4];
      if(!att_cont ||
         !att_cont.UpdateInputWeights(hub, context.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* att_scen = cHUBs[i * 6 + 5];
      if(!att_scen ||
         !att_scen.UpdateInputWeights(hub, scen.getOutput()))
         ReturnFalse;
      CNeuronBaseOCL* concat = cHUBs[i * 6 + 6];
      CNeuronBaseOCL* scal_and_shift = cHUBs[i * 6 + 7];
      if(!scal_and_shift ||
         !scal_and_shift.UpdateInputWeights(concat))
         ReturnFalse;
      hub = cHUBs[i * 6 + 8];
      if(!hub.UpdateInputWeights(concat))
         ReturnFalse;
     }
   CNeuronBaseOCL* att_seq = cHUBs[layers * 6 + 3];
   if(!att_seq ||
      !att_seq.UpdateInputWeights(hub, sequence.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* att_cont = cHUBs[layers * 6 + 4];
   if(!att_cont ||
      !att_cont.UpdateInputWeights(hub, context.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* att_scen = cHUBs[layers * 6 + 5];
   if(!att_scen ||
      !att_scen.UpdateInputWeights(hub, scen.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* concat = cHUBs[layers * 6 + 6];
   hub = cHUBs[layers * 6 + 7];
   if(!hub.UpdateInputWeights(concat))
      ReturnFalse;
//---
   if(!CNeuronPerTokenSwiGLU::updateInputWeights(hub))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::Save(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Save(file_handle))
      ReturnFalse;
//---
   dFileWriteUInt(file_handle, iDimension);
   dFileWriteUInt(file_handle, iSequenceUnits);
   dFileWriteUInt(file_handle, iSequenceVar);
   dFileWriteUInt(file_handle, iContextUnits);
   dFileWriteUInt(file_handle, iScenariosUnits);
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cSequence.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cScenarios.Save(file_handle))
      ReturnFalse;
   if(!cHUBs.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::Load(const int file_handle)
  {
   if(!CNeuronPerTokenSwiGLU::Load(file_handle))
      ReturnFalse;
//---
   dFileReadUInt(file_handle, iDimension);
   dFileReadUInt(file_handle, iSequenceUnits);
   dFileReadUInt(file_handle, iSequenceVar);
   dFileReadUInt(file_handle, iContextUnits);
   dFileReadUInt(file_handle, iScenariosUnits);
   if(!LoadInsideLayer(file_handle, cPrepare.AsObject()))
      ReturnFalse;
   if(!cSequence.Load(file_handle))
      ReturnFalse;
   if(!cContext.Load(file_handle))
      ReturnFalse;
   if(!cScenarios.Load(file_handle))
      ReturnFalse;
   if(!cHUBs.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::Clear(void)
  {
   if(!CNeuronPerTokenSwiGLU::Clear())
      ReturnFalse;
//---
   if(!cPrepare.Clear())
      ReturnFalse;
   if(!cSequence.ClearStates())
      ReturnFalse;
   if(!cContext.ClearStates())
      ReturnFalse;
   if(!cScenarios.ClearStates())
      ReturnFalse;
   if(!cHUBs.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronINFNetBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronPerTokenSwiGLU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronINFNetBlock* Source = source;
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cSequence, Source, tau);
   dWeightsUpdate(cContext, Source, tau);
   dWeightsUpdate(cScenarios, Source, tau);
   dWeightsUpdate(cHUBs, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronPerTokenSwiGLU::SetOpenCL(obj);
//---
   cPrepare.SetOpenCL(OpenCL);
   cSequence.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cScenarios.SetOpenCL(OpenCL);
   cHUBs.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronINFNetBlock::TrainMode(bool flag)
  {
   CNeuronPerTokenSwiGLU::TrainMode(flag);
//---
   cPrepare.TrainMode(bTrain);
   cSequence.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cScenarios.TrainMode(bTrain);
   cHUBs.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint dimension, uint units, uint embed_size,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, embed_size, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cNorm.Init(0, index, OpenCL, dimension * units, iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   index++;
   uint units_out = MathMax(dimension - 1, 1);
   uint win_out = (embed_size / 2 + units_out - 1) / units_out;
   if(!cProj[0].Init(0, index, OpenCL, MathMin(3, dimension), 1, win_out, units_out, units, optimization, iBatch))
      ReturnFalse;
   cProj[0].SetActivationFunction(SIGMOID);
   index++;
   if(!cProj[1].Init(embed_size, index, OpenCL, 2 * win_out * units_out, win_out * units_out, embed_size / 2, units - 1, 1, optimization, iBatch))
      ReturnFalse;
   cProj[1].SetActivationFunction(SIGMOID);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL* prev = cNorm.AsObject();
   for(uint i = 0; i < cProj.Size(); i++)
     {
      if(!cProj[i].FeedForward(prev))
         ReturnFalse;
      prev = cProj[i].AsObject();
     }
   if(!CNeuronBaseOCL::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   int last_proj = (int)cProj.Size() - 1;
   if(!CNeuronBaseOCL::calcInputGradients(cProj[last_proj].AsObject()))
      ReturnFalse;
   for(int i = last_proj - 1; i >= 0; i--)
      if(!cProj[i].CalcHiddenGradients(cProj[i + 1].AsObject()))
         ReturnFalse;
   if(!cNorm.CalcHiddenGradients(cProj[0].AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cNorm.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL* prev = cNorm.AsObject();
   for(uint i = 0; i < cProj.Size(); i++)
     {
      if(!cProj[i].UpdateInputWeights(prev))
         ReturnFalse;
      prev = cProj[i].AsObject();
     }
   if(!CNeuronBaseOCL::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cProj.Size(); i++)
      if(!cProj[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cProj.Size(); i++)
      if(!LoadInsideLayer(file_handle, cProj[i].AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSequenceTokenizer::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSequenceTokenizer* Source = source;
   dWeightsUpdate(cNorm, Source, tau);
   for(uint i = 0; i < cProj.Size(); i++)
      dWeightsUpdate(cProj[i], Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSequenceTokenizer::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cNorm.SetOpenCL(OpenCL);
   for(uint i = 0; i < cProj.Size(); i++)
      cProj[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSequenceTokenizer::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cNorm.TrainMode(bTrain);
   for(uint i = 0; i < cProj.Size(); i++)
      cProj[i].TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                      uint inpSize, uint tokenSize, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, tokenSize, optimization_type, batch))
      ReturnFalse;
//---
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
//---
   uint index = 0;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, inpSize, optimization, iBatch) ||
      !cFlow.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   CNeuronBatchNormOCL* norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, inpSize, optimization, iBatch) ||
      !cFlow.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
   index++;
   uint window = 3;
   uint units = (inpSize + window - 1) / window;
   CNeuronSpikeConvBlock* conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, window, units, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(tokenSize, index, OpenCL, 2 * window, window, window, MathMax(units - 1, 1), 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   CNeuronBaseOCL* neuron = cFlow[0];
   if(!neuron || NeuronOCL.Neurons() < neuron.Neurons() ||
      SecondInput.Total() < neuron.Neurons())
      ReturnFalse;
//---
   if(!Different(NeuronOCL.getOutput(), SecondInput, neuron.getOutput(), neuron.Neurons(), 0, 0, 0, 1))
      ReturnFalse;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      neuron = cFlow[i];
      if(!neuron ||
         !neuron.FeedForward(cFlow[i - 1]))
         ReturnFalse;
     }
//---
   if(!CNeuronBaseOCL::feedForward(neuron))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
   CNeuronBaseOCL* neuron = cFlow[0];
   if(!neuron || NeuronOCL.Neurons() < neuron.Neurons() ||
      SecondGradient.Total() < neuron.Neurons())
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//---
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      neuron = cFlow[i];
      if(!neuron ||
         !neuron.CalcHiddenGradients(cFlow[i + 1]))
         ReturnFalse;
     }
//---
   if(!DifferentGrad(NeuronOCL.getGradient(), SecondGradient, neuron.getGradient(), neuron.Neurons(), 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL);
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   CNeuronBaseOCL* neuron = cFlow[0];
   for(int i = 1; i < cFlow.Total(); i++)
     {
      neuron = cFlow[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(cFlow[i - 1]))
         ReturnFalse;
     }
//---
   if(!CNeuronBaseOCL::updateInputWeights(neuron))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDifferenceTokenizer::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronDifferenceTokenizer* Source = source;
   dWeightsUpdate(cFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDifferenceTokenizer::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDifferenceTokenizer::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint &dimensions[], uint units_s, uint heads, uint stack_size,
                                   uint layers, uint embed_size, uint candidates, uint topK,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() < 2 || layers < 1 || units_s < 1)
      ReturnFalse;
   uint iSequenceDim = units_s * dimensions[0];
   uint iContextDim = 0;
   for(uint i = 1; i < dimensions.Size(); i++)
      iContextDim += dimensions[i];
   uint iScenariosUnits = dimensions.Size() + 1;
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, (iScenariosUnits + 4)*embed_size,
                            optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   cSeqToken.Clear();
   cTransformer.Clear();
   cRecurrent.Clear();
   cSeqToken.SetOpenCL(OpenCL);
   cTransformer.SetOpenCL(OpenCL);
   cRecurrent.SetOpenCL(OpenCL);
//--- Sequence tokenizer
   uint index = 0;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iSequenceDim, optimization, iBatch) ||
      !cSeqToken.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   CNeuronSequenceTokenizer* seq_tok = new CNeuronSequenceTokenizer();
   if(!seq_tok ||
      !seq_tok.Init(0, index, OpenCL, dimensions[0], units_s, embed_size, optimization, iBatch) ||
      !cSeqToken.Add(seq_tok))
      DeleteObjAndFalse(seq_tok);
   seq_tok.SetActivationFunction(SoftPlus);
//--- Transformer
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iContextDim, optimization, iBatch) ||
      !cTransformer.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iContextDim + seq_tok.Neurons(), optimization, iBatch) ||
      !cTransformer.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   uint dims[];
   if(ArrayCopy(dims, dimensions, 0, 0, dimensions.Size()) < (int)dimensions.Size())
      ReturnFalse;
   dims[0] = seq_tok.Neurons();
   index++;
   CNeuronINFNetBlock* transf = new CNeuronINFNetBlock();
   if(!transf ||
      !transf.Init(0, index, OpenCL, dims, 1, heads, stack_size, layers, embed_size, candidates, topK, optimization, iBatch) ||
      !cTransformer.Add(transf))
      DeleteObjAndFalse(transf);
//--- Recurrent
   CNeuronLSTMOCL* lstm = NULL;
   uint size = seq_tok.Neurons();
   for(uint i = 0; i < layers; i++)
     {
      lstm = new CNeuronLSTMOCL();
      index++;
      if(!lstm ||
         !lstm.Init(0, index, OpenCL, size, optimization, iBatch) ||
         !lstm.SetInputs(size) ||
         !cRecurrent.Add(lstm))
         DeleteObjAndFalse(lstm);
      lstm.SetActivationFunction(TANH);
     }
   index++;
   CNeuronBatchNormOCL* norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, lstm.Neurons(), optimization, iBatch) ||
      !cRecurrent.Add(norm))
      DeleteObjAndFalse(norm);
//---
   index++;
   if(!cDifference.Init(0, index, OpenCL, norm.Neurons(), embed_size, optimization, iBatch))
      ReturnFalse;
   cDifference.SetActivationFunction(TANH);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSeqToken[0];
   CNeuronBaseOCL* context = cTransformer[0];
   if(!sequence || !context)
      ReturnFalse;
   uint iSequenceDim = sequence.Neurons();
   uint iContextDim = context.Neurons();
   if(!DeConcat(sequence.getOutput(), context.getOutput(), NeuronOCL.getOutput(),
                iSequenceDim, iContextDim, 1))
      ReturnFalse;
//--- Sequence tokenizer
   for(int i = 1; i < cSeqToken.Total(); i++)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.FeedForward(cSeqToken[i - 1]))
         ReturnFalse;
     }
   iSequenceDim = sequence.Neurons();
//--- Transformer
   context = cTransformer[1];
   if(!context ||
      !Concat(sequence.getOutput(), cTransformer[0].getOutput(), context.getOutput(),
              iSequenceDim, iContextDim, 1))
      ReturnFalse;
   for(int i = 2; i < cTransformer.Total(); i++)
     {
      context = cTransformer[i];
      if(!context ||
         !context.FeedForward(cTransformer[i - 1]))
         ReturnFalse;
     }
//--- Recurrent
   sequence = cRecurrent[0];
   if(!sequence ||
      !sequence.FeedForward(cSeqToken[-1]))
      ReturnFalse;
   for(int i = 1; i < cRecurrent.Total(); i++)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.FeedForward(cRecurrent[i - 1]))
         ReturnFalse;
     }
//---
   if(!cDifference.FeedForward(context, sequence.getOutput()))
      ReturnFalse;
   if(!Concat(context.getOutput(), sequence.getOutput(), cDifference.getOutput(), Output,
              context.Neurons(), sequence.Neurons(), cDifference.Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cRecurrent[-1];
   CNeuronBaseOCL* context = cTransformer[-1];
   if(!sequence || !context)
      ReturnFalse;
   uint iSequenceDim = sequence.Neurons();
   uint iContextDim = context.Neurons();
   if(!DeConcat(context.getGradient(), sequence.getGradient(), cDifference.getGradient(), Gradient,
                iContextDim, iSequenceDim, cDifference.Neurons(), 1))
      ReturnFalse;
   Deactivation(cDifference);
   Deactivation(context);
   Deactivation(sequence);
   CBufferFloat* temp = context.getGradient();
   if(!context.SetGradient(context.getPrevOutput(), false) ||
      !context.getGradient().Fill(0) ||
      !sequence.getPrevOutput().Fill(0))
      ReturnFalse;
   if(!context.CalcHiddenGradients(cDifference.AsObject(),
                                   sequence.getOutput(),
                                   sequence.getPrevOutput(),
                                   (ENUM_ACTIVATION)sequence.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(temp, context.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !context.SetGradient(temp, false) ||
      !SumAndNormalize(sequence.getGradient(), sequence.getPrevOutput(), sequence.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Recurrent
   for(int i = cRecurrent.Total() - 2; i >= 0; i--)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.CalcHiddenGradients(cRecurrent[i + 1]))
         ReturnFalse;
     }
//--- Transformer
   for(int i = cTransformer.Total() - 2; i >= 1; i--)
     {
      context = cTransformer[i];
      if(!context ||
         !context.CalcHiddenGradients(cTransformer[i + 1]))
         ReturnFalse;
     }
   context = cTransformer[0];
   sequence = cSeqToken[-1];
   if(!context || !sequence)
      ReturnFalse;
   iSequenceDim = sequence.Neurons();
   iContextDim = context.Neurons();
   if(!DeConcat(sequence.getPrevOutput(), context.getGradient(), cTransformer[1].getGradient(),
                iSequenceDim, iContextDim, 1))
      ReturnFalse;
//--- Sequence tokenizer
   if(!sequence.CalcHiddenGradients(cRecurrent[0]))
      ReturnFalse;
   if(sequence.Activation() != None)
      if(!DeActivation(sequence.getOutput(), sequence.getPrevOutput(),
                       sequence.getPrevOutput(), sequence.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(sequence.getGradient(), sequence.getPrevOutput(), sequence.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   for(int i = cSeqToken.Total() - 2; i >= 0; i--)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.CalcHiddenGradients(cSeqToken[i + 1]))
         ReturnFalse;
     }
   iSequenceDim = sequence.Neurons();
//---
   if(!Concat(sequence.getGradient(), context.getGradient(), NeuronOCL.getGradient(),
              iSequenceDim, iContextDim, 1))
      ReturnFalse;
   Deactivation(NeuronOCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//---
   CNeuronBaseOCL* sequence = cSeqToken[0];
   CNeuronBaseOCL* context = cTransformer[0];
//--- Sequence tokenizer
   for(int i = 1; i < cSeqToken.Total(); i++)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.UpdateInputWeights(cSeqToken[i - 1]))
         ReturnFalse;
     }
//--- Transformer
   for(int i = 2; i < cTransformer.Total(); i++)
     {
      context = cTransformer[i];
      if(!context ||
         !context.UpdateInputWeights(cTransformer[i - 1]))
         ReturnFalse;
     }
//--- Recurrent
   sequence = cRecurrent[0];
   if(!sequence ||
      !sequence.UpdateInputWeights(cSeqToken[-1]))
      ReturnFalse;
   for(int i = 1; i < cRecurrent.Total(); i++)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.UpdateInputWeights(cRecurrent[i - 1]))
         ReturnFalse;
     }
//---
   if(!cDifference.UpdateInputWeights(context, sequence.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cSeqToken.Save(file_handle))
      ReturnFalse;
   if(!cTransformer.Save(file_handle))
      ReturnFalse;
   if(!cRecurrent.Save(file_handle))
      ReturnFalse;
   if(!cDifference.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cSeqToken.Load(file_handle))
      ReturnFalse;
   if(!cTransformer.Load(file_handle))
      ReturnFalse;
   if(!cRecurrent.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDifference.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronReGENTADBackbone* Source = source;
   dWeightsUpdate(cSeqToken, Source, tau);
   dWeightsUpdate(cTransformer, Source, tau);
   dWeightsUpdate(cRecurrent, Source, tau);
   dWeightsUpdate(cDifference, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADBackbone::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cSeqToken.ClearStates())
      ReturnFalse;
   if(!cTransformer.ClearStates())
      ReturnFalse;
   if(!cRecurrent.ClearStates())
      ReturnFalse;
   if(!cDifference.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTADBackbone::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cSeqToken.SetOpenCL(OpenCL);
   cTransformer.SetOpenCL(OpenCL);
   cRecurrent.SetOpenCL(OpenCL);
   cDifference.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTADBackbone::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cSeqToken.TrainMode(bTrain);
   cTransformer.TrainMode(bTrain);
   cRecurrent.TrainMode(bTrain);
   cDifference.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBatchNormOCL* CNeuronReGENTADBackbone::GetNorm(void)
  {
   CNeuronBatchNormOCL* result = NULL;
   for(int i = 0; (i < cSeqToken.Total() && !result); i++)
     {
      if(!cSeqToken[i] || cSeqToken[i].Type() != defNeuronSequenceTokenizer)
         continue;
      result = ((CNeuronSequenceTokenizer*)cSeqToken[i]).GetNorm();
      break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint dimension_in, uint units_in, uint dimension_out,
                               uint units_out, CNeuronBatchNormOCL *norm,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRevINDenormOCL::Init(numOutputs, myIndex, open_cl, dimension_out * units_out, -1, NULL))
      ReturnFalse;
   activation = None;
   if(!!norm)
     {
      if(!SetNorm(norm))
         ReturnFalse;
     }
   iBatch = batch;
//---
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
//---
   uint index = 0;
   CNeuronTransposeOCL* transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, dimension_in, optimization, iBatch) ||
      !cFlow.Add(transp))
      DeleteObjAndFalse(transp);
   transp.SetActivationFunction(None);
   index++;
   uint dim_hide = (MathMax(units_in, units_out) + 1) / 2;
   CNeuronSpikeConvBlock* conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, dim_hide, dimension_in, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dim_hide, dim_hide, units_out, dimension_in, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, dimension_in, units_out, optimization, iBatch) ||
      !cFlow.Add(transp))
      DeleteObjAndFalse(transp);
   transp.SetActivationFunction(None);
   dim_hide = (MathMax(dimension_in, dimension_out) + 1) / 2;
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dimension_in, dimension_in, dim_hide, units_out, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dim_hide, dim_hide, dimension_out, units_out, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_out, dimension_out, optimization, iBatch) ||
      !cFlow.Add(transp))
      DeleteObjAndFalse(transp);
   transp.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   return CNeuronRevINDenormOCL::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* next = (cFlow.Total() == 0 ? NeuronOCL : cFlow[-1]);
   if(!CNeuronRevINDenormOCL::calcInputGradients(next))
      ReturnFalse;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(NeuronOCL != next)
      if(!NeuronOCL.CalcHiddenGradients(next))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   return CNeuronRevINDenormOCL::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::Save(const int file_handle)
  {
   if(!CNeuronRevINDenormOCL::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::Load(const int file_handle)
  {
   if(!CNeuronRevINDenormOCL::Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronRevINDenormOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronReGENTADHead* Source = source;
   dWeightsUpdate(cFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::Clear(void)
  {
   if(!CNeuronRevINDenormOCL::Clear())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTADHead::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronRevINDenormOCL::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTADHead::TrainMode(bool flag)
  {
   CNeuronRevINDenormOCL::TrainMode(flag);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTADHead::SetNorm(CNeuronBatchNormOCL *norm)
  {
   if(!norm)
      ReturnFalse;
   switch(norm.Type())
     {
      case defNeuronBatchNormOCL:
      case defNeuronBatchNormWithNoise:
         break;
      default:
         ReturnFalse;
     }
   if(BatchOptions != norm.BatchOptions)
     {
      DeleteObj(BatchOptions);
      BatchOptions = norm.BatchOptions;
     }
   iBatchSize = norm.iBatchSize;
   iBatchCount = iBatchSize;
   iBatch = iBatchSize;
   for(int i = 0; i < cFlow.Total(); i++)
      cFlow[i].iBatch = iBatch;
//---
   return !!BatchOptions;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint &dimensions[], uint units_s, uint heads, uint stack_size,
                           uint layers, uint embed_size, uint candidates, uint topK,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!cBackbone)
      cBackbone = CreateBackbone();
   if(!cBackbone ||
      !cBackbone.Init(0, 0, open_cl, dimensions, units_s, heads, stack_size, layers,
                      embed_size, candidates, topK, optimization_type, batch))
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, cBackbone.Neurons() + 3 * embed_size,
                            optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   for(uint i = 0; i < cHeads.Size(); i++)
     {
      cHeads[i].Clear();
      cHeads[i].SetOpenCL(OpenCL);
     }
//---
   CNeuronReGENTADHead*          head = NULL;
   CNeuronDifferenceTokenizer*   diff = NULL;
   uint backbone_units = cBackbone.Neurons() / embed_size;
   uint index = 1;
   for(int i = 0; i < 2; i++)
     {
      head = new CNeuronReGENTADHead();
      if(!head ||
         !head.Init(0, index, OpenCL, embed_size, backbone_units, dimensions[0], units_s,
                    cBackbone.GetNorm(), optimization, iBatch) ||
         !cHeads[i].Add(head))
         DeleteObjAndFalse(head);
      index++;
      diff = new CNeuronDifferenceTokenizer();
      if(!diff ||
         !diff.Init(0, index, OpenCL, head.Neurons(), embed_size, optimization, iBatch) ||
         !cHeads[i].Add(diff))
         DeleteObjAndFalse(diff);
      diff.SetActivationFunction(TANH);
      index++;
     }
//---
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, cBackbone.Neurons() + diff.Neurons(), optimization, iBatch) ||
      !cHeads[2].Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   head = new CNeuronReGENTADHead();
   if(!head ||
      !head.Init(0, index, OpenCL, embed_size, backbone_units + 1, dimensions[0], units_s,
                 cBackbone.GetNorm(), optimization, iBatch) ||
      !cHeads[2].Add(head))
      DeleteObjAndFalse(head);
   index++;
   diff = new CNeuronDifferenceTokenizer();
   if(!diff ||
      !diff.Init(0, index, OpenCL, head.Neurons(), embed_size, optimization, iBatch) ||
      !cHeads[2].Add(diff))
      DeleteObjAndFalse(diff);
   diff.SetActivationFunction(TANH);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !cBackbone)
      ReturnFalse;
//--- Forecats 1
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* prev = cBackbone.AsObject();
   for(int i = 0; i < cHeads[1].Total(); i++)
     {
      curr = cHeads[1][i];
      if(!curr ||
         !curr.FeedForward(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//--- Forecats 2
   curr = cHeads[2][0];
   if(!curr ||
      !Concat(cBackbone.getOutput(), prev.getOutput(), curr.getOutput(),
              cBackbone.Neurons(), prev.Neurons(), 1))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cHeads[2].Total(); i++)
     {
      curr = cHeads[2][i];
      if(!curr ||
         !curr.FeedForward(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//--- Backprop
   if(bTrain)
     {
      //--- Forecast 2
      curr = cHeads[2][-2];
      float error = 0;
      if(!curr ||
         !curr.calcOutputGradients(NeuronOCL.getOutput(), error))
         ReturnFalse;
      for(int i = cHeads[2].Total() - 3; i >= 0; i--)
        {
         curr = cHeads[2][i];
         if(!curr ||
            !curr.CalcHiddenGradients(cHeads[2][i + 1]))
            ReturnFalse;
        }
      for(int i = 1; i < cHeads[2].Total() - 1; i++)
        {
         curr = cHeads[2][i];
         if(!curr ||
            !curr.UpdateInputWeights(cHeads[2][i - 1]))
            ReturnFalse;
        }
      //--- Forecast 1
      curr = cHeads[2][0];
      prev = cHeads[1][-1];
      if(!DeConcat(cBackbone.getPrevOutput(), prev.getGradient(), curr.getGradient(),
                   cBackbone.Neurons(), prev.Neurons(), 1))
         ReturnFalse;
      Deactivation(prev);
      curr = cHeads[1][-2];
      if(!curr ||
         !curr.CalcHiddenGradients(prev, NeuronOCL.getOutput(),
                                   NeuronOCL.getGradient(),
                                   (ENUM_ACTIVATION)NeuronOCL.Activation()))
         ReturnFalse;
      CBufferFloat* temp = curr.getGradient();
      if(!curr.SetGradient(curr.getPrevOutput(), false) ||
         !curr.calcOutputGradients(NeuronOCL.getOutput(), error) ||
         !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
         !curr.SetGradient(temp, false))
         ReturnFalse;
      for(int i = cHeads[1].Total() - 3; i >= 0; i--)
        {
         curr = cHeads[1][i];
         if(!curr ||
            !curr.CalcHiddenGradients(cHeads[1][i + 1], NeuronOCL.getOutput(),
                                      NeuronOCL.getGradient(),
                                      (ENUM_ACTIVATION)NeuronOCL.Activation()))
            ReturnFalse;
        }
      if(!cBackbone.CalcHiddenGradients(curr) ||
         !SumAndNormalize(cBackbone.getGradient(), cBackbone.getPrevOutput(), cBackbone.getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
      if(!NeuronOCL.CalcHiddenGradients(cBackbone.AsObject()))
         ReturnFalse;
      if(!cBackbone.UpdateInputWeights(NeuronOCL))
         ReturnFalse;
      prev = cBackbone.AsObject();
      for(int i = 0; i < cHeads[1].Total(); i++)
        {
         curr = cHeads[1][i];
         if(!curr ||
            !curr.UpdateInputWeights(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//--- Backbone
   if(!cBackbone.FeedForward(NeuronOCL))
      ReturnFalse;
//--- Recover
   prev = cBackbone.AsObject();
   for(int i = 0; i < cHeads[0].Total(); i++)
     {
      curr = cHeads[0][i];
      if(!curr ||
         !curr.FeedForward(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!Concat(cBackbone.getOutput(), cHeads[0][-1].getOutput(), cHeads[1][-1].getOutput(), cHeads[2][-1].getOutput(), Output,
              cBackbone.Neurons(), cHeads[0][-1].Neurons(), cHeads[1][-1].Neurons(), cHeads[2][-1].Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !cBackbone)
      ReturnFalse;
//---
   if(!cHeads[0][-1] || !cHeads[1][-1] || !cHeads[2][-1] ||
      !DeConcat(cBackbone.getPrevOutput(), cHeads[0][-1].getGradient(), cHeads[1][-1].getGradient(), cHeads[2][-1].getGradient(), Gradient,
                cBackbone.Neurons(), cHeads[0][-1].Neurons(), cHeads[1][-1].Neurons(), cHeads[2][-1].Neurons(), 1))
      ReturnFalse;
//--- Forecast 2
   CNeuronBaseOCL* curr = cHeads[2][-1];
   Deactivation(curr)
   for(int i = cHeads[2].Total() - 2; i >= 0; i--)
     {
      curr = cHeads[2][i];
      if(!curr ||
         !curr.CalcHiddenGradients(cHeads[2][i + 1], NeuronOCL.getOutput(),
                                   NeuronOCL.getGradient(),
                                   (ENUM_ACTIVATION)NeuronOCL.Activation()))
         ReturnFalse;
     }
//--- Forecast 1
   curr = cHeads[2][0];
   CNeuronBaseOCL* prev = cHeads[1][-1];
   if(!DeConcat(cBackbone.getPrevOutput(), prev.getPrevOutput(), curr.getGradient(),
                cBackbone.Neurons(), prev.Neurons(), 1) ||
      !SumAndNormalize(prev.getGradient(), prev.getPrevOutput(), prev.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(prev);
   for(int i = cHeads[1].Total() - 2; i >= 0; i--)
     {
      curr = cHeads[1][i];
      if(!curr ||
         !curr.CalcHiddenGradients(cHeads[1][i + 1], NeuronOCL.getOutput(),
                                   NeuronOCL.getGradient(),
                                   (ENUM_ACTIVATION)NeuronOCL.Activation()))
         ReturnFalse;
     }
//--- Recovery
   prev = cHeads[0][-1];
   Deactivation(prev);
   curr = cHeads[0][-2];
   if(!curr ||
      !curr.CalcHiddenGradients(prev, NeuronOCL.getOutput(),
                                NeuronOCL.getGradient(),
                                (ENUM_ACTIVATION)NeuronOCL.Activation()))
      ReturnFalse;
   CBufferFloat* temp = curr.getGradient();
   float error = 0;
   if(!curr.SetGradient(curr.getPrevOutput(), false) ||
      !curr.calcOutputGradients(NeuronOCL.getOutput(), error) ||
      !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !curr.SetGradient(temp, false))
      ReturnFalse;
   for(int i = cHeads[0].Total() - 3; i >= 0; i--)
     {
      curr = cHeads[0][i];
      if(!curr ||
         !curr.CalcHiddenGradients(cHeads[0][i + 1], NeuronOCL.getOutput(),
                                   NeuronOCL.getGradient(),
                                   (ENUM_ACTIVATION)NeuronOCL.Activation()))
         ReturnFalse;
     }
   if(!cBackbone.CalcHiddenGradients(curr) ||
      !SumAndNormalize(cBackbone.getGradient(), cBackbone.getPrevOutput(), cBackbone.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cBackbone.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !cBackbone)
      ReturnFalse;
//--- Forecast 1
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* prev = cBackbone.AsObject();
   for(int i = 0; i < cHeads[1].Total(); i++)
     {
      curr = cHeads[1][i];
      if(!curr ||
         !curr.UpdateInputWeights(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//--- Forecast 2
   prev = cHeads[2][0];
   for(int i = 1; i < cHeads[2].Total(); i++)
     {
      curr = cHeads[2][i];
      if(!curr ||
         !curr.UpdateInputWeights(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//--- Backbone
   if(!cBackbone.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//--- Recover
   prev = cBackbone.AsObject();
   for(int i = 0; i < cHeads[0].Total(); i++)
     {
      curr = cHeads[0][i];
      if(!curr ||
         !curr.UpdateInputWeights(prev, NeuronOCL.getOutput()))
         ReturnFalse;
      prev = curr;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cBackbone ||
      !cBackbone.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cHeads.Size(); i++)
      if(!cHeads[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cBackbone)
      cBackbone = CreateBackbone();
   if(!cBackbone)
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cBackbone.AsObject()))
      ReturnFalse;
   for(uint i = 0; i < cHeads.Size(); i++)
     {
      if(!cHeads[i].Load(file_handle))
         ReturnFalse;
      for(int n = 0; n < cHeads[i].Total(); n++)
         if(cHeads[i][n].Type() == defNeuronReGENTADHead)
           {
            CNeuronReGENTADHead* head = cHeads[i][n];
            if(!head.SetNorm(cBackbone.GetNorm()))
               ReturnFalse;
           }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronReGENTAD* Source = source;
   if(!Source || !cBackbone || !Source.cBackbone ||
      !cBackbone.WeightsUpdate(Source.cBackbone.AsObject(), tau))
      ReturnFalse;
   for(uint i = 0; i < cHeads.Size(); i++)
      dWeightsUpdate(cHeads[i], Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronReGENTAD::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cBackbone ||
      !cBackbone.Clear())
      ReturnFalse;
   for(uint i = 0; i < cHeads.Size(); i++)
      if(!cHeads[i].ClearStates())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTAD::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!cBackbone)
      cBackbone.SetOpenCL(OpenCL);
   for(uint i = 0; i < cHeads.Size(); i++)
      cHeads[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronReGENTAD::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   if(!!cBackbone)
      cBackbone.TrainMode(bTrain);
   for(uint i = 0; i < cHeads.Size(); i++)
      cHeads[i].TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronGDformerBackbone::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint &dimensions[], uint units_s, uint heads, uint stack_size,
                                   uint layers, uint embed_size, uint candidates, uint topK,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() < 2 || layers < 1 || units_s < 1 || heads < 1 || embed_size < 1 ||
      embed_size % heads != 0)
      ReturnFalse;
   uint iSequenceDim = units_s * dimensions[0];
   uint iDCAlayers = 3;
   uint iContextUnits = dimensions.Size() - 1;
   uint iGDSequenceUnits = 1 + iDCAlayers;
   uint iOutputUnits = 3 * iGDSequenceUnits + iContextUnits + 3;
   uint iContextDim = 0;
   for(uint i = 1; i < dimensions.Size(); i++)
      iContextDim += dimensions[i];
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, iOutputUnits * embed_size, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   cSeqToken.Clear();
   cTransformer.Clear();
   cRecurrent.Clear();
   cSeqToken.SetOpenCL(OpenCL);
   cTransformer.SetOpenCL(OpenCL);
   cRecurrent.SetOpenCL(OpenCL);
//--- Sequence tokenizer
   uint index = 0;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iSequenceDim, optimization, iBatch) ||
      !cSeqToken.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   CNeuronSequenceTokenizer* seq_tok = new CNeuronSequenceTokenizer();
   if(!seq_tok ||
      !seq_tok.Init(0, index, OpenCL, dimensions[0], units_s, embed_size, optimization, iBatch) ||
      !cSeqToken.Add(seq_tok))
      DeleteObjAndFalse(seq_tok);
   seq_tok.SetActivationFunction(SoftPlus);
   CNeuronDictionaryCrossAtt* dca = NULL;
   for(uint i = 0; i < iDCAlayers; i++)
     {
      index++;
      dca = new CNeuronDictionaryCrossAtt();
      if(!dca ||
         !dca.Init(0, index, OpenCL, embed_size, 1, heads, stack_size, optimization, iBatch) ||
         !cSeqToken.Add(dca))
         DeleteObjAndFalse(dca);
     }
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 4 * embed_size, optimization, iBatch) ||
      !cSeqToken.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
//--- Transformer
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iContextDim, optimization, iBatch) ||
      !cTransformer.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, iContextDim + cSeqToken[-1].Neurons(), optimization, iBatch) ||
      !cTransformer.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   uint dims[];
   uint sequence_units = cSeqToken[-1].Neurons() / embed_size;
   if(sequence_units != iGDSequenceUnits ||
      ArrayResize(dims, sequence_units + iContextUnits) < int(sequence_units + iContextUnits))
      ReturnFalse;
   ArrayFill(dims, 0, sequence_units, embed_size);
   ArrayCopy(dims, dimensions, sequence_units, 1);
   index++;
   CNeuronINFNetBlock* transf = new CNeuronINFNetBlock();
   if(!transf ||
      !transf.Init(0, index, OpenCL, dims, sequence_units, heads, stack_size, layers, embed_size, candidates, topK, optimization, iBatch) ||
      !cTransformer.Add(transf))
      DeleteObjAndFalse(transf);
//--- Recurrent
   CNeuronLSTMOCL* lstm = NULL;
   uint size = cSeqToken[-1].Neurons();
   for(uint i = 0; i < layers; i++)
     {
      lstm = new CNeuronLSTMOCL();
      index++;
      if(!lstm ||
         !lstm.Init(0, index, OpenCL, size, optimization, iBatch) ||
         !lstm.SetInputs(size) ||
         !cRecurrent.Add(lstm))
         DeleteObjAndFalse(lstm);
      lstm.SetActivationFunction(TANH);
     }
   index++;
   CNeuronBatchNormOCL* norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, lstm.Neurons(), optimization, iBatch) ||
      !cRecurrent.Add(norm))
      DeleteObjAndFalse(norm);
//---
   index++;
   if(!cDifference.Init(0, index, OpenCL, norm.Neurons(), embed_size, optimization, iBatch))
      ReturnFalse;
   cDifference.SetActivationFunction(TANH);
   if(Neurons() != cTransformer[-1].Neurons() + cSeqToken[-1].Neurons() + cDifference.Neurons())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronGDformerBackbone::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cSeqToken[0];
   CNeuronBaseOCL* context = cTransformer[0];
   if(!sequence || !context)
      ReturnFalse;
   uint iSequenceDim = sequence.Neurons();
   uint iContextDim = context.Neurons();
   if(!DeConcat(sequence.getOutput(), context.getOutput(), NeuronOCL.getOutput(),
                iSequenceDim, iContextDim, 1))
      ReturnFalse;
//--- Sequence tokenizer
   CNeuronBaseOCL *sims[3] = {NULL};
   uint sims_pos = 0;
   for(int i = 1; i < cSeqToken.Total() - 1; i++)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.FeedForward(cSeqToken[i - 1]))
         ReturnFalse;
      if(sequence.Type() == defNeuronDictionaryCrossAtt)
        {
         if(sims_pos >= 3)
            ReturnFalse;
         CNeuronDictionaryCrossAtt* temp = sequence;
         sims[sims_pos] = temp.GetSimilarity();
         sims_pos++;
        }
     }
   iSequenceDim = sequence.Neurons();
   sequence = cSeqToken[-1];
   if(sims_pos != 3 || !sims[0] || !sims[1] || !sims[2])
      ReturnFalse;
   if(!sequence ||
      !Concat(cSeqToken[-2].getOutput(), sims[0].getOutput(), sims[1].getOutput(),
              sims[2].getOutput(), sequence.getOutput(),
              iSequenceDim, iSequenceDim, iSequenceDim, iSequenceDim, 1))
      ReturnFalse;
   iSequenceDim = sequence.Neurons();
//--- Transformer
   context = cTransformer[1];
   if(!context ||
      !Concat(sequence.getOutput(), cTransformer[0].getOutput(), context.getOutput(),
              iSequenceDim, iContextDim, 1))
      ReturnFalse;
   for(int i = 2; i < cTransformer.Total(); i++)
     {
      context = cTransformer[i];
      if(!context ||
         !context.FeedForward(cTransformer[i - 1]))
         ReturnFalse;
     }
//--- Recurrent
   sequence = cRecurrent[0];
   if(!sequence ||
      !sequence.FeedForward(cSeqToken[-1]))
      ReturnFalse;
   for(int i = 1; i < cRecurrent.Total(); i++)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.FeedForward(cRecurrent[i - 1]))
         ReturnFalse;
     }
//---
   if(!cDifference.FeedForward(context, sequence.getOutput()))
      ReturnFalse;
   if(!Concat(context.getOutput(), sequence.getOutput(), cDifference.getOutput(), Output,
              context.Neurons(), sequence.Neurons(), cDifference.Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronGDformerBackbone::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* sequence = cRecurrent[-1];
   CNeuronBaseOCL* context = cTransformer[-1];
   if(!sequence || !context)
      ReturnFalse;
   uint iSequenceDim = sequence.Neurons();
   uint iContextDim = context.Neurons();
   if(!DeConcat(context.getGradient(), sequence.getGradient(), cDifference.getGradient(), Gradient,
                iContextDim, iSequenceDim, cDifference.Neurons(), 1))
      ReturnFalse;
   Deactivation(cDifference);
   Deactivation(context);
   Deactivation(sequence);
   CBufferFloat* temp = context.getGradient();
   if(!context.SetGradient(context.getPrevOutput(), false))
      ReturnFalse;
   if(!context.CalcHiddenGradients(cDifference.AsObject(),
                                   sequence.getOutput(),
                                   sequence.getPrevOutput(),
                                   (ENUM_ACTIVATION)sequence.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(temp, context.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !context.SetGradient(temp, false) ||
      !SumAndNormalize(sequence.getGradient(), sequence.getPrevOutput(), sequence.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Recurrent
   for(int i = cRecurrent.Total() - 2; i >= 0; i--)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.CalcHiddenGradients(cRecurrent[i + 1]))
         ReturnFalse;
     }
//--- Transformer
   for(int i = cTransformer.Total() - 2; i >= 1; i--)
     {
      context = cTransformer[i];
      if(!context ||
         !context.CalcHiddenGradients(cTransformer[i + 1]))
         ReturnFalse;
     }
   context = cTransformer[0];
   sequence = cSeqToken[-1];
   if(!context || !sequence)
      ReturnFalse;
   iSequenceDim = sequence.Neurons();
   iContextDim = context.Neurons();
   if(!DeConcat(sequence.getPrevOutput(), context.getGradient(), cTransformer[1].getGradient(),
                iSequenceDim, iContextDim, 1))
      ReturnFalse;
//--- Sequence tokenizer
   if(!sequence.CalcHiddenGradients(cRecurrent[0]))
      ReturnFalse;
   if(sequence.Activation() != None)
      if(!DeActivation(sequence.getOutput(), sequence.getPrevOutput(),
                       sequence.getPrevOutput(), sequence.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(sequence.getGradient(), sequence.getPrevOutput(), sequence.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   CNeuronBaseOCL* sims[3] = {NULL};
   uint sims_pos = 0;
   for(int i = 1; i < cSeqToken.Total() - 1; i++)
     {
      if(!cSeqToken[i] || cSeqToken[i].Type() != defNeuronDictionaryCrossAtt)
         continue;
      if(sims_pos >= 3)
         ReturnFalse;
      CNeuronDictionaryCrossAtt* temp = cSeqToken[i];
      sims[sims_pos] = temp.GetSimilarity();
      sims_pos++;
     }
   if(sims_pos != 3 || !sims[0] || !sims[1] || !sims[2])
      ReturnFalse;
   iSequenceDim = sims[0].Neurons();
   if(!cSeqToken[-2] ||
      !DeConcat(cSeqToken[-2].getGradient(), sims[0].getGradient(), sims[1].getGradient(),
                sims[2].getGradient(), sequence.getGradient(),
                iSequenceDim, iSequenceDim, iSequenceDim, iSequenceDim, 1))
      ReturnFalse;
   for(uint i = 0; i < sims_pos; i++)
      Deactivation(sims[i]);
   for(int i = cSeqToken.Total() - 3; i >= 0; i--)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.CalcHiddenGradients(cSeqToken[i + 1]))
         ReturnFalse;
     }
   iSequenceDim = sequence.Neurons();
//---
   if(!Concat(sequence.getGradient(), context.getGradient(), NeuronOCL.getGradient(),
              iSequenceDim, iContextDim, 1))
      ReturnFalse;
   Deactivation(NeuronOCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronGDformerBackbone::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//---
   CNeuronBaseOCL* sequence = cSeqToken[0];
   CNeuronBaseOCL* context = cTransformer[0];
//--- Sequence tokenizer
   for(int i = 1; i < cSeqToken.Total() - 1; i++)
     {
      sequence = cSeqToken[i];
      if(!sequence ||
         !sequence.UpdateInputWeights(cSeqToken[i - 1]))
         ReturnFalse;
     }
//--- Transformer
   for(int i = 2; i < cTransformer.Total(); i++)
     {
      context = cTransformer[i];
      if(!context ||
         !context.UpdateInputWeights(cTransformer[i - 1]))
         ReturnFalse;
     }
//--- Recurrent
   sequence = cRecurrent[0];
   if(!sequence ||
      !sequence.UpdateInputWeights(cSeqToken[-1]))
      ReturnFalse;
   for(int i = 1; i < cRecurrent.Total(); i++)
     {
      sequence = cRecurrent[i];
      if(!sequence ||
         !sequence.UpdateInputWeights(cRecurrent[i - 1]))
         ReturnFalse;
     }
//---
   if(!cDifference.UpdateInputWeights(context, sequence.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint latent_dim, uint candidates, uint decisions,
                           uint topK, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(latent_dim == 0 || candidates == 0 || decisions == 0 ||
      topK == 0 || topK > candidates)
      ReturnFalse;
//---
   iCandidates = candidates;
   iDecisions = decisions;
   iDimension = latent_dim;
   iTopK = topK;
//---
   if(!CNeuronBaseOCL::Init(0, myIndex, open_cl, latent_dim, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   if(!cMeanDistance.Init(0, 0, OpenCL, iCandidates, optimization, iBatch))
      ReturnFalse;
   cMeanDistance.SetActivationFunction(None);
   if(!cScores.Init(0, 1, OpenCL, 1, iCandidates, iTopK, optimization, iBatch))
      ReturnFalse;
   cScores.SetActivationFunction(None);
//---
   int distance_total = int(iCandidates * iDecisions);
   int top_total = int(distance_total * iTopK);
   int candidates_total = int(iCandidates * iDimension);
   if(!cDistance.Init(0, 2, OpenCL, distance_total, optimization, iBatch) ||
      !cCandidatesGradient.BufferInit(candidates_total, 0))
      ReturnFalse;
   cDistance.SetActivationFunction(None);
   if(!cCandidatesGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   CBufferFloat* dist_index = cDistance.getPrevOutput();
   dist_index.BufferFree();
   if(!dist_index.BufferInit(top_total, 0) ||
      !dist_index.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::MeanScores(void)
  {
   if(!OpenCL || cMeanDistance.getOutputIndex() < 0)
      ReturnFalse;
   if(cDistance.Neurons() != int(iCandidates * iDecisions) ||
      cMeanDistance.Neurons() != int(iCandidates))
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iCandidates};
   const int kernel = def_k_MomADTTMMeanScores;
   ResetLastError();
   setBuffer(kernel, def_k_madttm_ms_distance, cDistance.getOutputIndex())
   setBuffer(kernel, def_k_madttm_ms_output, cMeanDistance.getOutputIndex())
   setArgument(kernel, def_k_madttm_ms_decisions, int(iDecisions))
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cMeanDistance.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::MeanScoresGradient(void)
  {
   if(!OpenCL || cMeanDistance.getGradientIndex() < 0)
      ReturnFalse;
   if(cMeanDistance.Neurons() != int(iCandidates) ||
      cDistance.Neurons() != int(iCandidates * iDecisions))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iCandidates, iDecisions};
   const int kernel = def_k_MomADTTMMeanScoresGrad;
   ResetLastError();
   setBuffer(kernel, def_k_madttm_msg_scores_gr, cMeanDistance.getGradientIndex())
   setBuffer(kernel, def_k_madttm_msg_distance_gr, cDistance.getGradientIndex())
   setArgument(kernel, def_k_madttm_msg_decisions, int(iDecisions))
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cDistance.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() != int(iCandidates * iDimension) ||
      Neurons() != int(iDimension) ||
      SecondInput.Total() != int(iDecisions * iDimension))
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!SparseSoftmaxHausdorff(NeuronOCL.getOutput(), SecondInput, cDistance.getOutput(),
                              cDistance.getPrevOutput(), int(iDimension), int(iTopK)))
      ReturnFalse;
   if(!MeanScores())
      ReturnFalse;
   if(!cScores.FeedForward(cMeanDistance.AsObject()))
      ReturnFalse;
   if(!SparseMatMul(cScores.GetIndexes(), cScores.getOutput(), NeuronOCL.getOutput(),
                    Output, 1, int(iTopK), int(iCandidates), int(iDimension)))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() != int(iCandidates * iDimension) ||
      Gradient.Total() != int(iDimension) ||
      SecondInput.Total() != int(iDecisions * iDimension))
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
   if(!!SecondGradient)
     {
      if(SecondGradient.Total() != int(iDecisions * iDimension))
         ReturnFalse;
      if(SecondGradient.GetIndex() < 0 && !SecondGradient.BufferCreate(OpenCL))
         ReturnFalse;
     }
//--- weighted latent output branch
   if(!cCandidatesGradient.Fill(0) ||
      !cScores.getGradient().Fill(0))
      ReturnFalse;
   if(!SparseMatMulGrad(cScores.GetIndexes(), cScores.getOutput(), cScores.getGradient(),
                        NeuronOCL.getOutput(), GetPointer(cCandidatesGradient), Gradient,
                        1, int(iTopK), int(iCandidates), int(iDimension)))
      ReturnFalse;
//--- sparse candidate-score branch
   if(!cMeanDistance.CalcHiddenGradients(cScores.AsObject()))
      ReturnFalse;
   if(!MeanScoresGradient())
      ReturnFalse;
//--- Hausdorff branch
   CBufferFloat *second_grad = SecondGradient;
   CBufferFloat tempSecondGradient;
   if(!second_grad)
     {
      if(!tempSecondGradient.BufferInit(int(iDecisions * iDimension), 0) ||
         !tempSecondGradient.BufferCreate(OpenCL))
         ReturnFalse;
      second_grad = GetPointer(tempSecondGradient);
     }
   bool result = SparseSoftmaxHausdorffGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                 SecondInput, second_grad, cDistance.getGradient(),
                 cDistance.getPrevOutput(), int(iDimension), int(iTopK));
   if(!SecondGradient)
      tempSecondGradient.BufferFree();
   if(!result)
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), GetPointer(cCandidatesGradient),
                       NeuronOCL.getGradient(), int(iDimension), false, 0, 0, 0, 1.0f))
      ReturnFalse;
   Deactivation(NeuronOCL);
   if(!!SecondGradient && SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iCandidates)
   dFileWriteUInt(file_handle, iDecisions)
   dFileWriteUInt(file_handle, iDimension)
   dFileWriteUInt(file_handle, iTopK)
   if(!cMeanDistance.Save(file_handle) ||
      !cScores.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   dFileReadUInt(file_handle, iCandidates)
   dFileReadUInt(file_handle, iDecisions)
   dFileReadUInt(file_handle, iDimension)
   dFileReadUInt(file_handle, iTopK)
   if(iDimension == 0 || iDimension > 64 || iCandidates == 0 || iDecisions == 0 ||
      iTopK == 0 || iTopK > iCandidates || iTopK > iDimension)
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMeanDistance.AsObject()) ||
      !LoadInsideLayer(file_handle, cScores.AsObject()))
      ReturnFalse;
   if(Neurons() != int(iDimension) || cMeanDistance.Neurons() != int(iCandidates) ||
      cScores.Neurons() != int(iTopK))
      ReturnFalse;
//---
   int distance_total = int(iCandidates * iDecisions);
   int top_total = int(distance_total * iTopK);
   int candidates_total = int(iCandidates * iDimension);
   if(!cDistance.Init(0, 2, OpenCL, distance_total, optimization, iBatch) ||
      !cCandidatesGradient.BufferInit(candidates_total, 0))
      ReturnFalse;
   cDistance.SetActivationFunction(None);
   if(!cCandidatesGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   CBufferFloat* dist_index = cDistance.getPrevOutput();
   dist_index.BufferFree();
   if(!dist_index.BufferInit(top_total, 0) ||
      !dist_index.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomADTTM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cMeanDistance.SetOpenCL(OpenCL);
   cScores.SetOpenCL(OpenCL);
   if(cDistance.Neurons() > 0)
      cDistance.SetOpenCL(OpenCL);
   if(cCandidatesGradient.Total() > 0)
      cCandidatesGradient.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomADTTM::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cMeanDistance.TrainMode(bTrain);
   cScores.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADTTM::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cMeanDistance.Clear() ||
      !cScores.Clear())
      ReturnFalse;
   if(!cDistance.Clear())
      ReturnFalse;
   if(cCandidatesGradient.Total() > 0 && !cCandidatesGradient.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint latent_dim, uint candidates, uint market_dim,
                           uint market_units, uint stack_size, uint topK,
                           uint heads, uint key_dimension,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(latent_dim == 0 || candidates == 0 || market_dim == 0 || market_units == 0 ||
      stack_size == 0 || topK == 0 || topK > candidates || heads == 0 ||
      key_dimension == 0)
      ReturnFalse;
//--- final MPI refine: Q_hist attends to market state
   if(!CNeuronCrossAttention::Init(numOutputs, myIndex, open_cl, latent_dim,
                                   key_dimension, heads, 1, market_dim,
                                   market_units, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//--- history stack and internal MPI components
   if(!cStack.Init(0, 0, OpenCL, stack_size, latent_dim, 1, optimization, iBatch))
      ReturnFalse;
   if(!cTTM.Init(0, 1, OpenCL, latent_dim, candidates, stack_size, topK,
                 optimization, iBatch))
      ReturnFalse;
   if(!cPolicyMarket.Init(0, 2, OpenCL, latent_dim, key_dimension, heads,
                          stack_size, market_dim, market_units, optimization, iBatch))
      ReturnFalse;
   if(!cTTMHistory.Init(0, 3, OpenCL, latent_dim, key_dimension, heads,
                        1, latent_dim, stack_size, optimization, iBatch))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(SecondInput.Total() != int(iWindow_K * iUnits_K))
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//--- push the previous MPI scenario before overwriting Output
   if(!cStack.FeedForward(GetPointer(this)))
      ReturnFalse;
//--- TTM builds the weighted scenario from the strategy pool and recent scenarios
   if(!cTTM.FeedForward(NeuronOCL, cStack.getOutput()))
      ReturnFalse;
//--- D_cur = actions stack attends to market
   if(!cPolicyMarket.FeedForward(cStack.AsObject(), SecondInput))
      ReturnFalse;
//--- Q_hist = TTM scenario attends to the current action-aware history
   if(!cTTMHistory.FeedForward(cTTM.AsObject(), cPolicyMarket.getOutput()))
      ReturnFalse;
//--- parent cross-attention is the final MPI market refine block
   if(!CNeuronCrossAttention::feedForward(cTTMHistory.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(SecondInput.Total() != int(iWindow_K * iUnits_K))
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//--- market gradient target for the final parent refine branch
   CBufferFloat tempMarketGradient;
   CBufferFloat *market_gradient = SecondGradient;
   if(!!market_gradient)
     {
      if(market_gradient.Total() != SecondInput.Total())
         ReturnFalse;
      if(market_gradient.GetIndex() < 0 && !market_gradient.BufferCreate(OpenCL))
         ReturnFalse;
      if(!market_gradient.Fill(0))
         ReturnFalse;
     }
   else
     {
      if(!tempMarketGradient.BufferInit(SecondInput.Total(), 0) ||
         !tempMarketGradient.BufferCreate(OpenCL))
         ReturnFalse;
      market_gradient = GetPointer(tempMarketGradient);
     }
//--- final MPI refine gradients: Q_hist and market context
   if(!CNeuronCrossAttention::calcInputGradients(cTTMHistory.AsObject(),
         SecondInput, market_gradient, None))
      ReturnFalse;
//--- TTM-history gradients: Q_ttm and D_cur
   if(!cTTM.CalcHiddenGradients(cTTMHistory.AsObject(), cPolicyMarket.getOutput(),
                                cPolicyMarket.getGradient(), None))
      ReturnFalse;
//--- TTM gradients to the strategy pool. The stack branch is intentionally stop-gradient.
   if(!NeuronOCL.CalcHiddenGradients(cTTM.AsObject(), cStack.getOutput(), NULL, None))
      ReturnFalse;
//--- policy-market gradients are needed for its weights and optional market input gradient
   CBufferFloat policyMarketGradient;
   if(!policyMarketGradient.BufferInit(SecondInput.Total(), 0) ||
      !policyMarketGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cStack.CalcHiddenGradients(cPolicyMarket.AsObject(), SecondInput,
                                  GetPointer(policyMarketGradient), None))
      ReturnFalse;
   if(cStack.getGradient().Total() > 0 && !cStack.getGradient().Fill(0))
      ReturnFalse;
//--- expose summed market gradient only when the caller provided a target buffer
   if(!!SecondGradient)
     {
      if(!SumAndNormalize(market_gradient, GetPointer(policyMarketGradient),
                          market_gradient, int(iWindow_K), false, 0, 0, 0, 1.0f))
         ReturnFalse;
      if(SecondActivation != None)
         if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
            ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      ReturnFalse;
   if(SecondInput.Total() != int(iWindow_K * iUnits_K))
      ReturnFalse;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!CNeuronCrossAttention::updateInputWeights(cTTMHistory.AsObject(), SecondInput))
      ReturnFalse;
   if(!cTTMHistory.UpdateInputWeights(cTTM.AsObject(), cPolicyMarket.getOutput()))
      ReturnFalse;
   if(!cPolicyMarket.UpdateInputWeights(cStack.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::Save(const int file_handle)
  {
   if(!CNeuronCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cStack.Save(file_handle) ||
      !cTTM.Save(file_handle) ||
      !cPolicyMarket.Save(file_handle) ||
      !cTTMHistory.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::Load(const int file_handle)
  {
   if(!CNeuronCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStack.AsObject()) ||
      !LoadInsideLayer(file_handle, cTTM.AsObject()) ||
      !LoadInsideLayer(file_handle, cPolicyMarket.AsObject()) ||
      !LoadInsideLayer(file_handle, cTTMHistory.AsObject()))
      ReturnFalse;
   if(cStack.GetDimension() != iWindow ||
      cTTM.Neurons() != int(iWindow) ||
      cPolicyMarket.Neurons() != int(iWindow * cStack.GetStackSize()) ||
      cTTMHistory.Neurons() != int(iWindow))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomADMPI::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronCrossAttention::SetOpenCL(obj);
   cStack.SetOpenCL(OpenCL);
   cTTM.SetOpenCL(OpenCL);
   cPolicyMarket.SetOpenCL(OpenCL);
   cTTMHistory.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomADMPI::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   Q_Embedding.TrainMode(bTrain);
   KV_Embedding.TrainMode(bTrain);
   Transpose.TrainMode(bTrain);
   MHAttentionOut.TrainMode(bTrain);
   W0.TrainMode(bTrain);
   AttentionOut.TrainMode(bTrain);
   FF[0].TrainMode(bTrain);
   FF[1].TrainMode(bTrain);
   if(!!cContext)
      cContext.TrainMode(bTrain);
   cStack.TrainMode(bTrain);
   cTTM.TrainMode(bTrain);
   cPolicyMarket.TrainMode(bTrain);
   cTTMHistory.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronMomADMPI *src = source;
   if(!CNeuronMH2AttentionOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cTTM.WeightsUpdate(GetPointer(src.cTTM), tau))
      ReturnFalse;
   if(!cPolicyMarket.WeightsUpdate(GetPointer(src.cPolicyMarket), tau))
      ReturnFalse;
   if(!cTTMHistory.WeightsUpdate(GetPointer(src.cTTMHistory), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomADMPI::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!Q_Embedding.Clear() ||
      !KV_Embedding.Clear() ||
      !Transpose.Clear() ||
      !MHAttentionOut.Clear() ||
      !W0.Clear() ||
      !AttentionOut.Clear() ||
      !FF[0].Clear() ||
      !FF[1].Clear())
      ReturnFalse;
   if(!!cContext && !cContext.Clear())
      ReturnFalse;
   if(!cStack.Clear() ||
      !cTTM.Clear() ||
      !cPolicyMarket.Clear() ||
      !cTTMHistory.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MSD::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint dimension, uint units, uint variables, uint quantiles,
                            float range, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(units < (10 * quantiles))
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, dimension * (2 * quantiles + 1)*variables, optimization_type, batch))
      ReturnFalse;
//---
   iDimension = dimension;
   iUnits = units;
   iVariables = variables;
   iQuantiles = quantiles;
   fQuantSize = range / iQuantiles;
   if(fQuantSize < FLT_EPSILON)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MSD::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
   if(NeuronOCL.Neurons() < int(iDimension * iUnits * iVariables))
      ReturnFalse;
//---
   uint global_work_offset[] = { 0, 0, 0 };
   uint global_work_size[] = { iDimension,
                               (uint)MathMin(iUnits, OpenCL.GetMaxLocalSize(1)),
                               iVariables
                             };
   uint local_work_size[] = { 1, global_work_size[1], 1 };
//---
   const int kernel = def_k_MarketStateDensity;
   setBuffer(kernel, def_k_msd_sequence, NeuronOCL.getOutputIndex());
   setBuffer(kernel, def_k_msd_distribution, getOutputIndex());
   setArgument(kernel, def_k_msd_sequence_size, iUnits);
   setArgument(kernel, def_k_msd_quantiles, iQuantiles);
   setArgument(kernel, def_k_msd_quant_step, fQuantSize);
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size);
#ifdef _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MSD::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iDimension);
   dFileWriteUInt(file_handle, iUnits);
   dFileWriteUInt(file_handle, iVariables);
   dFileWriteUInt(file_handle, iQuantiles);
   dFileWriteFloat(file_handle, fQuantSize);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MSD::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   dFileReadUInt(file_handle, iDimension);
   dFileReadUInt(file_handle, iUnits);
   dFileReadUInt(file_handle, iVariables);
   dFileReadUInt(file_handle, iQuantiles);
   dFileReadUFloat(file_handle, fQuantSize);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint window, uint variables, uint state_stack_size,
                                uint quant_stack_size, uint quantiles, float range,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(window == 0 || variables == 0 || state_stack_size < (10 * quantiles) ||
      quant_stack_size < 2 || quantiles == 0 || range <= 0)
      ReturnFalse;
   uint bins = 2 * quantiles + 1;
   uint quant_dimension = window * bins;
   uint quant_features = quant_dimension * variables;
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl,
                                   quant_stack_size, quant_stack_size, 1,
                                   1, quant_features, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = window;
   iVariables = variables;
   iStateStackSize = state_stack_size;
   iQuantStackSize = quant_stack_size;
   iQuantiles = quantiles;
   fRange = range;
//---
   if(!cNorm.Init(0, 0, OpenCL, window * variables, iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   if(!cStateStack.Init(0, 1, OpenCL, state_stack_size, window, variables, optimization, iBatch))
      ReturnFalse;
   if(!cMSD.Init(0, 2, OpenCL, window, state_stack_size, variables, quantiles, range, optimization, iBatch))
      ReturnFalse;
   if(!cAffine.Init(0, 3, OpenCL, window, window, 2, 1, variables, optimization, iBatch))
      ReturnFalse;
   cAffine.SetActivationFunction(None);
   if(!cQuantStack.Init(0, 4, OpenCL, quant_stack_size, 1, quant_features, optimization, iBatch))
      ReturnFalse;
   if(!cPrevQuantStack.Init(0, 5, OpenCL, quant_stack_size * quant_features, optimization, iBatch))
      ReturnFalse;
   cPrevQuantStack.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::AffineStack(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint quant_dimension = iWindow * (2 * iQuantiles + 1);
   uint quant_features = quant_dimension * iVariables;
   if(cQuantStack.Neurons() != int(iQuantStackSize * quant_features) ||
      cPrevQuantStack.Neurons() != cQuantStack.Neurons() ||
      cAffine.Neurons() != int(2 * iVariables))
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {quant_features, iQuantStackSize};
   uint kernel = def_k_CogDriverAffineStack;
   setBuffer(kernel, def_k_cdas_stack, cQuantStack.getOutputIndex())
   setBuffer(kernel, def_k_cdas_snapshot, cPrevQuantStack.getOutputIndex())
   setBuffer(kernel, def_k_cdas_affine, cAffine.getOutputIndex())
   setArgument(kernel, def_k_cdas_stack_size, (int)iQuantStackSize)
   setArgument(kernel, def_k_cdas_quant_dimension, (int)quant_dimension)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cQuantStack.getOutput().BufferRead() || !cPrevQuantStack.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::AffineStackGradient(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint quant_dimension = iWindow * (2 * iQuantiles + 1);
   uint kernel = def_k_CogDriverAffineStackGrad;
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {iVariables, 2};
   setBuffer(kernel, def_k_cdasg_snapshot, cPrevQuantStack.getOutputIndex())
   setBuffer(kernel, def_k_cdasg_stack_gr, cQuantStack.getGradientIndex())
   setBuffer(kernel, def_k_cdasg_affine_gr, cAffine.getGradientIndex())
   setArgument(kernel, def_k_cdasg_stack_size, (int)iQuantStackSize)
   setArgument(kernel, def_k_cdasg_quant_dimension, (int)quant_dimension)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cAffine.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < int(iWindow * iVariables))
      ReturnFalse;
   if(!cNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cStateStack.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!cMSD.FeedForward(cStateStack.AsObject()))
      ReturnFalse;
   if(!cAffine.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!AffineStack())
      ReturnFalse;
   if(!cQuantStack.FeedForward(cMSD.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cQuantStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::calcInputGradients(cQuantStack.AsObject()))
      ReturnFalse;
   if(!AffineStackGradient())
      ReturnFalse;
   if(!NeuronOCL.getGradient().Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cQuantStack.AsObject()))
      ReturnFalse;
   if(!cAffine.UpdateInputWeights(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iWindow);
   dFileWriteUInt(file_handle, iVariables);
   dFileWriteUInt(file_handle, iStateStackSize);
   dFileWriteUInt(file_handle, iQuantStackSize);
   dFileWriteUInt(file_handle, iQuantiles);
   dFileWriteFloat(file_handle, fRange);
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(!cStateStack.Save(file_handle))
      ReturnFalse;
   if(!cMSD.Save(file_handle))
      ReturnFalse;
   if(!cAffine.Save(file_handle))
      ReturnFalse;
   if(!cQuantStack.Save(file_handle))
      ReturnFalse;
   if(!cPrevQuantStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   dFileReadUInt(file_handle, iWindow);
   dFileReadUInt(file_handle, iVariables);
   dFileReadUInt(file_handle, iStateStackSize);
   dFileReadUInt(file_handle, iQuantStackSize);
   dFileReadUInt(file_handle, iQuantiles);
   dFileReadUFloat(file_handle, fRange);
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStateStack.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMSD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAffine.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cQuantStack.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPrevQuantStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverData::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cNorm.SetOpenCL(OpenCL);
   cStateStack.SetOpenCL(OpenCL);
   cMSD.SetOpenCL(OpenCL);
   cAffine.SetOpenCL(OpenCL);
   cQuantStack.SetOpenCL(OpenCL);
   cPrevQuantStack.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverData::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cNorm.TrainMode(bTrain);
   cStateStack.TrainMode(bTrain);
   cMSD.TrainMode(bTrain);
   cAffine.TrainMode(bTrain);
   cQuantStack.TrainMode(bTrain);
   cPrevQuantStack.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronCogDriverData* Source = source;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
   if(!cStateStack.WeightsUpdate(Source.cStateStack.AsObject(), tau))
      ReturnFalse;
   if(!cMSD.WeightsUpdate(Source.cMSD.AsObject(), tau))
      ReturnFalse;
   if(!cAffine.WeightsUpdate(Source.cAffine.AsObject(), tau))
      ReturnFalse;
   if(!cQuantStack.WeightsUpdate(Source.cQuantStack.AsObject(), tau))
      ReturnFalse;
   if(!cPrevQuantStack.WeightsUpdate(Source.cPrevQuantStack.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cNorm.Clear())
      ReturnFalse;
   if(!cStateStack.Clear())
      ReturnFalse;
   if(!cMSD.Clear())
      ReturnFalse;
   if(!cAffine.Clear())
      ReturnFalse;
   if(!cQuantStack.Clear())
      ReturnFalse;
   if(!cPrevQuantStack.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::InitStateBuffers(void)
  {
   uint memory_total = iMemoryLen * iVariables * iQueryDim;
   uint kv_total = 2 * iMemoryLen * iTotalQueryDim;
   uint self_kv_total = 2 * iVariables * iTotalQueryDim;
   if(memory_total == 0 || kv_total == 0 || self_kv_total == 0 || iTotalHeads == 0)
      ReturnFalse;
//---
   if(!cMemoryQuery.BufferInit(memory_total, 0) || !cMemoryQuery.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryAge.BufferInit(iMemoryLen * iVariables, 0) || !cMemoryAge.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryScore.BufferInit(iMemoryLen * iVariables, 0) || !cMemoryScore.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryValid.BufferInit(iMemoryLen * iVariables, 0) || !cMemoryValid.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryKV.BufferInit(kv_total, 0) || !cMemoryKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryKVGradient.BufferInit(kv_total, 0) || !cMemoryKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfKV.BufferInit(self_kv_total, 0) || !cSelfKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfKVGradient.BufferInit(self_kv_total, 0) || !cSelfKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLogSumExp.BufferInit(iTotalHeads, 0) || !cLogSumExp.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfLogSumExp.BufferInit(iTotalHeads, 0) || !cSelfLogSumExp.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cQueryGateGradient.BufferInit(iTotalQueryDim, 0) || !cQueryGateGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGateQueryGradient.BufferInit(iTotalQueryDim, 0) || !cGateQueryGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGateContextGradient.BufferInit(iTotalQueryDim, 0) || !cGateContextGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint input_dim, uint query_dim, uint variables, uint memory_len, uint heads,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(input_dim == 0 || query_dim == 0 || variables == 0 || memory_len == 0 || heads == 0 ||
      query_dim % heads != 0)
      ReturnFalse;
   uint total_query = variables * query_dim;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, total_query, optimization_type, batch))
      ReturnFalse;
//---
   iInputDim = input_dim;
   iQueryDim = query_dim;
   iVariables = variables;
   iHeads = heads;
   iHeadDim = query_dim / heads;
   iTotalQueryDim = total_query;
   iTotalHeads = variables * heads;
   iMemoryLen = memory_len;
   iMemoryCount = 0;
   bLastQueryValid = false;
//---
   if(!cQueryProjection.Init(0, 0, OpenCL, input_dim, input_dim, query_dim, variables, 1,
                             optimization, iBatch))
      ReturnFalse;
   cQueryProjection.SetActivationFunction(None);
   if(!cGateInput.Init(total_query, 1, OpenCL, 2 * total_query, optimization, iBatch))
      ReturnFalse;
   cGateInput.SetActivationFunction(None);
   if(!cGateProjection.Init(0, 2, OpenCL, total_query, optimization, iBatch))
      ReturnFalse;
   cGateProjection.SetActivationFunction(SIGMOID);
   if(!cMemoryContext.Init(0, 3, OpenCL, total_query, optimization, iBatch))
      ReturnFalse;
   cMemoryContext.SetActivationFunction(None);
   if(!cSelfContext.Init(0, 4, OpenCL, total_query, optimization, iBatch))
      ReturnFalse;
   cSelfContext.SetActivationFunction(None);
   if(!InitStateBuffers())
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::InsertPendingQuery(void)
  {
   if(!bLastQueryValid)
      return true;
   if(!OpenCL)
      ReturnFalse;
   CBufferFloat *pending = cQueryProjection.getOutput();
   if(!pending)
      ReturnFalse;
//--- insert the previous query independently for each variable
   uint kernel = def_k_CogDriverRankTCMInsertPending;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iVariables, iQueryDim};
   uint local_work_size[2] = {1, iQueryDim};
   setBuffer(kernel, def_k_cdrtcm_insert_memory, cMemoryQuery.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_insert_age, cMemoryAge.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_insert_score, cMemoryScore.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_insert_valid, cMemoryValid.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_insert_pending_query, pending.GetIndex())
   setArgument(kernel, def_k_cdrtcm_insert_memory_len, (int)iMemoryLen)
   setArgument(kernel, def_k_cdrtcm_insert_query_dim, (int)iQueryDim)
   setArgument(kernel, def_k_cdrtcm_insert_variables, (int)iVariables)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
   iMemoryCount = (uint)MathMin(iMemoryLen, iMemoryCount + 1);
#ifdef _DEBUG
   if(!cMemoryQuery.BufferRead() || !cMemoryAge.BufferRead() ||
      !cMemoryScore.BufferRead() || !cMemoryValid.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::BuildMemoryKV(void)
  {
   if(!OpenCL || iMemoryCount == 0)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverRankTCMBuildKV;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iMemoryCount, iVariables, iQueryDim};
   setBuffer(kernel, def_k_cdrtcm_kv_memory, cMemoryQuery.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_kv_age, cMemoryAge.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_kv_valid, cMemoryValid.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_kv_out, cMemoryKV.GetIndex())
   setArgument(kernel, def_k_cdrtcm_kv_memory_count, (int)iMemoryCount)
   setArgument(kernel, def_k_cdrtcm_kv_variables, (int)iVariables)
   setArgument(kernel, def_k_cdrtcm_kv_heads, (int)iHeads)
   setArgument(kernel, def_k_cdrtcm_kv_head_dim, (int)iHeadDim)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cMemoryKV.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::FlashAttentionForward(CBufferFloat *query, CBufferFloat *out)
  {
   if(!OpenCL || !query || !out || iMemoryCount == 0)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = {1,
                              (uint)MathMin(MathMax(iMemoryCount, iHeadDim), OpenCL.GetMaxLocalSize(1)),
                              iTotalHeads
                             };
   uint local_work_size[] = {1, global_work_size[1], 1};
   uint kernel = def_k_MHFlashAttention;
   setBuffer(kernel, def_k_mhflat_q, query.GetIndex())
   setBuffer(kernel, def_k_mhflat_kv, cMemoryKV.GetIndex())
   setBuffer(kernel, def_k_mhflat_logsumexp, cLogSumExp.GetIndex())
   setBuffer(kernel, def_k_mhflat_out, out.GetIndex())
   setArgument(kernel, def_k_mhflat_dimension, (int)iHeadDim)
   setArgument(kernel, def_k_mhflat_total_kv, (int)iMemoryCount)
   setArgument(kernel, def_k_mhflat_mask_future, 0)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::FlashAttentionBackward(CBufferFloat *query, CBufferFloat *query_gr,
      CBufferFloat *out, CBufferFloat *out_gr)
  {
   if(!OpenCL || !query || !query_gr || !out || !out_gr || iMemoryCount == 0)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = {MathMax((int)1, (int)iMemoryCount), iHeadDim, iTotalHeads};
   uint local_work_size[] = {1, global_work_size[1], 1};
   uint kernel = def_k_MHFlashAttentionGrad;
   setBuffer(kernel, def_k_mhflatg_q, query.GetIndex())
   setBuffer(kernel, def_k_mhflatg_q_gr, query_gr.GetIndex())
   setBuffer(kernel, def_k_mhflatg_kv, cMemoryKV.GetIndex())
   setBuffer(kernel, def_k_mhflatg_kv_gr, cMemoryKVGradient.GetIndex())
   setBuffer(kernel, def_k_mhflatg_logsumexp, cLogSumExp.GetIndex())
   setBuffer(kernel, def_k_mhflatg_out, out.GetIndex())
   setBuffer(kernel, def_k_mhflatg_out_gr, out_gr.GetIndex())
   setArgument(kernel, def_k_mhflatg_dimension, (int)iHeadDim)
   setArgument(kernel, def_k_mhflatg_total_q, 1)
   setArgument(kernel, def_k_mhflatg_total_kv, (int)iMemoryCount)
   setArgument(kernel, def_k_mhflatg_mask_future, 0)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!query_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::BuildSelfKV(CBufferFloat *source)
  {
   if(!OpenCL || !source)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverRankTCMSelfKV;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iVariables, iTotalQueryDim};
   setBuffer(kernel, def_k_cdrtcm_selfkv_source, source.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_selfkv_out, cSelfKV.GetIndex())
   setArgument(kernel, def_k_cdrtcm_selfkv_query_dim, (int)iQueryDim)
   setArgument(kernel, def_k_cdrtcm_selfkv_variables, (int)iVariables)
   setArgument(kernel, def_k_cdrtcm_selfkv_heads, (int)iHeads)
   setArgument(kernel, def_k_cdrtcm_selfkv_head_dim, (int)iHeadDim)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cSelfKV.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::SelfAttentionForward(CBufferFloat *query, CBufferFloat *out)
  {
   if(!OpenCL || !query || !out || iVariables == 0)
      ReturnFalse;
   if(!BuildSelfKV(query))
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = {1,
                              (uint)MathMin(MathMax(iVariables, iHeadDim), OpenCL.GetMaxLocalSize(1)),
                              iTotalHeads
                             };
   uint local_work_size[] = {1, global_work_size[1], 1};
   uint kernel = def_k_MHFlashAttention;
   setBuffer(kernel, def_k_mhflat_q, query.GetIndex())
   setBuffer(kernel, def_k_mhflat_kv, cSelfKV.GetIndex())
   setBuffer(kernel, def_k_mhflat_logsumexp, cSelfLogSumExp.GetIndex())
   setBuffer(kernel, def_k_mhflat_out, out.GetIndex())
   setArgument(kernel, def_k_mhflat_dimension, (int)iHeadDim)
   setArgument(kernel, def_k_mhflat_total_kv, (int)iVariables)
   setArgument(kernel, def_k_mhflat_mask_future, 0)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::SelfAttentionBackward(CBufferFloat *query, CBufferFloat *query_gr,
      CBufferFloat *out, CBufferFloat *out_gr)
  {
   if(!OpenCL || !query || !query_gr || !out || !out_gr || iVariables == 0)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = {MathMax((int)1, (int)iVariables), iHeadDim, iTotalHeads};
   uint local_work_size[] = {1, global_work_size[1], 1};
   uint kernel = def_k_MHFlashAttentionGrad;
   setBuffer(kernel, def_k_mhflatg_q, query.GetIndex())
   setBuffer(kernel, def_k_mhflatg_q_gr, query_gr.GetIndex())
   setBuffer(kernel, def_k_mhflatg_kv, cSelfKV.GetIndex())
   setBuffer(kernel, def_k_mhflatg_kv_gr, cSelfKVGradient.GetIndex())
   setBuffer(kernel, def_k_mhflatg_logsumexp, cSelfLogSumExp.GetIndex())
   setBuffer(kernel, def_k_mhflatg_out, out.GetIndex())
   setBuffer(kernel, def_k_mhflatg_out_gr, out_gr.GetIndex())
   setArgument(kernel, def_k_mhflatg_dimension, (int)iHeadDim)
   setArgument(kernel, def_k_mhflatg_total_q, 1)
   setArgument(kernel, def_k_mhflatg_total_kv, (int)iVariables)
   setArgument(kernel, def_k_mhflatg_mask_future, 0)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!query_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::MeanFlashScores(CBufferFloat *query)
  {
   if(!OpenCL || !query || iMemoryCount == 0)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverRankTCMMeanFlashScore;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iMemoryCount, iVariables};
   setBuffer(kernel, def_k_cdrtcm_score_query, query.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_score_kv, cMemoryKV.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_score_valid, cMemoryValid.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_score_out, cMemoryScore.GetIndex())
   setBuffer(kernel, def_k_cdrtcm_score_logsumexp, cLogSumExp.GetIndex())
   setArgument(kernel, def_k_cdrtcm_score_memory_count, (int)iMemoryCount)
   setArgument(kernel, def_k_cdrtcm_score_variables, (int)iVariables)
   setArgument(kernel, def_k_cdrtcm_score_heads, (int)iHeads)
   setArgument(kernel, def_k_cdrtcm_score_head_dim, (int)iHeadDim)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cMemoryScore.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < int(iInputDim * iVariables))
      ReturnFalse;
//--- Step 1: deferred memory mutation from the previous pass
   if(!InsertPendingQuery())
      ReturnFalse;
//--- Step 2: current query
   if(!cQueryProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   CBufferFloat *query = cQueryProjection.getOutput();
   if(!query)
      ReturnFalse;
//--- Step 3-5: attention, score, and fusion
   if(iMemoryCount == 0)
     {
      if(!SelfAttentionForward(query, Output))
         ReturnFalse;
      bLastQueryValid = true;
      return true;
     }
   if(!BuildMemoryKV())
      ReturnFalse;
   if(!FlashAttentionForward(query, cMemoryContext.getOutput()))
      ReturnFalse;
   if(!SelfAttentionForward(cMemoryContext.getOutput(), cSelfContext.getOutput()))
      ReturnFalse;
   if(!MeanFlashScores(query))
      ReturnFalse;
   if(!Concat(query, cSelfContext.getOutput(), cGateInput.getOutput(),
              (int)iTotalQueryDim, (int)iTotalQueryDim, 1))
      ReturnFalse;
   if(!cGateProjection.FeedForward(cGateInput.AsObject()))
      ReturnFalse;
   if(!GateElementMult(cSelfContext.getOutput(), query, cGateProjection.getOutput(), Output))
      ReturnFalse;
//--- cQueryProjection output is the pending query for the next forward pass.
   bLastQueryValid = true;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   CBufferFloat *query = cQueryProjection.getOutput();
   CBufferFloat *query_gr = cQueryProjection.getGradient();
   if(!query || !query_gr)
      ReturnFalse;
   if(!query_gr.Fill(0))
      ReturnFalse;
//---
   if(iMemoryCount == 0)
     {
      if(!cSelfKVGradient.Fill(0))
         ReturnFalse;
      if(!SelfAttentionBackward(query, query_gr, Output, Gradient))
         ReturnFalse;
     }
   else
     {
      if(!cMemoryContext.getGradient().Fill(0) ||
         !cSelfContext.getGradient().Fill(0) ||
         !cGateProjection.getGradient().Fill(0) ||
         !cGateInput.getGradient().Fill(0))
         ReturnFalse;
      if(!GateElementMultGrad(cSelfContext.getOutput(), cSelfContext.getGradient(),
                              query, query_gr,
                              cGateProjection.getOutput(), cGateProjection.getGradient(),
                              Gradient, None, None, SIGMOID))
         ReturnFalse;
      if(!cGateInput.CalcHiddenGradients(cGateProjection.AsObject()))
         ReturnFalse;
      if(!DeConcat(GetPointer(cGateQueryGradient), GetPointer(cGateContextGradient),
                   cGateInput.getGradient(), (int)iTotalQueryDim, (int)iTotalQueryDim, 1))
         ReturnFalse;
      if(!SumAndNormalize(query_gr, GetPointer(cGateQueryGradient), query_gr,
                          (int)iTotalQueryDim, false, 0, 0, 0, 1.0f))
         ReturnFalse;
      if(!SumAndNormalize(cSelfContext.getGradient(), GetPointer(cGateContextGradient),
                          cSelfContext.getGradient(), (int)iTotalQueryDim, false, 0, 0, 0, 1.0f))
         ReturnFalse;
      CBufferFloat *gate_gr = query_gr;
      CBufferFloat *flash_gr = GetPointer(cQueryGateGradient);
      if(!flash_gr || !flash_gr.Fill(0))
         ReturnFalse;
      if(!cQueryProjection.SetGradient(flash_gr, false))
         ReturnFalse;
      if(!cSelfKVGradient.Fill(0))
        {
         if(!cQueryProjection.SetGradient(gate_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!SelfAttentionBackward(cMemoryContext.getOutput(), cMemoryContext.getGradient(),
                                cSelfContext.getOutput(), cSelfContext.getGradient()))
        {
         if(!cQueryProjection.SetGradient(gate_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!cMemoryKVGradient.Fill(0))
        {
         if(!cQueryProjection.SetGradient(gate_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!FlashAttentionBackward(query, cQueryProjection.getGradient(),
                                 cMemoryContext.getOutput(), cMemoryContext.getGradient()))
        {
         if(!cQueryProjection.SetGradient(gate_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!SumAndNormalize(cQueryProjection.getGradient(), gate_gr, gate_gr,
                          (int)iTotalQueryDim, false, 0, 0, 0, 1.0f))
        {
         if(!cQueryProjection.SetGradient(gate_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!cQueryProjection.SetGradient(gate_gr, false))
         ReturnFalse;
      query_gr = gate_gr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(cQueryProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!cGateProjection.UpdateInputWeights(cGateInput.AsObject()))
      ReturnFalse;
   if(!cQueryProjection.UpdateInputWeights(NeuronOCL.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iInputDim);
   dFileWriteUInt(file_handle, iQueryDim);
   dFileWriteUInt(file_handle, iVariables);
   dFileWriteUInt(file_handle, iHeads);
   dFileWriteUInt(file_handle, iHeadDim);
   dFileWriteUInt(file_handle, iTotalQueryDim);
   dFileWriteUInt(file_handle, iTotalHeads);
   dFileWriteUInt(file_handle, iMemoryLen);
   dFileWriteUInt(file_handle, iMemoryCount);
   dFileWriteUInt(file_handle, uint(bLastQueryValid));
   if(!cQueryProjection.Save(file_handle))
      ReturnFalse;
   if(!cGateInput.Save(file_handle))
      ReturnFalse;
   if(!cGateProjection.Save(file_handle))
      ReturnFalse;
   if(!cMemoryContext.Save(file_handle))
      ReturnFalse;
   if(!cSelfContext.Save(file_handle))
      ReturnFalse;
   if(!cMemoryQuery.Save(file_handle) ||
      !cMemoryAge.Save(file_handle) ||
      !cMemoryScore.Save(file_handle) ||
      !cMemoryValid.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   uint last_valid = 0;
   dFileReadUInt(file_handle, iInputDim);
   dFileReadUInt(file_handle, iQueryDim);
   dFileReadUInt(file_handle, iVariables);
   dFileReadUInt(file_handle, iHeads);
   dFileReadUInt(file_handle, iHeadDim);
   dFileReadUInt(file_handle, iTotalQueryDim);
   dFileReadUInt(file_handle, iTotalHeads);
   dFileReadUInt(file_handle, iMemoryLen);
   dFileReadUInt(file_handle, iMemoryCount);
   dFileReadUInt(file_handle, last_valid);
   bLastQueryValid = (last_valid != 0);
   if(!LoadInsideLayer(file_handle, cQueryProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGateInput.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGateProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMemoryContext.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSelfContext.AsObject()))
      ReturnFalse;
   if(!cMemoryQuery.Load(file_handle) ||
      !cMemoryAge.Load(file_handle) ||
      !cMemoryScore.Load(file_handle) ||
      !cMemoryValid.Load(file_handle))
      ReturnFalse;
//--- transient buffers are rebuilt from dimensions
   if(!cMemoryKV.BufferInit(2 * iMemoryLen * iTotalQueryDim, 0) || !cMemoryKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cMemoryKVGradient.BufferInit(2 * iMemoryLen * iTotalQueryDim, 0) || !cMemoryKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfKV.BufferInit(2 * iVariables * iTotalQueryDim, 0) || !cSelfKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfKVGradient.BufferInit(2 * iVariables * iTotalQueryDim, 0) || !cSelfKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cLogSumExp.BufferInit(iTotalHeads, 0) || !cLogSumExp.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cSelfLogSumExp.BufferInit(iTotalHeads, 0) || !cSelfLogSumExp.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cQueryGateGradient.BufferInit(iTotalQueryDim, 0) || !cQueryGateGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGateQueryGradient.BufferInit(iTotalQueryDim, 0) || !cGateQueryGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGateContextGradient.BufferInit(iTotalQueryDim, 0) || !cGateContextGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverRankTCM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cQueryProjection.SetOpenCL(OpenCL);
   cGateInput.SetOpenCL(OpenCL);
   cGateProjection.SetOpenCL(OpenCL);
   cMemoryContext.SetOpenCL(OpenCL);
   cSelfContext.SetOpenCL(OpenCL);
   cMemoryQuery.BufferCreate(OpenCL);
   cMemoryAge.BufferCreate(OpenCL);
   cMemoryScore.BufferCreate(OpenCL);
   cMemoryValid.BufferCreate(OpenCL);
   cMemoryKV.BufferCreate(OpenCL);
   cMemoryKVGradient.BufferCreate(OpenCL);
   cSelfKV.BufferCreate(OpenCL);
   cSelfKVGradient.BufferCreate(OpenCL);
   cLogSumExp.BufferCreate(OpenCL);
   cSelfLogSumExp.BufferCreate(OpenCL);
   cQueryGateGradient.BufferCreate(OpenCL);
   cGateQueryGradient.BufferCreate(OpenCL);
   cGateContextGradient.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverRankTCM::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cQueryProjection.TrainMode(bTrain);
   cGateInput.TrainMode(bTrain);
   cGateProjection.TrainMode(bTrain);
   cMemoryContext.TrainMode(bTrain);
   cSelfContext.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronCogDriverRankTCM *Source = source;
   if(!cQueryProjection.WeightsUpdate(Source.cQueryProjection.AsObject(), tau))
      ReturnFalse;
   if(!cGateInput.WeightsUpdate(Source.cGateInput.AsObject(), tau))
      ReturnFalse;
   if(!cGateProjection.WeightsUpdate(Source.cGateProjection.AsObject(), tau))
      ReturnFalse;
   if(!cMemoryContext.WeightsUpdate(Source.cMemoryContext.AsObject(), tau))
      ReturnFalse;
   if(!cSelfContext.WeightsUpdate(Source.cSelfContext.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cQueryProjection.Clear())
      ReturnFalse;
   if(!cGateInput.Clear())
      ReturnFalse;
   if(!cGateProjection.Clear())
      ReturnFalse;
   if(!cMemoryContext.Clear())
      ReturnFalse;
   if(!cSelfContext.Clear())
      ReturnFalse;
   if(!cMemoryQuery.Fill(0) ||
      !cMemoryAge.Fill(0) ||
      !cMemoryScore.Fill(0) ||
      !cMemoryValid.Fill(0) ||
      !cMemoryKV.Fill(0) ||
      !cMemoryKVGradient.Fill(0) ||
      !cSelfKV.Fill(0) ||
      !cSelfKVGradient.Fill(0) ||
      !cLogSumExp.Fill(0) ||
      !cSelfLogSumExp.Fill(0) ||
      !cQueryGateGradient.Fill(0) ||
      !cGateQueryGradient.Fill(0) ||
      !cGateContextGradient.Fill(0))
      ReturnFalse;
   iMemoryCount = 0;
   bLastQueryValid = false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::ShiftState(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastShiftState;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iUnits * iWindow};
   setBuffer(kernel, def_k_cdfh_shift_plan, Output.GetIndex())
   setBuffer(kernel, def_k_cdfh_shift_confidence, cConfidence.GetIndex())
   setBuffer(kernel, def_k_cdfh_shift_shifted_plan, cPlanInput.getOutputIndex())
   setBuffer(kernel, def_k_cdfh_shift_shifted_confidence, cShiftedConfidence.GetIndex())
   setBuffer(kernel, def_k_cdfh_shift_delta_mask, cDeltaMask.GetIndex())
   setArgument(kernel, def_k_cdfh_shift_query_dim, (int)iWindow)
   setArgument(kernel, def_k_cdfh_shift_horizon, (int)iUnits)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cPlanInput.getOutput().BufferRead() || !cShiftedConfidence.BufferRead() || !cDeltaMask.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildContextKV(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastBuildKV;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iVariables, iUnits, iWindow};
   setBuffer(kernel, def_k_cdfh_kv_context, cContextProjection.getOutputIndex())
   setBuffer(kernel, def_k_cdfh_kv_out, cContextKV.GetIndex())
   setArgument(kernel, def_k_cdfh_kv_dimension, (int)iWindow)
   setArgument(kernel, def_k_cdfh_kv_horizon, (int)iUnits)
   setArgument(kernel, def_k_cdfh_kv_variables, (int)iVariables)
   setArgument(kernel, def_k_cdfh_kv_heads, (int)iHeads)
   setArgument(kernel, def_k_cdfh_kv_head_dim, (int)iWindowKey)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cContextKV.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildContextProjectionGradient(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastBuildKVGrad;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iVariables, iUnits, iWindow};
   setBuffer(kernel, def_k_cdfh_kvg_kv_gr, cContextKVGradient.GetIndex())
   setBuffer(kernel, def_k_cdfh_kvg_context_gr, cContextProjection.getGradientIndex())
   setArgument(kernel, def_k_cdfh_kvg_dimension, (int)iWindow)
   setArgument(kernel, def_k_cdfh_kvg_horizon, (int)iUnits)
   setArgument(kernel, def_k_cdfh_kvg_variables, (int)iVariables)
   setArgument(kernel, def_k_cdfh_kvg_heads, (int)iHeads)
   setArgument(kernel, def_k_cdfh_kvg_head_dim, (int)iWindowKey)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cContextProjection.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildErrorInput(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < int(iVariables * iWindow))
      ReturnFalse;
   if(!ConcatVecMatrix(NeuronOCL.getOutput(), Output, cErrorInput.getOutput(),
                       iVariables * iWindow, iUnits, iWindow, 1, false))
      ReturnFalse;
//---
   return cErrorHead.FeedForward(cErrorInput.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildErrorTarget(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastGradientRMS;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iUnits};
   setBuffer(kernel, def_k_cdfh_grms_gradient, getGradientIndex())
   setBuffer(kernel, def_k_cdfh_grms_error_target, cErrorTarget.GetIndex())
   setArgument(kernel, def_k_cdfh_grms_dimension, (int)iWindow)
   setArgument(kernel, def_k_cdfh_grms_horizon, (int)iUnits)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cErrorTarget.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildConfidence(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastConfidence;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iUnits};
   setBuffer(kernel, def_k_cdfh_conf_expected_error, cErrorHead.getOutputIndex())
   setBuffer(kernel, def_k_cdfh_conf_confidence, cConfidence.GetIndex())
   setArgument(kernel, def_k_cdfh_conf_total, (int)iUnits)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cConfidence.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::BuildDeltaNorm(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint kernel = def_k_CogDriverForecastDeltaNorm;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {iUnits};
   setBuffer(kernel, def_k_cdfh_delta_plan, Output.GetIndex())
   setBuffer(kernel, def_k_cdfh_delta_shifted_plan, cPlanInput.getOutputIndex())
   setBuffer(kernel, def_k_cdfh_delta_plan_delta, cPlanDelta.GetIndex())
   setBuffer(kernel, def_k_cdfh_delta_plan_delta_norm, cPlanDeltaNorm.GetIndex())
   setArgument(kernel, def_k_cdfh_delta_query_dim, (int)iWindow)
   setArgument(kernel, def_k_cdfh_delta_horizon, (int)iUnits)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!cPlanDelta.BufferRead() || !cPlanDeltaNorm.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                        uint dimension, uint horizon, uint variables, uint forecast_blocks,
                                        uint heads, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimension == 0 || horizon == 0 || variables == 0 || forecast_blocks == 0 || heads == 0 ||
      dimension % heads != 0)
      ReturnFalse;
   uint head_dim = dimension / heads;
   if(!CNeuronMLMHAttentionOCL::Init(numOutputs, myIndex, open_cl, dimension, head_dim,
                                     heads, horizon, forecast_blocks, optimization_type, batch))
      ReturnFalse;
//---
   iVariables = variables;
   SetActivationFunction(None);
//--- wrappers and trainable context projection
   if(!cPlanInput.Init(0, 0, OpenCL, iUnits * iWindow, optimization, iBatch))
      ReturnFalse;
   cPlanInput.SetActivationFunction(None);
   if(!cContextProjection.Init(0, 1, OpenCL, iWindow, iWindow,
                               2 * iUnits * iWindow, iVariables,
                               optimization, iBatch))
      ReturnFalse;
   cContextProjection.SetActivationFunction(None);
   if(!cQueryPE.Init(0, 2, OpenCL, iUnits * iWindow, optimization, iBatch))
      ReturnFalse;
   cQueryPE.SetActivationFunction(None);
//--- forecast blocks
   cCrossAttention.Clear();
   cCrossResidual.Clear();
   cCrossFF1.Clear();
   cCrossFF2.Clear();
   cCrossFFOutput.Clear();
   cSelfOutput.Clear();
   cCrossAttention.SetOpenCL(OpenCL);
   cCrossResidual.SetOpenCL(OpenCL);
   cCrossFF1.SetOpenCL(OpenCL);
   cCrossFF2.SetOpenCL(OpenCL);
   cCrossFFOutput.SetOpenCL(OpenCL);
   cSelfOutput.SetOpenCL(OpenCL);
   uint index = 3;
   uint plan_total = iUnits * iWindow;
   for(uint i = 0; i < iLayers; i++)
     {
      CCrossMHFlashAttention *cross = new CCrossMHFlashAttention();
      if(!cross ||
         !cross.Init(0, index++, OpenCL, iWindowKey, iHeads, false,
                     iUnits, iVariables * iUnits, optimization, iBatch) ||
         !cCrossAttention.Add(cross))
         DeleteObjAndFalse(cross);
      CNeuronBaseOCL *holder = new CNeuronBaseOCL();
      if(!holder ||
         !holder.Init(0, index++, OpenCL, plan_total, optimization, iBatch) ||
         !cCrossResidual.Add(holder))
         DeleteObjAndFalse(holder);
      holder.SetActivationFunction(None);
      CNeuronConvOCL *ff1 = new CNeuronConvOCL();
      if(!ff1 ||
         !ff1.Init(0, index++, OpenCL, iWindow, iWindow, 4 * iWindow, iUnits,
                   optimization, iBatch) ||
         !cCrossFF1.Add(ff1))
         DeleteObjAndFalse(ff1);
      ff1.SetActivationFunction(GELU);
      CNeuronConvOCL *ff2 = new CNeuronConvOCL();
      if(!ff2 ||
         !ff2.Init(0, index++, OpenCL, 4 * iWindow, 4 * iWindow, iWindow, iUnits,
                   optimization, iBatch) ||
         !cCrossFF2.Add(ff2))
         DeleteObjAndFalse(ff2);
      ff2.SetActivationFunction(None);
      holder = new CNeuronBaseOCL();
      if(!holder ||
         !holder.Init(0, index++, OpenCL, plan_total, optimization, iBatch) ||
         !cCrossFFOutput.Add(holder))
         DeleteObjAndFalse(holder);
      holder.SetActivationFunction(None);
      holder = new CNeuronBaseOCL();
      if(!holder ||
         !holder.Init(0, index++, OpenCL, plan_total, optimization, iBatch) ||
         !cSelfOutput.Add(holder))
         DeleteObjAndFalse(holder);
      holder.SetActivationFunction(None);
     }
   uint error_width = (iVariables + 1) * iWindow;
   if(!cErrorInput.Init(0, index++, OpenCL, iUnits * error_width, optimization, iBatch))
      ReturnFalse;
   cErrorInput.SetActivationFunction(None);
   if(!cErrorHead.Init(0, index++, OpenCL, error_width, error_width, 1, iUnits,
                       optimization, iBatch))
      ReturnFalse;
   cErrorHead.SetActivationFunction(SoftPlus);
//--- state and diagnostics
   uint context_kv_total = 2 * iVariables * iUnits * iWindow;
   if(!cContextKV.BufferInit(context_kv_total, 0) || !cContextKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cContextKVGradient.BufferInit(context_kv_total, 0) || !cContextKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cShiftedConfidence.BufferInit(iUnits, 0) || !cShiftedConfidence.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cDeltaMask.BufferInit(iUnits, 0) || !cDeltaMask.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cPlanDelta.BufferInit(plan_total, 0) || !cPlanDelta.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cPlanDeltaNorm.BufferInit(iUnits, 0) || !cPlanDeltaNorm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cConfidence.BufferInit(iUnits, 0) || !cConfidence.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cErrorTarget.BufferInit(iUnits, 0) || !cErrorTarget.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::CrossLayerForward(uint layer,
      CNeuronBaseOCL *plan,
      CNeuronBaseOCL *context)
  {
   if(!plan || !context || layer >= iLayers)
      ReturnFalse;
   CNeuronBaseOCL *query = plan;
   if(layer == 0)
     {
      if(!cQueryPE.FeedForward(plan))
         ReturnFalse;
      query = cQueryPE.AsObject();
     }
   CCrossMHFlashAttention *cross = cCrossAttention.At((int)layer);
   CNeuronBaseOCL *cross_res = cCrossResidual.At((int)layer);
   CNeuronConvOCL *ff1 = cCrossFF1.At((int)layer);
   CNeuronConvOCL *ff2 = cCrossFF2.At((int)layer);
   CNeuronBaseOCL *ff_out = cCrossFFOutput.At((int)layer);
   if(!cross || !cross_res || !ff1 || !ff2 || !ff_out)
      ReturnFalse;
   if(!cross.FeedForward(query, GetPointer(cContextKV)))
      ReturnFalse;
   if(!SumAndNormalize(cross.getOutput(), plan.getOutput(), cross_res.getOutput(), (int)iWindow))
      ReturnFalse;
   if(!ff1.FeedForward(cross_res))
      ReturnFalse;
   if(!ff2.FeedForward(ff1.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(ff2.getOutput(), cross_res.getOutput(), ff_out.getOutput(), (int)iWindow))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::CrossLayerBackward(uint layer, CNeuronBaseOCL *plan,
      CNeuronBaseOCL *context, CBufferFloat *out_grad)
  {
   if(!plan || !context || !out_grad || layer >= iLayers)
      ReturnFalse;
   CNeuronConvOCL *ff1 = cCrossFF1.At((int)layer);
   CNeuronConvOCL *ff2 = cCrossFF2.At((int)layer);
   CNeuronBaseOCL *cross_res = cCrossResidual.At((int)layer);
   CCrossMHFlashAttention *cross = cCrossAttention.At((int)layer);
   if(!ff1 || !ff2 || !cross_res || !cross)
      ReturnFalse;
   CBufferFloat *ff2_gr = ff2.getGradient();
   if(!ff2.SetGradient(out_grad, false))
      ReturnFalse;
   if(!ff1.CalcHiddenGradients(ff2.AsObject()))
     {
      if(!ff2.SetGradient(ff2_gr, false))
         ReturnFalse;
      ReturnFalse;
     }
   if(!ff2.SetGradient(ff2_gr, false))
      ReturnFalse;
   if(!cross_res.CalcHiddenGradients(ff1.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(out_grad, cross_res.getGradient(), cross_res.getGradient(),
                       (int)iWindow, false, 0, 0, 0, 1))
      ReturnFalse;
   CBufferFloat *cross_gr = cross.getGradient();
   if(!cross.SetGradient(cross_res.getGradient(), false))
      ReturnFalse;
   if(!cContextKVGradient.Fill(0))
     {
      if(!cross.SetGradient(cross_gr, false))
         ReturnFalse;
      ReturnFalse;
     }
   if(layer == 0)
     {
      if(!cPlanInput.getGradient().Fill(0))
        {
         if(!cross.SetGradient(cross_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!cQueryPE.CalcHiddenGradients(cross.AsObject(),
                                       GetPointer(cContextKV), GetPointer(cContextKVGradient), None))
        {
         if(!cross.SetGradient(cross_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!cPlanInput.CalcHiddenGradients(cQueryPE.AsObject()))
        {
         if(!cross.SetGradient(cross_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
     }
   else
     {
      if(!plan.CalcHiddenGradients(cross,
                                   GetPointer(cContextKV), GetPointer(cContextKVGradient), None))
        {
         if(!cross.SetGradient(cross_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
      if(!SumAndNormalize(cross_res.getGradient(), plan.getGradient(), plan.getGradient(),
                          (int)iWindow, false, 0, 0, 0, 1))
        {
         if(!cross.SetGradient(cross_gr, false))
            ReturnFalse;
         ReturnFalse;
        }
     }
   if(!cross.SetGradient(cross_gr, false))
      ReturnFalse;
//---
   return BuildContextProjectionGradient();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::CrossLayerUpdateWeights(uint layer, CNeuronBaseOCL *plan,
      CNeuronBaseOCL *context)
  {
   if(!plan || !context || layer >= iLayers)
      ReturnFalse;
   CNeuronBaseOCL *cross_res = cCrossResidual.At((int)layer);
   CNeuronConvOCL *ff1 = cCrossFF1.At((int)layer);
   CNeuronConvOCL *ff2 = cCrossFF2.At((int)layer);
   if(!cross_res || !ff1 || !ff2)
      ReturnFalse;
   if(layer == 0)
      if(!cQueryPE.UpdateInputWeights(cPlanInput.AsObject()))
         ReturnFalse;
   if(!ff1.UpdateInputWeights(cross_res))
      ReturnFalse;
   if(!ff2.UpdateInputWeights(ff1.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::SelfLayerForward(uint layer, CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || layer >= iLayers)
      ReturnFalse;
//--- Calculate Queries, Keys, Values
   CBufferFloat *inputs = NeuronOCL.getOutput();
   CBufferFloat *qkv = QKV_Tensors.At(layer * 2);
   if(!ConvolutionForward(QKV_Weights.At(layer * (optimization == SGD ? 2 : 3)), inputs, qkv,
                          iWindow, 3 * iWindowKey * iHeads, None))
      ReturnFalse;
//--- Score and attention
   CBufferFloat *scores = S_Tensors.At(layer * 2);
   if(!AttentionScore(qkv, scores, false))
      ReturnFalse;
   CBufferFloat *attn = AO_Tensors.At(layer * 2);
   if(!AttentionOut(qkv, scores, attn))
      ReturnFalse;
//--- Attention projection + residual/norm
   CBufferFloat *temp = FF_Tensors.At(layer * 6);
   if(!ConvolutionForward(FF_Weights.At(layer * (optimization == SGD ? 6 : 9)), attn, temp,
                          iWindowKey * iHeads, iWindow, None))
      ReturnFalse;
   if(!SumAndNormalize(temp, inputs, temp, iWindow))
      ReturnFalse;
//--- Self feed-forward + residual/norm
   inputs = temp;
   temp = FF_Tensors.At(layer * 6 + 1);
   if(!ConvolutionForward(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 1), inputs, temp,
                          iWindow, 4 * iWindow, GELU))
      ReturnFalse;
   CBufferFloat *out = Output;
   if(layer + 1 < iLayers)
     {
      CNeuronBaseOCL *holder = cSelfOutput.At((int)layer);
      if(!holder)
         ReturnFalse;
      out = holder.getOutput();
     }
   if(!ConvolutionForward(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 2), temp, out,
                          4 * iWindow, iWindow, activation))
      ReturnFalse;
   if(!SumAndNormalize(out, inputs, out, iWindow))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::SelfLayerBackward(uint layer, CNeuronBaseOCL *NeuronOCL, CBufferFloat *out_grad)
  {
   if(!NeuronOCL || !out_grad || layer >= iLayers)
      ReturnFalse;
//--- Passing gradient through self feed-forward layers
   if(!ConvolutionInputGradients(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 2),
                                 out_grad, FF_Tensors.At(layer * 6 + 1),
                                 FF_Tensors.At(layer * 6 + 4), 4 * iWindow, iWindow, None))
      ReturnFalse;
   CBufferFloat *temp = FF_Tensors.At(layer * 6 + 3);
   if(!ConvolutionInputGradients(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 1),
                                 FF_Tensors.At(layer * 6 + 4), FF_Tensors.At(layer * 6),
                                 temp, iWindow, 4 * iWindow, GELU))
      ReturnFalse;
   if(!SumAndNormalize(out_grad, temp, temp, iWindow, false))
      ReturnFalse;
   out_grad = temp;
//--- Passing gradient through self attention
   if(!ConvolutionInputGradients(FF_Weights.At(layer * (optimization == SGD ? 6 : 9)),
                                 out_grad, AO_Tensors.At(layer * 2),
                                 AO_Tensors.At(layer * 2 + 1), iWindowKey * iHeads, iWindow, None))
      ReturnFalse;
   if(!AttentionInsideGradients(QKV_Tensors.At(layer * 2), QKV_Tensors.At(layer * 2 + 1),
                                S_Tensors.At(layer * 2), AO_Tensors.At(layer * 2 + 1)))
      ReturnFalse;
   temp = NeuronOCL.getGradient();
   if(!ConvolutionInputGradients(QKV_Weights.At(layer * (optimization == SGD ? 2 : 3)),
                                 QKV_Tensors.At(layer * 2 + 1), NeuronOCL.getOutput(),
                                 temp, iWindow, 3 * iWindowKey * iHeads, None))
      ReturnFalse;
   if(!SumAndNormalize(out_grad, temp, temp, iWindow, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::SelfLayerUpdateWeights(uint layer, CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || layer >= iLayers)
      ReturnFalse;
   CBufferFloat *inputs = NeuronOCL.getOutput();
   if(!ConvolutuionUpdateWeights(QKV_Weights.At(layer * (optimization == SGD ? 2 : 3)),
                                 QKV_Tensors.At(layer * 2 + 1), inputs,
                                 (optimization == SGD ? QKV_Weights.At(layer * 2 + 1) : QKV_Weights.At(layer * 3 + 1)),
                                 (optimization == SGD ? NULL : QKV_Weights.At(layer * 3 + 2)),
                                 iWindow, 3 * iWindowKey * iHeads, 0, iHeads))
      ReturnFalse;
   if(!ConvolutuionUpdateWeights(FF_Weights.At(layer * (optimization == SGD ? 6 : 9)),
                                 FF_Tensors.At(layer * 6 + 3), AO_Tensors.At(layer * 2),
                                 (optimization == SGD ? FF_Weights.At(layer * 6 + 3) : FF_Weights.At(layer * 9 + 3)),
                                 (optimization == SGD ? NULL : FF_Weights.At(layer * 9 + 6)),
                                 iWindowKey * iHeads, iWindow, 0, 1))
      ReturnFalse;
   if(!ConvolutuionUpdateWeights(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 1),
                                 FF_Tensors.At(layer * 6 + 4), FF_Tensors.At(layer * 6),
                                 (optimization == SGD ? FF_Weights.At(layer * 6 + 4) : FF_Weights.At(layer * 9 + 4)),
                                 (optimization == SGD ? NULL : FF_Weights.At(layer * 9 + 7)),
                                 iWindow, 4 * iWindow, 0, 1))
      ReturnFalse;
   CBufferFloat *ff2_out_gr = Gradient;
   if(layer + 1 < iLayers)
     {
      CNeuronBaseOCL *holder = cSelfOutput.At((int)layer);
      if(!holder)
         ReturnFalse;
      ff2_out_gr = holder.getGradient();
     }
   if(!ConvolutuionUpdateWeights(FF_Weights.At(layer * (optimization == SGD ? 6 : 9) + 2),
                                 ff2_out_gr, FF_Tensors.At(layer * 6 + 1),
                                 (optimization == SGD ? FF_Weights.At(layer * 6 + 5) : FF_Weights.At(layer * 9 + 5)),
                                 (optimization == SGD ? NULL : FF_Weights.At(layer * 9 + 8)),
                                 4 * iWindow, iWindow, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || NeuronOCL.Neurons() < int(iVariables * iWindow))
      ReturnFalse;
//--- recurrent plan state is captured before Output is overwritten
   if(!ShiftState())
      ReturnFalse;
//--- variables x dimension context -> trainable K/V tokens for variables * horizon
   if(!cContextProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!BuildContextKV())
      ReturnFalse;
//--- forecast transformer blocks
   CNeuronBaseOCL *plan = cPlanInput.AsObject();
   CNeuronBaseOCL *context = cContextProjection.AsObject();
   for(uint i = 0; i < iLayers; i++)
     {
      CNeuronBaseOCL *ff_out = cCrossFFOutput.At((int)i);
      if(!ff_out)
         ReturnFalse;
      if(!CrossLayerForward(i, plan, context))
         ReturnFalse;
      if(!SelfLayerForward(i, ff_out))
         ReturnFalse;
      if(i + 1 < iLayers)
         plan = cSelfOutput.At((int)i);
     }
//--- side outputs
   if(!BuildErrorInput(NeuronOCL))
      ReturnFalse;
   if(!BuildConfidence())
      ReturnFalse;
   if(!BuildDeltaNorm())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//--- expected-error head learns decoder-gradient magnitude without feeding it back into z_pred
   float error = 0;
   if(!BuildErrorTarget())
      ReturnFalse;
   if(!cErrorHead.calcOutputGradients(GetPointer(cErrorTarget), error))
      ReturnFalse;
//--- gradients through forecast blocks; shifted plan branch is detached
   if(!cContextProjection.getGradient().Fill(0))
      ReturnFalse;
   CNeuronBaseOCL *context = cContextProjection.AsObject();
   for(int i = int(iLayers) - 1; i >= 0; i--)
     {
      CNeuronBaseOCL *self_input = cCrossFFOutput.At(i);
      CNeuronBaseOCL *self_out = cSelfOutput.At(i);
      if(!self_input || !self_out)
         ReturnFalse;
      CBufferFloat *out_grad = (i == int(iLayers) - 1 ? Gradient : self_out.getGradient());
      if(!SelfLayerBackward((uint)i, self_input, out_grad))
         ReturnFalse;
      CNeuronBaseOCL *plan = (i == 0 ? cPlanInput.AsObject() : cSelfOutput.At(i - 1));
      if(!CrossLayerBackward((uint)i, plan, context, self_input.getGradient()))
         ReturnFalse;
     }
//--- accumulated KV gradient -> projection gradient -> RankTCM output gradient
   if(!NeuronOCL.CalcHiddenGradients(cContextProjection.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL *plan = cPlanInput.AsObject();
   CNeuronBaseOCL *context = cContextProjection.AsObject();
   for(uint i = 0; i < iLayers; i++)
     {
      CNeuronBaseOCL *ff_out = cCrossFFOutput.At((int)i);
      if(!ff_out)
         ReturnFalse;
      if(!CrossLayerUpdateWeights(i, plan, context))
         ReturnFalse;
      if(!SelfLayerUpdateWeights(i, ff_out))
         ReturnFalse;
      if(i + 1 < iLayers)
         plan = cSelfOutput.At((int)i);
     }
   if(!cContextProjection.UpdateInputWeights(NeuronOCL.AsObject()))
      ReturnFalse;
   if(!cErrorHead.UpdateInputWeights(cErrorInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::Save(const int file_handle)
  {
   if(!CNeuronMLMHAttentionOCL::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iVariables);
   if(!cPlanInput.Save(file_handle) ||
      !cContextProjection.Save(file_handle) ||
      !cQueryPE.Save(file_handle) ||
      !cCrossAttention.Save(file_handle) ||
      !cCrossResidual.Save(file_handle) ||
      !cCrossFF1.Save(file_handle) ||
      !cCrossFF2.Save(file_handle) ||
      !cCrossFFOutput.Save(file_handle) ||
      !cSelfOutput.Save(file_handle) ||
      !cErrorInput.Save(file_handle) ||
      !cErrorHead.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::Load(const int file_handle)
  {
   if(!CNeuronMLMHAttentionOCL::Load(file_handle))
      ReturnFalse;
   dFileReadUInt(file_handle, iVariables);
   if(!LoadInsideLayer(file_handle, cPlanInput.AsObject()) ||
      !LoadInsideLayer(file_handle, cContextProjection.AsObject()) ||
      !LoadInsideLayer(file_handle, cQueryPE.AsObject()) ||
      !cCrossAttention.Load(file_handle) ||
      !cCrossResidual.Load(file_handle) ||
      !cCrossFF1.Load(file_handle) ||
      !cCrossFF2.Load(file_handle) ||
      !cCrossFFOutput.Load(file_handle) ||
      !cSelfOutput.Load(file_handle) ||
      !LoadInsideLayer(file_handle, cErrorInput.AsObject()) ||
      !LoadInsideLayer(file_handle, cErrorHead.AsObject()))
      ReturnFalse;
   cPlanInput.SetOpenCL(OpenCL);
   cContextProjection.SetOpenCL(OpenCL);
   cQueryPE.SetOpenCL(OpenCL);
   cCrossAttention.SetOpenCL(OpenCL);
   cCrossResidual.SetOpenCL(OpenCL);
   cCrossFF1.SetOpenCL(OpenCL);
   cCrossFF2.SetOpenCL(OpenCL);
   cCrossFFOutput.SetOpenCL(OpenCL);
   cSelfOutput.SetOpenCL(OpenCL);
   cErrorInput.SetOpenCL(OpenCL);
   cErrorHead.SetOpenCL(OpenCL);
//--- transient state is rebuilt from dimensions
   uint plan_total = iUnits * iWindow;
   uint context_kv_total = 2 * iVariables * iUnits * iWindow;
   if(!cContextKV.BufferInit(context_kv_total, 0) || !cContextKV.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cContextKVGradient.BufferInit(context_kv_total, 0) || !cContextKVGradient.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cShiftedConfidence.BufferInit(iUnits, 0) || !cShiftedConfidence.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cDeltaMask.BufferInit(iUnits, 0) || !cDeltaMask.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cPlanDelta.BufferInit(plan_total, 0) || !cPlanDelta.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cPlanDeltaNorm.BufferInit(iUnits, 0) || !cPlanDeltaNorm.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cConfidence.BufferInit(iUnits, 0) || !cConfidence.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cErrorTarget.BufferInit(iUnits, 0) || !cErrorTarget.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverForecastHead::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMLMHAttentionOCL::SetOpenCL(obj);
   cPlanInput.SetOpenCL(OpenCL);
   cContextProjection.SetOpenCL(OpenCL);
   cQueryPE.SetOpenCL(OpenCL);
   cCrossAttention.SetOpenCL(OpenCL);
   cCrossResidual.SetOpenCL(OpenCL);
   cCrossFF1.SetOpenCL(OpenCL);
   cCrossFF2.SetOpenCL(OpenCL);
   cCrossFFOutput.SetOpenCL(OpenCL);
   cSelfOutput.SetOpenCL(OpenCL);
   cErrorInput.SetOpenCL(OpenCL);
   cErrorHead.SetOpenCL(OpenCL);
   cContextKV.BufferCreate(OpenCL);
   cContextKVGradient.BufferCreate(OpenCL);
   cShiftedConfidence.BufferCreate(OpenCL);
   cDeltaMask.BufferCreate(OpenCL);
   cPlanDelta.BufferCreate(OpenCL);
   cPlanDeltaNorm.BufferCreate(OpenCL);
   cConfidence.BufferCreate(OpenCL);
   cErrorTarget.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverForecastHead::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cPlanInput.TrainMode(bTrain);
   cContextProjection.TrainMode(bTrain);
   cQueryPE.TrainMode(bTrain);
   cCrossAttention.TrainMode(bTrain);
   cCrossResidual.TrainMode(bTrain);
   cCrossFF1.TrainMode(bTrain);
   cCrossFF2.TrainMode(bTrain);
   cCrossFFOutput.TrainMode(bTrain);
   cSelfOutput.TrainMode(bTrain);
   cErrorInput.TrainMode(bTrain);
   cErrorHead.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   if(!CNeuronMLMHAttentionOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronCogDriverForecastHead *Source = source;
   if(!cPlanInput.WeightsUpdate(Source.cPlanInput.AsObject(), tau) ||
      !cContextProjection.WeightsUpdate(Source.cContextProjection.AsObject(), tau) ||
      !cQueryPE.WeightsUpdate(Source.cQueryPE.AsObject(), tau) ||
      !cCrossAttention.WeightsUpdate(GetPointer(Source.cCrossAttention), tau) ||
      !cCrossResidual.WeightsUpdate(GetPointer(Source.cCrossResidual), tau) ||
      !cCrossFF1.WeightsUpdate(GetPointer(Source.cCrossFF1), tau) ||
      !cCrossFF2.WeightsUpdate(GetPointer(Source.cCrossFF2), tau) ||
      !cCrossFFOutput.WeightsUpdate(GetPointer(Source.cCrossFFOutput), tau) ||
      !cSelfOutput.WeightsUpdate(GetPointer(Source.cSelfOutput), tau) ||
      !cErrorInput.WeightsUpdate(Source.cErrorInput.AsObject(), tau) ||
      !cErrorHead.WeightsUpdate(Source.cErrorHead.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastHead::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cPlanInput.Clear() ||
      !cContextProjection.Clear() ||
      !cQueryPE.Clear() ||
      !cCrossAttention.ClearStates() ||
      !cCrossResidual.ClearStates() ||
      !cCrossFF1.ClearStates() ||
      !cCrossFF2.ClearStates() ||
      !cCrossFFOutput.ClearStates() ||
      !cSelfOutput.ClearStates() ||
      !cErrorInput.Clear() ||
      !cErrorHead.Clear())
      ReturnFalse;
   if(!cContextKV.Fill(0) ||
      !cContextKVGradient.Fill(0) ||
      !cShiftedConfidence.Fill(0) ||
      !cDeltaMask.Fill(0) ||
      !cPlanDelta.Fill(0) ||
      !cPlanDeltaNorm.Fill(0) ||
      !cConfidence.Fill(0) ||
      !cErrorTarget.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint dimension, uint horizon, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimension == 0 || horizon == 0 || dimension % 2 != 0)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, horizon * (dimension + 1),
                            optimization_type, batch))
      ReturnFalse;
//---
   iDimension = dimension;
   iHorizon = horizon;
   SetActivationFunction(None);
   if(!cRoPE.Init(0, myIndex + 1, OpenCL, iHorizon, iDimension, 1, optimization, iBatch))
      ReturnFalse;
   cRoPE.SetActivationFunction(None);
   if(!cConfidenceGradient.BufferInit(iHorizon, 0) || !cConfidenceGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || NeuronOCL.Type() != defNeuronCogDriverForecastHead ||
      NeuronOCL.Neurons() != int(iHorizon * iDimension))
      ReturnFalse;
//---
   CNeuronCogDriverForecastHead *forecast = NeuronOCL;
   CBufferFloat *confidence = forecast.GetConfidence();
   if(!confidence || confidence.Total() < int(iHorizon))
      ReturnFalse;
   if(!cRoPE.FeedForward(NeuronOCL.AsObject()))
      ReturnFalse;
   if(!Concat(cRoPE.getOutput(), confidence, Output, (int)iDimension, 1, (int)iHorizon))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL || NeuronOCL.Type() != defNeuronCogDriverForecastHead)
      ReturnFalse;
   if(!DeConcat(cRoPE.getGradient(), GetPointer(cConfidenceGradient), Gradient,
                (int)iDimension, 1, (int)iHorizon))
      ReturnFalse;
   if(!cRoPE.CalcHiddenGradients(NeuronOCL.AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(),
                       NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, iDimension);
   dFileWriteUInt(file_handle, iHorizon);
   if(!cRoPE.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   dFileReadUInt(file_handle, iDimension);
   dFileReadUInt(file_handle, iHorizon);
   if(!LoadInsideLayer(file_handle, cRoPE.AsObject()))
      ReturnFalse;
   cRoPE.SetOpenCL(OpenCL);
   if(!cConfidenceGradient.BufferInit(iHorizon, 0) || !cConfidenceGradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return Clear();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverForecastToken::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cRoPE.SetOpenCL(OpenCL);
   cConfidenceGradient.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCogDriverForecastToken::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cRoPE.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCogDriverForecastToken::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear() ||
      !cRoPE.Clear() ||
      !cConfidenceGradient.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint window, uint window_out, uint variables,
                            uint stack_size, uint quantiles, float range,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl,  window + window_out,
                                   window + window_out, window_out, 2 * quantiles + 1, variables,
                                   optimization_type, batch))
      ReturnFalse;
//---
   cMSDFlow.Clear();
   cUncertainty.Clear();
   cMSDFlow.SetOpenCL(OpenCL);
   cUncertainty.SetOpenCL(OpenCL);
//--- MSD Flow
   uint index = 0;
   CNeuronBatchNormOCL* norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, window * variables, iBatch, optimization) ||
      !cMSDFlow.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
   index++;
   CNeuronAddToStack* stack = new CNeuronAddToStack();
   if(!stack ||
      !stack.Init(0, index, OpenCL, stack_size, window, variables, optimization, iBatch) ||
      !cMSDFlow.Add(stack))
      DeleteObjAndFalse(stack);
   index++;
   CNeuronUncAD_MSD* msd = new CNeuronUncAD_MSD();
   if(!msd ||
      !msd.Init(0, index, OpenCL, window, stack_size, variables, quantiles, range, optimization, iBatch) ||
      !cMSDFlow.Add(msd))
      DeleteObjAndFalse(msd);
//--- Uncertainty Flow
   index++;
   CNeuronMultiMixAttention* att = new CNeuronMultiMixAttention();
   if(!att ||
      !att.Init(0, index, OpenCL, window * (2 * quantiles + 1), variables, 2 * quantiles + 1, quantiles, 5, 2, optimization, iBatch) ||
      !cUncertainty.Add(att))
      DeleteObjAndFalse(att);
   index++;
   CNeuronDictionaryCrossAtt* dic = new CNeuronDictionaryCrossAtt();
   if(!dic ||
      !dic.Init(0, index, OpenCL, window * (2 * quantiles + 1), variables, 4, stack_size / 2, optimization, iBatch) ||
      !cUncertainty.Add(dic))
      DeleteObjAndFalse(dic);
   index++;
   CNeuronBaseOCL* concat = new CNeuronBaseOCL();
   if(!concat ||
      !concat.Init(0, index, OpenCL, 2 * dic.Neurons(), optimization, iBatch) ||
      !cUncertainty.Add(concat))
      DeleteObjAndFalse(concat);
   concat.SetActivationFunction(None);
   index++;
   CNeuronSpikeConvBlock* conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * window, 2 * window, window_out, 2 * quantiles + 1, variables, optimization, iBatch) ||
      !cUncertainty.Add(conv))
      DeleteObjAndFalse(conv);
//---
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, msd.Neurons() + conv.Neurons(), optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- MSD Flow
   for(int i = 0; i < cMSDFlow.Total(); i++)
     {
      curr = cMSDFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Uncertainty Flow
   for(int i = 0; i < cUncertainty.Total(); i++)
     {
      curr = cUncertainty[i];
      if(!curr)
         ReturnFalse;
      if(prev.Type() != defNeuronDictionaryCrossAtt)
        {
         if(!curr.FeedForward(prev))
            ReturnFalse;
        }
      else
        {
         CNeuronDictionaryCrossAtt* dic = prev;
         uint count = dic.GetUnits() * dic.GetVariables();
         uint dimension = dic.GetFilters();
         prev = dic.GetSimilarity();
         if(!prev ||
            !Concat(dic.getOutput(), prev.getOutput(), curr.getOutput(),
                    dimension, dimension, count))
            ReturnFalse;
        }
      prev = curr;
     }
//---
   prev = cMSDFlow[-1];
   uint count = cConv.GetUnits() * cConv.GetVariables();
   uint dim_msd = prev.Neurons() / count;
   uint dim_unc = curr.Neurons() / count;
   if(!Concat(prev.getOutput(), curr.getOutput(), cConcatenated.getOutput(),
              dim_msd, dim_unc, count))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cConcatenated.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || cMSDFlow.Total() < 1 || cUncertainty.Total() < 1)
      ReturnFalse;
//---
   CNeuronBaseOCL* msd = cMSDFlow[-1];
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* next = cUncertainty[-1];
   if(!msd || !next)
      ReturnFalse;
//--- Final block gradient
   if(!CNeuronSpikeConvBlock::calcInputGradients(cConcatenated.AsObject()))
      ReturnFalse;
//--- Split final concatenation and apply NLL gradient
   uint count = cConv.GetUnits() * cConv.GetVariables();
   if(count == 0)
      ReturnFalse;
   uint dim_msd = msd.Neurons() / count;
   uint dim_unc = next.Neurons() / count;
   if(dim_msd == 0 || dim_unc == 0)
      ReturnFalse;
   if(!DeConcat(msd.getGradient(), next.getGradient(), cConcatenated.getGradient(),
                dim_msd, dim_unc, count))
      ReturnFalse;
   if(!NLLGrad(next.getOutput(), msd.getGradient(), msd.getGradient(), next.getGradient()))
      ReturnFalse;
//--- Uncertainty flow gradient
   for(int i = cUncertainty.Total() - 2; i >= 0; i--)
     {
      curr = cUncertainty[i];
      if(!curr)
         ReturnFalse;
      if(curr.Type() == defNeuronDictionaryCrossAtt)
        {
         CNeuronDictionaryCrossAtt* dictionary = curr;
         CNeuronBaseOCL* similarity = dictionary.GetSimilarity();
         if(!similarity)
            ReturnFalse;
         count = dictionary.GetUnits() * dictionary.GetVariables();
         uint dimension = dictionary.GetFilters();
         if(count == 0 || dimension == 0)
            ReturnFalse;
         if(!DeConcat(dictionary.getGradient(), similarity.getGradient(), next.getGradient(),
                      dimension, dimension, count))
            ReturnFalse;
        }
      else
         if(!curr.CalcHiddenGradients(next))
            ReturnFalse;
      next = curr;
     }
//--- MSD flow intentionally does not propagate gradients to source
   if(!NeuronOCL.getGradient().Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || cMSDFlow.Total() < 1 || cUncertainty.Total() < 1)
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* prev = cMSDFlow[-1];
   if(!prev)
      ReturnFalse;
//--- Update only trainable uncertainty path and final block
   for(int i = 0; i < cUncertainty.Total(); i++)
     {
      curr = cUncertainty[i];
      if(!curr || !prev)
         ReturnFalse;
      if(prev.Type() != defNeuronDictionaryCrossAtt)
        {
         if(!curr.UpdateInputWeights(prev))
            ReturnFalse;
        }
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::updateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cMSDFlow.Save(file_handle))
      ReturnFalse;
   if(!cUncertainty.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!cMSDFlow.Load(file_handle))
      ReturnFalse;
   if(!cUncertainty.Load(file_handle))
      ReturnFalse;
//---
   CNeuronBaseOCL* msd = cMSDFlow[-1];
   CNeuronBaseOCL* uncertainty = cUncertainty[-1];
   if(!msd || !uncertainty)
      ReturnFalse;
   uint index = cMSDFlow.Total() + cUncertainty.Total();
   if(!cConcatenated.Init(0, index, OpenCL, msd.Neurons() + uncertainty.Neurons(), optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_MUE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cMSDFlow.SetOpenCL(OpenCL);
   cUncertainty.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_MUE::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cMSDFlow.TrainMode(bTrain);
   cUncertainty.TrainMode(bTrain);
   cConcatenated.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUncAD_MUE* Source = source;
   dWeightsUpdate(cMSDFlow, Source, tau);
   dWeightsUpdate(cUncertainty, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_MUE::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cMSDFlow.ClearStates())
      ReturnFalse;
   if(!cUncertainty.ClearStates())
      ReturnFalse;
   if(!cConcatenated.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint window, uint units, uint context_window,
                            uint context_units, uint heads, uint key_dimension,
                            uint candidates, uint topK,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(window == 0 || units == 0 || context_window == 0 || context_units == 0 ||
      heads == 0 || key_dimension == 0 || candidates == 0 || topK == 0)
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::Init(numOutputs, myIndex, open_cl,
                                          window, key_dimension, heads,
                                          units, context_window, context_units,
                                          optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cSelfAttention.Init(0, index, OpenCL, window, units, heads,
                           key_dimension, candidates, topK, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() < Neurons())
      ReturnFalse;
//---
   if(!cSelfAttention.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::feedForward(cSelfAttention.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
//---
   if(!CNeuronSpikeMHCrossAttention::calcInputGradients(cSelfAttention.AsObject(),
         SecondInput, SecondGradient, SecondActivation))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSelfAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
//---
   if(!cSelfAttention.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::updateInputWeights(cSelfAttention.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::Save(const int file_handle)
  {
   if(!CNeuronSpikeMHCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cSelfAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::Load(const int file_handle)
  {
   if(!CNeuronSpikeMHCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSelfAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_UGP::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeMHCrossAttention::SetOpenCL(obj);
   cSelfAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_UGP::TrainMode(bool flag)
  {
   CNeuronSpikeMHCrossAttention::TrainMode(flag);
   cSelfAttention.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeMHCrossAttention::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUncAD_UGP* Source = source;
   dWeightsUpdate(cSelfAttention, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGP::Clear(void)
  {
   if(!CNeuronSpikeMHCrossAttention::Clear())
      ReturnFalse;
   if(!cSelfAttention.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint window, uint units, uint context_window,
                             uint context_units, uint mode_units, uint heads,
                             uint key_dimension,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(window == 0 || units == 0 || context_window == 0 || context_units == 0 ||
      mode_units == 0 || heads == 0 || key_dimension == 0)
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::Init(numOutputs, myIndex, open_cl,
                                          window, key_dimension, heads,
                                          units, context_window, context_units,
                                          optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cMode.Init(0, index, OpenCL, window * mode_units, optimization, iBatch))
      ReturnFalse;
   cMode.SetActivationFunction(None);
   index++;
   if(!cModeFusion.Init(0, index, OpenCL, window, key_dimension, heads,
                        units, window, mode_units, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(NeuronOCL.Neurons() < Neurons())
      ReturnFalse;
//---
   if(bTrain)
      if(!cMode.FeedForward())
         ReturnFalse;
   if(!cModeFusion.FeedForward(NeuronOCL, cMode.getOutput()))
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::feedForward(cModeFusion.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
//---
   if(!CNeuronSpikeMHCrossAttention::calcInputGradients(cModeFusion.AsObject(),
         SecondInput, SecondGradient, SecondActivation))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cModeFusion.AsObject(), cMode.getOutput(),
                                     cMode.getGradient(), None))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
//---
   if(!cModeFusion.UpdateInputWeights(NeuronOCL, cMode.getOutput()))
      ReturnFalse;
   if(!cMode.UpdateInputWeights())
      ReturnFalse;
   if(!CNeuronSpikeMHCrossAttention::updateInputWeights(cModeFusion.AsObject(), SecondInput))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::Save(const int file_handle)
  {
   if(!CNeuronSpikeMHCrossAttention::Save(file_handle))
      ReturnFalse;
   if(!cMode.Save(file_handle))
      ReturnFalse;
   if(!cModeFusion.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::Load(const int file_handle)
  {
   if(!CNeuronSpikeMHCrossAttention::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMode.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cModeFusion.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_UGPL::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeMHCrossAttention::SetOpenCL(obj);
   cMode.SetOpenCL(OpenCL);
   cModeFusion.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD_UGPL::TrainMode(bool flag)
  {
   CNeuronSpikeMHCrossAttention::TrainMode(flag);
   cMode.TrainMode(bTrain);
   cModeFusion.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeMHCrossAttention::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUncAD_UGPL* Source = source;
   dWeightsUpdate(cMode, Source, tau);
   dWeightsUpdate(cModeFusion, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD_UGPL::Clear(void)
  {
   if(!CNeuronSpikeMHCrossAttention::Clear())
      ReturnFalse;
   if(!cMode.Clear())
      ReturnFalse;
   if(!cModeFusion.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint &dimensions[], uint units_s, uint heads,
                        uint stack_size, uint quantiles, uint embed_size,
                        uint candidates, uint topK, uint latent_count,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(dimensions.Size() < 2 || units_s == 0 || heads == 0 || stack_size == 0 ||
      quantiles == 0 || embed_size == 0 || candidates < 2 || topK == 0 ||
      topK > candidates || latent_count == 0 || embed_size % heads != 0)
      ReturnFalse;
//---
   iInputDim = dimensions[0] * units_s;
   for(uint i = 1; i < dimensions.Size(); i++)
     {
      if(dimensions[i] == 0)
         ReturnFalse;
      iInputDim += dimensions[i];
     }
   if(dimensions[0] == 0)
      ReturnFalse;
   iAgentUnits = units_s;
   iContextUnits = dimensions.Size() - 1;
   iEmbedSize = embed_size;
   iQuantiles = quantiles;
   iCandidates = candidates;
   iTopK = topK;
//---
   uint agent_dim = iAgentUnits * iEmbedSize;
   uint context_dim = iContextUnits * iEmbedSize;
   uint qm_size = iAgentUnits * iEmbedSize * (2 * iQuantiles + 1);
   uint fusion_size = agent_dim + context_dim + qm_size;
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, fusion_size, fusion_size,
                                   latent_count, 1, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint token_units = iAgentUnits + iContextUnits;
   uint all_dimensions[];
   if(ArrayResize(all_dimensions, token_units) < int(token_units))
      ReturnFalse;
   for(uint i = 0; i < iAgentUnits; i++)
      all_dimensions[i] = dimensions[0];
   for(uint i = 0; i < iContextUnits; i++)
      all_dimensions[iAgentUnits + i] = dimensions[i + 1];
//---
   uint index = 0;
   if(!cTokenizer.Init(0, index, OpenCL, all_dimensions, token_units, iEmbedSize,
                       token_units, iCandidates, iTopK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAgents.Init(0, index, OpenCL, agent_dim, optimization, iBatch))
      ReturnFalse;
   cAgents.SetActivationFunction(None);
   index++;
   if(!cContext.Init(0, index, OpenCL, context_dim, optimization, iBatch))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   index++;
   if(!cMUE.Init(0, index, OpenCL, iEmbedSize, iEmbedSize, iAgentUnits,
                 stack_size, iQuantiles, 1.0f, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cUGP.Init(0, index, OpenCL, iEmbedSize, iAgentUnits,
                 iEmbedSize, iAgentUnits * (2 * iQuantiles + 1),
                 heads, iEmbedSize / heads, candidates, topK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cUGPL.Init(0, index, OpenCL, iEmbedSize, iContextUnits, iEmbedSize, iAgentUnits,
                  candidates, heads, iEmbedSize / heads, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMapInteraction.Init(0, index, OpenCL, iEmbedSize, iEmbedSize / heads, heads,
                            iContextUnits, iEmbedSize, iAgentUnits * (2 * iQuantiles + 1),
                            optimization, iBatch))
      ReturnFalse;
//--- tau a
   index++;
   if(!cTauAValues.Init(0, index, OpenCL, iEmbedSize, iEmbedSize,
                        iCandidates * iEmbedSize, iAgentUnits, 1, optimization, iBatch))
      ReturnFalse;
   cTauAValues.SetActivationFunction(SoftPlus);
   index++;
   if(!cTauAScores.Init(0, index, OpenCL, iEmbedSize, iEmbedSize,
                        iCandidates, iAgentUnits, 1, optimization, iBatch))
      ReturnFalse;
   cTauAScores.SetActivationFunction(None);
   index++;
   if(!cTauASoftMax.Init(0, index, OpenCL, iAgentUnits, iCandidates, iTopK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTauAWeighted.Init(0, index, OpenCL, agent_dim, optimization, iBatch))
      ReturnFalse;
   cTauAWeighted.SetActivationFunction(None);
//--- tau e
   index++;
   if(!cTauEValues.Init(0, index, OpenCL, iEmbedSize, iEmbedSize,
                        iCandidates * iEmbedSize, iContextUnits, 1, optimization, iBatch))
      ReturnFalse;
   cTauEValues.SetActivationFunction(SoftPlus);
   index++;
   if(!cTauEScores.Init(0, index, OpenCL, iEmbedSize, iEmbedSize,
                        iCandidates, iContextUnits, 1, optimization, iBatch))
      ReturnFalse;
   cTauEScores.SetActivationFunction(None);
   index++;
   if(!cTauESoftMax.Init(0, index, OpenCL, iContextUnits, iCandidates, iTopK, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTauEWeighted.Init(0, index, OpenCL, iContextUnits * iEmbedSize, optimization, iBatch))
      ReturnFalse;
   cTauEWeighted.SetActivationFunction(None);
//--- fusion
   index++;
   if(!cFusionInput.Init(0, index, OpenCL, cTauAWeighted.Neurons() + cTauEWeighted.Neurons() + qm_size,
                         optimization, iBatch))
      ReturnFalse;
   cFusionInput.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::TauAForward(void)
  {
   if(!cTauAValues.FeedForward(cUGP.AsObject()) ||
      !cTauAScores.FeedForward(cUGP.AsObject()) ||
      !cTauASoftMax.FeedForward(cTauAScores.AsObject()))
      ReturnFalse;
   if(!SparseMatMul(cTauASoftMax.GetIndexes(), cTauASoftMax.getOutput(), cTauAValues.getOutput(),
                    cTauAWeighted.getOutput(), iAgentUnits, iTopK, iCandidates, iEmbedSize))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::TauEForward(void)
  {
   if(!cTauEValues.FeedForward(cMapInteraction.AsObject()) ||
      !cTauEScores.FeedForward(cMapInteraction.AsObject()) ||
      !cTauESoftMax.FeedForward(cTauEScores.AsObject()))
      ReturnFalse;
   if(!SparseMatMul(cTauESoftMax.GetIndexes(), cTauESoftMax.getOutput(), cTauEValues.getOutput(),
                    cTauEWeighted.getOutput(), iContextUnits, iTopK, iCandidates, iEmbedSize))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() < int(iInputDim))
      ReturnFalse;
//--- Unified tokenization: market agents first, context tokens second
   if(!cTokenizer.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!DeConcat(cAgents.getOutput(), cContext.getOutput(), cTokenizer.getOutput(),
                cAgents.Neurons(), cContext.Neurons(), 1))
      ReturnFalse;
//--- Uncertainty-guided prediction/planning graph
   if(!cMUE.FeedForward(cAgents.AsObject()))
      ReturnFalse;
   if(!cUGP.FeedForward(cAgents.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cUGPL.FeedForward(cContext.AsObject(), cUGP.getOutput()))
      ReturnFalse;
   if(!cMapInteraction.FeedForward(cUGPL.AsObject(), cMUE.getOutput()))
      ReturnFalse;
//--- Differentiable tau proposals
   if(!TauAForward() || !TauEForward())
      ReturnFalse;
//--- Proposal fusion for critic/actor head
   if(!Concat(cTauAWeighted.getOutput(), cTauEWeighted.getOutput(), cMUE.getOutput(),
              cFusionInput.getOutput(),
              cTauAWeighted.Neurons(), cTauEWeighted.Neurons(), cMUE.Neurons(), 1))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cFusionInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::TauAGradients(void)
  {
   if(!SparseMatMulGrad(cTauASoftMax.GetIndexes(), cTauASoftMax.getOutput(),
                        cTauASoftMax.getGradient(), cTauAValues.getOutput(),
                        cTauAValues.getGradient(), cTauAWeighted.getGradient(),
                        iAgentUnits, iTopK, iCandidates, iEmbedSize))
      ReturnFalse;
   if(!cTauAScores.CalcHiddenGradients(cTauASoftMax.AsObject()))
      ReturnFalse;
//--- Fan-in to Q_a''
   if(!cUGP.CalcHiddenGradients(cTauAValues.AsObject()))
      ReturnFalse;
   CBufferFloat* qaa_gr = cUGP.getGradient();
   if(!cUGP.SetGradient(cUGP.getPrevOutput(), false))
      ReturnFalse;
   if(!cUGP.CalcHiddenGradients(cTauAScores.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(qaa_gr, cUGP.getGradient(), qaa_gr, 1, false, 0, 0, 0, 1) ||
      !cUGP.SetGradient(qaa_gr, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::TauEGradients(void)
  {
   if(!SparseMatMulGrad(cTauESoftMax.GetIndexes(), cTauESoftMax.getOutput(),
                        cTauESoftMax.getGradient(), cTauEValues.getOutput(),
                        cTauEValues.getGradient(), cTauEWeighted.getGradient(),
                        iContextUnits, iTopK, iCandidates, iEmbedSize))
      ReturnFalse;
   if(!cTauEScores.CalcHiddenGradients(cTauESoftMax.AsObject()))
      ReturnFalse;
//--- Fan-in to Q_e'''
   if(!cMapInteraction.CalcHiddenGradients(cTauEValues.AsObject()))
      ReturnFalse;
   CBufferFloat* qeee_gr = cMapInteraction.getGradient();
   if(!cMapInteraction.SetGradient(cMapInteraction.getPrevOutput(), false))
      ReturnFalse;
   if(!cMapInteraction.CalcHiddenGradients(cTauEScores.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(qeee_gr, cMapInteraction.getGradient(), qeee_gr, 1, false, 0, 0, 0, 1) ||
      !cMapInteraction.SetGradient(qeee_gr, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Output fusion
   if(!CNeuronSpikeConvBlock::calcInputGradients(cFusionInput.AsObject()))
      ReturnFalse;
   if(!DeConcat(cTauAWeighted.getGradient(), cTauEWeighted.getGradient(), cMUE.getGradient(),
                cFusionInput.getGradient(), cTauAWeighted.Neurons(), cTauEWeighted.Neurons(),
                cMUE.Neurons(), 1))
      ReturnFalse;
//--- tau heads
   if(!TauAGradients() || !TauEGradients())
      ReturnFalse;
//--- Q_e''' = CrossAttention(Q_e'', Q_m')
   CBufferFloat* qm_gr = cMUE.getGradient();
   if(!cMUE.SetGradient(cMUE.getPrevOutput(), false))
      ReturnFalse;
   if(!cUGPL.CalcHiddenGradients(cMapInteraction.AsObject(), cMUE.getOutput(),
                                 cMUE.getGradient(), None))
      ReturnFalse;
   if(!SumAndNormalize(qm_gr, cMUE.getGradient(), qm_gr, 1, false, 0, 0, 0, 1) ||
      !cMUE.SetGradient(qm_gr, false))
      ReturnFalse;
//--- Q_e'' = UGPL(Q_e, Q_a'')
   CBufferFloat* qaa_gr = cUGP.getGradient();
   if(!cUGP.SetGradient(cUGP.getPrevOutput(), false))
      ReturnFalse;
   if(!cContext.CalcHiddenGradients(cUGPL.AsObject(), cUGP.getOutput(),
                                    cUGP.getGradient(), None))
      ReturnFalse;
   if(!SumAndNormalize(qaa_gr, cUGP.getGradient(), qaa_gr, 1, false, 0, 0, 0, 1) ||
      !cUGP.SetGradient(qaa_gr, false))
      ReturnFalse;
//--- Q_a'' = UGP(Q_a, Q_m')
   qm_gr = cMUE.getGradient();
   if(!cMUE.SetGradient(cMUE.getPrevOutput(), false))
      ReturnFalse;
   if(!cAgents.CalcHiddenGradients(cUGP.AsObject(), cMUE.getOutput(),
                                   cMUE.getGradient(), None))
      ReturnFalse;
   if(!SumAndNormalize(qm_gr, cMUE.getGradient(), qm_gr, 1, false, 0, 0, 0, 1) ||
      !cMUE.SetGradient(qm_gr, false))
      ReturnFalse;
//--- Q_m' = MUE(Q_a). MUE intentionally does not propagate source gradient.
   CBufferFloat* qa_gr = cAgents.getGradient();
   if(!cAgents.SetGradient(cAgents.getPrevOutput(), false))
      ReturnFalse;
   if(!cAgents.CalcHiddenGradients(cMUE.AsObject()))
      ReturnFalse;
   if(!cAgents.SetGradient(qa_gr, false))
      ReturnFalse;
//--- Unified tokenizer
   if(!Concat(cAgents.getGradient(), cContext.getGradient(), cTokenizer.getGradient(),
              cAgents.Neurons(), cContext.Neurons(), 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTokenizer.AsObject()))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMUE.UpdateInputWeights(cAgents.AsObject()))
      ReturnFalse;
   if(!cUGP.UpdateInputWeights(cAgents.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cUGPL.UpdateInputWeights(cContext.AsObject(), cUGP.getOutput()))
      ReturnFalse;
   if(!cMapInteraction.UpdateInputWeights(cUGPL.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cTauAValues.UpdateInputWeights(cUGP.AsObject()))
      ReturnFalse;
   if(!cTauAScores.UpdateInputWeights(cUGP.AsObject()))
      ReturnFalse;
   if(!cTauEValues.UpdateInputWeights(cMapInteraction.AsObject()))
      ReturnFalse;
   if(!cTauEScores.UpdateInputWeights(cMapInteraction.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cFusionInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iInputDim, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iAgentUnits, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iContextUnits, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iEmbedSize, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iQuantiles, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iCandidates, INT_VALUE) < INT_VALUE ||
      FileWriteInteger(file_handle, (int)iTopK, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   if(!cTokenizer.Save(file_handle) ||
      !cAgents.Save(file_handle) ||
      !cContext.Save(file_handle) ||
      !cMUE.Save(file_handle) ||
      !cUGP.Save(file_handle) ||
      !cUGPL.Save(file_handle) ||
      !cMapInteraction.Save(file_handle) ||
      !cTauAValues.Save(file_handle) ||
      !cTauAScores.Save(file_handle) ||
      !cTauASoftMax.Save(file_handle) ||
      !cTauAWeighted.Save(file_handle) ||
      !cTauEValues.Save(file_handle) ||
      !cTauEScores.Save(file_handle) ||
      !cTauESoftMax.Save(file_handle) ||
      !cTauEWeighted.Save(file_handle) ||
      !cFusionInput.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iInputDim = (uint)FileReadInteger(file_handle, INT_VALUE);
   iAgentUnits = (uint)FileReadInteger(file_handle, INT_VALUE);
   iContextUnits = (uint)FileReadInteger(file_handle, INT_VALUE);
   iEmbedSize = (uint)FileReadInteger(file_handle, INT_VALUE);
   iQuantiles = (uint)FileReadInteger(file_handle, INT_VALUE);
   iCandidates = (uint)FileReadInteger(file_handle, INT_VALUE);
   iTopK = (uint)FileReadInteger(file_handle, INT_VALUE);
//---
   if(!LoadInsideLayer(file_handle, cTokenizer.AsObject()) ||
      !LoadInsideLayer(file_handle, cAgents.AsObject()) ||
      !LoadInsideLayer(file_handle, cContext.AsObject()) ||
      !LoadInsideLayer(file_handle, cMUE.AsObject()) ||
      !LoadInsideLayer(file_handle, cUGP.AsObject()) ||
      !LoadInsideLayer(file_handle, cUGPL.AsObject()) ||
      !LoadInsideLayer(file_handle, cMapInteraction.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauAValues.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauAScores.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauASoftMax.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauAWeighted.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauEValues.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauEScores.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauESoftMax.AsObject()) ||
      !LoadInsideLayer(file_handle, cTauEWeighted.AsObject()) ||
      !LoadInsideLayer(file_handle, cFusionInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cTokenizer.SetOpenCL(OpenCL);
   cAgents.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cMUE.SetOpenCL(OpenCL);
   cUGP.SetOpenCL(OpenCL);
   cUGPL.SetOpenCL(OpenCL);
   cMapInteraction.SetOpenCL(OpenCL);
   cTauAValues.SetOpenCL(OpenCL);
   cTauAScores.SetOpenCL(OpenCL);
   cTauASoftMax.SetOpenCL(OpenCL);
   cTauAWeighted.SetOpenCL(OpenCL);
   cTauEValues.SetOpenCL(OpenCL);
   cTauEScores.SetOpenCL(OpenCL);
   cTauESoftMax.SetOpenCL(OpenCL);
   cTauEWeighted.SetOpenCL(OpenCL);
   cFusionInput.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronUncAD::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cTokenizer.TrainMode(bTrain);
   cAgents.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cMUE.TrainMode(bTrain);
   cUGP.TrainMode(bTrain);
   cUGPL.TrainMode(bTrain);
   cMapInteraction.TrainMode(bTrain);
   cTauAValues.TrainMode(bTrain);
   cTauAScores.TrainMode(bTrain);
   cTauASoftMax.TrainMode(bTrain);
   cTauAWeighted.TrainMode(bTrain);
   cTauEValues.TrainMode(bTrain);
   cTauEScores.TrainMode(bTrain);
   cTauESoftMax.TrainMode(bTrain);
   cTauEWeighted.TrainMode(bTrain);
   cFusionInput.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronUncAD* Source = source;
   dWeightsUpdate(cTokenizer, Source, tau);
   dWeightsUpdate(cMUE, Source, tau);
   dWeightsUpdate(cUGP, Source, tau);
   dWeightsUpdate(cUGPL, Source, tau);
   dWeightsUpdate(cMapInteraction, Source, tau);
   dWeightsUpdate(cTauAValues, Source, tau);
   dWeightsUpdate(cTauAScores, Source, tau);
   dWeightsUpdate(cTauEValues, Source, tau);
   dWeightsUpdate(cTauEScores, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronUncAD::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cTokenizer.Clear() ||
      !cAgents.Clear() ||
      !cContext.Clear() ||
      !cMUE.Clear() ||
      !cUGP.Clear() ||
      !cUGPL.Clear() ||
      !cMapInteraction.Clear() ||
      !cTauAValues.Clear() ||
      !cTauAScores.Clear() ||
      !cTauASoftMax.Clear() ||
      !cTauAWeighted.Clear() ||
      !cTauEValues.Clear() ||
      !cTauEScores.Clear() ||
      !cTauESoftMax.Clear() ||
      !cTauEWeighted.Clear() ||
      !cFusionInput.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::DetachTauEFromMPI(void)
  {
   if(!bTauELinked)
      return true;
//---
   int output_total = cTauEWeighted.Neurons();
   int gradient_total = cTauEWeighted.getGradient().Total();
   if(output_total <= 0 || gradient_total <= 0)
      ReturnFalse;
//---
   CBufferFloat *output = new CBufferFloat();
   CBufferFloat *gradient = new CBufferFloat();
   if(!output || !gradient)
     {
      DeleteObj(output);
      DeleteObj(gradient);
      ReturnFalse;
     }
   if(!output.BufferInit(output_total, 0) ||
      !gradient.BufferInit(gradient_total, 0))
     {
      DeleteObj(output);
      DeleteObj(gradient);
      ReturnFalse;
     }
   if(!cTauEWeighted.SetOutput(output, false))
     {
      DeleteObj(output);
      DeleteObj(gradient);
      ReturnFalse;
     }
   if(!cTauEWeighted.SetGradient(gradient, false))
     {
      DeleteObj(gradient);
      ReturnFalse;
     }
   bTauELinked = false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::LinkTauEToMPI(void)
  {
   if(bTauELinked)
      return true;
   if(cMPI.Neurons() != cTauEWeighted.Neurons() ||
      cMPI.getGradient().Total() != cTauEWeighted.getGradient().Total())
      ReturnFalse;
//---
   if(!cTauEWeighted.SetOutput(cMPI.getOutput(), true) ||
      !cTauEWeighted.SetGradient(cMPI.getGradient(), true))
      ReturnFalse;
   bTauELinked = true;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint &dimensions[], uint units_s, uint heads,
                        uint stack_size, uint quantiles, uint embed_size,
                        uint candidates, uint topK, uint latent_count,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!DetachTauEFromMPI())
      ReturnFalse;
   if(!CNeuronUncAD::Init(numOutputs, myIndex, open_cl, dimensions, units_s,
                          heads, stack_size, quantiles, embed_size, candidates,
                          topK, latent_count, optimization_type, batch))
      ReturnFalse;
//---
   uint mpi_latent = (uint)cTauEWeighted.Neurons();
   uint mpi_market_dim = iEmbedSize;
   uint mpi_market_units = iAgentUnits * (2 * iQuantiles + 1);
   if(mpi_latent == 0 || mpi_market_dim == 0 || mpi_market_units == 0 ||
      heads == 0 || (mpi_latent % heads) != 0)
      ReturnFalse;
   if(!cMPI.Init(0, 0, OpenCL, mpi_latent, iCandidates, mpi_market_dim,
                 mpi_market_units, 5, topK, heads, mpi_latent / heads,
                 optimization, iBatch))
      ReturnFalse;
//---
   return LinkTauEToMPI();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::TauEForward(void)
  {
   if(!bTauELinked && !LinkTauEToMPI())
      ReturnFalse;
   if(!cTauEValues.FeedForward(cMapInteraction.AsObject()))
      ReturnFalse;
   if(!cMPI.FeedForward(cTauEValues.AsObject(), cMUE.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::TauEGradients(void)
  {
   if(!bTauELinked && !LinkTauEToMPI())
      ReturnFalse;
//--- Preserve the fusion gradient into MUE and add MPI's second-input gradient.
   CBufferFloat *qm_gr = cMUE.getGradient();
   if(!cMUE.SetGradient(cMUE.getPrevOutput(), false))
      ReturnFalse;
   if(!cTauEValues.CalcHiddenGradients(cMPI.AsObject(), cMUE.getOutput(),
                                       cMUE.getGradient(), (ENUM_ACTIVATION)cMUE.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(qm_gr, cMUE.getGradient(), qm_gr, 1, false, 0, 0, 0, 1) ||
      !cMUE.SetGradient(qm_gr, false))
      ReturnFalse;
   if(!cMapInteraction.CalcHiddenGradients(cTauEValues.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cTokenizer.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMUE.UpdateInputWeights(cAgents.AsObject()))
      ReturnFalse;
   if(!cUGP.UpdateInputWeights(cAgents.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cUGPL.UpdateInputWeights(cContext.AsObject(), cUGP.getOutput()))
      ReturnFalse;
   if(!cMapInteraction.UpdateInputWeights(cUGPL.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cMPI.UpdateInputWeights(cMapInteraction.AsObject(), cMUE.getOutput()))
      ReturnFalse;
   if(!cTauAValues.UpdateInputWeights(cUGP.AsObject()))
      ReturnFalse;
   if(!cTauAScores.UpdateInputWeights(cUGP.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cFusionInput.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::Save(const int file_handle)
  {
   if(!CNeuronUncAD::Save(file_handle))
      ReturnFalse;
   if(!cMPI.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::Load(const int file_handle)
  {
   if(!DetachTauEFromMPI())
      ReturnFalse;
   if(!CNeuronUncAD::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMPI.AsObject()))
      ReturnFalse;
//---
   return LinkTauEToMPI();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomAD::SetOpenCL(COpenCLMy *obj)
  {
   DetachTauEFromMPI();
   CNeuronUncAD::SetOpenCL(obj);
   cMPI.SetOpenCL(OpenCL);
   if(cMPI.Neurons() == cTauEWeighted.Neurons() && cMPI.Neurons() > 0)
      LinkTauEToMPI();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMomAD::TrainMode(bool flag)
  {
   CNeuronUncAD::TrainMode(flag);
   cMPI.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   if(!CNeuronUncAD::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMomAD *Source = source;
   dWeightsUpdate(cMPI, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMomAD::Clear(void)
  {
   if(!CNeuronUncAD::Clear())
      ReturnFalse;
   if(!cMPI.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
