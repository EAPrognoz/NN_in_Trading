//+------------------------------------------------------------------+
//|                                   NeuroNet_InferenceSnapshot.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#ifndef NEURONET_INFERENCE_SNAPSHOT_WORK_MQH
#define NEURONET_INFERENCE_SNAPSHOT_WORK_MQH
//--- Restricted native frozen-inference state path. Ordinary Load bytes/reset unchanged.
//+------------------------------------------------------------------+
//| InferenceRuntimeLoadActive                                       |
//+------------------------------------------------------------------+
bool COpenCLMy::InferenceRuntimeLoadActive(void) const
  {
   return(CheckPointer(m_inference_owner) != POINTER_INVALID
          && m_inference_owner.InferenceRuntimeLoadActive(GetPointer(this)));
  }
//+------------------------------------------------------------------+
//| NNInferenceRemaining                                             |
//+------------------------------------------------------------------+
bool NNInferenceRemaining(const int handle, const ulong required)
  {
   if(handle == INVALID_HANDLE)
      return(false);
   const ulong position=FileTell(handle);
   const ulong size=FileSize(handle);
   return(position <= size && required <= size - position);
  }
//+------------------------------------------------------------------+
//| NNInferenceHeader                                                |
//+------------------------------------------------------------------+
bool NNInferenceHeader(const int handle, const bool restore, const uint value)
  {
   if(handle == INVALID_HANDLE)
      return(false);
   if(restore)
      return(NNInferenceRemaining(handle, 4) && (uint)FileReadInteger(handle) == value);
   return(FileWriteInteger(handle, (int)value) == 4);
  }
//+------------------------------------------------------------------+
//| NNInferenceFinite                                                |
//+------------------------------------------------------------------+
bool NNInferenceFinite(CBufferFloat *buffer)
  {
   if(buffer == NULL)
      return(true);
   if(CheckPointer(buffer) == POINTER_INVALID)
      return(false);
   if(buffer.Total() == 0)
      return(true);
   if(buffer.Total() < 0)
      return(false);
   if(buffer.GetIndex() >= 0)
     {
      COpenCLMy *receiver = dynamic_cast<COpenCLMy*>(buffer.GetOpenCL());
      if(!receiver || receiver.GetBufferSize(buffer.GetIndex()) != (ulong)buffer.Total() * sizeof(float))
         return(false);
      if(!buffer.BufferRead())
         return(false);
     }
   for(int i = 0; i < buffer.Total(); i++)
      if(!MathIsValidNumber((double)(*buffer)[i]))
         return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| Validate an optional finite buffer                               |
//+------------------------------------------------------------------+
//--- CNeuronBaseOCL::Load uses DeleteObj for absent parameter buffers but does not null the pointer.
bool NNInferenceOptionalFinite(CBufferFloat *buffer)
  {
   return(buffer == NULL || CheckPointer(buffer) == POINTER_INVALID || NNInferenceFinite(buffer));
  }
//+------------------------------------------------------------------+
//| NNInferenceOmittedBuffer                                         |
//+------------------------------------------------------------------+
bool NNInferenceOmittedBuffer(const int handle, const bool restore, CBufferFloat *buffer)
  {
   if(buffer == NULL || CheckPointer(buffer) == POINTER_INVALID || buffer.Total() <= 0 || !buffer.GetOpenCL() || buffer.GetIndex() < 0
      || !NNInferenceHeader(handle, restore, (uint)buffer.Total()))
      return(false);
   if(!restore)
     {
      if(!NNInferenceFinite(buffer))
         return(false);
      for(int i = 0; i < buffer.Total(); i++)
         if(FileWriteFloat(handle, (*buffer)[i]) != 4)
            return(false);
      return(true);
     }
   if(!NNInferenceRemaining(handle, (ulong)buffer.Total() * 4))
      return(false);
   float values[];
   if(ArrayResize(values, buffer.Total()) != buffer.Total())
      return(false);
   for(int i = 0; i < buffer.Total(); i++)
     {
      values[i] = FileReadFloat(handle);
      if(!MathIsValidNumber((double)values[i]))
         return(false);
     }
   for(int i = 0; i < buffer.Total(); i++)
      buffer.Update(i, values[i]);
   return(buffer.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Function NNInferenceRequired                                     |
//+------------------------------------------------------------------+
bool NNInferenceRequired(CBufferFloat *buffer, COpenCLMy *receiver, const int count);
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(NNInferenceHeader(handle, restore, (uint)bTrain) && NNInferenceHeader(handle, restore, (uint)Type())
          && NNInferenceHeader(handle, restore, (uint)Neurons()) && NNInferenceRequired(Output, OpenCL, Neurons())
          && NNInferenceRequired(PrevOutput, OpenCL, Neurons()) && CheckPointer(Gradient) != POINTER_INVALID
          && NNInferenceRequired(Gradient, OpenCL, Gradient.Total()) && NNInferenceOptionalFinite(Weights)
          && NNInferenceOptionalFinite(DeltaWeights) && NNInferenceOptionalFinite(FirstMomentum)
          && NNInferenceOptionalFinite(SecondMomentum));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 101)
          && cLevels.InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 102)
          && cGates.InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 103)
          && cHV.Neurons() == Neurons() && NNInferenceOmittedBuffer(handle, restore, cHV.getOutput())
          && NNInferenceOmittedBuffer(handle, restore, cHV.getPrevOutput()));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 201)
          && cActivation.InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 202)
          && cConv.InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 203)
          && cStack.InferenceRuntimeSidecar(handle, restore));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronCogDriverData::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   if(iWindow == 0 || iVariables == 0 || iQuantiles == 0 || (ulong)iStateStackSize < 10 * (ulong)iQuantiles
      || iQuantStackSize < 2 || !MathIsValidNumber((double)fRange) || fRange <= 0)
      return(false);
   const ulong raw = (ulong)iWindow * iVariables, bins = 2 * (ulong)iQuantiles + 1;
   if(raw > (ulong)INT_MAX || bins > (ulong)INT_MAX || raw * bins > (ulong)INT_MAX || (ulong)iStateStackSize * raw > (ulong)INT_MAX
      || (ulong)iQuantStackSize * raw * bins > (ulong)INT_MAX || cStateStack.Neurons() != (int)((ulong)iStateStackSize * raw)
      || cQuantStack.Neurons() != (int)((ulong)iQuantStackSize * raw * bins)
      || cPrevQuantStack.Neurons() != cQuantStack.Neurons())
      return(false);
   return(CNeuronSpikeConvBlock::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, iWindow)
          && NNInferenceHeader(handle, restore, iVariables) && NNInferenceHeader(handle, restore, iStateStackSize)
          && NNInferenceHeader(handle, restore, iQuantStackSize) && NNInferenceHeader(handle, restore, iQuantiles)
          && NNInferenceHeader(handle, restore, 301) && cNorm.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 302) && cStateStack.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 303) && cMSD.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 304) && cAffine.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 305) && cQuantStack.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 306) && cPrevQuantStack.InferenceRuntimeSidecar(handle, restore));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronCogDriverRankTCM::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   if(iInputDim == 0 || iQueryDim == 0 || iVariables == 0 || iHeads == 0 || iMemoryLen == 0 || iQueryDim % iHeads != 0
      || iHeadDim != iQueryDim / iHeads || (ulong)iVariables * iQueryDim != iTotalQueryDim
      || (ulong)iVariables * iHeads != iTotalHeads || iMemoryCount > iMemoryLen
      || (ulong)iMemoryLen * iTotalQueryDim > (ulong)INT_MAX
      || cMemoryQuery.Total() != (int)((ulong)iMemoryLen * iTotalQueryDim) || (ulong)iMemoryLen * iVariables > (ulong)INT_MAX
      || cMemoryAge.Total() != (int)((ulong)iMemoryLen * iVariables) || cMemoryScore.Total() != cMemoryAge.Total()
      || cMemoryValid.Total() != cMemoryAge.Total())
      return(false);
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, iInputDim)
          && NNInferenceHeader(handle, restore, iQueryDim) && NNInferenceHeader(handle, restore, iVariables)
          && NNInferenceHeader(handle, restore, iHeads) && NNInferenceHeader(handle, restore, iMemoryLen)
          && NNInferenceHeader(handle, restore, iMemoryCount) && NNInferenceHeader(handle, restore, (uint)bLastQueryValid)
          && NNInferenceHeader(handle, restore, 401) && cQueryProjection.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 402) && cGateInput.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 403) && cGateProjection.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 404) && cMemoryContext.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceHeader(handle, restore, 405) && cSelfContext.InferenceRuntimeSidecar(handle, restore)
          && NNInferenceFinite(GetPointer(cMemoryQuery)) && NNInferenceFinite(GetPointer(cMemoryAge))
          && NNInferenceFinite(GetPointer(cMemoryScore)) && NNInferenceFinite(GetPointer(cMemoryValid)));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronOMPBOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   if(bCurrentTransactionPending || cReference == NULL || cCurrent == NULL || iReferenceCount > iReferenceCapacity
      || iCurrentCount > iCurrentCapacity)
      return(false);
   if(!CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) || !NNInferenceHeader(handle, restore, iDimension)
      || !NNInferenceHeader(handle, restore, iVariables) || !NNInferenceHeader(handle, restore, iSamples)
      || !NNInferenceHeader(handle, restore, iReferenceCapacity)
      || !NNInferenceHeader(handle, restore, iCurrentCapacity) || !NNInferenceHeader(handle, restore, iReferenceCount)
      || !NNInferenceHeader(handle, restore, iCurrentCount))
      return(false);
   uint mode = (uint)eMode, anchor = (uint)bSourceAnchor;
   if(restore)
     {
      if(!NNInferenceRemaining(handle, 8))
         return(false);
      mode = (uint)FileReadInteger(handle);
      anchor = (uint)FileReadInteger(handle);
      if(mode != (uint)OMPB_INFERENCE || anchor > 1)
         return(false);
     }
   else
      if(mode != (uint)OMPB_INFERENCE || FileWriteInteger(handle, (int)mode) != 4
         || FileWriteInteger(handle, (int)anchor) != 4)
         return(false);
   if(!NNInferenceHeader(handle, restore, 501) || !cResidual.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 502) || !cGate.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 503) || !cReference.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 504) || !cCurrent.InferenceRuntimeSidecar(handle, restore))
      return(false);
   if(!NNInferenceHeader(handle, restore, 505) || !cAlpha.InferenceRuntimeSidecar(handle, restore))
      return(false);
   if(restore)
     {
      eMode = (ENUM_OMPB_MODE)mode;
      bSourceAnchor = (anchor == 1);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronScenarioForecast::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   if(!restore && bCodebookUpdate)
      return(false);
   const ulong vh = (ulong)iVariables * iHorizon;
   if(iVariables == 0 || iHorizon == 0 || iDimension == 0 || iScenarios == 0 || iTopK == 0 || iTopK > iScenarios
      || vh > (ulong)INT_MAX)
      return(false);
   const ulong vhd = vh * iDimension;
   if(vhd > (ulong)INT_MAX || (ulong)iScenarios * vhd > (ulong)INT_MAX / 2)
      return(false);
   const int conf = (int)((ulong)iScenarios * vh), latent = (int)((ulong)iScenarios * vhd), target = (int)vhd;
   if(!NNInferenceRequired(Output, OpenCL, latent) || !NNInferenceRequired(GetPointer(cConfidence), OpenCL, conf)
      || !NNInferenceRequired(GetPointer(cLogConfidence), OpenCL, conf)
      || !NNInferenceRequired(GetPointer(cLogProbabilities), OpenCL, (int)iScenarios)
      || !NNInferenceRequired(GetPointer(cResponsibilities), OpenCL, (int)iScenarios)
      || !NNInferenceRequired(GetPointer(cTarget), OpenCL, target) || cGeneratorInput.Neurons() != 2 * latent
      || cConfidenceInput.Neurons() != 2 * latent
      || cRouterInput.Neurons() != (int)(2 * (ulong)iScenarios * iVariables * iDimension) || cGenerator.Neurons() != latent
      || cConfidenceHead.Neurons() != conf || cRouter.Neurons() != (int)iScenarios
      || cRouterSoftmax.Neurons() != (int)iScenarios || cResponsibilityScores.Neurons() != (int)iScenarios
      || cTopKSelector.Neurons() != (int)iTopK)
      return(false);
   if(!CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) || !NNInferenceHeader(handle, restore, iVariables)
      || !NNInferenceHeader(handle, restore, iScenarios) || !NNInferenceHeader(handle, restore, iTopK)
      || !NNInferenceHeader(handle, restore, iHorizon) || !NNInferenceHeader(handle, restore, iDimension)
      || !NNInferenceHeader(handle, restore, 600) || !cGeneratorInput.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 601) || !cRouterInput.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 602) || !cConfidenceInput.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 603) || !cGenerator.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 604) || !cRouter.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 605) || !cConfidenceHead.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 606) || !cRouterSoftmax.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 607) || !cResponsibilityScores.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceHeader(handle, restore, 608) || !cTopKSelector.InferenceRuntimeSidecar(handle, restore)
      || !NNInferenceFinite(cCodebook.GetPrototypes()) || !NNInferenceFinite(cCodebook.GetEMASums())
      || !NNInferenceFinite(cCodebook.GetEMACounts()) || !NNInferenceFinite(cCodebook.GetUsage())
      || !NNInferenceFinite(cCodebook.GetInactivityAge()) || !NNInferenceFinite(GetPointer(cConfidence))
      || !NNInferenceFinite(GetPointer(cLogConfidence)) || !NNInferenceFinite(GetPointer(cLogProbabilities))
      || !NNInferenceFinite(GetPointer(cResponsibilities)) || !NNInferenceFinite(GetPointer(cTarget)))
      return(false);
   if(restore)
      bCodebookUpdate = false;
   return(true);
  }
//+------------------------------------------------------------------+
//| InferenceSnapshotGraph                                           |
//+------------------------------------------------------------------+
bool CNet::InferenceSnapshotGraph(void) const
  {
   if(!layers || (layers.Total() != 5 && layers.Total() != 7) || !opencl || !bOwnOpenCL || pendingOpenCL != NULL)
      return(false);
   const int n = layers.Total();
   const int tail[] =
     {
      defNeuronCogDriverData, defNeuronCogDriverRankTCM, defNeuronOMPBOCL, defNeuronScenarioForecast
     }
   ;
   for(int i = 0; i < n; i++)
     {
      CLayer *layer = layers.At(i);
      if(!layer || layer.Total() != 1)
         return(false);
      CNeuronBaseOCL *node = layer.At(0);
      if(!node || node.TrainMode())
         return(false);
      const int type = node.Type();
      if(i >= n - 4)
        {
         const int expected = tail[i - (n - 4)];
         if(type != expected &&
            !(expected == defNeuronScenarioForecast && type == defNeuronScenarioForecastPriceOCL))
            return(false);
        }
      else
         if(type != (i == 0 ? defNeuronBaseOCL : (i == 1 ? defNeuronBatchNormOCL : defNeuronDropoutOCL)))
            return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| SaveInferenceRuntimeSidecar                                      |
//+------------------------------------------------------------------+
bool CNet::SaveInferenceRuntimeSidecar(const int handle)
  {
   if(m_inference_load_active || bWeightsUpdate || !InferenceSnapshotGraph()
      || !NNInferenceHeader(handle, false, 0x49534631) || !NNInferenceHeader(handle, false, (uint)layers.Total()))
      return(false);
   for(int i = 0; i < layers.Total(); i++)
     {
      CLayer *layer = layers.At(i);
      CNeuronBaseOCL *node = layer.At(0);
      if(!NNInferenceHeader(handle, false, (uint)i) || !node.InferenceRuntimeSidecar(handle, false))
         return(false);
     }
   return(NNInferenceHeader(handle, false, 0x454e4431));
  }
//+------------------------------------------------------------------+
//| InferenceSnapshotLoadFile                                        |
//+------------------------------------------------------------------+
bool CNet::InferenceSnapshotLoadFile(const string file_name, const bool common)
  {
   const int handle = FileOpen(file_name, FILE_READ | FILE_BIN | (common ? FILE_COMMON : 0));
   if(handle == INVALID_HANDLE)
      return(false);
   bool accepted = false;
   do
     {
      if(FileSize(handle) < 48)
         break;
      const double error = FileReadDouble(handle), undefined = FileReadDouble(handle), forecast = FileReadDouble(handle);
      const long stamp = FileReadLong(handle);
      if(!MathIsValidNumber(error) || !MathIsValidNumber(undefined) || !MathIsValidNumber(forecast) || stamp <= 0
         || FileReadLong(handle) != -1 || FileReadInteger(handle) != layers.Type())
         break;
      const int count = FileReadInteger(handle);
      if((count != 5 && count != 7))
         break;
      int loaded = 0;
      for(int i = 0; i < count; i++)
        {
         CLayer *layer = new CLayer(0, handle, opencl);
         if(layer == NULL)
            break;
         const ulong position = FileTell(handle);
         const int tail[] =
           {
            defNeuronCogDriverData, defNeuronCogDriverRankTCM, defNeuronOMPBOCL, defNeuronScenarioForecast
           }
         ;
         const int expected = (i >= count - 4 ? tail[i - (count - 4)] : (i == 0 ? defNeuronBaseOCL : (i == 1 ? defNeuronBatchNormOCL : defNeuronDropoutOCL)));
         bool framed = (NNInferenceRemaining(handle, 20) && FileReadLong(handle) == -1
                        && FileReadInteger(handle) == layer.Type() && FileReadInteger(handle) == 1);
         if(framed)
           {
            const int actual = FileReadInteger(handle);
            framed = (actual == expected ||
                      (expected == defNeuronScenarioForecast && actual == defNeuronScenarioForecastPriceOCL));
           }
         if(!FileSeek(handle, (long)position, SEEK_SET) || !framed || !layer.Load(handle) || layer.Total() != 1
            || !layers.Add(layer))
           {
            delete layer;
            break;
           }
         loaded++;
        }
      if(loaded != count || FileTell(handle) != FileSize(handle) || !InferenceSnapshotGraph())
         break;
      recentAverageError = (float)error;
      accepted = true;
     }
   while(false);
   FileClose(handle);
   return(accepted);
  }
//+------------------------------------------------------------------+
//| LoadInferenceRuntimeSnapshot                                     |
//+------------------------------------------------------------------+
bool CNet::LoadInferenceRuntimeSnapshot(const string file_name, const int sidecar_handle, const bool common)
  {
   if(m_inference_load_active || pendingOpenCL != NULL || (layers && layers.Total() != 0) || (opencl && !bOwnOpenCL)
      || sidecar_handle == INVALID_HANDLE)
      return(false);
   if(!layers)
      layers = new CArrayLayer();
   if(!layers)
      return(false);
   if(!opencl && !OpenCLInit())
      return(false);
   m_inference_load_active = true;
   bool accepted = InferenceSnapshotLoadFile(file_name, common);
   if(accepted)
      accepted = TrainMode(false) && SetWeightsUpdate(false) && NNInferenceHeader(sidecar_handle, true, 0x49534631)
                 && NNInferenceHeader(sidecar_handle, true, (uint)layers.Total());
   if(accepted)
     {
      for(int i = 0; i < layers.Total(); i++)
        {
         CLayer *layer = layers.At(i);
         CNeuronBaseOCL *node = layer.At(0);
         if(!NNInferenceHeader(sidecar_handle, true, (uint)i) || !node.InferenceRuntimeSidecar(sidecar_handle, true))
           {
            accepted = false;
            break;
           }
        }
     }
   if(accepted)
      accepted = NNInferenceHeader(sidecar_handle, true, 0x454e4431)
                 && FileTell(sidecar_handle) == FileSize(sidecar_handle);
   m_inference_load_active = false;
   if(!accepted)
      layers.Clear();
   return(accepted);
  }
//+------------------------------------------------------------------+
//| LoadInferenceFinite                                              |
//+------------------------------------------------------------------+
bool CBufferFloat::LoadInferenceFinite(const int handle, COpenCLMy *receiver)
  {
   if(!receiver || !receiver.InferenceRuntimeLoadActive() || handle == INVALID_HANDLE)
      return(false);
   if(!NNInferenceRemaining(handle, 16))
      return(false);
   const ulong position = FileTell(handle), remaining = FileSize(handle) - position;
   const long marker = FileReadLong(handle);
   const int type = FileReadInteger(handle), count = FileReadInteger(handle);
   if(!FileSeek(handle, (long)position, SEEK_SET) || marker != -1 || type != Type() || count < 0 || (ulong)count > 268435456
      || (ulong)count * 4 > remaining - 16)
      return(false);
   ResetLastError();
   if(!CArrayFloat::Load(handle) || GetLastError() != 0 || m_data_total != count)
      return(false);
   for(int i = 0; i < count; i++)
      if(!MathIsValidNumber((double)m_data[i]))
         return(false);
   BufferFree();
   if(count > 0 && !BufferCreate(receiver))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
//| NNInferenceRequired                                              |
//+------------------------------------------------------------------+
bool NNInferenceRequired(CBufferFloat *buffer, COpenCLMy *receiver, const int count)
  {
   return(buffer != NULL && CheckPointer(buffer) != POINTER_INVALID && CheckPointer(receiver) != POINTER_INVALID
          && count > 0 && buffer.Total() == count && buffer.GetOpenCL() == receiver
          && buffer.GetIndex() >= 0 && receiver.GetBufferSize(buffer.GetIndex()) == (ulong)count * 4
          && NNInferenceFinite(buffer));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(!bTrain && MathIsValidNumber((double)OutProbability) && OutProbability >= 0 && OutProbability < 1
          && CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore)
          && CheckPointer(WeightsConv) != POINTER_INVALID && NNInferenceRequired(WeightsConv, OpenCL, WeightsConv.Total())
          && NNInferenceFinite(DeltaWeightsConv)
          && NNInferenceFinite(FirstMomentumConv) && NNInferenceFinite(SecondMomentumConv));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore)
          && CheckPointer(BatchOptions) != POINTER_INVALID
          && NNInferenceRequired(BatchOptions, OpenCL, BatchOptions.Total()));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CParams::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 701)
          && cOne.InferenceRuntimeSidecar(handle, restore));
  }
//+------------------------------------------------------------------+
//| InferenceRuntimeSidecar                                          |
//+------------------------------------------------------------------+
bool CNeuronBayesianLinearOCL::InferenceRuntimeSidecar(const int handle, const bool restore)
  {
   return(CNeuronBaseOCL::InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, iVariables)
          && NNInferenceHeader(handle, restore, iInputs) && NNInferenceHeader(handle, restore, iOutputs)
          && NNInferenceHeader(handle, restore, iSamples) && NNInferenceHeader(handle, restore, 801)
          && cWeights.InferenceRuntimeSidecar(handle, restore) && NNInferenceHeader(handle, restore, 802)
          && cBias.InferenceRuntimeSidecar(handle, restore));
  }
#endif
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
