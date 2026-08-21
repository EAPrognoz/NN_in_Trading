#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh: SpikeFlow
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDecouplingFlow::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                 uint window, uint step, uint units_count,
                                 uint filters, uint variables,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint units_out = (units_count - window + step) / step;
   if(!CNeuronLegendreWaveletsHL::Init(numOutputs, myIndex, open_cl, window, step, units_out, filters, variables, optimization_type, batch))
      ReturnFalse;
   if(!cTranspose.Init(0, 0, OpenCL, units_count, variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDecouplingFlow::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronLegendreWaveletsHL::feedForward(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDecouplingFlow::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronLegendreWaveletsHL::calcInputGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDecouplingFlow::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronLegendreWaveletsHL::SetOpenCL(obj);
   cTranspose.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDecouplingFlow::Save(const int file_handle)
  {
   if(!CNeuronLegendreWaveletsHL::Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDecouplingFlow::Load(const int file_handle)
  {
   if(!CNeuronLegendreWaveletsHL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint window, uint step, uint window_out, uint units_count,
                            uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units_count * variables, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   if(!cConv.Init(0, 0, OpenCL, window, step, window_out, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cConv.SetActivationFunction(TANH);
   if(!cLevels.Init(0, 1, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cLevels.SetActivationFunction(SIGMOID);
   CBufferFloat* temp = cLevels.getWeightsParams();
   if(!temp)
      ReturnFalse;
   if(!temp.Fill(-4))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::FloatToSpike(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = { Neurons() };
   uint kernel = def_k_FloatToSpike;
   setBuffer(kernel, def_k_fts_values, cConv.getOutputIndex())
   setBuffer(kernel, def_k_fts_levels, cLevels.getOutputIndex())
   setBuffer(kernel, def_k_fts_outputs, getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
//---
#ifdef _DEBUG
   cConv.getOutput().BufferRead();
   cLevels.getOutput().BufferRead();
   Output.BufferRead();
   Sleep(0);
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::FloatToSpikeGrad(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = { Neurons() };
   uint kernel = def_k_FloatToSpikeGrad;
   setBuffer(kernel, def_k_ftsg_values, cConv.getOutputIndex())
   setBuffer(kernel, def_k_ftsg_values_gr, cConv.getGradientIndex())
   setBuffer(kernel, def_k_ftsg_levels_gr, cLevels.getGradientIndex())
   setBuffer(kernel, def_k_ftsg_gradients, getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cConv.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cConv.FeedForward(NeuronOCL))
      ReturnFalse;
   if(bTrain)
      if(!cLevels.FeedForward())
         ReturnFalse;
//---
   return FloatToSpike();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!FloatToSpikeGrad())
      ReturnFalse;
   Deactivation(cConv)
   Deactivation(cLevels)
   if(!NeuronOCL.CalcHiddenGradients(cConv.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cConv.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cLevels.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeConv* Source = source;
   if(!cConv.WeightsUpdate(Source.cConv.AsObject(), tau))
      ReturnFalse;
   if(!cLevels.WeightsUpdate(Source.cLevels.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cConv.Save(file_handle))
      ReturnFalse;
   if(!cLevels.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cConv.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLevels.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cConv.SetOpenCL(OpenCL);
   cLevels.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint units, uint window, uint experts, float dropout,
                                       uint emb_dimension, uint sparse_dimension,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConv::Init(numOutputs, myIndex, open_cl, window, window, window,
                              units, 1, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cValue.Init(0, index, OpenCL, window, window, window, units, 1, optimization, iBatch))
      ReturnFalse;
   cValue.SetActivationFunction(None);
   index++;
   if(!cGraphs.Init(0, index, OpenCL, units, window, emb_dimension, experts, dropout, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cScores.Init(0, index, OpenCL, units, units, sparse_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAttention.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cAttention.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cValue.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cGraphs.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cScores.FeedForward(cGraphs.AsObject()))
      ReturnFalse;
   if(!SparseMatMul(cScores.GetIndexes(), cScores.getOutput(), cValue.getOutput(),
                    cAttention.getOutput(), cScores.Heads(), cScores.DimensionOut(),
                    cValue.GetUnits(), cValue.GetFilters()))
      ReturnFalse;
//---
   return CNeuronSpikeConv::feedForward(cAttention.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConv::calcInputGradients(cAttention.AsObject()))
      ReturnFalse;
   if(!SparseMatMulGrad(cScores.GetIndexes(), cScores.getOutput(), cScores.getGradient(),
                        cValue.getOutput(), cValue.getGradient(), cAttention.getGradient(),
                        cScores.Heads(), cScores.DimensionOut(),
                        cValue.GetUnits(), cValue.GetFilters()))
      ReturnFalse;
   if(!cGraphs.CalcHiddenGradients(cScores.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cGraphs.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(PrevOutput, false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cValue.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(PrevOutput, temp, temp, GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cValue.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cGraphs.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeConv::updateInputWeights(cAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConv::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSparseAttention* Source = source;
   if(!cValue.WeightsUpdate(Source.cValue.AsObject(), tau))
      ReturnFalse;
   if(!cGraphs.WeightsUpdate(Source.cGraphs.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::Save(const int file_handle)
  {
   if(!CNeuronSpikeConv::Save(file_handle))
      ReturnFalse;
   if(!cValue.Save(file_handle))
      ReturnFalse;
   if(!cGraphs.Save(file_handle))
      ReturnFalse;
   if(!cScores.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSparseAttention::Load(const int file_handle)
  {
   if(!CNeuronSpikeConv::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cValue.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGraphs.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScores.AsObject()))
      ReturnFalse;
//---
   int index = 3;
   if(!cAttention.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cAttention.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSparseAttention::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConv::SetOpenCL(obj);
//---
   cValue.SetOpenCL(OpenCL);
   cGraphs.SetOpenCL(OpenCL);
   cScores.SetOpenCL(OpenCL);
   cAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSparseAttention::TrainMode(bool flag)
  {
   CNeuronSpikeConv::TrainMode(flag);
//---
   cValue.TrainMode(bTrain);
   cGraphs.TrainMode(bTrain);
   cScores.TrainMode(bTrain);
   cAttention.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint units, uint window, uint experts, float dropout,
                               uint emb_dimension, uint sparse_dimension,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cTimeAttention.Init(0, index, OpenCL, units, window, experts, dropout, emb_dimension,
                           sparse_dimension, optimization, iBatch))
      ReturnFalse;
//---
   CNeuronTransposeOCL* transp = NULL;
   CNeuronGateLineAttention* line_att = NULL;
   caSpatialAttention.Clear();
   caSpatialAttention.SetOpenCL(OpenCL);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units, window, optimization, iBatch) ||
      !caSpatialAttention.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
   line_att = new CNeuronGateLineAttention();
   if(!line_att ||
      !line_att.Init(0, index, OpenCL, window, units, emb_dimension, experts, optimization, iBatch) ||
      !caSpatialAttention.Add(line_att))
     {
      DeleteObj(line_att)
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, window, units, optimization, iBatch) ||
      !caSpatialAttention.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
   if(!cResidual.Init(0, index, OpenCL, window * units, optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   CNeuronSpikeConv* conv = NULL;
   CNeuronMoEConv* moe = NULL;
   caFFN.Clear();
   caFFN.SetOpenCL(OpenCL);
   index++;
   moe = new CNeuronMoEConv();
   if(!moe ||
      !moe.Init(0, index, OpenCL, units, window, 2 * window, experts, dropout, sparse_dimension, optimization, iBatch) ||
      !caFFN.Add(moe))
     {
      DeleteObj(moe)
      ReturnFalse;
     }
   moe.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronSpikeConv();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * window, 2 * window, window, units, 1, optimization, iBatch) ||
      !caFFN.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
//---
   if(!cTimeAttention.FeedForward(prev))
      ReturnFalse;
   for(int i = 0; i < caSpatialAttention.Total(); i++)
     {
      current = caSpatialAttention[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
   if(!SumAndNormalize(prev.getOutput(), cTimeAttention.getOutput(), cResidual.getOutput(),
                       cTimeAttention.GetFilters(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(cResidual.getOutput(), NeuronOCL.getOutput(), cResidual.getOutput(),
                       cTimeAttention.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   prev = cResidual.AsObject();
   for(int i = 0; i < caFFN.Total(); i++)
     {
      current = caFFN[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
   if(!SumAndNormalize(cResidual.getOutput(), prev.getOutput(), Output,
                       cTimeAttention.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* next = NULL;
   CNeuronBaseOCL* current = caFFN[-1];
   if(!current ||
      !DeActivation(current.getOutput(), current.getGradient(),
                    Gradient, current.Activation()))
      ReturnFalse;
   for(int i = caFFN.Total() - 1; i >= 0; i--)
     {
      next = caFFN[i];
      current = (i == 0 ? cResidual.AsObject() : caFFN[i - 1]);
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
     }
   if(!SumAndNormalize(cResidual.getGradient(), Gradient, cResidual.getGradient(),
                       cTimeAttention.GetFilters(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   current = caSpatialAttention[-1];
   if(!DeActivation(current.getOutput(), current.getGradient(),
                    cResidual.getGradient(), current.Activation()))
      ReturnFalse;
   if(!DeActivation(cTimeAttention.getOutput(), cTimeAttention.getGradient(),
                    cResidual.getGradient(), cTimeAttention.Activation()))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), cResidual.getPrevOutput(),
                    cResidual.getGradient(), NeuronOCL.Activation()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cTimeAttention.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cResidual.getPrevOutput(),
                       cResidual.getPrevOutput(), cTimeAttention.GetFilters(),
                       false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = caSpatialAttention.Total() - 1; i >= 0; i--)
     {
      next = caSpatialAttention[i];
      current = (i == 0 ? NeuronOCL : caSpatialAttention[i - 1]);
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
     }
   if(!SumAndNormalize(NeuronOCL.getGradient(), cResidual.getPrevOutput(),
                       NeuronOCL.getGradient(), cTimeAttention.GetFilters(),
                       false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
//---
   if(!cTimeAttention.UpdateInputWeights(prev))
      ReturnFalse;
   for(int i = 0; i < caSpatialAttention.Total(); i++)
     {
      current = caSpatialAttention[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = cResidual.AsObject();
   for(int i = 0; i < caFFN.Total(); i++)
     {
      current = caFFN[i];
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
bool CNeuronSpikingBrain::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikingBrain* Source = source;
   if(!cTimeAttention.WeightsUpdate(Source.cTimeAttention.AsObject(), tau))
      ReturnFalse;
   if(!caSpatialAttention.WeightsUpdate(Source.caSpatialAttention.AsObject(), tau))
      ReturnFalse;
   if(!caFFN.WeightsUpdate(Source.caFFN.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTimeAttention.Save(file_handle))
      ReturnFalse;
   if(!caSpatialAttention.Save(file_handle))
      ReturnFalse;
   if(!caFFN.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTimeAttention.AsObject()))
      ReturnFalse;
   if(!caSpatialAttention.Load(file_handle))
      ReturnFalse;
   if(!caFFN.Load(file_handle))
      ReturnFalse;
//---
   int index = caSpatialAttention.Total() + 1;
   if(!cResidual.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikingBrain::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cTimeAttention.SetOpenCL(OpenCL);
   caSpatialAttention.SetOpenCL(OpenCL);
   caFFN.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikingBrain::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cTimeAttention.TrainMode(bTrain);
   caSpatialAttention.TrainMode(bTrain);
   caFFN.TrainMode(bTrain);
   cResidual.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikingBrain::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cTimeAttention.Clear())
      ReturnFalse;
   for(int i = 0; i < caSpatialAttention.Total(); i++)
      if(!caSpatialAttention[i] || !caSpatialAttention[i].Clear())
         ReturnFalse;
   for(int i = 0; i < caFFN.Total(); i++)
      if(!caFFN[i] || !caFFN[i].Clear())
         ReturnFalse;
   if(!cResidual.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                  uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   uint index = 0;
   if(!cHV.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cHV.SetActivationFunction(None);
   cHV.Clear();
//---
   index++;
   if(!cLevels.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cLevels.SetActivationFunction(SoftPlus);
   CBufferFloat* temp = cLevels.getWeightsParams();
   if(!temp ||
      !temp.Fill(-3.0f))
      ReturnFalse;
//---
   index++;
   if(!cGates.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cGates.SetActivationFunction(SIGMOID);
   temp = cGates.getWeightsParams();
   if(!temp ||
      !temp.Fill(0.0f))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cHV.SwapOutputs())
      ReturnFalse;
   if(bTrain)
     {
      if(!cLevels.FeedForward())
         ReturnFalse;
      if(!cGates.FeedForward())
         ReturnFalse;
     }
//---
   if(!GateElementMult(NeuronOCL.getOutput(), cHV.getPrevOutput(),
                       cGates.getOutput(), cHV.getOutput()))
      ReturnFalse;
   return FloatToSpike();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!FloatToSpikeGrad())
      ReturnFalse;
//---
   if(!GateElementMultGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                           cHV.getPrevOutput(), cLevels.getPrevOutput(),
                           cGates.getOutput(), cGates.getGradient(),
                           cHV.getGradient(), NeuronOCL.Activation(),
                           cHV.Activation(), cGates.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::FloatToSpike(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = { Neurons() };
   uint kernel = def_k_FloatToSpike;
   setBuffer(kernel, def_k_fts_values, cHV.getOutputIndex())
   setBuffer(kernel, def_k_fts_levels, cLevels.getOutputIndex())
   setBuffer(kernel, def_k_fts_outputs, getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
//---
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
bool CNeuronSpikeActivation::FloatToSpikeGrad(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = { Neurons() };
   uint kernel = def_k_FloatToSpikeGrad;
   setBuffer(kernel, def_k_ftsg_values, cHV.getOutputIndex())
   setBuffer(kernel, def_k_ftsg_values_gr, cHV.getGradientIndex())
   setBuffer(kernel, def_k_ftsg_levels_gr, cLevels.getGradientIndex())
   setBuffer(kernel, def_k_ftsg_gradients, getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cHV.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cLevels.UpdateInputWeights())
      ReturnFalse;
   if(!cGates.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeActivation* Source = source;
   if(!cLevels.WeightsUpdate(Source.cLevels.AsObject(), tau))
      ReturnFalse;
   if(!cGates.WeightsUpdate(Source.cGates.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cLevels.Save(file_handle))
      ReturnFalse;
   if(!cGates.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLevels.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGates.AsObject()))
      ReturnFalse;
//---
   uint index = 0;
   if(!cHV.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cHV.SetActivationFunction(None);
   cHV.Clear();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeActivation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cLevels.SetOpenCL(OpenCL);
   cGates.SetOpenCL(OpenCL);
   cHV.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeActivation::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cLevels.TrainMode(bTrain);
   cGates.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeActivation::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cHV.getOutput().Fill(0.0f))
      ReturnFalse;
   if(!cHV.getGradient().Fill(0.0f))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint chanels_in, uint chanels_out, uint units_in,
      uint units_out, uint group_size, uint groups,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, chanels_out * units_out, optimization_type, batch))
      ReturnFalse;
//---
   CNeuronConvOCL *conv = NULL;
   CNeuronMultiWindowsConvOCL *mwconv = NULL;
   CNeuronBatchNormOCL *norm = NULL;
   CNeuronSpikeActivation *spike = NULL;
   CNeuronTransposeOCL* transp = NULL;
//---
   cBottleneck.Clear();
   cBottleneck.SetOpenCL(OpenCL);
//--- Projection in
   uint index = 0;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, chanels_in, chanels_in, group_size * groups, units_in, 1, optimization, iBatch) ||
      !cBottleneck.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cBottleneck.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   index++;
   spike = new CNeuronSpikeActivation();
   if(!spike ||
      !spike.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
      !cBottleneck.Add(spike))
     {
      DeleteObj(spike)
      ReturnFalse;
     }
//--- Feature Extraction
   index++;
   uint windows[];
   if(ArrayResize(windows, groups) < (int)groups)
      ReturnFalse;
   ArrayFill(windows, 0, groups, group_size);
   mwconv = new CNeuronMultiWindowsConvOCL();
   if(!mwconv ||
      !mwconv.Init(0, index, OpenCL, windows, group_size, 1, units_in, optimization, iBatch) ||
      !cBottleneck.Add(mwconv))
     {
      DeleteObj(mwconv)
      ReturnFalse;
     }
   mwconv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, mwconv.Neurons(), iBatch, optimization) ||
      !cBottleneck.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   index++;
   spike = new CNeuronSpikeActivation();
   if(!spike ||
      !spike.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
      !cBottleneck.Add(spike))
     {
      DeleteObj(spike)
      ReturnFalse;
     }
//--- Projection out
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, group_size * groups, group_size * groups, chanels_out, units_in, 1, optimization, iBatch) ||
      !cBottleneck.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cBottleneck.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   if(units_in == units_out)
      return true;
   index++;
   spike = new CNeuronSpikeActivation();
   if(!spike ||
      !spike.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
      !cBottleneck.Add(spike))
     {
      DeleteObj(spike)
      ReturnFalse;
     }
//--- Projection units
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, chanels_out, optimization, iBatch) ||
      !cBottleneck.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, units_out, chanels_out, 1, optimization, iBatch) ||
      !cBottleneck.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cBottleneck.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, chanels_out, units_out, optimization, iBatch) ||
      !cBottleneck.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
//---
   for(int i = 0; i < cBottleneck.Total(); i++)
     {
      current = cBottleneck[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL *current = (cBottleneck.Total() > 0 ? cBottleneck[-1] : NeuronOCL);
   if(!CNeuronSpikeActivation::calcInputGradients(current))
      ReturnFalse;
   for(int i = cBottleneck.Total() - 1; i >= 0; i--)
     {
      current = (i > 0 ? cBottleneck[i - 1] : NeuronOCL);
      if(!current ||
         !current.CalcHiddenGradients(cBottleneck[i]))
         ReturnFalse;
     }
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *current = NULL;
//---
   for(int i = 0; i < cBottleneck.Total(); i++)
     {
      current = cBottleneck[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
   if(!cBottleneck.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
   if(!cBottleneck.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeResNeXtBottleneck::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
   cBottleneck.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikeResNeXtBottleneck* Source = source;
   if(!cBottleneck.WeightsUpdate(Source.cBottleneck.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeResNeXtBottleneck::TrainMode(bool flag)
  {
   CNeuronSpikeActivation::TrainMode(flag);
   cBottleneck.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBottleneck::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
   for(int i = 0; i < cBottleneck.Total(); i++)
      if(!cBottleneck[i] ||
         !cBottleneck[i].Clear())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtResidual::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                       uint chanels_in, uint chanels_out, uint units_in,
                                       uint units_out, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, chanels_out * units_out, optimization_type, batch))
      ReturnFalse;
//---
   CNeuronConvOCL *conv = NULL;
   CNeuronBatchNormOCL *norm = NULL;
   CNeuronSpikeActivation *spike = NULL;
   CNeuronTransposeOCL* transp = NULL;
//---
   cBottleneck.Clear();
   cBottleneck.SetOpenCL(OpenCL);
//--- Projection Chanels
   uint index = 0;
   if(chanels_in != chanels_out)
     {
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, chanels_in, chanels_in, chanels_out, units_in, 1, optimization, iBatch) ||
         !cBottleneck.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
      index++;
     }
//--- Projection Units
   if(units_in != units_out)
     {
      if(chanels_in != chanels_out)
        {
         norm = new CNeuronBatchNormOCL();
         if(!norm ||
            !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
            !cBottleneck.Add(norm))
           {
            DeleteObj(norm)
            ReturnFalse;
           }
         norm.SetActivationFunction(None);
         index++;
         spike = new CNeuronSpikeActivation();
         if(!spike ||
            !spike.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
            !cBottleneck.Add(spike))
           {
            DeleteObj(spike)
            ReturnFalse;
           }
         index++;
        }
      transp = new CNeuronTransposeOCL();
      if(!transp ||
         !transp.Init(0, index, OpenCL, units_in, chanels_out, optimization, iBatch) ||
         !cBottleneck.Add(transp))
        {
         DeleteObj(transp)
         ReturnFalse;
        }
      index++;
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, units_in, units_in, units_out, chanels_out, 1, optimization, iBatch) ||
         !cBottleneck.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
      index++;
      transp = new CNeuronTransposeOCL();
      if(!transp ||
         !transp.Init(0, index, OpenCL, chanels_out, units_out, optimization, iBatch) ||
         !cBottleneck.Add(transp))
        {
         DeleteObj(transp)
         ReturnFalse;
        }
      index++;
     }
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, Neurons(), iBatch, optimization) ||
      !cBottleneck.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint chanels_in, uint chanels_out, uint units_in,
                                    uint units_out, uint group_size, uint groups,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, chanels_out * units_out,
                                    optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!caTranspose[0].Init(0, index, OpenCL, units_in, chanels_in, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!caTranspose[1].Init(0, index, OpenCL, chanels_out, units_out, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!caBottleneck[0].Init(0, index, OpenCL, chanels_in, chanels_out, units_in, units_out,
                            group_size, groups, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!caBottleneck[1].Init(0, index, OpenCL, units_in, units_out, chanels_in, chanels_out,
                            group_size, groups, optimization, iBatch))
      ReturnFalse;
//---
   index++;
   if(!cResidual.Init(0, index, OpenCL, chanels_in, chanels_out, units_in, units_out,
                      optimization, iBatch))
      ReturnFalse;
//---
   CNeuronBaseOCL*      neuron = NULL;
   CNeuronBatchNormOCL* norm = NULL;
   cG.Clear();
   cG.SetOpenCL(OpenCL);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, Neurons(), optimization, iBatch) ||
      !cG.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
//---
   if(!cResidual.SetGradient(neuron.getGradient(), true) ||
      !caTranspose[1].SetGradient(neuron.getGradient(), true) ||
      !caBottleneck[0].SetGradient(neuron.getGradient(), true))
      ReturnFalse;
//---
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, Neurons(), iBatch, optimization) ||
      !cG.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!caTranspose[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[1].FeedForward(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!caTranspose[1].FeedForward(caBottleneck[1].AsObject()))
      ReturnFalse;
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = cG[0];
   if(!current)
      ReturnFalse;
   if(!SumAndNormalize(caBottleneck[0].getOutput(), caTranspose[1].getOutput(),
                       current.getOutput(), caTranspose[1].GetCount(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(current.getOutput(), cResidual.getOutput(),
                       current.getOutput(), caTranspose[1].GetCount(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.FeedForward(cG[i - 1]))
         ReturnFalse;
     }
//---
   return CNeuronSpikeActivation::feedForward(cG[-1]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cG[-1]))
      ReturnFalse;
   CNeuronBaseOCL* current = NULL;
   for(int i = cG.Total() - 2; i >= 0; i--)
     {
      current = cG[i];
      if(!current ||
         !current.CalcHiddenGradients(cG[i + 1]))
         ReturnFalse;
     }
//---
   if(!caBottleneck[1].CalcHiddenGradients(caTranspose[1].AsObject()))
      ReturnFalse;
   if(!caTranspose[0].CalcHiddenGradients(caBottleneck[1].AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(caTranspose[0].getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(caBottleneck[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp,
                       caTranspose[0].GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp,
                       caTranspose[0].GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!caBottleneck[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[1].UpdateInputWeights(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.UpdateInputWeights(cG[i - 1]))
         ReturnFalse;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(cG[-1]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeResNeXtBlock* Source = source;
   for(uint i = 0; i < caBottleneck.Size(); i++)
      if(!caBottleneck[i].WeightsUpdate(Source.caBottleneck[i].AsObject(), tau))
         ReturnFalse;
   if(!cResidual.WeightsUpdate(Source.cResidual.AsObject(), tau))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   for(int i = 0; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.WeightsUpdate(Source.cG[i], tau))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caBottleneck.Size(); i++)
      if(!caBottleneck[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < caTranspose.Size(); i++)
      if(!caTranspose[i].Save(file_handle))
         ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
   if(!cG.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caBottleneck.Size(); i++)
      if(!LoadInsideLayer(file_handle, caBottleneck[i].AsObject()))
         ReturnFalse;
   for(uint i = 0; i < caTranspose.Size(); i++)
      if(!LoadInsideLayer(file_handle, caTranspose[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
   if(!cG.Load(file_handle))
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = cG[0];
   if(!neuron ||
      !cResidual.SetGradient(neuron.getGradient(), true) ||
      !caTranspose[1].SetGradient(neuron.getGradient(), true) ||
      !caBottleneck[0].SetGradient(neuron.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeResNeXtBlock::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
//---
   for(uint i = 0; i < caBottleneck.Size(); i++)
      if(!caBottleneck[i].Clear())
         ReturnFalse;
   for(uint i = 0; i < caTranspose.Size(); i++)
      if(!caTranspose[i].Clear())
         ReturnFalse;
   if(!cResidual.Clear())
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   for(int i = 0; i < cG.Total(); i++)
     {
      current = cG[i];
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
void CNeuronSpikeResNeXtBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
//---
   for(uint i = 0; i < caBottleneck.Size(); i++)
      caBottleneck[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < caTranspose.Size(); i++)
      caTranspose[i].SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cG.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeResNeXtBlock::TrainMode(bool flag)
  {
   CNeuronSpikeActivation::TrainMode(flag);
//---
   for(uint i = 0; i < caBottleneck.Size(); i++)
      caBottleneck[i].TrainMode(bTrain);
   for(uint i = 0; i < caTranspose.Size(); i++)
      caTranspose[i].TrainMode(bTrain);
   cResidual.TrainMode(bTrain);
   cG.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::SpikeMHAttention(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = { iUnits, iUnits, iHeads };
   uint local_work_size[3] = { 1, iUnits, 1 };
   uint kernel = def_k_SpikeMHAttention;
   setBuffer(kernel, def_k_smhat_qkv, cQKV.getOutputIndex())
   setBuffer(kernel, def_k_smhat_diag_bias, cDiaganalBias.getOutputIndex())
   setBuffer(kernel, def_k_smhat_scores, ciScore)
   setBuffer(kernel, def_k_smhat_out, cMHAttentionOut.getOutputIndex())
   setArgument(kernel, def_k_smhat_dimension, int(iDimension))
   setArgument(kernel, def_k_smhat_mask_future, int(bMaskFuture))
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
bool CNeuronSSAM::SpikeMHAttentionGrad(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = { 0 };
   uint global_work_size[3] = { iUnits, iUnits, iHeads };
   uint local_work_size[3] = { 1, iUnits, 1 };
   uint kernel = def_k_SpikeMHAttentionGrad;
   setBuffer(kernel, def_k_smhat_gr_qkv, cQKV.getOutputIndex())
   setBuffer(kernel, def_k_smhat_gr_qkv_gr, cQKV.getGradientIndex())
   setBuffer(kernel, def_k_smhat_gr_diag_bias, cDiaganalBias.getOutputIndex())
   setBuffer(kernel, def_k_smhat_gr_diag_bias_gr, cDiaganalBias.getGradientIndex())
   setBuffer(kernel, def_k_smhat_gr_scores, ciScore)
   setBuffer(kernel, def_k_smhat_gr_gradients, cMHAttentionOut.getGradientIndex())
   setArgument(kernel, def_k_smhat_gr_dimension, int(iDimension))
   setArgument(kernel, def_k_smhat_gr_mask_future, int(bMaskFuture))
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cQKV.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint units, uint window, uint heads, uint dimension_k,
                       bool mask_future, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units * window, optimization_type, batch))
      ReturnFalse;
//---
   iHeads = (uint)MathMax(heads, 1);
   iDimension = (uint)MathMax(dimension_k, 1);
   iUnits = units;
   bMaskFuture = mask_future;
//---
   uint index = 0;
   if(!cQKV.Init(0, index, OpenCL, window, window, 3 * iHeads * iDimension, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cDiaganalBias.Init(0, index, OpenCL, iUnits, optimization, iBatch))
      ReturnFalse;
   cDiaganalBias.SetActivationFunction(None);
   CBufferFloat* temp = cDiaganalBias.getWeightsParams();
   if(!temp ||
      !temp.Fill(0))
      ReturnFalse;
   ciScore = OpenCL.AddBuffer(sizeof(float) * iUnits * iUnits * iHeads, CL_MEM_READ_WRITE);
   if(ciScore == INVALID_HANDLE)
      ReturnFalse;
   index++;
   if(!cMHAttentionOut.Init(0, index, OpenCL, iUnits * iDimension * iHeads, optimization, iBatch))
      ReturnFalse;
   cMHAttentionOut.SetActivationFunction(None);
   index++;
   if(!cW0.Init(0, index, OpenCL, iHeads * iDimension, iHeads * iDimension, window, iUnits, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cResidual.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
   if(!cResidual.SetGradient(cW0.getGradient(), true))
      ReturnFalse;
   index++;
   if(!cNorm.Init(0, index, OpenCL, Neurons(), iBatch, optimization))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cQKV.FeedForward(NeuronOCL))
      ReturnFalse;
   if(bTrain)
      if(!cDiaganalBias.FeedForward())
         ReturnFalse;
   if(!SpikeMHAttention())
      ReturnFalse;
   if(!cW0.FeedForward(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), cW0.getOutput(), cResidual.getOutput(),
                       Neurons() / iUnits, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!cNorm.FeedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return CNeuronSpikeActivation::feedForward(cNorm.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeActivation::calcInputGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cResidual.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!SpikeMHAttentionGrad())
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cQKV.AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() == None)
     {
      if(!SumAndNormalize(NeuronOCL.getGradient(), cResidual.getGradient(),
                          NeuronOCL.getGradient(), Neurons() / iUnits, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   else
     {
      if(!DeActivation(NeuronOCL.getOutput(), cResidual.getPrevOutput(),
                       cResidual.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
      if(!SumAndNormalize(NeuronOCL.getGradient(), cResidual.getPrevOutput(),
                          NeuronOCL.getGradient(), Neurons() / iUnits, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cQKV.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cDiaganalBias.UpdateInputWeights())
      ReturnFalse;
   if(!cW0.UpdateInputWeights(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cResidual.AsObject()))
      ReturnFalse;
//---
   return CNeuronSpikeActivation::updateInputWeights(cNorm.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSSAM* Source = source;
   if(!cQKV.WeightsUpdate(Source.cQKV.AsObject(), tau))
      ReturnFalse;
   if(!cDiaganalBias.WeightsUpdate(Source.cDiaganalBias.AsObject(), tau))
      ReturnFalse;
   if(!cW0.WeightsUpdate(Source.cW0.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
//---
   if(!cQKV.Save(file_handle))
      ReturnFalse;
   if(!cDiaganalBias.Save(file_handle))
      ReturnFalse;
   if(!cW0.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iUnits) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iHeads) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iDimension) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)bMaskFuture) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cQKV.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDiaganalBias.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cW0.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iUnits = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHeads = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimension = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   bMaskFuture = (bool)FileReadInteger(file_handle);
//---
   uint index = 4;
   if(!cResidual.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
   if(!cW0.SetGradient(cResidual.getGradient(), true))
      ReturnFalse;
   if(ciScore != INVALID_HANDLE)
      OpenCL.BufferFree(ciScore);
   ciScore = OpenCL.AddBuffer(sizeof(float) * iUnits * iUnits * iHeads, CL_MEM_READ_WRITE);
   if(ciScore == INVALID_HANDLE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAM::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
//---
   if(!cQKV.Clear())
      ReturnFalse;
   if(!cW0.Clear())
      ReturnFalse;
   if(!cNorm.Clear())
      ReturnFalse;
   if(!cResidual.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSSAM::SetOpenCL(COpenCLMy *obj)
  {
   if(OpenCL && ciScore != INVALID_HANDLE)
      OpenCL.BufferFree(ciScore);
//---
   CNeuronSpikeActivation::SetOpenCL(obj);
//---
   cQKV.SetOpenCL(OpenCL);
   cDiaganalBias.SetOpenCL(OpenCL);
   cW0.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   ciScore = OpenCL.AddBuffer(sizeof(float) * iUnits * iUnits * iHeads, CL_MEM_READ_WRITE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSSAM::TrainMode(bool flag)
  {
   CNeuronSpikeActivation::TrainMode(flag);
//---
   cQKV.TrainMode(bTrain);
   cDiaganalBias.TrainMode(bTrain);
   cW0.TrainMode(bTrain);
   cNorm.TrainMode(bTrain);
   cResidual.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint chanels_in, uint chanels_out, uint units_in,
                                   uint units_out, uint group_size, uint groups,
                                   uint heads, uint dimension_k,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeResNeXtBlock::Init(numOutputs, myIndex, open_cl, chanels_in, chanels_out,
                                      units_in, units_out, group_size, groups, optimization_type, batch))
      ReturnFalse;
//---
   if(!caSSAM[0].Init(0, 0, OpenCL, units_out, chanels_out, heads, dimension_k, true, optimization, iBatch))
      ReturnFalse;
   if(!caSSAM[1].Init(0, 0, OpenCL, chanels_out, units_out, heads, dimension_k, false, optimization, iBatch))
      ReturnFalse;
//---
   CBufferFloat* temp = caBottleneck[0].getGradient();
   if(!caBottleneck[0].SetGradient(caSSAM[0].getGradient(), false))
      ReturnFalse;
   if(!caSSAM[0].SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!caTranspose[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[1].FeedForward(caTranspose[0].AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      if(!caSSAM[i].FeedForward(caBottleneck[i].AsObject()))
         ReturnFalse;
     }
   if(!caTranspose[1].FeedForward(caSSAM[1].AsObject()))
      ReturnFalse;
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = cG[0];
   if(!current)
      ReturnFalse;
   if(!SumAndNormalize(caSSAM[0].getOutput(), caTranspose[1].getOutput(),
                       current.getOutput(), caTranspose[1].GetCount(), false, 0, 0, 0, 1) ||
      !SumAndNormalize(current.getOutput(), cResidual.getOutput(),
                       current.getOutput(), caTranspose[1].GetCount(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.FeedForward(cG[i - 1]))
         ReturnFalse;
     }
//---
   return CNeuronSpikeActivation::feedForward(cG[-1]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cG[-1]))
      ReturnFalse;
   CNeuronBaseOCL* current = NULL;
   for(int i = cG.Total() - 2; i >= 0; i--)
     {
      current = cG[i];
      if(!current ||
         !current.CalcHiddenGradients(cG[i + 1]))
         ReturnFalse;
     }
//---
   if(!caSSAM[1].CalcHiddenGradients(caTranspose[1].AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!caBottleneck[i].CalcHiddenGradients(caSSAM[i].AsObject()))
         ReturnFalse;
   if(!caTranspose[0].CalcHiddenGradients(caBottleneck[1].AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(caTranspose[0].getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(caBottleneck[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp,
                       caTranspose[0].GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp,
                       caTranspose[0].GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!caBottleneck[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!caBottleneck[1].UpdateInputWeights(caTranspose[0].AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!caSSAM[i].UpdateInputWeights(caBottleneck[i].AsObject()))
         ReturnFalse;
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.UpdateInputWeights(cG[i - 1]))
         ReturnFalse;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(cG[-1]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::Save(const int file_handle)
  {
   if(!CNeuronSpikeResNeXtBlock::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caSSAM.Size(); i++)
      if(!caSSAM[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::Load(const int file_handle)
  {
   if(!CNeuronSpikeResNeXtBlock::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < caSSAM.Size(); i++)
      if(!LoadInsideLayer(file_handle, caSSAM[i].AsObject()))
         ReturnFalse;
//---
   CBufferFloat* temp = caBottleneck[0].getGradient();
   if(!caBottleneck[0].SetGradient(caSSAM[0].getGradient(), false))
      ReturnFalse;
   if(!caSSAM[0].SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeResNeXtBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSSAMResNeXtBlock* Source = source;
   for(uint i = 0; i < caSSAM.Size(); i++)
      if(!caSSAM[i].WeightsUpdate(Source.caSSAM[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSSAMResNeXtBlock::Clear(void)
  {
   if(!CNeuronSpikeResNeXtBlock::Clear())
      ReturnFalse;
//---
   for(uint i = 0; i < caSSAM.Size(); i++)
      if(!caSSAM[i].Clear())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSSAMResNeXtBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeResNeXtBlock::SetOpenCL(obj);
//---
   for(uint i = 0; i < caSSAM.Size(); i++)
      caSSAM[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSSAMResNeXtBlock::TrainMode(bool flag)
  {
   CNeuronSpikeResNeXtBlock::TrainMode(flag);
//---
   for(uint i = 0; i < caSSAM.Size(); i++)
      caSSAM[i].TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint chanels_in, uint chanels_out, uint units_in, uint units_out,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, chanels_out * units_out, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   int index = 0;
   if(!cMaskTime.Init(0, index, OpenCL, units_in, optimization, iBatch))
      ReturnFalse;
   cMaskTime.SetActivationFunction(None);
   index++;
   if(!cMaskSpatial.Init(0, index, OpenCL, chanels_in, optimization, iBatch))
      ReturnFalse;
   cMaskSpatial.SetActivationFunction(None);
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, chanels_in * units_in * 5, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
   if(!cSpatialDScale.Init(0, index, OpenCL, chanels_in, chanels_in, chanels_out, units_in, 5, optimization, iBatch))
      ReturnFalse;
   cSpatialDScale.SetActivationFunction(SoftPlus);
   index++;
   if(!cTimeDScale.Init(0, index, OpenCL, units_in, units_in, units_out, chanels_out, 5, optimization, iBatch))
      ReturnFalse;
   cTimeDScale.SetActivationFunction(TANH);
   index++;
   if(!caTranspose[0].Init(0, index, OpenCL, 5, units_in, chanels_out, optimization, iBatch))
      ReturnFalse;
   if(!caTranspose[1].Init(0, index, OpenCL, 5, chanels_out, units_out, optimization, iBatch))
      ReturnFalse;
   if(!cOne.BufferInit(5, 1) ||
      !cOne.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!STFS(NeuronOCL))
      ReturnFalse;
   if(!cSpatialDScale.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   if(!caTranspose[0].FeedForward(cSpatialDScale.AsObject()))
      ReturnFalse;
   if(!cTimeDScale.FeedForward(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!caTranspose[1].FeedForward(cTimeDScale.AsObject()))
      ReturnFalse;
   if(!DeConcat(Output, caTranspose[1].getPrevOutput(), caTranspose[1].getOutput(), Neurons(), caTranspose[1].Neurons() - Neurons(), 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::STFS(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
   if(NeuronOCL.Neurons() < (cMaskSpatial.Neurons()*cMaskTime.Neurons()))
      ReturnFalse;
//--- Time mask
     {
      int total = cMaskTime.Neurons();
      vector<float> vec = vector<float>::Zeros(total);
      for(int i = 0; i < total / 2; i++)
        {
         int id = RND(total);
         vec[id] = 1;
        }
      CBufferFloat* out = cMaskTime.getOutput();
      if(!out ||
         !out.Fill(vec))
         ReturnFalse;
     }
//--- Spatial msak
     {
      int total = cMaskSpatial.Neurons();
      vector<float> vec = vector<float>::Zeros(total);
      for(int i = 0; i < total / 2; i++)
        {
         int id = RND(total);
         vec[id] = 1;
        }
      CBufferFloat* out = cMaskSpatial.getOutput();
      if(!out ||
         !out.Fill(vec))
         ReturnFalse;
     }
//---
   uint global_work_offset[3] = { 0, 0, 0 };
   uint global_work_size[3] = { cMaskTime.Neurons(), cMaskSpatial.Neurons(), 5 };
   uint local_work_size[3] = { 1, 1, global_work_size[2] };
   uint kernel = def_k_STFS;
   setBuffer(kernel, def_k_stfs_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_stfs_mask_time, cMaskTime.getOutputIndex())
   setBuffer(kernel, def_k_stfs_mask_spatial, cMaskSpatial.getOutputIndex())
   setBuffer(kernel, def_k_stfs_outputs, cConcatenated.getOutputIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cConcatenated.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::STFSGrad(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
   if(NeuronOCL.Neurons() < (cMaskSpatial.Neurons()*cMaskTime.Neurons()))
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = { cMaskTime.Neurons(), cMaskSpatial.Neurons(), 5 };
   uint local_work_size[3] = { 1, 1, 5 };
   uint kernel = def_k_STFSGrad;
   setBuffer(kernel, def_k_stfs_inputs, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_stfs_mask_time, cMaskTime.getOutputIndex())
   setBuffer(kernel, def_k_stfs_mask_spatial, cMaskSpatial.getOutputIndex())
   setBuffer(kernel, def_k_stfs_outputs, cConcatenated.getGradientIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!SumAndNormalize(Output, Gradient, PrevOutput, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!MatMul(GetPointer(cOne), PrevOutput, caTranspose[1].getGradient(), cOne.Total(), 1, Neurons(), 1, false))
      ReturnFalse;
   float error = 1;
   if(!caTranspose[1].calcOutputGradients(caTranspose[1].getGradient(), error))
      ReturnFalse;
   if(!cTimeDScale.CalcHiddenGradients(caTranspose[1].AsObject()))
      ReturnFalse;
   if(!caTranspose[0].CalcHiddenGradients(cTimeDScale.AsObject()))
      ReturnFalse;
   if(!cSpatialDScale.CalcHiddenGradients(caTranspose[0].AsObject()))
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cSpatialDScale.AsObject()))
      ReturnFalse;
   if(!STFSGrad(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cSpatialDScale.UpdateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
   if(!cTimeDScale.UpdateInputWeights(caTranspose[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTFS* Source = source;
   if(!cSpatialDScale.WeightsUpdate(Source.cSpatialDScale.AsObject(), tau))
      ReturnFalse;
   if(!cTimeDScale.WeightsUpdate(Source.cTimeDScale.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cSpatialDScale.Save(file_handle))
      ReturnFalse;
   if(!cTimeDScale.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFS::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cSpatialDScale.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTimeDScale.AsObject()))
      ReturnFalse;
//---
   uint units_in = cTimeDScale.GetWindow();
   uint units_out = cTimeDScale.GetFilters();
   uint chanels_in = cSpatialDScale.GetWindow();
   uint chanels_out = cSpatialDScale.GetFilters();
//---
   int index = 0;
   if(!cMaskTime.Init(0, index, OpenCL, units_in, optimization, iBatch))
      ReturnFalse;
   cMaskTime.SetActivationFunction(None);
   index++;
   if(!cMaskSpatial.Init(0, index, OpenCL, chanels_in, optimization, iBatch))
      ReturnFalse;
   cMaskSpatial.SetActivationFunction(None);
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, chanels_in * units_in * 5, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
   index += 2;
   if(!caTranspose[0].Init(0, index, OpenCL, 5, units_in, chanels_out, optimization, iBatch))
      ReturnFalse;
   if(!caTranspose[1].Init(0, index, OpenCL, 5, chanels_out, units_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTFS::SetActivationFunction(ENUM_ACTIVATION value)
  {
   cTimeDScale.SetActivationFunction(value);
   activation = None;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTFS::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cMaskTime.SetOpenCL(OpenCL);
   cMaskSpatial.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cSpatialDScale.SetOpenCL(OpenCL);
   cTimeDScale.SetOpenCL(OpenCL);
   caTranspose[0].SetOpenCL(OpenCL);
   caTranspose[1].SetOpenCL(OpenCL);
   cOne.BufferFree();
   cOne.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint units, uint chanels_in, uint chanels_out,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units * chanels_out, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   for(uint i = 0; i < cConcatenated.Size(); i++)
     {
      if(!cConcatenated[i].Init(0, index, OpenCL, units * (chanels_in + chanels_out), optimization, iBatch))
         ReturnFalse;
      cConcatenated[i].SetActivationFunction(None);
      index++;
     }
   if(!cZR.Init(0, index, OpenCL, chanels_in + chanels_out, chanels_in + chanels_out, 2 * chanels_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cZR.SetActivationFunction(SIGMOID);
   index++;
   if(!cZ.Init(0, index, OpenCL, cZR.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction((ENUM_ACTIVATION)cZR.Activation());
   index++;
   if(!cR.Init(0, index, OpenCL, cZR.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cR.SetActivationFunction((ENUM_ACTIVATION)cZR.Activation());
   index++;
   if(!cHt.Init(0, index, OpenCL, chanels_in + chanels_out, chanels_in + chanels_out, chanels_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cHt.SetActivationFunction(TANH);
   index++;
   if(!cH.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNorm.Init(0, index, OpenCL, Neurons(), iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint hidden = cHt.GetFilters();
   uint inputs = cHt.GetWindow() - hidden;
//---
   if(!cH.SwapOutputs())
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), cH.getPrevOutput(), cConcatenated[0].getOutput(), inputs, hidden, units))
      ReturnFalse;
   if(!cZR.FeedForward(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!DeConcat(cZ.getOutput(), cR.getOutput(), cZR.getOutput(), hidden, hidden, units))
      ReturnFalse;
   if(!ElementMult(cR.getOutput(), cH.getPrevOutput(), cR.getPrevOutput()))
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), cR.getPrevOutput(), cConcatenated[1].getOutput(), inputs, hidden, units))
      ReturnFalse;
   if(!cHt.FeedForward(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!GateElementMult(cHt.getOutput(), cH.getPrevOutput(), cZ.getOutput(), cH.getOutput()))
      ReturnFalse;
//---
   if(!cNorm.FeedForward(cH.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeActivation::feedForward(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint hidden = cHt.GetFilters();
   uint inputs = cHt.GetWindow() - hidden;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cH.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!GateElementMultGrad(cHt.getOutput(), cHt.getGradient(),
                           cH.getPrevOutput(), cHt.getPrevOutput(),
                           cZ.getOutput(), cZ.getGradient(),
                           cH.getGradient(), cHt.Activation(), cH.Activation(), cZR.Activation()))
      ReturnFalse;
   if(!cConcatenated[1].CalcHiddenGradients(cHt.AsObject()))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), cR.getPrevOutput(), cConcatenated[1].getGradient(), inputs, hidden, units))
      ReturnFalse;
   if(!ElementMultGrad(cR.getOutput(), cR.getGradient(),
                       cH.getPrevOutput(), cHt.getPrevOutput(),
                       cR.getPrevOutput(), cZR.Activation(), cH.Activation()))
      ReturnFalse;
   if(!Concat(cZ.getGradient(), cR.getGradient(), cZR.getGradient(), hidden, hidden, units))
      ReturnFalse;
   if(!cConcatenated[0].CalcHiddenGradients(cZR.AsObject()))
      ReturnFalse;
   if(!DeConcat(cConcatenated[0].getPrevOutput(), cH.getPrevOutput(), cConcatenated[0].getGradient(), inputs, hidden, units))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cConcatenated[0].getPrevOutput(), NeuronOCL.getGradient(),
                       inputs, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cZR.UpdateInputWeights(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!cHt.UpdateInputWeights(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cH.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeActivation::updateInputWeights(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeConvGRU* Source = source;
   if(!cZR.WeightsUpdate(Source.cZR.AsObject(), tau))
      ReturnFalse;
   if(!cHt.WeightsUpdate(Source.cHt.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
   if(!cZR.Save(file_handle))
      ReturnFalse;
   if(!cHt.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cZR.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHt.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint chanels_out = cHt.GetFilters();
   uint chanels_in = cHt.GetWindow() - chanels_out;
//---
   uint index = 0;
   for(uint i = 0; i < cConcatenated.Size(); i++)
     {
      if(!cConcatenated[i].Init(0, index, OpenCL, units * (chanels_in + chanels_out), optimization, iBatch))
         ReturnFalse;
      cConcatenated[i].SetActivationFunction(None);
      index++;
     }
   index += 2;
   if(!cZ.Init(0, index, OpenCL, cZR.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction((ENUM_ACTIVATION)cZR.Activation());
   index++;
   if(!cR.Init(0, index, OpenCL, cZR.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cR.SetActivationFunction((ENUM_ACTIVATION)cZR.Activation());
   index += 2;
   if(!cH.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeConvGRU::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
   cZR.SetOpenCL(OpenCL);
   cHt.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
   for(uint i = 0; i < cConcatenated.Size(); i++)
      cConcatenated[i].SetOpenCL(OpenCL);
   cZ.SetOpenCL(OpenCL);
   cR.SetOpenCL(OpenCL);
   cH.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
   if(!cH.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAddToStack::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint stack_size, uint dimension, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, stack_size * dimension * variables, optimization_type, batch))
      ReturnFalse;
//---
   activation = None;
   iStackSize = stack_size;
   iDimension = dimension;
   iVariables = variables;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAddToStack::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   if(NeuronOCL.Neurons() < int(iDimension * iVariables))
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = { iDimension,
                                MathMin((uint)OpenCL.GetMaxLocalSize(1), iStackSize),
                                iVariables
                              };
   uint local_work_size[3] = { 1, global_work_size[1], 1 };
   uint kernel = def_k_AddToStack;
   setBuffer(kernel, def_k_ats_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ats_stack, getOutputIndex())
   setArgument(kernel, def_k_ats_stack_size, (int)iStackSize)
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
bool CNeuronAddToStack::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
   if(NeuronOCL.Neurons() < int(iDimension * iVariables))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), PrevOutput, Gradient, iDimension, iDimension * (iStackSize - 1), iVariables))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAddToStack::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iStackSize) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iDimension) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iVariables) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronAddToStack::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iStackSize = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iDimension = (uint)FileReadInteger(file_handle);
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iVariables = (uint)FileReadInteger(file_handle);
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronStackCorrelation::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint stack_size, uint dimension, uint variables,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, variables * stack_size, optimization_type, batch))
      ReturnFalse;
   if(!cStack.Init(0, 0, OpenCL, stack_size, dimension, variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronStackCorrelation::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cStack.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!MatMul(cStack.getOutput(), NeuronOCL.getOutput(), Output, cStack.GetStackSize(),
              cStack.GetDimension(), 1, cStack.GetVariables(), true))
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
bool CNeuronStackCorrelation::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!MatMulGrad(cStack.getOutput(), cStack.getGradient(),
                  NeuronOCL.getOutput(), cStack.getPrevOutput(),
                  Gradient, cStack.GetStackSize(),
                  cStack.GetDimension(), 1, cStack.GetVariables(), true))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), cStack.getPrevOutput(),
                       cStack.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cStack.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cStack.getPrevOutput(),
                       NeuronOCL.getGradient(), cStack.GetDimension(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronStackCorrelation::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronStackCorrelation::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronStackCorrelation::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cStack.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronStackCorrelation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cStack.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint &chanels[], uint &units[], uint group_size,
                                    uint groups, uint heads, uint dimension_k, uint stack_size,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels.Size() != units.Size())
      ReturnFalse;
//---
   const uint layers = chanels.Size() - 1;
   if(!CNeuronSpikeConvGRU::Init(numOutputs, myIndex, open_cl, units[layers], chanels[layers] + stack_size,
                                 chanels[layers], optimization_type, batch))
      ReturnFalse;
//---
   CNeuronSSAMResNeXtBlock*   conv_res = NULL;
   CNeuronStackCorrelation*   correl = NULL;
   CNeuronSpikeConvGRU*       gru = NULL;
   CNeuronBaseOCL*            neuron = NULL;
//---
   cRecurrentLine.Clear();
   cFlow.Clear();
   cCorrelation.Clear();
   cRecurrentLine.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
//---
   uint index = 0;
   for(uint i = 0; i < layers; i++)
     {
      conv_res = new CNeuronSSAMResNeXtBlock();
      if(!conv_res ||
         !conv_res.Init(0, index, OpenCL, chanels[i], chanels[i + 1],
                        units[i], units[i + 1], group_size, groups, heads,
                        dimension_k, optimization, iBatch) ||
         !cRecurrentLine.Add(conv_res))
        {
         DeleteObj(conv_res)
         ReturnFalse;
        }
      index++;
      neuron = new CNeuronBaseOCL();
      if(!neuron ||
         !neuron.Init(0, index, OpenCL, units[i + 1] * (chanels[i + 1]*stack_size), optimization, iBatch) ||
         !cRecurrentLine.Add(neuron))
        {
         DeleteObj(neuron)
         ReturnFalse;
        }
      if(i == layers)
         break;
      //---
      if(i < layers - 1)
        {
         index++;
         gru = new CNeuronSpikeConvGRU();
         if(!gru ||
            !gru.Init(0, index, OpenCL, units[i + 1], chanels[i + 1] + stack_size, chanels[i + 1], optimization, iBatch) ||
            !cRecurrentLine.Add(gru))
           {
            DeleteObj(gru)
            ReturnFalse;
           }
        }
      //---
      index++;
      correl = new CNeuronStackCorrelation();
      if(!correl ||
         !correl.Init(0, index, OpenCL, stack_size, chanels[i + 1], units[i + 1], optimization, iBatch) ||
         !cCorrelation.Add(correl))
        {
         DeleteObj(correl)
         ReturnFalse;
        }
      //---
      if(i == 0)
         continue;
      index++;
      conv_res = new CNeuronSSAMResNeXtBlock();
      if(!conv_res ||
         !conv_res.Init(0, index, OpenCL, chanels[i], chanels[i + 1],
                        units[i], units[i + 1], group_size, groups, heads,
                        dimension_k, optimization, iBatch) ||
         !cFlow.Add(conv_res))
        {
         DeleteObj(conv_res)
         ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *recur = NeuronOCL;
   CNeuronBaseOCL *flow = cRecurrentLine[0];
   CNeuronStackCorrelation *correl = NULL;
   CNeuronSSAMResNeXtBlock* resnext = NULL;
   int layers = (int)Layers();
//---
   for(int i = 0; i < layers; i++)
     {
      //--- ResNeXt
      if(!cRecurrentLine[i * 3] || !cRecurrentLine[i * 3].FeedForward(recur))
         ReturnFalse;
      recur = cRecurrentLine[i * 3];
      //--- Correlation
      correl = cCorrelation[i];
      if(!correl ||
         !correl.FeedForward(flow))
         ReturnFalse;
      //--- Concatenate
      resnext = recur;
      recur = cRecurrentLine[i * 3 + 1];
      if(!recur ||
         !Concat(resnext.getOutput(), correl.getOutput(), recur.getOutput(), resnext.GetChannelsOut(), correl.GetStackSize(), resnext.GetUnitsOut()))
         ReturnFalse;
      if((i * 3 + 2) == cRecurrentLine.Total())
         break;
      //--- GRU
      if(!cRecurrentLine[i * 3 + 1] ||
         !cRecurrentLine[i * 3 + 2].FeedForward(recur))
         ReturnFalse;
      recur = cRecurrentLine[i * 3 + 1];
      if(!cFlow[i] ||
         !cFlow[i].FeedForward(flow))
         ReturnFalse;
      flow = cFlow[i];
     }
//---
   return CNeuronSpikeConvGRU::feedForward(recur);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//---
   CNeuronBaseOCL *recur = NeuronOCL;
   CNeuronBaseOCL *flow = cRecurrentLine[0];
   CNeuronStackCorrelation *correl = NULL;
   CNeuronSSAMResNeXtBlock* resnext = NULL;
   int layers = (int)Layers();
//---
   for(int i = 0; i < layers; i++)
     {
      //--- ResNeXt
      if(!cRecurrentLine[i * 3] || !cRecurrentLine[i * 3].UpdateInputWeights(recur))
         ReturnFalse;
      recur = cRecurrentLine[i * 3];
      //--- Correlation
      correl = cCorrelation[i];
      if(!correl ||
         !correl.UpdateInputWeights(flow))
         ReturnFalse;
      //--- Concatenate
      recur = cRecurrentLine[i * 3 + 1];
      if((i * 3 + 2) == cRecurrentLine.Total())
         break;
      //--- GRU
      if(!cRecurrentLine[i * 3 + 1] ||
         !cRecurrentLine[i * 3 + 2].UpdateInputWeights(recur))
         ReturnFalse;
      recur = cRecurrentLine[i * 3 + 1];
      if(!cFlow[i] ||
         !cFlow[i].UpdateInputWeights(flow))
         ReturnFalse;
      flow = cFlow[i];
     }
//---
   return CNeuronSpikeConvGRU::updateInputWeights(recur);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvGRU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTEFlowNetEncoder* Source = source;
   if(!cRecurrentLine.WeightsUpdate(Source.cRecurrentLine.AsObject(), tau))
      ReturnFalse;
   if(!cCorrelation.WeightsUpdate(Source.cCorrelation.AsObject(), tau))
      ReturnFalse;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvGRU::calcInputGradients(cRecurrentLine[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL *concat = cRecurrentLine[-1];
   CNeuronBaseOCL *flow = cFlow[-1];
   CNeuronStackCorrelation *correl = cCorrelation[-1];
   CNeuronSSAMResNeXtBlock* resnext = cRecurrentLine[-2];
   int layers = (int)Layers() - 1;
//---
   for(int i = layers; i >= 0; i--)
     {
      //--- Concatenate
      if(!resnext || !concat || !correl ||
         !DeConcat(resnext.getGradient(), correl.getGradient(), concat.getGradient(),
                   resnext.GetChannelsOut(), correl.GetStackSize(), resnext.GetUnitsOut()))
         ReturnFalse;
      if(i == 0)
        {
         CBufferFloat* temp = resnext.getGradient();
         //--- Correlation
         if(!resnext.SetGradient(resnext.getPrevOutput(), false) ||
            !resnext.CalcHiddenGradients(correl) ||
            !SumAndNormalize(temp, resnext.getGradient(), temp, resnext.GetChannelsOut(), false, 0, 0, 0, 1))
            ReturnFalse;
         //--- Flow
         if(i < layers)
            if(!resnext.CalcHiddenGradients(cFlow[i]) ||
               !SumAndNormalize(temp, resnext.getGradient(), temp, 1, false, 0, 0, 0, 1))
               ReturnFalse;
         if(!resnext.SetGradient(temp, false))
            ReturnFalse;
         //--- ResNeXt
         if(!NeuronOCL.CalcHiddenGradients(resnext))
            ReturnFalse;
        }
      else
        {
         //--- Correlation
         if(!flow ||
            !flow.CalcHiddenGradients(correl))
            ReturnFalse;
         //--- Flow
         if(i < layers)
           {
            CBufferFloat* temp = flow.getGradient();
            if(!flow.SetGradient(flow.getPrevOutput(), false) ||
               !flow.CalcHiddenGradients(cFlow[i]) ||
               !SumAndNormalize(temp, flow.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
               !flow.SetGradient(temp, false))
               ReturnFalse;
           }
         //--- ResNeXt
         if(!cRecurrentLine[i * 3 - 1] ||
            !cRecurrentLine[i * 3 - 1].CalcHiddenGradients(resnext))
            ReturnFalse;
         //---
         concat = cRecurrentLine[(i - 1) * 3 + 1];
         flow = cFlow[i - 2];
         correl = cCorrelation[i - 1];
         resnext = cRecurrentLine[(i - 1) * 3];
         //--- GRU
         if(!concat ||
            !concat.CalcHiddenGradients(cRecurrentLine[i * 3 - 1]))
            ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Save(file_handle))
      ReturnFalse;
//---
   if(!cRecurrentLine.Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
   if(!cCorrelation.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Load(file_handle))
      ReturnFalse;
//---
   if(!cRecurrentLine.Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
   if(!cCorrelation.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetEncoder::Clear(void)
  {
   if(!CNeuronSpikeConvGRU::Clear())
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = NULL;
   for(int i = 0; i < cRecurrentLine.Total(); i++)
     {
      neuron = cRecurrentLine[i];
      if(!neuron ||
         !neuron.Clear())
         ReturnFalse;
     }
   for(int i = 0; i < cCorrelation.Total(); i++)
     {
      neuron = cCorrelation[i];
      if(!neuron ||
         !neuron.Clear())
         ReturnFalse;
     }
   for(int i = 0; i < cFlow.Total(); i++)
     {
      neuron = cFlow[i];
      if(!neuron ||
         !neuron.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvGRU::SetOpenCL(obj);
//---
   cRecurrentLine.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetEncoder::TrainMode(bool flag)
  {
   CNeuronSpikeConvGRU::TrainMode(flag);
//---
   cRecurrentLine.TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
   cCorrelation.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBaseOCL* CNeuronSTEFlowNetEncoder::GetLayer(uint layer)
  {
   if(layer >= Layers())
      return NULL;
   if(layer == (Layers() - 1))
      return GetPointer(this);
//---
   return cRecurrentLine[layer * 3 + 2];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint chanels, uint units, uint group_size, uint groups,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, 2 * chanels * units, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   CNeuronSpikeResNeXtBottleneck* conv = NULL;
   CNeuronSpikeResNeXtBlock*      residual = NULL;
//---
   cBlock.Clear();
   cBlock.SetOpenCL(OpenCL);
//---
   uint index = 0;
   conv = new CNeuronSpikeResNeXtBottleneck();
   if(!conv ||
      !conv.Init(0, index, OpenCL, chanels, chanels, units, units, group_size, groups, optimization, iBatch) ||
      !cBlock.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   for(int i = 0; i < 2; i++)
     {
      index++;
      residual = new CNeuronSpikeResNeXtBlock();
      if(!residual ||
         !residual.Init(0, index, OpenCL, chanels, chanels, units, units, group_size, groups, optimization, iBatch) ||
         !cBlock.Add(residual))
        {
         DeleteObj(residual)
         ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(cBlock.Total() <= 0)
      ReturnFalse;
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
   for(int i = 0; i < cBlock.Total(); i++)
     {
      current = cBlock[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!Concat(prev.getOutput(), NeuronOCL.getOutput(), Output, 1, 1, Neurons() / 2))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(cBlock.Total() <= 0)
      ReturnFalse;
   CNeuronBaseOCL* current = cBlock[-1];
   CNeuronBaseOCL* next = NULL;
//---
   if(!DeConcat(current.getGradient(), PrevOutput, Gradient, 1, 1, Neurons() / 2))
      ReturnFalse;
   next = current;
   for(int i = cBlock.Total() - 2; i >= 0; i--)
     {
      current = cBlock[i];
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
      next = current;
     }
//---
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), PrevOutput, PrevOutput, NeuronOCL.Activation()))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(next) ||
      !SumAndNormalize(NeuronOCL.getGradient(), PrevOutput, PrevOutput, 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(cBlock.Total() <= 0)
      ReturnFalse;
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* current = NULL;
   for(int i = 0; i < cBlock.Total(); i++)
     {
      current = cBlock[i];
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
bool CNeuronSTEFlowNetResidualBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTEFlowNetResidualBlock* Source = source;
   if(!cBlock.WeightsUpdate(Source.cBlock.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::Clear(void)
  {
   for(int i = 0; i < cBlock.Total(); i++)
     {
      if(!cBlock[i] ||
         !cBlock[i].Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetResidualBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cBlock.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetResidualBlock::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cBlock.TrainMode(flag);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cBlock.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetResidualBlock::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cBlock.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint &chanels[], uint &units[], uint group_size, uint groups,
                                    uint heads, uint dimension_k,
                                    CNeuronSTEFlowNetEncoder *encoder,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels.Size() != units.Size())
      ReturnFalse;
//---
   const uint layers = chanels.Size() - 1;
   if(layers < 1)
      ReturnFalse;
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units[layers]*chanels[layers], optimization_type, batch))
      ReturnFalse;
   cEncoder = encoder;
   uint enc_layers = (!cEncoder ? 0 : cEncoder.Layers());
//---
   CNeuronSSAMResNeXtBlock*   conv_res = NULL;
   CNeuronConvOCL*            conv = NULL;
   CNeuronBatchNormOCL*       norm = NULL;
   CNeuronSpikeActivation*    sa = NULL;
   CNeuronSpikeConvGRU*       gru = NULL;
   CNeuronBaseOCL*            concat = NULL;
//---
   cMainLine.Clear();
   cFlow.Clear();
   cMainLine.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
//---
   uint index = 0;
//--- Main Line
   conv_res = new CNeuronSSAMResNeXtBlock();
   if(!conv_res ||
      !conv_res.Init(0, index, OpenCL, chanels[0], chanels[1],
                     units[0], units[1], group_size, groups, heads,
                     dimension_k, optimization, iBatch) ||
      !cMainLine.Add(conv_res))
     {
      DeleteObj(conv_res)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, chanels[1], chanels[1], chanels[1], units[1], 1, optimization, iBatch) ||
      !cMainLine.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cMainLine.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   uint ch_in = chanels[0];
//---
   for(uint i = 1; i < layers; i++)
     {
      //--- Main Line
      index++;
      sa = new CNeuronSpikeActivation();
      if(!sa ||
         !sa.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
         !cMainLine.Add(sa))
        {
         DeleteObj(sa)
         ReturnFalse;
        }
      //--- Flow
      index++;
      conv_res = new CNeuronSSAMResNeXtBlock();
      if(!conv_res ||
         !conv_res.Init(0, index, OpenCL, ch_in, chanels[i],
                        units[i - 1], units[i], group_size, groups, heads,
                        dimension_k, optimization, iBatch) ||
         !cFlow.Add(conv_res))
        {
         DeleteObj(conv_res)
         ReturnFalse;
        }
      index++;
      ch_in = 2 * chanels[i];
      if(enc_layers - i - 1 >= 0)
        {
         gru = cEncoder.GetLayer(enc_layers - i - 1);
         if(!gru)
            ReturnFalse;
         if(gru.GetUnits() != units[i])
            ReturnFalse;
         ch_in += gru.GetChanelsOut();
        }
      concat = new CNeuronBaseOCL();
      if(!concat ||
         !concat.Init(0, index, OpenCL, ch_in * units[i], optimization, iBatch) ||
         !cFlow.Add(concat))
        {
         DeleteObj(concat)
         ReturnFalse;
        }
      concat.SetActivationFunction(None);
      //--- Main Line
      conv_res = new CNeuronSSAMResNeXtBlock();
      if(!conv_res ||
         !conv_res.Init(0, index, OpenCL, ch_in, ch_in,
                        units[i], units[i + 1], group_size, groups, heads,
                        dimension_k, optimization, iBatch) ||
         !cMainLine.Add(conv_res))
        {
         DeleteObj(conv_res)
         ReturnFalse;
        }
      index++;
      conv = new CNeuronConvOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, chanels[i + 1], chanels[i + 1], chanels[i + 1], units[i + 1], 1, optimization, iBatch) ||
         !cMainLine.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
      index++;
      norm = new CNeuronBatchNormOCL();
      if(!norm ||
         !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
         !cMainLine.Add(norm))
        {
         DeleteObj(norm)
         ReturnFalse;
        }
      norm.SetActivationFunction(None);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Main Line
   CNeuronSSAMResNeXtBlock*   conv_res = NULL;
   CNeuronConvOCL*            conv = NULL;
   CNeuronBatchNormOCL*       norm = NULL;
   CNeuronSpikeActivation*    sa = NULL;
   CNeuronSpikeConvGRU*       gru = NULL;
//--- Flow
   CNeuronSSAMResNeXtBlock*   flow = NULL;
   CNeuronBaseOCL*            concat = NULL;
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   int layers = (cMainLine.Total() + 1) / 4;
   int enc_layers = int(!cEncoder ? 0 : cEncoder.Layers());
   int enc_dim = 0;
//---
   for(int i = 0; i < layers; i++)
     {
      conv_res = cMainLine[i * 4];
      conv = cMainLine[i * 4 + 1];
      norm = cMainLine[i * 4 + 2];
      sa = cMainLine[i * 4 + 3];
      if(enc_layers - i - 2 >= 0)
        {
         gru = cEncoder.GetLayer(uint(enc_layers - i - 2));
         if(!gru)
            ReturnFalse;
         enc_dim = (int)gru.GetChanelsOut();
        }
      else
        {
         gru = NULL;
         enc_dim = 0;
        }
      flow = cFlow[i * 2];
      concat = cFlow[i * 2 + 1];
      //--- Main Line
      if(!conv_res ||
         !conv_res.FeedForward(prev))
         ReturnFalse;
      if(!conv ||
         !conv.FeedForward(conv_res))
         ReturnFalse;
      if(!norm ||
         !norm.FeedForward(conv))
         ReturnFalse;
      if(i == (layers - 1))
         break;
      if(!sa ||
         !sa.FeedForward(norm))
         ReturnFalse;
      //--- Flow
      if(!flow ||
         !flow.FeedForward(prev))
         ReturnFalse;
      if(!concat)
         ReturnFalse;
      if(!gru)
        {
         if(!Concat(sa.getOutput(), flow.getOutput(), concat.getOutput(),
                    conv.GetFilters(), flow.GetChannelsOut(), conv.GetUnits()))
            ReturnFalse;
        }
      else
        {
         if(conv.GetUnits() != gru.GetUnits())
            ReturnFalse;
         if(!Concat(sa.getOutput(), flow.getOutput(), gru.getOutput(), concat.getOutput(),
                    conv.GetFilters(), flow.GetChannelsOut(), enc_dim, conv.GetUnits()))
            ReturnFalse;
        }
      prev = concat;
     }
//---
   if(!CNeuronSpikeActivation::feedForward(norm))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//--- Main Line
   CNeuronSSAMResNeXtBlock*   conv_res = NULL;
   CNeuronConvOCL*            conv = NULL;
   CNeuronBatchNormOCL*       norm = NULL;
   CNeuronSpikeActivation*    sa = NULL;
   CNeuronSpikeConvGRU*       gru = NULL;
//--- Flow
   CNeuronSSAMResNeXtBlock*   flow = NULL;
   CNeuronBaseOCL*            concat = NULL;
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   int layers = (cMainLine.Total() + 1) / 4;
   int enc_layers = int(!cEncoder ? 0 : cEncoder.Layers());
   int enc_dim = 0;
//---
   for(int i = 0; i < layers; i++)
     {
      conv_res = cMainLine[i * 4];
      conv = cMainLine[i * 4 + 1];
      norm = cMainLine[i * 4 + 2];
      sa = cMainLine[i * 4 + 3];
      if(enc_layers - i >= 0)
        {
         gru = cEncoder.GetLayer(uint(enc_layers - i - 1));
         if(!gru)
            ReturnFalse;
         enc_dim = (int)gru.GetChanelsOut();
        }
      else
        {
         gru = NULL;
         enc_dim = 0;
        }
      flow = cFlow[i * 2];
      concat = cFlow[i * 2 + 1];
      //--- Main Line
      if(!conv_res ||
         !conv_res.UpdateInputWeights(prev))
         ReturnFalse;
      if(!conv ||
         !conv.UpdateInputWeights(conv_res))
         ReturnFalse;
      if(!norm ||
         !norm.UpdateInputWeights(conv))
         ReturnFalse;
      if(i == (layers - 1))
         break;
      if(!sa ||
         !sa.UpdateInputWeights(norm))
         ReturnFalse;
      //--- Flow
      if(!flow ||
         !flow.UpdateInputWeights(prev))
         ReturnFalse;
      prev = concat;
     }
//---
   if(!CNeuronSpikeActivation::updateInputWeights(norm))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
//--- Main Line
   CNeuronSSAMResNeXtBlock*   conv_res = cMainLine[-3];
   CNeuronConvOCL*            conv = cMainLine[-2];
   CNeuronBatchNormOCL*       norm = cMainLine[-1];
   CNeuronSpikeActivation*    sa = cMainLine[-4];
   CNeuronSpikeConvGRU*       gru = NULL;
//--- Flow
   CNeuronSSAMResNeXtBlock*   flow = cFlow[-2];
//---
   int layers = cMainLine.Total() / 4;
   CNeuronBaseOCL* prev = (layers > 1 ? cFlow[-1] : NeuronOCL);
   int enc_layers = int(!cEncoder ? 0 : cEncoder.Layers());
   int enc_dim = 0;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(norm))
      ReturnFalse;
   if(!conv ||
      !conv.CalcHiddenGradients(norm))
      ReturnFalse;
   if(!conv_res ||
      !conv_res.CalcHiddenGradients(conv))
      ReturnFalse;
   if(!prev ||
      !prev.CalcHiddenGradients(conv_res))
      ReturnFalse;
//---
   for(int i = layers - 1; i >= 0; i--)
     {
      if(enc_layers - i - 2 >= 0)
        {
         gru = cEncoder.GetLayer(uint(enc_layers - i - 1));
         if(!gru)
            ReturnFalse;
         enc_dim = (int)gru.GetChanelsOut();
        }
      else
        {
         gru = NULL;
         enc_dim = 0;
        }
      if(!sa || !flow)
         ReturnFalse;
      if(!gru)
        {
         if(!DeConcat(sa.getGradient(), flow.getGradient(), prev.getGradient(),
                      conv.GetFilters(), flow.GetChannelsOut(), conv.GetUnits()))
            ReturnFalse;
        }
      else
         if(!DeConcat(sa.getGradient(), flow.getGradient(), gru.getGradient(), prev.getGradient(),
                      conv.GetFilters(), flow.GetChannelsOut(), enc_dim, conv.GetUnits()))
            ReturnFalse;
      //--- Flow
      prev = (i > 0 ? cFlow[i * 2 - 1] : NeuronOCL);
      if(!prev ||
         !prev.CalcHiddenGradients(flow))
         ReturnFalse;
      //--- Main Line
      conv_res = cMainLine[i * 4];
      conv = cMainLine[i * 4 + 1];
      norm = cMainLine[i * 4 + 2];
      if(!norm.CalcHiddenGradients(sa))
         ReturnFalse;
      if(!conv ||
         !conv.CalcHiddenGradients(norm))
         ReturnFalse;
      if(!conv_res ||
         !conv_res.CalcHiddenGradients(conv))
         ReturnFalse;
      CBufferFloat* temp = prev.getGradient();
      if(!prev.SetGradient(prev.getPrevOutput(), false) ||
         !prev.CalcHiddenGradients(conv_res) ||
         !SumAndNormalize(temp, prev.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
         !prev.SetGradient(temp, false))
         ReturnFalse;
      if(i == 0)
         continue;
      sa = cMainLine[i * 4 - 1];
      //--- Flow
      flow = cFlow[(i - 1) * 2];
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
   if(!cMainLine.Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
   if(!cMainLine.Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTEFlowNetDecoder* Source = source;
   if(!cMainLine.WeightsUpdate(Source.cMainLine.AsObject(), tau))
      ReturnFalse;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNetDecoder::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = NULL;
   for(int i = 0; i < cMainLine.Total(); i++)
     {
      neuron = cMainLine[i];
      if(!neuron ||
         !neuron.Clear())
         ReturnFalse;
     }
   for(int i = 0; i < cFlow.Total(); i++)
     {
      neuron = cFlow[i];
      if(!neuron ||
         !neuron.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetDecoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
   cMainLine.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNetDecoder::TrainMode(bool flag)
  {
   CNeuronSpikeActivation::TrainMode(flag);
   cMainLine.TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint &chanels[], uint &units[], uint group_size,
                             uint groups, uint heads, uint dimension_k, uint stack_size,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels.Size() != units.Size())
      ReturnFalse;
//---
   uint rev_chanels[], rev_units[];
   if(ArrayResize(rev_chanels, chanels.Size()) < 0 ||
      ArrayResize(rev_units, units.Size()) < 0)
      ReturnFalse;
   uint total = chanels.Size();
   for(uint i = 0; i < total; i++)
      rev_chanels[total - i - 1] = chanels[i];
   rev_chanels[0] *= 2;
   for(uint i = 0; i < total; i++)
      rev_units[total - i - 1] = units[i];
//---
   if(!cSTEEncoder.Init(0, 0, open_cl, chanels, units, group_size, groups, heads,
                        dimension_k, stack_size, optimization_type, batch))
      ReturnFalse;
   if(!cSTEResidualBlock.Init(0, 1, open_cl, chanels[total - 1], units[total - 1], group_size, groups,
                              optimization, iBatch))
      ReturnFalse;
   rev_chanels[0] = cSTEResidualBlock.Neurons() / rev_units[0];
   if(!CNeuronSTEFlowNetDecoder::Init(numOutputs, myIndex, open_cl, rev_chanels,
                                      rev_units, group_size, groups, heads, dimension_k,
                                      cSTEEncoder.AsObject(), optimization_type, batch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cSTEEncoder.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cSTEResidualBlock.FeedForward(cSTEEncoder.AsObject()))
      ReturnFalse;
//---
   return CNeuronSTEFlowNetDecoder::feedForward(cSTEResidualBlock.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cSTEEncoder.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSTEResidualBlock.UpdateInputWeights(cSTEEncoder.AsObject()))
      ReturnFalse;
//---
   return CNeuronSTEFlowNetDecoder::updateInputWeights(cSTEResidualBlock.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSTEFlowNetDecoder::calcInputGradients(cSTEResidualBlock.AsObject()))
      ReturnFalse;
   if(!cSTEEncoder.CalcHiddenGradients(cSTEResidualBlock.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSTEEncoder.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::Save(const int file_handle)
  {
   if(!CNeuronSTEFlowNetDecoder::Save(file_handle))
      ReturnFalse;
   if(!cSTEEncoder.Save(file_handle))
      ReturnFalse;
   if(!cSTEResidualBlock.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::Load(const int file_handle)
  {
   if(!CNeuronSTEFlowNetDecoder::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSTEEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSTEResidualBlock.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSTEFlowNetDecoder::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTEFlowNet* Source = source;
   if(!cSTEEncoder.WeightsUpdate(Source.cSTEEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cSTEResidualBlock.WeightsUpdate(Source.cSTEResidualBlock.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTEFlowNet::Clear(void)
  {
   if(!CNeuronSTEFlowNetDecoder::Clear())
      ReturnFalse;
   if(!cSTEEncoder.Clear())
      ReturnFalse;
   if(!cSTEResidualBlock.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNet::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSTEFlowNetDecoder::SetOpenCL(obj);
   cSTEEncoder.SetOpenCL(OpenCL);
   cSTEResidualBlock.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTEFlowNet::TrainMode(bool flag)
  {
   CNeuronSTEFlowNetDecoder::TrainMode(flag);
   cSTEEncoder.TrainMode(bTrain);
   cSTEResidualBlock.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleStackCorrelation::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint stack_size, uint dimension, uint variables, uint levels,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronStackCorrelation::Init(numOutputs, myIndex, open_cl, stack_size, dimension,
                                     variables * levels, optimization_type, batch))
      ReturnFalse;
   iLevels = levels;
   if(!cScales.Init(0, 0, OpenCL, dimension * levels * variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleStackCorrelation::AggregationByTime(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   if(NeuronOCL.Neurons() < int(GetDimension()*GetVariables()))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { GetDimension(),
                                GetVariables()
                              };
   uint kernel = def_k_AggregationByTime;
   setBuffer(kernel, def_k_agt_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_agt_stack, cStack.getOutputIndex())
   setBuffer(kernel, def_k_agt_outputs, cScales.getOutputIndex())
   setArgument(kernel, def_k_agt_stack_size, (int)cStack.GetStackSize())
   setArgument(kernel, def_k_agt_levels, (int)GetLevels())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cStack.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleStackCorrelation::AggregationByTimeGrad(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   if(NeuronOCL.Neurons() < int(GetDimension()*GetVariables()))
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { GetDimension(),
                                GetVariables()
                              };
   uint kernel = def_k_AggregationByTimeGrad;
   setBuffer(kernel, def_k_agtg_inputs_gr, NeuronOCL.getGradientIndex())
   setBuffer(kernel, def_k_agtg_outputs_gr, cStack.getGradientIndex())
   setArgument(kernel, def_k_agtg_levels, (int)GetLevels())
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
bool CNeuronMultiScaleStackCorrelation::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!AggregationByTime(NeuronOCL))
      ReturnFalse;
   if(!cStack.FeedForward(cScales.AsObject()))
      ReturnFalse;
   if(!MatMul(cStack.getOutput(), NeuronOCL.getOutput(), Output, GetStackSize()*GetLevels(),
              GetDimension(), 1, GetVariables(), true))
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
bool CNeuronMultiScaleStackCorrelation::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!MatMulGrad(cStack.getOutput(), cStack.getGradient(),
                  NeuronOCL.getOutput(), cScales.getGradient(),
                  Gradient, GetStackSize()*GetLevels(),
                  GetDimension(), 1, GetVariables(), true))
      ReturnFalse;
   if(!AggregationByTimeGrad(NeuronOCL))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cScales.getGradient(), NeuronOCL.getGradient(), GetDimension(), false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleStackCorrelation::Save(const int file_handle)
  {
   if(!CNeuronStackCorrelation::Save(file_handle))
      ReturnFalse;
   if(!cScales.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iLevels) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultiScaleStackCorrelation::Load(const int file_handle)
  {
   if(!CNeuronStackCorrelation::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScales.AsObject()))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iLevels = FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultiScaleStackCorrelation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronStackCorrelation::SetOpenCL(obj);
   cScales.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint units, uint chanels_in, uint chanels_out,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvGRU::Init(numOutputs, myIndex, open_cl, units, chanels_in, chanels_out, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cProjectionX.Init(0, index, OpenCL, chanels_in, chanels_in, chanels_out, units, optimization, iBatch))
      ReturnFalse;
   cProjectionX.SetActivationFunction(None);
   index++;
   for(uint i = 0; i < cConcatenated2.Size(); i++)
     {
      if(!cConcatenated2[i].Init(0, index, OpenCL, 2 * units * chanels_in, optimization, iBatch))
         ReturnFalse;
      cConcatenated[i].SetActivationFunction(None);
      index++;
     }
   if(!cTranspose[0].Init(0, index, OpenCL, units, chanels_out, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose[1].Init(0, index, OpenCL, units, chanels_out, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose[2].Init(0, index, OpenCL, chanels_out, units, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cZR2.Init(0, index, OpenCL, 2 * units, 2 * units, 2 * units, chanels_out, 1, optimization, iBatch))
      ReturnFalse;
   cZR2.SetActivationFunction(SIGMOID);
   index++;
   if(!cZ2.Init(0, index, OpenCL, cZR2.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cZ2.SetActivationFunction((ENUM_ACTIVATION)cZR2.Activation());
   index++;
   if(!cR2.Init(0, index, OpenCL, cZR2.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cR2.SetActivationFunction((ENUM_ACTIVATION)cZR2.Activation());
   index++;
   if(!cHt2.Init(0, index, OpenCL, 2 * units, 2 * units, units, chanels_out, 1, optimization, iBatch))
      ReturnFalse;
   cHt2.SetActivationFunction(TANH);
   index++;
   if(!cH2.Init(0, index, OpenCL, units * chanels_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjectionX.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cTranspose[0].FeedForward(cH.AsObject()))
      ReturnFalse;
   if(!cTranspose[1].FeedForward(cProjectionX.AsObject()))
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint hidden = cHt.GetFilters();
   uint inputs = cHt.GetWindow() - hidden;
//---
   if(!Concat(cTranspose[0].getOutput(), cTranspose[1].getOutput(), cConcatenated2[0].getOutput(), units, units, hidden))
      ReturnFalse;
   if(!cZR2.FeedForward(cConcatenated2[0].AsObject()))
      ReturnFalse;
   if(!DeConcat(cZ2.getOutput(), cR2.getOutput(), cZR2.getOutput(), units, units, hidden))
      ReturnFalse;
   if(!ElementMult(cR2.getOutput(), cTranspose[0].getOutput(), cR2.getPrevOutput()))
      ReturnFalse;
   if(!Concat(cR2.getPrevOutput(), cTranspose[1].getOutput(), cConcatenated2[1].getOutput(), units, units, hidden))
      ReturnFalse;
   if(!cHt2.FeedForward(cConcatenated2[1].AsObject()))
      ReturnFalse;
   if(!GateElementMult(cHt2.getOutput(), cTranspose[0].getOutput(), cZ2.getOutput(), cH2.getOutput()))
      ReturnFalse;
   if(!cTranspose[2].FeedForward(cH2.AsObject()))
      ReturnFalse;
//---
   if(!Concat(NeuronOCL.getOutput(), cTranspose[2].getOutput(), cConcatenated[0].getOutput(), inputs, hidden, units))
      ReturnFalse;
   if(!cZR.FeedForward(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!DeConcat(cZ.getOutput(), cR.getOutput(), cZR.getOutput(), hidden, hidden, units))
      ReturnFalse;
   if(!ElementMult(cR.getOutput(), cTranspose[2].getOutput(), cR.getPrevOutput()))
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), cR.getPrevOutput(), cConcatenated[1].getOutput(), inputs, hidden, units))
      ReturnFalse;
   if(!cHt.FeedForward(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!GateElementMult(cHt.getOutput(), cTranspose[2].getOutput(), cZ.getOutput(), cH.getOutput()))
      ReturnFalse;
//---
   if(!cNorm.FeedForward(cH.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeActivation::feedForward(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint hidden = cHt.GetFilters();
   uint inputs = cHt.GetWindow() - hidden;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cH.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!GateElementMultGrad(cHt.getOutput(), cHt.getGradient(),
                           cTranspose[2].getOutput(), cTranspose[2].getGradient(),
                           cZ.getOutput(), cZ.getGradient(),
                           cH.getGradient(), cHt.Activation(), cTranspose[2].Activation(), cZR.Activation()))
      ReturnFalse;
   if(!cConcatenated[1].CalcHiddenGradients(cHt.AsObject()))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), cR.getPrevOutput(), cConcatenated[1].getGradient(), inputs, hidden, units))
      ReturnFalse;
   if(!ElementMultGrad(cR.getOutput(), cR.getGradient(),
                       cTranspose[2].getOutput(), cTranspose[2].getPrevOutput(),
                       cR.getPrevOutput(), cZR.Activation(), cH.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(cTranspose[2].getGradient(), cTranspose[2].getPrevOutput(),
                       cTranspose[2].getGradient(), hidden, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!Concat(cZ.getGradient(), cR.getGradient(), cZR.getGradient(), hidden, hidden, units))
      ReturnFalse;
   if(!cConcatenated[0].CalcHiddenGradients(cZR.AsObject()))
      ReturnFalse;
   if(!DeConcat(cConcatenated[0].getPrevOutput(), cTranspose[2].getPrevOutput(), cConcatenated[0].getGradient(), inputs, hidden, units))
      ReturnFalse;
   if(cTranspose[2].Activation() != None)
      if(!DeActivation(cTranspose[2].getOutput(), cTranspose[2].getPrevOutput(),
                       cTranspose[2].getPrevOutput(), cTranspose[2].Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cTranspose[2].getGradient(), cTranspose[2].getPrevOutput(),
                       cTranspose[2].getGradient(), hidden, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cConcatenated[0].getPrevOutput(),
                       NeuronOCL.getGradient(), inputs, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   if(!cH2.CalcHiddenGradients(cTranspose[2].AsObject()))
      ReturnFalse;
   if(!GateElementMultGrad(cHt2.getOutput(), cHt2.getGradient(), cTranspose[0].getOutput(),
                           cTranspose[0].getGradient(), cZ2.getOutput(), cZ2.getGradient(),
                           cH2.getGradient(), cHt2.Activation(), cTranspose[1].Activation(), cZR2.Activation()))
      ReturnFalse;
   if(!cConcatenated2[1].CalcHiddenGradients(cHt2.AsObject()))
      ReturnFalse;
   if(!DeConcat(cR2.getPrevOutput(), cTranspose[1].getGradient(), cConcatenated2[1].getOutput(), units, units, hidden))
      ReturnFalse;
   if(!ElementMultGrad(cR2.getOutput(), cR2.getGradient(), cTranspose[0].getOutput(),
                       cTranspose[0].getGradient(), cR2.getPrevOutput(),
                       cZR2.Activation(), cTranspose[0].Activation()))
      ReturnFalse;
   if(!Concat(cZ2.getGradient(), cR2.getGradient(), cZR2.getGradient(), units, units, hidden))
      ReturnFalse;
   if(!cConcatenated2[0].CalcHiddenGradients(cZR2.AsObject()))
      ReturnFalse;
   if(!DeConcat(cTranspose[0].getGradient(), cTranspose[1].getPrevOutput(), cConcatenated2[0].getGradient(), units, units, hidden))
      ReturnFalse;
   if(!SumAndNormalize(cTranspose[1].getGradient(), cTranspose[1].getPrevOutput(),
                       cTranspose[1].getGradient(), units, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(cTranspose[1])
   if(!cProjectionX.CalcHiddenGradients(cTranspose[1].AsObject()))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cProjectionX.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, inputs, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cProjectionX.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cZR2.UpdateInputWeights(cConcatenated2[0].AsObject()))
      ReturnFalse;
   if(!cHt2.UpdateInputWeights(cConcatenated2[1].AsObject()))
      ReturnFalse;
   if(!cZR.UpdateInputWeights(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!cHt.UpdateInputWeights(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cH.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeActivation::updateInputWeights(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvGRU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeConvGRU2D* Source = source;
   if(!cProjectionX.WeightsUpdate(Source.cProjectionX.AsObject(), tau))
      ReturnFalse;
   if(!cZR2.WeightsUpdate(Source.cZR2.AsObject(), tau))
      ReturnFalse;
   if(!cHt2.WeightsUpdate(Source.cHt2.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Save(file_handle))
      ReturnFalse;
   if(!cProjectionX.Save(file_handle))
      ReturnFalse;
   if(!cZR2.Save(file_handle))
      ReturnFalse;
   if(!cHt2.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvGRU2D::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjectionX.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cZR2.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHt2.AsObject()))
      ReturnFalse;
//---
   uint units = cHt.GetUnits();
   uint chanels_out = cHt.GetFilters();
   uint chanels_in = cHt.GetWindow() - chanels_out;
//---
   uint index = 1;
   for(uint i = 0; i < cConcatenated2.Size(); i++)
     {
      if(!cConcatenated2[i].Init(0, index, OpenCL, 2 * units * chanels_in, optimization, iBatch))
         ReturnFalse;
      cConcatenated[i].SetActivationFunction(None);
      index++;
     }
   if(!cTranspose[0].Init(0, index, OpenCL, units, chanels_out, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose[1].Init(0, index, OpenCL, units, chanels_out, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose[2].Init(0, index, OpenCL, chanels_out, units, optimization, iBatch))
      ReturnFalse;
   index += 2;
   if(!cZ2.Init(0, index, OpenCL, cZR2.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cZ2.SetActivationFunction((ENUM_ACTIVATION)cZR2.Activation());
   index++;
   if(!cR2.Init(0, index, OpenCL, cZR2.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cR2.SetActivationFunction((ENUM_ACTIVATION)cZR2.Activation());
   index += 2;
   if(!cH2.Init(0, index, OpenCL, units * chanels_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeConvGRU2D::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvGRU::SetOpenCL(obj);
   cProjectionX.SetOpenCL(OpenCL);
   cZR2.SetOpenCL(OpenCL);
   cHt2.SetOpenCL(OpenCL);
   for(uint i = 0; i < cConcatenated2.Size(); i++)
      cConcatenated2[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < cTranspose.Size(); i++)
      cTranspose[i].SetOpenCL(OpenCL);
   cZ2.SetOpenCL(OpenCL);
   cR2.SetOpenCL(OpenCL);
   cH2.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint &units[], uint &chanels[], uint group_size,
                       uint groups, uint stack_size, uint levels,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(units.Size() != chanels.Size())
      ReturnFalse;
   uint layers = units.Size() - 1;
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units[layers]*chanels[layers], optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   CNeuronSpikeResNeXtBlock*           resblock = NULL;
   CNeuronSpikeConvGRU2D*              gru = NULL;
   CNeuronMultiScaleStackCorrelation*  correlation = NULL;
   CNeuronBaseOCL*                     neuron = NULL;
   CNeuronBatchNormOCL*                norm = NULL;
//---
   cEncoder.Clear();
   cContext.Clear();
   cCorrelation.Clear();
   cG.Clear();
   cEncoder.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
   cG.SetOpenCL(OpenCL);
//---
   for(uint i = 0; i < layers; i++)
     {
      resblock = new CNeuronSpikeResNeXtBlock();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, chanels[i], chanels[i + 1], units[i], units[i + 1], group_size, groups, optimization, iBatch) ||
         !cEncoder.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      index++;
      resblock = new CNeuronSpikeResNeXtBlock();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, chanels[i], chanels[i + 1], units[i], units[i + 1], group_size, groups, optimization, iBatch) ||
         !cContext.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      index++;
     }
//---
   gru = new CNeuronSpikeConvGRU2D();
   if(!gru ||
      !gru.Init(0, index, OpenCL, units[layers], chanels[layers], chanels[layers], optimization, iBatch) ||
      !cContext.Add(gru))
     {
      DeleteObj(gru)
      ReturnFalse;
     }
   index++;
//---
   correlation = new CNeuronMultiScaleStackCorrelation();
   if(!correlation ||
      !correlation.Init(0, index, OpenCL, stack_size, chanels[layers], units[layers], levels, optimization, iBatch) ||
      !cCorrelation.Add(correlation))
     {
      DeleteObj(correlation)
      ReturnFalse;
     }
   index++;
   resblock = new CNeuronSpikeResNeXtBlock();
   if(!resblock ||
      !resblock.Init(0, index, OpenCL, correlation.Neurons() / units[layers], chanels[layers], units[layers], units[layers], group_size, groups, optimization, iBatch) ||
      !cCorrelation.Add(resblock))
     {
      DeleteObj(resblock)
      ReturnFalse;
     }
   index++;
//---
   if(!cConcatenated.Init(0, index, OpenCL, 3 * chanels[layers]*units[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   if(!cGRU.Init(0, index, OpenCL, units[layers], 3 * chanels[layers], chanels[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   if(!cResidual.Init(0, index, OpenCL, chanels[0], chanels[layers], units[0], units[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, Neurons(), optimization, iBatch) ||
      !cG.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   if(!neuron.SetGradient(cGRU.getGradient(), true) ||
      !cResidual.SetGradient(cGRU.getGradient(), true))
      ReturnFalse;
//---
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, Neurons(), iBatch, optimization) ||
      !cG.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = NeuronOCL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = cEncoder[-1];
   for(int i = 0; i < cCorrelation.Total(); i++)
     {
      current = cCorrelation[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   uint units = cGRU.GetUnits();
   if(!Concat(cEncoder[-1].getOutput(), cContext[-1].getOutput(), cCorrelation[-1].getOutput(), cConcatenated.getOutput(),
              cEncoder[-1].Neurons() / units, cContext[-1].Neurons() / units, cCorrelation[-1].Neurons() / units, units))
      ReturnFalse;
   if(!cGRU.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
//---
   current = cG[0];
   if(!current ||
      !SumAndNormalize(cGRU.getOutput(), cResidual.getOutput(), current.getOutput(), cGRU.GetChanelsOut(), false, 0, 0, 0, 1))
      ReturnFalse;
   prev = current;
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cG[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* next = cG[-1];
   for(int i = cG.Total() - 2; i >= 0; i--)
     {
      current = cG[i];
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
      next = current;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.CalcHiddenGradients(cGRU.AsObject()))
      ReturnFalse;
   uint units = cGRU.GetChanelsOut();
   if(!DeConcat(cEncoder[-1].getGradient(), cContext[-1].getGradient(), cCorrelation[-1].getGradient(), cConcatenated.getGradient(),
                cEncoder[-1].Neurons() / units, cContext[-1].Neurons() / units, cCorrelation[-1].Neurons() / units, units))
      ReturnFalse;
//---
   next = cCorrelation[-1];
   for(int i = cCorrelation.Total() - 2; i >= 0; i--)
     {
      current = cCorrelation[i];
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
      next = current;
     }
//---
   current = cEncoder[-1];
   if(!current)
      ReturnFalse;
   CBufferFloat* temp = current.getGradient();
   if(!current.SetGradient(current.getPrevOutput(), false) ||
      !current.CalcHiddenGradients(next) ||
      !SumAndNormalize(temp, current.getGradient(), temp, temp.Total() / units, false, 0, 0, 0, 1) ||
      !current.SetGradient(temp, false))
      ReturnFalse;
//---
   next = current;
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      current = cEncoder[i];
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
      next = current;
     }
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
//---
   next = cContext[-1];
   for(int i = cContext.Total() - 2; i >= 0; i--)
     {
      current = cContext[i];
      if(!current ||
         !current.CalcHiddenGradients(next))
         ReturnFalse;
      next = current;
     }
   temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(next) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = NeuronOCL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = cEncoder[-1];
   for(int i = 0; i < cCorrelation.Total(); i++)
     {
      current = cCorrelation[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!cGRU.UpdateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
//---
   prev = cG[0];
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronRAFT* Source = source;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cContext.WeightsUpdate(Source.cContext.AsObject(), tau))
      ReturnFalse;
   if(!cCorrelation.WeightsUpdate(Source.cCorrelation.AsObject(), tau))
      ReturnFalse;
   if(!cG.WeightsUpdate(Source.cG.AsObject(), tau))
      ReturnFalse;
   if(!cGRU.WeightsUpdate(Source.cGRU.AsObject(), tau))
      ReturnFalse;
   if(!cResidual.WeightsUpdate(Source.cResidual.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
//---
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cCorrelation.Save(file_handle))
      ReturnFalse;
   if(!cG.Save(file_handle))
      ReturnFalse;
   if(!cGRU.Save(file_handle))
      ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
//---
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
   if(!cContext.Load(file_handle))
      ReturnFalse;
   if(!cCorrelation.Load(file_handle))
      ReturnFalse;
   if(!cG.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGRU.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.Init(0, 0, OpenCL, cContext[-1].Neurons() + cEncoder[-1].Neurons() + cCorrelation[-1].Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronRAFT::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
   cG.SetOpenCL(OpenCL);
   cGRU.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronRAFT::TrainMode(bool flag)
  {
   CNeuronSpikeActivation::TrainMode(flag);
   cEncoder.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cCorrelation.TrainMode(bTrain);
   cG.TrainMode(bTrain);
   cGRU.TrainMode(bTrain);
   cResidual.TrainMode(bTrain);
   cConcatenated.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRAFT::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
//---
   if(!cEncoder.ClearStates())
      ReturnFalse;
   if(!cContext.ClearStates())
      ReturnFalse;
   if(!cCorrelation.ClearStates())
      ReturnFalse;
   if(!cG.ClearStates())
      ReturnFalse;
   if(!cGRU.Clear())
      ReturnFalse;
   if(!cResidual.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint segments, uint segment_size, uint dimension,
      uint variables, uint levels,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronAddToStack::Init(numOutputs, myIndex, open_cl, segments, segment_size * levels,
                               variables, optimization_type, batch))
      ReturnFalse;
   if(!cCorrelation.Init(0, 0, OpenCL, segment_size, dimension, variables, levels, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cCorrelation.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronAddToStack::feedForward(cCorrelation.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronAddToStack::calcInputGradients(cCorrelation.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cCorrelation.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::Save(const int file_handle)
  {
   if(!CNeuronAddToStack::Save(file_handle))
      ReturnFalse;
   if(!cCorrelation.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::Load(const int file_handle)
  {
   if(!CNeuronAddToStack::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCorrelation.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMultScalStackCorrelBySegments::Clear(void)
  {
   if(!CNeuronAddToStack::Clear())
      ReturnFalse;
   if(!cCorrelation.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultScalStackCorrelBySegments::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronAddToStack::SetOpenCL(obj);
   cCorrelation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMultScalStackCorrelBySegments::SetActivationFunction(ENUM_ACTIVATION value)
  {
   cCorrelation.SetActivationFunction(value);
   activation = (ENUM_ACTIVATION)cCorrelation.Activation();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint units, uint &windows[], uint window_out,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(windows.Size() != 2)
      ReturnFalse;
//---
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, 2 * window_out, 2 * window_out,
                            window_out, units, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   cEncoder.Clear();
   cEncoder.SetOpenCL(OpenCL);
   CNeuronMultiWindowsConvOCL*   mw_conv  =  NULL;
   CNeuronBatchNormOCL*          normal   =  NULL;
//---
   mw_conv = new CNeuronMultiWindowsConvOCL();
   if(!mw_conv ||
      !mw_conv.Init(0, index, OpenCL, windows, 2 * window_out, units, 1, optimization, iBatch) ||
      !cEncoder.Add(mw_conv))
     {
      DeleteObj(mw_conv)
      ReturnFalse;
     }
   mw_conv.SetActivationFunction(SoftPlus);
   index++;
   normal = new CNeuronBatchNormOCL();
   if(!normal ||
      !normal.Init(0, index, OpenCL, mw_conv.Neurons(), iBatch, optimization) ||
      !cEncoder.Add(normal))
     {
      DeleteObj(normal)
      ReturnFalse;
     }
   index++;
//---
   uint temp[] = {2 * window_out, 2 * window_out};
   mw_conv = new CNeuronMultiWindowsConvOCL();
   if(!mw_conv ||
      !mw_conv.Init(0, index, OpenCL, temp, window_out, units, 1, optimization, iBatch) ||
      !cEncoder.Add(mw_conv))
     {
      DeleteObj(mw_conv)
      ReturnFalse;
     }
   mw_conv.SetActivationFunction(SoftPlus);
   index++;
   normal = new CNeuronBatchNormOCL();
   if(!normal ||
      !normal.Init(0, index, OpenCL, mw_conv.Neurons(), iBatch, optimization) ||
      !cEncoder.Add(normal))
     {
      DeleteObj(normal)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronConvOCL::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronConvOCL::calcInputGradients(cEncoder[-1]))
      ReturnFalse;
   CNeuronBaseOCL* next = cEncoder[-1];
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cEncoder[i];
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
bool CNeuronTMAMFE::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronConvOCL::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMFE::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronConvOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronTMAMFE* Source = source;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTMAMFE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTMAMFE::TrainMode(bool flag)
  {
   CNeuronConvOCL::TrainMode(flag);
   cEncoder.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint window, uint window_key, uint units,
                         uint segments, uint heads,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units * segments,
                            optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   uint index = 0;
   if(!cCrossAttention.Init(0, index, OpenCL, window, window_key, units * segments,
                            heads, window, segments, optimization, iBatch))
      ReturnFalse;
   index++;
//---
   cProjection.Clear();
   cProjection.SetOpenCL(OpenCL);
//---
   CNeuronBaseOCL*      neuron =  NULL;
   CNeuronConvOCL*      conv  =  NULL;
   CNeuronBatchNormOCL* norm  =  NULL;
//--- Concatenated
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 2 * Neurons(), optimization, iBatch) ||
      !cProjection.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
//--- Projection
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * window, 2 * window, window, units * segments, 1, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cProjection.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   index++;
//--- Feed Forward
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, 2 * window, units * segments, 1, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * window, 2 * window, window, units * segments, 1, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   conv.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!cCrossAttention.FeedForward(NeuronOCL, NeuronOCL.getOutput()))
      ReturnFalse;
   CNeuronBaseOCL* neuron = cProjection[0];
   uint window = cCrossAttention.GetWindow();
   uint units = cCrossAttention.GetUnits();
   uint segments = cCrossAttention.GetUnitsKV();
   if(!neuron ||
      !Concat(NeuronOCL.getOutput(), cCrossAttention.getOutput(), neuron.getOutput(),
              window, window, units))
      ReturnFalse;
//---
   for(int i = 1; i < cProjection.Total(); i++)
     {
      neuron = cProjection[i];
      if(!neuron ||
         !neuron.FeedForward(cProjection[i - 1]))
         ReturnFalse;
     }
//---
   if(!SumAndNormalize(NeuronOCL.getOutput(), neuron.getOutput(), Output, window * segments, true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = cProjection[-1];
   if(!neuron ||
      !DeActivation(neuron.getOutput(), neuron.getGradient(), Gradient, neuron.Activation()))
      ReturnFalse;
   for(int i = cProjection.Total() - 2; i >= 0; i--)
     {
      neuron = cProjection[0];
      if(!neuron ||
         !neuron.CalcHiddenGradients(cProjection[i + 1]))
         ReturnFalse;
     }
//---
   uint window = cCrossAttention.GetWindow();
   uint units = cCrossAttention.GetUnits();
   if(!DeConcat(PrevOutput, cCrossAttention.getGradient(), neuron.getGradient(),
                window, window, units))
      ReturnFalse;
   Deactivation(cCrossAttention)
//---
   if(!SumAndNormalize(PrevOutput, Gradient, PrevOutput, window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), PrevOutput, PrevOutput, NeuronOCL.Activation()))
         ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cCrossAttention.AsObject(), NeuronOCL.getOutput(),
                                     cProjection[-1].getPrevOutput(), (ENUM_ACTIVATION)NeuronOCL.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cProjection[-1].getPrevOutput(), NeuronOCL.getGradient(), window, false, 0, 0, 0, 1) ||
      !SumAndNormalize(NeuronOCL.getGradient(), PrevOutput, NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cCrossAttention.UpdateInputWeights(NeuronOCL, NeuronOCL.getOutput()))
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = cProjection[0];
   for(int i = 1; i < cProjection.Total(); i++)
     {
      neuron = cProjection[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(cProjection[i - 1]))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronTMAMPA* Source = source;
   if(!cCrossAttention.WeightsUpdate(Source.cCrossAttention.AsObject(), tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cCrossAttention.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMAMPA::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cCrossAttention.AsObject()))
      ReturnFalse;
   if(!cProjection.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTMAMPA::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cCrossAttention.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                      uint &units[], uint &chanels[], uint group_size, uint groups,
                      uint segments, uint segment_size, uint levels, uint window_key,
                      uint heads, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(units.Size() != chanels.Size())
      ReturnFalse;
   uint layers = units.Size() - 1;
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units[layers]*chanels[layers], optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   CNeuronSpikeResNeXtBlock*              resblock    = NULL;
   CNeuronSpikeConvGRU2D*                 gru         = NULL;
   CNeuronMultScalStackCorrelBySegments*  correlation = NULL;
   CNeuronTMAMFE*                         mfe         = NULL;
   CNeuronTMAMPA*                         mpa         = NULL;
   CNeuronTransposeRCDOCL*                transpose   = NULL;
   CNeuronBaseOCL*                        neuron      = NULL;
   CNeuronBatchNormOCL*                   norm        = NULL;
//---
   cEncoder.Clear();
   cContext.Clear();
   cCorrelation.Clear();
   cG.Clear();
   cEncoder.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
   cG.SetOpenCL(OpenCL);
//---
   for(uint i = 0; i < layers; i++)
     {
      resblock = new CNeuronSpikeResNeXtBlock();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, chanels[i], chanels[i + 1], units[i], units[i + 1], group_size, groups, optimization, iBatch) ||
         !cEncoder.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      index++;
      resblock = new CNeuronSpikeResNeXtBlock();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, chanels[i], chanels[i + 1], units[i], units[i + 1], group_size, groups, optimization, iBatch) ||
         !cContext.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      index++;
     }
//---
   gru = new CNeuronSpikeConvGRU2D();
   if(!gru ||
      !gru.Init(0, index, OpenCL, units[layers], chanels[layers], chanels[layers], optimization, iBatch) ||
      !cContext.Add(gru))
     {
      DeleteObj(gru)
      ReturnFalse;
     }
   index++;
//---
   if(!cStatesStack.Init(0, index, OpenCL, segments, chanels[layers], units[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   correlation = new CNeuronMultScalStackCorrelBySegments();
   if(!correlation ||
      !correlation.Init(0, index, OpenCL, segments, segment_size, chanels[layers], units[layers], levels, optimization, iBatch) ||
      !cCorrelation.Add(correlation))
     {
      DeleteObj(correlation)
      ReturnFalse;
     }
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, correlation.Neurons() + cStatesStack.Neurons(), optimization, iBatch) ||
      !cCorrelation.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   index++;
     {
      uint temp[] = { cStatesStack.GetDimension(), correlation.GetDimension() };
      mfe = new CNeuronTMAMFE();
      if(!mfe ||
         !mfe.Init(0, index, OpenCL, segments * units[layers], temp, chanels[layers], optimization, iBatch) ||
         !cCorrelation.Add(mfe))
        {
         DeleteObj(mfe)
         ReturnFalse;
        }
     }
   index++;
   transpose = new CNeuronTransposeRCDOCL();
   if(!transpose ||
      !transpose.Init(0, index, OpenCL, units[layers], segments, chanels[layers], optimization, iBatch) ||
      !cCorrelation.Add(transpose))
     {
      DeleteObj(transpose)
      ReturnFalse;
     }
   index++;
   for(int i = 0; i < 2; i++)
     {
      mpa = new CNeuronTMAMPA();
      if(!mpa ||
         !mpa.Init(0, index, OpenCL, chanels[layers], window_key, segments, units[layers], heads, optimization, iBatch) ||
         !cCorrelation.Add(mpa))
        {
         DeleteObj(mpa)
         ReturnFalse;
        }
      index++;
     }
   transpose = new CNeuronTransposeRCDOCL();
   if(!transpose ||
      !transpose.Init(0, index, OpenCL, segments, units[layers], chanels[layers], optimization, iBatch) ||
      !cCorrelation.Add(transpose))
     {
      DeleteObj(transpose)
      ReturnFalse;
     }
   index++;
//---
   if(!cConcatenated.Init(0, index, OpenCL, units[layers] * ((segments + 2) * chanels[layers]), optimization, iBatch))
      ReturnFalse;
   index++;
//---
   if(!cGRU.Init(0, index, OpenCL, units[layers], (segments + 2) * chanels[layers], chanels[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   if(!cResidual.Init(0, index, OpenCL, chanels[0], chanels[layers], units[0], units[layers], optimization, iBatch))
      ReturnFalse;
   index++;
//---
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, Neurons(), optimization, iBatch) ||
      !cG.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   if(!neuron.SetGradient(cGRU.getGradient(), true) ||
      !cResidual.SetGradient(cGRU.getGradient(), true))
      ReturnFalse;
//---
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, Neurons(), iBatch, optimization) ||
      !cG.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = NeuronOCL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!cStatesStack.FeedForward(cEncoder[-1]))
      ReturnFalse;
   CNeuronMultScalStackCorrelBySegments* correlation = cCorrelation[0];
   if(!correlation ||
      !correlation.FeedForward(cEncoder[-1]))
      ReturnFalse;
//---
   uint variables = cStatesStack.GetVariables();
   uint segments = cStatesStack.GetStackSize();
   uint dimension = cStatesStack.GetDimension();
   uint corr_dim = correlation.GetDimension();
//---
   prev = cCorrelation[1];
   if(!Concat(cStatesStack.getOutput(), correlation.getOutput(), prev.getOutput(),
              dimension, corr_dim, variables * segments))
      ReturnFalse;
   for(int i = 2; i < cCorrelation.Total(); i++)
     {
      current = cCorrelation[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!Concat(cEncoder[-1].getOutput(), cContext[-1].getOutput(), cCorrelation[-1].getOutput(),
              cConcatenated.getOutput(), dimension, dimension, segments * dimension, variables))
      ReturnFalse;
   if(!cGRU.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
//---
   current = cG[0];
   if(!current ||
      !SumAndNormalize(cGRU.getOutput(), cResidual.getOutput(), current.getOutput(), cGRU.GetChanelsOut(), false, 0, 0, 0, 1))
      ReturnFalse;
   prev = current;
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cG[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = cG.Total() - 2; i >= 0; i--)
     {
      curr = cG[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cG[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.CalcHiddenGradients(cGRU.AsObject()))
      ReturnFalse;
//---
   uint variables = cStatesStack.GetVariables();
   uint segments = cStatesStack.GetStackSize();
   uint dimension = cStatesStack.GetDimension();
//---
   if(!cEncoder[-1] || !cContext[-1] || !cCorrelation[-1] ||
      !DeConcat(cEncoder[-1].getGradient(), cContext[-1].getGradient(), cCorrelation[-1].getGradient(),
                cConcatenated.getGradient(), dimension, dimension, segments * dimension, variables))
      ReturnFalse;
   Deactivation(cContext[-1])
   Deactivation(cCorrelation[-1])
   if(cEncoder[-1].Activation() != None)
      if(!DeActivation(cEncoder[-1].getOutput(), cEncoder[-1].getPrevOutput(),
                       cEncoder[-1].getGradient(), cEncoder[-1].Activation()))
         ReturnFalse;
//---
   for(int i = cCorrelation.Total() - 2; i >= 1; i--)
     {
      curr = cCorrelation[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cCorrelation[i + 1]))
         ReturnFalse;
     }
   CNeuronMultScalStackCorrelBySegments* correlation = cCorrelation[0];
   uint corr_dim = correlation.GetDimension();
   if(!DeConcat(cStatesStack.getGradient(), correlation.getGradient(), curr.getGradient(),
                dimension, corr_dim, variables * segments))
      ReturnFalse;
   curr = cEncoder[-1];
   if(!curr.CalcHiddenGradients(cStatesStack.AsObject()) ||
      !SumAndNormalize(curr.getGradient(), curr.getPrevOutput(),
                       curr.getPrevOutput(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!curr.CalcHiddenGradients(correlation) ||
      !SumAndNormalize(curr.getGradient(), curr.getPrevOutput(),
                       curr.getGradient(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cEncoder[i + 1]))
         ReturnFalse;
     }
   if(!NeuronOCL.CalcHiddenGradients(curr))
      ReturnFalse;
//---
   for(int i = cContext.Total() - 2; i >= 0; i--)
     {
      curr = cContext[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cContext[i + 1]))
         ReturnFalse;
     }
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(curr) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = NeuronOCL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   CNeuronMultScalStackCorrelBySegments* correlation = cCorrelation[0];
   if(!correlation ||
      !correlation.UpdateInputWeights(cEncoder[-1]))
      ReturnFalse;
//---
   prev = cCorrelation[1];
   for(int i = 2; i < cCorrelation.Total(); i++)
     {
      current = cCorrelation[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   if(!cGRU.UpdateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
//---
   prev = cG[0];
   for(int i = 1; i < cG.Total(); i++)
     {
      current = cG[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::Save(const int file_handle)
  {
   if(!CNeuronRAFT::Save(file_handle))
      ReturnFalse;
   if(!cStatesStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::Load(const int file_handle)
  {
   if(!CNeuronRAFT::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStatesStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTMA::Clear(void)
  {
   if(!CNeuronRAFT::Clear())
      ReturnFalse;
   if(!cStatesStack.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTMA::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronRAFT::SetOpenCL(obj);
   cStatesStack.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint units, uint &chanels_in[], uint chanels_out,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels_in.Size() != 2)
      ReturnFalse;
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units * chanels_out,
                                    optimization_type, batch))
      ReturnFalse;
//--- Concatenate [2*F0,S0,M0]*Units
   uint index = 0;
   if(!cConcatenated[0].Init(0, index, OpenCL, (2 * chanels_in[0] + chanels_in[1] + chanels_out)*units,
                             optimization, iBatch))
      ReturnFalse;
   cConcatenated[0].SetActivationFunction(None);
//--- [6*F,3*S,3*M]*Units
   index++;
   uint windows[] = { chanels_in[0], chanels_in[0], chanels_out, chanels_in[1] };
   if(!cFSM.Init(0, index, OpenCL, windows, 3 * chanels_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cFSM.SetActivationFunction(None);
//--- [S,M]*Units
   index++;
   if(!cSM.Init(0, index, OpenCL, 2 * chanels_out * units, optimization, iBatch))
      ReturnFalse;
   cSM.SetActivationFunction(None);
//--- [F,S,M]*Units
   index++;
   if(!cConcatenated[1].Init(0, index, OpenCL, (chanels_in[0] + 2 * chanels_out)*units,
                             optimization, iBatch))
      ReturnFalse;
//--- [S]*Units
   index++;
   if(!cOs.Init(0, index, OpenCL, chanels_in[0] + 2 * chanels_out, chanels_in[0] + 2 * chanels_out,
                chanels_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cOs.SetActivationFunction(SIGMOID);
//--- [S]*Units
   index++;
   if(!cOt.Init(0, index, OpenCL, 2 * chanels_out, 2 * chanels_out, chanels_out, units, 1,
                optimization, iBatch))
      ReturnFalse;
   cOt.SetActivationFunction(TANH);
//--- [S]*Units
   index++;
   if(!cFo.Init(0, index, OpenCL, chanels_out * units, optimization, iBatch))
      ReturnFalse;
   cFo.SetActivationFunction(None);
//--- [S]*Units
   index++;
   if(!cS.Init(0, index, OpenCL, units, 2, chanels_out, optimization, iBatch))
      ReturnFalse;
   cS.SetActivationFunction(None);
//--- [S]*Units
   index++;
   if(!cNorm.Init(0, index, OpenCL, Neurons(), iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
//---
   uint Fdimension = cFSM.GetWindow(0);
   uint Sdimension = cS.GetDimension();
   uint Mdimension = cFSM.GetWindow(3);
   uint units = cFSM.GetUnits();
//---
   if(!cS.FeedForward(cSM.AsObject()))
      ReturnFalse;
//---
   if(!cSM.SwapOutputs())
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), NeuronOCL.getOutput(), cS.getOutput(), SecondInput,
              cConcatenated[0].getOutput(), Fdimension, Fdimension, Sdimension, Mdimension, units))
      ReturnFalse;
   if(!cFSM.FeedForward(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!GRU())
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), cSM.getOutput(), cConcatenated[1].getOutput(),
              Fdimension, 2 * Sdimension, units))
      ReturnFalse;
   if(!cOs.FeedForward(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!cOt.FeedForward(cSM.AsObject()))
      ReturnFalse;
   if(!ElementMult(cOs.getOutput(), cOt.getOutput(), cFo.getOutput()))
      ReturnFalse;
   if(!cNorm.FeedForward(cFo.AsObject()))
      ReturnFalse;
//---
   return CNeuronSpikeActivation::feedForward(cNorm.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::GRU(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { cFSM.GetUnits(),
                                cSM.Neurons() / cFSM.GetUnits()
                              };
   uint kernel = def_k_GRU;
   setBuffer(kernel, def_k_gru_XH, cFSM.getOutputIndex())
   setBuffer(kernel, def_k_gru_prev_state, cSM.getPrevOutIndex())
   setBuffer(kernel, def_k_gru_outputs, cSM.getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cSM.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::GRU_grad(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { cFSM.GetUnits(),
                                cSM.Neurons() / cFSM.GetUnits()
                              };
   uint kernel = def_k_GRU_Grad;
   setBuffer(kernel, def_k_gru_gr_XH, cFSM.getOutputIndex())
   setBuffer(kernel, def_k_gru_gr_XH_gr, cFSM.getGradientIndex())
   setBuffer(kernel, def_k_gru_gr_prev_state, cSM.getPrevOutIndex())
   setBuffer(kernel, def_k_gru_gr_outputs_gr, cSM.getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cFSM.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient || !SecondInput)
      ReturnFalse;
//---
   uint Fdimension = cFSM.GetWindow(0);
   uint Sdimension = cS.GetDimension();
   uint Mdimension = cFSM.GetWindow(3);
   uint units = cFSM.GetUnits();
//---
   if(!CNeuronSpikeActivation::calcInputGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cFo.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!ElementMultGrad(cOs.getOutput(), cOs.getGradient(),
                       cOt.getOutput(), cOt.getGradient(),
                       cFo.getGradient(), cOs.Activation(), cOt.Activation()))
      ReturnFalse;
   if(!cSM.CalcHiddenGradients(cOt.AsObject()))
      ReturnFalse;
   if(!cConcatenated[1].CalcHiddenGradients(cOs.AsObject()))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), cConcatenated[1].getPrevOutput(),
                cConcatenated[1].getGradient(), Fdimension, 2 * Sdimension, units))
      ReturnFalse;
   if(cSM.Activation() != None)
      if(!DeActivation(cSM.getOutput(), cConcatenated[1].getPrevOutput(),
                       cConcatenated[1].getPrevOutput(), cSM.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cSM.getGradient(), cConcatenated[1].getGradient(), cSM.getGradient(),
                       2 * Sdimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!GRU_grad())
      ReturnFalse;
   if(!cConcatenated[0].CalcHiddenGradients(cFSM.AsObject()))
      ReturnFalse;
   if(!DeConcat(cConcatenated[0].getPrevOutput(), cConcatenated[1].getPrevOutput(),
                cS.getPrevOutput(), SecondGradient, cConcatenated[0].getGradient(),
                Fdimension, Fdimension, Sdimension, Mdimension, units))
      ReturnFalse;
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cConcatenated[0].getPrevOutput(),
                       NeuronOCL.getGradient(), Fdimension, false, 0, 0, 0, 1) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cConcatenated[1].getPrevOutput(),
                       NeuronOCL.getGradient(), Fdimension, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cFSM.UpdateInputWeights(cConcatenated[0].AsObject()))
      ReturnFalse;
   if(!cOs.UpdateInputWeights(cConcatenated[1].AsObject()))
      ReturnFalse;
   if(!cOt.UpdateInputWeights(cSM.AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cFo.AsObject()))
      ReturnFalse;
//---
   return CNeuronSpikeActivation::updateInputWeights(cNorm.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSTConvGRU* Source = source;
   if(!cFSM.WeightsUpdate(Source.cFSM.AsObject(), tau))
      ReturnFalse;
   if(!cOs.WeightsUpdate(Source.cOs.AsObject(), tau))
      ReturnFalse;
   if(!cOt.WeightsUpdate(Source.cOt.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
   if(!cFSM.Save(file_handle))
      ReturnFalse;
   if(!cOs.Save(file_handle))
      ReturnFalse;
   if(!cOt.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFSM.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cOs.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cOt.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
//---
   uint Fdimension = cFSM.GetWindow(0);
   uint Sdimension = cFSM.GetWindow(2);
   uint Mdimension = cFSM.GetWindow(3);
   uint units = cFSM.GetUnits();
//--- Concatenate [2*F0,S0,M0]*Units
   uint index = 0;
   if(!cConcatenated[0].Init(0, index, OpenCL, (2 * Fdimension + Sdimension + Mdimension)*units,
                             optimization, iBatch))
      ReturnFalse;
   cConcatenated[0].SetActivationFunction(None);
//--- [S,M]*Units
   index += 2;
   if(!cSM.Init(0, index, OpenCL, 2 * Sdimension * units, optimization, iBatch))
      ReturnFalse;
   cSM.SetActivationFunction(None);
//--- [F,S,M]*Units
   index++;
   if(!cConcatenated[1].Init(0, index, OpenCL, cOs.GetWindow()*units,
                             optimization, iBatch))
      ReturnFalse;
//--- [S]*Units
   index += 3;
   if(!cFo.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cFo.SetActivationFunction(None);
//--- [S]*Units
   index++;
   if(!cS.Init(0, index, OpenCL, units, 2, Sdimension, optimization, iBatch))
      ReturnFalse;
   cS.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTConvGRU::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
   if(!cSM.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSTConvGRU::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
//---
   cFSM.SetOpenCL(OpenCL);
   cOs.SetOpenCL(OpenCL);
   cOt.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
   for(uint i = 0; i < cConcatenated.Size(); i++)
      cConcatenated[i].SetOpenCL(OpenCL);
   cSM.SetOpenCL(OpenCL);
   cFo.SetOpenCL(OpenCL);
   cS.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &chanels[],
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint layers = chanels.Size() - 1;
   if(layers < 2)
      ReturnFalse;
   if(!CNeuronSpikeConvGRU::Init(numOutputs, myIndex, open_cl, units, chanels[layers - 1],
                                 chanels[layers], optimization_type, batch))
      ReturnFalse;
//---
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
//---
   uint index = 0;
   uint windows[] = { chanels[0], chanels[layers] };
   CNeuronSpikeSTConvGRU* st_gru = new CNeuronSpikeSTConvGRU();
   if(!st_gru ||
      !st_gru.Init(0, index, OpenCL, units, windows, chanels[1], optimization, iBatch) ||
      !cFlow.Add(st_gru))
     {
      DeleteObj(st_gru)
      ReturnFalse;
     }
//---
   CNeuronSpikeConvGRU* gru = NULL;
   for(uint i = 1; i < layers - 1; i++)
     {
      CNeuronSpikeConvGRU* gru = new CNeuronSpikeConvGRU();
      if(!gru ||
         !gru.Init(0, index, OpenCL, units, chanels[i], chanels[i + 1], optimization, iBatch) ||
         !cFlow.Add(gru))
        {
         DeleteObj(gru)
         ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *curr = cFlow[0];
   if(!curr ||
      !curr.FeedForward(prev, cH.getOutput()))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   return CNeuronSpikeConvGRU::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvGRU::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL *next = cFlow[-1];
   CNeuronBaseOCL *curr = NULL;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next, cH.getPrevOutput(), cH.getGradient(),
                                     (ENUM_ACTIVATION)cH.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *curr = cFlow[0];
   if(!curr ||
      !curr.UpdateInputWeights(prev, cH.getOutput()))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   return CNeuronSpikeConvGRU::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvGRU::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronFERENet* Source = source;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvGRU::Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFERENet::Clear(void)
  {
   if(!CNeuronSpikeConvGRU::Clear())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFERENet::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvGRU::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::GRU(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { cXH.GetUnits(),
                                cFlow.GetFilters()
                              };
   uint kernel = def_k_GRU;
   setBuffer(kernel, def_k_gru_XH, cXH.getOutputIndex())
   setBuffer(kernel, def_k_gru_prev_state, cFlow.getOutputIndex())
   setBuffer(kernel, def_k_gru_outputs, getOutputIndex())
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
bool CNeuronFGDModule::GRU_grad(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = { cXH.GetUnits(),
                                cFlow.GetFilters()
                              };
   uint kernel = def_k_GRU_Grad;
   setBuffer(kernel, def_k_gru_gr_XH, cXH.getOutputIndex())
   setBuffer(kernel, def_k_gru_gr_XH_gr, cXH.getGradientIndex())
   setBuffer(kernel, def_k_gru_gr_prev_state, cFlow.getOutputIndex())
   setBuffer(kernel, def_k_gru_gr_outputs_gr, getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cXH.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint units,
                            uint main_dimension, uint flow_dimension, uint dimension_out,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * dimension_out,
                            optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   uint index = 0;
   if(!cNorm.Init(0, index, OpenCL, main_dimension * units, iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   index++;
   if(!cD.Init(0, index, OpenCL, main_dimension, main_dimension, dimension_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cD.SetActivationFunction(TANH);
   index++;
   if(!cFlowIn.Init(0, index, OpenCL, flow_dimension * units, optimization, iBatch))
      ReturnFalse;
   DeleteObj(cFlowIn.getOutput())
   cFlowIn.SetActivationFunction(None);
   index++;
   if(!cFlow.Init(0, index, OpenCL, flow_dimension, flow_dimension, dimension_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cFlow.SetActivationFunction(None);
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, 2 * dimension_out * units, optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
   index++;
   uint windows[] = { dimension_out, dimension_out };
   if(!cXH.Init(0, index, OpenCL, windows, 3 * dimension_out, units, 1, optimization, iBatch))
      ReturnFalse;
   cXH.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!cFlowIn.SetOutput(SecondInput, false))
      ReturnFalse;
//---
   if(!cNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cD.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!cFlow.FeedForward(cFlowIn.AsObject()))
      ReturnFalse;
   if(!Concat(cD.getOutput(), cFlow.getOutput(), cConcatenated.getOutput(),
              cD.GetFilters(), cFlow.GetFilters(), cFlow.GetUnits()))
      ReturnFalse;
   if(!cXH.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
   if(!GRU())
      ReturnFalse;
   if(!cD.getGradient().Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
   if(!cFlowIn.SetGradient(SecondGradient, true))
      ReturnFalse;
   cFlowIn.SetActivationFunction(SecondActivation);
//---
   if(!GRU_grad())
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cXH.AsObject()))
      ReturnFalse;
   if(!DeConcat(cD.getPrevOutput(), cFlow.getGradient(), cConcatenated.getGradient(),
                cD.GetFilters(), cFlow.GetFilters(), cFlow.GetUnits()))
      ReturnFalse;
   if(!SumAndNormalize(cD.getGradient(), cD.getPrevOutput(), cD.getGradient(), cD.GetFilters(), false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(cFlow)
   Deactivation(cD)
   if(!cFlowIn.CalcHiddenGradients(cFlow.AsObject()))
      ReturnFalse;
   if(!cNorm.CalcHiddenGradients(cD.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cNorm.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cD.UpdateInputWeights(cNorm.AsObject()))
      ReturnFalse;
   if(!cFlow.UpdateInputWeights(cFlowIn.AsObject()))
      ReturnFalse;
   if(!cXH.UpdateInputWeights(cConcatenated.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronFGDModule* Source = source;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
   if(!cD.WeightsUpdate(Source.cD.AsObject(), tau))
      ReturnFalse;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
   if(!cXH.WeightsUpdate(Source.cXH.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(!cD.Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
   if(!cXH.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFGDModule::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cD.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFlow.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cXH.AsObject()))
      ReturnFalse;
//---
   if(!cFlowIn.Init(0, 2, OpenCL, cFlow.GetWindow() * cFlow.GetUnits(), optimization, iBatch))
      ReturnFalse;
   cFlowIn.SetActivationFunction(None);
   if(!cConcatenated.Init(0, 4, OpenCL, cFlow.Neurons() + cD.Neurons(), optimization, iBatch))
      ReturnFalse;
   cConcatenated.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFGDModule::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cNorm.SetOpenCL(OpenCL);
   cD.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
   cXH.SetOpenCL(OpenCL);
   cFlowIn.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint units, uint &chanels[],
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels.Size() < 2)
      ReturnFalse;
   if(!CNeuronSpikeActivation::Init(numOutputs, myIndex, open_cl, units * chanels[0], optimization_type, batch))
      ReturnFalse;
   uint layers = chanels.Size() - 1;
//--- Encoder
   int index = 0;
   if(!cEncoder.Init(0, index, OpenCL, units, chanels, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNeck.Init(0, index, OpenCL, chanels[layers], units, 32, 8, optimization, iBatch))
      ReturnFalse;
//--- Decoder
   cDecoder.Clear();
   cDecoder.SetOpenCL(OpenCL);
   CNeuronFGDModule* fgd = NULL;
   CNeuronBaseOCL*   concat = cNeck.AsObject();
   uint flow_dimension = chanels[0];
   for(uint i = 0; i < layers; i++)
     {
      uint main_dimension = concat.Neurons() / units;
      fgd = new CNeuronFGDModule();
      if(!fgd ||
         !fgd.Init(0, index, OpenCL, units, main_dimension, flow_dimension, chanels[layers - i - 1], optimization, iBatch) ||
         !cDecoder.Add(fgd))
        {
         DeleteObj(fgd)
         ReturnFalse;
        }
      if(i == (layers - 1))
         break;
      index++;
      CNeuronBaseOCL* encoder = cEncoder.GetLayer(-int(i + 1));
      CNeuronBaseOCL* d = fgd.GetD();
      if(!encoder)
        {
         concat = new CNeuronBaseOCL();
         if(!concat ||
            !concat.Init(0, index, OpenCL, fgd.Neurons() + d.Neurons(), optimization, iBatch) ||
            !cDecoder.Add(concat))
           {
            DeleteObj(concat)
            ReturnFalse;
           }
        }
      else
        {
         concat = new CNeuronBaseOCL();
         if(!concat ||
            !concat.Init(0, index, OpenCL, fgd.Neurons() + d.Neurons() + encoder.Neurons(), optimization, iBatch) ||
            !cDecoder.Add(concat))
           {
            DeleteObj(concat)
            ReturnFalse;
           }
        }
      index++;
     }
//---
   CNeuronBatchNormOCL* norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, fgd.Neurons(), iBatch, optimization) ||
      !cDecoder.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEncoder.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNeck.FeedForward(cEncoder.AsObject()))
      ReturnFalse;
//--- Decoder
   int layers = cDecoder.Total() / 2;
   CNeuronFGDModule* fgd = NULL;
   CNeuronBaseOCL*   d = NULL;
   CNeuronBaseOCL*   encoder = NULL;
   CNeuronBaseOCL*   prev = cNeck.AsObject();
   CNeuronBaseOCL*   flow = cDecoder[-2];
   if(!flow.SwapOutputs())
      ReturnFalse;
   uint units = cEncoder.GetUnits();
   if(!flow)
      ReturnFalse;
   for(int i = 0; i < layers; i++)
     {
      fgd = cDecoder[i * 2];
      if(!fgd ||
         !fgd.FeedForward(prev, flow.getPrevOutput()))
         ReturnFalse;
      prev = cDecoder[i * 2 + 1];
      if(!prev)
         ReturnFalse;
      if(i == (layers - 1))
        {
         if(!prev.FeedForward(fgd))
            ReturnFalse;
         break;
        }
      d = fgd.GetD();
      encoder = cEncoder[-(i + 1)];
      if(!encoder)
        {
         if(!Concat(fgd.getOutput(), d.getOutput(), prev.getOutput(),
                    fgd.Neurons() / units, d.Neurons() / units, units))
            ReturnFalse;
        }
      else
        {
         if(!Concat(fgd.getOutput(), d.getOutput(), encoder.getOutput(), prev.getOutput(),
                    fgd.Neurons() / units, d.Neurons() / units, encoder.Neurons() / units, units))
            ReturnFalse;
        }
     }
//---
   return CNeuronSpikeActivation::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeActivation::calcInputGradients(cDecoder[-1]))
      ReturnFalse;
//--- Decoder
   int layers = cDecoder.Total() / 2;
   CNeuronFGDModule* fgd = cDecoder[-2];
   CNeuronBaseOCL*   d = NULL;
   CNeuronBaseOCL*   encoder = NULL;
   CNeuronBaseOCL*   next = cDecoder[-1];
   CNeuronBaseOCL*   flow = cDecoder[-2];
   if(!fgd ||
      !fgd.CalcHiddenGradients(next))
      ReturnFalse;
   uint units = cEncoder.GetUnits();
   for(int i = layers - 2; i >= 0; i--)
     {
      next = cDecoder[i * 2 + 1];
      if(!next ||
         !next.CalcHiddenGradients(fgd, flow.getOutput(), flow.getGradient(), (ENUM_ACTIVATION)flow.Activation()))
         ReturnFalse;
      //---
      fgd = cDecoder[i * 2];
      if(!fgd)
         ReturnFalse;
      d = fgd.GetD();
      encoder = cEncoder[-(i + 1)];
      if(!encoder)
        {
         if(!DeConcat(fgd.getGradient(), d.getGradient(), next.getGradient(),
                      fgd.Neurons() / units, d.Neurons() / units, units))
            ReturnFalse;
        }
      else
        {
         if(!DeConcat(fgd.getGradient(), d.getGradient(), encoder.getGradient(), next.getGradient(),
                      fgd.Neurons() / units, d.Neurons() / units, encoder.Neurons() / units, units))
            ReturnFalse;
        }
      Deactivation(fgd);
     }
//---
   if(!cNeck.CalcHiddenGradients(fgd, flow.getOutput(), flow.getGradient(), (ENUM_ACTIVATION)flow.Activation()))
      ReturnFalse;
   if(!cEncoder.CalcHiddenGradients(cNeck.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cEncoder.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEncoder.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cNeck.UpdateInputWeights(cEncoder.AsObject()))
      ReturnFalse;
//--- Decoder
   int layers = cDecoder.Total() / 2;
   CNeuronFGDModule* fgd = NULL;
   CNeuronBaseOCL*   prev = cNeck.AsObject();
   CNeuronBaseOCL*   flow = cDecoder[-2];
   uint units = cEncoder.GetUnits();
   if(!flow)
      ReturnFalse;
   for(int i = 0; i < layers; i++)
     {
      fgd = cDecoder[i * 2];
      if(!fgd ||
         !fgd.UpdateInputWeights(prev, flow.getPrevOutput()))
         ReturnFalse;
      prev = cDecoder[i * 2 + 1];
      if(!prev)
         ReturnFalse;
      if(i == (layers - 1))
         if(!prev.UpdateInputWeights(fgd))
            ReturnFalse;
     }
//---
   return CNeuronSpikeActivation::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeActivation::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronEVMGRFlowNet* Source = source;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cNeck.WeightsUpdate(Source.cNeck.AsObject(), tau))
      ReturnFalse;
   if(!cDecoder.WeightsUpdate(Source.cDecoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::Save(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cNeck.Save(file_handle))
      ReturnFalse;
   if(!cDecoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::Load(const int file_handle)
  {
   if(!CNeuronSpikeActivation::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNeck.AsObject()))
      ReturnFalse;
   if(!cDecoder.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVMGRFlowNet::Clear(void)
  {
   if(!CNeuronSpikeActivation::Clear())
      ReturnFalse;
   if(!cEncoder.Clear())
      ReturnFalse;
   if(!cNeck.Clear())
      ReturnFalse;
   if(!cDecoder.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEVMGRFlowNet::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeActivation::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
   cNeck.SetOpenCL(OpenCL);
   cDecoder.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint window, uint step, uint window_out, uint units_count, uint variables,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, window_out * units_count * variables, batch, optimization_type))
      ReturnFalse;
   activation = None;
   if(!cActivation.Init(0, 0, OpenCL, (window - step + step * units_count)*variables, optimization, iBatch))
      ReturnFalse;
   if(!cConv.Init(0, 1, OpenCL, window, step, window_out, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cConv.SetActivationFunction(None);
   if(!cStack.Init(0, 2, OpenCL, 50, cActivation.Neurons(), 1, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cActivation.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cConv.FeedForward(cActivation.AsObject()))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::feedForward(cConv.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronBatchNormOCL::calcInputGradients(cConv.AsObject()))
      ReturnFalse;
   if(!cActivation.CalcHiddenGradients(cConv.AsObject()))
      ReturnFalse;
   if(!cStack.FeedForward(cActivation.AsObject()) ||
      !DiversityLoss(cStack.AsObject(), cStack.GetStackSize(), cStack.GetDimension(), false) ||
      !SumAndNormalize(cActivation.getGradient(), cStack.getGradient(), cActivation.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cActivation.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cActivation.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cConv.UpdateInputWeights(cActivation.AsObject()))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::updateInputWeights(cConv.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::Save(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Save(file_handle))
      ReturnFalse;
   if(!cActivation.Save(file_handle))
      ReturnFalse;
   if(!cConv.Save(file_handle))
      ReturnFalse;
   if(!cStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Append the trainable spike-convolution block graph.               |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(!CNeuronBatchNormOCL::AppendParameterFingerprint(fingerprint) ||
      !cActivation.AppendParameterFingerprint(fingerprint) ||
      !cConv.AppendParameterFingerprint(fingerprint))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::Load(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActivation.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConv.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::Clear(void)
  {
   if(!CNeuronBatchNormOCL::Clear())
      ReturnFalse;
   if(!cActivation.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeConvBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBatchNormOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeConvBlock* Source = source;
   if(!cActivation.WeightsUpdate(Source.cActivation.AsObject(), tau))
      ReturnFalse;
   if(!cConv.WeightsUpdate(Source.cConv.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeConvBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBatchNormOCL::SetOpenCL(obj);
   cActivation.SetOpenCL(OpenCL);
   cConv.SetOpenCL(OpenCL);
   cStack.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeConvBlock::TrainMode(bool flag)
  {
   CNeuronBatchNormOCL::TrainMode(flag);
   cActivation.TrainMode(bTrain);
   cConv.TrainMode(bTrain);
   cStack.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint units, uint window, uint variables,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units * variables, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
   CNeuronSpikeConvBlock *block = new CNeuronSpikeConvBlock();
   if(!block ||
      !block.Init(0, 0, OpenCL, window, window, 2 * window, units, variables, optimization, iBatch) ||
      !cFlow.Add(block))
     {
      DeleteObj(block)
      ReturnFalse;
     }
   block = new CNeuronSpikeConvBlock();
   if(!block ||
      !block.Init(0, 1, OpenCL, 2 * window, 2 * window, window, units, variables, optimization, iBatch) ||
      !cFlow.Add(block))
     {
      DeleteObj(block)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!SumAndNormalize(NeuronOCL.getOutput(), prev.getOutput(), Output, 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* next = cFlow[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next ||
      !DeActivation(next.getOutput(), next.getGradient(), Gradient, next.Activation()))
      ReturnFalse;
//---
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
   next = cFlow[-1];
   if(!DeActivation(NeuronOCL.getOutput(), next.getPrevOutput(), Gradient, NeuronOCL.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), next.getPrevOutput(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
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
bool CNeuronMSRes::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Append the trainable residual-flow graph.                          |
//+------------------------------------------------------------------+
bool CNeuronMSRes::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(!CNeuronBaseOCL::AppendParameterFingerprint(fingerprint))
      ReturnFalse;
   for(int i = 0; i < cFlow.Total(); i++)
     {
      CNeuronBaseOCL *layer = cFlow.At(i);
      if(!layer || !layer.AppendParameterFingerprint(fingerprint))
         ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   //--- cFlow is reconstructed from the serialized layer list.  It must use
   //--- the parent context while CLayer creates each nested OpenCL neuron.
   cFlow.SetOpenCL(OpenCL);
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::WeightsUpdate(CNeuronBaseOCL *source, float tau)\
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronMSRes* Source = source;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMSRes::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMSRes::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMSRes::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint units, uint window, uint heads, uint dimension_k,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   uint index = 0;
   if(!cQK.Init(0, index, OpenCL, window, window, dimension_k * heads * 2, units, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSpikeQK.Init(0, index, OpenCL, cQK.Neurons(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cQ.Init(0, index, OpenCL, cQK.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cQ.SetActivationFunction((ENUM_ACTIVATION)cSpikeQK.Activation());
   index++;
   if(!cK.Init(0, index, OpenCL, cQ.Neurons(), optimization, iBatch))
      ReturnFalse;
   cK.SetActivationFunction((ENUM_ACTIVATION)cSpikeQK.Activation());
   index++;
   uint windows[];
   if(ArrayResize(windows, heads) < (int)heads)
      ReturnFalse;
   ArrayFill(windows, 0, heads, dimension_k);
   if(!cScore.Init(0, index, OpenCL, windows, 1, units, 1, optimization, iBatch))
      ReturnFalse;
   cScore.SetActivationFunction(None);
   index++;
   if(!cSpikeScore.Init(0, index, OpenCL, cScore.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   cProject.Clear();
   cProject.SetOpenCL(OpenCL);
   index++;
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, cK.Neurons(), optimization, iBatch) ||
      !cProject.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   index++;
   CNeuronSpikeConvBlock* conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dimension_k * heads, dimension_k * heads, window, units, 1, optimization, iBatch) ||
      !cProject.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cQK.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint units = cQK.GetUnits();
   uint window = cQK.GetWindow();
   uint dimension_k = cScore.GetWindow(0);
   uint hd = cQK.GetFilters() / 2;
   uint heads = hd / dimension_k;
//---
   if(!cSpikeQK.FeedForward(cQK.AsObject()))
      ReturnFalse;
   if(!DeConcat(cQ.getOutput(), cK.getOutput(), cSpikeQK.getOutput(), hd, hd, units))
      ReturnFalse;
//---
   if(!cScore.FeedForward(cQ.AsObject()))
      ReturnFalse;
   if(!cSpikeScore.FeedForward(cScore.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = cProject[0];
   if(!neuron ||
      !ScalarToVector(cSpikeScore.getOutput(), cK.getOutput(), neuron.getOutput(), dimension_k))
      ReturnFalse;
   for(int i = 1; i < cProject.Total(); i++)
     {
      neuron = cProject[i];
      if(!neuron ||
         !neuron.FeedForward(cProject[i - 1]))
         ReturnFalse;
     }
//---
   if(!SumAndNormalize(NeuronOCL.getOutput(), neuron.getOutput(), Output, window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint units = cQK.GetUnits();
   uint window = cQK.GetWindow();
   uint dimension_k = cScore.GetWindow(0);
   uint hd = cQK.GetFilters() / 2;
   uint heads = hd / dimension_k;
//---
   CNeuronBaseOCL* neuron = cProject[-1];
   if(!neuron ||
      !DeActivation(neuron.getOutput(), neuron.getGradient(), Gradient, neuron.Activation()))
      ReturnFalse;
   for(int i = cProject.Total() - 2; i >= 0; i--)
     {
      neuron = cProject[i];
      if(!neuron ||
         !neuron.CalcHiddenGradients(cProject[i + 1]))
         ReturnFalse;
     }
   if(!ScalarToVectorGrad(cSpikeScore.getOutput(), cSpikeScore.getGradient(),
                          cK.getOutput(), cK.getGradient(),
                          neuron.getGradient(), dimension_k))
      ReturnFalse;
//---
   if(!cScore.CalcHiddenGradients(cSpikeScore.AsObject()))
      ReturnFalse;
   if(!cQ.CalcHiddenGradients(cScore.AsObject()))
      ReturnFalse;
//---
   if(!Concat(cQ.getGradient(), cK.getGradient(), cSpikeQK.getGradient(), hd, hd, units))
      ReturnFalse;
   if(!cQK.CalcHiddenGradients(cSpikeQK.AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cQK.AsObject()))
      ReturnFalse;
   neuron = cProject[-1];
   if(!DeActivation(NeuronOCL.getOutput(), neuron.getPrevOutput(), Gradient, NeuronOCL.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), neuron.getPrevOutput(), NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cQK.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSpikeQK.UpdateInputWeights(cQK.AsObject()))
      ReturnFalse;
//---
   if(!cScore.UpdateInputWeights(cQ.AsObject()))
      ReturnFalse;
   if(!cSpikeScore.UpdateInputWeights(cScore.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* neuron = NULL;
   for(int i = 1; i < cProject.Total(); i++)
     {
      neuron = cProject[i];
      if(!neuron ||
         !neuron.UpdateInputWeights(cProject[i - 1]))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cQK.Save(file_handle))
      ReturnFalse;
   if(!cSpikeQK.Save(file_handle))
      ReturnFalse;
//---
   if(!cScore.Save(file_handle))
      ReturnFalse;
   if(!cSpikeScore.Save(file_handle))
      ReturnFalse;
//---
   if(!cProject.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cQK.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpikeQK.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cScore.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpikeScore.AsObject()))
      ReturnFalse;
   if(!cProject.Load(file_handle))
      ReturnFalse;
//---
   uint index = 2;
   if(!cQ.Init(0, index, OpenCL, cQK.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cQ.SetActivationFunction((ENUM_ACTIVATION)cSpikeQK.Activation());
   index++;
   if(!cK.Init(0, index, OpenCL, cQ.Neurons(), optimization, iBatch))
      ReturnFalse;
   cK.SetActivationFunction((ENUM_ACTIVATION)cSpikeQK.Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeQKAttention* Source = source;
   if(!cQK.WeightsUpdate(Source.cQK.AsObject(), tau))
      ReturnFalse;
   if(!cSpikeQK.WeightsUpdate(Source.cSpikeQK.AsObject(), tau))
      ReturnFalse;
//---
   if(!cScore.WeightsUpdate(Source.cScore.AsObject(), tau))
      ReturnFalse;
   if(!cSpikeScore.WeightsUpdate(Source.cSpikeScore.AsObject(), tau))
      ReturnFalse;
//---
   if(!cProject.WeightsUpdate(Source.cProject.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeQKAttention::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cQK.Clear())
      ReturnFalse;
   if(!cSpikeQK.Clear())
      ReturnFalse;
//---
   if(!cScore.Clear())
      ReturnFalse;
   if(!cSpikeScore.Clear())
      ReturnFalse;
//---
   if(!cProject.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeQKAttention::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cQK.SetOpenCL(OpenCL);
   cSpikeQK.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
   cSpikeScore.SetOpenCL(OpenCL);
   cProject.SetOpenCL(OpenCL);
   cQ.SetOpenCL(OpenCL);
   cK.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeQKAttention::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cQK.TrainMode(bTrain);
   cSpikeQK.TrainMode(bTrain);
   cScore.TrainMode(bTrain);
   cSpikeScore.TrainMode(bTrain);
   cProject.TrainMode(bTrain);
   cQ.TrainMode(bTrain);
   cK.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint units, uint window, uint heads, uint dimension_k,
                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMSRes::Init(numOutputs, myIndex, open_cl, units, window, 1, optimization_type, batch))
      ReturnFalse;
   if(!cAttention.Init(0, 0, OpenCL, units, window, heads, dimension_k, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cAttention.FeedForward(NeuronOCL))
      ReturnFalse;
   return CNeuronMSRes::feedForward(cAttention.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronMSRes::calcInputGradients(cAttention.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cAttention.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   return CNeuronMSRes::updateInputWeights(cAttention.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::Save(const int file_handle)
  {
   if(!CNeuronMSRes::Save(file_handle))
      ReturnFalse;
   if(!cAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::Load(const int file_handle)
  {
   if(!CNeuronMSRes::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAttention.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMSRes::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSDSA* Source = source;
   if(!cAttention.WeightsUpdate(Source.cAttention.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDSA::Clear(void)
  {
   if(!CNeuronMSRes::Clear())
      ReturnFalse;
   if(!cAttention.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDSA::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMSRes::SetOpenCL(obj);
   cAttention.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDSA::TrainMode(bool flag)
  {
   CNeuronMSRes::TrainMode(flag);
   cAttention.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint units, uint &windows[], uint &heads[],
                                 uint &layers[], uint dimension_k,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint block = layers.Size();
   if(block <= 0 ||
      block > heads.Size() ||
      (block + 1) > windows.Size())
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, 2 * windows[1], 2 * windows[1],
                                   windows[0], units, 1, optimization_type, batch))
      ReturnFalse;
//---
   cEncoder.Clear();
   cEncoder.SetOpenCL(OpenCL);
//---
   uint index = 0;
   CNeuronSDSA* sdsa = NULL;
   for(uint i = 0; i < layers[0]; i++)
     {
      sdsa = new CNeuronSDSA();
      if(!sdsa ||
         !sdsa.Init(0, index, OpenCL, units, windows[0], heads[0], dimension_k, optimization, iBatch) ||
         !cEncoder.Add(sdsa))
        {
         DeleteObj(sdsa)
         ReturnFalse;
        }
      index++;
     }
//---
   CNeuronSpikeConvBlock* spm = new CNeuronSpikeConvBlock();
   if(!spm ||
      !spm.Init(0, index, OpenCL, 2 * windows[0], 2 * windows[0], 2 * windows[1], (units + 1) / 2, 1, optimization, iBatch) ||
      !cEncoder.Add(spm))
     {
      DeleteObj(spm)
      ReturnFalse;
     }
   index++;
//---
   if(block == 1)
     {
      CNeuronMSRes* neck = new CNeuronMSRes();
      if(!neck ||
         !neck.Init(0, index, OpenCL, units, windows[1], 1, optimization, iBatch))
        {
         DeleteObj(neck)
         ReturnFalse;
        }
      cNeck = neck;
     }
   else
     {
      uint new_layers[], new_windows[], new_heads[];
      if(ArrayResize(new_layers, block - 1) < 0 ||
         ArrayResize(new_windows, block) < 0 ||
         ArrayResize(new_heads, block - 1) < 0)
         ReturnFalse;
      if(ArrayCopy(new_heads, heads, 0, 1, block - 1) < (int(block) - 1) ||
         ArrayCopy(new_layers, layers, 0, 1, block - 1) < (int(block) - 1) ||
         ArrayCopy(new_windows, windows, 0, 1, block) < int(block))
         ReturnFalse;
      CNeuronSDformerUBlock* neck = new CNeuronSDformerUBlock();
      if(!neck ||
         !neck.Init(0, index, OpenCL, units, new_windows, new_heads,
                    new_layers, dimension_k, optimization, iBatch))
        {
         DeleteObj(neck)
         ReturnFalse;
        }
      cNeck = neck;
     }
   index++;
//---
   if(!cConcat.Init(0, index, OpenCL, 2 * windows[1]*units, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *curr = NULL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cNeck ||
      !cNeck.FeedForward(prev))
      ReturnFalse;
   if(!Concat(prev.getOutput(), cNeck.getOutput(), cConcat.getOutput(), GetWindow() / 2, GetWindow() / 2, GetUnits()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !cNeck)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
   if(!DeConcat(cConcat.getPrevOutput(), cNeck.getGradient(), cConcat.getGradient(),
                GetWindow() / 2, GetWindow() / 2, GetUnits()))
      ReturnFalse;
   Deactivation(cNeck)
//---
   CNeuronBaseOCL* curr = cEncoder[-1];
   if(!curr ||
      !curr.CalcHiddenGradients(cNeck))
      ReturnFalse;
   if(!DeActivation(curr.getOutput(), cConcat.getPrevOutput(), cConcat.getPrevOutput(), curr.Activation()) ||
      !SumAndNormalize(curr.getGradient(), cConcat.getPrevOutput(), curr.getGradient(), GetWindow() / 2, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cEncoder[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(curr))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *prev = NeuronOCL;
   CNeuronBaseOCL *curr = NULL;
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cNeck ||
      !cNeck.UpdateInputWeights(prev))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cNeck ||
      !cNeck.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
//---
   DeleteObj(cNeck)
   CNeuronMSRes* msres = NULL;
   CNeuronSDformerUBlock* ublock = NULL;
   uint new_layers[] = {1}, new_windows[] = {1, 1}, new_heads[] = {1};
   switch(FileReadInteger(file_handle))
     {
      case defNeuronMSRes:
         msres = new CNeuronMSRes();
         if(!msres ||
            !msres.Init(0, cEncoder.Total(), OpenCL, 3, 2, 1, optimization, iBatch))
           {
            DeleteObj(msres)
            ReturnFalse;
           }
         cNeck = msres;
         break;
      case defNeuronSDformerUBlock:
         ublock = new CNeuronSDformerUBlock();
         if(!ublock ||
            !ublock.Init(0, cEncoder.Total(), OpenCL, 3, new_windows, new_heads,
                         new_layers, 3, optimization, iBatch))
           {
            DeleteObj(ublock)
            ReturnFalse;
           }
         cNeck = ublock;
         break;
      default:
         ReturnFalse;
         break;
     }
   if(!cNeck ||
      !cNeck.Load(file_handle))
      ReturnFalse;
//---
   if(!cConcat.Init(0, cEncoder.Total() + 1, OpenCL, GetWindow()*GetUnits(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSDformerUBlock* Source = source;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cNeck ||
      !cNeck.WeightsUpdate(Source.cNeck.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerUBlock::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cEncoder.ClearStates())
      ReturnFalse;
   if(!cNeck ||
      !cNeck.Clear())
      ReturnFalse;
//---
   if(!cConcat.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDformerUBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//---
   cEncoder.SetOpenCL(OpenCL);
   if(!!cNeck)
      cNeck.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDformerUBlock::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//---
   cEncoder.TrainMode(bTrain);
   if(!!cNeck)
      cNeck.TrainMode(bTrain);
   cConcat.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronSDformerUBlock::GetStack(void)
  {
   CBufferFloat* result = new CBufferFloat();
   if(!result)
      return NULL;
//---
   uint window = GetFilters();
   uint units = GetUnits();
//---
   if(!cNeck ||
      cNeck.Type() != Type())
     {
      if(!result.BufferInit(window * units, 0) ||
         !result.BufferCreate(OpenCL))
        {
         DeleteObj(result)
         return NULL;
        }
      if(!Concat(Output, Output, result, window, 0, units))
        {
         DeleteObj(result)
         return NULL;
        }
      return result;
     }
//---
   CNeuronSDformerUBlock* block = cNeck;
   CBufferFloat* neck = block.GetStack();
   uint neck_window = neck.Total() / units;
   if(!result.BufferInit((window + neck_window)*units, 0) ||
      !result.BufferCreate(OpenCL))
     {
      DeleteObj(result)
      DeleteObj(neck)
      return NULL;
     }
   if(!Concat(Output, neck, result, window, neck_window, units))
     {
      DeleteObj(neck)
      DeleteObj(result)
      return NULL;
     }
   DeleteObj(neck)
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint units, uint &windows[], uint &heads[],
                               uint &layers[], uint dimension_k,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(layers.Size() < 3)
      ReturnFalse;
   if(heads.Size() != (layers.Size() - 2))
      ReturnFalse;
   if(windows.Size() < (layers[0] + layers.Size() - 1))
      ReturnFalse;
//---
   uint block = layers.Size() - 2;
   uint new_layers[], new_windows[];
   if(ArrayResize(new_layers, block) < 0 ||
      ArrayResize(new_windows, block + 1) < 0)
      ReturnFalse;
   if(ArrayCopy(new_layers, layers, 0, 2, block) < int(block) ||
      ArrayCopy(new_windows, windows, 0, layers[0], block + 1) < int(block + 1))
      ReturnFalse;
//---
   if(!CNeuronSDformerUBlock::Init(numOutputs, myIndex, open_cl, units, new_windows, heads, new_layers, dimension_k, optimization_type, batch))
      ReturnFalse;
//---
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
//---
   CNeuronSpikeConvBlock* conv = NULL;
   uint index = 0;
   for(uint i = 0; i < layers[0]; i++)
     {
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, windows[i], windows[i], windows[i + 1], units, 1, optimization, iBatch) ||
         !cFlow.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      index++;
     }
//---
   CNeuronMSRes* msres = NULL;
   for(uint i = 0; i < layers[1]; i++)
     {
      msres = new CNeuronMSRes();
      if(!msres ||
         !msres.Init(0, index, OpenCL, units, windows[layers[0]], 1, optimization, iBatch) ||
         !cFlow.Add(msres))
        {
         DeleteObj(msres)
         ReturnFalse;
        }
      index++;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
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
//---
   return CNeuronSDformerUBlock::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSDformerUBlock::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cFlow[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(curr))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
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
//---
   return CNeuronSDformerUBlock::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::Save(const int file_handle)
  {
   if(!CNeuronSDformerUBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::Load(const int file_handle)
  {
   if(!CNeuronSDformerUBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::Clear(void)
  {
   if(!CNeuronSDformerUBlock::Clear())
      ReturnFalse;
//---
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSDformerFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSDformerUBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSDformerFlow* Source = source;
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDformerFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSDformerUBlock::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSDformerFlow::TrainMode(bool flag)
  {
   CNeuronSDformerUBlock::TrainMode(flag);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint stack_size, uint dimension,
                                 uint patch_dimension, uint variables,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronAddToStack::Init(numOutputs, myIndex, open_cl, stack_size, patch_dimension, variables, optimization_type, batch))
      ReturnFalse;
   if(!cBlock.Init(0, 0, OpenCL, dimension, dimension, patch_dimension, variables, 1, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cBlock.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronAddToStack::feedForward(cBlock.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronAddToStack::calcInputGradients(cBlock.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cBlock.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cBlock.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronAddToStack::updateInputWeights(cBlock.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronAddToStack::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikePatchStak* Source = source;
   if(!cBlock.WeightsUpdate(Source.cBlock.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::Save(const int file_handle)
  {
   if(!CNeuronAddToStack::Save(file_handle))
      ReturnFalse;
   if(!cBlock.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::Load(const int file_handle)
  {
   if(!CNeuronAddToStack::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cBlock.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikePatchStak::Clear(void)
  {
   if(!CNeuronAddToStack::Clear())
      ReturnFalse;
   if(!cBlock.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikePatchStak::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronAddToStack::SetOpenCL(obj);
   cBlock.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikePatchStak::TrainMode(bool flag)
  {
   CNeuronAddToStack::TrainMode(flag);
   cBlock.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                uint window_in, uint window_out, uint units_in, uint units_out,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units_out, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   iWindowOut = window_out;
   iUnitsOut = units_out;
//---
   int index = 0;
   CNeuronSpikeConvBlock *conv = NULL;
   CNeuronTransposeOCL *transp = NULL;
//--- Projection Time
   cProjectionX_Time.Clear();
   cProjectionX_Time.SetOpenCL(OpenCL);
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, window_in, optimization, iBatch) ||
      !cProjectionX_Time.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, iUnitsOut, window_in, 1, optimization, iBatch) ||
      !cProjectionX_Time.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, window_in, iUnitsOut, optimization, iBatch) ||
      !cProjectionX_Time.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_in, window_in, iWindowOut, iUnitsOut, 1, optimization, iBatch) ||
      !cProjectionX_Time.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//--- Projection Variables
   cProjectionX_Variable.Clear();
   cProjectionX_Variable.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_in, window_in, iUnitsOut, units_in, 1, optimization, iBatch) ||
      !cProjectionX_Variable.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, iUnitsOut, optimization, iBatch) ||
      !cProjectionX_Variable.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, iWindowOut, iUnitsOut, 1, optimization, iBatch) ||
      !cProjectionX_Variable.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//--- HiddenState
   index++;
   if(!cHiddenStates.Init(0, index, OpenCL, 2 * iUnitsOut * iWindowOut, optimization, iBatch))
      ReturnFalse;
//--- A*H
   index++;
   if(!cA.Init(0, index, OpenCL, iWindowOut, iWindowOut, 2 * iWindowOut, iUnitsOut, 2, optimization, iBatch))
      ReturnFalse;
//--- BC
   index++;
   if(!cBC_Time.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut + 1, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cBC_Variable.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut + 1, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
//--- B
   index++;
   if(!cB_Time.Init(0, index, OpenCL, iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cB_Time.SetActivationFunction(None);
   index++;
   if(!cB_Variable.Init(0, index, OpenCL, iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cB_Variable.SetActivationFunction(None);
//--- C
   index++;
   if(!cC_Time.Init(0, index, OpenCL, iUnitsOut * iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cC_Time.SetActivationFunction(None);
   index++;
   if(!cC_Variable.Init(0, index, OpenCL, iUnitsOut * iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cC_Variable.SetActivationFunction(None);
//--- Delta
   index++;
   if(!cDelta_Time.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cDelta_Variable.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *x_time = NULL;
   CNeuronBaseOCL *x_var = NULL;
//--- Projection Time
   int total = cProjectionX_Time.Total();
   for(int i = 0; i < total; i++)
     {
      x_time = cProjectionX_Time.At(i);
      if(!x_time ||
         !x_time.FeedForward(inp))
         ReturnFalse;
      inp = x_time;
     }
//--- Projection Variable
   inp = NeuronOCL;
   total = cProjectionX_Variable.Total();
   for(int i = 0; i < total; i++)
     {
      x_var = cProjectionX_Variable.At(i);
      if(!x_var ||
         !x_var.FeedForward(inp))
         ReturnFalse;
      inp = x_var;
     }
//---
   if(!cA.FeedForward(cHiddenStates.AsObject()))
      ReturnFalse;
   if(!cBC_Time.FeedForward(x_time) ||
      !cBC_Variable.FeedForward(x_var))
      ReturnFalse;
   if(!DeConcat(cB_Time.getOutput(), cC_Time.getOutput(), cBC_Time.getOutput(), 1, iUnitsOut, iUnitsOut) ||
      !DeConcat(cB_Variable.getOutput(), cC_Variable.getOutput(), cBC_Variable.getOutput(), 1, iUnitsOut, iUnitsOut))
      ReturnFalse;
//---
   if(!cDelta_Time.FeedForward(x_time) ||
      !cDelta_Variable.FeedForward(x_var))
      ReturnFalse;
//---
   if(!cHiddenStates.SwapOutputs())
      ReturnFalse;
//---
   return feedForwardSSM2D();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::feedForwardSSM2D(void)
  {
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!OpenCL || !x_time || !x_var)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnitsOut, iWindowOut};
   uint local_work_size[2] = {global_work_size[0], 1};
   int kernel = def_k_SSM2D_FeedForward;
//---
   ResetLastError();
   setBuffer(kernel, def_k_ssm2d_ah, cA.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_b_time, cB_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_b_var, cB_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_c_time, cC_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_c_var, cC_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_delta_time, cDelta_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_delta_var, cDelta_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_hidden, cHiddenStates.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_px_time, x_time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_px_var, x_var.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_y, getOutputIndex())
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
bool CNeuronSpike2DSSMOCL::calcInputGradientsSSM2D(void)
  {
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!OpenCL || !x_time || !x_var)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnitsOut, iWindowOut};
   uint local_work_size[2] = {1, global_work_size[1]};
   int kernel = def_k_SSM2D_CalcHiddenGradient;
//---
   ResetLastError();
   setBuffer(kernel, def_k_ssm2dhg_ah, cA.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_ah, cA.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_b_time, cB_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_b_time, cB_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_b_var, cB_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_b_var, cB_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_c_time, cC_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_c_time, cC_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_c_var, cC_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_c_var, cC_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_delta_time, cDelta_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_delta_time, cDelta_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_delta_var, cDelta_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_delta_var, cDelta_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_hidden, cHiddenStates.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_px_time, x_time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_px_time, x_time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_px_var, x_var.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_px_var, x_var.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_y, getGradientIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cA.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!calcInputGradientsSSM2D())
      ReturnFalse;
//---
   if(!Concat(cB_Time.getGradient(), cC_Time.getGradient(), cBC_Time.getGradient(), 1, iUnitsOut, iUnitsOut) ||
      !Concat(cB_Variable.getGradient(), cC_Variable.getGradient(), cBC_Variable.getGradient(), 1, iUnitsOut, iUnitsOut))
      ReturnFalse;
//--- Deactivation
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!x_time || !x_var)
      ReturnFalse;
   Deactivation(x_time)
   Deactivation(x_var)
   Deactivation(cBC_Time)
   Deactivation(cBC_Variable)
   Deactivation(cDelta_Time)
   Deactivation(cDelta_Variable)
   Deactivation(cA)
//--- Gradient to projections X
   CBufferFloat *grad_x_time = x_time.getGradient();
   CBufferFloat *grad_x_var = x_var.getGradient();
   if(!x_time.SetGradient(x_time.getPrevOutput(), false) ||
      !x_var.SetGradient(x_var.getPrevOutput(), false))
      ReturnFalse;
//--- BC -> X
   if(!x_time.CalcHiddenGradients(cBC_Time.AsObject()) ||
      !SumAndNormalize(grad_x_time, x_time.getGradient(), grad_x_time, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_var.CalcHiddenGradients(cBC_Variable.AsObject()) ||
      !SumAndNormalize(grad_x_var, x_var.getGradient(), grad_x_var, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Delta -> X
   if(!x_time.CalcHiddenGradients(cDelta_Time.AsObject()) ||
      !SumAndNormalize(grad_x_time, x_time.getGradient(), grad_x_time, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_var.CalcHiddenGradients(cDelta_Variable.AsObject()) ||
      !SumAndNormalize(grad_x_var, x_var.getGradient(), grad_x_var, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_time.SetGradient(grad_x_time, false) ||
      !x_var.SetGradient(grad_x_var, false))
      ReturnFalse;
//--- Projection Variable
   int total = cProjectionX_Variable.Total() - 2;
   for(int i = total; i >= 0; i--)
     {
      x_var = cProjectionX_Variable[i];
      if(!x_var ||
         !x_var.CalcHiddenGradients(cProjectionX_Variable[i + 1]))
         ReturnFalse;
     }
//--- Projection Time
   total = cProjectionX_Time.Total() - 2;
   for(int i = total; i >= 0; i--)
     {
      x_time = cProjectionX_Time[i];
      if(!x_time ||
         !x_time.CalcHiddenGradients(cProjectionX_Time[i + 1]))
         ReturnFalse;
     }
//--- Projections -> inputs
   if(!NeuronOCL.CalcHiddenGradients(x_var.AsObject()))
      ReturnFalse;
   grad_x_time = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(x_time.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(x_time.AsObject()) ||
      !SumAndNormalize(grad_x_time, NeuronOCL.getGradient(), grad_x_time, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(grad_x_time, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *x_time = NULL;
   CNeuronBaseOCL *x_var = NULL;
//--- Projection Time
   int total = cProjectionX_Time.Total();
   for(int i = 0; i < total; i++)
     {
      x_time = cProjectionX_Time.At(i);
      if(!x_time ||
         !x_time.UpdateInputWeights(inp))
         ReturnFalse;
      inp = x_time;
     }
//--- Projection Variable
   inp = NeuronOCL;
   total = cProjectionX_Variable.Total();
   for(int i = 0; i < total; i++)
     {
      x_var = cProjectionX_Variable.At(i);
      if(!x_var ||
         !x_var.UpdateInputWeights(inp))
         ReturnFalse;
      inp = x_var;
     }
//---
   cHiddenStates.SwapOutputs();
   if(!cA.UpdateInputWeights(cHiddenStates.AsObject()))
      ReturnFalse;
   cHiddenStates.SwapOutputs();
   if(!cBC_Time.UpdateInputWeights(x_time) ||
      !cBC_Variable.UpdateInputWeights(x_var))
      ReturnFalse;
//---
   if(!cDelta_Time.UpdateInputWeights(x_time) ||
      !cDelta_Variable.UpdateInputWeights(x_var))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpike2DSSMOCL* Source = source;
//--- Projections
   if(!cProjectionX_Time.WeightsUpdate(Source.cProjectionX_Time.AsObject(), tau) ||
      !cProjectionX_Variable.WeightsUpdate(Source.cProjectionX_Variable.AsObject(), tau))
      ReturnFalse;
//---
   if(!cA.WeightsUpdate(Source.cA.AsObject(), tau))
      ReturnFalse;
   if(!cBC_Time.WeightsUpdate(Source.cBC_Time.AsObject(), tau) ||
      !cBC_Variable.WeightsUpdate(Source.cBC_Variable.AsObject(), tau))
      ReturnFalse;
   if(!cDelta_Time.WeightsUpdate(Source.cDelta_Time.AsObject(), tau) ||
      !cDelta_Variable.WeightsUpdate(Source.cDelta_Variable.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cProjectionX_Time.Save(file_handle) ||
      !cProjectionX_Variable.Save(file_handle))
      ReturnFalse;
   if(!cA.Save(file_handle))
      ReturnFalse;
   if(!cBC_Time.Save(file_handle) ||
      !cBC_Variable.Save(file_handle))
      ReturnFalse;
   if(!cDelta_Time.Save(file_handle) ||
      !cDelta_Variable.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!cProjectionX_Time.Load(file_handle) ||
      !cProjectionX_Variable.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cBC_Time.AsObject()) ||
      !LoadInsideLayer(file_handle, cBC_Variable.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDelta_Time.AsObject()) ||
      !LoadInsideLayer(file_handle, cDelta_Variable.AsObject()))
      ReturnFalse;
//---
   iWindowOut = cA.GetWindow();
   iUnitsOut = cA.GetUnits();
//--- HiddenState
   int index = cProjectionX_Time.Total() + cProjectionX_Variable.Total();
   if(!cHiddenStates.Init(0, index, OpenCL, 2 * iUnitsOut * iWindowOut, optimization, iBatch))
      ReturnFalse;
//--- B
   index += 4;
   if(!cB_Time.Init(0, index, OpenCL, iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cB_Time.SetActivationFunction(None);
   index++;
   if(!cB_Variable.Init(0, index, OpenCL, iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cB_Variable.SetActivationFunction(None);
//--- C
   index++;
   if(!cC_Time.Init(0, index, OpenCL, iUnitsOut * iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cC_Time.SetActivationFunction(None);
   index++;
   if(!cC_Variable.Init(0, index, OpenCL, iUnitsOut * iUnitsOut, optimization, iBatch))
      ReturnFalse;
   cC_Variable.SetActivationFunction(None);
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpike2DSSMOCL::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cProjectionX_Time.ClearStates() ||
      !cProjectionX_Variable.ClearStates())
      ReturnFalse;
   if(!cA.Clear())
      ReturnFalse;
   if(!cBC_Time.Clear() ||
      !cBC_Variable.Clear())
      ReturnFalse;
   if(!cDelta_Time.Clear() ||
      !cDelta_Variable.Clear())
      ReturnFalse;
//--- HiddenState
   if(!cHiddenStates.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpike2DSSMOCL::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cProjectionX_Time.SetOpenCL(OpenCL);
   cProjectionX_Variable.SetOpenCL(OpenCL);
   cA.SetOpenCL(OpenCL);
   cBC_Time.SetOpenCL(OpenCL);
   cBC_Variable.SetOpenCL(OpenCL);
   cDelta_Time.SetOpenCL(OpenCL);
   cDelta_Variable.SetOpenCL(OpenCL);
   cHiddenStates.SetOpenCL(OpenCL);
   cB_Time.SetOpenCL(OpenCL);
   cB_Variable.SetOpenCL(OpenCL);
   cC_Time.SetOpenCL(OpenCL);
   cC_Variable.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpike2DSSMOCL::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
//---
   cProjectionX_Time.TrainMode(bTrain);
   cProjectionX_Variable.TrainMode(bTrain);
   cA.TrainMode(bTrain);
   cBC_Time.TrainMode(bTrain);
   cBC_Variable.TrainMode(bTrain);
   cDelta_Time.TrainMode(bTrain);
   cDelta_Variable.TrainMode(bTrain);
   cHiddenStates.TrainMode(bTrain);
   cB_Time.TrainMode(bTrain);
   cB_Variable.TrainMode(bTrain);
   cC_Time.TrainMode(bTrain);
   cC_Variable.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint window, uint inside_dimension, uint units_count,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, inside_dimension,
                                   inside_dimension, window, units_count, 1, optimization_type, batch))
      ReturnFalse;
   if(!cXZProject.Init(0, 0, OpenCL, window, window, 2 * inside_dimension, units_count,
                       1, optimization, iBatch))
      ReturnFalse;
   if(!cX.Init(0, 1, OpenCL, inside_dimension * units_count, optimization, iBatch))
      ReturnFalse;
   cX.SetActivationFunction(None);
   if(!cZ.Init(0, 2, OpenCL, inside_dimension * units_count, optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction(None);
   if(!cSSM.Init(0, 3, OpenCL, inside_dimension, inside_dimension, units_count, units_count,
                 optimization, iBatch))
      ReturnFalse;
   if(!cZSSM.Init(0, 4, OpenCL, inside_dimension * units_count, optimization, iBatch))
      ReturnFalse;
   cZSSM.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cXZProject.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!DeConcat(cX.getOutput(), cZ.getOutput(), cXZProject.getOutput(), cXZProject.GetFilters() / 2,
                cXZProject.GetFilters() / 2, cXZProject.GetUnits()))
      ReturnFalse;
   if(!cSSM.FeedForward(cX.AsObject()))
      ReturnFalse;
   if(!ElementMult(cSSM.getOutput(), cZ.getOutput(), cZSSM.getOutput()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cZSSM.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::calcInputGradients(cZSSM.AsObject()))
      ReturnFalse;
   if(!ElementMultGrad(cSSM.getOutput(), cSSM.getGradient(),
                       cZ.getOutput(), cZ.getGradient(),
                       cZSSM.getGradient(), cSSM.Activation(), cZ.Activation()))
      ReturnFalse;
   if(!cX.CalcHiddenGradients(cSSM.AsObject()))
      ReturnFalse;
   if(!Concat(cX.getGradient(), cZ.getGradient(), cXZProject.getGradient(), cXZProject.GetFilters() / 2,
              cXZProject.GetFilters() / 2, cXZProject.GetUnits()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cXZProject.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cXZProject.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cSSM.UpdateInputWeights(cX.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cZSSM.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeMamba2D* Source = source;
   if(!cXZProject.WeightsUpdate(Source.cXZProject.AsObject(), tau))
      ReturnFalse;
   if(!cSSM.WeightsUpdate(Source.cSSM.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cXZProject.Save(file_handle))
      ReturnFalse;
   if(!cSSM.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cXZProject.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSSM.AsObject()))
      ReturnFalse;
//---
   if(!cX.Init(0, 1, OpenCL, cSSM.Neurons(), optimization, iBatch))
      ReturnFalse;
   cX.SetActivationFunction(None);
   if(!cZ.Init(0, 2, OpenCL, cSSM.Neurons(), optimization, iBatch))
      ReturnFalse;
   cZ.SetActivationFunction(None);
   if(!cZSSM.Init(0, 4, OpenCL, cSSM.Neurons(), optimization, iBatch))
      ReturnFalse;
   cZSSM.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMamba2D::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cXZProject.Clear())
      ReturnFalse;
   if(!cSSM.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMamba2D::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cXZProject.SetOpenCL(OpenCL);
   cSSM.SetOpenCL(OpenCL);
   cX.SetOpenCL(OpenCL);
   cZ.SetOpenCL(OpenCL);
   cZSSM.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint stack_size, uint dimension, uint patch_dimension,
                             uint ssm_dimension, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, ssm_dimension * (stack_size - 2),
                                   ssm_dimension * (stack_size - 2), dimension, variables, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cEmbeddings.Init(0, index, OpenCL, stack_size, dimension, patch_dimension, variables, optimization, iBatch))
      ReturnFalse;
//---
   cProjection.Clear();
   cProjection.SetOpenCL(OpenCL);
   CNeuronSpikeConvBlock* conv = NULL;
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * patch_dimension, patch_dimension, patch_dimension, stack_size - 1, variables, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * patch_dimension, patch_dimension, ssm_dimension, stack_size - 2, variables, optimization, iBatch) ||
      !cProjection.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
//---
   cSSM.Clear();
   cSSM.SetOpenCL(OpenCL);
   CNeuronSpikeMamba2D* ssm = NULL;
   for(int i = 0; i < 2; i++)
     {
      index++;
      ssm = new CNeuronSpikeMamba2D();
      if(!ssm ||
         !ssm.Init(0, index, OpenCL, ssm_dimension, ssm_dimension, (stack_size - 2)*variables, optimization, iBatch) ||
         !cSSM.Add(ssm))
        {
         DeleteObj(ssm)
         ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//---
   if(!cEmbeddings.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cEmbeddings.AsObject();
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cProjection.Total(); i++)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   uint units = GetUnits() * GetVariables();
   for(int i = 0; i < cSSM.Total(); i++)
     {
      curr = cSSM[i];
      if(!curr ||
         !curr.FeedForward(prev) ||
         !SumAndNormalize(curr.getOutput(), prev.getOutput(), curr.getOutput(),
                          curr.Neurons() / units, true, 0, 0, 0, 1))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cSSM[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   uint units = GetUnits() * GetVariables();
   for(int i = cSSM.Total() - 2; i >= 0; i--)
     {
      curr = cSSM[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cSSM[i + 1]) ||
         !SumAndNormalize(curr.getGradient(), cSSM[i + 1].getGradient(), curr.getGradient(),
                          curr.Neurons() / units, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   CNeuronBaseOCL* next = curr;
   for(int i = cProjection.Total() - 2; i >= 0; i--)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!cEmbeddings.CalcHiddenGradients(next))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cEmbeddings.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//---
   if(!cEmbeddings.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cEmbeddings.AsObject();
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cProjection.Total(); i++)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   uint units = GetUnits() * GetVariables();
   for(int i = 0; i < cSSM.Total(); i++)
     {
      curr = cSSM[i];
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
bool CNeuronSpikeSTSSM::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cEmbeddings.Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cSSM.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cEmbeddings.AsObject()))
      ReturnFalse;
   if(!cProjection.Load(file_handle))
      ReturnFalse;
   if(!cSSM.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSTSSM* Source = source;
   if(!cEmbeddings.WeightsUpdate(Source.cEmbeddings.AsObject(), tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   if(!cSSM.WeightsUpdate(Source.cSSM.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSTSSM::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cEmbeddings.Clear())
      ReturnFalse;
   if(!cProjection.ClearStates())
      ReturnFalse;
   if(!cSSM.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSTSSM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//---
   cEmbeddings.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
   cSSM.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSTSSM::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//---
   cEmbeddings.TrainMode(bTrain);
   cProjection.TrainMode(bTrain);
   cSSM.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                              uint &stack_size[], uint &dimension[], uint patch_dimension,
                              uint ssm_dimension, uint &variables[],
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint layers = stack_size.Size();
   if(layers < 1 ||
      dimension.Size() != (layers + 1) ||
      variables.Size() != (layers + 1))
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, dimension[layers - 1],
                                   dimension[layers - 1], dimension[layers],
                                   variables[layers], 1, optimization_type, batch))
      ReturnFalse;
//---
   cFlow.Clear();;
   cFlow.SetOpenCL(OpenCL);
   CNeuronTransposeOCL*    transp = NULL;
   CNeuronSpikeConvBlock*  conv = NULL;
   CNeuronSpikeSTSSM*      ssm = NULL;
   uint index = 0;
   for(uint i = 0; i < layers; i++)
     {
      ssm = new CNeuronSpikeSTSSM();
      if(!ssm ||
         !ssm.Init(0, index, OpenCL, stack_size[i], dimension[i], patch_dimension,
                   ssm_dimension, variables[i], optimization, iBatch) ||
         !cFlow.Add(ssm))
        {
         DeleteObj(ssm)
         ReturnFalse;
        }
      index++;
      if(variables[i] != variables[i + 1])
        {
         transp = new CNeuronTransposeOCL();
         if(!transp ||
            !transp.Init(0, index, OpenCL, variables[i], dimension[i], optimization, iBatch) ||
            !cFlow.Add(transp))
           {
            DeleteObj(transp)
            ReturnFalse;
           }
         index++;
         conv = new CNeuronSpikeConvBlock();
         if(!conv ||
            !conv.Init(0, index, OpenCL, variables[i], variables[i], variables[i + 1],
                       dimension[i], 1, optimization, iBatch) ||
            !cFlow.Add(conv))
           {
            DeleteObj(conv)
            ReturnFalse;
           }
         index++;
         transp = new CNeuronTransposeOCL();
         if(!transp ||
            !transp.Init(0, index, OpenCL, dimension[i], variables[i + 1], optimization, iBatch) ||
            !cFlow.Add(transp))
           {
            DeleteObj(transp)
            ReturnFalse;
           }
         index++;
        }
      if(i == (layers - 1))
         break;
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, dimension[i], dimension[i], dimension[i + 1],
                    variables[i + 1], 1, optimization, iBatch) ||
         !cFlow.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      index++;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      if(curr.Type() == defNeuronSpikeSTSSM)
         if(!SumAndNormalize(prev.getOutput(), curr.getOutput(), curr.getOutput(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
      prev = curr;
     }
//---
   return CNeuronSpikeConvBlock::feedForward(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cFlow[-1];
   CNeuronBaseOCL* curr = NULL;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      if(next.Type() == defNeuronSpikeSTSSM)
        {
         if(curr.Activation() != None)
           {
            if(!DeActivation(curr.getOutput(), next.getPrevOutput(), next.getGradient(), curr.Activation()) ||
               !SumAndNormalize(next.getPrevOutput(), curr.getGradient(), curr.getGradient(), 1, false, 0, 0, 0, 1))
               ReturnFalse;
           }
         else
            if(!SumAndNormalize(next.getGradient(), curr.getGradient(), curr.getGradient(), 1, false, 0, 0, 0, 1))
               ReturnFalse;
        }
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
   if(next.Type() == defNeuronSpikeSTSSM)
     {
      if(NeuronOCL.Activation() != None)
        {
         if(!DeActivation(NeuronOCL.getOutput(), next.getPrevOutput(), next.getGradient(), NeuronOCL.Activation()) ||
            !SumAndNormalize(next.getPrevOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
        }
      else
         if(!SumAndNormalize(next.getGradient(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
            ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
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
//---
   return CNeuronSpikeConvBlock::updateInputWeights(prev);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronESTMEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   if(!cFlow.WeightsUpdate(((CNeuronESTMEncoder*)source).cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronESTMEncoder::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronESTMEncoder::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint stack_size, uint dimension,
                             uint patch_dimension, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikePatchStak::Init(numOutputs, myIndex, open_cl, stack_size,
                                   dimension, patch_dimension, variables, optimization_type, batch))
      ReturnFalse;
   if(!cState.Init(0, 0, OpenCL, dimension * variables, optimization, iBatch))
      ReturnFalse;
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::CalcFlow(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
   if(NeuronOCL.Neurons() > cState.Neurons())
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = { NeuronOCL.Neurons() };
   uint kernel = def_k_CalcFlow;
   setBuffer(kernel, def_k_cf_value, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_cf_prev_value, cState.getPrevOutIndex())
   setBuffer(kernel, def_k_cf_flow, cState.getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cState.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!CalcFlow(NeuronOCL))
      ReturnFalse;
//---
   return CNeuronSpikePatchStak::feedForward(cState.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikePatchStak::calcInputGradients(cState.AsObject()))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                    cState.getGradient(), NeuronOCL.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   return CNeuronSpikePatchStak::updateInputWeights(cState.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikePatchStak::Load(file_handle))
      ReturnFalse;
   if(!cState.Init(0, 0, OpenCL, cBlock.GetWindow() * cBlock.GetUnits() * cBlock.GetVariables(), optimization, iBatch))
      ReturnFalse;
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateFlow::Clear(void)
  {
   if(!CNeuronSpikePatchStak::Clear())
      ReturnFalse;
   if(!cState.Clear() ||
      !cState.getPrevOutput() ||
      !cState.getPrevOutput().Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCreateFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikePatchStak::SetOpenCL(obj);
   cState.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                      uint dimension, uint units, uint variables, uint window_out,
                                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
     {
      uint windows[] = {dimension, 1};
      if(!CNeuronTMAMFE::Init(numOutputs, myIndex, open_cl, units * variables, windows, window_out, optimization_type, batch))
         ReturnFalse;
     }
//---
   uint index = 0;
   if(!cInpNorm.Init(0, index, OpenCL, variables * dimension * units, iBatch, optimization))
      ReturnFalse;
   cInpNorm.SetActivationFunction(None);
   index++;
   if(!cRefFrame.Init(0, index, OpenCL, variables * dimension, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
   index++;
   if(!cRefFrameT.Init(0, index, OpenCL, variables, dimension, optimization, iBatch))
      ReturnFalse;
   cRefFrameT.SetActivationFunction(None);
     {
      uint windows[];
      if(ArrayResize(windows, units) < (int)units)
         ReturnFalse;
      ArrayFill(windows, 0, units, dimension);
      index++;
      if(!cWarp.Init(0, index, OpenCL, windows, dimension, variables, 1, optimization, iBatch))
         ReturnFalse;
      cWarp.SetActivationFunction(None);
     }
   index++;
   if(!cWarpNorm.Init(0, index, OpenCL, cWarp.Neurons(), iBatch, optimization))
      ReturnFalse;
   cWarpNorm.SetActivationFunction(None);
   index++;
   if(!cCorrelation.Init(0, index, OpenCL, units * variables, optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
   index++;
   if(!cConcat.Init(0, index, OpenCL, (dimension + 1)*units * variables, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cInpNorm.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint dimension = cRefFrameT.GetWindow();
   uint variables = cRefFrameT.GetCount();
   uint units = cCorrelation.Neurons() / variables;
   uint ref_position = (units + 1) / 2;
   if(!DeConcat(cConcat.getOutput(), cRefFrame.getOutput(), cConcat.getOutput(), cInpNorm.getOutput(),
                (ref_position - 1)*dimension, dimension, (units - ref_position)*dimension, variables))
      ReturnFalse;
   if(!cRefFrameT.FeedForward(cRefFrame.AsObject()))
      ReturnFalse;
   if(!cWarp.FeedForward(cInpNorm.AsObject()))
      ReturnFalse;
   if(!cWarpNorm.FeedForward(cWarp.AsObject()))
      ReturnFalse;
   if(!MatMul(cWarpNorm.getOutput(), cRefFrameT.getOutput(), cCorrelation.getOutput(), units, dimension, 1, variables, true))
      ReturnFalse;
   if(!Concat(cInpNorm.getOutput(), cCorrelation.getOutput(), cConcat.getOutput(), dimension, 1, units * variables))
      ReturnFalse;
   if(!CNeuronTMAMFE::feedForward(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint dimension = cRefFrameT.GetWindow();
   uint variables = cRefFrameT.GetCount();
   uint units = cCorrelation.Neurons() / variables;
   uint ref_position = (units + 1) / 2;
//---
   if(!CNeuronTMAMFE::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
   if(!DeConcat(cInpNorm.getPrevOutput(), cCorrelation.getGradient(), cConcat.getGradient(), dimension, 1, units * variables))
      ReturnFalse;
   if(cInpNorm.Activation() != None)
      if(!DeActivation(cInpNorm.getOutput(), cInpNorm.getPrevOutput(), cInpNorm.getPrevOutput(), cInpNorm.Activation()))
         ReturnFalse;
   Deactivation(cCorrelation)
//---
   if(!MatMulGrad(cWarpNorm.getOutput(), cWarpNorm.getGradient(),
                  cRefFrameT.getOutput(), cRefFrameT.getGradient(),
                  cCorrelation.getOutput(), units, dimension, 1, variables, true))
      ReturnFalse;
   Deactivation(cWarpNorm)
   Deactivation(cRefFrameT)
//---
   if(!cWarp.CalcHiddenGradients(cWarpNorm.AsObject()))
      ReturnFalse;
//---
   if(!cInpNorm.CalcHiddenGradients(cWarp.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cInpNorm.getPrevOutput(), cInpNorm.getGradient(),
                       cInpNorm.getGradient(), dimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cInpNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cInpNorm.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   if(!cWarp.UpdateInputWeights(cInpNorm.AsObject()))
      ReturnFalse;
   if(!cWarpNorm.UpdateInputWeights(cWarp.AsObject()))
      ReturnFalse;
   if(!CNeuronTMAMFE::updateInputWeights(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronTMAMFE::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronBiDirectCorrelation* Source = source;
   if(!cInpNorm.WeightsUpdate(Source.cInpNorm.AsObject(), tau))
      ReturnFalse;
   if(!cWarp.WeightsUpdate(Source.cWarp.AsObject(), tau))
      ReturnFalse;
   if(!cWarpNorm.WeightsUpdate(Source.cWarpNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::Save(const int file_handle)
  {
   if(!CNeuronTMAMFE::Save(file_handle))
      ReturnFalse;
//---
   if(!cInpNorm.Save(file_handle))
      ReturnFalse;
   if(!cRefFrameT.Save(file_handle))
      ReturnFalse;
   if(!cWarp.Save(file_handle))
      ReturnFalse;
   if(!cWarpNorm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBiDirectCorrelation::Load(const int file_handle)
  {
   if(!CNeuronTMAMFE::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cInpNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cRefFrameT.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWarp.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWarpNorm.AsObject()))
      ReturnFalse;
//---
   uint dimension = cRefFrameT.GetWindow();
   uint variables = cRefFrameT.GetCount();
   uint units = cCorrelation.Neurons() / variables;
//---
   uint index = 1;
   if(!cRefFrame.Init(0, index, OpenCL, variables * dimension, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
   index += 4;
   if(!cCorrelation.Init(0, index, OpenCL, units * variables, optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
   index++;
   if(!cConcat.Init(0, index, OpenCL, (dimension + 1)*units * variables, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBiDirectCorrelation::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronTMAMFE::SetOpenCL(obj);
//---
   cInpNorm.SetOpenCL(OpenCL);
   cRefFrameT.SetOpenCL(OpenCL);
   cWarp.SetOpenCL(OpenCL);
   cWarpNorm.SetOpenCL(OpenCL);
   cRefFrame.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint dimension, uint units, uint variables,
                        uint window_key, uint heads,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, dimension * units * variables,
                            optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   cWarp.Clear();
   cWarp.SetOpenCL(OpenCL);
   CNeuronMultiWindowsConvOCL* mwconv = NULL;
   CNeuronBatchNormOCL*        norm = NULL;
   CNeuronBaseOCL*             neuron = NULL;
   uint index = 0;
     {
      uint windows[];
      if(ArrayResize(windows, units) < (int)units)
         ReturnFalse;
      ArrayFill(windows, 0, units, dimension);
      mwconv = new CNeuronMultiWindowsConvOCL();
      if(!mwconv ||
         !mwconv.Init(0, index, OpenCL, windows, dimension, variables, 1, optimization, iBatch) ||
         !cWarp.Add(mwconv))
        {
         DeleteObj(mwconv)
         ReturnFalse;
        }
      mwconv.SetActivationFunction(None);
     }
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, mwconv.Neurons(), iBatch, optimization) ||
      !cWarp.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   norm.SetActivationFunction(None);
   index++;
   if(!cRefFrame.Init(0, index, OpenCL, dimension * variables, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
   index++;
   if(!cCrossAttention.Init(0, index, OpenCL, dimension, window_key, units * variables,
                            heads, dimension, units * variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cSpatialAttention.Init(0, index, OpenCL, dimension, dimension, dimension, units,
                              variables, optimization, iBatch))
      ReturnFalse;
   cSpatialAttention.SetActivationFunction(SIGMOID);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cWarp.Total(); i++)
     {
      curr = cWarp[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cSpatialAttention.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint dimension = cSpatialAttention.GetWindow();
   uint units = cSpatialAttention.GetUnits();
   uint variables = cSpatialAttention.GetVariables();
   if(!DeConcat(cRefFrame.getOutput(), prev.getPrevOutput(), NeuronOCL.getOutput(),
                dimension, dimension * (units - 1), variables))
      ReturnFalse;
   if(!SumVecMatrix(cRefFrame.getOutput(), prev.getOutput(), prev.getOutput(), dimension, variables, 0, 0, 0, 1))
      ReturnFalse;
   if(!cCrossAttention.FeedForward(NeuronOCL, prev.getOutput()))
      ReturnFalse;
   if(!ElementMult(cSpatialAttention.getOutput(), cCrossAttention.getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!ElementMultGrad(cSpatialAttention.getOutput(), cSpatialAttention.getGradient(),
                       cCrossAttention.getOutput(), cCrossAttention.getGradient(),
                       Gradient, cSpatialAttention.Activation(), cCrossAttention.Activation()))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cWarp[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next ||
      !NeuronOCL.CalcHiddenGradients(cCrossAttention.AsObject(), next.getOutput(),
                                     next.getGradient(), (ENUM_ACTIVATION)next.Activation()))
      ReturnFalse;
//---
   for(int i = cWarp.Total() - 2; i >= 0; i--)
     {
      curr = cWarp[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   uint dimension = cSpatialAttention.GetWindow();
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cSpatialAttention.getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(next) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSpatialAttention.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cWarp.Total(); i++)
     {
      curr = cWarp[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cSpatialAttention.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cCrossAttention.UpdateInputWeights(NeuronOCL, prev.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSATMA* Source = source;
   if(!cWarp.WeightsUpdate(Source.cWarp.AsObject(), tau))
      ReturnFalse;
   if(!cSpatialAttention.WeightsUpdate(Source.cSpatialAttention.AsObject(), tau))
      ReturnFalse;
   if(!cCrossAttention.WeightsUpdate(Source.cCrossAttention.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cWarp.Save(file_handle))
      ReturnFalse;
   if(!cSpatialAttention.Save(file_handle))
      ReturnFalse;
   if(!cCrossAttention.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSATMA::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cWarp.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSpatialAttention.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossAttention.AsObject()))
      ReturnFalse;
//---
   uint dimension = cSpatialAttention.GetWindow();
   uint variables = cSpatialAttention.GetVariables();
   uint index = cWarp.Total();
   if(!cRefFrame.Init(0, index, OpenCL, dimension * variables, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSATMA::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cWarp.SetOpenCL(OpenCL);
   cSpatialAttention.SetOpenCL(OpenCL);
   cCrossAttention.SetOpenCL(OpenCL);
   cRefFrame.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                      uint dimension, uint embed_size,
                      uint stack_size, uint variables,
                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, dimension, dimension,
                                   dimension, stack_size * variables, 1, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   CNeuronBatchNormOCL*                norm = NULL;
   CNeuronCreateFlow*                  stack = NULL;
   CNeuronSpikeConvBlock*              conv = NULL;
   CNeuronMultiWindowsConvWPadOCL*     mwconv = NULL;
   CNeuronMSRes*                       resblock = NULL;
   CNeuronBiDirectCorrelation*         correlation = NULL;
   CNeuronSATMA*                       satma = NULL;
//---
   cPrepare.Clear();
   cPrepare.SetOpenCL(OpenCL);
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, dimension * variables, iBatch, optimization) ||
      !cPrepare.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   index++;
   stack = new CNeuronCreateFlow();
   if(!stack ||
      !stack.Init(0, index, OpenCL, stack_size, dimension, embed_size, variables, optimization, iBatch) ||
      !cPrepare.Add(stack))
     {
      DeleteObj(stack)
      ReturnFalse;
     }
//---
   cEncoder.Clear();
   cContext.Clear();
   cEncoder.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
//---
   uint windows[] = {embed_size, 3 * embed_size, 5 * embed_size, 7 * embed_size};
   mwconv = new CNeuronMultiWindowsConvWPadOCL();
   index++;
   if(!mwconv ||
      !mwconv.Init(0, index, OpenCL, windows, embed_size, embed_size, stack_size, variables, optimization, iBatch) ||
      !cEncoder.Add(mwconv))
     {
      DeleteObj(mwconv)
      ReturnFalse;
     }
   mwconv.SetActivationFunction(None);
   mwconv = new CNeuronMultiWindowsConvWPadOCL();
   index++;
   if(!mwconv ||
      !mwconv.Init(0, index, OpenCL, windows, embed_size, embed_size, stack_size, variables, optimization, iBatch) ||
      !cContext.Add(mwconv))
     {
      DeleteObj(mwconv)
      ReturnFalse;
     }
   mwconv.SetActivationFunction(None);
   uint mult = windows.Size();
   while(mult > 1)
     {
      index++;
      resblock = new CNeuronMSRes();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, mult * stack_size, embed_size, variables, optimization, iBatch) ||
         !cEncoder.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      index++;
      resblock = new CNeuronMSRes();
      if(!resblock ||
         !resblock.Init(0, index, OpenCL, mult * stack_size, embed_size, variables, optimization, iBatch) ||
         !cContext.Add(resblock))
        {
         DeleteObj(resblock)
         ReturnFalse;
        }
      mult = (mult + 1) / 2;
      index++;
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, 2 * embed_size, 2 * embed_size, embed_size, mult * stack_size, variables, optimization, iBatch) ||
         !cEncoder.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      index++;
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, 2 * embed_size, 2 * embed_size, embed_size, mult * stack_size, variables, optimization, iBatch) ||
         !cContext.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
     }
//---
   index++;
   correlation = new CNeuronBiDirectCorrelation();
   if(!correlation ||
      !correlation.Init(0, index, OpenCL, embed_size, stack_size, variables, embed_size, optimization, iBatch) ||
      !cEncoder.Add(correlation))
     {
      DeleteObj(correlation)
      ReturnFalse;
     }
   index++;
   satma = new CNeuronSATMA();
   if(!satma ||
      !satma.Init(0, index, OpenCL, embed_size, stack_size, variables, (embed_size + 3) / 4, 4, optimization, iBatch) ||
      !cEncoder.Add(satma))
     {
      DeleteObj(satma)
      ReturnFalse;
     }
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, 2 * embed_size * stack_size * variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cGRU.Init(0, index, OpenCL, stack_size * variables, 2 * embed_size, dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cTranspose.Init(0, index, OpenCL, variables, stack_size, dimension, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
//---
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      current = cPrepare[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = cPrepare[-1];
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.FeedForward(prev))
         ReturnFalse;
      prev = current;
     }
//---
   uint units = cGRU.GetUnits();
   if(!Concat(cEncoder[-1].getOutput(), cContext[-1].getOutput(), cConcatenated.getOutput(),
              cEncoder[-1].Neurons() / units, cContext[-1].Neurons() / units, units))
      ReturnFalse;
   if(!cGRU.FeedForward(cConcatenated.AsObject()))
      ReturnFalse;
//---
   if(!cTranspose.FeedForward(cGRU.AsObject()))
      ReturnFalse;
   if(!SumVecMatrix(cPrepare[0].getOutput(), cTranspose.getOutput(), cTranspose.getOutput(),
                    cPrepare[0].Neurons(), 1, 0, 0, 0, 1))
      ReturnFalse;
//---
   return CNeuronSpikeConvBlock::feedForward(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cTranspose.AsObject()))
      ReturnFalse;
//---
   if(!cGRU.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   if(!cConcatenated.CalcHiddenGradients(cGRU.AsObject()))
      ReturnFalse;
   uint units = cGRU.GetUnits();
   if(!cEncoder[-1] || !cContext[-1] ||
      !DeConcat(cEncoder[-1].getGradient(), cContext[-1].getGradient(), cConcatenated.getGradient(),
                cEncoder[-1].Neurons() / units, cContext[-1].Neurons() / units, units))
      ReturnFalse;
   Deactivation(cEncoder[-1])
   Deactivation(cContext[-1])
//---
   CNeuronBaseOCL* current = NULL;
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      current = cEncoder[i];
      if(!current ||
         !current.CalcHiddenGradients(cEncoder[i + 1]))
         ReturnFalse;
     }
//---
   for(int i = cContext.Total() - 2; i >= 0; i--)
     {
      current = cContext[i];
      if(!current ||
         !current.CalcHiddenGradients(cContext[i + 1]))
         ReturnFalse;
     }
//---
   current = cPrepare[-1];
   if(!current ||
      !current.CalcHiddenGradients(cEncoder[0]))
      ReturnFalse;
   CBufferFloat* temp = current.getGradient();
   if(!current.SetGradient(current.getPrevOutput(), false))
      ReturnFalse;
   if(!current.CalcHiddenGradients(cContext[0]) ||
      !SumAndNormalize(temp, current.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !current.SetGradient(temp, false))
      ReturnFalse;
//---
   for(int i = cPrepare.Total() - 2; i >= 0; i--)
     {
      current = cPrepare[i];
      if(!current ||
         !current.CalcHiddenGradients(cPrepare[i + 1]))
         ReturnFalse;
     }
//---
   if(!SumVecMatrixGrad(current.getPrevOutput(), cTranspose.getPrevOutput(), cTranspose.getGradient(),
                        current.Neurons(), 1, 0, 0, 0, 1))
      ReturnFalse;
   if(current.Activation() != None)
      if(!DeActivation(current.getOutput(), current.getPrevOutput(), current.getPrevOutput(), current.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(current.getPrevOutput(), current.getGradient(), current.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(current))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* current = NULL;
   CNeuronBaseOCL* prev = NeuronOCL;
//---
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      current = cPrepare[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      current = cEncoder[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   prev = cPrepare[-1];
   for(int i = 0; i < cContext.Total(); i++)
     {
      current = cContext[i];
      if(!current ||
         !current.UpdateInputWeights(prev))
         ReturnFalse;
      prev = current;
     }
//---
   return CNeuronSpikeConvBlock::updateInputWeights(cTranspose.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronBAT* Source = source;
   if(!cPrepare.WeightsUpdate(Source.cPrepare.AsObject(), tau))
      ReturnFalse;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cContext.WeightsUpdate(Source.cContext.AsObject(), tau))
      ReturnFalse;
   if(!cGRU.WeightsUpdate(Source.cGRU.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cGRU.Save(file_handle))
      ReturnFalse;
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Load(file_handle))
      ReturnFalse;
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
   if(!cContext.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGRU.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
//---
   if(!cConcatenated.Init(0, 0, OpenCL, cContext[-1].Neurons() + cEncoder[-1].Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBAT::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cPrepare.SetOpenCL(OpenCL);
   cEncoder.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cConcatenated.SetOpenCL(OpenCL);
   cGRU.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBAT::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cPrepare.TrainMode(bTrain);
   cEncoder.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cGRU.TrainMode(bTrain);
   cTranspose.TrainMode(bTrain);
   cConcatenated.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBAT::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cPrepare.ClearStates())
      ReturnFalse;
   if(!cEncoder.ClearStates())
      ReturnFalse;
   if(!cContext.ClearStates())
      ReturnFalse;
   if(!cGRU.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronBAT::getWeights(void)
  {
   CBufferFloat* result = NULL;
   for(int layerNum = 0; (layerNum < cPrepare.Total() && !result && !IsStopped()); layerNum++)
     {
      CNeuronBaseOCL *temp = cPrepare[layerNum];
      CNeuronConvOCL *conv = NULL;
      CNeuronConcatenate *conc = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            result = conv.GetWeightsConv();
            break;
         case defNeuronConcatenate:
            conc = temp;
            result = conc.getConcatWeights();
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            result = batch.getBatchOptions();
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            result = lstm.getLSTMWeights();
            break;
         default:
            result = temp.getWeights();
            break;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint window_out, uint units_count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, window + window_out,
                                   window + window_out, window_out, units_count,
                                   variables, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   uint windows[1] = {window * 3};
   CNeuronSpikeConvBlock*           conv = NULL;
   CNeuronMultiWindowsConvWPadOCL*  pad = NULL;
   cScale.Clear();
   cScale.SetOpenCL(OpenCL);
//---
   pad = new CNeuronMultiWindowsConvWPadOCL();
   if(!pad ||
      !pad.Init(0, index, OpenCL, windows, window, (window_out + 3) / 4, units_count, variables, optimization, iBatch) ||
      !cScale.Add(pad))
     {
      DeleteObj(pad)
      ReturnFalse;
     }
   pad.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, pad.GetFilters(), pad.GetFilters(), (window_out + 1) / 2, units_count, variables, optimization, iBatch) ||
      !cScale.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   windows[0] = conv.GetFilters() * 3;
   pad = new CNeuronMultiWindowsConvWPadOCL();
   if(!pad ||
      !pad.Init(0, index, OpenCL, windows, conv.GetFilters(), conv.GetFilters(), units_count, variables, optimization, iBatch) ||
      !cScale.Add(pad))
     {
      DeleteObj(pad)
      ReturnFalse;
     }
   pad.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, pad.GetFilters(), pad.GetFilters(), window_out, units_count, variables, optimization, iBatch) ||
      !cScale.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
//---
   if(!cConcat.Init(0, index, OpenCL, (window + window_out)*units_count * variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScale.Total(); i++)
     {
      curr = cScale[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   uint units = GetUnits() * GetVariables();
   uint window_in = NeuronOCL.Neurons() / units;
   uint window_prev = prev.Neurons() / units;
   if(!Concat(NeuronOCL.getOutput(), prev.getOutput(), cConcat.getOutput(), window_in, window_prev, units))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cScale[-1];
   CNeuronBaseOCL* curr = NULL;
   uint units = GetUnits() * GetVariables();
   uint window_in = NeuronOCL.Neurons() / units;
   uint window_prev = cConcat.Neurons() / units - window_in;
   if(!next ||
      !DeConcat(cConcat.getPrevOutput(), next.getGradient(), cConcat.getGradient(), window_in, window_prev, units))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), cConcat.getPrevOutput(), cConcat.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   for(int i = cScale.Total() - 2; i >= 0; i--)
     {
      curr = cScale[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cConcat.getPrevOutput(), NeuronOCL.getGradient(), window_in, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScale.Total(); i++)
     {
      curr = cScale[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::updateInputWeights(cConcat.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSCM* Source = source;
   if(!cScale.WeightsUpdate(Source.cScale.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cScale.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!cScale.Load(file_handle))
      ReturnFalse;
//---
   if(!cConcat.Init(0, cScale.Total(), OpenCL, cConv.GetWindow() * GetUnits() * GetVariables(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSCM::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cScale.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSCM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cScale.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSCM::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cScale.TrainMode(bTrain);
   cConcat.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                     uint chanels_in, uint chanels_out,
                                     uint depth_window, uint depth_step,
                                     uint units, uint variables,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, chanels_in, chanels_in,
                                   chanels_out, units, variables, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cTranspose[0].Init(0, index, OpenCL, variables, units * depth_step, chanels_in, optimization, iBatch))
      ReturnFalse;
   cTranspose[0].SetActivationFunction(None);
   index++;
   if(!cTranspose[1].Init(0, index, OpenCL, variables, chanels_in, units, optimization, iBatch))
      ReturnFalse;
   cTranspose[1].SetActivationFunction(None);
   index++;
//---
   uint windows[] = { depth_window };
   if(!cDepthConv.Init(0, index, OpenCL, windows, depth_step, 1, units, chanels_in * variables, optimization, iBatch))
      ReturnFalse;
   cDepthConv.SetActivationFunction(None);
   index++;
   if(!cNorm.Init(0, index, OpenCL, cDepthConv.Neurons(), iBatch, optimization))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   int total_in = NeuronOCL.Neurons();
   uint units = total_in / cTranspose[0].GetVariables();
   units = (units + cTranspose[0].GetWindow() - 1) / cTranspose[0].GetWindow();
   if(units != cTranspose[0].GetCount())
     {
      if(!cTranspose[0].Init(0, 0, OpenCL, cTranspose[0].GetVariables(), units,
                             cTranspose[0].GetWindow(), optimization, iBatch))
         ReturnFalse;
      cTranspose[0].SetActivationFunction(None);
     }
   if(!cTranspose[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cDepthConv.FeedForward(cTranspose[0].AsObject()))
      ReturnFalse;
   if(!cTranspose[1].FeedForward(cDepthConv.AsObject()))
      ReturnFalse;
   if(!cNorm.FeedForward(cTranspose[1].AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::calcInputGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cTranspose[1].CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!cDepthConv.CalcHiddenGradients(cTranspose[1].AsObject()))
      ReturnFalse;
   if(!cTranspose[0].CalcHiddenGradients(cDepthConv.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cTranspose[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cDepthConv.UpdateInputWeights(cTranspose[0].AsObject()))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cTranspose[1].AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeDepthWiseConv* Source = source;
   if(!cDepthConv.WeightsUpdate(Source.cDepthConv.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < cTranspose.Size(); i++)
      if(!cTranspose[i].Save(file_handle))
         ReturnFalse;
   if(!cDepthConv.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseConv::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   for(uint i = 0; i < cTranspose.Size(); i++)
      if(!LoadInsideLayer(file_handle, cTranspose[i].AsObject()))
         ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDepthConv.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeDepthWiseConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//---
   for(uint i = 0; i < cTranspose.Size(); i++)
      cTranspose[i].SetOpenCL(OpenCL);
   cDepthConv.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeDepthWiseConv::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//---
   for(uint i = 0; i < cTranspose.Size(); i++)
      cTranspose[i].TrainMode(bTrain);
   cDepthConv.TrainMode(bTrain);
   cNorm.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint units_count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables,
                                 batch, optimization_type))
      ReturnFalse;
   activation = None;
   if(!cProduct.Init(0, 0, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cProduct.SetActivationFunction(None);
   if(!cActivation.Init(0, 1, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cConv.Init(0, 2, OpenCL, window, window, 3, 1, units_count, variables, optimization, iBatch))
      ReturnFalse;
   cConv.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(!ElementMult(NeuronOCL.getOutput(), SecondInput, cProduct.getOutput()))
      ReturnFalse;
   if(!cActivation.FeedForward(cProduct.AsObject()))
      ReturnFalse;
   if(!cConv.FeedForward(cActivation.AsObject()))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::feedForward(cConv.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), Output, Output, GetWindow(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
   if(!CNeuronBatchNormOCL::calcInputGradients(cConv.AsObject()))
      ReturnFalse;
   if(!cActivation.CalcHiddenGradients(cConv.AsObject()))
      ReturnFalse;
   if(!cProduct.CalcHiddenGradients(cActivation.AsObject()))
      ReturnFalse;
//---
   if(!ElementMultGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                       SecondInput, SecondGradient, cProduct.getGradient(),
                       NeuronOCL.Activation(), SecondActivation))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), cProduct.getPrevOutput(), Gradient, NeuronOCL.Activation()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cProduct.getPrevOutput(), NeuronOCL.getGradient(),
                       GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cActivation.UpdateInputWeights(cProduct.AsObject()))
      ReturnFalse;
   if(!cConv.UpdateInputWeights(cActivation.AsObject()))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::updateInputWeights(cConv.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::Save(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Save(file_handle))
      ReturnFalse;
   if(!cActivation.Save(file_handle))
      ReturnFalse;
   if(!cConv.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::Load(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cActivation.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConv.AsObject()))
      ReturnFalse;
   if(!cProduct.Init(0, 0, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cProduct.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::Clear(void)
  {
   if(!CNeuronBatchNormOCL::Clear())
      ReturnFalse;
   if(!cActivation.Clear())
      ReturnFalse;
   if(!cConv.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeFAM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBatchNormOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeFAM* Source = source;
   if(!cActivation.WeightsUpdate(Source.cActivation.AsObject(), tau))
      ReturnFalse;
   if(!cConv.WeightsUpdate(Source.cConv.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeFAM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBatchNormOCL::SetOpenCL(obj);
   cProduct.SetOpenCL(OpenCL);
   cActivation.SetOpenCL(OpenCL);
   cConv.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeFAM::TrainMode(bool flag)
  {
   CNeuronBatchNormOCL::TrainMode(flag);
   cActivation.TrainMode(bTrain);
   cConv.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint window_out,
                           uint units_count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
// 3 * 32 * 6 * 1
   if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, 3 * window_out * units_count * variables,
                                 batch, optimization_type))
      ReturnFalse;
//---
   cSCMs.Clear();
   cFAMs.Clear();
   cAFFs.Clear();
   cProjects.Clear();
   cProjectsOut.Clear();
   cFeatExtracts.Clear();
   cSCMs.SetOpenCL(OpenCL);
   cFAMs.SetOpenCL(OpenCL);
   cAFFs.SetOpenCL(OpenCL);
   cProjects.SetOpenCL(OpenCL);
   cProjectsOut.SetOpenCL(OpenCL);
   cFeatExtracts.SetOpenCL(OpenCL);
//---
   CNeuronSpikeConvBlock*           conv = NULL;
   CNeuronSpikeDepthWiseConv*       pad_conv = NULL;
   CNeuronSpikeSCM*                 scm = NULL;
   CNeuronSpikeFAM*                 fam = NULL;
   CNeuronBaseOCL*                  neuron = NULL;
   uint index = 0;
//---
// 576
   if(!cConcatOut.Init(0, index, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 32 * 6 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, window, window_out, 3, 1, units_count, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
   uint u = (units_count + 1) / 2;
// 64 * 3 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, window_out, 2 * window_out, 3, 2, u, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
   u = (u + 1) / 2;
// 128 * 2 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 2 * window_out, 4 * window_out, 3, 2, u, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 256 * 2 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 4 * window_out, 8 * window_out, 3, 1, u, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   u *= 2;
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 128 * 4 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 4 * window_out, 4 * window_out, 3, 1, u, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   u *= 2;
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 32 * 8 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, window_out, window_out, 5, 1, units_count, variables, optimization, iBatch) ||
      !cFeatExtracts.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
//---
   index++;
   neuron = new CNeuronBaseOCL();
// 5 * 32 * 6 * 1
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 5 * window_out * units_count * variables, optimization, iBatch) ||
      !cProjects.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
// 64 * 6 * 1
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 5 * window_out, 5 * window_out, 2 * window_out, units_count, variables, optimization, iBatch) ||
      !cProjects.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   neuron = new CNeuronBaseOCL();
// 5 * 32 * 6 * 1
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 5 * window_out * units_count * variables, optimization, iBatch) ||
      !cProjects.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
// 32 * 6 * 1
   if(!conv ||
      !conv.Init(0, index, OpenCL, 5 * window_out, 5 * window_out, window_out, units_count, variables, optimization, iBatch) ||
      !cProjects.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 32 * 6 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 4 * window_out, window_out, 3, 2, units_count, variables, optimization, iBatch) ||
      !cProjectsOut.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 32 * 6 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 2 * window_out, window_out, 3, 2, units_count, variables, optimization, iBatch) ||
      !cProjectsOut.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
//---
   index++;
   neuron = new CNeuronBaseOCL();
// 7 * 32 * 6 * 1
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 7 * window_out * units_count * variables, optimization, iBatch) ||
      !cAFFs.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
// 32 * 6 * 1
   if(!conv ||
      !conv.Init(0, index, OpenCL, 7 * window_out, 7 * window_out, window_out, units_count, variables, optimization, iBatch) ||
      !cAFFs.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 32 * 6 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, window_out, window_out, 3, 1, units_count, variables, optimization, iBatch) ||
      !cAFFs.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
// 64 * 3 * 1
   if(!conv ||
      !conv.Init(0, index, OpenCL, 14 * window_out, 14 * window_out, 2 * window_out, (units_count + 1) / 2, variables, optimization, iBatch) ||
      !cAFFs.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   pad_conv = new CNeuronSpikeDepthWiseConv();
// 64 * 3 * 1
   if(!pad_conv ||
      !pad_conv.Init(0, index, OpenCL, 2 * window_out, 2 * window_out, 3, 1, (units_count + 1) / 2, variables, optimization, iBatch) ||
      !cAFFs.Add(pad_conv))
     {
      DeleteObj(pad_conv)
      ReturnFalse;
     }
   pad_conv.SetActivationFunction(None);
//---
   index++;
   fam = new CNeuronSpikeFAM();
   pad_conv = cFeatExtracts[2];
// 128 * 6 * 1
   if(!fam ||
      !fam.Init(0, index, OpenCL, pad_conv.GetFilters(), pad_conv.GetUnits(), variables, optimization, iBatch) ||
      !cFAMs.Add(fam))
     {
      DeleteObj(fam)
      ReturnFalse;
     }
   index++;
   fam = new CNeuronSpikeFAM();
   pad_conv = cFeatExtracts[1];
// 64 * 6 * 1
   if(!fam ||
      !fam.Init(0, index, OpenCL, pad_conv.GetFilters(), pad_conv.GetUnits(), variables, optimization, iBatch) ||
      !cFAMs.Add(fam))
     {
      DeleteObj(fam)
      ReturnFalse;
     }
//---
   index++;
   scm = new CNeuronSpikeSCM();
// 64 * 3 * 1
   if(!scm ||
      !scm.Init(0, index, OpenCL, 2 * window, 2 * window_out, (units_count + 1) / 2, variables, optimization, iBatch) ||
      !cSCMs.Add(scm))
     {
      DeleteObj(scm)
      ReturnFalse;
     }
   u = scm.GetUnits();
   index++;
   scm = new CNeuronSpikeSCM();
// 128 * 2 * 1
   if(!scm ||
      !scm.Init(0, index, OpenCL, 4 * window, 4 * window_out, (u + 1) / 2, variables, optimization, iBatch) ||
      !cSCMs.Add(scm))
     {
      DeleteObj(scm)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cSCMs.Total(); i++)
     {
      curr = cSCMs[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cFeatExtracts[0] ||
      !cFeatExtracts[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cFeatExtracts[1] ||
      !cFeatExtracts[1].FeedForward(cFeatExtracts[0]))
      ReturnFalse;
   if(!cFAMs[1] ||
      !cFAMs[1].FeedForward(cFeatExtracts[1], cSCMs[0].getOutput()))
      ReturnFalse;
   if(!cFeatExtracts[2] ||
      !cFeatExtracts[2].FeedForward(cFAMs[1]))
      ReturnFalse;
   if(!cFAMs[0] ||
      !cFAMs[0].FeedForward(cFeatExtracts[2], cSCMs[1].getOutput()))
      ReturnFalse;
//--- AAFF
   CNeuronSpikeConvBlock* conv = cAFFs[1];
   if(!conv)
      ReturnFalse;
   uint units = conv.GetUnits() * conv.GetVariables();
   uint window1 = cFeatExtracts[0].Neurons() / units;
   uint window2 = cFAMs[1].Neurons() / units;
   uint window3 = cFAMs[0].Neurons() / units;
   prev = cAFFs[0];
   if(!prev ||
      !Concat(cFeatExtracts[0].getOutput(), cFAMs[1].getOutput(), cFAMs[0].getOutput(),
              prev.getOutput(), window1, window2, window3, units))
      ReturnFalse;
   for(int i = 1; i < cAFFs.Total(); i += 2)
     {
      conv = cAFFs[i];
      curr = cAFFs[i + 1];
      if(!conv ||
         !conv.FeedForward(prev))
         ReturnFalse;
      if(!curr ||
         !curr.FeedForward(conv))
         ReturnFalse;
     }
//---
   prev = cProjectsOut[0];
   curr = cFeatExtracts[3];
   if(!prev ||
      !prev.FeedForward(cFAMs[0]))
      ReturnFalse;
   if(!curr ||
      !curr.FeedForward(cFAMs[0]))
      ReturnFalse;
//---
   window1 = curr.Neurons() / units;
   window2 = cAFFs[4].Neurons() / units;
   prev = cProjects[0];
   if(!prev ||
      !Concat(curr.getOutput(), cAFFs[4].getOutput(), prev.getOutput(),
              window1, window2, units))
      ReturnFalse;
   if(!cProjects[1] ||
      !cProjects[1].FeedForward(prev))
      ReturnFalse;
   if(!cProjectsOut[1] ||
      !cProjectsOut[1].FeedForward(cProjects[1]))
      ReturnFalse;
   if(!cFeatExtracts[4] ||
      !cFeatExtracts[4].FeedForward(cProjects[1]))
      ReturnFalse;
//---
   window1 = cFeatExtracts[4].Neurons() / units;
   window2 = cAFFs[2].Neurons() / units;
   prev = cProjects[2];
   if(!prev ||
      !Concat(cFeatExtracts[4].getOutput(), cAFFs[2].getOutput(), prev.getOutput(),
              window1, window2, units))
      ReturnFalse;
   if(!cProjects[3] ||
      !cProjects[3].FeedForward(prev))
      ReturnFalse;
   if(!cFeatExtracts[5] ||
      !cFeatExtracts[5].FeedForward(cProjects[3]))
      ReturnFalse;
   window1 = cProjectsOut[0].Neurons() / units;
   window2 = cProjectsOut[1].Neurons() / units;
   window3 = cFeatExtracts[5].Neurons() / units;
//---
   if(!Concat(cProjectsOut[0].getOutput(), cProjectsOut[1].getOutput(), cFeatExtracts[5].getOutput(),
              cConcatOut.getOutput(), window1, window2, window3, units))
      ReturnFalse;
   window1 = NeuronOCL.Neurons() / units;
   if(!SumVecMatrix(NeuronOCL.getOutput(), cConcatOut.getOutput(), cConcatOut.getOutput(),
                    window1, units, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::feedForward(cConcatOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronBatchNormOCL::calcInputGradients(cConcatOut.AsObject()))
      ReturnFalse;
   CNeuronSpikeConvBlock* conv = cAFFs[1];
   if(!conv)
      ReturnFalse;
   uint units = conv.GetUnits() * conv.GetVariables();
   uint window1 = NeuronOCL.Neurons() / units;
   if(!SumVecMatrixGrad(cConcatOut.getPrevOutput(), cConcatOut.getGradient(),
                        cConcatOut.getGradient(), window1, units, 0, 0, 0, 1))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), cConcatOut.getPrevOutput(),
                       cConcatOut.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   if(!cProjectsOut[0] || !cProjectsOut[1] || !cFeatExtracts[5])
      ReturnFalse;
   window1 = cProjectsOut[0].Neurons() / units;
   uint window2 = cProjectsOut[1].Neurons() / units;
   uint window3 = cFeatExtracts[5].Neurons() / units;
   if(!DeConcat(cProjectsOut[0].getGradient(), cProjectsOut[1].getGradient(), cFeatExtracts[5].getGradient(),
                cConcatOut.getGradient(), window1, window2, window3, units))
      ReturnFalse;
   Deactivation(cFeatExtracts[5])
   for(int i = 0; i < 2; i++)
     {
      Deactivation(cProjectsOut[i])
     }
//---
   if(!cProjects[3] ||
      !cProjects[3].CalcHiddenGradients(cFeatExtracts[5]))
      ReturnFalse;
   if(!cProjects[2] ||
      !cProjects[2].CalcHiddenGradients(cProjects[3]))
      ReturnFalse;
   if(!cFeatExtracts[4] || !cAFFs[2])
      ReturnFalse;
   window1 = cFeatExtracts[4].Neurons() / units;
   window2 = cAFFs[2].Neurons() / units;
   if(!DeConcat(cFeatExtracts[4].getGradient(), cAFFs[2].getGradient(), cProjects[2].getGradient(),
                window1, window2, units))
      ReturnFalse;
   Deactivation(cFeatExtracts[4])
   Deactivation(cAFFs[2])
   CNeuronBaseOCL* curr = cProjects[1];
   if(!curr ||
      !curr.CalcHiddenGradients(cFeatExtracts[4]))
      ReturnFalse;
   CBufferFloat* temp = curr.getGradient();
   if(!curr.SetGradient(curr.getPrevOutput(), false) ||
      !curr.CalcHiddenGradients(cProjectsOut[1]) ||
      !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !curr.SetGradient(temp, false))
      ReturnFalse;
   if(!cProjects[0] ||
      !cProjects[0].CalcHiddenGradients(curr))
      ReturnFalse;
//---
   if(!cFeatExtracts[3] || !cAFFs[4])
      ReturnFalse;
   window1 = cFeatExtracts[3].Neurons() / units;
   window2 = cAFFs[4].Neurons() / units;
   if(!DeConcat(cFeatExtracts[3].getGradient(), cAFFs[4].getGradient(), cProjects[0].getGradient(),
                window1, window2, units))
      ReturnFalse;
   Deactivation(cFeatExtracts[3])
   Deactivation(cAFFs[4])
//---
   curr = cFAMs[0];
   if(!curr ||
      !curr.CalcHiddenGradients(cFeatExtracts[3]))
      ReturnFalse;
   temp = curr.getGradient();
   if(!curr.SetGradient(curr.getPrevOutput(), false) ||
      !curr.CalcHiddenGradients(cProjectsOut[0]) ||
      !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !curr.SetGradient(temp, false))
      ReturnFalse;
//---
   curr = cAFFs[0];
   if(!cAFFs[3] || !curr ||
      !cAFFs[3].CalcHiddenGradients(cAFFs[4]) ||
      !curr.CalcHiddenGradients(cAFFs[3]))
      ReturnFalse;
   temp = curr.getGradient();
   if(!curr.SetGradient(curr.getPrevOutput(), false) ||
      !cAFFs[1] ||
      !cAFFs[1].CalcHiddenGradients(cAFFs[2]) ||
      !curr.CalcHiddenGradients(cAFFs[1]) ||
      !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !curr.SetGradient(temp, false))
      ReturnFalse;
//---
   window1 = cFeatExtracts[0].Neurons() / units;
   window2 = cFAMs[1].Neurons() / units;
   window3 = cFAMs[0].Neurons() / units;
   if(!DeConcat(cFeatExtracts[0].getPrevOutput(), cFAMs[1].getPrevOutput(), cFAMs[0].getPrevOutput(),
                curr.getGradient(), window1, window2, window3, units))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      curr = cFAMs[1];
      if(curr.Activation() != None)
         if(!DeActivation(curr.getOutput(), curr.getPrevOutput(),
                          curr.getPrevOutput(), curr.Activation()))
            ReturnFalse;
     }
   curr = cFeatExtracts[0];
   if(curr.Activation() != None)
      if(!DeActivation(curr.getOutput(), curr.getPrevOutput(),
                       curr.getPrevOutput(), curr.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cFAMs[0].getPrevOutput(), cFAMs[0].getGradient(),
                       cFAMs[0].getGradient(), window3, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cFeatExtracts[2] ||
      !cFeatExtracts[2].CalcHiddenGradients(cFAMs[0], cSCMs[1].getOutput(),
            cSCMs[1].getGradient(),
            (ENUM_ACTIVATION)cSCMs[1].Activation()))
      ReturnFalse;
   if(!cFAMs[1].CalcHiddenGradients(cFeatExtracts[2]) ||
      !SumAndNormalize(cFAMs[1].getPrevOutput(), cFAMs[1].getGradient(),
                       cFAMs[1].getGradient(), window2, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cFeatExtracts[1] ||
      !cFeatExtracts[1].CalcHiddenGradients(cFAMs[1], cSCMs[0].getOutput(),
            cSCMs[0].getPrevOutput(),
            (ENUM_ACTIVATION)cSCMs[0].Activation()))
      ReturnFalse;
   if(!cFeatExtracts[0] ||
      !cFeatExtracts[0].CalcHiddenGradients(cFeatExtracts[1]))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cFeatExtracts[0]) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cConcatOut.getPrevOutput(),
                       cConcatOut.getPrevOutput(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cSCMs[0].CalcHiddenGradients(cSCMs[1]) ||
      !SumAndNormalize(cSCMs[0].getPrevOutput(), cSCMs[0].getGradient(),
                       cSCMs[0].getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cSCMs[0]) ||
      !SumAndNormalize(NeuronOCL.getGradient(), cConcatOut.getPrevOutput(),
                       NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cSCMs.Total(); i++)
     {
      curr = cSCMs[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cFeatExtracts[0] ||
      !cFeatExtracts[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cFeatExtracts[1] ||
      !cFeatExtracts[1].UpdateInputWeights(cFeatExtracts[0]))
      ReturnFalse;
   if(!cFAMs[1] ||
      !cFAMs[1].UpdateInputWeights(cFeatExtracts[1], cSCMs[0].getOutput()))
      ReturnFalse;
   if(!cFeatExtracts[2] ||
      !cFeatExtracts[2].UpdateInputWeights(cFAMs[1]))
      ReturnFalse;
   if(!cFAMs[0] ||
      !cFAMs[0].UpdateInputWeights(cFeatExtracts[2], cSCMs[1].getOutput()))
      ReturnFalse;
//--- AAFF
   prev = cAFFs[0];
   CNeuronSpikeConvBlock* conv = NULL;
   for(int i = 1; i < cAFFs.Total(); i += 2)
     {
      conv = cAFFs[i];
      curr = cAFFs[i + 1];
      if(!conv ||
         !conv.UpdateInputWeights(prev))
         ReturnFalse;
      if(!curr ||
         !curr.UpdateInputWeights(conv))
         ReturnFalse;
     }
//---
   prev = cProjectsOut[0];
   curr = cFeatExtracts[3];
   if(!prev ||
      !prev.UpdateInputWeights(cFAMs[0]))
      ReturnFalse;
   if(!curr ||
      !curr.UpdateInputWeights(cFAMs[0]))
      ReturnFalse;
//---
   prev = cProjects[0];
   if(!cProjects[1] ||
      !cProjects[1].UpdateInputWeights(prev))
      ReturnFalse;
   if(!cProjectsOut[1] ||
      !cProjectsOut[1].UpdateInputWeights(cProjects[1]))
      ReturnFalse;
   if(!cFeatExtracts[4] ||
      !cFeatExtracts[4].UpdateInputWeights(cProjects[1]))
      ReturnFalse;
//---
   prev = cProjects[2];
   if(!cProjects[3] ||
      !cProjects[3].UpdateInputWeights(prev))
      ReturnFalse;
   if(!cFeatExtracts[5] ||
      !cFeatExtracts[5].UpdateInputWeights(cProjects[3]))
      ReturnFalse;
   if(!CNeuronBatchNormOCL::updateInputWeights(cConcatOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::Save(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Save(file_handle))
      ReturnFalse;
   if(!cSCMs.Save(file_handle))
      ReturnFalse;
   if(!cFAMs.Save(file_handle))
      ReturnFalse;
   if(!cAFFs.Save(file_handle))
      ReturnFalse;
   if(!cProjects.Save(file_handle))
      ReturnFalse;
   if(!cProjectsOut.Save(file_handle))
      ReturnFalse;
   if(!cFeatExtracts.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::Load(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Load(file_handle))
      ReturnFalse;
   if(!cSCMs.Load(file_handle))
      ReturnFalse;
   if(!cFAMs.Load(file_handle))
      ReturnFalse;
   if(!cAFFs.Load(file_handle))
      ReturnFalse;
   if(!cProjects.Load(file_handle))
      ReturnFalse;
   if(!cProjectsOut.Load(file_handle))
      ReturnFalse;
   if(!cFeatExtracts.Load(file_handle))
      ReturnFalse;
//---
   if(!cConcatOut.Init(0, 0, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBatchNormOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeMDC* Source = source;
   dWeightsUpdate(cSCMs, Source, tau)
   dWeightsUpdate(cFAMs, Source, tau)
   dWeightsUpdate(cAFFs, Source, tau)
   dWeightsUpdate(cProjects, Source, tau)
   dWeightsUpdate(cProjectsOut, Source, tau)
   dWeightsUpdate(cFeatExtracts, Source, tau)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMDC::Clear(void)
  {
   if(!CNeuronBatchNormOCL::Clear())
      ReturnFalse;
   if(!cSCMs.ClearStates())
      ReturnFalse;
   if(!cFAMs.ClearStates())
      ReturnFalse;
   if(!cAFFs.ClearStates())
      ReturnFalse;
   if(!cProjects.ClearStates())
      ReturnFalse;
   if(!cProjectsOut.ClearStates())
      ReturnFalse;
   if(!cFeatExtracts.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMDC::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBatchNormOCL::SetOpenCL(obj);
   cSCMs.SetOpenCL(OpenCL);
   cFAMs.SetOpenCL(OpenCL);
   cAFFs.SetOpenCL(OpenCL);
   cProjects.SetOpenCL(OpenCL);
   cProjectsOut.SetOpenCL(OpenCL);
   cFeatExtracts.SetOpenCL(OpenCL);
   cConcatOut.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMDC::TrainMode(bool flag)
  {
   CNeuronBatchNormOCL::TrainMode(flag);
   cSCMs.TrainMode(bTrain);
   cFAMs.TrainMode(bTrain);
   cAFFs.TrainMode(bTrain);
   cProjects.TrainMode(bTrain);
   cProjectsOut.TrainMode(bTrain);
   cFeatExtracts.TrainMode(bTrain);
   cConcatOut.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                      uint window, uint units_count, uint variables,
                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   iDimension = window;
   iUnits = units_count;
   iVariables = variables;
//---
   uint index = 0;
   if(!cConcat.Init(0, index, OpenCL, 4 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   CNeuronConvOCL*      conv = NULL;
   CNeuronBatchNormOCL* norm = NULL;
   CNeuronSoftMaxOCL*   softmax = NULL;
   cScores.Clear();
   cScores.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 4 * iDimension, 4 * iDimension, iDimension, iUnits, iVariables, optimization, iBatch) ||
      !cScores.Add(conv))
      ReturnFalse;
   conv.SetActivationFunction(None);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cScores.Add(norm))
      ReturnFalse;
   norm.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, iDimension, iDimension, 4, iUnits, iVariables, optimization, iBatch) ||
      !cScores.Add(conv))
      ReturnFalse;
   conv.SetActivationFunction(None);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cScores.Add(norm))
      ReturnFalse;
   norm.SetActivationFunction(SoftPlus);
   index++;
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, norm.Neurons(), optimization, iBatch) ||
      !cScores.Add(softmax))
      ReturnFalse;
   softmax.SetHeads(iUnits * iVariables);
   index++;
   if(!cTranspose.Init(0, index, OpenCL, iUnits * iVariables, 4, iDimension, optimization, iBatch))
      ReturnFalse;
   cTranspose.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(!Concat(NeuronOCL.getOutput(), SecondInput, cConcat.getOutput(), iDimension, 3 * iDimension, iUnits * iVariables))
      ReturnFalse;
   CNeuronBaseOCL* prev = cConcat.AsObject();
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScores.Total(); i++)
     {
      curr = cScores[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cTranspose.FeedForward(cConcat.AsObject()))
      ReturnFalse;
   if(!MatMul(cTranspose.getOutput(), curr.getOutput(), Output, iDimension, 4, 1, iUnits * iVariables, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                    CBufferFloat *SecondInput,
                                    CBufferFloat *SecondGradient,
                                    ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = cScores[-1];
   if(!curr ||
      !MatMulGrad(cTranspose.getOutput(), cTranspose.getGradient(),
                  curr.getOutput(), curr.getGradient(),
                  Gradient, iDimension, 4, 1, iUnits * iVariables, true))
      ReturnFalse;
   Deactivation(curr)
   Deactivation(cTranspose)
   if(!cConcat.CalcHiddenGradients(cTranspose.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = cConcat.getGradient();
   if(!cConcat.SetGradient(cTranspose.getPrevOutput(), false))
      ReturnFalse;
//---
   for(int i = cScores.Total() - 2; i >= 0; i--)
     {
      curr = cScores[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cScores[i + 1]))
         ReturnFalse;
     }
   if(!cConcat.CalcHiddenGradients(curr) ||
      !SumAndNormalize(cConcat.getGradient(), temp, temp, iDimension, false, 0, 0, 0, 1) ||
      !cConcat.SetGradient(temp, false))
      ReturnFalse;
//---
   if(!DeConcat(NeuronOCL.getGradient(), SecondGradient, cConcat.getGradient(),
                iDimension, 3 * iDimension, iUnits * iVariables))
      ReturnFalse;
   Deactivation(NeuronOCL)
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   CNeuronBaseOCL* prev = cConcat.AsObject();
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScores.Total(); i++)
     {
      curr = cScores[i];
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
bool CNeuronMDS::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronMDS* Source = source;
   dWeightsUpdate(cScores, Source, tau)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cScores.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronMDS::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!cScores.Load(file_handle))
      ReturnFalse;
//---
   if(!cScores[0] || cScores[0].Type() != defNeuronConvOCL)
      ReturnFalse;
   CNeuronConvOCL* conv = cScores[0];
   iDimension = conv.GetFilters();
   iUnits = conv.GetUnits();
   iVariables = conv.GetVariables();
//---
   if(!cConcat.Init(0, 0, OpenCL, 4 * Neurons(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
   if(!cTranspose.Init(0, cScores.Total() + 1, OpenCL, iUnits * iVariables, 4, iDimension, optimization, iBatch))
      ReturnFalse;
   cTranspose.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronMDS::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cScores.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
   cTranspose.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint units_count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMDS::Init(numOutputs, myIndex, open_cl, window, units_count, variables, optimization_type, batch))
      ReturnFalse;
   if(!cConv.Init(0, 0, OpenCL, window, window, window, units_count, variables, optimization, iBatch))
      ReturnFalse;
   if(!cMDC.Init(0, 1, OpenCL, window, window, units_count, variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cConv.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cMDC.FeedForward(cConv.AsObject()))
      ReturnFalse;
   if(!CNeuronMDS::feedForward(cConv.AsObject(), cMDC.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronMDS::calcInputGradients(cConv.AsObject(), cMDC.getOutput(),
                                      cMDC.getGradient(), (ENUM_ACTIVATION)cMDC.Activation()))
      ReturnFalse;
   CBufferFloat* temp = cConv.getGradient();
   if(!cConv.SetGradient(cConv.getPrevOutput(), false) ||
      !cConv.CalcHiddenGradients(cMDC.AsObject()) ||
      !SumAndNormalize(cConv.getGradient(), temp, temp, cConv.GetFilters(), false, 0, 0, 0, 1) ||
      !cConv.SetGradient(temp, false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cConv.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cConv.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cMDC.UpdateInputWeights(cConv.AsObject()))
      ReturnFalse;
   if(!CNeuronMDS::updateInputWeights(cConv.AsObject(), cMDC.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMDS::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikeADM* Source = source;
   dWeightsUpdate(cConv, Source, tau)
   dWeightsUpdate(cMDC, Source, tau)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::Save(const int file_handle)
  {
   if(!CNeuronMDS::Save(file_handle))
      ReturnFalse;
   if(!cConv.Save(file_handle))
      ReturnFalse;
   if(!cMDC.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::Load(const int file_handle)
  {
   if(!CNeuronMDS::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConv.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMDC.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeADM::Clear(void)
  {
   if(!CNeuronMDS::Clear())
      ReturnFalse;
   if(!cConv.Clear())
      ReturnFalse;
   if(!cMDC.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeADM::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMDS::SetOpenCL(obj);
   cConv.SetOpenCL(OpenCL);
   cMDC.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeADM::TrainMode(bool flag)
  {
   CNeuronMDS::TrainMode(flag);
   cConv.TrainMode(bTrain);
   cMDC.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBATvADM::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint embed_size,
                          uint stack_size, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBAT::Init(numOutputs, myIndex, open_cl, dimension, embed_size,
                        stack_size, variables, optimization_type, batch))
      ReturnFalse;
//---
   int index = cPrepare.Total();
   CNeuronSpikeADM* amd = new CNeuronSpikeADM();
   if(!amd ||
      !amd.Init(0, index, OpenCL, embed_size, stack_size, variables, optimization, iBatch) ||
      !cPrepare.Add(amd))
     {
      DeleteObj(amd)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint window, uint &inside_dimensions[], uint window_out,
                                   uint units_count, uint variables,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint layers = inside_dimensions.Size();
   if(layers <= 0)
      ReturnFalse;
   if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, window_out * units_count * variables,
                                 batch, optimization_type))
      ReturnFalse;
   activation = None;
//---
   iUnits = units_count;
   iVariables = variables;
//---
   cEstimator.Clear();
   cEstimator.SetOpenCL(OpenCL);
   CNeuronMultiWindowsConvWPadOCL*  conv = NULL;
   CNeuronBatchNormOCL*             norm = NULL;
   CNeuronBaseOCL*                  concat = NULL;
   uint index = 0;
   uint inp_window = window;
   uint windows[1] = { 3 * inp_window };
   for(uint i = 0; i < layers; i++)
     {
      conv = new CNeuronMultiWindowsConvWPadOCL();
      if(!conv ||
         !conv.Init(0, index, OpenCL, windows, inp_window, inside_dimensions[i],
                    iUnits, iVariables, optimization, iBatch) ||
         !cEstimator.Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      conv.SetActivationFunction(SoftPlus);
      index++;
      inp_window += conv.GetFilters();
      windows[0] =  3 * inp_window;
      norm = new CNeuronBatchNormOCL();
      if(!norm ||
         !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
         !cEstimator.Add(norm))
        {
         DeleteObj(norm)
         ReturnFalse;
        }
      norm.SetActivationFunction(None);
      index++;
      concat = new CNeuronBaseOCL();
      if(!concat ||
         !concat.Init(0, index, OpenCL, inp_window * iUnits * iVariables, optimization, iBatch) ||
         !cEstimator.Add(concat))
        {
         DeleteObj(concat)
         ReturnFalse;
        }
      index++;
      concat.SetActivationFunction(None);
     }
//---
   conv = new CNeuronMultiWindowsConvWPadOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, windows, inp_window, window_out,
                 iUnits, iVariables, optimization, iBatch) ||
      !cEstimator.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* residual = NeuronOCL;
   CNeuronBaseOCL* prev = residual;
   CNeuronBaseOCL* curr = NULL;
   if(!residual)
      ReturnFalse;
//---
   uint count = iUnits * iVariables;
   uint res_window = residual.Neurons() / count;
   int total = cEstimator.Total();
//---
   for(int i = 0; i < total; i++)
     {
      curr = cEstimator[i];
      if(!curr)
         ReturnFalse;
      if(curr.Type() == defNeuronBaseOCL)
        {
         if(!Concat(residual.getOutput(), prev.getOutput(), curr.getOutput(),
                    res_window, prev.Neurons() / count, count))
            ReturnFalse;
         residual = curr;
         res_window = curr.Neurons() / count;
        }
      else
         if(!curr.FeedForward(prev))
            ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronBatchNormOCL::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronBatchNormOCL::calcInputGradients(cEstimator[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cEstimator[-1];
   CNeuronBaseOCL* curr = NULL;
   CNeuronBaseOCL* residual = NULL;
   uint count = iUnits * iVariables;
//---
   for(int i = cEstimator.Total() - 2; i >= 0; i--)
     {
      curr = cEstimator[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      if(curr.Type() == defNeuronBaseOCL && i > 0)
        {
         int concat_win = curr.Neurons() / (int)count;
         if(!!residual)
            if(!SumAndNormalize(curr.getGradient(), residual.getPrevOutput(), curr.getGradient(),
                                concat_win, false, 0, 0, 0, 1))
               ReturnFalse;
         next = cEstimator[i - 1];
         if(!next)
            ReturnFalse;
         int prev_win = next.Neurons() / (int)count;
         if(!DeConcat(curr.getPrevOutput(), next.getGradient(), curr.getGradient(),
                      concat_win - prev_win, prev_win, count))
            ReturnFalse;
         residual = curr;
         Deactivation(next);
         i--;
        }
      else
         next = curr;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
   if(!!residual)
     {
      if(NeuronOCL.Activation() != None)
         if(!DeActivation(NeuronOCL.getOutput(), residual.getPrevOutput(),
                          residual.getPrevOutput(), NeuronOCL.Activation()))
            ReturnFalse;
      if(!SumAndNormalize(NeuronOCL.getGradient(), residual.getPrevOutput(), NeuronOCL.getGradient(),
                          NeuronOCL.Neurons() / count, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   int total = cEstimator.Total();
//---
   for(int i = 0; i < total; i++)
     {
      curr = cEstimator[i];
      if(!curr)
         ReturnFalse;
      if(curr.Type() != defNeuronBaseOCL)
         if(!curr.UpdateInputWeights(prev))
            ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronBatchNormOCL::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBatchNormOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronCDCSelfCorrector* Source = source;
   if(!cEstimator.WeightsUpdate(Source.cEstimator.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::Save(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Save(file_handle))
      ReturnFalse;
   if(!cEstimator.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::Load(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Load(file_handle))
      ReturnFalse;
   if(!cEstimator.Load(file_handle))
      ReturnFalse;
//---
   if(!cEstimator[0] ||
      cEstimator[0].Type() != defNeuronMultiWindowsConvWPadOCL)
      ReturnFalse;
   CNeuronConvOCL* conv = cEstimator[0];
   iUnits = conv.GetUnits();
   iVariables = conv.GetVariables();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCDCSelfCorrector::Clear(void)
  {
   if(!CNeuronBatchNormOCL::Clear())
      ReturnFalse;
   if(!cEstimator.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCDCSelfCorrector::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBatchNormOCL::SetOpenCL(obj);
   cEstimator.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCDCSelfCorrector::TrainMode(bool flag)
  {
   CNeuronBatchNormOCL::TrainMode(flag);
   cEstimator.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint &inside_dimensions[], uint window_out,
                           uint units_count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, window, window, window_out,
                                   units_count, variables, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cEsimator.Init(0, index, OpenCL, window, inside_dimensions, 2 * window, units_count,
                      variables, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cFlowCorrector.Init(0, index, OpenCL, cEsimator.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cFlowCorrector.SetActivationFunction(None);
   index++;
   if(!cFlowCorrelation.Init(0, index, OpenCL, window, (window + 3) / 4, units_count * variables,
                             4, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cFlowMask.Init(0, index, OpenCL, cFlowCorrector.Neurons(), optimization, iBatch))
      ReturnFalse;
   cFlowMask.SetActivationFunction(SIGMOID);
   index++;
   if(!cFlow.Init(0, index, OpenCL, cFlowCorrector.Neurons(), optimization, iBatch))
      ReturnFalse;
   cFlow.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEsimator.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint count = cConv.GetUnits() * cConv.GetVariables();
   uint window = cConv.GetWindow();
   if(!DeConcat(cFlowCorrector.getOutput(), cFlowMask.getOutput(),
                cEsimator.getOutput(), window, window, count))
      ReturnFalse;
   dActivation(cFlowCorrector)
   dActivation(cFlowMask)
   if(!SumAndNormalize(cFlowCorrector.getOutput(), NeuronOCL.getOutput(),
                       cFlowCorrector.getOutput(), window, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cFlowCorrelation.FeedForward(cFlowCorrector.AsObject()))
      ReturnFalse;
   if(!GateElementMult(NeuronOCL.getOutput(), cFlowCorrelation.getOutput(),
                       cFlowMask.getOutput(), cFlow.getOutput()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::feedForward(cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cFlow.AsObject()))
      ReturnFalse;
   if(!GateElementMultGrad(NeuronOCL.getOutput(), cFlowMask.getPrevOutput(),
                           cFlowCorrelation.getOutput(), cFlowCorrelation.getGradient(),
                           cFlowMask.getOutput(), cFlowMask.getGradient(),
                           cFlow.getGradient(), NeuronOCL.Activation(),
                           cFlowCorrelation.Activation(), cFlowMask.Activation()))
      ReturnFalse;
   if(!cFlowCorrector.CalcHiddenGradients(cFlowCorrelation.AsObject()))
      ReturnFalse;
   uint count = cConv.GetUnits() * cConv.GetVariables();
   uint window = cFlowCorrector.Neurons() / count;
   if(!Concat(cFlowCorrector.getGradient(), cFlowMask.getGradient(),
              cEsimator.getGradient(), window, window, count))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cEsimator.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cFlowMask.getPrevOutput(),
                       NeuronOCL.getGradient(), window, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cEsimator.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cFlowCorrelation.UpdateInputWeights(cFlowCorrector.AsObject()))
      ReturnFalse;
   if(!CNeuronSpikeConvBlock::updateInputWeights(cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikeCDC* Source = source;
   if(!cEsimator.WeightsUpdate(Source.cEsimator.AsObject(), tau))
      ReturnFalse;
   if(!cFlowCorrelation.WeightsUpdate(Source.cFlowCorrelation.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cEsimator.Save(file_handle))
      ReturnFalse;
   if(!cFlowCorrelation.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEsimator.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFlowCorrelation.AsObject()))
      ReturnFalse;
//---
   uint index = 1;
   if(!cFlowCorrector.Init(0, index, OpenCL, cEsimator.Neurons() / 2, optimization, iBatch))
      ReturnFalse;
   cFlowCorrector.SetActivationFunction(None);
   index += 2;
   if(!cFlowMask.Init(0, index, OpenCL, cFlowCorrector.Neurons(), optimization, iBatch))
      ReturnFalse;
   cFlowMask.SetActivationFunction(SIGMOID);
   index++;
   if(!cFlow.Init(0, index, OpenCL, cFlowCorrector.Neurons(), optimization, iBatch))
      ReturnFalse;
   cFlow.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeCDC::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
   if(!cEsimator.Clear())
      ReturnFalse;
   if(!cFlowCorrelation.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeCDC::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cEsimator.SetOpenCL(OpenCL);
   cFlowCorrector.SetOpenCL(OpenCL);
   cFlowCorrelation.SetOpenCL(OpenCL);
   cFlowMask.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeCDC::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cEsimator.TrainMode(bTrain);
   cFlowCorrector.TrainMode(bTrain);
   cFlowCorrelation.TrainMode(bTrain);
   cFlowMask.TrainMode(bTrain);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint dimension, uint embed_size,
                               uint stack_size, uint variables,
                               uint max_displacement,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint inside_dimensions[] = {32, 32, 32, 16, 8};
   if(!CNeuronSpikeCDC::Init(numOutputs, myIndex, open_cl, embed_size, inside_dimensions,
                             dimension, stack_size, variables, optimization_type, batch))
      ReturnFalse;
//---
   CNeuronCreateFlow*               stack = NULL;
   CNeuronBatchNormOCL*             norm = NULL;
   CNeuronSpikeADM*                 adm = NULL;
   CNeuronMultiWindowsConvWPadOCL*  padconv = NULL;
   CNeuronSpikeConvBlock*           conv = NULL;
   CNeuronSpikeCDC*                 cdc = NULL;
//---
   uint index = 0;
   cPrepare.Clear();
   cPrepare.SetOpenCL(OpenCL);
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, dimension * variables, iBatch, optimization) ||
      !cPrepare.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   index++;
   stack = new CNeuronCreateFlow();
   if(!stack ||
      !stack.Init(0, index, OpenCL, stack_size, dimension, embed_size, variables, optimization, iBatch) ||
      !cPrepare.Add(stack))
     {
      DeleteObj(stack)
      ReturnFalse;
     }
   index++;
   adm = new CNeuronSpikeADM();
   if(!adm ||
      !adm.Init(0, index, OpenCL, embed_size, stack_size, variables, optimization, iBatch) ||
      !cPrepare.Add(adm))
     {
      DeleteObj(adm)
      ReturnFalse;
     }
   index++;
//---
   int shifts[];
   if(ArrayResize(shifts, max_displacement) <= 0)
      ReturnFalse;
   uint count = 0;
   for(uint i = 0; i < max_displacement; i++)
     {
      if(i > 2 && bool(i & 1))
         continue;
      shifts[count] = (int)(i + 1);
      count++;
     }
   if(ArrayResize(shifts, count) < (int)count)
      ReturnFalse;
   ibShifts = OpenCL.AddBufferFromArray(shifts, 0, count, CL_MEM_READ_ONLY);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
   iMaxDisplacement = max_displacement;
//---
   uint window_in = embed_size;
   uint window_out = 16;
   uint units_in = stack_size;
   uint units_out = (units_in + 1) / 2;
   uint windows[] = { 3 * window_in };
//---
   for(int i = 0; i < 3; i++)
     {
      cEncoder[i].Clear();
      cDecoder[i].Clear();
      cEncoder[i].SetOpenCL(OpenCL);
      cDecoder[i].SetOpenCL(OpenCL);
      //--- Encoder
      padconv = new CNeuronMultiWindowsConvWPadOCL();
      if(!padconv ||
         !padconv.Init(0, index, OpenCL, windows, 2 * window_in, window_out,
                       units_out, variables, optimization, iBatch) ||
         !cEncoder[i].Add(padconv))
        {
         DeleteObj(padconv)
         ReturnFalse;
        }
      padconv.SetActivationFunction(SoftPlus);
      index++;
      norm = new CNeuronBatchNormOCL();
      if(!norm ||
         !norm.Init(0, index, OpenCL, padconv.Neurons(), iBatch, optimization) ||
         !cEncoder[i].Add(norm))
        {
         DeleteObj(norm)
         ReturnFalse;
        }
      norm.SetActivationFunction(None);
      index++;
      windows[0] = 3 * window_out;
      for(int j = 0; j < 2; j++)
        {
         padconv = new CNeuronMultiWindowsConvWPadOCL();
         if(!padconv ||
            !padconv.Init(0, index, OpenCL, windows, window_out, window_out,
                          units_out, variables, optimization, iBatch) ||
            !cEncoder[i].Add(padconv))
           {
            DeleteObj(padconv)
            ReturnFalse;
           }
         padconv.SetActivationFunction(SoftPlus);
         index++;
         norm = new CNeuronBatchNormOCL();
         if(!norm ||
            !norm.Init(0, index, OpenCL, padconv.Neurons(), iBatch, optimization) ||
            !cEncoder[i].Add(norm))
           {
            DeleteObj(norm)
            ReturnFalse;
           }
         norm.SetActivationFunction(None);
         index++;
        }
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, window_out, window_out, window_out, units_out,
                    variables, optimization, iBatch) ||
         !cEncoder[i].Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      index++;
      //---
      if(!cCorrelations[i].Init(0, index, OpenCL, 2 * count * units_out * variables,
                                optimization, iBatch))
         ReturnFalse;
      cCorrelations[i].SetActivationFunction(None);
      index++;
      if(!cConcat[i].Init(0, index, OpenCL, 2 * (window_out + count) * units_out * variables,
                          optimization, iBatch))
         ReturnFalse;
      cConcat[i].SetActivationFunction(None);
      index++;
      //--- Decoder
      windows[0] = 2 * (window_out + count);
      padconv = new CNeuronMultiWindowsConvWPadOCL();
      if(!padconv ||
         !padconv.Init(0, index, OpenCL, windows, windows[0], window_out,
                       units_out, variables, optimization, iBatch) ||
         !cDecoder[i].Add(padconv))
        {
         DeleteObj(padconv)
         ReturnFalse;
        }
      padconv.SetActivationFunction(SoftPlus);
      index++;
      norm = new CNeuronBatchNormOCL();
      if(!norm ||
         !norm.Init(0, index, OpenCL, padconv.Neurons(), iBatch, optimization) ||
         !cDecoder[i].Add(norm))
        {
         DeleteObj(norm)
         ReturnFalse;
        }
      norm.SetActivationFunction(None);
      index++;
      windows[0] = 3 * window_out;
      padconv = new CNeuronMultiWindowsConvWPadOCL();
      if(!padconv ||
         !padconv.Init(0, index, OpenCL, windows, window_out, 2 * window_in,
                       units_out, variables, optimization, iBatch) ||
         !cDecoder[i].Add(padconv))
        {
         DeleteObj(padconv)
         ReturnFalse;
        }
      padconv.SetActivationFunction(SoftPlus);
      index++;
      norm = new CNeuronBatchNormOCL();
      if(!norm ||
         !norm.Init(0, index, OpenCL, padconv.Neurons(), iBatch, optimization) ||
         !cDecoder[i].Add(norm))
        {
         DeleteObj(norm)
         ReturnFalse;
        }
      norm.SetActivationFunction(None);
      index++;
      windows[0] = 3 * window_in;
      for(int j = 0; j < 2; j++)
        {
         padconv = new CNeuronMultiWindowsConvWPadOCL();
         if(!padconv ||
            !padconv.Init(0, index, OpenCL, windows, window_in, window_in,
                          units_in, variables, optimization, iBatch) ||
            !cDecoder[i].Add(padconv))
           {
            DeleteObj(padconv)
            ReturnFalse;
           }
         padconv.SetActivationFunction(SoftPlus);
         index++;
         norm = new CNeuronBatchNormOCL();
         if(!norm ||
            !norm.Init(0, index, OpenCL, padconv.Neurons(), iBatch, optimization) ||
            !cDecoder[i].Add(norm))
           {
            DeleteObj(norm)
            ReturnFalse;
           }
         norm.SetActivationFunction(None);
         index++;
        }
      conv = new CNeuronSpikeConvBlock();
      if(!conv ||
         !conv.Init(0, index, OpenCL, window_in, window_in, window_in, units_in,
                    variables, optimization, iBatch) ||
         !cDecoder[i].Add(conv))
        {
         DeleteObj(conv)
         ReturnFalse;
        }
      index++;
      if(i > 0)
        {
         cdc = new CNeuronSpikeCDC();
         if(!cdc ||
            !cdc.Init(0, index, OpenCL, window_in, inside_dimensions, window_in, units_in, variables, optimization, iBatch) ||
            !cDecoder[i].Add(cdc))
           {
            DeleteObj(cdc)
            ReturnFalse;
           }
        }
      index++;
      //---
      window_in = window_out;
      window_out = 2 * window_in;
      units_in = units_out;
      units_out = (units_in + 1) / 2;
      windows[0] = 3 * window_in;
     }
//---
   cDecoder[3].Clear();
   cDecoder[3].SetOpenCL(OpenCL);
   CNeuronBaseOCL* neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, window_in * units_in * variables, optimization, iBatch) ||
      !cDecoder[3].Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   neuron.Clear();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   CNeuronSpikeConvBlock* conv = NULL;
//---
   for(int l = 0; l < cPrepare.Total(); l++)
     {
      curr = cPrepare[l];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   for(uint i = 0; i < cEncoder.Size(); i++)
     {
      for(int l = 0; l < cEncoder[i].Total(); l++)
        {
         curr = cEncoder[i][l];
         if(!curr ||
            !curr.FeedForward(prev))
            ReturnFalse;
         prev = curr;
        }
      if(prev.Type() != defNeuronSpikeConvBlock)
         ReturnFalse;
      conv = prev;
      if(!DilatedCorrelation(prev, cCorrelations[i].AsObject(), ibShifts,
                             conv.GetFilters(), conv.GetUnits()*conv.GetVariables()))
         ReturnFalse;
     }
//---
   CNeuronMultiWindowsConvWPadOCL* padconv = NULL;
   for(int i = int(cDecoder.Size()) - 2; i >= 0; i--)
     {
      if(!cDecoder[i][0] ||
         cDecoder[i][0].Type() != defNeuronMultiWindowsConvWPadOCL)
         ReturnFalse;
      if(!cEncoder[i][-1] ||
         !cDecoder[i + 1][-1])
         ReturnFalse;
      //---
      padconv = cDecoder[i][0];
      uint units = padconv.GetUnits() * padconv.GetVariables();
      uint window1 = cEncoder[i][-1].Neurons() / units;
      uint window2 = cDecoder[i][-1].Neurons() / units;
      uint window3 = cCorrelations[i].Neurons() / units;
      if(!Concat(cEncoder[i][-1].getOutput(), cDecoder[i + 1][-1].getOutput(),
                 cCorrelations[i].getOutput(), cConcat[i].getOutput(),
                 window1, window2, window3, units))
         ReturnFalse;
      prev = cConcat[i].AsObject();
      for(int l = 0; l < cDecoder[i].Total(); l++)
        {
         curr = cDecoder[i][l];
         if(!curr)
            ReturnFalse;
         if(curr.Type() == defNeuronSpikeCDC)
            if(!SumAndNormalize(cEncoder[i - 1][-1].getOutput(), prev.getOutput(), prev.getOutput(),
                                ((CNeuronSpikeCDC*)curr).GetWindow(), false, 0, 0, 0, 1))
               ReturnFalse;
         if(!curr.FeedForward(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//---
   if(!SumAndNormalize(cPrepare[-1].getOutput(), prev.getOutput(), prev.getOutput(),
                       GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronSpikeCDC::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeCDC::calcInputGradients(cDecoder[0][-1]))
      ReturnFalse;
//--- Decoder
   CNeuronBaseOCL* next = cDecoder[0][-1];
   CNeuronBaseOCL* curr = NULL;
   CNeuronMultiWindowsConvWPadOCL* padconv = NULL;
   for(uint i = 0; i < cDecoder.Size() - 1; i++)
     {
      for(int l = cDecoder[i].Total() - 2; l >= 0; l--)
        {
         curr = cDecoder[i][l];
         if(!curr ||
            !curr.CalcHiddenGradients(next))
            ReturnFalse;
         next = curr;
        }
      if(!cConcat[i].CalcHiddenGradients(next))
         ReturnFalse;
      if(next.Type() != defNeuronMultiWindowsConvWPadOCL)
         ReturnFalse;
      if(!cEncoder[i][-1] ||
         !cDecoder[i + 1][-1])
         ReturnFalse;
      padconv = next;
      next = cDecoder[i + 1][-1];
      curr = cEncoder[i][-1];
      uint units = padconv.GetUnits() * padconv.GetVariables();
      uint window1 = curr.Neurons() / units;
      uint window2 = next.Neurons() / units;
      uint window3 = cCorrelations[i].Neurons() / units;
      if(!DeConcat(cCorrelations[i].getPrevOutput(), next.getGradient(),
                   cCorrelations[i].getGradient(), cConcat[i].getGradient(),
                   window1, window2, window3, units))
         ReturnFalse;
      Deactivation(next);
      if(!DilatedCorrelationGrad(curr, cCorrelations[i].AsObject(), ibShifts, window1, units))
         ReturnFalse;
      if(i == cEncoder.Size() - 1)
        {
         if(!SumAndNormalize(curr.getGradient(), cCorrelations[i].getPrevOutput(),
                             curr.getGradient(), window1, false, 0, 0, 0, 1))
            ReturnFalse;
         Deactivation(curr)
        }
      else
        {
         if(!SumAndNormalize(curr.getGradient(), cCorrelations[i].getPrevOutput(),
                             cCorrelations[i].getPrevOutput(), window1, false, 0, 0, 0, 1))
            ReturnFalse;
         if(curr.Activation() != None)
            if(!DeActivation(curr.getOutput(), cCorrelations[i].getPrevOutput(),
                             cCorrelations[i].getPrevOutput(), curr.Activation()))
               ReturnFalse;
        }
     }
//--- Encoder
   for(int i = int(cEncoder.Size()) - 1; i >= 0; i--)
     {
      next = cEncoder[i][-1];
      for(int l = cEncoder[i].Total() - 2; l >= 0; l--)
        {
         curr = cEncoder[i][l];
         if(!curr ||
            !curr.CalcHiddenGradients(next))
            ReturnFalse;
         next = curr;
        }
      if(i == 0)
         continue;
      curr = cEncoder[i - 1][-1];
      if(!curr ||
         !SumAndNormalize(curr.getGradient(), cCorrelations[i - 1].getPrevOutput(), curr.getGradient(),
                          ((CNeuronSpikeConvBlock*)curr).GetFilters(), false, 0, 0, 0, 1))
         ReturnFalse;
     }
//--- Prepare
   for(int l = cPrepare.Total() - 1; l >= 0; l--)
     {
      curr = cPrepare[l];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int l = 0; l < cPrepare.Total(); l++)
     {
      curr = cPrepare[l];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   for(uint i = 0; i < cEncoder.Size(); i++)
     {
      for(int l = 0; l < cEncoder[i].Total(); l++)
        {
         curr = cEncoder[i][l];
         if(!curr ||
            !curr.UpdateInputWeights(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//---
   for(int i = int(cDecoder.Size()) - 2; i >= 0; i--)
     {
      prev = cConcat[i].AsObject();
      for(int l = 0; l < cDecoder[i].Total(); l++)
        {
         curr = cDecoder[i][l];
         if(!curr ||
            !curr.UpdateInputWeights(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//---
   if(!CNeuronSpikeCDC::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeCDC::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeEEMFLow* Source = source;
   for(uint i = 0; i < cEncoder.Size(); i++)
      if(!cEncoder[i].WeightsUpdate(Source.cEncoder[i].AsObject(), tau))
         ReturnFalse;
   for(uint i = 0; i < cDecoder.Size(); i++)
      if(!cDecoder[i].WeightsUpdate(Source.cDecoder[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::Save(const int file_handle)
  {
   if(!CNeuronSpikeCDC::Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cEncoder.Size(); i++)
      if(!cEncoder[i].Save(file_handle))
         ReturnFalse;
   for(uint i = 0; i < cDecoder.Size(); i++)
      if(!cDecoder[i].Save(file_handle))
         ReturnFalse;
   if(FileWriteInteger(file_handle, iMaxDisplacement, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::Load(const int file_handle)
  {
   if(!CNeuronSpikeCDC::Load(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Load(file_handle))
      ReturnFalse;
   for(uint i = 0; i < cEncoder.Size(); i++)
      if(!cEncoder[i].Load(file_handle))
         ReturnFalse;
   for(uint i = 0; i < cDecoder.Size(); i++)
      if(!cDecoder[i].Load(file_handle))
         ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iMaxDisplacement = (uint)FileReadInteger(file_handle);
//---
   OpenCL.BufferFree(ibShifts);
//---
   int shifts[];
   if(ArrayResize(shifts, iMaxDisplacement) <= 0)
      ReturnFalse;
   uint count = 0;
   for(uint i = 0; i < iMaxDisplacement; i++)
     {
      if(i > 2 && bool(i & 1))
         continue;
      shifts[count] = (int)(i + 1);
      count++;
     }
   if(ArrayResize(shifts, count) < (int)count)
      ReturnFalse;
   ibShifts = OpenCL.AddBufferFromArray(shifts, 0, count, CL_MEM_READ_ONLY);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
//---
   int index = 0;
   CNeuronSpikeConvBlock* conv = NULL;
   for(uint i = 0; i < cEncoder.Size(); i++)
     {
      index += cEncoder[i].Total();
      conv = cEncoder[i][-1];
      uint units = conv.GetUnits();
      uint variables = conv.GetVariables();
      uint window_out = conv.GetFilters();
      //---
      if(!cCorrelations[i].Init(0, index, OpenCL, 2 * count * units * variables,
                                optimization, iBatch))
         ReturnFalse;
      cCorrelations[i].SetActivationFunction(None);
      index++;
      if(!cConcat[i].Init(0, index, OpenCL, 2 * (window_out + count) * units * variables,
                          optimization, iBatch))
         ReturnFalse;
      cConcat[i].SetActivationFunction(None);
      index += cDecoder[i].Total() + 1;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeEEMFLow::Clear(void)
  {
   if(!CNeuronSpikeCDC::Clear())
      ReturnFalse;
//---
   if(!cPrepare.ClearStates())
      ReturnFalse;
   for(uint i = 0; i < cEncoder.Size(); i++)
      if(!cEncoder[i].ClearStates())
         ReturnFalse;
   for(uint i = 0; i < cDecoder.Size(); i++)
      if(!cDecoder[i].ClearStates())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeEEMFLow::SetOpenCL(COpenCLMy *obj)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibShifts);
//---
   CNeuronSpikeCDC::SetOpenCL(obj);
//---
   cPrepare.SetOpenCL(OpenCL);
   for(uint i = 0; i < cEncoder.Size(); i++)
      cEncoder[i].SetOpenCL(OpenCL);
   for(uint i = 0; i < cDecoder.Size(); i++)
      cDecoder[i].SetOpenCL(OpenCL);
//---
   int shifts[];
   ArrayResize(shifts, iMaxDisplacement);
   uint count = 0;
   for(uint i = 0; i < iMaxDisplacement; i++)
     {
      if(i > 2 && bool(i & 1))
         continue;
      shifts[count] = (int)(i + 1);
      count++;
     }
   ArrayResize(shifts, count);
   ibShifts = OpenCL.AddBufferFromArray(shifts, 0, count, CL_MEM_READ_ONLY);
//---
   for(uint i = 0; i < cEncoder.Size(); i++)
     {
      cCorrelations[i].SetOpenCL(OpenCL);
      cConcat[i].SetOpenCL(OpenCL);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeEEMFLow::TrainMode(bool flag)
  {
   CNeuronSpikeCDC::TrainMode(flag);
//---
   cPrepare.TrainMode(bTrain);
   for(uint i = 0; i < cEncoder.Size(); i++)
      cEncoder[i].TrainMode(bTrain);
   for(uint i = 0; i < cDecoder.Size(); i++)
      cDecoder[i].TrainMode(bTrain);
   for(uint i = 0; i < cEncoder.Size(); i++)
     {
      cCorrelations[i].TrainMode(bTrain);
      cConcat[i].TrainMode(bTrain);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronSpikeEEMFLow::getWeights(void)
  {
   CBufferFloat* result = NULL;
   for(int layerNum = 0; (layerNum < cPrepare.Total() && !result && !IsStopped()); layerNum++)
     {
      CNeuronBaseOCL *temp = cPrepare[layerNum];
      CNeuronConvOCL *conv = NULL;
      CNeuronConcatenate *conc = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            result = conv.GetWeightsConv();
            break;
         case defNeuronConcatenate:
            conc = temp;
            result = conc.getConcatWeights();
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            result = batch.getBatchOptions();
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            result = lstm.getLSTMWeights();
            break;
         default:
            result = temp.getWeights();
            break;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                     uint dimension, uint units,
                                     uint bottleneck, uint max_displacement,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, 2 * bottleneck, 2 * bottleneck,
                                   dimension, units, 1, optimization_type, batch))
      ReturnFalse;
//---
   int count = 0;
   ibShifts = CreateShifts(max_displacement, count);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
   iMaxDisplacement = max_displacement;
//---
   uint index = 0;
   if(!cCorrelation.Init(0, index, OpenCL, units * count, optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
   index++;
   if(!cWarpCorrs.Init(0, index, OpenCL, count, count, bottleneck, units, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cWarpInps.Init(0, index, OpenCL, dimension, dimension, bottleneck, units, 1, optimization, iBatch))
      ReturnFalse;
   index++;
//---
   cProjection.Clear();
   cProjection.SetOpenCL(OpenCL);
   CNeuronBaseOCL*   neuron = NULL;
   CNeuronMSRes*     res_block = NULL;
//---
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, cWarpCorrs.Neurons() + cWarpInps.Neurons(), optimization, iBatch) ||
      !cProjection.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
   res_block = new CNeuronMSRes();
   if(!res_block ||
      !res_block.Init(0, index, OpenCL, units, 2 * bottleneck, 1, optimization, iBatch) ||
      !cProjection.Add(res_block))
     {
      DeleteObj(res_block)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cWarpInps.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!DilatedCorrelation(NeuronOCL, cCorrelation.AsObject(), ibShifts,
                          cWarpInps.GetWindow(), cWarpInps.GetUnits()))
      ReturnFalse;
   if(!cWarpCorrs.FeedForward(cCorrelation.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = cProjection[0];
   CNeuronBaseOCL* prev = NULL;
//---
   if(!curr ||
      !Concat(cWarpInps.getOutput(), cWarpCorrs.getOutput(), curr.getOutput(),
              cWarpInps.GetFilters(), cWarpCorrs.GetFilters(), cWarpInps.GetUnits()))
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
bool CNeuronCorrelationEncoder::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cProjection[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cProjection[-1];
   CNeuronBaseOCL* curr = NULL;
   for(int i = cProjection.Total() - 2; i >= 0; i--)
     {
      curr = cProjection[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   if(!DeConcat(cWarpInps.getGradient(), cWarpCorrs.getGradient(), next.getGradient(),
                cWarpInps.GetFilters(), cWarpCorrs.GetFilters(), cWarpInps.GetUnits()))
      ReturnFalse;
   if(!cCorrelation.CalcHiddenGradients(cWarpCorrs.AsObject()))
      ReturnFalse;
   if(!DilatedCorrelationGrad(NeuronOCL, cCorrelation.AsObject(), ibShifts,
                              cWarpInps.GetWindow(), cWarpInps.GetUnits()))
      ReturnFalse;
//---
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(PrevOutput, false) ||
      !NeuronOCL.CalcHiddenGradients(cWarpInps.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, cWarpInps.GetWindow(), false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cWarpInps.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cWarpCorrs.UpdateInputWeights(cCorrelation.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = cProjection[0];
   CNeuronBaseOCL* prev = NULL;
//---
   prev = curr;
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
bool CNeuronCorrelationEncoder::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, iMaxDisplacement, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   if(!cWarpCorrs.Save(file_handle))
      ReturnFalse;
   if(!cWarpInps.Save(file_handle))
      ReturnFalse;
//---
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::Load(const int file_handle)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibShifts);
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iMaxDisplacement = FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cWarpCorrs.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cWarpInps.AsObject()))
      ReturnFalse;
//---
   if(!cProjection.Load(file_handle))
      ReturnFalse;
//---
   int count = 0;
   ibShifts = CreateShifts(iMaxDisplacement, count);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
//---
   uint index = 0;
   if(!cCorrelation.Init(0, index, OpenCL, cWarpCorrs.GetUnits() * count, optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronCorrelationEncoder* Source = source;
//---
   if(!cWarpCorrs.WeightsUpdate(Source.cWarpCorrs.AsObject(), tau))
      ReturnFalse;
   if(!cWarpInps.WeightsUpdate(Source.cWarpInps.AsObject(), tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCorrelationEncoder::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cWarpCorrs.Clear())
      ReturnFalse;
   if(!cWarpInps.Clear())
      ReturnFalse;
   if(!cProjection.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCorrelationEncoder::SetOpenCL(COpenCLMy *obj)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibShifts);
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cWarpCorrs.SetOpenCL(OpenCL);
   cWarpInps.SetOpenCL(OpenCL);
   cProjection.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
//---
   int count = 0;
   ibShifts = CreateShifts(iMaxDisplacement, count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCorrelationEncoder::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cWarpCorrs.TrainMode(bTrain);
   cWarpInps.TrainMode(bTrain);
   cProjection.TrainMode(bTrain);
   cCorrelation.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_in, uint units_in, uint bottleneck,
                          uint dimension_out, uint units_out,
                          uint &shifts[], uint max_displacement,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, bottleneck, bottleneck,
                                   dimension_out, units_out, 1, optimization_type, batch))
      ReturnFalse;
//---
   for(uint i = 0; i < cFeatureEncoder.Size(); i++)
     {
      cFeatureEncoder[i].Clear();
      cFeatureEncoder[i].SetOpenCL(OpenCL);
     }
   cPrepare.Clear();
   cConextEncoder.Clear();
   cCorrelationUpsampling.Clear();
   cFlow.Clear();
   cPrepare.SetOpenCL(OpenCL);
   cConextEncoder.SetOpenCL(OpenCL);
   cCorrelationUpsampling.SetOpenCL(OpenCL);
   cFlow.SetOpenCL(OpenCL);
//---
   CNeuronBaseOCL*               neuron = NULL;
   CNeuronBatchNormOCL*          norm = NULL;
   CNeuronCreateFlow*            stack = NULL;
   CNeuronSpikeADM*              adm = NULL;
   CNeuronSpikeDepthWiseConv*    dw_conv = NULL;
   CNeuronSpikeConvBlock*        conv = NULL;
   CNeuronTransposeOCL*          transp = NULL;
   CNeuronMultiScaleExcitation*  excitation = NULL;
   CNeuronSpikeConvGRU2D*        gru = NULL;
   uint index = 0;
//--- Prepare
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, dimension_in, iBatch, optimization) ||
      !cPrepare.Add(norm))
     {
      DeleteObj(norm)
      ReturnFalse;
     }
   index++;
   stack = new CNeuronCreateFlow();
   if(!stack ||
      !stack.Init(0, index, OpenCL, units_in, dimension_in, dimension_in, 1, optimization, iBatch) ||
      !cPrepare.Add(stack))
     {
      DeleteObj(stack)
      ReturnFalse;
     }
   index++;
   adm = new CNeuronSpikeADM();
   if(!adm ||
      !adm.Init(0, index, OpenCL, dimension_in, units_in, 1, optimization, iBatch) ||
      !cPrepare.Add(adm))
     {
      DeleteObj(adm)
      ReturnFalse;
     }
   index++;
//--- Feature Encoder HD
   uint units = (units_in + 1) / 2;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, dimension_in, dimension_in, 5, 2, units, 1, optimization, iBatch) ||
      !cFeatureEncoder[0].Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
   units = (units + 1) / 2;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, dimension_in, bottleneck, 5, 2, units, 1, optimization, iBatch) ||
      !cFeatureEncoder[0].Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
   if(!cDiffEncoder.Init(0, index, OpenCL, bottleneck, units, bottleneck, shifts, optimization, iBatch))
      ReturnFalse;
   index++;
//--- Feature Encoder LD
   units = (units + 1) / 2;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, bottleneck, 2 * bottleneck, 5, 2, units, 1, optimization, iBatch) ||
      !cFeatureEncoder[1].Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
   if(!cCorrelation.Init(0, index, OpenCL, 2 * bottleneck, units, bottleneck, max_displacement, optimization, iBatch))
      ReturnFalse;
   index++;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, 2 * bottleneck, 2 * bottleneck, 5, 1, units, 1, optimization, iBatch) ||
      !cCorrelationUpsampling.Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
//--- Conext Encoder
   units = (units_in + 1) / 2;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, dimension_in, dimension_in, 10, 2, units, 1, optimization, iBatch) ||
      !cConextEncoder.Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
   units = (units + 1) / 2;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, dimension_in, bottleneck, 10, 2, units, 1, optimization, iBatch) ||
      !cConextEncoder.Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
//--- Flow
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 4 * bottleneck, optimization, iBatch) ||
      !cFlow.Add(neuron))
     {
      DeleteObj(neuron)
      ReturnFalse;
     }
   neuron.SetActivationFunction(None);
   index++;
   excitation = new CNeuronMultiScaleExcitation();
   if(!excitation ||
      !excitation.Init(0, index, OpenCL, bottleneck, units, 4, (bottleneck + 1) / 2, optimization, iBatch) ||
      !cFlow.Add(excitation))
     {
      DeleteObj(excitation)
      ReturnFalse;
     }
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 4 * bottleneck, 4 * bottleneck, bottleneck, units, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, bottleneck, bottleneck, 3, 1, units, 1, optimization, iBatch) ||
      !cFlow.Add(dw_conv))
     {
      DeleteObj(dw_conv)
      ReturnFalse;
     }
   index++;
   gru = new CNeuronSpikeConvGRU2D();
   if(!gru ||
      !gru.Init(0, index, OpenCL, units, bottleneck, bottleneck, optimization, iBatch) ||
      !cFlow.Add(gru))
     {
      DeleteObj(gru)
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units, bottleneck, optimization, iBatch) ||
      !cFlow.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
   index++;
   conv = CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units, units, units_out, bottleneck, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
     {
      DeleteObj(conv)
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, bottleneck, units_out, optimization, iBatch) ||
      !cFlow.Add(transp))
     {
      DeleteObj(transp)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Prepare
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
     {
      for(int i = 0; i < cFeatureEncoder[fe].Total(); i++)
        {
         curr = cFeatureEncoder[fe][i];
         if(!curr ||
            !curr.FeedForward(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//--- Context Encoder
   prev = cPrepare[-1];
   for(int i = 0; i < cConextEncoder.Total(); i++)
     {
      curr = cConextEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Moution Encoder
   if(!cDiffEncoder.FeedForward(cFeatureEncoder[0][-1].AsObject()))
      ReturnFalse;
   if(!cCorrelation.FeedForward(cFeatureEncoder[1][-1].AsObject()))
      ReturnFalse;
   prev = cCorrelation.AsObject();
   for(int i = 0; i < cCorrelationUpsampling.Total(); i++)
     {
      curr = cCorrelationUpsampling[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   curr = cFlow[0];
   uint units = cDiffEncoder.GetUnits();
   uint bottleneck = cDiffEncoder.GetFilters();
   if(!curr ||
      !Concat(cFeatureEncoder[0][-1].getOutput(), cConextEncoder[-1].getOutput(),
              cDiffEncoder.getOutput(), prev.getOutput(), curr.getOutput(),
              bottleneck, bottleneck, bottleneck, bottleneck, units))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//--- Flow
   CNeuronBaseOCL* next = cFlow[-1];
   CNeuronBaseOCL* curr = NULL;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//---
   uint units = cDiffEncoder.GetUnits();
   uint bottleneck = cDiffEncoder.GetFilters();
   if(!cConextEncoder[-1] ||
      !cCorrelationUpsampling[-1])
      ReturnFalse;
   if(!DeConcat(cDiffEncoder.getPrevOutput(), cConextEncoder[-1].getGradient(),
                cDiffEncoder.getGradient(), cCorrelationUpsampling[-1].getGradient(),
                next.getGradient(), bottleneck, bottleneck, bottleneck, bottleneck, units))
      ReturnFalse;
   Deactivation(cConextEncoder[-1])
   Deactivation(cCorrelationUpsampling[-1])
//--- Correlation Encoder
   next = cCorrelationUpsampling[-1];
   for(int i = cCorrelationUpsampling.Total() - 2; i >= 0; i--)
     {
      curr = cCorrelationUpsampling[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!cCorrelation.CalcHiddenGradients(next))
      ReturnFalse;
   next = cCorrelation.AsObject();
//--- Feature Encoder LD
   for(int i = cFeatureEncoder[1].Total() - 1; i >= 0; i--)
     {
      curr = cFeatureEncoder[1][i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Feature Encoder HD
   curr = cFeatureEncoder[0][-1];
   if(!curr ||
      !curr.CalcHiddenGradients(next))
      ReturnFalse;
   if(curr.Activation() != None)
      if(!DeActivation(curr.getOutput(), cDiffEncoder.getPrevOutput(),
                       cDiffEncoder.getPrevOutput(), curr.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(curr.getGradient(), cDiffEncoder.getPrevOutput(),
                       cDiffEncoder.getPrevOutput(), bottleneck, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!curr.CalcHiddenGradients(cDiffEncoder.AsObject()) ||
      !SumAndNormalize(curr.getGradient(), cDiffEncoder.getPrevOutput(),
                       curr.getPrevOutput(), bottleneck, false, 0, 0, 0, 1))
      ReturnFalse;
   next = curr;
   for(int i = cFeatureEncoder[0].Total() - 2; i >= 0; i--)
     {
      curr = cFeatureEncoder[0][i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Context Encoder
   next = cConextEncoder[-1];
   for(int i = cConextEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cConextEncoder[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Prepare
   curr = cPrepare[-1];
   if(!curr ||
      !curr.CalcHiddenGradients(next))
      ReturnFalse;
   CBufferFloat* temp = curr.getGradient();
   if(!curr.SetGradient(curr.getPrevOutput(), false) ||
      !curr.CalcHiddenGradients(cFeatureEncoder[0][0]) ||
      !SumAndNormalize(temp, curr.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !curr.SetGradient(temp, false))
      ReturnFalse;
   next = curr;
   for(int i = cPrepare.Total() - 2; i >= 0; i--)
     {
      curr = cPrepare[i];
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
bool CNeuronEDCFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//--- Prepare
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
     {
      for(int i = 0; i < cFeatureEncoder[fe].Total(); i++)
        {
         curr = cFeatureEncoder[fe][i];
         if(!curr ||
            !curr.UpdateInputWeights(prev))
            ReturnFalse;
         prev = curr;
        }
     }
//--- Context Encoder
   prev = cPrepare[-1];
   for(int i = 0; i < cConextEncoder.Total(); i++)
     {
      curr = cConextEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Moution Encoder
   if(!cDiffEncoder.UpdateInputWeights(cFeatureEncoder[0][-1].AsObject()))
      ReturnFalse;
   if(!cCorrelation.UpdateInputWeights(cFeatureEncoder[1][-1].AsObject()))
      ReturnFalse;
   prev = cCorrelation.AsObject();
   for(int i = 0; i < cCorrelationUpsampling.Total(); i++)
     {
      curr = cCorrelationUpsampling[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   prev = cFlow[0];
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
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
bool CNeuronEDCFlow::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//--- Prepare
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      if(!cFeatureEncoder[fe].Save(file_handle))
         ReturnFalse;
//--- Context Encoder
   if(!cConextEncoder.Save(file_handle))
      ReturnFalse;
//--- Moution Encoder
   if(!cDiffEncoder.Save(file_handle))
      ReturnFalse;
   if(!cCorrelation.Save(file_handle))
      ReturnFalse;
   if(!cCorrelationUpsampling.Save(file_handle))
      ReturnFalse;
//--- Flow
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//--- Prepare
   if(!cPrepare.Load(file_handle))
      ReturnFalse;
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      if(!cFeatureEncoder[fe].Load(file_handle))
         ReturnFalse;
//--- Context Encoder
   if(!cConextEncoder.Load(file_handle))
      ReturnFalse;
//--- Moution Encoder
   if(!LoadInsideLayer(file_handle, cDiffEncoder.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCorrelation.AsObject()))
      ReturnFalse;
   if(!cCorrelationUpsampling.Load(file_handle))
      ReturnFalse;
//--- Flow
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronEDCFlow* Source = source;
//--- Prepare
   if(!cPrepare.WeightsUpdate(Source.cPrepare.AsObject(), tau))
      ReturnFalse;
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      if(!cFeatureEncoder[fe].WeightsUpdate(Source.cFeatureEncoder[fe].AsObject(), tau))
         ReturnFalse;
//--- Context Encoder
   if(!cConextEncoder.WeightsUpdate(Source.cConextEncoder.AsObject(), tau))
      ReturnFalse;
//--- Moution Encoder
   if(!cDiffEncoder.WeightsUpdate(Source.cDiffEncoder.AsObject(), tau))
      ReturnFalse;
   if(!cCorrelation.WeightsUpdate(Source.cCorrelation.AsObject(), tau))
      ReturnFalse;
   if(!cCorrelationUpsampling.WeightsUpdate(Source.cCorrelationUpsampling.AsObject(), tau))
      ReturnFalse;
//--- Flow
   if(!cFlow.WeightsUpdate(Source.cFlow.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEDCFlow::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//--- Prepare
   if(!cPrepare.ClearStates())
      ReturnFalse;
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      if(!cFeatureEncoder[fe].ClearStates())
         ReturnFalse;
//--- Context Encoder
   if(!cConextEncoder.ClearStates())
      ReturnFalse;
//--- Moution Encoder
   if(!cDiffEncoder.Clear())
      ReturnFalse;
   if(!cCorrelation.Clear())
      ReturnFalse;
   if(!cCorrelationUpsampling.ClearStates())
      ReturnFalse;
//--- Flow
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEDCFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//--- Prepare
   cPrepare.SetOpenCL(OpenCL);
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      cFeatureEncoder[fe].SetOpenCL(OpenCL);
//--- Context Encoder
   cConextEncoder.SetOpenCL(OpenCL);
//--- Moution Encoder
   cDiffEncoder.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
   cCorrelationUpsampling.SetOpenCL(OpenCL);
//--- Flow
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEDCFlow::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//--- Prepare
   cPrepare.TrainMode(bTrain);
//--- Feature Encoder
   for(uint fe = 0; fe < cFeatureEncoder.Size(); fe++)
      cFeatureEncoder[fe].TrainMode(bTrain);
//--- Context Encoder
   cConextEncoder.TrainMode(bTrain);
//--- Moution Encoder
   cDiffEncoder.TrainMode(bTrain);
   cCorrelation.TrainMode(bTrain);
   cCorrelationUpsampling.TrainMode(bTrain);
//--- Flow
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CNeuronEDCFlow::getWeights(void)
  {
   CBufferFloat* result = NULL;
   for(int layerNum = 0; (layerNum < cPrepare.Total() && !result && !IsStopped()); layerNum++)
     {
      CNeuronBaseOCL *temp = cPrepare[layerNum];
      CNeuronConvOCL *conv = NULL;
      CNeuronConcatenate *conc = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            result = conv.GetWeightsConv();
            break;
         case defNeuronConcatenate:
            conc = temp;
            result = conc.getConcatWeights();
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            result = batch.getBatchOptions();
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            result = lstm.getLSTMWeights();
            break;
         default:
            result = temp.getWeights();
            break;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint stack_size, uint dimension, uint patch_dimension, uint variables,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikePatchStak::Init(numOutputs, myIndex, open_cl, stack_size,
                                   2 * dimension, patch_dimension, variables, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cState.Init(0, index, OpenCL, dimension * variables, optimization, iBatch))
      ReturnFalse;
   cState.SetActivationFunction(None);
   index++;
   if(!cConcat.Init(0, index, OpenCL, 2 * dimension * variables, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!CalcFlow(NeuronOCL))
      ReturnFalse;
   uint dimension = cBlock.GetWindow() / 2;
   uint units = cBlock.GetUnits() * cBlock.GetVariables();
   if(!Concat(NeuronOCL.getOutput(), cState.getOutput(), cConcat.getOutput(),
              dimension, dimension, units))
      ReturnFalse;
   if(!Normalize(cConcat.getOutput(), dimension))
      ReturnFalse;
//---
   return CNeuronSpikePatchStak::feedForward(cConcat.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikePatchStak::calcInputGradients(cConcat.AsObject()))
      ReturnFalse;
   uint dimension = cBlock.GetWindow() / 2;
   uint units = cBlock.GetUnits() * cBlock.GetVariables();
   if(!DeConcat(NeuronOCL.getGradient(), cState.getGradient(), cConcat.getGradient(),
                dimension, dimension, units))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cState.getGradient(), NeuronOCL.getGradient(),
                       dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(NeuronOCL)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   return CNeuronSpikePatchStak::updateInputWeights(cConcat.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::Load(const int file_handle)
  {
   if(!CNeuronCreateFlow::Load(file_handle))
      ReturnFalse;
   if(!cConcat.Init(0, 0, OpenCL, 2 * cState.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronCreateICEFlow::Clear(void)
  {
   if(!CNeuronCreateFlow::Clear())
      ReturnFalse;
   if(!cConcat.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronCreateICEFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronCreateFlow::SetOpenCL(obj);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMFE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint units, uint &windows[], uint window_out,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint wsize = windows.Size();
   if(wsize < 2 || wsize % 2 != 0)
      ReturnFalse;
   uint modals = wsize / 2;
//---
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, 2 * window_out, 2 * window_out,
                                   window_out, modals * units, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   cEncoder.Clear();
   cEncoder.SetOpenCL(OpenCL);
   CNeuronMultiWindowsConvOCL*   mw_conv  =  NULL;
   CNeuronBatchNormOCL*          normal   =  NULL;
   CNeuronSpikeActivation*       activat  =  NULL;
//---
   mw_conv = new CNeuronMultiWindowsConvOCL();
   if(!mw_conv ||
      !mw_conv.Init(0, index, OpenCL, windows, 2 * window_out, units, 1, optimization, iBatch) ||
      !cEncoder.Add(mw_conv))
     {
      DeleteObj(mw_conv)
      ReturnFalse;
     }
   mw_conv.SetActivationFunction(None);
   index++;
   normal = new CNeuronBatchNormOCL();
   if(!normal ||
      !normal.Init(0, index, OpenCL, mw_conv.Neurons(), iBatch, optimization) ||
      !cEncoder.Add(normal))
     {
      DeleteObj(normal)
      ReturnFalse;
     }
   index++;
   activat = new CNeuronSpikeActivation();
   if(!activat ||
      !activat.Init(0, index, OpenCL, normal.Neurons(), optimization, iBatch) ||
      !cEncoder.Add(activat))
     {
      DeleteObj(activat)
      ReturnFalse;
     }
   index++;
//---
   uint temp[];
   if(ArrayResize(temp, wsize) < int(wsize))
      ReturnFalse;
   ArrayFill(temp, 0, wsize, 2 * window_out);
   mw_conv = new CNeuronMultiWindowsConvOCL();
   if(!mw_conv ||
      !mw_conv.Init(0, index, OpenCL, temp, window_out, units, 1, optimization, iBatch) ||
      !cEncoder.Add(mw_conv))
     {
      DeleteObj(mw_conv)
      ReturnFalse;
     }
   mw_conv.SetActivationFunction(None);
   index++;
   normal = new CNeuronBatchNormOCL();
   if(!normal ||
      !normal.Init(0, index, OpenCL, mw_conv.Neurons(), iBatch, optimization) ||
      !cEncoder.Add(normal))
     {
      DeleteObj(normal)
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMFE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMFE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cEncoder[-1]))
      ReturnFalse;
   CNeuronBaseOCL* next = cEncoder[-1];
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cEncoder[i];
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
bool CNeuronSpikeMFE::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
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
bool CNeuronSpikeMFE::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMFE::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMFE::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikeMFE* Source = source;
   if(!cEncoder.WeightsUpdate(Source.cEncoder.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMFE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
   cEncoder.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMFE::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
   cEncoder.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                        uint window, uint window_key, uint heads,
                                        uint units_count, uint window_k, uint units_k,
                                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMSRes::Init(numOutputs, myIndex, open_cl, units_count, window, 1, optimization_type, batch))
      ReturnFalse;
//---
   iHeads = heads;
   uint index = 0;
   if(!cQ_Embedding.Init(0, index, OpenCL, window, window, iHeads * window_key, units_count, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cKV_Embedding.Init(0, index, OpenCL, window_k, window_k, 2 * iHeads * window_key, units_k, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cContext.Init(0, index, OpenCL, cKV_Embedding.GetWindow()*cKV_Embedding.GetUnits()*cKV_Embedding.GetVariables(), optimization, iBatch))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   index++;
   ibScoreIndex = OpenCL.AddBuffer(sizeof(float) * cQ_Embedding.GetUnits() * cKV_Embedding.GetUnits() * iHeads, CL_MEM_READ_WRITE);
   if(ibScoreIndex == INVALID_HANDLE)
      ReturnFalse;
   if(!cMHAttentionOut.Init(0, index, OpenCL, cQ_Embedding.GetFilters()*cQ_Embedding.GetUnits()*cQ_Embedding.GetVariables(), optimization, iBatch))
      ReturnFalse;
   cMHAttentionOut.SetActivationFunction(None);
   index++;
   if(!cW0.Init(0, index, OpenCL, cQ_Embedding.GetFilters(), cQ_Embedding.GetFilters(), cQ_Embedding.GetWindow(), cQ_Embedding.GetUnits(), cQ_Embedding.GetVariables(), optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAttentionOut.Init(0, index, OpenCL, cW0.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context)
  {
   if(cContext.getOutput() != Context)
      if(!cContext.SetOutput(Context))
         ReturnFalse;
//---
   if(!cQ_Embedding.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cKV_Embedding.FeedForward(cContext.AsObject()))
      ReturnFalse;
   if(!Attention())
      ReturnFalse;
   if(!cW0.FeedForward(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), cW0.getOutput(), cAttentionOut.getOutput(), cW0.GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronMSRes::feedForward(cAttentionOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::Attention(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {cQ_Embedding.GetUnits(), cKV_Embedding.GetUnits(), iHeads};
   uint local_work_size[3] = {1, global_work_size[1], 1};
   ResetLastError();
   uint kernel = def_k_MH2AttentionOut;
   setBuffer(kernel, def_k_mh2ao_q, cQ_Embedding.getOutputIndex())
   setBuffer(kernel, def_k_mh2ao_kv, cKV_Embedding.getOutputIndex())
   setBuffer(kernel, def_k_mh2ao_score, ibScoreIndex)
   setBuffer(kernel, def_k_mh2ao_out, cMHAttentionOut.getOutputIndex())
   setArgument(kernel, def_k_mh2ao_dimension, (int)(cQ_Embedding.GetFilters() / iHeads))
   setArgument(kernel, def_k_mh2ao_heads_kv, (int)iHeads)
   setArgument(kernel, def_k_mh2ao_mask, 0)
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
bool CNeuronSpikeMHCrossAttention::AttentionGradients(void)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {cQ_Embedding.GetUnits(), cQ_Embedding.GetFilters() / iHeads, iHeads};
   ResetLastError();
   uint kernel = def_k_MH2AttentionInsideGradients;
   setBuffer(kernel, def_k_mh2aig_q, cQ_Embedding.getOutputIndex())
   setBuffer(kernel, def_k_mh2aig_qg, cQ_Embedding.getGradientIndex())
   setBuffer(kernel, def_k_mh2aig_kv, cKV_Embedding.getOutputIndex())
   setBuffer(kernel, def_k_mh2aig_kvg, cKV_Embedding.getGradientIndex())
   setBuffer(kernel, def_k_mh2aig_score, ibScoreIndex)
   setBuffer(kernel, def_k_mh2aig_outg, cMHAttentionOut.getGradientIndex())
   setArgument(kernel, def_k_mh2aig_kunits, cKV_Embedding.GetUnits())
   setArgument(kernel, def_k_mh2aig_heads_kv, (int)iHeads)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cQ_Embedding.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
   if(!prevLayer)
      ReturnFalse;
   if(cContext.getGradient() != SecondGradient)
      if(!cContext.SetGradient(SecondGradient))
         ReturnFalse;
   cContext.SetActivationFunction(SecondActivation);
//---
   if(!CNeuronMSRes::calcInputGradients(cAttentionOut.AsObject()))
      ReturnFalse;
   if(!DeActivation(cW0.getOutput(), cW0.getGradient(),
                    cAttentionOut.getGradient(), cW0.Activation()))
      ReturnFalse;
   if(!cMHAttentionOut.CalcHiddenGradients(cW0.AsObject()))
      ReturnFalse;
   if(!AttentionGradients())
      ReturnFalse;
   Deactivation(cQ_Embedding)
   Deactivation(cKV_Embedding)
//---
   if(!cContext.CalcHiddenGradients(cKV_Embedding.AsObject()))
      ReturnFalse;
   Deactivation(cContext)
//---
   if(!prevLayer.CalcHiddenGradients(cQ_Embedding.AsObject()))
      ReturnFalse;
   if(!DeActivation(prevLayer.getOutput(), cAttentionOut.getPrevOutput(),
                    cAttentionOut.getGradient(), prevLayer.Activation()))
      ReturnFalse;
   if(!SumAndNormalize(prevLayer.getGradient(), cAttentionOut.getPrevOutput(),
                       prevLayer.getGradient(), cQ_Embedding.GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context)
  {
   if(!cQ_Embedding.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cKV_Embedding.UpdateInputWeights(cContext.AsObject()))
      ReturnFalse;
   if(!cW0.UpdateInputWeights(cMHAttentionOut.AsObject()))
      ReturnFalse;
   if(!CNeuronMSRes::updateInputWeights(cAttentionOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::Save(const int file_handle)
  {
   if(!CNeuronMSRes::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iHeads), INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(!cQ_Embedding.Save(file_handle))
      ReturnFalse;
   if(!cKV_Embedding.Save(file_handle))
      ReturnFalse;
   if(!cW0.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::Load(const int file_handle)
  {
   if(!CNeuronMSRes::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iHeads = (uint)FileReadInteger(file_handle);
   if(!LoadInsideLayer(file_handle, cQ_Embedding.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cKV_Embedding.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cW0.AsObject()))
      ReturnFalse;
//---
   uint index = 2;
   if(!cContext.Init(0, index, OpenCL, cKV_Embedding.GetWindow()*cKV_Embedding.GetUnits()*cKV_Embedding.GetVariables(), optimization, iBatch))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   index++;
   ibScoreIndex = OpenCL.AddBuffer(sizeof(float) * cQ_Embedding.GetUnits() * cKV_Embedding.GetUnits() * iHeads, CL_MEM_READ_WRITE);
   if(ibScoreIndex == INVALID_HANDLE)
      ReturnFalse;
   if(!cMHAttentionOut.Init(0, index, OpenCL, cQ_Embedding.GetFilters()*cQ_Embedding.GetUnits()*cQ_Embedding.GetVariables(), optimization, iBatch))
      ReturnFalse;
   cMHAttentionOut.SetActivationFunction(None);
   index += 2;
   if(!cAttentionOut.Init(0, index, OpenCL, cW0.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMSRes::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeMHCrossAttention* Source = source;
   if(!cQ_Embedding.WeightsUpdate(Source.cQ_Embedding.AsObject(), tau))
      ReturnFalse;
   if(!cKV_Embedding.WeightsUpdate(Source.cKV_Embedding.AsObject(), tau))
      ReturnFalse;
   if(!cW0.WeightsUpdate(Source.cW0.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMHCrossAttention::Clear(void)
  {
   if(!CNeuronMSRes::Clear())
      ReturnFalse;
//---
   if(!cQ_Embedding.Clear())
      ReturnFalse;
   if(!cKV_Embedding.Clear())
      ReturnFalse;
   if(!cW0.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMHCrossAttention::SetOpenCL(COpenCLMy *obj)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibScoreIndex);
   CNeuronMSRes::SetOpenCL(obj);
   cQ_Embedding.SetOpenCL(OpenCL);
   cKV_Embedding.SetOpenCL(OpenCL);
   cW0.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cMHAttentionOut.SetOpenCL(OpenCL);
   cAttentionOut.SetOpenCL(OpenCL);
//---
   ibScoreIndex = OpenCL.AddBuffer(sizeof(float) * cQ_Embedding.GetUnits() * cKV_Embedding.GetUnits() * iHeads, CL_MEM_READ_WRITE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMHCrossAttention::TrainMode(bool flag)
  {
   CNeuronMSRes::TrainMode(flag);
   cQ_Embedding.TrainMode(bTrain);
   cKV_Embedding.TrainMode(bTrain);
   cW0.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cMHAttentionOut.TrainMode(bTrain);
   cAttentionOut.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint dimension_in, uint dimension_out,
                                 uint units_count, uint depth,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, dimension_out, dimension_out,
                                   dimension_out, units_count, 1, optimization_type, batch))
      ReturnFalse;
//--- Residual
   uint index = 0;
   if(!cResidual.Init(0, index, OpenCL, dimension_in, dimension_in, dimension_out, units_count,
                      1, optimization, iBatch))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//--- Main
   cMain.Clear();
   cMain.SetOpenCL(OpenCL);
   CNeuronSpikeConvBlock* conv = NULL;
   CNeuronSpikeDepthWiseConv* dw_conv = NULL;
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, dimension_in, dimension_in, dimension_out, units_count, 1, optimization, iBatch) ||
      !cMain.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   dw_conv = new CNeuronSpikeDepthWiseConv();
   if(!dw_conv ||
      !dw_conv.Init(0, index, OpenCL, dimension_out, dimension_out, depth, 1, units_count, 1, optimization, iBatch) ||
      !cMain.Add(dw_conv))
      DeleteObjAndFalse(dw_conv);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   if(!cResidual.FeedForward(prev))
      ReturnFalse;
   for(int i = 0; i < cMain.Total(); i++)
     {
      curr = cMain[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
   if(!SumAndNormalize(Output, cResidual.getOutput(), Output, GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cMain[-1];
   CNeuronBaseOCL* curr = NULL;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(next))
      ReturnFalse;
   if(!DeActivation(cResidual.getOutput(), cResidual.getGradient(), Gradient, cResidual.Activation()))
      ReturnFalse;
//---
   for(int i = cMain.Total() - 2; i >= 0; i--)
     {
      curr = cMain[i];
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
      !NeuronOCL.CalcHiddenGradients(cResidual.AsObject()) ||
      !SumAndNormalize(NeuronOCL.getGradient(), temp, temp, GetWindow(), false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   if(!cResidual.UpdateInputWeights(prev))
      ReturnFalse;
   for(int i = 0; i < cMain.Total(); i++)
     {
      curr = cMain[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeMixFusion* Source = source;
   if(!cResidual.WeightsUpdate(Source.cResidual.AsObject(), tau))
      ReturnFalse;
   if(!cMain.WeightsUpdate(Source.cMain.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cResidual.Save(file_handle))
      ReturnFalse;
   if(!cMain.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
   if(!cMain.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeMixFusion::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//---
   if(!cResidual.Clear())
      ReturnFalse;
   if(!cMain.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMixFusion::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//---
   cResidual.SetOpenCL(OpenCL);
   cMain.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeMixFusion::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//---
   cResidual.TrainMode(bTrain);
   cMain.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint dimension_in, uint units_in, uint bottleneck,
                         uint dimension_out, uint units_out, uint max_displacement,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeConvBlock::Init(numOutputs, myIndex, open_cl, bottleneck, bottleneck,
                                   dimension_out, units_out, 1, optimization_type, batch))
      ReturnFalse;
//--- Shifts
   int count = 0;
   ibShifts = CreateShifts(max_displacement, count);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
   iMaxDisplacement = max_displacement;
//--- Create Flow
   uint index = 0;
   if(!cNorm.Init(0, index, OpenCL, dimension_in, iBatch, optimization))
      ReturnFalse;
   SetActivationFunction(None);
   index++;
   if(!cICE.Init(0, index, OpenCL, units_in, dimension_in, bottleneck, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cEvents.Init(0, index, OpenCL, units_in, dimension_in, bottleneck, 1, optimization, iBatch))
      ReturnFalse;
   index++;
//---
   cEncoder.Clear();
   cEncoder.SetOpenCL(OpenCL);
   cFlow.Clear();
   cFlow.SetOpenCL(OpenCL);
//---
   CNeuronBaseOCL*                  neuron = NULL;
   CNeuronBatchNormOCL*             norm = NULL;
   CNeuronMultiWindowsConvWPadOCL*  mw_conv = NULL;
   CNeuronSpikeConvBlock*           conv = NULL;
   CNeuronTransposeOCL*             transp = NULL;
   CNeuronSpikeConvGRU2D*           gru = NULL;
//--- Feature Encoder
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 2 * bottleneck * units_in, optimization, iBatch) ||
      !cEncoder.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, bottleneck, bottleneck, bottleneck, 2 * units_in, 1, optimization, iBatch) ||
      !cEncoder.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, 2 * bottleneck, optimization, iBatch) ||
      !cEncoder.Add(transp))
      DeleteObjAndFalse(transp);
   index++;
   uint windows_depth[] = { iMaxDisplacement };
   for(int i = 0; i < 2; i++)
     {
      mw_conv = new CNeuronMultiWindowsConvWPadOCL();
      if(!mw_conv ||
         !mw_conv.Init(0, index, OpenCL, windows_depth, 1, 1, units_in, 2 * bottleneck, optimization, iBatch) ||
         !cEncoder.Add(mw_conv))
         DeleteObjAndFalse(mw_conv);
      mw_conv.SetActivationFunction(SoftPlus);
      index++;
     }
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, 2 * bottleneck, optimization, iBatch) ||
      !cEncoder.Add(transp))
      DeleteObjAndFalse(transp);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, bottleneck, bottleneck, bottleneck, 2 * units_in, 1, optimization, iBatch) ||
      !cEncoder.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, bottleneck, bottleneck, 2 * bottleneck, 2 * units_in, 1, optimization, iBatch) ||
      !cEncoder.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
//--- Moution Encoder
   if(!cICEEmbedding.Init(0, index, OpenCL, bottleneck * units_in, optimization, iBatch))
      ReturnFalse;
   cICEEmbedding.SetActivationFunction(None);
   index++;
   if(!cEventsEmbedding.Init(0, index, OpenCL, bottleneck * units_in, optimization, iBatch))
      ReturnFalse;
   cEventsEmbedding.SetActivationFunction(None);
   index++;
   if(!cContext.Init(0, index, OpenCL, 2 * bottleneck * units_in, optimization, iBatch))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   index++;
   if(!cICECorrelation.Init(0, index, OpenCL, count * units_in, optimization, iBatch))
      ReturnFalse;
   cICECorrelation.SetActivationFunction(None);
   index++;
   if(!cEventsCorrelation.Init(0, index, OpenCL, count * units_in, optimization, iBatch))
      ReturnFalse;
   cEventsCorrelation.SetActivationFunction(None);
   index++;
   if(!cFeatureVsCorrelation.Init(0, index, OpenCL, 2 * (bottleneck + count)*units_in, optimization, iBatch))
      ReturnFalse;
   cFeatureVsCorrelation.SetActivationFunction(None);
   index++;
   uint windows[] = { bottleneck, count, bottleneck, count };
   if(!cMoutionEncoder.Init(0, index, OpenCL, units_in, windows, bottleneck, optimization, iBatch))
      ReturnFalse;
   index++;
//--- Aggregation
   if(!cTranspose.Init(0, index, OpenCL, units_in, 2, bottleneck, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cCrossModalAggregation.Init(0, index, OpenCL, bottleneck, (bottleneck + 3) / 4, 4, 2 * units_in, bottleneck, units_in, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cMixFusion.Init(0, index, OpenCL, 2 * bottleneck, bottleneck, units_in, iMaxDisplacement, optimization, iBatch))
      ReturnFalse;
   index++;
//--- Flow
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 3 * bottleneck * units_in, optimization, iBatch) ||
      !cFlow.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 3 * bottleneck, 3 * bottleneck, bottleneck, units_in, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   gru = new CNeuronSpikeConvGRU2D();
   if(!gru ||
      !gru.Init(0, index, OpenCL, units_in, bottleneck, bottleneck, optimization, iBatch) ||
      !cFlow.Add(gru))
      DeleteObjAndFalse(gru);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, bottleneck, optimization, iBatch) ||
      !cFlow.Add(transp))
      DeleteObjAndFalse(transp);
   index++;
   conv = CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, units_out, bottleneck, 1, optimization, iBatch) ||
      !cFlow.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, bottleneck, units_out, optimization, iBatch) ||
      !cFlow.Add(transp))
      DeleteObjAndFalse(transp);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Create Flow
   if(!cNorm.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cICE.FeedForward(cNorm.AsObject()))
      ReturnFalse;
   if(!cEvents.FeedForward(cICE.GetState()))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cEncoder[0];
   CNeuronBaseOCL* curr = NULL;
   uint bottleneck = cICE.GetDimension();
   uint units = cICE.GetStackSize();
   uint correl = cICECorrelation.Neurons() / units;
//--- Feature Encoder
   if(!prev ||
      !Concat(cICE.getOutput(), cEvents.getOutput(), prev.getOutput(),
              bottleneck, bottleneck, units))
      ReturnFalse;
   for(int i = 1; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Moution Encoder
   if(!DeConcat(cICEEmbedding.getOutput(), cContext.getOutput(), cEventsEmbedding.getOutput(),
                prev.getOutput(), bottleneck, 2 * bottleneck, bottleneck, units))
      ReturnFalse;
   if(!DilatedCorrelation(cICEEmbedding.AsObject(), cICECorrelation.AsObject(), ibShifts, bottleneck, units))
      ReturnFalse;
   if(!DilatedCorrelation(cEventsEmbedding.AsObject(), cEventsCorrelation.AsObject(), ibShifts, bottleneck, units))
      ReturnFalse;
   if(!Concat(cICEEmbedding.getOutput(), cICECorrelation.getOutput(),
              cEventsEmbedding.getOutput(), cEventsCorrelation.getOutput(),
              cFeatureVsCorrelation.getOutput(),
              bottleneck, correl, bottleneck, correl, units))
      ReturnFalse;
   if(!cMoutionEncoder.FeedForward(cFeatureVsCorrelation.AsObject()))
      ReturnFalse;
//--- Aggregation
   if(!cTranspose.FeedForward(cMoutionEncoder.AsObject()))
      ReturnFalse;
   if(!cCrossModalAggregation.FeedForward(cMoutionEncoder.AsObject(), cTranspose.getOutput()))
      ReturnFalse;
   if(!cMixFusion.FeedForward(cContext.AsObject()))
      ReturnFalse;
//--- Flow
   prev = cFlow[0];
   if(!prev ||
      !Concat(cCrossModalAggregation.getOutput(), cMixFusion.getOutput(), prev.getOutput(),
              2 * bottleneck, bottleneck, units))
      ReturnFalse;
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeConvBlock::calcInputGradients(cFlow[-1]))
      ReturnFalse;
//--- Flow
   CNeuronBaseOCL* next = cFlow[-1];
   CNeuronBaseOCL*curr = NULL;
   for(int i = cFlow.Total() - 2; i >= 0; i--)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Aggregation
   uint bottleneck = cICE.GetDimension();
   uint units = cICE.GetStackSize();
   uint correl = cICECorrelation.Neurons() / units;
//---
   if(!DeConcat(cCrossModalAggregation.getGradient(), cMixFusion.getGradient(), next.getGradient(),
                2 * bottleneck, bottleneck, units))
      ReturnFalse;
   Deactivation(cCrossModalAggregation);
   Deactivation(cMixFusion);
   if(!cContext.CalcHiddenGradients(cMixFusion.AsObject()))
      ReturnFalse;
//--- Moution Encoder
   if(!cMoutionEncoder.CalcHiddenGradients(cCrossModalAggregation.AsObject(),
                                           cTranspose.getOutput(),
                                           cTranspose.getGradient(),
                                           (ENUM_ACTIVATION)cTranspose.Activation()))
      ReturnFalse;
   CBufferFloat* temp = cMoutionEncoder.getGradient();
   if(!cMoutionEncoder.SetGradient(cMoutionEncoder.getPrevOutput(), false) ||
      !cMoutionEncoder.CalcHiddenGradients(cTranspose.AsObject()) ||
      !SumAndNormalize(cMoutionEncoder.getGradient(), temp, temp, bottleneck, false, 0, 0, 0, 1) ||
      !cMoutionEncoder.SetGradient(temp, false))
      ReturnFalse;
   if(!cFeatureVsCorrelation.CalcHiddenGradients(cMoutionEncoder.AsObject()))
      ReturnFalse;
   if(!DeConcat(cICEEmbedding.getPrevOutput(), cICECorrelation.getGradient(),
                cEventsEmbedding.getPrevOutput(), cEventsCorrelation.getGradient(),
                cFeatureVsCorrelation.getGradient(),
                bottleneck, correl, bottleneck, correl, units))
      ReturnFalse;
   if(!DilatedCorrelationGrad(cICEEmbedding.AsObject(), cICECorrelation.AsObject(), ibShifts, bottleneck, units))
      ReturnFalse;
   if(!DilatedCorrelationGrad(cEventsEmbedding.AsObject(), cEventsCorrelation.AsObject(), ibShifts, bottleneck, units))
      ReturnFalse;
   if(!SumAndNormalize(cICEEmbedding.getGradient(), cICEEmbedding.getPrevOutput(),
                       cICEEmbedding.getGradient(), bottleneck, false, 0, 0, 0, 1) ||
      !SumAndNormalize(cEventsEmbedding.getGradient(), cEventsEmbedding.getPrevOutput(),
                       cEventsEmbedding.getGradient(), bottleneck, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(cICEEmbedding);
   Deactivation(cEventsEmbedding);
//--- Feature Encoder
   next = cEncoder[-1];
   if(!next ||
      !Concat(cICEEmbedding.getGradient(), cContext.getGradient(), cEventsEmbedding.getGradient(),
              next.getGradient(), bottleneck, 2 * bottleneck, bottleneck, units))
      ReturnFalse;
   Deactivation(next);
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
//--- Create Flow
   if(!DeConcat(cICE.getGradient(), cEvents.getGradient(), next.getGradient(),
                bottleneck, bottleneck, units))
      ReturnFalse;
   Deactivation(cICE);
   Deactivation(cEvents);
   curr = cICE.GetState();
   if(!curr ||
      !curr.CalcHiddenGradients(cEvents.AsObject()))
      ReturnFalse;
   if(!cNorm.CalcHiddenGradients(cICE.AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
//--- Create Flow
   if(!cNorm.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cICE.UpdateInputWeights(cNorm.AsObject()))
      ReturnFalse;
   if(!cEvents.UpdateInputWeights(cICE.GetState()))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cEncoder[0];
   CNeuronBaseOCL* curr = NULL;
//--- Feature Encoder
   for(int i = 1; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//--- Moution Encoder
   if(!cMoutionEncoder.UpdateInputWeights(cFeatureVsCorrelation.AsObject()))
      ReturnFalse;
//--- Aggregation
   if(!cCrossModalAggregation.UpdateInputWeights(cMoutionEncoder.AsObject(), cTranspose.getOutput()))
      ReturnFalse;
   if(!cMixFusion.UpdateInputWeights(cContext.AsObject()))
      ReturnFalse;
//--- Flow
   for(int i = 1; i < cFlow.Total(); i++)
     {
      curr = cFlow[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeConvBlock::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::Save(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iMaxDisplacement), INT_VALUE) < INT_VALUE)
      ReturnFalse;
//--- Create Flow
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(!cICE.Save(file_handle))
      ReturnFalse;
   if(!cEvents.Save(file_handle))
      ReturnFalse;
//--- Feature Encoder
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
//--- Moution Encoder
   if(!cMoutionEncoder.Save(file_handle))
      ReturnFalse;
//--- Aggregation
   if(!cTranspose.Save(file_handle))
      ReturnFalse;
   if(!cCrossModalAggregation.Save(file_handle))
      ReturnFalse;
   if(!cMixFusion.Save(file_handle))
      ReturnFalse;
//--- Flow
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikeConvBlock::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iMaxDisplacement = uint(FileReadInteger(file_handle, INT_VALUE));
//--- Shifts
   int count = 0;
   ibShifts = CreateShifts(iMaxDisplacement, count);
   if(ibShifts == INVALID_HANDLE)
      ReturnFalse;
//--- Create Flow
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cICE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cEvents.AsObject()))
      ReturnFalse;
//--- Feature Encoder
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
//--- Moution Encoder
   if(!LoadInsideLayer(file_handle, cMoutionEncoder.AsObject()))
      ReturnFalse;
//--- Aggregation
   if(!LoadInsideLayer(file_handle, cTranspose.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCrossModalAggregation.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cMixFusion.AsObject()))
      ReturnFalse;
//--- Flow
   if(!cFlow.Load(file_handle))
      ReturnFalse;
//---
   uint index = 3 + cEncoder.Total();
   uint bottleneck = cMoutionEncoder.GetFilters();
   uint units = cICE.GetStackSize();
   if(!cICEEmbedding.Init(0, index, OpenCL, bottleneck * units, optimization, iBatch))
      ReturnFalse;
   cICEEmbedding.SetActivationFunction(None);
   index++;
   if(!cEventsEmbedding.Init(0, index, OpenCL, bottleneck * units, optimization, iBatch))
      ReturnFalse;
   cEventsEmbedding.SetActivationFunction(None);
   index++;
   if(!cContext.Init(0, index, OpenCL, 2 * bottleneck * units, optimization, iBatch))
      ReturnFalse;
   cContext.SetActivationFunction(None);
   index++;
   if(!cICECorrelation.Init(0, index, OpenCL, count * units, optimization, iBatch))
      ReturnFalse;
   cICECorrelation.SetActivationFunction(None);
   index++;
   if(!cEventsCorrelation.Init(0, index, OpenCL, count * units, optimization, iBatch))
      ReturnFalse;
   cEventsCorrelation.SetActivationFunction(None);
   index++;
   if(!cFeatureVsCorrelation.Init(0, index, OpenCL, 2 * (bottleneck + count)*units, optimization, iBatch))
      ReturnFalse;
   cFeatureVsCorrelation.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeConvBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSTFlow* Source = source;
//--- Create Flow
   dWeightsUpdate(cNorm, Source, tau);
   dWeightsUpdate(cICE, Source, tau);
   dWeightsUpdate(cEvents, Source, tau);
//--- Feature Encoder
   dWeightsUpdate(cEncoder, Source, tau);
//--- Moution Encoder
   dWeightsUpdate(cMoutionEncoder, Source, tau);
//--- Aggregation
   dWeightsUpdate(cCrossModalAggregation, Source, tau);
   dWeightsUpdate(cMixFusion, Source, tau);
//--- Flow
   dWeightsUpdate(cFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSTFlow::Clear(void)
  {
   if(!CNeuronSpikeConvBlock::Clear())
      ReturnFalse;
//--- Create Flow
   if(!cNorm.Clear())
      ReturnFalse;
   if(!cICE.Clear())
      ReturnFalse;
   if(!cEvents.Clear())
      ReturnFalse;
//--- Feature Encoder
   if(!cEncoder.ClearStates())
      ReturnFalse;
//---
   if(!cMoutionEncoder.Clear())
      ReturnFalse;
   if(!cTranspose.Clear())
      ReturnFalse;
   if(!cCrossModalAggregation.Clear())
      ReturnFalse;
   if(!cMixFusion.Clear())
      ReturnFalse;
//--- Flow
   if(!cFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTFlow::SetOpenCL(COpenCLMy *obj)
  {
   if(!!OpenCL)
      OpenCL.BufferFree(ibShifts);
//---
   CNeuronSpikeConvBlock::SetOpenCL(obj);
//--- Create Flow
   cNorm.SetOpenCL(OpenCL);
   cICE.SetOpenCL(OpenCL);
   cEvents.SetOpenCL(OpenCL);
//--- Feature Encoder
   cEncoder.SetOpenCL(OpenCL);
//--- Moution Encoder
   cMoutionEncoder.SetOpenCL(OpenCL);
//--- Aggregation
   cTranspose.SetOpenCL(OpenCL);
   cCrossModalAggregation.SetOpenCL(OpenCL);
   cMixFusion.SetOpenCL(OpenCL);
//--- Flow
   cFlow.SetOpenCL(OpenCL);
//---
   cICEEmbedding.SetOpenCL(OpenCL);
   cEventsEmbedding.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cICECorrelation.SetOpenCL(OpenCL);
   cEventsCorrelation.SetOpenCL(OpenCL);
   cFeatureVsCorrelation.SetOpenCL(OpenCL);
   cFeatureVsCorrelation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSTFlow::TrainMode(bool flag)
  {
   CNeuronSpikeConvBlock::TrainMode(flag);
//--- Create Flow
   cNorm.TrainMode(bTrain);
   cICE.TrainMode(bTrain);
   cEvents.TrainMode(bTrain);
//--- Feature Encoder
   cEncoder.TrainMode(bTrain);
//--- Moution Encoder
   cMoutionEncoder.TrainMode(bTrain);
//--- Aggregation
   cTranspose.TrainMode(bTrain);
   cCrossModalAggregation.TrainMode(bTrain);
   cMixFusion.TrainMode(bTrain);
//--- Flow
   cFlow.TrainMode(bTrain);
//---
   cICEEmbedding.TrainMode(bTrain);
   cEventsEmbedding.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cICECorrelation.TrainMode(bTrain);
   cEventsCorrelation.TrainMode(bTrain);
   cFeatureVsCorrelation.TrainMode(bTrain);
   cFeatureVsCorrelation.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                       uint stack_size, uint dimension, uint patch_dimension,
                       uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronAddToStack::Init(numOutputs, myIndex, open_cl, stack_size,
                               patch_dimension, variables, optimization_type, batch))
      ReturnFalse;
//---
   cPrepare.Clear();
   cPrepare.SetOpenCL(OpenCL);
   CNeuronBatchNormOCL* norm = NULL;
   CNeuronSpikeQKAttention* att = NULL;
   CNeuronSpikeConvBlock* conv = NULL;
   uint index = 0;
   uint units = MathMax(dimension - 2, 1);
//---
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, dimension * variables, iBatch, optimization) ||
      !cPrepare.Add(norm))
      ReturnFalse;
   norm.SetActivationFunction(None);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, MathMin(dimension, 3), 1, patch_dimension, units, variables, optimization, iBatch) ||
      !cPrepare.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
   att = new CNeuronSpikeQKAttention();
   if(!att ||
      !att.Init(0, index, OpenCL, units * variables, patch_dimension, 4, (patch_dimension + 3) / 4, optimization, iBatch) ||
      !cPrepare.Add(att))
      DeleteObjAndFalse(att);
   index++;
   conv = new CNeuronSpikeConvBlock();
   if(!conv ||
      !conv.Init(0, index, OpenCL, patch_dimension * units, patch_dimension * units, patch_dimension, variables, 1, optimization, iBatch) ||
      !cPrepare.Add(conv))
      DeleteObjAndFalse(conv);
   index++;
//---
   if(!cA.Init(0, index, OpenCL, patch_dimension * (patch_dimension + 3) / 4 * variables, optimization, iBatch))
      ReturnFalse;
   cA.SetActivationFunction(None);
   CBufferFloat* temp = cA.getWeightsParams();
   if(!temp ||
      !temp.Fill(0))
      ReturnFalse;
   index++;
   if(!cBCDxE.Init(0, index, OpenCL, patch_dimension, patch_dimension, patch_dimension * (2 * patch_dimension + 1 + (patch_dimension + 3) / 4), variables, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cB.Init(0, index, OpenCL, patch_dimension * patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cB.SetActivationFunction(None);
   index++;
   if(!cC.Init(0, index, OpenCL, patch_dimension * patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cC.SetActivationFunction(None);
   index++;
   if(!cDx.Init(0, index, OpenCL, patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cDx.SetActivationFunction(None);
   index++;
   if(!cE.Init(0, index, OpenCL, cA.Neurons(), optimization, iBatch))
      ReturnFalse;
   cE.SetActivationFunction(None);
   index++;
   if(!cH.Init(0, index, OpenCL, patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cH.SetActivationFunction(None);
   if(!cH.Clear())
      ReturnFalse;
   index++;
//---
   if(!cAE.Init(0, index, OpenCL, cA.Neurons(), optimization, iBatch))
      ReturnFalse;
   cAE.SetActivationFunction(None);
   index++;
   if(!cAET.Init(0, index, OpenCL, variables, patch_dimension, (patch_dimension + 3) / 4, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAED.Init(0, index, OpenCL, patch_dimension * patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cAED.SetActivationFunction(None);
   index++;
//---
   if(!cAH.Init(0, index, OpenCL, patch_dimension * variables, optimization, iBatch))
      ReturnFalse;
   cAH.SetActivationFunction(None);
   if(!cBX.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cBX.SetActivationFunction(None);
   index++;
   if(!cHc.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cHc.SetActivationFunction(None);
   index++;
   if(!cCHc.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cCHc.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
     {
      if(!cA.FeedForward())
         ReturnFalse;
     }
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cBCDxE.FeedForward(prev))
      ReturnFalse;
   if(!DeConcat(cB.getOutput(), cC.getOutput(), cDx.getOutput(), cE.getOutput(), cBCDxE.getOutput(),
                cB.Neurons() / iVariables, cC.Neurons() / iVariables,
                cDx.Neurons() / iVariables, cE.Neurons() / iVariables, iVariables))
      ReturnFalse;
   if(!PerturbedMatrix(cA.AsObject(), cE.AsObject(), cAE.AsObject(), 0.01f, cE.Neurons() / iVariables, iVariables))
      ReturnFalse;
   uint AE_dimension = cAE.Neurons() / (iDimension * iVariables);
   if(!Normalize(cAE.getOutput(), AE_dimension))
      ReturnFalse;
   if(!cAET.FeedForward(cAE.AsObject()))
      ReturnFalse;
   if(!MatMul(cAE.getOutput(), cAET.getOutput(), cAED.getOutput(), iDimension, AE_dimension, iDimension, iVariables, true))
      ReturnFalse;
//---
   if(!cH.SwapOutputs())
      ReturnFalse;
   if(!MatMul(cAED.getOutput(), cH.getPrevOutput(), cAH.getOutput(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   if(!MatMul(cB.getOutput(), prev.getOutput(), cBX.getOutput(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   if(!SumAndNormalize(cAH.getOutput(), cBX.getOutput(), cHc.getOutput(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!MatMul(cC.getOutput(), cHc.getOutput(), cCHc.getOutput(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   if(!SumAndNormalize(cCHc.getOutput(), cDx.getOutput(), cH.getOutput(), iDimension, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronAddToStack::feedForward(cH.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronAddToStack::calcInputGradients(cH.AsObject()))
      ReturnFalse;
//---
   if(!DeActivation(cCHc.getOutput(), cCHc.getGradient(), cH.getGradient(), cCHc.Activation()))
      ReturnFalse;
   if(!DeActivation(cDx.getOutput(), cDx.getGradient(), cH.getGradient(), cDx.Activation()))
      ReturnFalse;
   if(!MatMulGrad(cC.getOutput(), cC.getGradient(), cHc.getOutput(), cHc.getGradient(),
                  cCHc.getGradient(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   Deactivation(cC);
   Deactivation(cHc);
//---
   if(!DeActivation(cAH.getOutput(), cAH.getGradient(), cHc.getGradient(), cAH.Activation()))
      ReturnFalse;
   if(!DeActivation(cBX.getOutput(), cBX.getGradient(), cHc.getGradient(), cBX.Activation()))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cPrepare[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next ||
      !MatMulGrad(cB.getOutput(), cB.getGradient(), next.getOutput(), cBX.getPrevOutput(),
                  cBX.getGradient(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   if(!MatMulGrad(cAED.getOutput(), cAED.getGradient(), cH.getPrevOutput(), cAH.getPrevOutput(),
                  cAH.getGradient(), iDimension, iDimension, 1, iVariables, true))
      ReturnFalse;
   uint AE_dimension = cAE.Neurons() / (iDimension * iVariables);
   if(!MatMulGrad(cAE.getOutput(), cAE.getPrevOutput(), cAET.getOutput(), cAET.getGradient(),
                  cAED.getGradient(), iDimension, AE_dimension, iDimension, iVariables, true))
      ReturnFalse;
   if(!cAE.CalcHiddenGradients(cAET.AsObject()) ||
      !SumAndNormalize(cAE.getGradient(), cAE.getPrevOutput(), cAE.getGradient(), AE_dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!PerturbedMatrixGrad(cA.AsObject(), cE.AsObject(), cAE.AsObject(), 0.01f, cE.Neurons() / iVariables, iVariables))
      ReturnFalse;
   Deactivation(cA);
   Deactivation(cE);
   if(!Concat(cB.getGradient(), cC.getGradient(), cDx.getGradient(), cE.getGradient(), cBCDxE.getGradient(),
              cB.Neurons() / iVariables, cC.Neurons() / iVariables,
              cDx.Neurons() / iVariables, cE.Neurons() / iVariables, iVariables))
      ReturnFalse;
   if(!next.CalcHiddenGradients(cBCDxE.AsObject()))
      ReturnFalse;
   if(!next.Activation() != None)
      if(!DeActivation(next.getOutput(), next.getPrevOutput(), next.getPrevOutput(), next.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(next.getGradient(), next.getPrevOutput(), next.getGradient(), iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = cPrepare.Total() - 2; i >= 0; i--)
     {
      curr = cPrepare[i];
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
bool CNeuronPSSE::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cA.UpdateInputWeights())
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cBCDxE.UpdateInputWeights(prev))
      ReturnFalse;
   if(!CNeuronAddToStack::updateInputWeights(cH.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronAddToStack::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronPSSE* Source = source;
   dWeightsUpdate(cA, Source, tau);
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cBCDxE, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::Save(const int file_handle)
  {
   if(!CNeuronAddToStack::Save(file_handle))
      ReturnFalse;
   if(!cA.Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cBCDxE.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::Load(const int file_handle)
  {
   if(!CNeuronAddToStack::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!cPrepare.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cBCDxE.AsObject()))
      ReturnFalse;
//---
   uint index = cPrepare.Total() + 2;
   if(!cB.Init(0, index, OpenCL, iDimension * iDimension * iVariables, optimization, iBatch))
      ReturnFalse;
   cB.SetActivationFunction(None);
   index++;
   if(!cC.Init(0, index, OpenCL, cB.Neurons(), optimization, iBatch))
      ReturnFalse;
   cC.SetActivationFunction(None);
   index++;
   if(!cDx.Init(0, index, OpenCL, iDimension * iVariables, optimization, iBatch))
      ReturnFalse;
   cDx.SetActivationFunction(None);
   index++;
   if(!cE.Init(0, index, OpenCL, cA.Neurons(), optimization, iBatch))
      ReturnFalse;
   cE.SetActivationFunction(None);
   index++;
   if(!cH.Init(0, index, OpenCL, iDimension * iVariables, optimization, iBatch))
      ReturnFalse;
   cH.SetActivationFunction(None);
   if(!cH.Clear())
      ReturnFalse;
   index++;
//---
   if(!cAE.Init(0, index, OpenCL, cA.Neurons(), optimization, iBatch))
      ReturnFalse;
   cAE.SetActivationFunction(None);
   index++;
   uint AE_dimension = cAE.Neurons() / (iDimension * iVariables);
   if(!cAET.Init(0, index, OpenCL, iVariables, iDimension, AE_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cAED.Init(0, index, OpenCL, cB.Neurons(), optimization, iBatch))
      ReturnFalse;
   cAED.SetActivationFunction(None);
   index++;
//---
   if(!cAH.Init(0, index, OpenCL, iDimension * iVariables, optimization, iBatch))
      ReturnFalse;
   cAH.SetActivationFunction(None);
   if(!cBX.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cBX.SetActivationFunction(None);
   index++;
   if(!cHc.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cHc.SetActivationFunction(None);
   index++;
   if(!cCHc.Init(0, index, OpenCL, cAH.Neurons(), optimization, iBatch))
      ReturnFalse;
   cCHc.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSE::Clear(void)
  {
   if(!CNeuronAddToStack::Clear())
      ReturnFalse;
//---
   if(!cPrepare.ClearStates())
      ReturnFalse;
   if(!cBCDxE.Clear())
      ReturnFalse;
   if(!cH.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPSSE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronAddToStack::SetOpenCL(obj);
//---
   cA.SetOpenCL(OpenCL);
   cPrepare.SetOpenCL(OpenCL);
   cBCDxE.SetOpenCL(OpenCL);
   cB.SetOpenCL(OpenCL);
   cC.SetOpenCL(OpenCL);
   cDx.SetOpenCL(OpenCL);
   cE.SetOpenCL(OpenCL);
   cH.SetOpenCL(OpenCL);
   cAE.SetOpenCL(OpenCL);
   cAET.SetOpenCL(OpenCL);
   cAED.SetOpenCL(OpenCL);
   cAH.SetOpenCL(OpenCL);
   cBX.SetOpenCL(OpenCL);
   cHc.SetOpenCL(OpenCL);
   cCHc.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPSSE::TrainMode(bool flag)
  {
   CNeuronAddToStack::TrainMode(flag);
//---
   cA.TrainMode(bTrain);
   cPrepare.TrainMode(bTrain);
   cBCDxE.TrainMode(bTrain);
   cB.TrainMode(bTrain);
   cC.TrainMode(bTrain);
   cDx.TrainMode(bTrain);
   cE.TrainMode(bTrain);
   cH.TrainMode(bTrain);
   cAE.TrainMode(bTrain);
   cAET.TrainMode(bTrain);
   cAED.TrainMode(bTrain);
   cAH.TrainMode(bTrain);
   cBX.TrainMode(bTrain);
   cHc.TrainMode(bTrain);
   cCHc.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
      uint chanels_in, uint chanels_out, uint depth_window,
      uint depth_step, uint units, uint variables,
      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeDepthWiseConv::Init(numOutputs, myIndex, open_cl, chanels_in,
                                       chanels_out, depth_window, depth_step,
                                       units, variables, optimization_type, batch))
      ReturnFalse;
   if(!cResidual.Init(0, 0, OpenCL, chanels_in * depth_window, chanels_in * depth_step,
                      chanels_out, units, variables, optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(Gradient, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!CNeuronSpikeDepthWiseConv::feedForward(NeuronOCL))
      ReturnFalse;
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!SumAndNormalize(Output, cResidual.getOutput(), Output, GetFilters(), true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeDepthWiseConv::calcInputGradients(NeuronOCL))
      ReturnFalse;
   CBufferFloat* temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!CNeuronSpikeDepthWiseConv::updateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeDepthWiseConv::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeDepthWiseResidual* Source = source;
   dWeightsUpdate(cResidual, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::Save(const int file_handle)
  {
   if(!CNeuronSpikeDepthWiseConv::Save(file_handle))
      ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeDepthWiseResidual::Load(const int file_handle)
  {
   if(!CNeuronSpikeDepthWiseConv::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeDepthWiseResidual::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeDepthWiseConv::SetOpenCL(obj);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeDepthWiseResidual::TrainMode(bool flag)
  {
   CNeuronSpikeDepthWiseConv::TrainMode(flag);
   cResidual.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                        uint chanels_in, uint chanels_out, uint &depth_window[],
                                        uint &depth_step[], uint units, uint variables,
                                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMSRes::Init(numOutputs, myIndex, open_cl, units, chanels_out,
                          variables, optimization_type, batch))
      ReturnFalse;
//---
   uint layers = depth_window.Size();
   if(layers == 0 || layers != depth_step.Size())
      ReturnFalse;
   uint units_dw[];
   if(ArrayResize(units_dw, layers) < (int)layers)
      ReturnFalse;
   units_dw[layers - 1] = units;
   for(int i = (int)layers - 2; i >= 0; i--)
      units_dw[i] = units_dw[i + 1] * depth_step[i + 1];
   uint chanels_dw_in = chanels_in;
   CNeuronSpikeDepthWiseResidual* conv = NULL;
   cSKFlow.Clear();
   cSKFlow.SetOpenCL(OpenCL);
   for(uint i = 0; i < layers; i++)
     {
      conv = new CNeuronSpikeDepthWiseResidual();
      if(!conv ||
         !conv.Init(0, i, OpenCL, chanels_dw_in, chanels_out, depth_window[i], depth_step[i],
                    units_dw[i], variables, optimization, iBatch) ||
         !cSKFlow.Add(conv))
         DeleteObjAndFalse(conv);
      chanels_dw_in = chanels_out;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cSKFlow.Total(); i++)
     {
      curr = cSKFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronMSRes::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronMSRes::calcInputGradients(cSKFlow[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cSKFlow[-1];
   CNeuronBaseOCL* curr = NULL;
   for(int i = cSKFlow.Total() - 2; i >= 0; i--)
     {
      curr = cSKFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cSKFlow.Total(); i++)
     {
      curr = cSKFlow[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronMSRes::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::Save(const int file_handle)
  {
   if(!CNeuronMSRes::Save(file_handle))
      ReturnFalse;
//---
   if(!cSKFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::Load(const int file_handle)
  {
   if(!CNeuronMSRes::Load(file_handle))
      ReturnFalse;
//---
   if(!cSKFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSuperKernelBlock::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMSRes::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSuperKernelBlock* Source = source;
   dWeightsUpdate(cSKFlow, Source, tau)
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSuperKernelBlock::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMSRes::SetOpenCL(obj);
   cSKFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSuperKernelBlock::TrainMode(bool flag)
  {
   CNeuronMSRes::TrainMode(flag);
   cSKFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint chanels, uint units, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(chanels < 2)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, chanels * units * variables, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   iChanels = chanels;
   iUnits = units;
   iVariables = variables;
//---
   cQKV.Clear();
   cQKV.SetOpenCL(OpenCL);
   CNeuronSpikeActivation* act = NULL;
   CNeuronMultiWindowsConvOCL* conv = NULL;
   CNeuronBatchNormOCL* norm = NULL;
   uint index = 0;
//---
   act = new CNeuronSpikeActivation();
   if(!act ||
      !act.Init(0, index, OpenCL, 2 * chanels * units * variables, optimization, iBatch) ||
      !cQKV.Add(act))
      DeleteObjAndFalse(act);
//---
   index++;
   uint windows[] = {chanels, chanels};
   conv = new CNeuronMultiWindowsConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, windows, chanels, units, variables, optimization, iBatch) ||
      !cQKV.Add(conv))
      DeleteObjAndFalse(conv);
   conv.SetActivationFunction(None);
//---
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, conv.Neurons(), iBatch, optimization) ||
      !cQKV.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
//---
   index++;
   if(!cQ.Init(0, index, OpenCL, (chanels + 1) / 2 * units * variables, optimization, iBatch))
      ReturnFalse;
   cQ.SetActivationFunction(None);
   cQ.Clear();
   index++;
   if(!cK.Init(0, index, OpenCL, (chanels + 1) / 2 * units * variables, optimization, iBatch))
      ReturnFalse;
   cK.SetActivationFunction(None);
   cK.Clear();
   index++;
   if(!cKT.Init(0, index, OpenCL, variables, units, (chanels + 1) / 2, optimization, iBatch))
      ReturnFalse;
   cKT.SetActivationFunction(None);
   index++;
   if(!cV.Init(0, index, OpenCL, chanels * units * variables, optimization, iBatch))
      ReturnFalse;
   cV.SetActivationFunction(None);
//---
   cScore.Clear();
   cScore.SetOpenCL(OpenCL);
   CNeuronBaseOCL* neuron = NULL;
   CNeuronSoftMaxOCL* softmax = NULL;
//---
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, units * units * variables, optimization, iBatch) ||
      !cScore.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, neuron.Neurons(), optimization, iBatch) ||
      !cScore.Add(softmax))
      DeleteObjAndFalse(softmax);
   softmax.SetHeads(units * variables);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cQKV.Total(); i++)
     {
      curr = cQKV[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!DeConcat(cV.getOutput(), cQ.getOutput(), cK.getOutput(), prev.getOutput(),
                iChanels, iChanels / 2, iChanels / 2, iUnits * iVariables))
      ReturnFalse;
   if(!cKT.FeedForward(cK.AsObject()))
      ReturnFalse;
   prev = cScore[0];
   if(!MatMul(cQ.getOutput(), cKT.getOutput(), prev.getOutput(), iUnits, iChanels / 2, iUnits, iVariables, true))
      ReturnFalse;
   for(int i = 1; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!DeConcat(PrevOutput, Output, NeuronOCL.getOutput(), iChanels, iChanels, iUnits * iVariables))
      ReturnFalse;
   if(!MatMul(prev.getOutput(), cV.getOutput(), Output, iUnits, iUnits, iChanels, iVariables, true))
      ReturnFalse;
   if(!SumAndNormalize(PrevOutput, Output, Output, iChanels, true, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* next = cScore[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next ||
      !MatMulGrad(next.getOutput(), next.getGradient(), cV.getOutput(), cV.getGradient(),
                  Gradient, iUnits, iUnits, iChanels, iVariables, true))
      ReturnFalse;
   for(int i = cScore.Total() - 2; i >= 0; i--)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!MatMulGrad(cQ.getOutput(), cQ.getGradient(), cKT.getOutput(), cKT.getGradient(),
                  next.getGradient(), iUnits, iChanels / 2, iUnits, iVariables, true))
      ReturnFalse;
   if(!cK.CalcHiddenGradients(cKT.AsObject()))
      ReturnFalse;
   next = cQKV[-1];
   if(!next ||
      !Concat(cV.getGradient(), cQ.getGradient(), cK.getGradient(), next.getGradient(),
              iChanels, iChanels / 2, iChanels / 2, iUnits * iVariables))
      ReturnFalse;
   for(int i = cQKV.Total() - 1; i >= 0; i--)
     {
      curr = cQKV[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
   if(!PrevOutput.Fill(0) ||
      !Concat(PrevOutput, Gradient, next.getPrevOutput(), iChanels, iChanels, iUnits * iVariables))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), next.getPrevOutput(), next.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(next.getPrevOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), iChanels, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cQKV.Total(); i++)
     {
      curr = cQKV[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   for(int i = 1; i < cScore.Total(); i++)
     {
      curr = cScore[i];
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
bool CNeuronSpikeGMA::Save(int const file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cQKV.Save(file_handle))
      ReturnFalse;
   if(!cKT.Save(file_handle))
      ReturnFalse;
   if(!cScore.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::Load(int const file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!cQKV.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cKT.AsObject()))
      ReturnFalse;
   if(!cScore.Load(file_handle))
      ReturnFalse;
//---
   iUnits = cKT.GetCount();
   iVariables = cKT.GetVariables();
   iChanels = Neurons() / (iUnits * iVariables);
//---
   uint index = cQKV.Total();
   if(!cQ.Init(0, index, OpenCL, (iChanels + 1) / 2 * iUnits * iVariables, optimization, iBatch))
      ReturnFalse;
   cQ.SetActivationFunction(None);
   cQ.Clear();
   index++;
   if(!cK.Init(0, index, OpenCL, (iChanels + 1) / 2 * iUnits * iVariables, optimization, iBatch))
      ReturnFalse;
   cK.SetActivationFunction(None);
   cK.Clear();
   index += 2;
   if(!cV.Init(0, index, OpenCL, iChanels * iUnits * iVariables, optimization, iBatch))
      ReturnFalse;
   cV.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeGMA::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronSpikeGMA* Source = source;
   dWeightsUpdate(cQKV, Source, tau);
   dWeightsUpdate(cKT, Source, tau);
   dWeightsUpdate(cScore, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeGMA::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cQKV.SetOpenCL(OpenCL);
   cKT.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
   cQ.SetOpenCL(OpenCL);
   cK.SetOpenCL(OpenCL);
   cV.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeGMA::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cQKV.TrainMode(bTrain);
   cKT.TrainMode(bTrain);
   cScore.TrainMode(bTrain);
   cQ.TrainMode(bTrain);
   cK.TrainMode(bTrain);
   cV.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint stack_size, uint dimension_in, uint dimension_out,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronAddToStack::Init(numOutputs, myIndex, open_cl, stack_size, dimension_out,
                               1, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cPSSE.Init(0, index, OpenCL, 3, dimension_in, 5 * dimension_out, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cRefFrame.Init(0, index, OpenCL, 5 * dimension_out, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
   index++;
   if(!cRefFrameT.Init(0, index, OpenCL, 5, dimension_out, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cForwardBackward.Init(0, index, OpenCL, 10 * dimension_out, optimization, iBatch))
      ReturnFalse;
   cForwardBackward.SetActivationFunction(None);
   index++;
   if(!cCorrelation.Init(0, index, OpenCL, 10 * 5, optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
   index++;
//---
   cMixFusion.Clear();
   cMixFusion.SetOpenCL(OpenCL);
   CNeuronBaseOCL*   neuron = NULL;
   CNeuronTransposeRCDOCL* transp = NULL;
   CNeuronSpikeSuperKernelBlock* sk_block = NULL;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, cForwardBackward.Neurons() + cCorrelation.Neurons(), optimization, iBatch) ||
      !cMixFusion.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   transp = new CNeuronTransposeRCDOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, 2, 5, neuron.Neurons() / 10, optimization, iBatch) ||
      !cMixFusion.Add(transp))
      DeleteObjAndFalse(transp);
   index++;
   uint windows[] = {3, transp.GetWindow()};
   uint steps[] = {1, 1};
   sk_block = new CNeuronSpikeSuperKernelBlock();
   if(!sk_block ||
      !sk_block.Init(0, index, OpenCL, transp.GetCount()*transp.GetDimension(), dimension_out, windows, steps,
                     transp.GetWindow(), 1, optimization, iBatch) ||
      !cMixFusion.Add(sk_block))
      DeleteObjAndFalse(sk_block);
   index++;
//---
   if(!cContext.Init(0, index, OpenCL, cRefFrame.Neurons() / transp.GetWindow(), dimension_out, windows, steps,
                     transp.GetWindow(), 1, optimization, iBatch))
      ReturnFalse;
//---
   cMotionEncoder.Clear();
   cMotionEncoder.SetOpenCL(OpenCL);
   CNeuronSpikeGMA* gma = NULL;
//---
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 2 * dimension_out * transp.GetWindow(), optimization, iBatch) ||
      !cMotionEncoder.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   gma = new CNeuronSpikeGMA();
   if(!gma ||
      !gma.Init(0, index, OpenCL, dimension_out, cRefFrameT.GetCount(), 1, optimization, iBatch) ||
      !cMotionEncoder.Add(gma))
      DeleteObjAndFalse(gma);
   index++;
   windows[0] = cRefFrameT.GetCount();
   windows[1] = 2 * windows[0];
   steps[0] = 1;
   steps[1] = windows[0];
   sk_block = new CNeuronSpikeSuperKernelBlock();
   if(!sk_block ||
      !sk_block.Init(0, index, OpenCL, dimension_out, dimension_out, windows, steps, 1, 1, optimization, iBatch) ||
      !cMotionEncoder.Add(sk_block))
      DeleteObjAndFalse(sk_block);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPSSE.FeedForward(NeuronOCL))
      ReturnFalse;
   uint units = cRefFrameT.GetCount();
   if(!DeConcat(cRefFrame.getPrevOutput(), cRefFrame.getOutput(), cForwardBackward.getPrevOutput(),
                cPSSE.getOutput(), units * iDimension, units * iDimension, units * iDimension, 1))
      ReturnFalse;
   if(!Concat(cRefFrame.getPrevOutput(), cForwardBackward.getPrevOutput(), cForwardBackward.getOutput(),
              units * iDimension, units * iDimension, 1))
      ReturnFalse;
   if(!cRefFrameT.FeedForward(cRefFrame.AsObject()))
      ReturnFalse;
   if(!MatMul(cForwardBackward.getOutput(), cRefFrameT.getOutput(), cCorrelation.getOutput(),
              units, iDimension, units, 2, false))
      ReturnFalse;
   if(!cContext.FeedForward(cRefFrame.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cMixFusion[0];
   CNeuronBaseOCL* curr = NULL;
   if(!prev ||
      !Concat(cForwardBackward.getOutput(), cCorrelation.getOutput(), prev.getOutput(), iDimension, units, 2 * units))
      ReturnFalse;
   for(int i = 1; i < cMixFusion.Total(); i++)
     {
      curr = cMixFusion[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   curr = cMotionEncoder[0];
   if(!curr ||
      !Concat(cContext.getOutput(), prev.getOutput(), curr.getOutput(), iDimension, iDimension, units))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cMotionEncoder.Total(); i++)
     {
      curr = cMotionEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronAddToStack::feedForward(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronAddToStack::calcInputGradients(cMotionEncoder[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   for(int i = cMotionEncoder.Total() - 2; i >= 0; i--)
     {
      curr = cMotionEncoder[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cMotionEncoder[i + 1]))
         ReturnFalse;
     }
//---
   uint units = cRefFrameT.GetCount();
   curr = cMixFusion[-1];
   if(!curr ||
      !DeConcat(cContext.getGradient(), curr.getGradient(), cMotionEncoder[0].getGradient(), iDimension, iDimension, units))
      ReturnFalse;
   for(int i = cMixFusion.Total() - 2; i >= 0; i--)
     {
      curr = cMixFusion[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cMixFusion[i + 1]))
         ReturnFalse;
     }
   if(!DeConcat(cForwardBackward.getGradient(), cCorrelation.getGradient(), curr.getGradient(), iDimension, units, 2 * units))
      ReturnFalse;
   if(!MatMulGrad(cForwardBackward.getOutput(), cForwardBackward.getPrevOutput(),
                  cRefFrameT.getOutput(), cRefFrameT.getGradient(), cCorrelation.getGradient(),
                  units, iDimension, units, 2, false))
      ReturnFalse;
   if(!SumAndNormalize(cForwardBackward.getGradient(), cForwardBackward.getPrevOutput(), cForwardBackward.getGradient(),
                       iDimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!cRefFrame.CalcHiddenGradients(cContext.AsObject()))
      ReturnFalse;
   CBufferFloat* temp = cRefFrame.getGradient();
   if(!cRefFrame.SetGradient(cRefFrame.getPrevOutput(), false) ||
      !cRefFrame.CalcHiddenGradients(cRefFrameT.AsObject()) ||
      !SumAndNormalize(cRefFrame.getGradient(), temp, temp, iDimension, false, 0, 0, 0, 1) ||
      !cRefFrame.SetGradient(temp, false))
      ReturnFalse;
   if(!DeConcat(cRefFrameT.getPrevOutput(), cForwardBackward.getPrevOutput(), cForwardBackward.getGradient(),
                units * iDimension, units * iDimension, 1))
      ReturnFalse;
   if(!Concat(cRefFrameT.getPrevOutput(), cRefFrame.getGradient(), cForwardBackward.getPrevOutput(),
              cPSSE.getGradient(), units * iDimension, units * iDimension, units * iDimension, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cPSSE.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPSSE.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cContext.UpdateInputWeights(cRefFrame.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cMixFusion[0];
   CNeuronBaseOCL* curr = NULL;
   for(int i = 1; i < cMixFusion.Total(); i++)
     {
      curr = cMixFusion[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   prev = cMotionEncoder[0];
   for(int i = 1; i < cMotionEncoder.Total(); i++)
     {
      curr = cMotionEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!CNeuronAddToStack::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronAddToStack::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronETROF* Source = source;
   dWeightsUpdate(cPSSE, Source, tau);
   dWeightsUpdate(cContext, Source, tau);
   dWeightsUpdate(cMixFusion, Source, tau);
   dWeightsUpdate(cMotionEncoder, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::Save(const int file_handle)
  {
   if(!CNeuronAddToStack::Save(file_handle))
      ReturnFalse;
//---
   if(!cPSSE.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cMixFusion.Save(file_handle))
      ReturnFalse;
   if(!cMotionEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::Load(const int file_handle)
  {
   if(!CNeuronAddToStack::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cPSSE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cContext.AsObject()))
      ReturnFalse;
   if(!cMixFusion.Load(file_handle))
      ReturnFalse;
   if(!cMotionEncoder.Load(file_handle))
      ReturnFalse;
//---
   uint index = 1;
   if(!cRefFrame.Init(0, index, OpenCL, cPSSE.Neurons() / 3, optimization, iBatch))
      ReturnFalse;
   cRefFrame.SetActivationFunction(None);
   index++;
   if(!cRefFrameT.Init(0, index, OpenCL, cRefFrame.Neurons() / iDimension, iDimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cForwardBackward.Init(0, index, OpenCL, 2 * cRefFrame.Neurons(), optimization, iBatch))
      ReturnFalse;
   cForwardBackward.SetActivationFunction(None);
   index++;
   if(!cCorrelation.Init(0, index, OpenCL, 2 * cRefFrameT.GetCount()*cRefFrameT.GetCount(), optimization, iBatch))
      ReturnFalse;
   cCorrelation.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronETROF::Clear(void)
  {
   if(!CNeuronAddToStack::Clear())
      ReturnFalse;
//---
   if(!cPSSE.Clear())
      ReturnFalse;
   if(!cContext.Clear())
      ReturnFalse;
   if(!cMixFusion.ClearStates())
      ReturnFalse;
   if(!cMotionEncoder.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronETROF::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronAddToStack::SetOpenCL(obj);
//---
   cPSSE.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cMixFusion.SetOpenCL(OpenCL);
   cMotionEncoder.SetOpenCL(OpenCL);
   cRefFrame.SetOpenCL(OpenCL);
   cRefFrameT.SetOpenCL(OpenCL);
   cForwardBackward.SetOpenCL(OpenCL);
   cCorrelation.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronETROF::TrainMode(bool flag)
  {
   CNeuronAddToStack::TrainMode(flag);
//---
   cPSSE.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cMixFusion.TrainMode(bTrain);
   cMotionEncoder.TrainMode(bTrain);
   cRefFrame.TrainMode(bTrain);
   cRefFrameT.TrainMode(bTrain);
   cForwardBackward.TrainMode(bTrain);
   cCorrelation.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint forecast, uint dimension, uint hidden_dimension,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   uint windows[] = {3, 5};
   uint steps[] = {1, 1};
   if(!CNeuronSpikeSuperKernelBlock::Init(numOutputs, myIndex, open_cl, hidden_dimension, dimension,
                                          windows, steps, forecast, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cETROF.Init(0, index, OpenCL, forecast, dimension, hidden_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cLocal.Init(0, index, OpenCL, hidden_dimension, hidden_dimension, windows, steps, forecast, 1, optimization, iBatch))
      ReturnFalse;
   index++;
//---
   cGlobal.Clear();
   cGlobal.SetOpenCL(OpenCL);
   CNeuronBaseOCL* neuron = NULL;
   CNeuronSpikeGMA* gma = NULL;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 2 * hidden_dimension * forecast, optimization, iBatch) ||
      !cGlobal.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   gma = new CNeuronSpikeGMA();
   if(!gma ||
      !gma.Init(0, index, OpenCL, hidden_dimension, forecast, 1, optimization, iBatch) ||
      !cGlobal.Add(gma))
      DeleteObjAndFalse(gma);
   index++;
//---
   cUpdateFlow.Clear();
   cUpdateFlow.SetOpenCL(OpenCL);
   CNeuronSpikeSuperKernelBlock* sk_blok = NULL;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, 4 * hidden_dimension * forecast, optimization, iBatch) ||
      !cUpdateFlow.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   sk_blok = new CNeuronSpikeSuperKernelBlock();
   if(!sk_blok ||
      !sk_blok.Init(0, index, OpenCL, 4 * hidden_dimension, hidden_dimension, windows, steps, forecast, 1, optimization, iBatch) ||
      !cUpdateFlow.Add(sk_blok))
      DeleteObjAndFalse(sk_blok);
   if(!sk_blok.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cETROF.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cLocal.FeedForward(cETROF.AsObject()))
      ReturnFalse;
//---
   CNeuronSpikeDepthWiseResidual* resid = cSKFlow[0];
   uint hidden_dimension = resid.GetWindow();
   uint units = cLocal.Neurons() / hidden_dimension;
   uint dimension = Neurons() / units;
//---
   CNeuronBaseOCL* prev = cGlobal[0];
   CNeuronBaseOCL* curr = NULL;
   if(!prev ||
      !Concat(cLocal.getOutput(), cETROF.getOutput(), prev.getOutput(), hidden_dimension, hidden_dimension, units))
      ReturnFalse;
   for(int i = 1; i < cGlobal.Total(); i++)
     {
      curr = cGlobal[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   curr = cUpdateFlow[0];
   prev = cUpdateFlow[-1];
   if(!prev || !curr ||
      !Concat(cLocal.getOutput(), cETROF.getOutput(), cGlobal[-1].getOutput(), prev.getOutput(), curr.getOutput(),
              hidden_dimension, hidden_dimension, hidden_dimension, hidden_dimension, units))
      ReturnFalse;
   prev = curr;
   for(int i = 1; i < cUpdateFlow.Total(); i++)
     {
      curr = cUpdateFlow[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeSuperKernelBlock::feedForward(prev))
      ReturnFalse;
   if(!SumVecMatrix(NeuronOCL.getOutput(), Output, Output, dimension, 1, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeSuperKernelBlock::calcInputGradients(cUpdateFlow[-1]))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   for(int i = cUpdateFlow.Total() - 2; i >= 0; i--)
     {
      curr = cUpdateFlow[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cUpdateFlow[i + 1]))
         ReturnFalse;
     }
//---
   CNeuronSpikeDepthWiseResidual* resid = cSKFlow[0];
   uint hidden_dimension = resid.GetWindow();
   uint units = cLocal.Neurons() / hidden_dimension;
   uint dimension = Neurons() / units;
//---
   curr = cGlobal[-1];
   if(!curr ||
      !DeConcat(cLocal.getPrevOutput(), cETROF.getPrevOutput(), curr.getGradient(), cUpdateFlow[0].getPrevOutput(),
                cUpdateFlow[0].getGradient(), hidden_dimension, hidden_dimension, hidden_dimension, hidden_dimension, units))
      ReturnFalse;
   Deactivation(curr);
   for(int i = cGlobal.Total() - 2; i >= 0; i--)
     {
      curr = cGlobal[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cGlobal[i + 1]))
         ReturnFalse;
     }
//---
   if(!DeConcat(cLocal.getGradient(), cETROF.getGradient(), curr.getGradient(), hidden_dimension, hidden_dimension, units))
      ReturnFalse;
   if(!SumAndNormalize(cLocal.getGradient(), cLocal.getPrevOutput(), cLocal.getGradient(), hidden_dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!SumAndNormalize(cETROF.getGradient(), cETROF.getPrevOutput(), cETROF.getPrevOutput(), hidden_dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   Deactivation(cLocal);
//---
   if(!cETROF.CalcHiddenGradients(cLocal.AsObject()))
      ReturnFalse;
   if(cETROF.Activation() != None)
      if(!DeActivation(cETROF.getOutput(), cETROF.getPrevOutput(), cETROF.getPrevOutput(), cETROF.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(cETROF.getGradient(), cETROF.getPrevOutput(), cETROF.getGradient(), hidden_dimension, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cETROF.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cETROF.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cLocal.UpdateInputWeights(cETROF.AsObject()))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cGlobal[0];
   CNeuronBaseOCL* curr = NULL;
   for(int i = 1; i < cGlobal.Total(); i++)
     {
      curr = cGlobal[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   prev = cUpdateFlow[0];
   for(int i = 1; i < cUpdateFlow.Total(); i++)
     {
      curr = cUpdateFlow[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!CNeuronSpikeSuperKernelBlock::updateInputWeights(prev))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::Save(const int file_handle)
  {
   if(!CNeuronSpikeSuperKernelBlock::Save(file_handle))
      ReturnFalse;
   if(!cETROF.Save(file_handle))
      ReturnFalse;
   if(!cLocal.Save(file_handle))
      ReturnFalse;
//---
   if(!cGlobal.Save(file_handle))
      ReturnFalse;
   if(!cUpdateFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikeSuperKernelBlock::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cETROF.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLocal.AsObject()))
      ReturnFalse;
//---
   if(!cGlobal.Load(file_handle))
      ReturnFalse;
   if(!cUpdateFlow.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeSuperKernelBlock::WeightsUpdate(source, tau))
      ReturnFalse
      //---
      CNeuronPSSEFlow* Source = source;
   dWeightsUpdate(cETROF, Source, tau);
   dWeightsUpdate(cLocal, Source, tau);
   dWeightsUpdate(cGlobal, Source, tau);
   dWeightsUpdate(cUpdateFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSSEFlow::Clear(void)
  {
   if(!CNeuronSpikeSuperKernelBlock::Clear())
      ReturnFalse;
   if(!cETROF.Clear())
      ReturnFalse;
   if(!cLocal.Clear())
      ReturnFalse;
//---
   if(!cGlobal.ClearStates())
      ReturnFalse;
   if(!cUpdateFlow.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPSSEFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeSuperKernelBlock::SetOpenCL(obj);
   cETROF.SetOpenCL(OpenCL);
   cLocal.SetOpenCL(OpenCL);
   cGlobal.SetOpenCL(OpenCL);
   cUpdateFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPSSEFlow::TrainMode(bool flag)
  {
   CNeuronSpikeSuperKernelBlock::TrainMode(flag);
   cETROF.TrainMode(bTrain);
   cLocal.TrainMode(bTrain);
   cGlobal.TrainMode(bTrain);
   cUpdateFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint stack_size, uint dimension,
                           uint patch_dimension, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeADM::Init(numOutputs, myIndex, open_cl, patch_dimension,
                             stack_size, variables, optimization_type, batch))
      ReturnFalse;
   if(!cFlow.Init(0, 0, OpenCL, stack_size, dimension, patch_dimension,
                  variables, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cFlow.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeADM::feedForward(cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!CNeuronSpikeADM::calcInputGradients(cFlow.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cFlow.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!CNeuronSpikeADM::updateInputWeights(cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::Save(const int file_handle)
  {
   if(!CNeuronSpikeADM::Save(file_handle))
      ReturnFalse;
   if(!cFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::Load(const int file_handle)
  {
   if(!CNeuronSpikeADM::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeADM::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeUSR* Source = source;
   dWeightsUpdate(cFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeUSR::Clear(void)
  {
   if(!CNeuronSpikeADM::Clear())
      ReturnFalse;
   if(!cFlow.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeUSR::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeADM::SetOpenCL(obj);
   cFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeUSR::TrainMode(bool flag)
  {
   CNeuronSpikeADM::TrainMode(flag);
   cFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                              uint window, uint units, uint window_out,
                              uint candidates, uint stack_size,
                              ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units * window_out, optimization_type, batch))
      ReturnFalse;
   activation = None;
//---
   uint index = 0;
   if(!cCandidates.Init(0, index, OpenCL, units * window_out * candidates, optimization, iBatch))
      ReturnFalse;
   cCandidates.SetActivationFunction(None);
   CBufferFloat* temp = cCandidates.getWeightsParams();
   if(!temp ||
      !temp.Fill(0))
      ReturnFalse;
   index++;
//---
   cScores.Clear();
   cScores.SetOpenCL(OpenCL);
   CNeuronSpikeDepthWiseConv* dw_conv = NULL;
   CNeuronSoftMaxOCL* softmax = NULL;
   uint window_in = window;
   for(int i = 0; i < 2; i++)
     {
      dw_conv = new CNeuronSpikeDepthWiseConv();
      if(!dw_conv ||
         !dw_conv.Init(0, index, OpenCL, window_in, candidates, 3, 1, units, 1, optimization, iBatch) ||
         !cScores.Add(dw_conv))
         DeleteObjAndFalse(dw_conv);
      index++;
      window_in = candidates;
     }
   softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, units * candidates, optimization, iBatch) ||
      !cScores.Add(softmax))
      DeleteObjAndFalse(softmax);
   index++;
   if(!cStack.Init(0, index, OpenCL, stack_size, softmax.Neurons(), 1, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
      if(!cCandidates.FeedForward())
         ReturnFalse;
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScores.Total(); i++)
     {
      curr = cScores[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   CNeuronSoftMaxOCL* softmax = prev;
   uint units = softmax.Heads();
   uint candidates = softmax.Neurons() / units;
   uint dimension = Neurons() / units;
   if(!MatMul(cCandidates.getOutput(), softmax.getOutput(), Output, dimension, candidates, 1, units, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronSoftMaxOCL* softmax = cScores[-1];
   if(!cStack.FeedForward(softmax))
      ReturnFalse;
//---
   uint units = softmax.Heads();
   uint candidates = softmax.Neurons() / units;
   uint dimension = Neurons() / units;
   if(!DiversityLoss(cStack.AsObject(), cStack.GetStackSize(), candidates, false))
      ReturnFalse;
//---
   if(!MatMulGrad(cCandidates.getOutput(), cCandidates.getGradient(),
                  softmax.getOutput(), softmax.getGradient(),
                  Gradient, dimension, candidates, 1, units, true))
      ReturnFalse;
   if(!SumAndNormalize(softmax.getGradient(), cStack.getGradient(), softmax.getGradient(), candidates, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   CNeuronBaseOCL* curr = NULL;
   for(int i = cScores.Total() - 2; i >= 0; i--)
     {
      curr = cScores[i];
      if(!curr ||
         !curr.CalcHiddenGradients(cScores[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(curr))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cCandidates.UpdateInputWeights())
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScores.Total(); i++)
     {
      curr = cScores[i];
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
bool CNeuronInitialFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
//---
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronInitialFlow* Source = source;
   dWeightsUpdate(cCandidates, Source, tau);
   dWeightsUpdate(cScores, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::Save(const int file_handle)
  {
//---
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cCandidates.Save(file_handle))
      ReturnFalse;
   if(!cScores.Save(file_handle))
      ReturnFalse;
   if(!cStack.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::Load(const int file_handle)
  {
//---
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cCandidates.AsObject()))
      ReturnFalse;
   if(!cScores.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cStack.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronInitialFlow::Clear(void)
  {
//---
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   if(!cCandidates.Clear())
      ReturnFalse;
   if(!cScores.ClearStates())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronInitialFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cCandidates.SetOpenCL(OpenCL);
   cScores.SetOpenCL(OpenCL);
   cStack.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronInitialFlow::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cCandidates.TrainMode(bTrain);
   cScores.TrainMode(bTrain);
   cStack.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint units, uint window, uint second_window, uint window_out,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMSRes::Init(numOutputs, myIndex, open_cl, units, window_out, 1, optimization_type, batch))
      ReturnFalse;
//---
   uint windows[] = {3, 5};
   uint steps[] = {1, 1};
   uint index = 0;
   if(!cUpsampleFlow.Init(0, index, OpenCL, window, window_out, windows, steps, units, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcat.Init(0, index, OpenCL, (window + second_window)*units, optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
   index++;
   if(!cConvGRU.Init(0, index, OpenCL, units, (window + second_window), window_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      ReturnFalse;
   if(!cUpsampleFlow.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   uint units = GetUnits();
   uint window_flow = NeuronOCL.Neurons() / units;;
   uint second_window = SecondInput.Total() / units;
   if(!Concat(NeuronOCL.getOutput(), SecondInput, cConcat.getOutput(), window_flow, second_window, units))
      ReturnFalse;
   if(!cConvGRU.FeedForward(cConcat.AsObject()))
      ReturnFalse;
   if(!CNeuronMSRes::feedForward(cConvGRU.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cUpsampleFlow.getOutput(), Output, Output, GetWindow(), false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
//---
   if(!CNeuronMSRes::calcInputGradients(cConvGRU.AsObject()))
      ReturnFalse;
   if(!cConcat.CalcHiddenGradients(cConvGRU.AsObject()))
      ReturnFalse;
//---
   if(!DeActivation(cUpsampleFlow.getOutput(), cUpsampleFlow.getGradient(), Gradient, cUpsampleFlow.Activation()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cUpsampleFlow.AsObject()))
      ReturnFalse;
//---
   uint units = GetUnits();
   uint window_flow = NeuronOCL.Neurons() / units;;
   uint second_window = SecondInput.Total() / units;
   if(!DeConcat(cConcat.getPrevOutput(), SecondGradient, cConcat.getGradient(), window_flow, second_window, units))
      ReturnFalse;
   if(SecondActivation != None)
      if(!DeActivation(SecondInput, SecondGradient, SecondGradient, SecondActivation))
         ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), cConcat.getPrevOutput(), cConcat.getPrevOutput(), NeuronOCL.Activation()))
         ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), cConcat.getPrevOutput(), NeuronOCL.getGradient(), window_flow, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   if(!cUpsampleFlow.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   if(!cConvGRU.UpdateInputWeights(cConcat.AsObject()))
      ReturnFalse;
   if(!CNeuronMSRes::updateInputWeights(cConvGRU.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMSRes::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSpikeSMR* Source = source;
   dWeightsUpdate(cUpsampleFlow, Source, tau);
   dWeightsUpdate(cConvGRU, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::Save(const int file_handle)
  {
   if(!CNeuronMSRes::Save(file_handle))
      ReturnFalse;
   if(!cUpsampleFlow.Save(file_handle))
      ReturnFalse;
   if(!cConvGRU.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::Load(const int file_handle)
  {
   if(!CNeuronMSRes::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cUpsampleFlow.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cConvGRU.AsObject()))
      ReturnFalse;
//---
   if(!cConcat.Init(0, 1, OpenCL, cConvGRU.GetChanelsIn()*cConvGRU.GetUnits(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSpikeSMR::Clear(void)
  {
   if(!CNeuronMSRes::Clear())
      ReturnFalse;
   if(!cUpsampleFlow.Clear())
      ReturnFalse;
   if(!cConvGRU.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSMR::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMSRes::SetOpenCL(obj);
   cUpsampleFlow.SetOpenCL(OpenCL);
   cConvGRU.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSpikeSMR::TrainMode(bool flag)
  {
   CNeuronMSRes::TrainMode(flag);
   cUpsampleFlow.TrainMode(bTrain);
   cConvGRU.TrainMode(bTrain);
   cConcat.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint forecast, uint dimension, uint hidden_dimension,
                          uint layers, uint candidates,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSpikeSMR::Init(numOutputs, myIndex, open_cl, (forecast + 1) / 2, hidden_dimension,
                             2 * hidden_dimension, 2 * dimension, optimization_type, batch))
      ReturnFalse;
//--- Prepare
   cPrepare.Clear();
   cPrepare.SetOpenCL(OpenCL);
   CNeuronBatchNormOCL* norm = NULL;
   CNeuronSpikeUSR*     usr = NULL;
   uint index = 0;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, dimension, iBatch, optimization) ||
      !cPrepare.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
   index++;
   usr = new CNeuronSpikeUSR();
   if(!usr ||
      !usr.Init(0, index, OpenCL, forecast, dimension, hidden_dimension, 1, optimization, iBatch) ||
      !cPrepare.Add(usr))
      DeleteObjAndFalse(usr);
//--- Encoder / Decoder
   cEncoder.Clear();
   cDecoder.Clear();
   cEncoder.SetOpenCL(OpenCL);
   cDecoder.SetOpenCL(OpenCL);
   CNeuronBaseOCL* neuron = NULL;
   CNeuronSpikeSuperKernelBlock* block = NULL;
   CNeuronSpikeSMR* smr = NULL;
   uint windows[] = { 3, 5};
   uint steps_enc[] = {1, 2};
   uint steps_dec[] = {1, 1};
   uint units_in = forecast;
   uint units_out = MathMax((units_in + 1) / 2, 1);
   for(uint i = 0; i < layers; i++)
     {
      index++;
      block = new CNeuronSpikeSuperKernelBlock();
      if(!block ||
         !block.Init(0, index, OpenCL, hidden_dimension, hidden_dimension,
                     windows, steps_enc, units_out, 1, optimization, iBatch) ||
         !cEncoder.Add(block))
         DeleteObjAndFalse(block);
      if(i > 0)
        {
         index++;
         smr = new CNeuronSpikeSMR();
         if(!smr ||
            !smr.Init(0, index, OpenCL, units_out, hidden_dimension, (1 + int(i != (layers - 1))) * hidden_dimension,
                      2 * hidden_dimension, optimization, iBatch) ||
            !cDecoder.Add(smr))
            DeleteObjAndFalse(smr);
        }
      if(i < (layers - 1))
        {
         index++;
         neuron = new CNeuronBaseOCL();
         if(!neuron ||
            !neuron.Init(0, index, OpenCL, units_out * hidden_dimension * 2, optimization, iBatch) ||
            !cDecoder.Add(neuron))
            DeleteObjAndFalse(neuron);
         neuron.SetActivationFunction(None);
        }
      units_in = units_out;
      units_out = MathMax((units_in + 1) / 2, 1);
     }
   index++;
   if(!cInitFlow.Init(0, index, OpenCL, hidden_dimension, units_in, hidden_dimension,
                      candidates, 2 * forecast, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cInitFlow.FeedForward(prev))
      ReturnFalse;
   CNeuronBaseOCL* flow = cInitFlow.AsObject();
   for(int i = cDecoder.Total() - 1; i >= 0; i -= 2)
     {
      curr = cDecoder[i];
      if(!curr ||
         !curr.FeedForward(flow, prev.getOutput()))
         ReturnFalse;
      flow = curr;
      if(i <= 0)
         continue;
      prev = cDecoder[i - 1];
      if(!prev)
         ReturnFalse;
      CNeuronSpikeSMR* smr = flow;
      uint dimension = smr.GetWindow() / 2;
      uint units = prev.Neurons() / (2 * dimension);
      if(!cEncoder[i / 2] ||
         !Concat(smr.GetHidden(), cEncoder[i / 2].getOutput(), prev.getOutput(), dimension, dimension, units))
         ReturnFalse;
     }
   if(!CNeuronSpikeSMR::feedForward(flow, prev.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = (cDecoder.Total() >= 1 ? cDecoder[0] : cEncoder[-1]);
   CNeuronBaseOCL* flow = (cDecoder.Total() >= 2 ? cDecoder[1] : cInitFlow.AsObject());
   CNeuronBaseOCL* curr = NULL;
//---
   if(!CNeuronSpikeSMR::calcInputGradients(flow, prev.getOutput(),
                                           prev.getGradient(),
                                           (ENUM_ACTIVATION) prev.Activation()))
      ReturnFalse;
//---
   for(int i = 1; i < cDecoder.Total(); i += 2)
     {
      flow = cDecoder[i];
      prev = ((i + 1) < cDecoder.Total() ? cDecoder[i + 1] : cEncoder[-1]);
      curr = ((i + 2) < cDecoder.Total() ? cDecoder[i + 2] : cInitFlow.AsObject());
      if(!curr || !prev ||
         !curr.CalcHiddenGradients(flow, prev.getOutput(),
                                   prev.getGradient(),
                                   (ENUM_ACTIVATION) prev.Activation()))
         ReturnFalse;
     }
//---
   CBufferFloat* temp = prev.getGradient();
   if(!prev.SetGradient(prev.getPrevOutput(), false) ||
      !prev.CalcHiddenGradients(cInitFlow.AsObject()) ||
      !SumAndNormalize(temp, prev.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !prev.SetGradient(temp, false))
      ReturnFalse;
   curr = prev;
//---
   for(int i = cEncoder.Total() - 2; i >= 0; i--)
     {
      prev = cEncoder[i];
      if(!prev ||
         !prev.CalcHiddenGradients(curr))
         ReturnFalse;
      curr = cDecoder[i * 2];
      uint dimension = ((CNeuronSpikeSuperKernelBlock*)prev).GetWindow();
      uint units = ((CNeuronSpikeSuperKernelBlock*)prev).GetUnits();
      if(!curr ||
         !DeConcat(curr.getPrevOutput(), prev.getPrevOutput(), curr.getGradient(), dimension, dimension, units))
         ReturnFalse;
      if(prev.Activation() != None)
         if(!DeActivation(prev.getOutput(), prev.getPrevOutput(), prev.getPrevOutput(), prev.Activation()))
            ReturnFalse;
      if(!SumAndNormalize(prev.getGradient(), prev.getPrevOutput(), prev.getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
      curr = prev;
     }
//---
   for(int i = cPrepare.Total() - 1; i >= 0; i--)
     {
      prev = cPrepare[i];
      if(!prev ||
         !prev.CalcHiddenGradients(curr))
         ReturnFalse;
      curr = prev;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(curr))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
//---
   for(int i = 0; i < cPrepare.Total(); i++)
     {
      curr = cPrepare[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   for(int i = 0; i < cEncoder.Total(); i++)
     {
      curr = cEncoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
   if(!cInitFlow.UpdateInputWeights(prev))
      ReturnFalse;
   CNeuronBaseOCL* flow = cInitFlow.AsObject();
   for(int i = cDecoder.Total() - 1; i >= 0; i -= 2)
     {
      curr = cDecoder[i];
      if(!curr ||
         !curr.UpdateInputWeights(flow, prev.getOutput()))
         ReturnFalse;
      flow = curr;
      if(i <= 0)
         continue;
      prev = cDecoder[i - 1];
     }
   if(!CNeuronSpikeSMR::updateInputWeights(flow, prev.getOutput()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::Save(const int file_handle)
  {
   if(!CNeuronSpikeSMR::Save(file_handle))
      ReturnFalse;
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cEncoder.Save(file_handle))
      ReturnFalse;
   if(!cDecoder.Save(file_handle))
      ReturnFalse;
   if(!cInitFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikeSMR::Load(file_handle))
      ReturnFalse;
   if(!cPrepare.Load(file_handle))
      ReturnFalse;
   if(!cEncoder.Load(file_handle))
      ReturnFalse;
   if(!cDecoder.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInitFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeSMR::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronEVAFlow* Source = source;
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cEncoder, Source, tau);
   dWeightsUpdate(cDecoder, Source, tau);
   dWeightsUpdate(cInitFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEVAFlow::Clear(void)
  {
   if(!CNeuronSpikeSMR::Clear())
      ReturnFalse;
   if(!cPrepare.ClearStates())
      ReturnFalse;
   if(!cEncoder.ClearStates())
      ReturnFalse;
   if(!cDecoder.ClearStates())
      ReturnFalse;
   if(!cInitFlow.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEVAFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeSMR::SetOpenCL(obj);
   cPrepare.SetOpenCL(OpenCL);
   cEncoder.SetOpenCL(OpenCL);
   cDecoder.SetOpenCL(OpenCL);
   cInitFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEVAFlow::TrainMode(bool flag)
  {
   CNeuronSpikeSMR::TrainMode(flag);
   cPrepare.TrainMode(bTrain);
   cEncoder.TrainMode(bTrain);
   cDecoder.TrainMode(bTrain);
   cInitFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint units, uint mf_dimension, uint context_size,
                             uint inside_dimension, uint chanels_out,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMSRes::Init(numOutputs, myIndex, open_cl, units, chanels_out, 1, optimization_type, batch))
      ReturnFalse;
//---
   cContextProjection.Clear();
   cMFProjection.Clear();
   cContextProjection.SetOpenCL(OpenCL);
   cMFProjection.SetOpenCL(OpenCL);
//---
   CNeuronBaseOCL*               neuron = NULL;
   CNeuronSpikeActivation*       activat = NULL;
   CNeuronBatchNormOCL*          norm = NULL;
   CNeuronSpikeSuperKernelBlock* block = NULL;
//--- Context Projection
   uint index = 0;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, context_size, optimization, iBatch) ||
      !cContextProjection.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   activat = new CNeuronSpikeActivation();
   if(!activat ||
      !activat.Init(inside_dimension, index, OpenCL, neuron.Neurons(), optimization, iBatch) ||
      !cContextProjection.Add(activat))
      DeleteObjAndFalse(activat);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, inside_dimension, optimization, iBatch) ||
      !cContextProjection.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, neuron.Neurons(), iBatch, optimization) ||
      !cContextProjection.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
   index++;
//--- Motion Feature
   uint windows[] = {3, 5};
   uint steps[] = {1, 1};
   uint chan_in = mf_dimension;
   for(int i = 0; i < 3; i++)
     {
      block = new CNeuronSpikeSuperKernelBlock();
      if(!block ||
         !block.Init(0, index, OpenCL, chan_in, inside_dimension, windows, steps, units, 1, optimization, iBatch) ||
         !cMFProjection.Add(block))
         DeleteObjAndFalse(block);
      index++;
      chan_in = inside_dimension;
     }
//---
   if(!cContextVsMF.Init(0, index, OpenCL, block.Neurons(), optimization, iBatch))
      ReturnFalse;
   cContextVsMF.SetActivationFunction(None);
   index++;
   if(!cGRU.Init(0, index, OpenCL, units, inside_dimension, chanels_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* context = cContextProjection[0];
   CNeuronBaseOCL* mf = NULL;
//---
   if(!context.SetOutput(SecondInput, true))
      ReturnFalse;
//---
   for(int i = 0; i < cMFProjection.Total(); i++)
     {
      mf = cMFProjection[i];
      if(!mf || !mf.FeedForward(prev))
         ReturnFalse;
      prev = mf;
     }
//---
   prev = context;
   for(int i = 1; i < cContextProjection.Total(); i++)
     {
      context = cContextProjection[i];
      if(!context ||
         !context.FeedForward(prev))
         ReturnFalse;
      prev = context;
     }
//---
   if(!SumVecMatrix(context.getOutput(), mf.getOutput(), cContextVsMF.getOutput(), context.Neurons(), 1, 0, 0, 0, 1))
      ReturnFalse;
   if(!cGRU.FeedForward(cContextVsMF.AsObject()))
      ReturnFalse;
   if(!CNeuronMSRes::feedForward(cGRU.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
      CBufferFloat *SecondInput,
      CBufferFloat *SecondGradient,
      ENUM_ACTIVATION SecondActivation = None)
  {
   if(!NeuronOCL || !SecondGradient)
      ReturnFalse;
   if(!cContextProjection[0] ||
      !cContextProjection[0].SetGradient(SecondGradient, true))
      ReturnFalse;
   cContextProjection[0].SetActivationFunction(SecondActivation);
//---
   if(!CNeuronMSRes::calcInputGradients(cGRU.AsObject()))
      ReturnFalse;
   if(!cContextVsMF.CalcHiddenGradients(cGRU.AsObject()))
      ReturnFalse;
   CNeuronBaseOCL* context = cContextProjection[-1];
   CNeuronBaseOCL* mf = cMFProjection[-1];
   if(!context || !mf ||
      !SumVecMatrixGrad(context.getGradient(), mf.getGradient(), cContextVsMF.getGradient(), context.Neurons(), 1, 0, 0, 0, 1))
      ReturnFalse;
//---
   for(int i = cContextProjection.Total() - 2; i >= 0; i--)
     {
      context = cContextProjection[i];
      if(!context ||
         !context.CalcHiddenGradients(cContextProjection[i + 1]))
         ReturnFalse;
     }
//---
   for(int i = cMFProjection.Total() - 2; i >= 0; i--)
     {
      mf = cMFProjection[i];
      if(!mf ||
         !mf.CalcHiddenGradients(cMFProjection[i + 1]))
         ReturnFalse;
     }
//---
   if(!NeuronOCL.CalcHiddenGradients(mf))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)
  {
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* context = cContextProjection[0];
   CNeuronBaseOCL* mf = NULL;
//---
   for(int i = 0; i < cMFProjection.Total(); i++)
     {
      mf = cMFProjection[i];
      if(!mf || !mf.UpdateInputWeights(prev))
         ReturnFalse;
      prev = mf;
     }
//---
   prev = context;
   for(int i = 1; i < cContextProjection.Total(); i++)
     {
      context = cContextProjection[i];
      if(!context ||
         !context.UpdateInputWeights(prev))
         ReturnFalse;
      prev = context;
     }
//---
   if(!cGRU.UpdateInputWeights(cContextVsMF.AsObject()))
      ReturnFalse;
   if(!CNeuronMSRes::updateInputWeights(cGRU.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::Save(const int file_handle)
  {
   if(!CNeuronMSRes::Save(file_handle))
      ReturnFalse;
//---
   if(!cMFProjection.Save(file_handle))
      ReturnFalse;
   if(!cContextProjection.Save(file_handle))
      ReturnFalse;
   if(!cGRU.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::Load(const int file_handle)
  {
   if(!CNeuronMSRes::Load(file_handle))
      ReturnFalse;
//---
   if(!cMFProjection.Load(file_handle))
      ReturnFalse;
   if(!cContextProjection.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cGRU.AsObject()))
      ReturnFalse;
//---
   if(!cMFProjection[-1] ||
      !cContextVsMF.Init(0, cMFProjection.Total() + cContextProjection.Total(), OpenCL, cMFProjection[-1].Neurons(), optimization, iBatch))
      ReturnFalse;
   cContextVsMF.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronMSRes::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronResFlowHTR* Source = source;
   dWeightsUpdate(cMFProjection, Source, tau);
   dWeightsUpdate(cContextProjection, Source, tau);
   dWeightsUpdate(cGRU, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowHTR::Clear(void)
  {
   if(!CNeuronMSRes::Clear())
      ReturnFalse;
//---
   if(!cMFProjection.ClearStates())
      ReturnFalse;
   if(!cContextProjection.ClearStates())
      ReturnFalse;
   if(!cGRU.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlowHTR::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronMSRes::SetOpenCL(obj);
//---
   cMFProjection.SetOpenCL(OpenCL);
   cContextProjection.SetOpenCL(OpenCL);
   cGRU.SetOpenCL(OpenCL);
   cContextVsMF.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlowHTR::TrainMode(bool flag)
  {
   CNeuronMSRes::TrainMode(flag);
//---
   cMFProjection.TrainMode(bTrain);
   cContextProjection.TrainMode(bTrain);
   cGRU.TrainMode(bTrain);
   cContextVsMF.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint hidden_dimension,
                          uint ltr_forecast, uint htr_points_to_segment,
                          uint ltr_layers, uint ltr_candidates,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
     {
      uint windows[] = {3, 5};
      uint steps[] = {1, 1};
      if(!CNeuronSpikeSuperKernelBlock::Init(numOutputs, myIndex, open_cl, hidden_dimension, dimension, windows,
                                             steps, ltr_forecast * htr_points_to_segment, 1, optimization_type, batch))
         ReturnFalse;
     }
//---
   uint index = 0;
   if(!cPrepare.Init(0, index, OpenCL, 2 * htr_points_to_segment, dimension, hidden_dimension, 1, optimization, iBatch))
      ReturnFalse;
//--- Context
   cContext.Clear();
   cContext.SetOpenCL(OpenCL);
   CNeuronBaseOCL* neuron = NULL;
   CNeuronSpikeSuperKernelBlock* block = NULL;
   CNeuronSpikeActivation* activ = NULL;
   CNeuronBatchNormOCL* norm = NULL;
   index++;
     {
      uint windows[] = {3, 5, 7};
      uint steps[] = {2, 2, 2};
      block = new CNeuronSpikeSuperKernelBlock();
      if(!block ||
         !block.Init(0, index, OpenCL, hidden_dimension, hidden_dimension, windows, steps,
                     (htr_points_to_segment + 3) / 4, 1, optimization, iBatch) ||
         !cContext.Add(block))
         DeleteObjAndFalse(block);
     }
   index++;
   activ = new CNeuronSpikeActivation();
   if(!activ ||
      !activ.Init(hidden_dimension, index, OpenCL, block.Neurons(), optimization, iBatch) ||
      !cContext.Add(activ))
      DeleteObjAndFalse(activ);
   index++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, index, OpenCL, hidden_dimension, optimization, iBatch) ||
      !cContext.Add(neuron))
      DeleteObjAndFalse(neuron);
   neuron.SetActivationFunction(None);
   index++;
   norm = new CNeuronBatchNormOCL();
   if(!norm ||
      !norm.Init(0, index, OpenCL, neuron.Neurons(), iBatch, optimization) ||
      !cContext.Add(norm))
      DeleteObjAndFalse(norm);
   norm.SetActivationFunction(None);
//---
   index++;
   if(!cLTR.Init(0, index, OpenCL, ltr_forecast, cContext[-1].Neurons(), hidden_dimension,
                 ltr_layers, ltr_candidates, optimization, iBatch))
      ReturnFalse;
//---
   cMFE.Clear();
   cMFE.SetOpenCL(OpenCL);
   index++;
     {
      uint windows[] = {3, 5};
      uint steps[] = {1, 1};
      block = new CNeuronSpikeSuperKernelBlock();
      if(!block ||
         !block.Init(0, index, OpenCL, hidden_dimension, htr_points_to_segment * hidden_dimension,
                     windows, steps, ltr_forecast, 1, optimization, iBatch) ||
         !cMFE.Add(block))
         DeleteObjAndFalse(block);
     }
//---
   index++;
   if(!cUpsampleLTR.Init(0, index, OpenCL, cLTR.Neurons()*htr_points_to_segment, optimization, iBatch))
      ReturnFalse;
   cUpsampleLTR.SetActivationFunction(None);
   index++;
   if(!cMFEvsUpLTR.Init(0, index, OpenCL, cUpsampleLTR.Neurons() + cMFE[-1].Neurons(), optimization, iBatch))
      ReturnFalse;
   cMFEvsUpLTR.SetActivationFunction(None);
   index++;
   if(!cHTR.Init(0, index, OpenCL, ltr_forecast * htr_points_to_segment, 2 * hidden_dimension, hidden_dimension,
                 hidden_dimension, hidden_dimension, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cUpdatedFlow.Init(0, index, OpenCL, cHTR.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cPrepare.AsObject();
   CNeuronBaseOCL*curr = NULL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      curr = cContext[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cLTR.FeedForward(prev))
      ReturnFalse;
//---
   uint hidden_dimension = cLTR.GetWindow();
   uint ltr_forecat = cLTR.GetUnits();
   uint htr_points_to_segment = cUpsampleLTR.Neurons() / cLTR.Neurons();
   if(!LinearUpsample(cLTR.AsObject(), cUpsampleLTR.AsObject(), htr_points_to_segment,
                      ltr_forecat, hidden_dimension))
      ReturnFalse;
//---
   prev = cUpsampleLTR.AsObject();
   for(int i = 0; i < cMFE.Total(); i++)
     {
      curr = cMFE[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!Concat(prev.getOutput(), cUpsampleLTR.getOutput(), cMFEvsUpLTR.getOutput(), hidden_dimension,
              hidden_dimension, ltr_forecat * htr_points_to_segment))
      ReturnFalse;
   if(!cHTR.FeedForward(cMFEvsUpLTR.AsObject(), cContext[-1].getOutput()))
      ReturnFalse;
//---
   if(!SumAndNormalize(cUpsampleLTR.getOutput(), cHTR.getOutput(), cUpdatedFlow.getOutput(), hidden_dimension, false, 0, 0, 0, 1) ||
      !SumVecMatrix(cPrepare.getOutput(), cUpdatedFlow.getOutput(), cUpdatedFlow.getOutput(), hidden_dimension, 1, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronSpikeSuperKernelBlock::feedForward(cUpdatedFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeSuperKernelBlock::calcInputGradients(cUpdatedFlow.AsObject()))
      ReturnFalse;
   if(!DeActivation(cHTR.getOutput(), cHTR.getGradient(), cUpdatedFlow.getGradient(), cHTR.Activation()))
      ReturnFalse;
   if(!cContext[-1] ||
      !cMFEvsUpLTR.CalcHiddenGradients(cHTR.AsObject(), cContext[-1].getOutput(),
                                       cContext[-1].getGradient(),
                                       (ENUM_ACTIVATION)cContext[-1].Activation()))
      ReturnFalse;
//---
   uint hidden_dimension = cLTR.GetWindow();
   uint ltr_forecat = cLTR.GetUnits();
   uint htr_points_to_segment = cUpsampleLTR.Neurons() / cLTR.Neurons();
//---
   if(!cMFE[-1] ||
      !DeConcat(cMFE[-1].getGradient(), cUpsampleLTR.getGradient(), cMFEvsUpLTR.getGradient(),
                hidden_dimension, hidden_dimension, ltr_forecat * htr_points_to_segment))
      ReturnFalse;
   for(int i = cMFE.Total() - 2; i >= 0; i--)
     {
      if(!cMFE[i] ||
         !cMFE[i].CalcHiddenGradients(cMFE[i + 1]))
         ReturnFalse;
     }
   if(!cUpsampleLTR.CalcHiddenGradients(cMFE[0]))
      ReturnFalse;
//---
   if(!SumAndNormalize(cUpsampleLTR.getOutput(), cHTR.getOutput(), cUpsampleLTR.getGradient(),
                       hidden_dimension, false, 0, 0, 0, 1) ||
      !SumAndNormalize(cUpsampleLTR.getGradient(), cUpdatedFlow.getGradient(), cUpsampleLTR.getGradient(),
                       hidden_dimension, false, 0, 0, 0, 1))
      ReturnFalse;
   float error = 0;
   if(!cUpsampleLTR.calcOutputGradients(cUpsampleLTR.getGradient(), error))
      ReturnFalse;
   if(!LinearUpsampleGrad(cLTR.AsObject(), cUpsampleLTR.AsObject(), htr_points_to_segment,
                          ltr_forecat, hidden_dimension))
      ReturnFalse;
   Deactivation(cLTR);
   CNeuronBaseOCL* next = cLTR.AsObject();
   for(int i = cContext.Total() - 1; i >= 0; i--)
     {
      if(!cContext[i] ||
         !cContext[i].CalcHiddenGradients(next))
         ReturnFalse;
      next = cContext[i];
     }
//---
   if(!cPrepare.CalcHiddenGradients(next))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cPrepare.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cPrepare.AsObject();
   CNeuronBaseOCL*curr = NULL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      curr = cContext[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cLTR.UpdateInputWeights(prev))
      ReturnFalse;
//---
   prev = cUpsampleLTR.AsObject();
   for(int i = 0; i < cMFE.Total(); i++)
     {
      curr = cMFE[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cHTR.UpdateInputWeights(cMFEvsUpLTR.AsObject(), cContext[-1].getOutput()))
      ReturnFalse;
//---
   if(!CNeuronSpikeSuperKernelBlock::updateInputWeights(cUpdatedFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::Save(const int file_handle)
  {
   if(!CNeuronSpikeSuperKernelBlock::Save(file_handle))
      ReturnFalse;
//---
   if(!cPrepare.Save(file_handle))
      ReturnFalse;
   if(!cContext.Save(file_handle))
      ReturnFalse;
   if(!cLTR.Save(file_handle))
      ReturnFalse;
   if(!cMFE.Save(file_handle))
      ReturnFalse;
   if(!cHTR.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::Load(const int file_handle)
  {
   if(!CNeuronSpikeSuperKernelBlock::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cPrepare.AsObject()))
      ReturnFalse;
   if(!cContext.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cLTR.AsObject()))
      ReturnFalse;
   if(!cMFE.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHTR.AsObject()))
      ReturnFalse;
//---
   uint hidden_dimension = cLTR.GetWindow();
   uint ltr_forecat = cLTR.GetUnits();
   uint htr_points_to_segment = cHTR.Neurons() / cLTR.Neurons();
//---
   uint index = 2 + cContext.Total() + cMFE.Total();
   if(!cUpsampleLTR.Init(0, index, OpenCL, cHTR.Neurons(), optimization, iBatch))
      ReturnFalse;
   cUpsampleLTR.SetActivationFunction(None);
   index++;
   if(!cMFEvsUpLTR.Init(0, index, OpenCL, cUpsampleLTR.Neurons() + cMFE[-1].Neurons(), optimization, iBatch))
      ReturnFalse;
   cMFEvsUpLTR.SetActivationFunction(None);
   index += 2;
   if(!cUpdatedFlow.Init(0, index, OpenCL, cHTR.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronSpikeSuperKernelBlock::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronResFlow* Source = source;
   dWeightsUpdate(cPrepare, Source, tau);
   dWeightsUpdate(cContext, Source, tau);
   dWeightsUpdate(cLTR, Source, tau);
   dWeightsUpdate(cMFE, Source, tau);
   dWeightsUpdate(cHTR, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlow::Clear(void)
  {
   if(!CNeuronSpikeSuperKernelBlock::Clear())
      ReturnFalse;
//---
   if(!cPrepare.Clear())
      ReturnFalse;
   if(!cContext.ClearStates())
      ReturnFalse;
   if(!cLTR.Clear())
      ReturnFalse;
   if(!cMFE.ClearStates())
      ReturnFalse;
   if(!cHTR.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlow::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronSpikeSuperKernelBlock::SetOpenCL(obj);
//---
   cPrepare.SetOpenCL(OpenCL);
   cContext.SetOpenCL(OpenCL);
   cLTR.SetOpenCL(OpenCL);
   cMFE.SetOpenCL(OpenCL);
   cHTR.SetOpenCL(OpenCL);
   cUpsampleLTR.SetOpenCL(OpenCL);
   cMFEvsUpLTR.SetOpenCL(OpenCL);
   cUpdatedFlow.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlow::TrainMode(bool flag)
  {
   CNeuronSpikeSuperKernelBlock::TrainMode(flag);
//---
   cPrepare.TrainMode(bTrain);
   cContext.TrainMode(bTrain);
   cLTR.TrainMode(bTrain);
   cMFE.TrainMode(bTrain);
   cHTR.TrainMode(bTrain);
   cUpsampleLTR.TrainMode(bTrain);
   cMFEvsUpLTR.TrainMode(bTrain);
   cUpdatedFlow.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint dimension, uint hidden_dimension, uint ltr_forecast,
                                 uint htr_points_to_segment, uint ltr_layers, uint ltr_candidates,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronResFlow::Init(numOutputs, myIndex, open_cl, dimension, hidden_dimension,
                            ltr_forecast, htr_points_to_segment, ltr_layers,
                            ltr_candidates, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cTailAwareMoE.Init(0, index, OpenCL, hidden_dimension, ltr_forecast,
                          GetUnits(), ltr_candidates, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cConcat.Init(0, index, OpenCL, 4 * hidden_dimension * GetUnits(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
   index++;
   uint windows[] = {3, 5};
   uint steps[] = {1, 1};
   if(!cAdaptFlow.Init(0, index, OpenCL, 4 * hidden_dimension, hidden_dimension,
                       windows, steps, GetUnits(), 1, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cPrepare.AsObject();
   CNeuronBaseOCL*curr = NULL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      curr = cContext[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cLTR.FeedForward(prev))
      ReturnFalse;
//---
   uint hidden_dimension = cLTR.GetWindow();
   uint ltr_forecat = cLTR.GetUnits();
   uint htr_points_to_segment = cUpsampleLTR.Neurons() / cLTR.Neurons();
   if(!LinearUpsample(cLTR.AsObject(), cUpsampleLTR.AsObject(), htr_points_to_segment,
                      ltr_forecat, hidden_dimension))
      ReturnFalse;
//---
   prev = cUpsampleLTR.AsObject();
   for(int i = 0; i < cMFE.Total(); i++)
     {
      curr = cMFE[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!Concat(prev.getOutput(), cUpsampleLTR.getOutput(), cMFEvsUpLTR.getOutput(), hidden_dimension,
              hidden_dimension, GetUnits()))
      ReturnFalse;
   if(!cHTR.FeedForward(cMFEvsUpLTR.AsObject(), cContext[-1].getOutput()))
      ReturnFalse;
//---
   if(!cTailAwareMoE.FeedForward(cLTR.AsObject()))
      ReturnFalse;
   if(!Concat(cMFEvsUpLTR.getOutput(), cHTR.getOutput(), cTailAwareMoE.getOutput(),
              cConcat.getOutput(), 2 * hidden_dimension, hidden_dimension, hidden_dimension, GetUnits()))
      ReturnFalse;
   if(!cAdaptFlow.FeedForward(cConcat.AsObject()))
      ReturnFalse;
//---
   if(!SumVecMatrix(cPrepare.getOutput(), cAdaptFlow.getOutput(), cUpdatedFlow.getOutput(),
                    hidden_dimension, 1, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronSpikeSuperKernelBlock::feedForward(cUpdatedFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSpikeSuperKernelBlock::calcInputGradients(cUpdatedFlow.AsObject()))
      ReturnFalse;
   if(!DeActivation(cAdaptFlow.getOutput(), cAdaptFlow.getGradient(),
                    cUpdatedFlow.getGradient(), cAdaptFlow.Activation()))
      ReturnFalse;
   if(!cConcat.CalcHiddenGradients(cAdaptFlow.AsObject()))
      ReturnFalse;
//---Tail-Aware
   if(!DeActivation(cTailAwareMoE.getOutput(), cTailAwareMoE.getGradient(),
                    cUpdatedFlow.getGradient(), cTailAwareMoE.Activation()))
      ReturnFalse;
   if(!cLTR.CalcHiddenGradients(cTailAwareMoE.AsObject()))
      ReturnFalse;
//--- HTR
   if(!DeActivation(cHTR.getOutput(), cHTR.getGradient(), cUpdatedFlow.getGradient(), cHTR.Activation()))
      ReturnFalse;
   if(!cContext[-1] ||
      !cMFEvsUpLTR.CalcHiddenGradients(cHTR.AsObject(), cContext[-1].getOutput(),
                                       cContext[-1].getGradient(),
                                       (ENUM_ACTIVATION)cContext[-1].Activation()))
      ReturnFalse;
//--- MFE
   uint hidden_dimension = cLTR.GetWindow();
   uint ltr_forecat = cLTR.GetUnits();
   uint htr_points_to_segment = cUpsampleLTR.Neurons() / cLTR.Neurons();
//---
   if(!cMFE[-1] ||
      !DeConcat(cMFE[-1].getGradient(), cUpsampleLTR.getGradient(), cMFEvsUpLTR.getGradient(),
                hidden_dimension, hidden_dimension, ltr_forecat * htr_points_to_segment))
      ReturnFalse;
   for(int i = cMFE.Total() - 2; i >= 0; i--)
     {
      if(!cMFE[i] ||
         !cMFE[i].CalcHiddenGradients(cMFE[i + 1]))
         ReturnFalse;
     }
   if(!cUpsampleLTR.CalcHiddenGradients(cMFE[0]))
      ReturnFalse;
//--- LTR
   if(!DeActivation(cUpsampleLTR.getOutput(), cUpsampleLTR.getGradient(),
                    cUpdatedFlow.getGradient(), cUpsampleLTR.Activation()))
      ReturnFalse;
   if(!LinearUpsampleGrad(cLTR.AsObject(), cUpsampleLTR.AsObject(), htr_points_to_segment,
                          ltr_forecat, hidden_dimension))
      ReturnFalse;
   Deactivation(cLTR);
   CNeuronBaseOCL* next = cLTR.AsObject();
   for(int i = cContext.Total() - 1; i >= 0; i--)
     {
      if(!cContext[i] ||
         !cContext[i].CalcHiddenGradients(next))
         ReturnFalse;
      next = cContext[i];
     }
//---
   if(!cPrepare.CalcHiddenGradients(next))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cPrepare.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cPrepare.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   CNeuronBaseOCL* prev = cPrepare.AsObject();
   CNeuronBaseOCL*curr = NULL;
   for(int i = 0; i < cContext.Total(); i++)
     {
      curr = cContext[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cLTR.UpdateInputWeights(prev))
      ReturnFalse;
//---
   prev = cUpsampleLTR.AsObject();
   for(int i = 0; i < cMFE.Total(); i++)
     {
      curr = cMFE[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   if(!cHTR.UpdateInputWeights(cMFEvsUpLTR.AsObject(), cContext[-1].getOutput()))
      ReturnFalse;
//---
   if(!cTailAwareMoE.UpdateInputWeights(cLTR.AsObject()))
      ReturnFalse;
   if(!cAdaptFlow.UpdateInputWeights(cConcat.AsObject()))
      ReturnFalse;
//---
   if(!CNeuronSpikeSuperKernelBlock::updateInputWeights(cUpdatedFlow.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::Save(const int file_handle)
  {
   if(!CNeuronResFlow::Save(file_handle))
      ReturnFalse;
   if(!cTailAwareMoE.Save(file_handle))
      ReturnFalse;
   if(!cAdaptFlow.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::Load(const int file_handle)
  {
   if(!CNeuronResFlow::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTailAwareMoE.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cAdaptFlow.AsObject()))
      ReturnFalse;
//---
   if(!cConcat.Init(0, 1, OpenCL, 4 * cLTR.GetWindow() * GetUnits(), optimization, iBatch))
      ReturnFalse;
   cConcat.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::Clear(void)
  {
   if(!CNeuronResFlow::Clear())
      ReturnFalse;
   if(!cTailAwareMoE.Clear())
      ReturnFalse;
   if(!cAdaptFlow.Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResFlowLattice::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronResFlow::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronResFlowLattice* Source = source;
   dWeightsUpdate(cTailAwareMoE, Source, tau);
   dWeightsUpdate(cAdaptFlow, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlowLattice::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronResFlow::SetOpenCL(obj);
   cTailAwareMoE.SetOpenCL(OpenCL);
   cAdaptFlow.SetOpenCL(OpenCL);
   cConcat.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResFlowLattice::TrainMode(bool flag)
  {
   CNeuronResFlow::TrainMode(flag);
   cTailAwareMoE.TrainMode(bTrain);
   cAdaptFlow.TrainMode(bTrain);
   cConcat.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+

