//+------------------------------------------------------------------+
//|                                                   NetGenetic.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define MaxMutation     0.5f
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNetGenetic : public CNet
  {
protected:
   uint              i_PopulationSize;
   vectorf           v_Probability;
   vectorf           v_Rewards;
   matrixf           m_Weights;
   matrixf           m_WeightsConv;

   //---
   virtual bool      CreatePopulation(void);
   virtual int       GetAction(CBufferFloat * probability);
   virtual bool      GetWeights(uint layer);
   float             NextGenerationWeight(matrixf &array, uint shift, vectorf &probability);
   float             GenerateWeight(uint total);

public:
                     CNetGenetic();
                    ~CNetGenetic();
   //---
   virtual bool              Create(CArrayObj *Description, uint population_size);
   virtual bool              SetPopulationSize(uint size);
   virtual bool              feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true);
   virtual bool              Rewards(CArrayFloat *rewards);
   virtual bool              NextGeneration(float quantile, float mutation, float &average, float &mamximum);
   virtual bool              Load(string file_name, uint population_size, bool common = true);
   virtual bool              SaveModel(string file_name, int model, bool common = true);
   //---
   virtual bool              CopyModel(CArrayLayer *source, uint model);
   virtual bool              Detach(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNetGenetic::CNetGenetic() :  i_PopulationSize(100)
  {
   SetPopulationSize(i_PopulationSize);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNetGenetic::~CNetGenetic()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::Create(CArrayObj *Description, uint population_size)
  {
   if(CheckPointer(Description) == POINTER_INVALID)
      return false;
//---
   if(!SetPopulationSize(population_size))
      return false;
   CNet::Create(Description);
   return CreatePopulation();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::SetPopulationSize(uint size)
  {
   i_PopulationSize = size;
   v_Probability = vectorf::Zeros(i_PopulationSize);
   v_Rewards = vectorf::Zeros(i_PopulationSize);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::Load(string file_name, uint population_size, bool common = true)
  {
   float temp;
   datetime dt;
   if(!CNet::Load(file_name, temp, temp, temp, dt, common))
      return false;
//---
   if(!SetPopulationSize(population_size))
      return false;
   return CreatePopulation();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::CreatePopulation(void)
  {
   if(!layers || layers.Total() < 2)
      return false;
//---
   CLayer *layer = layers.At(0);
   if(!layer || !layer.At(0))
      return false;
//---
   CNeuronBaseOCL *neuron_ocl = layer.At(0);
   int prev_count = neuron_ocl.Neurons();
   for(int i = 1; i < layers.Total(); i++)
     {
      layer = layers.At(i);
      if(!layer || !layer.At(0))
         return false;
      //---
      neuron_ocl = layer.At(0);
      CLayerDescription *desc = neuron_ocl.GetLayerInfo();
      int outputs = neuron_ocl.getConnections();
      //---
      for(uint n = layer.Total(); n < i_PopulationSize; n++)
        {
         CNeuronConvOCL *neuron_conv_ocl = NULL;
         CNeuronProofOCL *neuron_proof_ocl = NULL;
         CNeuronAttentionOCL *neuron_attention_ocl = NULL;
         CNeuronMLMHAttentionOCL *neuron_mlattention_ocl = NULL;
         CNeuronDropoutOCL *dropout = NULL;
         CNeuronBatchNormOCL *batch = NULL;
         CVAE *vae = NULL;
         CNeuronLSTMOCL *lstm = NULL;
         switch(layer.At(0).Type())
           {
            case defNeuron:
            case defNeuronBaseOCL:
               neuron_ocl = new CNeuronBaseOCL();
               if(CheckPointer(neuron_ocl) == POINTER_INVALID)
                  return false;
               if(!neuron_ocl.Init(outputs, n, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_ocl;
                  return false;
                 }
               neuron_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_ocl))
                 {
                  delete neuron_ocl;
                  return false;
                 }
               neuron_ocl = NULL;
               break;
            //---
            case defNeuronConvOCL:
               neuron_conv_ocl = new CNeuronConvOCL();
               if(CheckPointer(neuron_conv_ocl) == POINTER_INVALID)
                  return false;
               if(!neuron_conv_ocl.Init(outputs, n, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_conv_ocl;
                  return false;
                 }
               neuron_conv_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_conv_ocl))
                 {
                  delete neuron_conv_ocl;
                  return false;
                 }
               neuron_conv_ocl = NULL;
               break;
            //---
            case defNeuronProofOCL:
               neuron_proof_ocl = new CNeuronProofOCL();
               if(!neuron_proof_ocl)
                  return false;
               if(!neuron_proof_ocl.Init(outputs, n, opencl, desc.window, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_proof_ocl;
                  return false;
                 }
               neuron_proof_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_proof_ocl))
                 {
                  delete neuron_proof_ocl;
                  return false;
                 }
               neuron_proof_ocl = NULL;
               break;
            //---
            case defNeuronAttentionOCL:
               neuron_attention_ocl = new CNeuronAttentionOCL();
               if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
                  return false;
               if(!neuron_attention_ocl.Init(outputs, n, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_attention_ocl;
                  return false;
                 }
               neuron_attention_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_attention_ocl))
                 {
                  delete neuron_attention_ocl;
                  return false;
                 }
               neuron_attention_ocl = NULL;
               break;
            //---
            case defNeuronMHAttentionOCL:
               neuron_attention_ocl = new CNeuronMHAttentionOCL();
               if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
                  return false;
               if(!neuron_attention_ocl.Init(outputs, n, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_attention_ocl;
                  return false;
                 }
               neuron_attention_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_attention_ocl))
                 {
                  delete neuron_attention_ocl;
                  return false;
                 }
               neuron_attention_ocl = NULL;
               break;
            //---
            case defNeuronMLMHAttentionOCL:
               neuron_mlattention_ocl = new CNeuronMLMHAttentionOCL();
               if(CheckPointer(neuron_mlattention_ocl) == POINTER_INVALID)
                  return false;
               if(!neuron_mlattention_ocl.Init(outputs, n, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete neuron_mlattention_ocl;
                  return false;
                 }
               neuron_mlattention_ocl.SetActivationFunction(desc.activation);
               if(!layer.Add(neuron_mlattention_ocl))
                 {
                  delete neuron_mlattention_ocl;
                  return false;
                 }
               neuron_mlattention_ocl = NULL;
               break;
            //---
            case defNeuronDropoutOCL:
               dropout = new CNeuronDropoutOCL();
               if(CheckPointer(dropout) == POINTER_INVALID)
                  return false;
               if(!dropout.Init(outputs, n, opencl, desc.count, desc.probability, desc.optimization, desc.batch))
                 {
                  delete dropout;
                  return false;
                 }
               if(!layer.Add(dropout))
                 {
                  delete dropout;
                  return false;
                 }
               dropout = NULL;
               break;
            //---
            case defNeuronBatchNormOCL:
               batch = new CNeuronBatchNormOCL();
               if(CheckPointer(batch) == POINTER_INVALID)
                  return false;
               if(!batch.Init(outputs, n, opencl, desc.count, desc.batch, desc.optimization))
                 {
                  delete batch;
                  return false;
                 }
               batch.SetActivationFunction(desc.activation);
               if(!layer.Add(batch))
                 {
                  delete batch;
                  return false;
                 }
               batch = NULL;
               break;
            //---
            case defNeuronVAEOCL:
               vae = new CVAE();
               if(!vae)
                  return false;
               if(!vae.Init(outputs, n, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  delete vae;
                  return false;
                 }
               if(!layer.Add(vae))
                 {
                  delete vae;
                  return false;
                 }
               vae = NULL;
               break;
            case defNeuronLSTMOCL:
               lstm = new CNeuronLSTMOCL();
               if(!lstm)
                  return false;
               if(!lstm.Init(outputs, n, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  delete lstm;
                  return false;
                 }
               if(!lstm.SetInputs(prev_count))
                 {
                  delete lstm;
                  return false;
                 }
               if(!layer.Add(lstm))
                 {
                  delete lstm;
                  return false;
                 }
               lstm = NULL;
               break;
            //---
            case defNeuronSoftMaxOCL:
               neuron_ocl = new CNeuronSoftMaxOCL();
               if(!neuron_ocl)
                  return false;
               if(!neuron_ocl.Init(outputs, n, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_ocl;
                  return false;
                 }
               if(!layer.Add(neuron_ocl))
                 {
                  delete neuron_ocl;
                  return false;
                 }
               neuron_ocl = NULL;
               break;
            //---
            default:
               return false;
               break;
           }
        }
      if(layer.Total() > (int)i_PopulationSize)
         layer.Resize(i_PopulationSize);
      delete desc;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true)
  {
   if(CheckPointer(layers) == POINTER_INVALID || CheckPointer(inputVals) == POINTER_INVALID || layers.Total() <= 1)
      return false;
//---
   CLayer *previous = NULL;
   CLayer *current = layers.At(0);
   int total = MathMin(current.Total(), inputVals.Total());
   CNeuronBase *neuron = NULL;
   if(CheckPointer(opencl) == POINTER_INVALID)
      return false;
   CNeuronBaseOCL *neuron_ocl = current.At(0);
   CBufferFloat *inputs = neuron_ocl.getOutput();
   int total_data = inputVals.Total();
   if(!inputs.Resize(total_data))
      return false;
   for(int d = 0; d < total_data; d++)
     {
      int pos = d;
      int dim = 0;
      if(window > 1)
        {
         dim = d % window;
         pos = (d - dim) / window;
        }
      float value = pos / pow(10000, (2 * dim + 1) / (float)(window + 1));
      value = (float)(tem ? (dim % 2 == 0 ? sin(value) : cos(value)) : 0);
      value += inputVals.At(d);
      if(!inputs.Update(d, value))
         return false;
     }
   if(!inputs.BufferWrite())
      return false;
//---
   CNeuronBaseOCL *current_ocl;
   for(int l = 1; l < layers.Total(); l++)
     {
      previous = current;
      current = layers.At(l);
      if(CheckPointer(current) == POINTER_INVALID)
         return false;
      //---
      for(uint n = 0; n < i_PopulationSize; n++)
        {
         current_ocl = current.At(n);
         if(!current_ocl.FeedForward(previous.At(l == 1 ? 0 : n)))
            return false;
         continue;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::Rewards(CArrayFloat *rewards)
  {
   if(!rewards || !layers || layers.Total() < 2)
      return false;
//---
   CLayer *output = layers.At(layers.Total() - 1);
   if(!output)
      return false;
//---
   for(int i = 0; i < output.Total(); i++)
     {
      CNeuronBaseOCL *neuron = output.At(i);
      if(!neuron)
         continue;
      int action = GetAction(neuron.getOutput());
      v_Rewards[i] += rewards.At(action >= 0 ? action : rewards.Minimum(0, WHOLE_ARRAY));
     }
//---
   v_Probability = v_Rewards - v_Rewards.Min();
   if(!v_Probability.Clip(0, v_Probability.Max()))
      return false;
   v_Probability = v_Probability / v_Probability.Sum();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CNetGenetic::GetAction(CBufferFloat * probability)
  {
   vectorf prob;
   if(!probability.GetData(prob))
      return -1;
   prob = prob.CumSum();
   prob = prob / prob.Max();
   int err_code;
   float random = (float)MathRandomNormal(0.5, 0.5, err_code);
   if(random >= 1)
      return (int)prob.Size() - 1;
   for(int i = 0; i < (int)prob.Size(); i++)
      if(random <= prob[i] && prob[i] > 0)
         return i;
//---
   return -1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::NextGeneration(float quantile, float mutation, float &average, float &maximum)
  {
   maximum = v_Rewards.Max();
   mutation = MathMin(mutation, MaxMutation);
   v_Probability = v_Rewards - v_Rewards.Quantile(quantile);
   if(!v_Probability.Clip(0, v_Probability.Max()))
      return false;
   v_Probability = v_Probability / v_Probability.Sum();
   average = v_Rewards.Average(v_Probability);
//---
   if(!v_Probability.Resize(i_PopulationSize + 1))
      return false;
   v_Probability[i_PopulationSize] = mutation;
   v_Probability = (v_Probability / (1 + mutation)).CumSum();
//---
   for(int l = 1; l < layers.Total(); l++)
     {
      if(!GetWeights(l))
        {
         PrintFormat("Error of load weights from layer %d", l);
         return false;
        }
      CLayer* layer = layers.At(l);
      if(!layer)
         return false;
      for(uint i = 0; i < i_PopulationSize; i++)
        {
         CNeuronBaseOCL* neuron = layer.At(i);
         CBufferFloat* weights = neuron.getWeights();
         if(!!weights)
           {
            for(int w = 0; w < weights.Total(); w++)
               if(!weights.Update(w, NextGenerationWeight(m_Weights, w, v_Probability)))
                 {
                  Print("Error of update weights");
                  return false;
                 }
            weights.BufferWrite();
           }
         if(neuron.Type() != defNeuronConvOCL)
            continue;
         CNeuronConvOCL* temp = neuron;
         weights = temp.GetWeightsConv();
         for(int w = 0; w < weights.Total(); w++)
            if(!weights.Update(w, NextGenerationWeight(m_WeightsConv, w, v_Probability)))
              {
               Print("Error of update weights");
               return false;
              }
         weights.BufferWrite();
        }
     }
   v_Rewards.Fill(0);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNetGenetic::NextGenerationWeight(matrixf &array, uint shift, vectorf &probability)
  {
   int err_code;
   float random = (float)MathRandomNormal(0.5, 0.5, err_code);
   for(int i = 0; i < (int)probability.Size(); i++)
      if(probability[i] > 0 && random <= probability[i] && i < (int)array.Rows())
        {
         return array[i, shift];
        }
//---
   return GenerateWeight((uint)array.Cols());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::GetWeights(uint layer)
  {
   if(!layers)
      return false;
   CLayer *current_layer = layers.At(layer);
   if(!current_layer)
      return false;
   int total = current_layer.Total();
   m_Weights = matrixf::Zeros(total, 0);
   m_WeightsConv = matrixf::Zeros(total, 0);
   for(int m = 0; m < total; m++)
     {
      CNeuronBaseOCL *neuron = current_layer.At(m);
      if(!neuron)
         return false;
      vectorf buffer;
      if(!!neuron.getWeights())
        {
         neuron.getWeights().GetData(buffer);
         if(m_Weights.Cols() <= 0)
            if(!m_Weights.Resize(total, buffer.Size()))
               return false;
         if(!m_Weights.Row(buffer, m))
            continue;
        }
      //---
      if(neuron.Type() != defNeuronConvOCL)
         continue;
      CNeuronConvOCL *temp = neuron;
      if(temp.GetWeightsConv().GetData(buffer) <= 0)
         return false;
      if(m_WeightsConv.Cols() != buffer.Size())
         if(!m_WeightsConv.Resize(total, buffer.Size()))
            return false;
      if(!m_WeightsConv.Row(buffer, m))
         return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::SaveModel(string file_name, int model, bool common = true)
  {
   if(model < 0 || model >= (int)i_PopulationSize)
      model = (int)v_Probability.ArgMax();
//---
   CNetGenetic *new_model = new CNetGenetic();
   if(!new_model)
      return false;
   if(!new_model.CopyModel(layers, model))
     {
      new_model.Detach();
      delete new_model;
      return false;
     }
   bool result = new_model.Save(file_name, 0, 0, 0, 0, common);
   new_model.Detach();
   delete new_model;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::CopyModel(CArrayLayer *source, uint model)
  {
   if(!source)
      return false;
   if(!layers)
     {
      layers = new CArrayLayer();
      if(!layers)
         return false;
     }
   else
      layers.Clear();
   for(int l = 0; l < source.Total(); l++)
     {
      CLayer* source_layer = source.At(l);
      if(!source_layer)
         return false;
      CLayer* new_layer = new CLayer(source_layer.Outputs(), -1, opencl);
      if(!new_layer)
         return false;
      if(!new_layer.Add(source_layer.At(l == 0 ? 0 : model)))
         return false;
      if(!layers.Add(new_layer))
         return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetGenetic::Detach(void)
  {
   if(!layers || layers.Total() <= 0)
      return true;
   int total = layers.Total();
   for(int i = 0; i < total; i++)
     {
      CLayer *layer = layers.At(i);
      if(!layer)
         continue;
      for(int n = layer.Total() - 1; n >= 0; n--)
         CObject *temp = layer.Detach(n);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNetGenetic::GenerateWeight(uint total)
  {
   xor128;
   float result = (float)rnd_w / UINT_MAX;
   if(result == 0)
      return GenerateWeight(total);
   float k = (float)(1 / sqrt(total + 1));
   result = (2 * result * k - k) * WeightsMultiplier;
//---
   return result;
  }
//+------------------------------------------------------------------+

