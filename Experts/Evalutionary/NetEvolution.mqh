//+------------------------------------------------------------------+
//|                                                 NetEvalution.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//---
#include "NetGenetic.mqh"
//---
class CNetEvolution : protected CNetGenetic
  {
protected:
   virtual bool      GetWeights(uint layer) override;

public:
                     CNetEvolution() {};
                    ~CNetEvolution() {};
   //---
   virtual bool              Create(CArrayObj *Description, uint population_size) override;
   virtual bool              SetPopulationSize(uint size) override;
   virtual bool              feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true) override;
   virtual bool              Rewards(CArrayFloat *rewards) override;
   virtual bool              NextGeneration(float mutation, float &average, float &mamximum);
   virtual bool              Load(string file_name, uint population_size, bool common = true) override;
   virtual bool              Save(string file_name, bool common = true);
   //---
   virtual bool              GetLayerOutput(uint layer, CBufferFloat *&result) override;
   virtual void              getResults(CBufferFloat *&resultVals);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::Create(CArrayObj *Description, uint population_size)
  {
   if(!CNetGenetic::Create(Description, population_size))
      return false;
   float average, maximum;
   return NextGeneration(0, average, maximum);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::SetPopulationSize(uint size)
  {
   return CNetGenetic::SetPopulationSize(size);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true)
  {
   return CNetGenetic::feedForward(inputVals, window, tem);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::Rewards(CArrayFloat *rewards)
  {
   if(!CNetGenetic::Rewards(rewards))
      return false;
//---
   v_Probability = v_Rewards - v_Rewards.Mean();
   v_Probability = v_Probability / MathAbs(v_Probability).Sum();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::NextGeneration(float mutation, float &average, float &maximum)
  {
   maximum = v_Rewards.Max();
   average = v_Rewards.Mean();
   mutation = MathMin(mutation, MaxMutation);
   v_Probability = v_Rewards - v_Rewards.Mean();
   float Sum = MathAbs(v_Probability).Sum();
   if(Sum == 0)
      v_Probability[0] = 1;
   else
      v_Probability = v_Probability / Sum;
//---
   for(int l = 1; l < layers.Total(); l++)
     {
      CLayer *layer = layers.At(l);
      if(!layer)
         return false;
      if(layer.Total() < (int)i_PopulationSize)
         if(!CreatePopulation())
            return false;
      if(!GetWeights(l))
         return false;
      for(uint i = 0; i < i_PopulationSize; i++)
        {
         CNeuronBaseOCL* neuron = layer.At(i);
         if(!neuron)
           return false;
         CBufferFloat* weights = neuron.getWeights();
         if(!!weights)
           {
            for(int w = 0; w < weights.Total(); w++)
              {
               if(mutation > 0)
                 {
                  int err_code;
                  float random = (float)MathRandomNormal(0.5, 0.5, err_code);
                  if(mutation > random)
                    {
                     if(!weights.Update(w, GenerateWeight((uint)m_Weights.Cols())))
                       {
                        Print("Error of update weights");
                        return false;
                       }
                     continue;
                    }
                 }
               if(!MathIsValidNumber(m_Weights[0, w]))
                 {
                  if(!weights.Update(w, GenerateWeight((uint)m_Weights.Cols())))
                    {
                     Print("Error of update weights");
                     return false;
                    }
                  continue;
                 }
               if(!weights.Update(w, m_Weights[0, w] + GenerateWeight((uint)m_Weights.Cols())))
                 {
                  Print("Error of update weights");
                  return false;
                 }
              }
            weights.BufferWrite();
           }
         if(neuron.Type() != defNeuronConvOCL)
            continue;
         CNeuronConvOCL* temp = neuron;
         weights = temp.GetWeightsConv();
         for(int w = 0; w < weights.Total(); w++)
           {
            if(mutation > 0)
              {
               int err_code;
               float random = (float)MathRandomNormal(0.5, 0.5, err_code);
               if(mutation > random)
                 {
                  if(!weights.Update(w, GenerateWeight((uint)m_WeightsConv.Cols())))
                    {
                     Print("Error of update weights");
                     return false;
                    }
                  continue;
                 }
              }
            if(!MathIsValidNumber(m_WeightsConv[0, w]))
              {
               if(!weights.Update(w, GenerateWeight((uint)m_WeightsConv.Cols())))
                 {
                  Print("Error of update weights");
                  return false;
                 }
               continue;
              }
            if(!weights.Update(w, m_WeightsConv[0, w] + GenerateWeight((uint)m_WeightsConv.Cols())))
              {
               Print("Error of update weights");
               return false;
              }
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
bool CNetEvolution::GetWeights(uint layer)
  {
   if(v_Probability.Sum() == 0)
      return false;
   if(!CNetGenetic::GetWeights(layer))
      return false;
//---
   if(m_Weights.Cols() > 0)
     {
      vectorf mean = m_Weights.Mean(0);
      matrixf temp = matrixf::Zeros(1, m_Weights.Cols());
      if(!temp.Row(mean, 0))
         return false;
      temp = (matrixf::Ones(m_Weights.Rows(), 1)).MatMul(temp);
      m_Weights = m_Weights - temp;
      mean = mean + m_Weights.Transpose().MatMul(v_Probability) * lr;
      if(!m_Weights.Resize(1, m_Weights.Cols()))
         return false;
      if(!m_Weights.Row(mean, 0))
         return false;
     }
//---
   if(m_WeightsConv.Cols() > 0)
     {
      vectorf mean = m_WeightsConv.Mean(0);
      matrixf temp = matrixf::Zeros(1, m_WeightsConv.Cols());
      if(!temp.Row(mean, 0))
         return false;
      temp = (matrixf::Ones(m_WeightsConv.Rows(), 1)).MatMul(temp);
      m_WeightsConv = m_WeightsConv - temp;
      mean = mean + m_WeightsConv.Transpose().MatMul(v_Probability) * lr;
      if(!m_WeightsConv.Resize(1, m_WeightsConv.Cols()))
         return false;
      if(!m_WeightsConv.Row(mean, 0))
         return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::Load(string file_name, uint population_size, bool common = true)
  {
   if(!CNetGenetic::Load(file_name, population_size, common))
      return false;
   v_Rewards.Fill(0);
   float average, maximum;
   if(!NextGeneration(0, average, maximum))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::GetLayerOutput(uint layer, CBufferFloat *&result)
  {
   return CNet::GetLayerOutput(layer, result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNetEvolution::getResults(CBufferFloat *&resultVals)
  {
   CNetGenetic::getResults(resultVals);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetEvolution::Save(string file_name, bool common = true)
  {
   return CNetGenetic::SaveModel(file_name, -1, common);
  }
//+------------------------------------------------------------------+

