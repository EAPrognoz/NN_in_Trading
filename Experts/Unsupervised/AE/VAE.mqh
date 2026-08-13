//+------------------------------------------------------------------+
//|                                                          VAE.mqh |
//|                                                   Copyright DNG® |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//---
#ifndef lr
#include "..\..\NeuroNet_DNG\NeuroNet.mqh"
#endif
#define class_vae
//---
class CVAE : public CNeuronBaseOCL
  {
protected:
   float             m_fKLD_Mult;
   CBufferFloat*     m_cRandom;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::VAE().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) { return true; }  ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.


public:
                     CVAE();
                    ~CVAE();
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual void      SetKLDMult(float value) { m_fKLD_Mult = value;}
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   ///\ingroup neuron_base_gr
   ///@{
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.
   ///@}
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronVAEOCL; }///< Identificator of class.@return Type of class
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CVAE::CVAE()   : m_fKLD_Mult(0.01f)
  {
   m_cRandom = new CBufferFloat();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CVAE::~CVAE()
  {
   if(!!m_cRandom)
      delete m_cRandom;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CVAE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      return false;
//---
   if(!m_cRandom)
     {
      m_cRandom = new CBufferFloat();
      if(!m_cRandom)
         return false;
     }
   if(!m_cRandom.BufferInit(numNeurons, 0.0))
      return false;
   if(!m_cRandom.BufferCreate(OpenCL))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CVAE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      return false;
   if(NeuronOCL.Neurons() % 2 != 0 ||
      NeuronOCL.Neurons() / 2 != Neurons())
      return false;
//---
   double random[];
   if(!MathRandomNormal(0, 1, m_cRandom.Total(), random))
      return false;
   if(!m_cRandom.AssignArray(random))
      return false;
   if(!m_cRandom.BufferWrite())
      return false;
//---
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_inputs, NeuronOCL.getOutput().GetIndex()))
      return false;
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_random, m_cRandom.GetIndex()))
      return false;
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_outputd, Output.GetIndex()))
      return false;
   uint off_set[] = {0};
   uint NDrange[] = {Neurons()};
   if(!OpenCL.Execute(def_k_VAEFeedForward, 1, off_set, NDrange))
      return false;
   //if(!Output.BufferRead())
   //   return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CVAE::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      return false;
//---
   if(!OpenCL.SetArgumentBuffer(def_k_VAECalcHiddenGradient, def_k_vaehg_input, NeuronOCL.getOutput().GetIndex()))
      return false;
   if(!OpenCL.SetArgumentBuffer(def_k_VAECalcHiddenGradient, def_k_vaehg_inp_grad, NeuronOCL.getGradient().GetIndex()))
      return false;
   if(!OpenCL.SetArgumentBuffer(def_k_VAECalcHiddenGradient, def_k_vaehg_random, m_cRandom.GetIndex()))
      return false;
   if(!OpenCL.SetArgumentBuffer(def_k_VAECalcHiddenGradient, def_k_vaehg_gradient, Gradient.GetIndex()))
      return false;
   if(!OpenCL.SetArgument(def_k_VAECalcHiddenGradient, def_k_vaehg_kld_mult, m_fKLD_Mult))
      return false;
   uint off_set[] = {0};
   uint NDrange[] = {Neurons()};
   if(!OpenCL.Execute(def_k_VAECalcHiddenGradient, 1, off_set, NDrange))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CVAE::Save(const int file_handle)
  {
//---
   if(!CNeuronBaseOCL::Save(file_handle))
      return false;
   if(FileWriteFloat(file_handle, m_fKLD_Mult) < sizeof(m_fKLD_Mult))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CVAE::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      return false;
   m_fKLD_Mult = FileReadFloat(file_handle);
//---
   if(!m_cRandom)
     {
      m_cRandom = new CBufferFloat();
      if(!m_cRandom)
         return false;
     }
   m_cRandom.BufferFree();
   if(!m_cRandom.BufferInit(Neurons(), 0.0))
      return false;
   if(!m_cRandom.BufferCreate(OpenCL))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CVAE::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!m_cRandom)
      m_cRandom.BufferCreate(obj);
  }
//+------------------------------------------------------------------+

