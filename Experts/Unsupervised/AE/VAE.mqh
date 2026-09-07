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
enum ENUM_VAE_SAMPLE_MODE
  {
   VAE_SAMPLE_RANDOM = 0,
   VAE_SAMPLE_MEAN
  };
//---
class CVAE : public CNeuronBaseOCL
  {
protected:
   float             m_fKLD_Mult;
   CBufferFloat*     m_cRandom;
   ENUM_VAE_SAMPLE_MODE  m_eSampleMode;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::VAE().@param NeuronOCL Pointer to previos layer.
   bool              ValidateCheckedScratch(CNeuronBaseOCL *NeuronOCL,
         CBufferFloat *numeric_flags,
         CBufferFloat *numeric_ones,
         CBufferFloat *numeric_scalar);
   virtual bool      VAEFeedForward(CNeuronBaseOCL *NeuronOCL,
                                    CBufferFloat *numeric_flags);

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) { return true; }  ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.


public:
                     CVAE();
                    ~CVAE();
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual void      SetKLDMult(float value) { m_fKLD_Mult = value; }
   virtual void      SetSampleMode(ENUM_VAE_SAMPLE_MODE mode)
     {
      m_eSampleMode = mode;
     }
   virtual ENUM_VAE_SAMPLE_MODE SampleMode(void) const
     {
      return(m_eSampleMode);
     }
   virtual bool      FeedForwardChecked(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *numeric_flags,
                                        CBufferFloat *numeric_ones,
                                        CBufferFloat *numeric_scalar,
                                        bool &rejected);
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
   virtual bool      SupportsOpenCLChecked(void) const override;
   virtual bool      SetOpenCLChecked(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//| Construct the VAE sampler with random sampling as the default.   |
//+------------------------------------------------------------------+
CVAE::CVAE()
   : m_fKLD_Mult(0.01f),
     m_eSampleMode(VAE_SAMPLE_RANDOM)
  {
   m_cRandom = new CBufferFloat();
  }
//+------------------------------------------------------------------+
//| Release the VAE random sampling buffer.                          |
//+------------------------------------------------------------------+
CVAE::~CVAE()
  {
   if(!!m_cRandom)
      delete m_cRandom;
  }
//+------------------------------------------------------------------+
//| Initialize the VAE layer and its sampling buffer.                |
//+------------------------------------------------------------------+
bool CVAE::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//--- Initialize the inherited layer before allocating the random source.
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      return(false);
//--- Allocate the deterministic/random sampling buffer.
   if(!m_cRandom)
     {
      m_cRandom = new CBufferFloat();
      if(!m_cRandom)
         return(false);
     }
   if(!m_cRandom.BufferInit(numNeurons, 0.0))
      return(false);
   if(!m_cRandom.BufferCreate(OpenCL))
      return(false);
//--- Initialization completed successfully.
   return(true);
  }
//+------------------------------------------------------------------+
//| Execute one deterministic or random VAE forward sample.          |
//+------------------------------------------------------------------+
bool CVAE::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Validate the posterior input layout.
   if(!OpenCL || !NeuronOCL)
      return(false);
   if(NeuronOCL.Neurons() % 2 != 0 ||
      NeuronOCL.Neurons() / 2 != Neurons())
      return(false);
//--- Select the deterministic mean sampler or generate random samples.
   if(m_eSampleMode == VAE_SAMPLE_MEAN)
     {
      if(!m_cRandom.Fill(0.0f))
         return(false);
     }
   else
     {
      double random[];
      if(!MathRandomNormal(0, 1, m_cRandom.Total(), random))
         return(false);
      if(!m_cRandom.AssignArray(random))
         return(false);
      if(!m_cRandom.BufferWrite())
         return(false);
     }
//--- Execute the selected sample through the VAE kernel.
   return(VAEFeedForward(NeuronOCL, NULL));
  }
//+------------------------------------------------------------------+
//| Validates caller scratch against every VAE kernel operand.      |
//+------------------------------------------------------------------+
bool CVAE::ValidateCheckedScratch(CNeuronBaseOCL *NeuronOCL,
                                  CBufferFloat *numeric_flags,
                                  CBufferFloat *numeric_ones,
                                  CBufferFloat *numeric_scalar)
  {
//--- Resolve the exact posterior, local-draw, and output operands.
   if(!OpenCL || !NeuronOCL || !m_cRandom || !Output || !NeuronOCL.getOutput() || Neurons() < 1 ||
      NeuronOCL.Neurons() % 2 != 0 || NeuronOCL.Neurons() / 2 != Neurons())
      return(false);
   CBufferFloat *posterior = NeuronOCL.getOutput();
   const ulong posterior_total = 2 * ulong(Neurons());
//--- Reject invalid scratch ownership, spans, and aliases before sampling mutates a draw.
   if(!numeric_flags || !numeric_ones || !numeric_scalar || numeric_flags == numeric_ones ||
      numeric_flags == numeric_scalar || numeric_ones == numeric_scalar ||
      posterior.GetOpenCL() != OpenCL || m_cRandom.GetOpenCL() != OpenCL || Output.GetOpenCL() != OpenCL ||
      numeric_flags.GetOpenCL() != OpenCL || numeric_ones.GetOpenCL() != OpenCL ||
      numeric_scalar.GetOpenCL() != OpenCL || posterior.GetIndex() < 0 || m_cRandom.GetIndex() < 0 ||
      Output.GetIndex() < 0 || numeric_flags.GetIndex() < 0 || numeric_ones.GetIndex() < 0 ||
      numeric_scalar.GetIndex() < 0 || ulong(posterior.Total()) < posterior_total || m_cRandom.Total() < Neurons() ||
      Output.Total() < Neurons() || numeric_flags.Total() < Neurons() || numeric_ones.Total() < Neurons() ||
      numeric_scalar.Total() != 1 || numeric_flags.GetIndex() == numeric_ones.GetIndex() ||
      numeric_flags.GetIndex() == numeric_scalar.GetIndex() || numeric_ones.GetIndex() == numeric_scalar.GetIndex() ||
      numeric_flags.GetIndex() == posterior.GetIndex() || numeric_flags.GetIndex() == m_cRandom.GetIndex() ||
      numeric_flags.GetIndex() == Output.GetIndex() || numeric_ones.GetIndex() == posterior.GetIndex() ||
      numeric_ones.GetIndex() == m_cRandom.GetIndex() || numeric_ones.GetIndex() == Output.GetIndex() ||
      numeric_scalar.GetIndex() == posterior.GetIndex() || numeric_scalar.GetIndex() == m_cRandom.GetIndex() ||
      numeric_scalar.GetIndex() == Output.GetIndex())
      return(false);
//--- Finalize the pre-dispatch scratch validation.
   return(true);
  }
//+------------------------------------------------------------------+
//| Dispatch the selected VAE sample with an optional numeric flag. |
//+------------------------------------------------------------------+
bool CVAE::VAEFeedForward(CNeuronBaseOCL *NeuronOCL,
                          CBufferFloat *numeric_flags)
  {
//--- Validate the VAE buffers required by the checked kernel binding.
   if(!OpenCL || !NeuronOCL || !m_cRandom)
      return(false);
   const ulong posterior_total = 2 * ulong(Neurons());
   if(numeric_flags && (!NeuronOCL.getOutput() || !Output || Neurons() < 1 ||
                        NeuronOCL.getOutput().GetOpenCL() != OpenCL ||
                        m_cRandom.GetOpenCL() != OpenCL || Output.GetOpenCL() != OpenCL ||
                        numeric_flags.GetOpenCL() != OpenCL || NeuronOCL.getOutput().GetIndex() < 0 ||
                        m_cRandom.GetIndex() < 0 || Output.GetIndex() < 0 || numeric_flags.GetIndex() < 0 ||
                        ulong(NeuronOCL.getOutput().Total()) < posterior_total || m_cRandom.Total() < Neurons() ||
                        Output.Total() < Neurons() || numeric_flags.Total() < Neurons() ||
                        numeric_flags.GetIndex() == NeuronOCL.getOutput().GetIndex() ||
                        numeric_flags.GetIndex() == m_cRandom.GetIndex() ||
                        numeric_flags.GetIndex() == Output.GetIndex()))
      return(false);
//--- Bind the posterior, selected local draw, output, and optional status.
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_inputs, NeuronOCL.getOutput().GetIndex()))
      return(false);
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_random, m_cRandom.GetIndex()))
      return(false);
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_outputd, Output.GetIndex()))
      return(false);
   if(!OpenCL.SetArgumentBuffer(def_k_VAEFeedForward, def_k_vaeff_numeric_flags,
                                (numeric_flags ? numeric_flags.GetIndex() : Output.GetIndex())))
      return(false);
   if(!OpenCL.SetArgument(def_k_VAEFeedForward, def_k_vaeff_flags_enabled, (numeric_flags ? 1 : 0)))
      return(false);
   uint off_set[] = {0};
   uint NDrange[] = {Neurons()};
   if(!OpenCL.Execute(def_k_VAEFeedForward, 1, off_set, NDrange))
      return(false);
//--- Finalize the VAE kernel dispatch.
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs one checked VAE sample and reduces its output status.       |
//+------------------------------------------------------------------+
bool CVAE::FeedForwardChecked(CNeuronBaseOCL *NeuronOCL,
                              CBufferFloat *numeric_flags,
                              CBufferFloat *numeric_ones,
                              CBufferFloat *numeric_scalar,
                              bool &rejected)
  {
   rejected = false;
//--- Reject malformed posterior layouts and scratch aliases before sampling.
   if(!ValidateCheckedScratch(NeuronOCL, numeric_flags, numeric_ones, numeric_scalar))
      return(false);
//--- Preserve the selected mean/random sampling mode before dispatch.
   if(m_eSampleMode == VAE_SAMPLE_MEAN)
     {
      if(!m_cRandom || !m_cRandom.Fill(0.0f))
         return(false);
     }
   else
     {
      double random[];
      if(!m_cRandom || !MathRandomNormal(0, 1, m_cRandom.Total(), random) ||
         !m_cRandom.AssignArray(random) || !m_cRandom.BufferWrite())
         return(false);
     }
//--- Dispatch and immediately reduce the VAE-written output prefix.
   if(!VAEFeedForward(NeuronOCL, numeric_flags))
      return(false);
//--- Finalize the checked VAE dispatch.
   return(ReduceNumericFlags(numeric_flags, numeric_ones, numeric_scalar,
                             (uint)Neurons(), rejected));
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
//| Checks checked-transfer support of VAE base and random holder.   |
//+------------------------------------------------------------------+
bool CVAE::SupportsOpenCLChecked(void) const
  {
   return(Type() == defNeuronVAEOCL && CheckPointer(m_cRandom) != POINTER_INVALID);
  }
//+------------------------------------------------------------------+
//| Transfers VAE random data before inherited context publication.  |
//+------------------------------------------------------------------+
bool CVAE::SetOpenCLChecked(COpenCLMy *obj)
  {
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      return(false);
//--- Synchronize the random holder only when it still belongs to another context.
   if(m_cRandom.Total() > 0 && m_cRandom.GetIndex() >= 0 &&
      m_cRandom.GetOpenCL() != obj && !m_cRandom.BufferRead())
      return(false);
//--- Recreate a changed or host-only random holder before base publication.
   if(m_cRandom.Total() > 0 &&
      (m_cRandom.GetOpenCL() != obj || m_cRandom.GetIndex() < 0) &&
      !m_cRandom.BufferCreate(obj))
      return(false);
//--- Finalize by transferring inherited base owners last.
   return(CNeuronBaseOCL::SetOpenCLChecked(obj));
  }
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
