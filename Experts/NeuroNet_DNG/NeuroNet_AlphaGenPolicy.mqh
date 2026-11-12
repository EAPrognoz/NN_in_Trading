//+------------------------------------------------------------------+
//| AlphaGen internal policy implementations.                        |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGEN_POLICY_MQH
#define NEURONET_ALPHAGEN_POLICY_MQH

//+------------------------------------------------------------------+
//| Bind a non-owning view of a device tensor.                       |
//+------------------------------------------------------------------+
bool CAlphaGenBufferView::Bind(COpenCLMy *context, const int index, const int count)
  {
   if(!BufferInit(count, 0))
      ReturnFalse;
   OpenCL = context;
   m_myIndex = index;
   return(true);
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CAlphaGenBufferView::~CAlphaGenBufferView(void)
  {
   m_myIndex = -1;
   OpenCL = NULL;
  }

//+------------------------------------------------------------------+
//| Draw an initialization weight from this layer RNG.               |
//+------------------------------------------------------------------+
float CNeuronAlphaGenDenseOCL::GenerateWeight(void)
  {
   m_iRandom ^= m_iRandom << 13;
   m_iRandom ^= m_iRandom >> 17;
   m_iRandom ^= m_iRandom << 5;
   return((float)((double)m_iRandom / 4294967296.0));
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CNeuronAlphaGenDenseOCL::CNeuronAlphaGenDenseOCL(void) : m_iRandom(1), m_iRows(0)
  {
  }

//+------------------------------------------------------------------+
//| Configure native parameters and policy workspace.                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Configure(COpenCLMy *context, const int inputs, const int outputs, const uint seed,
                                   const ENUM_ACTIVATION function)
  {
   m_iRandom = seed == 0 ? 1 : seed;
   m_iRows = outputs;
   if(!Init(outputs, 0, context, inputs, ADAM, 1))
      ReturnFalse;
   SetActivationFunction(function);
   if(outputs > 0 &&
      (!m_cWeightGradient.BufferInit((inputs + 1) * outputs, 0) || !m_cWeightGradient.BufferCreate(OpenCL)))
      ReturnFalse;
   return(AllocateChecks());
  }

//+------------------------------------------------------------------+
//| Validate serialized parameter geometry.                          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Geometry(const int inputs, const int outputs)
  {
   if(Neurons() != inputs || m_iRows != outputs || activation != None)
      return(false);
   if(outputs == 0)
      return(!Weights);
   return(Weights != NULL && FirstMomentum != NULL && SecondMomentum != NULL &&
           Weights.Total() == (inputs + 1) * outputs && FirstMomentum.Total() == Weights.Total() &&
           SecondMomentum.Total() == Weights.Total());
  }

//+------------------------------------------------------------------+
//| Allocate dense numeric-check workspace.                          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::AllocateChecks(void)
  {
   return(m_cFlags.BufferInit(Neurons(), 0) && m_cFlags.BufferCreate(OpenCL) && m_cOnes.BufferInit(Neurons(), 1) &&
           m_cOnes.BufferCreate(OpenCL) && m_cScalar.BufferInit(1, 0) && m_cScalar.BufferCreate(OpenCL));
  }

//+------------------------------------------------------------------+
//| Run checked native dense feed forward.                           |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::ForwardDense(CNeuronBaseOCL *source)
  {
   bool rejected = false;
   if(!feedForwardChecked(source, GetPointer(m_cFlags)) ||
      !ReduceNumericFlags(GetPointer(m_cFlags), GetPointer(m_cOnes), GetPointer(m_cScalar), Neurons(), rejected))
      ReturnFalse;
   return(!rejected);
  }

//+------------------------------------------------------------------+
//| Copy the current source tensor to the layer output.              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Copy(CBufferFloat *source)
  {
   return(source != NULL && source.Total() >= Neurons() && CopyBufferRaw(source, Output, Neurons()));
  }

//+------------------------------------------------------------------+
//| Materialize the complete dense weight gradient.                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Outer(CNeuronBaseOCL *next)
  {
   if(!next || m_iRows != next.Neurons())
      ReturnFalse;
   setBuffer(def_k_AGDenseGradient, def_ag_AGDenseGradient_input, getOutputIndex())
   setBuffer(def_k_AGDenseGradient, def_ag_AGDenseGradient_gradient, next.getGradientIndex())
   setArgument(def_k_AGDenseGradient, def_ag_AGDenseGradient_inputs, Neurons())
   setArgument(def_k_AGDenseGradient, def_ag_AGDenseGradient_outputs, m_iRows)
   setBuffer(def_k_AGDenseGradient, def_ag_AGDenseGradient_weights_gradient, m_cWeightGradient.GetIndex())
   uint global_work_offset[] = { 0, 0 };
   uint global_work_size[] = { (uint)(Neurons() + 1), (uint)m_iRows };
   kernelExecute(def_k_AGDenseGradient, global_work_offset, global_work_size)
   return(true);
  }

//+------------------------------------------------------------------+
//| Expose the materialized parameter gradient.                      |
//+------------------------------------------------------------------+
CBufferFloat *CNeuronAlphaGenDenseOCL::WeightGradient(void)
  {
   return(GetPointer(m_cWeightGradient));
  }

//+------------------------------------------------------------------+
//| Add parameters, moments and gradients to numeric preflight.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Check(CNeuronAlphaGenEvaluatorOCL *evaluator)
  {
   return(evaluator.Numeric(GetPointer(m_cWeightGradient), false) && evaluator.Numeric(Weights, false) &&
           evaluator.Numeric(FirstMomentum, false) && evaluator.Numeric(SecondMomentum, false));
  }

//+------------------------------------------------------------------+
//| Apply native Adam after successful example preflight.            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::Apply(void)
  {
   if(m_iRows < 1)
      ReturnFalse;
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights, Weights.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights_gradient, m_cWeightGradient.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_m, FirstMomentum.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_v, SecondMomentum.GetIndex())
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_l, lr)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b1, b1)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b2, b2)
   uint global_work_offset[] = { 0, 0 };
   uint global_work_size[] = { (uint)(Neurons() + 1), (uint)m_iRows };
   kernelExecute(def_k_LSTM_UpdateWeightsAdam, global_work_offset, global_work_size)
   t++;
   return(true);
  }

//+------------------------------------------------------------------+
//| Synchronize and serialize native parameters and RNG state.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::SaveState(const int handle)
  {
   if(m_iRows > 0 &&
      (CheckPointer(Weights) == POINTER_INVALID || CheckPointer(FirstMomentum) == POINTER_INVALID ||
       CheckPointer(SecondMomentum) == POINTER_INVALID || !Weights.BufferRead() ||
       !FirstMomentum.BufferRead() || !SecondMomentum.BufferRead()))
      ReturnFalse;
   if(CheckPointer(Output) == POINTER_INVALID || CheckPointer(PrevOutput) == POINTER_INVALID ||
      CheckPointer(Gradient) == POINTER_INVALID || !Output.BufferRead() || !PrevOutput.BufferRead() ||
      !Gradient.BufferRead() || !Save(handle))
      ReturnFalse;
   dFileWriteUInt(handle, m_iRandom)
   return(true);
  }

//+------------------------------------------------------------------+
//| Restore native parameters and allocate transient workspace.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenDenseOCL::LoadState(const int handle, COpenCLMy *context)
  {
   OpenCL = context;
   uint type = 0;
   dFileReadUInt(handle, type)
   if(!Load(handle))
      ReturnFalse;
   dFileReadUInt(handle, m_iRandom)
   m_iRows = getConnections();
   if(m_iRows > 0 && (!m_cWeightGradient.BufferInit(Weights.Total(), 0) || !m_cWeightGradient.BufferCreate(OpenCL)))
      ReturnFalse;
   return(AllocateChecks());
  }

//+------------------------------------------------------------------+
//| Draw an initialization weight from this layer RNG.               |
//+------------------------------------------------------------------+
float CNeuronAlphaGenPolicyLSTMOCL::GenerateWeight(void)
  {
   m_iRandom ^= m_iRandom << 13;
   m_iRandom ^= m_iRandom >> 17;
   m_iRandom ^= m_iRandom << 5;
   return((float)((double)m_iRandom / 4294967296.0));
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CNeuronAlphaGenPolicyLSTMOCL::CNeuronAlphaGenPolicyLSTMOCL(void) : m_iRandom(1)
  {
  }

//+------------------------------------------------------------------+
//| Initialize this recurrent layer RNG.                             |
//+------------------------------------------------------------------+
void CNeuronAlphaGenPolicyLSTMOCL::Seed(const uint seed)
  {
   m_iRandom = seed == 0 ? 1 : seed;
  }

//+------------------------------------------------------------------+
//| Validate serialized parameter geometry.                          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyLSTMOCL::Geometry(const int inputs, const int hidden, const int capacity)
  {
   return(m_iInputs == inputs && Neurons() == hidden && m_iCapacity == capacity &&
           m_cWeightsLSTM.Total() == (inputs + hidden + 1) * hidden * 4);
  }

//+------------------------------------------------------------------+
//| Add parameters, moments and gradients to numeric preflight.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyLSTMOCL::Check(CNeuronAlphaGenEvaluatorOCL *evaluator)
  {
   CAlphaGenBufferView view;
   if(!view.Bind(OpenCL, m_iWeightsGradient, m_cWeightsLSTM.Total()))
      ReturnFalse;
   return(evaluator.Numeric(GetPointer(view), false) && evaluator.Numeric(GetPointer(m_cWeightsLSTM), false) &&
           evaluator.Numeric(GetPointer(m_cFirstMomentumLSTM), false) &&
           evaluator.Numeric(GetPointer(m_cSecondMomentumLSTM), false));
  }

//+------------------------------------------------------------------+
//| Synchronize and serialize native parameters and RNG state.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyLSTMOCL::SaveState(const int handle)
  {
   if(!Output.BufferRead() || !PrevOutput.BufferRead() || !Gradient.BufferRead() || !Save(handle))
      ReturnFalse;
   dFileWriteUInt(handle, m_iRandom)
   dFileWriteUInt(handle, m_iCapacity)
   return(true);
  }

//+------------------------------------------------------------------+
//| Restore native parameters and allocate transient workspace.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyLSTMOCL::LoadState(const int handle, COpenCLMy *context)
  {
   OpenCL = context;
   uint type = 0;
   dFileReadUInt(handle, type)
   if(!Load(handle))
      ReturnFalse;
   dFileReadUInt(handle, m_iRandom)
   uint capacity = 0;
   dFileReadUInt(handle, capacity)
   if(capacity > INT_MAX)
      ReturnFalse;
   return(RestoreWorkspace(context, (int)capacity));
  }

//+------------------------------------------------------------------+
//| Draw a uniform sample from the policy RNG.                       |
//+------------------------------------------------------------------+
float CNeuronAlphaGenPolicyOCL::Uniform(void)
  {
   m_iRandom ^= m_iRandom << 13;
   m_iRandom ^= m_iRandom >> 17;
   m_iRandom ^= m_iRandom << 5;
   return((float)(((double)m_iRandom + 0.5) / 4294967296.0));
  }

//+------------------------------------------------------------------+
//| Read the configured token embedding dimension.                   |
//+------------------------------------------------------------------+
int CNeuronAlphaGenPolicyOCL::EmbeddingDim(void)
  {
   return((int)m_cConfig.At(AG_CFG_EMBEDDING));
  }

//+------------------------------------------------------------------+
//| Read the configured search recurrent dimension.                  |
//+------------------------------------------------------------------+
int CNeuronAlphaGenPolicyOCL::SearchHidden(void)
  {
   return((int)m_cConfig.At(AG_CFG_SEARCH_HIDDEN));
  }

//+------------------------------------------------------------------+
//| Read the configured trading recurrent dimension.                 |
//+------------------------------------------------------------------+
int CNeuronAlphaGenPolicyOCL::TradeHidden(void)
  {
   return((int)m_cConfig.At(AG_CFG_TRADE_HIDDEN));
  }

//+------------------------------------------------------------------+
//| Read the configured trading projection dimension.                |
//+------------------------------------------------------------------+
int CNeuronAlphaGenPolicyOCL::MLP(void)
  {
   return((int)m_cConfig.At(AG_CFG_MLP));
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::Allocate(CBufferFloat &buffer, const int count)
  {
   return(buffer.BufferInit(count, 0) && buffer.BufferCreate(OpenCL));
  }

//+------------------------------------------------------------------+
//| Allocate policy distribution buffers and configuration.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::Setup(COpenCLMy *context, const float &config[], const uint seed)
  {
   OpenCL = context;
   m_iRandom = seed == 0 ? 1 : seed;
   m_bReady = false;
   if(!context || ArraySize(config) != AG_CFG_SIZE)
      ReturnFalse;
   for(int j = 0; j < AG_CFG_SIZE; j++)
      if(!MathIsValidNumber(config[j]))
         ReturnFalse;
   if(config[AG_CFG_EMBEDDING] < 1 || config[AG_CFG_SEARCH_HIDDEN] < 1 || config[AG_CFG_TRADE_HIDDEN] < 1 ||
      config[AG_CFG_MLP] < 1 || config[AG_CFG_MAX_TOKENS] < 1 || config[AG_CFG_W] < 1 ||
      !m_cConfig.AssignArray(config) || !m_cConfig.BufferCreate(OpenCL) || !Allocate(m_cSample, AG_ACTIONS) ||
      !Allocate(m_cAction, AG_ACTIONS) || !Allocate(m_cMetadata, 7) || !Allocate(m_cNoise, AG_ACTIONS))
      ReturnFalse;
   return(true);
  }

//+------------------------------------------------------------------+
//| Map recurrent sequences to heads or head gradients.              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::SequenceMap(CBufferFloat *source, CBufferFloat *output, const int length, const int hidden,
                                      const int mode, CBufferFloat *account)
  {
   setBuffer(def_k_AGSequence, def_ag_AGSequence_input, source.GetIndex())
   setBuffer(def_k_AGSequence, def_ag_AGSequence_output, output.GetIndex())
   setArgument(def_k_AGSequence, def_ag_AGSequence_length, length)
   setArgument(def_k_AGSequence, def_ag_AGSequence_hidden, hidden)
   setArgument(def_k_AGSequence, def_ag_AGSequence_mode, mode)
   setBuffer(def_k_AGSequence, def_ag_AGSequence_account, account.GetIndex())
   uint global_work_offset[] = { 0 };
   uint global_work_size[] = { (uint)(mode == 0 ? hidden : mode == 2 ? hidden + AG_ACCOUNT : length * hidden) };
   kernelExecute(def_k_AGSequence, global_work_offset, global_work_size)
   return(true);
  }

//+------------------------------------------------------------------+
//| Sample or replay the categorical/Gaussian distribution.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::Distribution(CNeuronBaseOCL *head, CBufferFloat *mask, const bool categorical, const int mode)
  {
   m_iMode = mode;
   m_bReady = false;
   if(mode < 0 || mode > 2)
      ReturnFalse;
   if(mode == 0)
     {
      if(categorical)
         m_cNoise.Update(0, Uniform());
      else
         for(int j = 0; j < AG_ACTIONS; j += 2)
           {
            double radius = MathSqrt(-2.0 * MathLog((double)Uniform()));
            double angle = 2.0 * M_PI * (double)Uniform();
            m_cNoise.Update(j, (float)(radius * MathCos(angle)));
            m_cNoise.Update(j + 1, (float)(radius * MathSin(angle)));
           }
      if(!m_cNoise.BufferWrite())
         ReturnFalse;
     }
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_head, head.getOutputIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_mask, mask.GetIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_noise, m_cNoise.GetIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_cfg, m_cConfig.GetIndex())
   setArgument(def_k_AGDistribution, def_ag_AGDistribution_categorical, (int)categorical)
   setArgument(def_k_AGDistribution, def_ag_AGDistribution_mode, mode)
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_sample, m_cSample.GetIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_action, m_cAction.GetIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_metadata, m_cMetadata.GetIndex())
   setBuffer(def_k_AGDistribution, def_ag_AGDistribution_gradient, head.getGradientIndex())
   uint global_work_offset[] = { 0 };
   uint global_work_size[] = { 1 };
   kernelExecute(def_k_AGDistribution, global_work_offset, global_work_size)
   return(true);
  }

//+------------------------------------------------------------------+
//| Preflight only the active device tensor span.                    |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::NumericSpan(CNeuronAlphaGenEvaluatorOCL *evaluator, CBufferFloat *buffer, const int count)
  {
   CAlphaGenBufferView view;
   if(!view.Bind(OpenCL, buffer.GetIndex(), count))
      ReturnFalse;
   return(evaluator.Numeric(GetPointer(view), false));
  }

//+------------------------------------------------------------------+
//| Apply strict ReLU or its derivative.                             |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::ReLU(CNeuronBaseOCL *layer, const int mode)
  {
   setBuffer(def_k_AGReLU, def_ag_AGReLU_input, layer.getOutputIndex())
   setBuffer(def_k_AGReLU, def_ag_AGReLU_gradient, layer.getGradientIndex())
   setArgument(def_k_AGReLU, def_ag_AGReLU_count, layer.Neurons())
   setArgument(def_k_AGReLU, def_ag_AGReLU_mode, mode)
   uint global_work_offset[] = { 0 };
   uint global_work_size[] = { (uint)layer.Neurons() };
   kernelExecute(def_k_AGReLU, global_work_offset, global_work_size)
   return(true);
  }

//+------------------------------------------------------------------+
//| Serialize shared policy configuration and RNG state.             |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::SaveCommon(const int handle)
  {
   dFileWriteUInt(handle, m_iRandom)
   if(!m_cConfig.BufferRead() || !m_cConfig.Save(handle))
      ReturnFalse;
   return(true);
  }

//+------------------------------------------------------------------+
//| Restore policy configuration and RNG state.                      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenPolicyOCL::LoadCommon(const int handle, COpenCLMy *context)
  {
   uint seed;
   dFileReadUInt(handle, seed)
   CBufferFloat config;
   if(!config.Load(handle))
      ReturnFalse;
   float values[];
   config.GetData(values);
   return(Setup(context, values, seed));
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CNeuronAlphaGenPolicyOCL::CNeuronAlphaGenPolicyOCL(void) : m_iRandom(1), m_bReady(false), m_iMode(0)
  {
  }

//+------------------------------------------------------------------+
//| Expose the immutable policy sample device buffer.                |
//+------------------------------------------------------------------+
CBufferFloat *CNeuronAlphaGenPolicyOCL::Sample(void)
  {
   return(GetPointer(m_cSample));
  }

//+------------------------------------------------------------------+
//| Expose the final Action6 device buffer.                          |
//+------------------------------------------------------------------+
CBufferFloat *CNeuronAlphaGenPolicyOCL::Action(void)
  {
   return(GetPointer(m_cAction));
  }

//+------------------------------------------------------------------+
//| Expose likelihood, entropy, value and loss diagnostics.          |
//+------------------------------------------------------------------+
CBufferFloat *CNeuronAlphaGenPolicyOCL::Metadata(void)
  {
   return(GetPointer(m_cMetadata));
  }

//+------------------------------------------------------------------+
//| Allocate embedding and sequence gradient workspace.              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Workspace(void)
  {
   int capacity = (int)m_cConfig.At(AG_CFG_MAX_TOKENS);
   return(Allocate(m_cEmbedded, capacity * EmbeddingDim()) &&
           Allocate(m_cEmbeddedGradient, capacity * EmbeddingDim()) &&
           Allocate(m_cFirstGradient, capacity * SearchHidden()));
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CNeuronAlphaGenSearchPolicyOCL::CNeuronAlphaGenSearchPolicyOCL(void) : m_cTokens(NULL), m_iLength(0)
  {
  }

//+------------------------------------------------------------------+
//| Configure native parameters and policy workspace.                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Configure(COpenCLMy *context, const float &config[], const uint seed)
  {
   if(!Setup(context, config, seed))
      ReturnFalse;
   int capacity = (int)config[AG_CFG_MAX_TOKENS];
   m_cFirst.Seed(seed + 3);
   m_cSecond.Seed(seed + 4);
   return(m_cEmbedding.Configure(context, EmbeddingDim() - 1, AG_VOCABULARY, seed + 1) &&
           m_cMean.Configure(context, SearchHidden(), AG_VOCABULARY + 1, seed + 2) &&
           m_cHead.Configure(context, AG_VOCABULARY + 1, 0, seed + 5) &&
           m_cFirst.Configure(context, EmbeddingDim(), SearchHidden(), capacity) &&
           m_cSecond.Configure(context, SearchHidden(), SearchHidden(), capacity) && Workspace());
  }

//+------------------------------------------------------------------+
//| Encode a complete prefix and evaluate the search heads.          |
//+------------------------------------------------------------------+
int CNeuronAlphaGenSearchPolicyOCL::BatchWorkStride(void)
  {
   return(EmbeddingDim() + 9 * SearchHidden() +
          3 * ((int)m_cConfig.At(AG_CFG_MAX_TOKENS) + 1) * AG_VOCABULARY);
  }

bool CNeuronAlphaGenSearchPolicyOCL::GenerateBatch(const int count, CBufferFloat *results, CBufferFloat *workspace)
  {
   int max_tokens = (int)m_cConfig.At(AG_CFG_MAX_TOKENS);
   if(count < 1 || !results || !workspace ||
      results.Total() < (long)count * (1 + max_tokens * (4 + AG_VOCABULARY)) ||
      workspace.Total() < (long)count * BatchWorkStride())
      ReturnFalse;
   Uniform();
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_embedding, m_cEmbedding.getWeightsIndex())
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_first_lstm, m_cFirst.RecurrentWeightsIndex())
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_second_lstm, m_cSecond.RecurrentWeightsIndex())
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_mean, m_cMean.getWeightsIndex())
   setArgument(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_seed, m_iRandom)
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_results, results.GetIndex())
   setBuffer(def_k_AGSearchGenerate, def_ag_AGSearchGenerate_workspace, workspace.GetIndex())
   long kernel_limit=OpenCL.GetKernelInfoInteger(def_k_AGSearchGenerate,CL_KERNEL_WORK_GROUP_SIZE);
   long lane_limit=MathMin(OpenCL.GetMaxWorkGroupSize(),OpenCL.GetMaxLocalSize(0));
   lane_limit=MathMin(lane_limit,kernel_limit);
   if(lane_limit<1)
      ReturnFalse;
   uint lanes=1;
   while(lanes<64 && 2*(long)lanes<=lane_limit)
      lanes*=2;
   if((long)count*lanes>UINT_MAX)
      ReturnFalse;
   uint offset[] = { 0 }, global[] = { (uint)count*lanes }, local[] = { lanes };
   kernelExecuteLoc(def_k_AGSearchGenerate, offset, global, local)
   return(OpenCL.WaitAlphaGenKernel(def_k_AGSearchGenerate));
  }

bool CNeuronAlphaGenSearchPolicyOCL::ForwardPrefix(CBufferFloat *tokens, const int length, CBufferFloat *mask, const int mode)
  {
   m_bReady = false;
   if(!tokens || !mask || length < 1 || length > (int)m_cConfig.At(AG_CFG_MAX_TOKENS) || tokens.Total() < length ||
      mask.Total() < AG_VOCABULARY)
      ReturnFalse;
   m_cTokens = tokens;
   m_iLength = length;
   setBuffer(def_k_AGEmbedding, def_ag_AGEmbedding_tokens, tokens.GetIndex())
   setArgument(def_k_AGEmbedding, def_ag_AGEmbedding_length, length)
   setBuffer(def_k_AGEmbedding, def_ag_AGEmbedding_weights, m_cEmbedding.getWeightsIndex())
   setArgument(def_k_AGEmbedding, def_ag_AGEmbedding_dimension, EmbeddingDim())
   setBuffer(def_k_AGEmbedding, def_ag_AGEmbedding_output, m_cEmbedded.GetIndex())
   setArgument(def_k_AGEmbedding, def_ag_AGEmbedding_position_scale, m_cConfig.At(AG_CFG_POSITION_SCALE))
   uint global_work_offset[] = { 0, 0 };
   uint global_work_size[] = { (uint)EmbeddingDim(), (uint)length };
   kernelExecute(def_k_AGEmbedding, global_work_offset, global_work_size)
   if(!m_cFirst.ResetAnchor() || !m_cSecond.ResetAnchor() || !m_cFirst.Sequence(GetPointer(m_cEmbedded), length) ||
      !m_cSecond.Sequence(m_cFirst.SequenceOutput(), length) ||
      !SequenceMap(m_cSecond.SequenceOutput(), m_cMean.getOutput(), length, SearchHidden(), 0, mask) ||
      !m_cHead.ForwardDense(GetPointer(m_cMean)))
      ReturnFalse;
   return(Distribution(GetPointer(m_cHead), mask, true, mode));
  }

//+------------------------------------------------------------------+
//| Calculate all example gradients before numeric preflight.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Backward(CNeuronAlphaGenEvaluatorOCL *evaluator)
  {
   m_bReady = false;
   if(!evaluator || m_iMode != 2 || !m_cMean.CalcHiddenGradients(GetPointer(m_cHead)) ||
      !m_cMean.Outer(GetPointer(m_cHead)) ||
      !SequenceMap(m_cMean.getGradient(), m_cSecond.SequenceGradient(), m_iLength, SearchHidden(), 1,
                   GetPointer(m_cNoise)) ||
      !m_cSecond.Backward(m_cFirst.SequenceOutput(), GetPointer(m_cFirstGradient)) ||
      !CopyBufferRaw(GetPointer(m_cFirstGradient), m_cFirst.SequenceGradient(), m_cFirstGradient.Total()) ||
      !m_cFirst.Backward(GetPointer(m_cEmbedded), GetPointer(m_cEmbeddedGradient)))
      ReturnFalse;
   setBuffer(def_k_AGEmbeddingBackward, def_ag_AGEmbeddingBackward_tokens, m_cTokens.GetIndex())
   setArgument(def_k_AGEmbeddingBackward, def_ag_AGEmbeddingBackward_length, m_iLength)
   setBuffer(def_k_AGEmbeddingBackward, def_ag_AGEmbeddingBackward_gradient, m_cEmbeddedGradient.GetIndex())
   setArgument(def_k_AGEmbeddingBackward, def_ag_AGEmbeddingBackward_dimension, EmbeddingDim())
   setBuffer(def_k_AGEmbeddingBackward, def_ag_AGEmbeddingBackward_weights_gradient,
             m_cEmbedding.WeightGradient().GetIndex())
   uint global_work_offset[] = { 0, 0 };
   uint global_work_size[] = { (uint)EmbeddingDim(), AG_VOCABULARY };
   kernelExecute(def_k_AGEmbeddingBackward, global_work_offset, global_work_size)
   if(!evaluator.Numeric(GetPointer(m_cMetadata), true) || !evaluator.Numeric(m_cHead.getOutput(), false) ||
      !evaluator.Numeric(m_cHead.getGradient(), false) || !evaluator.Numeric(m_cMean.getGradient(), false) ||
      !NumericSpan(evaluator, GetPointer(m_cEmbeddedGradient), m_iLength * EmbeddingDim()) ||
      !NumericSpan(evaluator, GetPointer(m_cFirstGradient), m_iLength * SearchHidden()) || !m_cMean.Check(evaluator) ||
      !m_cEmbedding.Check(evaluator) || !m_cFirst.Check(evaluator) || !m_cSecond.Check(evaluator))
      ReturnFalse;
   m_bReady = evaluator.NumericOK();
   return(m_bReady);
  }

//+------------------------------------------------------------------+
//| Apply native Adam after successful example preflight.            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Apply(void)
  {
   if(!m_bReady)
      ReturnFalse;
   m_bReady = false;
   return(m_cEmbedding.Apply() && m_cFirst.Apply() && m_cSecond.Apply() && m_cMean.Apply());
  }

//+------------------------------------------------------------------+
//| Serialize the complete trainable policy state.                   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Save(const int handle)
  {
   return(SaveCommon(handle) && m_cEmbedding.SaveState(handle) && m_cMean.SaveState(handle) &&
           m_cHead.SaveState(handle) && m_cFirst.SaveState(handle) && m_cSecond.SaveState(handle));
  }

//+------------------------------------------------------------------+
//| Restore and validate the complete trainable policy state.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenSearchPolicyOCL::Load(const int handle, COpenCLMy *context)
  {
   return(LoadCommon(handle, context) && m_cEmbedding.LoadState(handle, context) &&
           m_cMean.LoadState(handle, context) && m_cHead.LoadState(handle, context) &&
           m_cFirst.LoadState(handle, context) && m_cSecond.LoadState(handle, context) &&
           m_cEmbedding.Geometry(EmbeddingDim() - 1, AG_VOCABULARY) &&
           m_cMean.Geometry(SearchHidden(), AG_VOCABULARY + 1) && m_cHead.Geometry(AG_VOCABULARY + 1, 0) &&
           m_cFirst.Geometry(EmbeddingDim(), SearchHidden(), (int)m_cConfig.At(AG_CFG_MAX_TOKENS)) &&
           m_cSecond.Geometry(SearchHidden(), SearchHidden(), (int)m_cConfig.At(AG_CFG_MAX_TOKENS)) && Workspace());
  }

//+------------------------------------------------------------------+
//| Initialize or release this policy helper.                        |
//+------------------------------------------------------------------+
CNeuronAlphaGenTradingPolicyOCL::CNeuronAlphaGenTradingPolicyOCL(void) : m_cSignal(NULL), m_iLength(0)
  {
  }

//+------------------------------------------------------------------+
//| Configure native parameters and policy workspace.                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Configure(COpenCLMy *context, const float &config[], const uint seed)
  {
   if(!Setup(context, config, seed))
      ReturnFalse;
   m_cRecurrent.Seed(seed + 1);
   return(m_cRecurrent.Configure(context, 1, TradeHidden(), (int)config[AG_CFG_W]) &&
           m_cInput.Configure(context, TradeHidden() + AG_ACCOUNT, MLP(), seed + 2) &&
           m_cHidden1.Configure(context, MLP(), MLP(), seed + 3, None) &&
           m_cHidden2.Configure(context, MLP(), AG_ACTIONS * 2 + 1, seed + 4, None) &&
           m_cHead.Configure(context, AG_ACTIONS * 2 + 1, 0, seed + 5) &&
           Allocate(m_cSignalGradient, (int)config[AG_CFG_W]));
  }

//+------------------------------------------------------------------+
//| Process closed-bar signals without running account heads.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Encode(CBufferFloat *signal, const int length, CBufferFloat *anchor)
  {
   m_bReady = false;
   if(!signal || length < 1)
      ReturnFalse;
   m_cSignal = signal;
   m_iLength = length;
   if(anchor != NULL && !m_cRecurrent.SetAnchor(anchor))
      ReturnFalse;
   return(m_cRecurrent.Sequence(signal, length));
  }

//+------------------------------------------------------------------+
//| Evaluate heads using the current account snapshot.               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Heads(CBufferFloat *account, const int mode)
  {
   m_bReady = false;
   if(!account || account.Total() != AG_ACCOUNT || m_iLength < 1)
      ReturnFalse;
   if(!SequenceMap(m_cRecurrent.SequenceOutput(), m_cInput.getOutput(), m_iLength, TradeHidden(), 2, account) ||
      !m_cHidden1.ForwardDense(GetPointer(m_cInput)) || !ReLU(GetPointer(m_cHidden1), 0) ||
      !m_cHidden2.ForwardDense(GetPointer(m_cHidden1)) || !ReLU(GetPointer(m_cHidden2), 0) ||
      !m_cHead.ForwardDense(GetPointer(m_cHidden2)))
      ReturnFalse;
   return(Distribution(GetPointer(m_cHead), GetPointer(m_cNoise), false, mode));
  }

//+------------------------------------------------------------------+
//| Encode a sequence and evaluate its trading distribution.         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Forward(CBufferFloat *signal, const int length, CBufferFloat *anchor, CBufferFloat *account,
                                const int mode)
  {
   return(Encode(signal, length, anchor) && Heads(account, mode));
  }

//+------------------------------------------------------------------+
//| Calculate all example gradients before numeric preflight.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Backward(CNeuronAlphaGenEvaluatorOCL *evaluator)
  {
   m_bReady = false;
   if(!evaluator || m_iMode != 2 || !m_cHidden2.CalcHiddenGradients(GetPointer(m_cHead)) ||
      !ReLU(GetPointer(m_cHidden2), 1) || !m_cHidden1.CalcHiddenGradients(GetPointer(m_cHidden2)) ||
      !ReLU(GetPointer(m_cHidden1), 1) || !m_cInput.CalcHiddenGradients(GetPointer(m_cHidden1)) ||
      !m_cInput.Outer(GetPointer(m_cHidden1)) || !m_cHidden1.Outer(GetPointer(m_cHidden2)) ||
      !m_cHidden2.Outer(GetPointer(m_cHead)) ||
      !SequenceMap(m_cInput.getGradient(), m_cRecurrent.SequenceGradient(), m_iLength, TradeHidden(), 3,
                   GetPointer(m_cNoise)) ||
      !m_cRecurrent.Backward(m_cSignal, GetPointer(m_cSignalGradient)))
      ReturnFalse;
   if(!evaluator.Numeric(GetPointer(m_cMetadata), true) || !evaluator.Numeric(m_cHead.getOutput(), false) ||
      !evaluator.Numeric(m_cHead.getGradient(), false) || !evaluator.Numeric(m_cInput.getGradient(), false) ||
      !evaluator.Numeric(m_cHidden1.getGradient(), false) || !evaluator.Numeric(m_cHidden2.getGradient(), false) ||
      !NumericSpan(evaluator, GetPointer(m_cSignalGradient), m_iLength) || !m_cInput.Check(evaluator) ||
      !m_cHidden1.Check(evaluator) || !m_cHidden2.Check(evaluator) || !m_cRecurrent.Check(evaluator))
      ReturnFalse;
   m_bReady = evaluator.NumericOK();
   return(m_bReady);
  }

//+------------------------------------------------------------------+
//| Apply native Adam after successful example preflight.            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Apply(void)
  {
   if(!m_bReady)
      ReturnFalse;
   m_bReady = false;
   return(m_cRecurrent.Apply() && m_cInput.Apply() && m_cHidden1.Apply() && m_cHidden2.Apply());
  }

//+------------------------------------------------------------------+
//| Expose the incoming recurrent anchor.                            |
//+------------------------------------------------------------------+
CBufferFloat *CNeuronAlphaGenTradingPolicyOCL::Anchor(void)
  {
   return(m_cRecurrent.Anchor());
  }

//+------------------------------------------------------------------+
//| Commit the last encoded sequence anchor.                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::CommitAnchor(void)
  {
   return(m_cRecurrent.CommitAnchor());
  }

//+------------------------------------------------------------------+
//| Reset the recurrent anchor for explicit warmup.                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::ResetAnchor(void)
  {
   return(m_cRecurrent.ResetAnchor());
  }

//+------------------------------------------------------------------+
//| Serialize the complete trainable policy state.                   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Save(const int handle)
  {
   return(SaveCommon(handle) && m_cRecurrent.SaveState(handle) && m_cInput.SaveState(handle) &&
           m_cHidden1.SaveState(handle) && m_cHidden2.SaveState(handle) && m_cHead.SaveState(handle));
  }

//+------------------------------------------------------------------+
//| Restore and validate the complete trainable policy state.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenTradingPolicyOCL::Load(const int handle, COpenCLMy *context)
  {
   return(LoadCommon(handle, context) && m_cRecurrent.LoadState(handle, context) &&
           m_cInput.LoadState(handle, context) && m_cHidden1.LoadState(handle, context) &&
           m_cHidden2.LoadState(handle, context) && m_cHead.LoadState(handle, context) &&
           m_cRecurrent.Geometry(1, TradeHidden(), (int)m_cConfig.At(AG_CFG_W)) &&
           m_cInput.Geometry(TradeHidden() + AG_ACCOUNT, MLP()) && m_cHidden1.Geometry(MLP(), MLP()) &&
           m_cHidden2.Geometry(MLP(), AG_ACTIONS * 2 + 1) && m_cHead.Geometry(AG_ACTIONS * 2 + 1, 0) &&
           Allocate(m_cSignalGradient, (int)m_cConfig.At(AG_CFG_W)));
  }

#endif
