//+------------------------------------------------------------------+
//| NeuroNet_AlphaGen.mqh                                            |
//+------------------------------------------------------------------+
#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif
//+------------------------------------------------------------------+
//| Validate geometry and allocate exact configured device storage.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Configure(COpenCLMy *context, const float &config[])
  {
   if(!context || ArraySize(config) != AG_CFG_SIZE)
      ReturnFalse;
   for(int i = 0; i < AG_CFG_SIZE; i++)
      if(!MathIsValidNumber(config[i]))
         ReturnFalse;
   m_iWindow = (int)config[AG_CFG_W];
   m_iMaxTokens = (int)config[AG_CFG_MAX_TOKENS];
   m_iCapacity = (int)config[AG_CFG_POOL] + 1;
   long stack = (long)m_iWindow * m_iMaxTokens;
   long mask = (long)AG_VOCABULARY * (m_iMaxTokens + 1) * 3;
   long fit = (long)m_iCapacity * (m_iCapacity + 1);
   if(m_iWindow < 1 || m_iMaxTokens < 2 || m_iCapacity < 2 || config[AG_CFG_HORIZON] < 1 ||
      config[AG_CFG_IC_WINDOW] < 1 || config[AG_CFG_NORM_WINDOW] < 1 || config[AG_CFG_LOOKBACK] < 0 ||
      m_iWindow < config[AG_CFG_LOOKBACK] +
                     MathMax(config[AG_CFG_IC_WINDOW] + config[AG_CFG_HORIZON], config[AG_CFG_NORM_WINDOW]) ||
      stack > INT_MAX || mask > INT_MAX || fit > INT_MAX)
      ReturnFalse;
   if(!CNeuronBaseOCL::Init(0, 0, context, m_iWindow, ADAM, 1) || !m_cConfig.AssignArray(config) ||
      !m_cConfig.BufferCreate(OpenCL) || !m_cMask.BufferInit(AG_VOCABULARY, 0) || !m_cMask.BufferCreate(OpenCL) ||
      !m_cMaskWork.BufferInit((int)mask, 0) || !m_cMaskWork.BufferCreate(OpenCL) ||
      !m_cStack.BufferInit((int)stack, 0) || !m_cStack.BufferCreate(OpenCL) ||
      !m_cValidity.BufferInit((int)MathMax(stack, 3 * (m_iMaxTokens + 1)), 0) || !m_cValidity.BufferCreate(OpenCL) ||
      !m_cRowStatus.BufferInit(m_iWindow, 0) || !m_cRowStatus.BufferCreate(OpenCL) ||
      !m_cFitWork.BufferInit((int)fit, 0) || !m_cFitWork.BufferCreate(OpenCL) || !m_cNumeric.BufferInit(2, 0) ||
      !m_cNumeric.BufferCreate(OpenCL))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Compute the exact next-token mask without reading it to CPU.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Mask(CBufferFloat *tokens, const int length)
  {
   if(!tokens || length < 0 || length >= m_iMaxTokens || tokens.Total() < MathMax(length, 1))
      ReturnFalse;
   setBuffer(def_k_AGMask, def_ag_AGMask_tokens, tokens.GetIndex())
   setArgument(def_k_AGMask, def_ag_AGMask_length, length)
   setBuffer(def_k_AGMask, def_ag_AGMask_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGMask, def_ag_AGMask_mask, m_cMask.GetIndex())
   setBuffer(def_k_AGMask, def_ag_AGMask_workspace, m_cMaskWork.GetIndex())
   uint offset[] = { 0 }, global[] = { AG_VOCABULARY };
   kernelExecute(def_k_AGMask, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluate a completed expression on channel-major closed bars.    |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Evaluate(CBufferFloat *market, CBufferFloat *tokens, const int length)
  {
   if(!market || !tokens || market.Total() < 5 * m_iWindow || length < 2 || length > m_iMaxTokens ||
      tokens.Total() < length)
      ReturnFalse;
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_market, market.GetIndex())
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_tokens, tokens.GetIndex())
   setArgument(def_k_AGEvaluate, def_ag_AGEvaluate_length, length)
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_stack, m_cStack.GetIndex())
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_validity, m_cValidity.GetIndex())
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_result, getOutputIndex())
   setBuffer(def_k_AGEvaluate, def_ag_AGEvaluate_status, m_cRowStatus.GetIndex())
   uint offset[] = { 0 }, global[] = { 128 }, local[] = { 128 };
   kernelExecuteLoc(def_k_AGEvaluate, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
//| Fit stable affine statistics on exactly the required rows.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Standardize(CBufferFloat *source, CBufferFloat *validity, const int offset,
                                         const int count, CBufferFloat *normalized, CBufferFloat *statistics,
                                         const int stats_offset)
  {
   if(!source || !validity || !normalized || !statistics || offset < 0 || count < 1 ||
      count > source.Total() - offset || count > validity.Total() - offset || normalized.Total() < count ||
      stats_offset < 0 || statistics.Total() - stats_offset < AG_STAT_FIELDS)
      ReturnFalse;
   setBuffer(def_k_AGStandardize, def_ag_AGStandardize_input, source.GetIndex())
   setBuffer(def_k_AGStandardize, def_ag_AGStandardize_validity, validity.GetIndex())
   setArgument(def_k_AGStandardize, def_ag_AGStandardize_offset, offset)
   setArgument(def_k_AGStandardize, def_ag_AGStandardize_count, count)
   setBuffer(def_k_AGStandardize, def_ag_AGStandardize_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGStandardize, def_ag_AGStandardize_normalized, normalized.GetIndex())
   setBuffer(def_k_AGStandardize, def_ag_AGStandardize_statistics, statistics.GetIndex())
   setArgument(def_k_AGStandardize, def_ag_AGStandardize_stats_offset, stats_offset)
   uint starts[] = { 0 }, global[] = { 1 };
   kernelExecute(def_k_AGStandardize, starts, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Fit ridge entirely on the device, including its IC diagnostic.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Ridge(CBufferFloat *columns, CBufferFloat *target, const int count, const int factors,
                                   CBufferFloat *weights, CBufferFloat *metrics)
  {
   if(!columns || !target || !weights || !metrics || count < 1 || factors < 1 || factors > m_iCapacity ||
      (long)count * factors > columns.Total() || target.Total() < count || weights.Total() < factors ||
      metrics.Total() < 3)
      ReturnFalse;
   setBuffer(def_k_AGRidge, def_ag_AGRidge_z, columns.GetIndex())
   setBuffer(def_k_AGRidge, def_ag_AGRidge_target, target.GetIndex())
   setArgument(def_k_AGRidge, def_ag_AGRidge_count, count)
   setArgument(def_k_AGRidge, def_ag_AGRidge_columns, factors)
   setBuffer(def_k_AGRidge, def_ag_AGRidge_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGRidge, def_ag_AGRidge_scratch, m_cFitWork.GetIndex())
   setBuffer(def_k_AGRidge, def_ag_AGRidge_weights, weights.GetIndex())
   setBuffer(def_k_AGRidge, def_ag_AGRidge_metrics, metrics.GetIndex())
   uint offset[] = { 0 }, global[] = { 1 };
   kernelExecute(def_k_AGRidge, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Finalize a mature immutable rollout's GAE and actor advantages.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::GAE(CBufferFloat *rollout, const int count, const bool categorical)
  {
   if(!rollout || count < 1 || (long)count * AG_GAE_FIELDS > rollout.Total())
      ReturnFalse;
   setBuffer(def_k_AGGAE, def_ag_AGGAE_rollout, rollout.GetIndex())
   setArgument(def_k_AGGAE, def_ag_AGGAE_count, count)
   setBuffer(def_k_AGGAE, def_ag_AGGAE_cfg, m_cConfig.GetIndex())
   setArgument(def_k_AGGAE, def_ag_AGGAE_token_policy, (int)categorical)
   uint offset[] = { 0 }, global[] = { 1 };
   kernelExecute(def_k_AGGAE, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Accumulate numeric preflight results before touching parameters. |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Numeric(CBufferFloat *source, const bool reset)
  {
   if(!source || source.Total() < 1)
      ReturnFalse;
   setBuffer(def_k_AGNumeric, def_ag_AGNumeric_input, source.GetIndex())
   setArgument(def_k_AGNumeric, def_ag_AGNumeric_count, source.Total())
   setBuffer(def_k_AGNumeric, def_ag_AGNumeric_status, m_cNumeric.GetIndex())
   setArgument(def_k_AGNumeric, def_ag_AGNumeric_reset, (int)reset)
   uint offset[] = { 0 }, global[] = { 1 };
   kernelExecute(def_k_AGNumeric, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Read one control scalar; historical arrays never leave device.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::NumericOK(void)
  {
   if(!m_cNumeric.BufferRead())
      ReturnFalse;
   return(m_cNumeric.At(0) == AG_OK);
  }
//+------------------------------------------------------------------+
//| Allocate native recurrent parameters and the full BPTT tape.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Configure(COpenCLMy *context, const int inputs, const int hidden, const int capacity)
  {
   if(!context || inputs < 1 || hidden < 1 || capacity < 1 || (long)capacity * hidden * 6 > INT_MAX ||
      (long)(inputs + hidden + 1) * hidden * 4 > INT_MAX)
      ReturnFalse;
   if(!CNeuronLSTMOCL::Init(0, 0, context, hidden, ADAM, 1) || !SetInputs(inputs))
      ReturnFalse;
   return(RestoreWorkspace(context, capacity));
  }
//+------------------------------------------------------------------+
//| Allocate transient tape after Load without touching parameters.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::RestoreWorkspace(COpenCLMy *context, const int capacity)
  {
   if(context != OpenCL || capacity < 1 || m_iInputs < 1 || m_iVariables != 1 ||
      (long)capacity * Neurons() * 6 > INT_MAX)
      ReturnFalse;
   int hidden = (int)Neurons();
   m_iCapacity = capacity;
   if(!m_cTape.BufferInit(capacity * hidden * 6, 0) || !m_cTape.BufferCreate(OpenCL) ||
      !m_cAnchor.BufferInit(hidden * 2, 0) || !m_cAnchor.BufferCreate(OpenCL) ||
      !m_cSequence.BufferInit(capacity * hidden, 0) || !m_cSequence.BufferCreate(OpenCL) ||
      !m_cSequenceGradient.BufferInit(capacity * hidden, 0) || !m_cSequenceGradient.BufferCreate(OpenCL) ||
      !m_cBPTTWork.BufferInit(hidden * 6, 0) || !m_cBPTTWork.BufferCreate(OpenCL))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Forward one sequence from its explicit incoming anchor.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Sequence(CBufferFloat *source, const int length)
  {
   if(!source || length < 1 || length > m_iCapacity || (long)length * m_iInputs > source.Total())
      ReturnFalse;
   m_iLength = length;
   setBuffer(def_k_AGLSTMForward, def_ag_AGLSTMForward_input, source.GetIndex())
   setArgument(def_k_AGLSTMForward, def_ag_AGLSTMForward_length, length)
   setArgument(def_k_AGLSTMForward, def_ag_AGLSTMForward_inputs, m_iInputs)
   setArgument(def_k_AGLSTMForward, def_ag_AGLSTMForward_hidden, (int)Neurons())
   setBuffer(def_k_AGLSTMForward, def_ag_AGLSTMForward_weights, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_AGLSTMForward, def_ag_AGLSTMForward_anchor, m_cAnchor.GetIndex())
   setBuffer(def_k_AGLSTMForward, def_ag_AGLSTMForward_tape, m_cTape.GetIndex())
   setBuffer(def_k_AGLSTMForward, def_ag_AGLSTMForward_output, m_cSequence.GetIndex())
   uint offset[] = { 0 }, global[] = { 128 }, local[] = { 128 };
   kernelExecuteLoc(def_k_AGLSTMForward, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
//| Backpropagate one sequence; do not update any parameter yet.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Backward(CBufferFloat *source, CBufferFloat *input_gradient)
  {
   if(!source || !input_gradient || m_iLength < 1 || (long)m_iLength * m_iInputs > source.Total() ||
      (long)m_iLength * m_iInputs > input_gradient.Total())
      ReturnFalse;
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_input, source.GetIndex())
   setArgument(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_length, m_iLength)
   setArgument(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_inputs, m_iInputs)
   setArgument(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_hidden, (int)Neurons())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_weights, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_anchor, m_cAnchor.GetIndex())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_tape, m_cTape.GetIndex())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_gradient, m_cSequenceGradient.GetIndex())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_input_gradient, input_gradient.GetIndex())
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_weights_gradient, m_iWeightsGradient)
   setBuffer(def_k_AGLSTMBackward, def_ag_AGLSTMBackward_scratch, m_cBPTTWork.GetIndex())
   uint offset[] = { 0 }, global[] = { 128 }, local[] = { 128 };
   kernelExecuteLoc(def_k_AGLSTMBackward, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
//| Native Adam over all four gates, after whole-example preflight.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Apply(void)
  {
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights_gradient, m_iWeightsGradient)
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_m, m_cFirstMomentumLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_v, m_cSecondMomentumLSTM.GetIndex())
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_l, lr)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b1, b1)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b2, b2)
   uint offset[] = { 0, 0 }, global[] = { (uint)(m_iInputs + Neurons() + 1), Neurons() * 4 };
   kernelExecute(def_k_LSTM_UpdateWeightsAdam, offset, global)
   t++;
   return(true);
  }
//+------------------------------------------------------------------+
//| Start an independent prefix or an explicitly requested warmup.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::ResetAnchor(void)
  {
   for(int i = 0; i < m_cAnchor.Total(); i++)
      if(!m_cAnchor.Update(i, 0))
         ReturnFalse;
   if(!m_cAnchor.BufferWrite())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Restore an example's constant incoming h/c without host copying. |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::SetAnchor(CBufferFloat *anchor)
  {
   return(anchor && anchor.Total() == m_cAnchor.Total() &&
           CopyBufferRaw(anchor, GetPointer(m_cAnchor), m_cAnchor.Total()));
  }
//+------------------------------------------------------------------+
//| Commit the sequence end h/c to the current working anchor.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::CommitAnchor(void)
  {
   if(m_iLength < 1)
      ReturnFalse;
   setBuffer(def_k_AGSequence, def_ag_AGSequence_input, m_cTape.GetIndex())
   setBuffer(def_k_AGSequence, def_ag_AGSequence_output, m_cAnchor.GetIndex())
   setArgument(def_k_AGSequence, def_ag_AGSequence_length, m_iLength)
   setArgument(def_k_AGSequence, def_ag_AGSequence_hidden, (int)Neurons())
   setArgument(def_k_AGSequence, def_ag_AGSequence_mode, 4)
   setBuffer(def_k_AGSequence, def_ag_AGSequence_account, m_cAnchor.GetIndex())
   uint offset[] = { 0 }, global[] = { (uint)m_cAnchor.Total() };
   kernelExecute(def_k_AGSequence, offset, global)
   return(true);
  }
