//+------------------------------------------------------------------+
//| NeuroNet_AlphaGen.mqh                                            |
//+------------------------------------------------------------------+
#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
#ifndef NEURONET_ALPHA_GEN_MQH
#define NEURONET_ALPHA_GEN_MQH
//+------------------------------------------------------------------+
//| Configure - Configure device state                               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Configure(COpenCLMy*context,const float&config[])
  {
   if(!context||ArraySize(config)!=AG_CFG_SIZE)
      ReturnFalse;

   for(int i=0;i<AG_CFG_SIZE;i++)
      if(!MathIsValidNumber(config[i]))
         ReturnFalse;

   m_iWindow=(int)config[AG_CFG_W];
   m_iMaxTokens=(int)config[AG_CFG_MAX_TOKENS];
   m_iCapacity=(int)config[AG_CFG_POOL]+1;
   long stack=(long)m_iWindow*m_iMaxTokens;
   long mask=(long)AG_VOCABULARY*(m_iMaxTokens+1)*3;
   long fit=(long)m_iCapacity*(m_iCapacity+1);

   if(m_iWindow<1||m_iMaxTokens<2||m_iCapacity<2||config[AG_CFG_HORIZON]<1||
      config[AG_CFG_IC_WINDOW]<1||config[AG_CFG_NORM_WINDOW]<1||config[AG_CFG_LOOKBACK]<0||
      m_iWindow<config[AG_CFG_LOOKBACK]+
      MathMax(config[AG_CFG_IC_WINDOW]+config[AG_CFG_HORIZON],config[AG_CFG_NORM_WINDOW])||
      stack>INT_MAX||mask>INT_MAX||fit>INT_MAX)
      ReturnFalse;

   if(!CNeuronBaseOCL::Init(0,0,context,m_iWindow,ADAM,1)||!m_cConfig.AssignArray(config)||
      !m_cConfig.BufferCreate(OpenCL)||!m_cMask.BufferInit(AG_VOCABULARY,0)||!m_cMask.BufferCreate(OpenCL)||
      !m_cMaskWork.BufferInit((int)mask,0)||!m_cMaskWork.BufferCreate(OpenCL)||
      !m_cStack.BufferInit((int)stack,0)||!m_cStack.BufferCreate(OpenCL)||
      !m_cValidity.BufferInit((int)MathMax(stack,3*(m_iMaxTokens+1)),0)||!m_cValidity.BufferCreate(OpenCL)||
      !m_cRowStatus.BufferInit(m_iWindow,0)||!m_cRowStatus.BufferCreate(OpenCL)||
      !m_cFitWork.BufferInit((int)fit,0)||!m_cFitWork.BufferCreate(OpenCL)||!m_cNumeric.BufferInit(2+AG_NUMERIC_GROUPS,0)||
      !m_cNumeric.BufferCreate(OpenCL))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Mask - Build the next-token mask                                 |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Mask(CBufferFloat*tokens,const int length)
  {
   if(!tokens||length<0||length>=m_iMaxTokens||tokens.Total()<MathMax(length,1))
      ReturnFalse;

#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES

//--- a device copy fences earlier queue work without touching the formula.
   setBuffer(def_k_AGCopySlice,def_ag_AGCopySlice_source,m_cConfig.GetIndex())
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_source_offset,AG_CFG_MAX_TOKENS)
   setBuffer(def_k_AGCopySlice,def_ag_AGCopySlice_destination,m_cNumeric.GetIndex())
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_destination_offset,1)
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_count,1)
   uint probe_offset[]={0};
   uint probe_global[]={1};
   //+------------------------------------------------------------------+
   //| kernelExecute - Implement kernelExecute                          |
   //+------------------------------------------------------------------+
   kernelExecute(def_k_AGCopySlice,probe_offset,probe_global)

   if(!OpenCL.WaitAlphaGenKernel(def_k_AGCopySlice))
     {
      PrintFormat("AlphaGen PRE-MASK queue failed: AGMask not submitted; prefix=%d max=%d token_count=%d token_buffer=%d mask_count=%d work_count=%d",
                  length,m_iMaxTokens,tokens.Total(),tokens.GetIndex(),m_cMask.Total(),m_cMaskWork.Total());
      ReturnFalse;
     }
   float marker[1]={-1};

   ResetLastError();
   bool probe_read=OpenCL.BufferRead(m_cNumeric.GetIndex(),marker,sizeof(float),0,1);

   if(!probe_read||marker[0]!=(float)m_iMaxTokens)
     {
      PrintFormat("AlphaGen PRE-MASK marker failed: AGMask not submitted; prefix=%d read=%d error=%d marker=%.9g expected=%d",
                  length,(int)probe_read,GetLastError(),marker[0],m_iMaxTokens);
      ReturnFalse;
     }
#endif
   setBuffer(def_k_AGMask,def_ag_AGMask_tokens,tokens.GetIndex())
   setArgument(def_k_AGMask,def_ag_AGMask_length,length)
   setBuffer(def_k_AGMask,def_ag_AGMask_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGMask,def_ag_AGMask_mask,m_cMask.GetIndex())
   setBuffer(def_k_AGMask,def_ag_AGMask_workspace,m_cMaskWork.GetIndex())
   uint offset[]={0};
   uint global[]={AG_VOCABULARY};
   //+------------------------------------------------------------------+
   //| kernelExecute - Implement kernelExecute                          |
   //+------------------------------------------------------------------+
   kernelExecute(def_k_AGMask,offset,global)
#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES
   if(!OpenCL.WaitAlphaGenKernel(def_k_AGMask))
     {
      PrintFormat("AlphaGen MASK failed after successful queue probe: prefix=%d max=%d token_count=%d token_buffer=%d mask_count=%d work_count=%d expected_work=%d",
                  length,m_iMaxTokens,tokens.Total(),tokens.GetIndex(),m_cMask.Total(),m_cMaskWork.Total(),AG_VOCABULARY*(m_iMaxTokens+1)*3);
      ReturnFalse;
     }
#endif
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluate - Evaluate a token expression                           |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Evaluate(CBufferFloat*market,CBufferFloat*tokens,const int length)
  {
   if(!market||!tokens||market.Total()<5*m_iWindow||length<2||length>m_iMaxTokens||
      tokens.Total()<length)
      ReturnFalse;

   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_market,market.GetIndex())
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_tokens,tokens.GetIndex())
   setArgument(def_k_AGEvaluate,def_ag_AGEvaluate_length,length)
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_stack,m_cStack.GetIndex())
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_validity,m_cValidity.GetIndex())
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_result,getOutputIndex())
   setBuffer(def_k_AGEvaluate,def_ag_AGEvaluate_status,m_cRowStatus.GetIndex())
   uint lanes=OpenCL.AlphaGenGroupSize(def_k_AGEvaluate,128,1);

   if(lanes==0)
      ReturnFalse;

   uint offset[]={0};
   uint global[]={lanes};
   uint local[]={lanes};

   kernelExecuteLoc(def_k_AGEvaluate,offset,global,local)

   return(true);
  }
//+------------------------------------------------------------------+
//| Standardize - Fit affine normalization statistics                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Standardize(CBufferFloat*source,CBufferFloat*validity,const int offset,
                                              const int count,CBufferFloat*normalized,CBufferFloat*statistics,
                                              const int stats_offset)
  {
   if(!source||!validity||!normalized||!statistics||offset<0||count<1||
      count>source.Total()-offset||count>validity.Total()-offset||normalized.Total()<count||
      stats_offset<0||statistics.Total()-stats_offset<AG_STAT_FIELDS)
      ReturnFalse;

   setBuffer(def_k_AGStandardize,def_ag_AGStandardize_input,source.GetIndex())
   setBuffer(def_k_AGStandardize,def_ag_AGStandardize_validity,validity.GetIndex())
   setArgument(def_k_AGStandardize,def_ag_AGStandardize_offset,offset)
   setArgument(def_k_AGStandardize,def_ag_AGStandardize_count,count)
   setBuffer(def_k_AGStandardize,def_ag_AGStandardize_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGStandardize,def_ag_AGStandardize_normalized,normalized.GetIndex())
   setBuffer(def_k_AGStandardize,def_ag_AGStandardize_statistics,statistics.GetIndex())
   setArgument(def_k_AGStandardize,def_ag_AGStandardize_stats_offset,stats_offset)
   uint starts[]={0};
   uint global[]={1};

   kernelExecute(def_k_AGStandardize,starts,global)

   return(true);
  }
//+------------------------------------------------------------------+
//| Ridge - Fit ridge weights and calculate IC                       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Ridge(CBufferFloat*columns,CBufferFloat*target,const int count,const int factors,
                                        CBufferFloat *weights,CBufferFloat*metrics)
  {
   if(!columns||!target||!weights||!metrics||count<1||factors<1||factors>m_iCapacity||
      (long)count*factors>columns.Total()||target.Total()<count||weights.Total()<factors||
      metrics.Total()<3)
      ReturnFalse;

   setBuffer(def_k_AGRidge,def_ag_AGRidge_z,columns.GetIndex())
   setBuffer(def_k_AGRidge,def_ag_AGRidge_target,target.GetIndex())
   setArgument(def_k_AGRidge,def_ag_AGRidge_count,count)
   setArgument(def_k_AGRidge,def_ag_AGRidge_columns,factors)
   setBuffer(def_k_AGRidge,def_ag_AGRidge_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGRidge,def_ag_AGRidge_scratch,m_cFitWork.GetIndex())
   setBuffer(def_k_AGRidge,def_ag_AGRidge_weights,weights.GetIndex())
   setBuffer(def_k_AGRidge,def_ag_AGRidge_metrics,metrics.GetIndex())
   uint offset[]={0};
   uint global[]={1};

   kernelExecute(def_k_AGRidge,offset,global)

   return(true);
  }
//+------------------------------------------------------------------+
//| GAE - Calculate rollout advantages                               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::GAE(CBufferFloat*rollout,const int count,const bool categorical)
  {
   if(!rollout||count<1||(long)count*AG_GAE_FIELDS>rollout.Total())
      ReturnFalse;

   setBuffer(def_k_AGGAE,def_ag_AGGAE_rollout,rollout.GetIndex())
   setArgument(def_k_AGGAE,def_ag_AGGAE_count,count)
   setBuffer(def_k_AGGAE,def_ag_AGGAE_cfg,m_cConfig.GetIndex())
   setArgument(def_k_AGGAE,def_ag_AGGAE_token_policy,(int)categorical)
   uint offset[]={0};
   uint global[]={1};

   kernelExecute(def_k_AGGAE,offset,global)

   return(true);
  }
//+------------------------------------------------------------------+
//| Numeric - Accumulate numeric validation flags                    |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::Numeric(CBufferFloat*source,const bool reset)
  {
   return(NumericBatch(source,NULL,NULL,NULL,reset));
  }
//+------------------------------------------------------------------+
//| NumericBatch - Validate four buffers with one local reduction    |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::NumericBatch(CBufferFloat*source,CBufferFloat*b,CBufferFloat*c,CBufferFloat*d,const bool reset)
  {
   if(!source||source.Total()<1)
      ReturnFalse;

   if((b&&b.Total()<1)||(c&&c.Total()<1)||(d&&d.Total()<1))
      ReturnFalse;

   setBuffer(def_k_AGNumeric,4,b?b.GetIndex():source.GetIndex())
   setArgument(def_k_AGNumeric,5,b?b.Total():0)
   setBuffer(def_k_AGNumeric,6,c?c.GetIndex():source.GetIndex())
   setArgument(def_k_AGNumeric,7,c?c.Total():0)
   setBuffer(def_k_AGNumeric,8,d?d.GetIndex():source.GetIndex())
   setArgument(def_k_AGNumeric,9,d?d.Total():0)
   setBuffer(def_k_AGNumeric,def_ag_AGNumeric_input,source.GetIndex())
   setArgument(def_k_AGNumeric,def_ag_AGNumeric_count,source.Total())
   setBuffer(def_k_AGNumeric,def_ag_AGNumeric_status,m_cNumeric.GetIndex())
   setArgument(def_k_AGNumeric,def_ag_AGNumeric_reset,(int)reset)
   uint lanes=OpenCL.AlphaGenGroupSize(def_k_AGNumeric,64,1);

   if(lanes==0)
      ReturnFalse;

   uint offset[]={0};
   uint global[]={lanes*AG_NUMERIC_GROUPS};
   uint local[]={lanes};

   kernelExecuteLoc(def_k_AGNumeric,offset,global,local)

   return(true);
  }
//+------------------------------------------------------------------+
//| NumericOK - Read the numeric validation result                   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenEvaluatorOCL::NumericOK(void)
  {
   setBuffer(def_k_AGNumericFinish,0,m_cNumeric.GetIndex())
   uint offset[]={0};
   uint global[]={1};

   kernelExecute(def_k_AGNumericFinish,offset,global)
   float status[1];

   if(!OpenCL.BufferRead(m_cNumeric.GetIndex(),status,0,0,1))
      ReturnFalse;

   return(status[0]==AG_OK);
  }
//+------------------------------------------------------------------+
//| Configure - Configure device state                               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Configure(COpenCLMy*context,const int inputs,const int hidden,const int capacity)
  {
   if(!context||inputs<1||hidden<1||capacity<1||(long)capacity*hidden*6>INT_MAX||
      (long)(inputs+hidden+1)*hidden*4>INT_MAX)
      ReturnFalse;

   if(!CNeuronLSTMOCL::Init(0,0,context,hidden,ADAM,1)||!SetInputs(inputs))
      ReturnFalse;

   return(RestoreWorkspace(context,capacity));
  }
//+------------------------------------------------------------------+
//| RestoreWorkspace - Allocate transient recurrent buffers          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::RestoreWorkspace(COpenCLMy*context,const int capacity)
  {
   if(context!=OpenCL||capacity<1||m_iInputs<1||m_iVariables!=1||
      (long)capacity*Neurons()*6>INT_MAX)
      ReturnFalse;

   int hidden=(int)Neurons();

   m_iCapacity=capacity;
   long local_bytes=0;
   long state_bytes=(long)hidden*22*sizeof(float);

   //--- one local-state implementation on every OpenCL device.
   if(!OpenCL.GetLocalMemorySize(local_bytes)||state_bytes>local_bytes||state_bytes>INT_MAX)
     {
      PrintFormat("AlphaGen LSTM local memory insufficient: required=%I64d available=%I64d",
                  state_bytes,local_bytes);
      ReturnFalse;
     }

   if(!m_cTape.BufferInit(capacity*hidden*6,0)||!m_cTape.BufferCreate(OpenCL)||
      !m_cAnchor.BufferInit(hidden*2,0)||!m_cAnchor.BufferCreate(OpenCL)||
      !m_cSequence.BufferInit(capacity*hidden,0)||!m_cSequence.BufferCreate(OpenCL)||
      !m_cSequenceGradient.BufferInit(capacity*hidden,0)||!m_cSequenceGradient.BufferCreate(OpenCL)||
      !m_cBPTTWork.BufferInit(capacity*hidden*4,0)||!m_cBPTTWork.BufferCreate(OpenCL))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Sequence - Run the recurrent forward pass                        |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Sequence(CBufferFloat*source,const int length)
  {
   if(!source||length<1||length>m_iCapacity||(long)length*m_iInputs>source.Total())
      ReturnFalse;

   m_iLength=length;
   setBuffer(def_k_AGLSTMForward,def_ag_AGLSTMForward_input,source.GetIndex())
   setArgument(def_k_AGLSTMForward,def_ag_AGLSTMForward_length,length)
   setArgument(def_k_AGLSTMForward,def_ag_AGLSTMForward_inputs,m_iInputs)
   setArgument(def_k_AGLSTMForward,def_ag_AGLSTMForward_hidden,(int)Neurons())
   setBuffer(def_k_AGLSTMForward,def_ag_AGLSTMForward_weights,m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_AGLSTMForward,def_ag_AGLSTMForward_anchor,m_cAnchor.GetIndex())
   setBuffer(def_k_AGLSTMForward,def_ag_AGLSTMForward_tape,m_cTape.GetIndex())
   setBuffer(def_k_AGLSTMForward,def_ag_AGLSTMForward_output,m_cSequence.GetIndex())
   if(!OpenCL.SetArgumentLocalMemory(def_k_AGLSTMForward,def_ag_AGLSTMForward_state,
                                    (int)((long)Neurons()*22*sizeof(float))))
      ReturnFalse;

   uint lanes=OpenCL.AlphaGenGroupSize(def_k_AGLSTMForward,128,4);

   if(lanes==0)
      ReturnFalse;

   uint offset[]={0};
   uint global[]={lanes};
   uint local[]={lanes};

   kernelExecuteLoc(def_k_AGLSTMForward,offset,global,local)

   return(true);
  }
//+------------------------------------------------------------------+
//| Backward - Compute sequence gradients                            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Backward(CBufferFloat*source,CBufferFloat*input_gradient)
  {
   if(!source||!input_gradient||m_iLength<1||(long)m_iLength*m_iInputs>source.Total()||
      (long)m_iLength*m_iInputs>input_gradient.Total())
      ReturnFalse;

   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_input,source.GetIndex())
   setArgument(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_length,m_iLength)
   setArgument(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_inputs,m_iInputs)
   setArgument(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_hidden,(int)Neurons())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_weights,m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_anchor,m_cAnchor.GetIndex())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_tape,m_cTape.GetIndex())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_gradient,m_cSequenceGradient.GetIndex())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_input_gradient,input_gradient.GetIndex())
   setBuffer(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_gate_gradient,m_cBPTTWork.GetIndex())
   long local_bytes=0;
   long state_bytes=(long)Neurons()*6*sizeof(float);

   if(!OpenCL.GetLocalMemorySize(local_bytes)||local_bytes<=state_bytes)
      ReturnFalse;

   long max_columns=(local_bytes-state_bytes)/(4*sizeof(float));
   int  tile_columns=(int)MathMin((long)m_iInputs+Neurons(),max_columns);
   long scratch_bytes=state_bytes+(long)tile_columns*4*sizeof(float);

   if(tile_columns<1||scratch_bytes>INT_MAX||
      !OpenCL.SetArgumentLocalMemory(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_scratch,
                                    (int)scratch_bytes))
      ReturnFalse;

   setArgument(def_k_AGLSTMBackward,def_ag_AGLSTMBackward_tile_columns,tile_columns)
   uint lanes=OpenCL.AlphaGenGroupSize(def_k_AGLSTMBackward,256,4);

   if(lanes==0)
      ReturnFalse;

   uint offset[]={0};
   uint global[]={lanes};
   uint local[]={lanes};

   kernelExecuteLoc(def_k_AGLSTMBackward,offset,global,local)
   setBuffer(def_k_AGLSTMWeightGradient,0,source.GetIndex())
   setArgument(def_k_AGLSTMWeightGradient,1,m_iLength)
   setArgument(def_k_AGLSTMWeightGradient,2,m_iInputs)
   setArgument(def_k_AGLSTMWeightGradient,3,(int)Neurons())
   setBuffer(def_k_AGLSTMWeightGradient,4,m_cAnchor.GetIndex())
   setBuffer(def_k_AGLSTMWeightGradient,5,m_cTape.GetIndex())
   setBuffer(def_k_AGLSTMWeightGradient,6,m_cBPTTWork.GetIndex())
   setBuffer(def_k_AGLSTMWeightGradient,7,m_iWeightsGradient)
   lanes=OpenCL.AlphaGenGroupSize(def_k_AGLSTMWeightGradient,64,1);

   if(lanes==0)
      ReturnFalse;

   uint count=(uint)((m_iInputs+Neurons()+1)*Neurons()*4);

   global[0]=((count+lanes-1)/lanes)*lanes;
   local[0]=lanes;
   kernelExecuteLoc(def_k_AGLSTMWeightGradient,offset,global,local)

   return(true);
  }
//+------------------------------------------------------------------+
//| Apply - Apply Adam to recurrent weights                          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::Apply(void)
  {
   setBuffer(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_weights,m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_weights_gradient,m_iWeightsGradient)
   setBuffer(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_matrix_m,m_cFirstMomentumLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_matrix_v,m_cSecondMomentumLSTM.GetIndex())
   setArgument(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_l,lr)
   setArgument(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_b1,b1)
   setArgument(def_k_LSTM_UpdateWeightsAdam,def_k_lstmuw_b2,b2)
   uint offset[]={0,0};
   uint global[]={(uint)(m_iInputs+Neurons()+1),Neurons()*4};

   kernelExecute(def_k_LSTM_UpdateWeightsAdam,offset,global)
   t++;

   return(true);
  }
//+------------------------------------------------------------------+
//| ResetAnchor - Clear the recurrent anchor                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::ResetAnchor(void)
  {
   for(int i=0;i<m_cAnchor.Total();i++)
      if(!m_cAnchor.Update(i,0))
         ReturnFalse;

   if(!m_cAnchor.BufferWrite())
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| SetAnchor - Restore the incoming recurrent anchor                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::SetAnchor(CBufferFloat*anchor)
  {
   return(anchor&&anchor.Total()==m_cAnchor.Total()&&

          CopyBufferRaw(anchor,GetPointer(m_cAnchor),m_cAnchor.Total()));
  }
//+------------------------------------------------------------------+
//| CommitAnchor - Commit the final recurrent state                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenLSTMOCL::CommitAnchor(void)
  {
   if(m_iLength<1)
      ReturnFalse;

   setBuffer(def_k_AGSequence,def_ag_AGSequence_input,m_cTape.GetIndex())
   setBuffer(def_k_AGSequence,def_ag_AGSequence_output,m_cAnchor.GetIndex())
   setArgument(def_k_AGSequence,def_ag_AGSequence_length,m_iLength)
   setArgument(def_k_AGSequence,def_ag_AGSequence_hidden,(int)Neurons())
   setArgument(def_k_AGSequence,def_ag_AGSequence_mode,4)
   setBuffer(def_k_AGSequence,def_ag_AGSequence_account,m_cAnchor.GetIndex())
   uint offset[]={0};
   uint global[]={(uint)m_cAnchor.Total()};

   kernelExecute(def_k_AGSequence,offset,global)

   return(true);
  }

#endif // NEURONET_ALPHA_GEN_MQH
