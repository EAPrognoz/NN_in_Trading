//+------------------------------------------------------------------+
//| Device-resident Prepared/Active pool and normalization state.     |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGENPOOL_MQH
#define NEURONET_ALPHAGENPOOL_MQH
//+------------------------------------------------------------------+
//| Allocate geometry from the single application configuration.     |
//+------------------------------------------------------------------+
bool CAlphaGenPool::Configure(COpenCLMy *context,const float &config[])
  {
   if(!context || ArraySize(config)!=AG_CFG_SIZE)
      ReturnFalse;
   for(int i=0;i<AG_CFG_SIZE;i++)
      if(!MathIsValidNumber(config[i]))
         ReturnFalse;
   // Reconfiguration must not retain indices belonging to an earlier context.
   m_cEvaluationWork.Clear();
   m_cEvaluated.Clear();
   if(!m_cEvaluationWork.BufferCreate(context) || !m_cEvaluated.BufferCreate(context))
      ReturnFalse;
   OpenCL=context;
   for(int i=AG_CFG_W;i<=AG_CFG_CONTEXT;i++)
      if(config[i]!=(float)(int)config[i])
         ReturnFalse;
   m_iLookback=(int)config[AG_CFG_LOOKBACK];
   m_iWindow=(int)config[AG_CFG_W];m_iTokens=(int)config[AG_CFG_MAX_TOKENS];
   m_iCapacity=(int)config[AG_CFG_POOL]+1;m_iIC=(int)config[AG_CFG_IC_WINDOW];
   m_iNorm=(int)config[AG_CFG_NORM_WINDOW];m_iContext=(int)config[AG_CFG_CONTEXT];
   long bank=10+(long)m_iCapacity*(13+m_iTokens);
   long ring=11+(long)m_iNorm+m_iContext;
   long stack=(long)m_iWindow*m_iTokens;
   long work=stack+MathMax(stack,3*(m_iTokens+1))+4*(long)m_iWindow+m_iIC+7+
             (long)m_iCapacity*m_iIC+(long)m_iCapacity*(m_iCapacity+1)+m_iCapacity+3+m_iWindow;
   if(m_iLookback<0 || m_iWindow<1 || m_iTokens<2 || m_iCapacity<2 || m_iIC<2 || m_iNorm<2 ||
      m_iContext<1 || config[AG_CFG_HORIZON]<1 || config[AG_CFG_RIDGE]<=0 ||
      config[AG_CFG_SIGNAL_CLIP]<=0 || config[AG_CFG_EPS_VAR]<=0 ||
      m_iWindow<config[AG_CFG_LOOKBACK]+MathMax(m_iIC+config[AG_CFG_HORIZON],m_iNorm) ||
      bank*4>INT_MAX || ring*2>INT_MAX || work>INT_MAX)
      ReturnFalse;
   m_iBankSize=(int)bank;m_iRingSize=(int)ring;m_iWorkSize=(int)work;
   if(!m_cConfig.AssignArray(config) || !m_cConfig.BufferCreate(OpenCL) ||
      !Allocate(m_cBanks,4*m_iBankSize) || !Allocate(m_cRings,2*m_iRingSize) ||
      !Allocate(m_cWork,m_iWorkSize) || !Allocate(m_cControl,AG_POOL_CONTROL_SIZE) ||
      !Allocate(m_cSignal,1) || !Allocate(m_cContext,m_iContext) ||
      !Allocate(m_cPreparedSignal,1) || !Allocate(m_cPreparedContext,m_iContext) || !Allocate(m_cDummy,1))
      ReturnFalse;
   // The first accepted formula receives ID one; zero remains unassigned.
   if(!m_cBanks.Update(1,1) || !m_cBanks.BufferWrite())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Dispatch a transaction; only the final result scalars need reading. |
//+------------------------------------------------------------------+
bool CAlphaGenPool::Transaction(CBufferFloat *market,CBufferFloat *tokens,const int length,const int mode)
  {
   if(!OpenCL || !market || !tokens || m_iWindow<1 ||
      ((mode==0 || mode==1) && market.Total()<5*m_iWindow) ||
      (mode==0 && (length<2 || length>m_iTokens || tokens.Total()<length)))
      ReturnFalse;
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_market,market.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_tokens,tokens.GetIndex())
   setArgument(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_length,length)
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_banks,m_cBanks.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_work,m_cWork.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_rings,m_cRings.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_control,m_cControl.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_signal,m_cSignal.GetIndex())
   setBuffer(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_context,m_cContext.GetIndex())
   setArgument(def_k_AGPoolTransaction,def_ag_AGPoolTransaction_mode,mode)
   uint global_work_offset[]={0},global_work_size[]={1};
   kernelExecute(def_k_AGPoolTransaction,global_work_offset,global_work_size)
   return(true);
  }
//+------------------------------------------------------------------+
//| Evaluate formulas in parallel and apply ordered pool updates.    |
//+------------------------------------------------------------------+
bool CAlphaGenPool::CommitSearchBatch(CBufferFloat *market,CBufferFloat *candidates,const int count,
                                      const int first_row,CBufferFloat *prefixes,CBufferFloat *masks,
                                      CBufferFloat *samples,CBufferFloat *metadata,CBufferFloat *gae,
                                      CBufferFloat *summary)
  {
   if(!OpenCL || !market || !candidates || !samples || !summary || count<1 ||
      market.Total()<5*m_iWindow || (long)candidates.Total()<(long)count*(1+(long)m_iTokens*(4+AG_VOCABULARY)))
      ReturnFalse;
   long stride=(long)m_iWindow*m_iTokens+MathMax((long)m_iWindow*m_iTokens,3*(long)(m_iTokens+1));
   long work_count=stride*count,cache_count=2*(long)m_iWindow*count;
   if(work_count>INT_MAX || cache_count>INT_MAX ||
      (m_cEvaluationWork.Total()<work_count && !Allocate(m_cEvaluationWork,(int)work_count)) ||
      (m_cEvaluated.Total()<cache_count && !Allocate(m_cEvaluated,(int)cache_count)))
      ReturnFalse;
   setBuffer(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_market,market.GetIndex())
   setBuffer(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_candidates,candidates.GetIndex())
   setArgument(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_batch,count)
   setBuffer(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_workspace,m_cEvaluationWork.GetIndex())
   setBuffer(def_k_AGPoolEvaluateBatch,def_ag_AGPoolEvaluateBatch_evaluated,m_cEvaluated.GetIndex())
   long kernel_limit=OpenCL.GetKernelInfoInteger(def_k_AGPoolEvaluateBatch,CL_KERNEL_WORK_GROUP_SIZE);
   long commit_limit=OpenCL.GetKernelInfoInteger(def_k_AGPoolCommitBatch,CL_KERNEL_WORK_GROUP_SIZE);
   long lane_limit=MathMin(OpenCL.GetMaxWorkGroupSize(),OpenCL.GetMaxLocalSize(0));
   lane_limit=MathMin(lane_limit,MathMin(kernel_limit,commit_limit));
   if(lane_limit<1)
      ReturnFalse;
   // Tree reductions in commit use a power-of-two group, at most 64 lanes.
   uint lanes=1;
   while(lanes<64 && 2*(long)lanes<=lane_limit)
      lanes*=2;
   uint eval_offset[]={0},eval_global[]={(uint)count*lanes},eval_local[]={lanes};
#ifdef ALPHAGEN_STUDY
   Comment(StringFormat("\n\nAlphaGen Study | GPU formula evaluation | %d parallel groups x %d lanes",count,lanes));
   ChartRedraw();
#endif
   ulong eval_started=GetMicrosecondCount();
   if(!OpenCL.Execute(def_k_AGPoolEvaluateBatch,1,eval_offset,eval_global,eval_local) ||
      !OpenCL.WaitAlphaGenKernel(def_k_AGPoolEvaluateBatch))
      ReturnFalse;
   if(first_row==0)
      PrintFormat("AlphaGen parallel formula evaluation: groups=%d lanes=%d us=%I64u",count,lanes,GetMicrosecondCount()-eval_started);
   int capacity = samples.Total();
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_market, market.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_candidates, candidates.GetIndex())
   setArgument(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_batch, count)
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_cfg, m_cConfig.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_banks, m_cBanks.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_work, m_cWork.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_rings, m_cRings.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_control, m_cControl.GetIndex())
   setArgument(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_first_row, first_row)
   setArgument(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_capacity, capacity)
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_prefixes, prefixes.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_masks, masks.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_samples, samples.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_metadata, metadata.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_gae, gae.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_summary, summary.GetIndex())
   setBuffer(def_k_AGPoolCommitBatch, def_ag_AGPoolCommitBatch_evaluated, m_cEvaluated.GetIndex())
   uint offset[] = { 0 }, global[] = { lanes }, local[] = { lanes };
   for(int candidate=0; candidate<count; candidate++)
     {
      offset[0]=(uint)candidate*lanes;
      if(first_row==0)
         PrintFormat("AlphaGen pool commit BEGIN candidate=%d/%d first_row=%d parallel_lanes=%d",candidate+1,count,first_row,lanes);
#ifdef ALPHAGEN_STUDY
      Comment(StringFormat("\n\nAlphaGen Study | GPU pool commit | candidate %d/%d | %d parallel lanes | PPO first row %d",candidate+1,count,lanes,first_row));
      ChartRedraw();
#endif
      ResetLastError();
      ulong started=GetMicrosecondCount();
      bool queued=OpenCL.Execute(def_k_AGPoolCommitBatch,1,offset,global,local);
      int enqueue_error=GetLastError();
      ulong enqueue_us=GetMicrosecondCount()-started;
      bool complete=queued && OpenCL.WaitAlphaGenKernel(def_k_AGPoolCommitBatch);
      if(first_row==0 || !complete)
         PrintFormat("AlphaGen pool commit END candidate=%d/%d queued=%d enqueue_error=%d enqueue_us=%I64u complete=%d total_us=%I64u",
                     candidate+1,count,(int)queued,enqueue_error,enqueue_us,(int)complete,GetMicrosecondCount()-started);
      if(!complete)
         ReturnFalse;
     }
   if(!summary.BufferRead() || summary.At(0) != AG_OK)
      ReturnFalse;
   for(int i = 0; i < AG_POOL_CONTROL_SIZE; i++)
      if(!m_cControl.Update(i, summary.At(4 + i)))
         ReturnFalse;
   return(true);
  }

bool CAlphaGenPool::SignalStep(CBufferFloat *market,const int mode,const bool first)
  {
   if(!OpenCL || !market || market.Total()<5*m_iWindow)
      ReturnFalse;
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_market,market.GetIndex())
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_cfg,m_cConfig.GetIndex())
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_banks,m_cBanks.GetIndex())
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_work,m_cWork.GetIndex())
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_rings,m_cRings.GetIndex())
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_control,m_cControl.GetIndex())
   int signal_index=mode==2?m_cPreparedSignal.GetIndex():m_cSignal.GetIndex();
   int context_index=mode==2?m_cPreparedContext.GetIndex():m_cContext.GetIndex();
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_signal,signal_index)
   setBuffer(def_k_AGPoolSignal,def_ag_AGPoolSignal_context,context_index)
   setArgument(def_k_AGPoolSignal,def_ag_AGPoolSignal_mode,mode)
   int start=first?(mode==2?m_iLookback:MathMax(m_iLookback,m_iWindow-m_iContext)):m_iWindow-1;
   setArgument(def_k_AGPoolSignal,def_ag_AGPoolSignal_start,start)
   uint global_work_offset[]={0},global_work_size[]={1};
   kernelExecute(def_k_AGPoolSignal,global_work_offset,global_work_size)
   return(true);
  }
//+------------------------------------------------------------------+
//| Serialization is the only full device-state readback.            |
//+------------------------------------------------------------------+
bool CAlphaGenPool::Save(const int file_handle)
  {
   if(!OpenCL || file_handle==INVALID_HANDLE)
      ReturnFalse;
   dFileWriteUInt(file_handle,m_iBankSize)
   dFileWriteUInt(file_handle,m_iRingSize)
   if(!m_cBanks.BufferRead() || !m_cBanks.Save(file_handle))
      ReturnFalse;
   if(!m_cRings.BufferRead() || !m_cRings.Save(file_handle))
      ReturnFalse;
   if(!m_cControl.BufferRead() || !m_cControl.Save(file_handle))
      ReturnFalse;
   if(!m_cSignal.BufferRead() || !m_cSignal.Save(file_handle))
      ReturnFalse;
   if(!m_cContext.BufferRead() || !m_cContext.Save(file_handle))
      ReturnFalse;
   if(!m_cPreparedSignal.BufferRead() || !m_cPreparedSignal.Save(file_handle) ||
      !m_cPreparedContext.BufferRead() || !m_cPreparedContext.Save(file_handle))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Load into temporary host buffers; validate shape before commit.  |
//+------------------------------------------------------------------+
bool CAlphaGenPool::Load(const int file_handle)
  {
   if(!OpenCL || file_handle==INVALID_HANDLE)
      ReturnFalse;
   uint bank=0,ring=0;
   dFileReadUInt(file_handle,bank)
   dFileReadUInt(file_handle,ring)
   if(bank!=(uint)m_iBankSize || ring!=(uint)m_iRingSize)
      ReturnFalse;
   CBufferFloat banks,rings,control,signal,context,prepared_signal,prepared_context;
   if(!banks.Load(file_handle) || banks.Total()!=4*m_iBankSize ||
      !rings.Load(file_handle) || rings.Total()!=2*m_iRingSize ||
      !control.Load(file_handle) || control.Total()!=AG_POOL_CONTROL_SIZE ||
      !signal.Load(file_handle) || signal.Total()!=1 ||
      !context.Load(file_handle) || context.Total()!=m_iContext ||
      !prepared_signal.Load(file_handle) || prepared_signal.Total()!=1 ||
      !prepared_context.Load(file_handle) || prepared_context.Total()!=m_iContext)
      ReturnFalse;
   // Every persisted scalar must be finite before any working state changes.
   for(int i=0;i<banks.Total();i++) if(!MathIsValidNumber(banks.At(i))) {ReturnFalse;}
   for(int i=0;i<rings.Total();i++) if(!MathIsValidNumber(rings.At(i))) {ReturnFalse;}
   for(int i=0;i<control.Total();i++) if(!MathIsValidNumber(control.At(i))) {ReturnFalse;}
   for(int i=0;i<context.Total();i++) if(!MathIsValidNumber(context.At(i)) || MathAbs(context.At(i))>1) {ReturnFalse;}
   for(int i=0;i<prepared_context.Total();i++) if(!MathIsValidNumber(prepared_context.At(i)) || MathAbs(prepared_context.At(i))>1) {ReturnFalse;}
   if(!MathIsValidNumber(signal.At(0)) || !MathIsValidNumber(prepared_signal.At(0)))
      ReturnFalse;
   for(int b=0;b<4;b++)
     {
      int offset=b*m_iBankSize,count=(int)banks.At(offset);
      if(count<0 || count>(b==2?m_iCapacity:m_iCapacity-1) || banks.At(offset)!=(float)count)
         ReturnFalse;
      for(int k=1;k<9;k++)
        {
         float limb=banks.At(offset+k);
         if(limb<0 || limb>65535 || limb!=(float)(int)limb)
            ReturnFalse;
        }
      if(banks.At(offset+9)!=0 && banks.At(offset+9)!=1)
         ReturnFalse;
      for(int j=0;j<count;j++)
        {
         int record=offset+10+j*(13+m_iTokens),length=(int)banks.At(record);
         if(length<2 || length>m_iTokens || banks.At(record)!=(float)length)
            ReturnFalse;
         for(int k=1;k<5;k++)
           {
            float limb=banks.At(record+k);
            if(limb<0 || limb>65535 || limb!=(float)(int)limb)
               ReturnFalse;
           }
         bool zero_id=true;
         for(int k=1;k<5;k++) if(banks.At(record+k)!=0) zero_id=false;
         if(zero_id)
            ReturnFalse;
         if(b!=2)
           {
            bool greater=false;
            for(int k=3;k>=0;k--)
              {
               float next=banks.At(offset+1+k),id=banks.At(record+1+k);
               if(next!=id) {greater=next>id;break;}
              }
            if(!greater)
               ReturnFalse;
           }
         for(int previous=0;previous<j;previous++)
           {
            int other=offset+10+previous*(13+m_iTokens);
            bool same_id=true,same_formula=banks.At(other)==(float)length;
            for(int k=1;k<5;k++) if(banks.At(record+k)!=banks.At(other+k)) same_id=false;
            for(int k=0;k<length && same_formula;k++)
               if(banks.At(record+5+k)!=banks.At(other+5+k)) same_formula=false;
            if(same_id || same_formula)
               ReturnFalse;
           }
         if(banks.At(record+5+length-1)!=AG_STOP)
            ReturnFalse;
         if(banks.At(offset+9)==1 && (banks.At(record+5+m_iTokens)<=0 ||
            banks.At(record+7+m_iTokens)<=0 || banks.At(record+9+m_iTokens)<=0 ||
            banks.At(record+10+m_iTokens)!=AG_OK))
            ReturnFalse;
         for(int k=0;k<length;k++)
           {
            float token=banks.At(record+5+k);
            if(token<0 || token>=AG_VOCABULARY || token!=(float)(int)token)
               ReturnFalse;
           }
        }
     }
   for(int b=0;b<2;b++)
     {
      int offset=b*m_iRingSize;
      float count=rings.At(offset),position=rings.At(offset+1),used=rings.At(offset+2);
      if(count<0 || count>m_iNorm || count!=(float)(int)count ||
         position<0 || position>=m_iNorm || position!=(float)(int)position ||
         used<0 || used>m_iContext || used!=(float)(int)used)
         ReturnFalse;
      if(count==m_iNorm && rings.At(offset+10)==AG_OK &&
         (rings.At(offset+3)<=0 || rings.At(offset+5)<=0 || rings.At(offset+7)<=0))
         ReturnFalse;
     }
   if(!m_cBanks.AssignArray(GetPointer(banks)) || !m_cBanks.BufferWrite())
      ReturnFalse;
   if(!m_cRings.AssignArray(GetPointer(rings)) || !m_cRings.BufferWrite())
      ReturnFalse;
   if(!m_cControl.AssignArray(GetPointer(control)) || !m_cControl.BufferWrite())
      ReturnFalse;
   if(!m_cSignal.AssignArray(GetPointer(signal)) || !m_cSignal.BufferWrite())
      ReturnFalse;
   if(!m_cContext.AssignArray(GetPointer(context)) || !m_cContext.BufferWrite() ||
      !m_cPreparedSignal.AssignArray(GetPointer(prepared_signal)) || !m_cPreparedSignal.BufferWrite() ||
      !m_cPreparedContext.AssignArray(GetPointer(prepared_context)) || !m_cPreparedContext.BufferWrite())
      ReturnFalse;
   return(true);
  }
#endif
