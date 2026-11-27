//+------------------------------------------------------------------+
//|                                        NeuroNet_AlphaQCMPool.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCMPOOL_MQH
#define NEURONET_ALPHAQCMPOOL_MQH

enum ENUM_AQCM_POOL_KERNEL
  {
   AQCMPoolOrderedKernel=500,
   AQCMPoolScoreKernel,
   AQCMPoolAdvanceKernel,
   AQCMPoolEvaluatePreparedKernel,
   AQCMPoolHistoryGlobalKernel
  };

//+------------------------------------------------------------------+
//| QCMStatus - Return the failure of the last AlphaQCM pool call    |
//+------------------------------------------------------------------+
int CAlphaGenPool::QCMStatus(void) const
  {
   return(m_bQCMUncertain?AQCM_UPDATE_INCONSISTENT:m_iQCMStatus);
  }

//+------------------------------------------------------------------+
//| QCMValidLoadedActive - Check restored Active metadata on host    |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMValidLoadedActive(void) const
  {
   const float active=m_cBanks.At(m_iBankSize);
   const float fit=m_cBanks.At(m_iBankSize+9);
   const float ready=m_cControl.At(AG_POOL_ACTIVE_READY);
   if(active!=m_cControl.At(AG_POOL_ACTIVE_COUNT)||(ready!=0.0f&&ready!=1.0f)||
      (active==0.0f&&(fit!=0.0f||ready!=0.0f))||(active>0.0f&&fit!=1.0f))
     {
      ReturnFalseEx("Invalid saved AlphaQCM Active count, fit or ready flag")
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMValidateWorkspace - Check the pool and minimum scratch floor  |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMValidateWorkspace(const int candidates)
  {
   if(!OpenCL||m_iWindow<1||m_iContext<1||m_iCapacity<2||m_iTokens<2||candidates<0)
     {
      ReturnFalse
     }
   const long span=(long)m_iWindow+m_iContext-1;
   const long stack=(long)m_iWindow*m_iTokens;
   const long stack_elements=stack+MathMax(stack,3L*(m_iTokens+1));
   const long formula=candidates>0?stack_elements:0;
   const long history_formula=stack_elements+7L*m_iWindow;
   const long history=2L*m_iCapacity*span;
   const long evaluated=2L*m_iWindow*candidates;
   const long persistent=(long)m_cConfig.Total()+m_cBanks.Total()+m_cRings.Total()+
                         m_cWork.Total()+m_cControl.Total()+m_cSignal.Total()+m_cContext.Total()+
                         m_cPreparedSignal.Total()+m_cPreparedContext.Total()+m_cDummy.Total();
   const long elements=persistent+formula+history+evaluated+5L*span+2L*span+1+(candidates>0?3:0);
   const long largest=MathMax(MathMax((long)m_cBanks.Total(),(long)m_cWork.Total()),
                              MathMax(MathMax(formula,history),MathMax(evaluated,5L*span)));
   const long memory=OpenCL.GetDeviceInfoInteger(CL_DEVICE_GLOBAL_MEM_SIZE);
   const long allocation=OpenCL.GetDeviceInfo(0x1010); // CL_DEVICE_MAX_MEM_ALLOC_SIZE
   if(memory<elements*(long)sizeof(float)||allocation<largest*(long)sizeof(float))
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: pool needs %I64d bytes and a %I64d-byte buffer",
                                 elements*(long)sizeof(float),largest*(long)sizeof(float)))
     }
   PrintFormat("AlphaQCM pool workspace floor: bytes=%I64d largest_buffer=%I64d history_bars=%I64d "
               "candidates=%d, global-history fallback adds %I64d bytes at formula chunk=1",
               elements*(long)sizeof(float),largest*(long)sizeof(float),span,candidates,
               (history_formula-formula)*(long)sizeof(float));
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMEnsure - Allocate scratch within the supported buffer range   |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMEnsure(CBufferFloat &buffer,const long count)
  {
   if(count<1||count>INT_MAX/(int)sizeof(float))
     {
      ReturnFalse
     }
   if(buffer.Total()>=count&&buffer.GetIndex()>=0)
      return(true);
   return(Allocate(buffer,(int)count));
  }

// A single physical portion completes before rebinding any donor kernel.
// An unknown completion excludes further dispatch and checkpointing.
//+------------------------------------------------------------------+
//| QCMRun - Run a pool kernel and wait for completion               |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMRun(const int kernel,const int groups)
  {
   if(m_bQCMUncertain||groups<1||!OpenCL)
     {
      m_iQCMStatus=AG_TECHNICAL;
      ReturnFalse
     }
   const uint minimum=kernel==AQCMPoolOrderedKernel?4:2;
   const uint lanes=OpenCL.AlphaGenGroupSize(kernel,64,minimum);
   if(!lanes||(long)groups*lanes>UINT_MAX)
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: pool kernel %s needs at least %u parallel lanes (%d groups)",
                                 OpenCL.GetKernelName(kernel),minimum,groups))
     }
   uint offset[]={0},global[]={(uint)groups*lanes},local[]={lanes};
   m_iQCMStatus=AQCM_OPENCL_ERROR;
   kernelExecuteLoc(kernel,offset,global,local)
   if(!OpenCL.WaitAlphaGenKernel(kernel))
     {
      if(OpenCL.ExecutionStatus(kernel)==CL_COMPLETE)
         return(true);
      m_bQCMUncertain=true;
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMBindAdvance - Bind one chronological slice of the ring        |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMBindAdvance(const int begin,const int end)
  {
   setArgument(AQCMPoolAdvanceKernel,10,begin)
   setArgument(AQCMPoolAdvanceKernel,11,end)
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMReadStatus - Check completed pool control and numeric status  |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMReadStatus(const int changes,const uint offset)
  {
   if(!m_cControl.BufferRead())
     {
      m_iQCMStatus=AQCM_OPENCL_ERROR;
      m_bQCMUncertain=true;
      ReturnFalse
     }
   const float status=m_cControl.At(AG_POOL_STATUS);
   // Matches the private numeric status in AlphaQCM_Pool.cl.
   if(status!=-1.0f&&status!=AG_TECHNICAL)
      return(true);
   if(changes>=0)
     {
      float changed[1];
      if(!OpenCL.BufferRead(changes,changed,offset,0,1)||changed[0]<0.0f)
         m_bQCMUncertain=true;
     }
   m_iQCMStatus=status==-1.0f?AQCM_NUMERIC_ERROR:AG_TECHNICAL;
   ReturnFalseEx(m_iQCMStatus==AQCM_NUMERIC_ERROR?"NUMERIC_ERROR: AlphaQCM pool":"AlphaQCM pool technical failure")
  }

//+------------------------------------------------------------------+
//| QCMCache - Evaluate independent formulas in bounded GPU chunks   |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMCache(CBufferFloat *market,CBufferFloat *source,CBufferFloat *result,
                             const int count,const bool packed)
  {
   long stride=(long)m_iWindow*m_iTokens;
   stride+=MathMax(stride,3*(m_iTokens+1));
   if(!QCMEnsure(result,MathMax(1,2*(long)m_iWindow*count)))
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx("RESOURCE_UNSUPPORTED: AlphaQCM formula outputs")
     }
   if(count==0)
      return(true);
   int chunk=MathMin(8,count);
   while(chunk>0&&!QCMEnsure(m_cEvaluationWork,stride*chunk))
      chunk/=2;
   if(chunk==0)
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx("RESOURCE_UNSUPPORTED: scratch for one AlphaQCM formula")
     }
   m_iQCMStatus=AQCM_OPENCL_ERROR;
   if(packed&&chunk==count)
     {
      setBuffer(def_k_AGPoolEvaluateBatch,0,market.GetIndex())
      setBuffer(def_k_AGPoolEvaluateBatch,1,source.GetIndex())
      setArgument(def_k_AGPoolEvaluateBatch,2,count)
      setBuffer(def_k_AGPoolEvaluateBatch,3,m_cConfig.GetIndex())
      setBuffer(def_k_AGPoolEvaluateBatch,4,m_cEvaluationWork.GetIndex())
      setBuffer(def_k_AGPoolEvaluateBatch,5,result.GetIndex())
      return(QCMRun(def_k_AGPoolEvaluateBatch,count));
     }
   int layout=packed?1:0;
   setBuffer(AQCMPoolEvaluatePreparedKernel,0,market.GetIndex())
   setBuffer(AQCMPoolEvaluatePreparedKernel,1,m_cConfig.GetIndex())
   setBuffer(AQCMPoolEvaluatePreparedKernel,2,source.GetIndex())
   setBuffer(AQCMPoolEvaluatePreparedKernel,3,m_cEvaluationWork.GetIndex())
   setBuffer(AQCMPoolEvaluatePreparedKernel,4,result.GetIndex())
   setArgument(AQCMPoolEvaluatePreparedKernel,6,layout)
   for(int first=0;first<count;first+=chunk)
     {
      setArgument(AQCMPoolEvaluatePreparedKernel,5,first)
      if(!QCMRun(AQCMPoolEvaluatePreparedKernel,MathMin(chunk,count-first)))
        {
         ReturnFalse
        }
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMOrdered - Complete one ordered fit transaction                |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMOrdered(CBufferFloat *market,CBufferFloat *packed,CBufferFloat *values,
                               const int index,const int mode)
  {
   if(!QCMEnsure(m_cQCMFit,3))
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx("RESOURCE_UNSUPPORTED: AlphaQCM fit status buffer")
     }
   m_iQCMStatus=AQCM_OPENCL_ERROR;
   setBuffer(AQCMPoolOrderedKernel,0,market.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,1,m_cConfig.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,2,m_cBanks.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,3,m_cWork.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,4,m_cRings.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,5,m_cControl.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,6,m_cQCMFit.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,7,packed.GetIndex())
   setBuffer(AQCMPoolOrderedKernel,8,values.GetIndex())
   setArgument(AQCMPoolOrderedKernel,9,index)
   setArgument(AQCMPoolOrderedKernel,10,mode)
   if(!QCMRun(AQCMPoolOrderedKernel,1))
     {
      ReturnFalse
     }
   if(!QCMReadStatus(-1))
     {
      ReturnFalse
     }
   if(!m_cQCMFit.BufferRead())
     {
      m_bQCMUncertain=true;
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMRefresh - Refresh the pool's pinned market baseline           |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMRefresh(CBufferFloat *market,const bool pin,const ulong window)
  {
   m_iQCMStatus=AG_TECHNICAL;
   if(m_bQCMUncertain||!OpenCL||!market||(long)market.Total()<5*(long)m_iWindow)
     {
      ReturnFalse
     }
   m_iQCMBatch=0;
   if(window!=0&&m_bQCMBaseline&&window==m_uQCMWindow&&market.GetIndex()==m_iQCMMarket)
      return(QCMOrdered(market,GetPointer(m_cDummy),GetPointer(m_cDummy),0,pin?3:4));
   m_bQCMBaseline=false;
   if(!m_cControl.BufferRead())
     {
      m_iQCMStatus=AQCM_OPENCL_ERROR;
      ReturnFalse
     }
   int count=(int)m_cControl.At(AG_POOL_PREPARED_COUNT);
   if(count<0||count>=m_iCapacity||
      !QCMCache(market,GetPointer(m_cBanks),GetPointer(m_cHistoryValues),count,false)||
      !QCMOrdered(market,GetPointer(m_cDummy),GetPointer(m_cHistoryValues),0,pin?2:0))
     {
      ReturnFalse
     }
   m_bQCMBaseline=m_cControl.At(AG_POOL_STATUS)==AG_OK;
   m_uQCMWindow=window;m_iQCMMarket=market.GetIndex();
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMEvaluateBatch - Evaluate the packed candidate batch           |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMEvaluateBatch(CBufferFloat *market,CBufferFloat *packed,const int count)
  {
   m_iQCMStatus=AG_TECHNICAL;
   m_iQCMBatch=0;
   if(m_bQCMUncertain||!m_bQCMBaseline||!market||!packed||count<1||
      m_iQCMMarket!=market.GetIndex()||(long)market.Total()<5*(long)m_iWindow||
      packed.Total()<(long)count*(1+(long)m_iTokens*(4+AG_VOCABULARY))||
      !QCMCache(market,packed,GetPointer(m_cEvaluated),count,true))
     {
      ReturnFalse
     }
   m_iQCMBatch=count;m_iQCMNext=0;m_iQCMPacked=packed.GetIndex();
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMCommitCandidate - Commit the next ordered candidate           |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMCommitCandidate(CBufferFloat *market,CBufferFloat *packed,const int count,const int index)
  {
   m_iQCMStatus=AG_TECHNICAL;
   if(m_bQCMUncertain||!m_bQCMBaseline||!market||!packed||
      market.GetIndex()!=m_iQCMMarket||packed.GetIndex()!=m_iQCMPacked||
      count!=m_iQCMBatch||index!=m_iQCMNext||index<0||index>=count||
      !QCMOrdered(market,packed,GetPointer(m_cEvaluated),index,1))
     {
      ReturnFalse
     }
   if(m_cControl.At(AG_POOL_STATUS)!=AG_TECHNICAL)
      m_iQCMNext++;
   return(true);
  }

//+------------------------------------------------------------------+
//| QCMSignalStep - Dispatch one pool signal step                    |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMSignalStep(CBufferFloat *market,const int mode,const bool first,const int count)
  {
   m_iQCMStatus=AG_TECHNICAL;
   int span=count>0?count:m_iWindow;
   if(m_bQCMUncertain||!OpenCL||!market||span<m_iWindow||(long)market.Total()<5*(long)span)
     {
      ReturnFalse
     }
   int start=first?(mode>=2?m_iLookback:MathMax(m_iLookback,span-m_iContext)):span-1;
   long available=0;
   m_iQCMStatus=AQCM_OPENCL_ERROR;
   if(!OpenCL.GetLocalMemorySize(available))
     {
      ReturnFalse
     }
   if(!QCMEnsure(m_cHistoryValues,2*(long)m_iCapacity*span)||
      !QCMEnsure(m_cQCMScores,2*(long)span+1))
     {
      m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalseEx("RESOURCE_UNSUPPORTED: AlphaQCM signal values and scores")
     }
   long tile_budget=available>512?available-512:0;
   int tile=(int)MathMin(span,tile_budget/(8*(long)m_iTokens));
   while(tile>m_iLookback)
     {
      long slots=(long)m_iTokens*tile;
      long bytes=4*(slots+MathMax(slots,3*(m_iTokens+1)));
      if(bytes>INT_MAX||!OpenCL.SetArgumentLocalMemory(def_k_AGPoolEvaluateHistory,8,(int)bytes))
        {
         tile/=2;
         continue;
        }
      long actual=OpenCL.GetKernelInfoInteger(def_k_AGPoolEvaluateHistory,CL_KERNEL_LOCAL_MEM_SIZE);
      if(actual<0)
        {
         ReturnFalse
        }
      if(actual<=available)
         break;
      tile-=(int)MathMax(1,(actual-available+8*m_iTokens-1)/(8*m_iTokens));
     }
   if(tile>m_iLookback)
     {
      setBuffer(def_k_AGPoolEvaluateHistory,0,market.GetIndex())
      setBuffer(def_k_AGPoolEvaluateHistory,1,m_cConfig.GetIndex())
      setBuffer(def_k_AGPoolEvaluateHistory,2,m_cBanks.GetIndex())
      setBuffer(def_k_AGPoolEvaluateHistory,3,m_cHistoryValues.GetIndex())
      setArgument(def_k_AGPoolEvaluateHistory,4,mode)
      setArgument(def_k_AGPoolEvaluateHistory,5,span)
      setArgument(def_k_AGPoolEvaluateHistory,6,start)
      setArgument(def_k_AGPoolEvaluateHistory,7,tile)
      if(!QCMRun(def_k_AGPoolEvaluateHistory,m_iCapacity))
        {
         ReturnFalse
        }
     }
   else
     {
      long stack=(long)m_iWindow*m_iTokens;
      long stride=stack+MathMax(stack,3*(m_iTokens+1))+7*(long)m_iWindow;
      int chunk=MathMin(8,m_iCapacity);
      while(chunk>0&&!QCMEnsure(m_cEvaluationWork,stride*chunk))
         chunk/=2;
      if(!chunk)
        {
         m_iQCMStatus=AQCM_RESOURCE_UNSUPPORTED;
         ReturnFalseEx("RESOURCE_UNSUPPORTED: global scratch for one AlphaQCM history formula")
        }
      setBuffer(AQCMPoolHistoryGlobalKernel,0,market.GetIndex())
      setBuffer(AQCMPoolHistoryGlobalKernel,1,m_cConfig.GetIndex())
      setBuffer(AQCMPoolHistoryGlobalKernel,2,m_cBanks.GetIndex())
      setBuffer(AQCMPoolHistoryGlobalKernel,3,m_cHistoryValues.GetIndex())
      setBuffer(AQCMPoolHistoryGlobalKernel,4,m_cEvaluationWork.GetIndex())
      setArgument(AQCMPoolHistoryGlobalKernel,6,mode)
      setArgument(AQCMPoolHistoryGlobalKernel,7,start)
      setArgument(AQCMPoolHistoryGlobalKernel,8,span)
      for(int base=0;base<m_iCapacity;base+=chunk)
        {
         setArgument(AQCMPoolHistoryGlobalKernel,5,base)
         if(!QCMRun(AQCMPoolHistoryGlobalKernel,MathMin(chunk,m_iCapacity-base)))
           {
            ReturnFalse
           }
        }
     }

   setBuffer(AQCMPoolScoreKernel,0,m_cConfig.GetIndex())
   setBuffer(AQCMPoolScoreKernel,1,m_cBanks.GetIndex())
   setBuffer(AQCMPoolScoreKernel,2,m_cHistoryValues.GetIndex())
   setBuffer(AQCMPoolScoreKernel,3,m_cQCMScores.GetIndex())
   setArgument(AQCMPoolScoreKernel,4,start)
   setArgument(AQCMPoolScoreKernel,5,span)
   setArgument(AQCMPoolScoreKernel,6,mode)
   if(!QCMRun(AQCMPoolScoreKernel,span-start))
     {
      ReturnFalse
     }
   int signal_index=mode==2?m_cPreparedSignal.GetIndex():m_cSignal.GetIndex();
   int context_index=mode==2?m_cPreparedContext.GetIndex():m_cContext.GetIndex();
   setBuffer(AQCMPoolAdvanceKernel,0,m_cConfig.GetIndex())
   setBuffer(AQCMPoolAdvanceKernel,1,m_cBanks.GetIndex())
   setBuffer(AQCMPoolAdvanceKernel,2,m_cRings.GetIndex())
   setBuffer(AQCMPoolAdvanceKernel,3,m_cControl.GetIndex())
   setBuffer(AQCMPoolAdvanceKernel,4,signal_index)
   setBuffer(AQCMPoolAdvanceKernel,5,context_index)
   setBuffer(AQCMPoolAdvanceKernel,6,m_cQCMScores.GetIndex())
   setArgument(AQCMPoolAdvanceKernel,7,start)
   setArgument(AQCMPoolAdvanceKernel,8,span)
   setArgument(AQCMPoolAdvanceKernel,9,mode)
   for(int begin=start;begin<span;)
     {
      int end=begin+MathMin(32,span-begin);
      if(!QCMBindAdvance(begin,end)||!QCMRun(AQCMPoolAdvanceKernel,1))
        {
         if(begin!=start)
            m_bQCMUncertain=true;
         ReturnFalse
        }
      begin=end;
     }
   // A numerical failure after a chronological ring update cannot be retried.
   return(QCMReadStatus(m_cQCMScores.GetIndex(),(uint)(2*(long)span*sizeof(float))));
  }

//+------------------------------------------------------------------+
//| QCMPrepareAppend - Prepare the signal history for appending      |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMPrepareAppend(CBufferFloat *market,const bool first)
  {
   if(first&&m_iHistoryCount>0)
     {
      if(!QCMSignalStep(GetPointer(m_cHistory),2,true,m_iHistoryCount))
        {
         ReturnFalse
        }
      m_iHistoryCount=0;
      return(true);
     }
   return(QCMSignalStep(market,2,first,0));
  }

//+------------------------------------------------------------------+
//| QCMRewarmActive - Rebuild active signal state for a new stream   |
//+------------------------------------------------------------------+
bool CAlphaGenPool::QCMRewarmActive(CBufferFloat *market)
  {
   // A new stream may reuse a timestamp and a device market buffer.
   m_bQCMBaseline=false;m_iQCMBatch=0;m_uQCMWindow=0;
   if(m_iHistoryCount>0)
     {
      if(!QCMSignalStep(GetPointer(m_cHistory),3,true,m_iHistoryCount))
        {
         ReturnFalse
        }
      m_iHistoryCount=0;
      return(true);
     }
   return(QCMSignalStep(market,3,true,0));
  }

#endif
