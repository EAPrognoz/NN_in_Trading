//+------------------------------------------------------------------+
//| NeuroNet_AlphaGenComposite.mqh                                   |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGEN_COMPOSITE_MQH
#define NEURONET_ALPHAGEN_COMPOSITE_MQH
//+------------------------------------------------------------------+
//| CNeuronAlphaGenOCL - Initialize protocol state                   |
//+------------------------------------------------------------------+
CNeuronAlphaGenOCL::CNeuronAlphaGenOCL(void)
  {
   m_cService=NULL;
   m_iPhase=AG_PREPARE;
   m_iStatus=AG_WAIT;
   m_iPrefixLength=0;
   m_iEpisodeStart=0;
   m_iTokenCount=0;
   m_iTradeCount=0;
   m_iTrainingKind=0;
   m_iPass=0;
   m_iExample=0;
   m_iTrials=0;
   m_iBudget=0;
   m_iBatchSize=0;
   m_iBatchGenerated=0;
   m_iBatchProcessed=0;
   m_iPendingTrade=-1;
   m_uBoundary=0;
   m_uObserved=0;
   m_uActionBoundary=0;
   m_uEvent=0;
   m_uDecision=0;
   m_uSeed=0;
   m_uTokenUpdates=0;
   m_uTradeUpdates=0;
   m_iFinanceRandom=1;
   m_bConfigured=false;
   m_bFrame=false;
   m_bGradientReady=false;
   m_bRejected=false;
   m_bWarmBudget=false;
   m_bSearchBudgetPending=false;
   m_bPublicationPermit=true;
   m_bPoolWarmup=false;
   m_bWorkReady=false;
   m_bCorrupted=false;
   m_bNextReady=false;
  }
//+------------------------------------------------------------------+
//| Init - Initialize the composite neuron                           |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Init(uint outputs,uint index,COpenCLMy*context,uint count,ENUM_OPTIMIZATION method,uint batch)
  {
   if(count!=AG_ACTIONS||outputs!=0)
      ReturnFalse;

   return(CNeuronBaseOCL::Init(0,index,context,AG_ACTIONS,ADAM,1));
  }
//+------------------------------------------------------------------+
//| Allocate - Allocate a checked device buffer                      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Allocate(CBufferFloat&buffer,const long count)
  {
   if(count<1||count>INT_MAX||!buffer.BufferInit((int)count,0)||!buffer.BufferCreate(OpenCL))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Configure - Configure device state                               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Configure(const float&config[],const ulong seed)
  {
   if(m_bConfigured||ArraySize(config)!=AG_CFG_SIZE)
      ReturnFalse;

   m_uSeed=seed;
   ArrayCopy(m_fConfig,config);

   if(!m_cEvaluator.Configure(OpenCL,config)||!m_cPool.Configure(OpenCL,config)||
      !m_cSearch.Configure(OpenCL,config,(uint)(seed^(seed>>32))+0x9e3779b9)||
      !m_cTrading.Configure(OpenCL,config,(uint)(seed^(seed>>32))+0x85ebca6b))
      ReturnFalse;

   m_iFinanceRandom=(uint)(seed^(seed>>32))^0xc2b2ae35;

   if(m_iFinanceRandom==0)
      m_iFinanceRandom=1;

   if(!Workspace())
      ReturnFalse;

   m_bConfigured=true;

   return(true);
  }
//+------------------------------------------------------------------+
//| Workspace - Allocate rollout and replay buffers                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Workspace(void)
  {
   int max_tokens=(int)m_fConfig[AG_CFG_MAX_TOKENS];
   int token_capacity=(int)m_fConfig[AG_CFG_TOKEN_ROLLOUT]+max_tokens;
   int trade_capacity=(int)m_fConfig[AG_CFG_TRADE_ROLLOUT];
   int context=(int)m_fConfig[AG_CFG_CONTEXT];
   int hidden=(int)m_fConfig[AG_CFG_TRADE_HIDDEN];

   if(token_capacity<=max_tokens||trade_capacity<1||context<1||hidden<1||
      !Allocate(m_cAccount,AG_ACCOUNT)||!Allocate(m_cWorkMetadata,(long)context*8)||
      !Allocate(m_cTradeContextMetadata,(long)trade_capacity*context*8)||
      !Allocate(m_cTradeIdentifiers,(long)trade_capacity*8)||!Allocate(m_cWorkAnchor,hidden*2)||
      !Allocate(m_cStageAnchor,hidden*2)||!Allocate(m_cEvicted,1)||!Allocate(m_cPrefix,max_tokens+1)||
      !Allocate(m_cFormula,max_tokens)||!Allocate(m_cSearchMarket,(long)m_fConfig[AG_CFG_W]*5+AG_ACCOUNT)||
      !Allocate(m_cTokenPrefixes,(long)token_capacity*(max_tokens+1))||
      !Allocate(m_cTokenMasks,(long)token_capacity*AG_VOCABULARY)||!Allocate(m_cTokenSamples,token_capacity)||
      !Allocate(m_cTokenMetadata,(long)token_capacity*7)||
      !Allocate(m_cTokenGAE,(long)token_capacity*AG_GAE_FIELDS)||
      !Allocate(m_cTradeContexts,(long)trade_capacity*context)||
      !Allocate(m_cTradeAnchors,(long)trade_capacity*hidden*2)||
      !Allocate(m_cTradeAccounts,(long)trade_capacity*AG_ACCOUNT)||
      !Allocate(m_cTradeSamples,(long)trade_capacity*AG_ACTIONS)||
      !Allocate(m_cTradeActions,(long)trade_capacity*AG_ACTIONS)||
      !Allocate(m_cTradeMetadata,(long)trade_capacity*7)||
      !Allocate(m_cTradeGAE,(long)trade_capacity*AG_GAE_FIELDS)||!Allocate(m_cReplayPrefix,max_tokens+1)||
      !Allocate(m_cReplayMask,AG_VOCABULARY)||!Allocate(m_cReplayContext,context)||
      !Allocate(m_cReplayAnchor,hidden*2)||!Allocate(m_cReplayAccount,AG_ACCOUNT)||
      ArrayResize(m_iTokenLengths,token_capacity)!=token_capacity||
      ArrayResize(m_uTradeDecisions,trade_capacity)!=trade_capacity||
      ArrayResize(m_fTradeRewards,trade_capacity)!=trade_capacity)
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Slice - Copy a device buffer slice                               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Slice(CBufferFloat*source,const int source_offset,CBufferFloat*destination,
                               const int destination_offset,const int count)
  {
   if(!source||!destination||count<1||source_offset<0||destination_offset<0||
      count>source.Total()-source_offset||count>destination.Total()-destination_offset)
      ReturnFalse;

   setBuffer(def_k_AGCopySlice,def_ag_AGCopySlice_source,source.GetIndex())
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_source_offset,source_offset)
   setBuffer(def_k_AGCopySlice,def_ag_AGCopySlice_destination,destination.GetIndex())
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_destination_offset,destination_offset)
   setArgument(def_k_AGCopySlice,def_ag_AGCopySlice_count,count)
   uint offset[]={0};
   uint global[]={(uint)count};

   kernelExecute(def_k_AGCopySlice,offset,global)

   return(true);
  }
//+------------------------------------------------------------------+
//| Scalar - Write a host control scalar                             |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Scalar(CBufferFloat*buffer,const int index,const float value)
  {
   if(!buffer||index<0||index>=buffer.Total())
      ReturnFalse;

   ulong byte_offset=(ulong)index*sizeof(float);

   if(byte_offset>UINT_MAX)
      ReturnFalse;

   float values[]={value};

   if(!OpenCL.BufferWrite(buffer.GetIndex(),values,(uint)byte_offset,0,1))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Identifier - Assemble exact-width identifiers                    |
//+------------------------------------------------------------------+
ulong CNeuronAlphaGenOCL::Identifier(CBufferFloat*buffer,const int offset)const
  {
   ulong result=0;

   for(int i=0;i<4;i++)
      result|=((ulong)(uint)buffer.At(offset+i))<<(16*i);

   return(result);
  }
//+------------------------------------------------------------------+
//| PoolControl - Read pool operation status                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::PoolControl(void)
  {
   if(!m_cPool.Control().BufferRead())
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Metadata - Store timestamp and active pool version               |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Metadata(const bool fresh)
  {
   int window=(int)m_fConfig[AG_CFG_W];
   int count=(int)m_fConfig[AG_CFG_CONTEXT];
   int version_offset=10+((int)m_fConfig[AG_CFG_POOL]+1)*(13+(int)m_fConfig[AG_CFG_MAX_TOKENS])+5;

   if(m_cService.Total()<AG_CTX_SIZE+window*5||count>window)
      ReturnFalse;

   setBuffer(def_k_AGContextMetadata,def_ag_AGContextMetadata_service,m_cService.GetIndex())
   setArgument(def_k_AGContextMetadata,def_ag_AGContextMetadata_window,window)
   setBuffer(def_k_AGContextMetadata,def_ag_AGContextMetadata_banks,m_cPool.Banks().GetIndex())
   setArgument(def_k_AGContextMetadata,def_ag_AGContextMetadata_version_offset,version_offset)
   setBuffer(def_k_AGContextMetadata,def_ag_AGContextMetadata_metadata,m_cWorkMetadata.GetIndex())
   setArgument(def_k_AGContextMetadata,def_ag_AGContextMetadata_count,count)
   setArgument(def_k_AGContextMetadata,def_ag_AGContextMetadata_fresh,(int)fresh)
   uint offset[]={0};
   uint global[]={1};

   kernelExecute(def_k_AGContextMetadata,offset,global)

   return(true);
  }
//+------------------------------------------------------------------+
//| ZeroOutput - Clear service-phase outputs                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::ZeroOutput(void)
  {
   for(int i=0;i<AG_ACTIONS;i++)
      if(!Output.Update(i,0))
         ReturnFalse;

   return(Output.BufferWrite());
  }
//+------------------------------------------------------------------+
//| feedForward - Dispatch the current protocol phase                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::feedForward(CNeuronBaseOCL*source,CBufferFloat*context)
  {
   if(context)
      m_cService=context;

   return(feedForward(source));
  }
//+------------------------------------------------------------------+
//| feedForward - Dispatch the current protocol phase                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::feedForward(CNeuronBaseOCL*source)
  {
   if(!m_bConfigured||!source||!m_cService||m_cService.Total()<AG_CTX_SIZE||
      source.Neurons()!=(uint)(5*m_fConfig[AG_CFG_W]+AG_ACCOUNT)||m_bFrame||m_bCorrupted)
      ReturnFalse;

//--- the caller writes this host-origin context before forwarding it.
//--- at() uses its existing CPU view; GPU kernels only read the service data.
   m_iPhase=(int)m_cService.At(AG_CTX_PHASE);
   int   flags=(int)m_cService.At(AG_CTX_FLAGS);
   ulong boundary=Identifier(m_cService,AG_CTX_BOUNDARY);

   m_iStatus=AG_OK;

   if(m_iPhase==AG_BACKWARD)
      return(Replay());

   if(!ZeroOutput())
      ReturnFalse;

   if(m_iPhase==AG_FINALIZED)
     {
      m_iStatus=AG_FINALIZED;

      return(true);
     }

   if(m_iPhase==AG_FLUSH||m_iPhase==AG_EXIT_PENDING)
      return(Flush());

   if(m_iPhase==AG_OBSERVE)
      return(Observe(source,flags,boundary));

   if(m_iPhase==AG_PREPARE)
     {
      if(!bTrain)
        {
         m_iStatus=AG_WAIT;

         return(true);
        }

      if(m_bPoolWarmup&&(flags&AG_FLAG_WARMUP)!=0)
         return(PoolWarmup(source,flags));

      return(SearchStep(source,boundary));
     }

   if(m_iPhase==AG_ACT||m_iPhase==AG_INFER)
      return(Act(source,boundary));

   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Observe - Advance the recurrent observation state                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Observe(CNeuronBaseOCL*source,const int flags,const ulong boundary)
  {
   m_bNextReady=false;

   if(!Slice(source.getOutput(),(int)m_fConfig[AG_CFG_W]*5,GetPointer(m_cAccount),0,AG_ACCOUNT))
      ReturnFalse;

   if(!PoolControl())
      ReturnFalse;

   if(m_cPool.Control().At(AG_POOL_ACTIVE_READY)!=1)
     {
      m_iStatus=AG_WAIT;
      m_bWorkReady=false;

      return(true);
     }

   if((flags&AG_FLAG_WARMUP)!=0)
     {
      if((flags&AG_FLAG_RESET_WORK)!=0)
        {
         if(!m_cPool.ResetContext()||!m_cWorkAnchor.BufferInit(m_cWorkAnchor.Total(),0)||
            !m_cWorkAnchor.BufferWrite())
            ReturnFalse;
        }

      if(!m_cPool.Warmup(source.getOutput(),(flags&AG_FLAG_RESET_WORK)!=0))
         ReturnFalse;
     }
   else
      if(boundary!=m_uObserved)
        {
         if(m_bWorkReady)
           {
            if(!Slice(m_cPool.Context(),0,GetPointer(m_cEvicted),0,1)||
               !m_cTrading.Encode(GetPointer(m_cEvicted),1,GetPointer(m_cWorkAnchor))||!m_cTrading.CommitAnchor()||
               !Slice(m_cTrading.Anchor(),0,GetPointer(m_cWorkAnchor),0,m_cWorkAnchor.Total()))
               ReturnFalse;
           }

         if(!m_cPool.Observe(source.getOutput()))
            ReturnFalse;
        }

   if(!PoolControl())
      ReturnFalse;

   if(m_cPool.Control().At(AG_POOL_SIGNAL_READY)!=1)
     {
      m_bWorkReady=false;
      m_iStatus=m_cPool.Control().At(AG_POOL_STATUS)==AG_WAIT?AG_WAIT:AG_INVALID;

      return(true);
     }

   if(!Metadata((flags&AG_FLAG_WARMUP)!=0))
      ReturnFalse;

   m_bWorkReady=true;
   m_uObserved=boundary;

   if(!m_cTrading.Encode(m_cPool.Context(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cWorkAnchor)))
      ReturnFalse;

//--- a next-state value is needed only for an actual unresolved action.
   if(m_iPendingTrade>=0&&(flags&AG_FLAG_VALUE)!=0)
     {
      if(!m_cTrading.Heads(GetPointer(m_cAccount),1))
         ReturnFalse;

      m_bNextReady=true;
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| ProbeSearchQueue - Check search queue completion                 |
//+------------------------------------------------------------------+
void CNeuronAlphaGenOCL::ProbeSearchQueue(const string stage,const float marker)
  {
#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES
   ResetLastError();
   bool queued=Scalar(GetPointer(m_cBatchWorkspace),0,marker)&&
               Scalar(GetPointer(m_cBatchSummary),0,-1.0f)&&
               Slice(GetPointer(m_cBatchWorkspace),0,GetPointer(m_cBatchSummary),0,1);
   int  enqueue_error=GetLastError();
   bool complete=queued&&OpenCL.WaitAlphaGenKernel(def_k_AGCopySlice);

   ResetLastError();
   float result[1]={-1.0f};
   bool  read=queued&&OpenCL.BufferRead(m_cBatchSummary.GetIndex(),result,0,0,1);
   int   read_error=GetLastError();
   int   status=queued?(int)OpenCL.ExecutionStatus(def_k_AGCopySlice):-1;

   PrintFormat("AlphaGen OpenCL probe %s: queued=%d enqueue_error=%d complete=%d status=%d read=%d read_error=%d marker=%.0f expected=%.0f",
               stage,(int)queued,enqueue_error,(int)complete,status,(int)read,read_error,result[0],marker);
#endif
  }
//+------------------------------------------------------------------+
//| EnableParallelSearch - Configure parallel candidate search       |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::EnableParallelSearch(void)
  {
   const int batch=8;
   const int maximum=(int)m_fConfig[AG_CFG_MAX_TOKENS];

   if(!m_bConfigured||
      !Allocate(m_cBatchCandidates,(long)batch*(1+maximum*(4+AG_VOCABULARY)))||
      !Allocate(m_cBatchWorkspace,(long)batch*m_cSearch.BatchWorkStride())||
      !Allocate(m_cBatchSummary,4+AG_POOL_CONTROL_SIZE+(long)batch*maximum))
      ReturnFalse;

   ProbeSearchQueue("setup",13579.0f);
   m_iBatchSize=batch;

   return(true);
  }
//+------------------------------------------------------------------+
//| SearchBatch - Generate and commit a candidate batch              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::SearchBatch(CNeuronBaseOCL*source)
  {
   int count=MathMin(m_iBatchSize,m_iBudget-m_iTrials);
   int first_row=m_iTokenCount;

   if(m_iTrials==0&&first_row==0&&m_uTokenUpdates==0)
      ProbeSearchQueue("before generation",24680.0f);
   ulong started=GetMicrosecondCount();

   if(!m_cSearch.GenerateBatch(count,GetPointer(m_cBatchCandidates),GetPointer(m_cBatchWorkspace)))
     {
      m_iStatus=AG_TECHNICAL;
      ReturnFalse;
     }

   if(m_iTrials==0&&first_row==0&&m_uTokenUpdates==0)
     {
      ResetLastError();
      bool read=m_cBatchCandidates.BufferRead();
      int  read_error=GetLastError();

      ResetLastError();
      int execution=(int)OpenCL.ExecutionStatus(def_k_AGSearchGenerate);
      int execution_error=GetLastError();
      int stride=1+(int)m_fConfig[AG_CFG_MAX_TOKENS]*(4+AG_VOCABULARY);

      string lengths="";
      bool   valid=read;

      for(int i=0;i<count&&read;i++)
        {
         float length=m_cBatchCandidates.At(i*stride);

         lengths+=StringFormat(" %.9g",length);
         valid=valid&&length>=2&&length<=(int)m_fConfig[AG_CFG_MAX_TOKENS];
        }
      PrintFormat("AlphaGen first GPU generation: us=%I64u candidates=%d read=%d read_error=%d execution=%d execution_error=%d lengths:%s",
                  GetMicrosecondCount()-started,count,(int)read,read_error,execution,execution_error,lengths);

      if(!valid)
        {
         m_iStatus=AG_TECHNICAL;
         ReturnFalse;
        }
     }

   if(!m_cPool.CommitSearchBatch(source.getOutput(),GetPointer(m_cBatchCandidates),count,first_row,
                                 GetPointer(m_cTokenPrefixes),GetPointer(m_cTokenMasks),
                                 GetPointer(m_cTokenSamples),GetPointer(m_cTokenMetadata),
                                 GetPointer(m_cTokenGAE),GetPointer(m_cBatchSummary)))
     {
      m_iStatus=AG_TECHNICAL;
      ReturnFalse;
     }
   int processed=(int)m_cBatchSummary.At(1);
   int rows=(int)m_cBatchSummary.At(2);
   int maximum=(int)m_fConfig[AG_CFG_MAX_TOKENS];

   if(processed<1||processed>count||rows<first_row||
      rows>ArraySize(m_iTokenLengths)||rows-first_row>processed*maximum)
     {
      m_iStatus=AG_TECHNICAL;
      PrintFormat("AlphaGen invalid batch summary: requested=%d processed=%.9g first_row=%d rows=%.9g capacity=%d max_tokens=%d target=%d status=%.9g",
                  count,m_cBatchSummary.At(1),first_row,m_cBatchSummary.At(2),ArraySize(m_iTokenLengths),
                  maximum,(int)m_fConfig[AG_CFG_TOKEN_ROLLOUT],m_cBatchSummary.At(0));
      ReturnFalse;
     }

   for(int i=first_row;i<rows;i++)
     {
      int length=(int)m_cBatchSummary.At(4+AG_POOL_CONTROL_SIZE+i-first_row);

      if(length<1||length>maximum)
         ReturnFalse;

      m_iTokenLengths[i]=length;
     }
   int rejected=(int)m_cBatchSummary.At(3);

   if(rejected>0)
      PrintFormat("AlphaGen excluded %d technical pool candidates in a search batch",rejected);
   m_iBatchGenerated=count;
   m_iBatchProcessed=processed;
   m_iTrials+=processed;
   m_iTokenCount=rows;

   if(m_iTokenCount>=(int)m_fConfig[AG_CFG_TOKEN_ROLLOUT]&&!BeginTraining(1))
      ReturnFalse;

   m_iStatus=AG_WORKING;

   return(true);
  }
//+------------------------------------------------------------------+
//| SearchStep - Advance causal token search                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::SearchStep(CNeuronBaseOCL*source,const ulong boundary)
  {
   if(m_iPendingTrade>=0)
     {
      m_iStatus=AG_WAIT;

      return(true);
     }

   if(HasTraining())
     {
      m_iStatus=AG_WORKING;

      return(true);
     }

   if(m_bPoolWarmup)
     {
      m_iStatus=AG_PREPARING_POOL;

      return(true);
     }

//--- a timer may finish fewer than one episode before the market advances.
//--- keep the issued budget until its completion/publication check has run.
   if(!m_bSearchBudgetPending&&boundary!=m_uBoundary&&m_iPrefixLength==0)
     {
      if(!PoolControl())
         ReturnFalse;

      m_uBoundary=boundary;
      m_iTrials=0;
      m_iBatchGenerated=0;
      m_iBatchProcessed=0;
      bool active=m_cPool.Control().At(AG_POOL_ACTIVE_READY)==1;

      m_iBudget=(int)m_fConfig[!active&&!m_bWarmBudget?AG_CFG_WARMUP_TRIALS:AG_CFG_SEARCH_TRIALS];
      m_bWarmBudget=true;
      m_bSearchBudgetPending=true;
     }

   if(m_iTrials>=m_iBudget)
     {
      if(!PoolControl())
         ReturnFalse;

      if(m_bPublicationPermit&&PreparedCount()>0)
        {
         if(!m_cPool.Prepare(source.getOutput()))
            ReturnFalse;

         if(!PoolControl())
            ReturnFalse;

         if(m_cPool.Control().At(AG_POOL_STATUS)==AG_OK)
           {
            m_bPoolWarmup=true;
            m_bSearchBudgetPending=false;
            m_uBoundary=boundary;
            m_iStatus=AG_PREPARING_POOL;

            return(true);
           }
        }
      m_bSearchBudgetPending=false;
      m_uBoundary=boundary;
      m_iStatus=AG_OK;

      return(true);
     }

   if(m_iBatchSize>0)
      return(SearchBatch(source));

   if(m_iPrefixLength==0)
     {
      m_iEpisodeStart=m_iTokenCount;

      if(!Scalar(GetPointer(m_cPrefix),0,AG_BOS)||
         !Slice(source.getOutput(),0,GetPointer(m_cSearchMarket),0,m_cSearchMarket.Total()))
         ReturnFalse;

#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES
      if(!OpenCL.WaitAlphaGenKernel(def_k_AGCopySlice))
        {
         PrintFormat("AlphaGen search market copy failed before mask probe: count=%d source=%d destination=%d",
                     m_cSearchMarket.Total(),source.getOutputIndex(),m_cSearchMarket.GetIndex());
         ReturnFalse;
        }
#endif
     }

   if(!m_cEvaluator.Mask(GetPointer(m_cFormula),m_iPrefixLength)||
      !m_cSearch.ForwardPrefix(GetPointer(m_cPrefix),m_iPrefixLength+1,m_cEvaluator.MaskBuffer(),0))
      ReturnFalse;

#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES
   if(!OpenCL.WaitAlphaGenKernel(def_k_AGDistribution))
      ReturnFalse;

#endif

//--- the terminal read fences mask, policy and the complete saved token record.
   CBufferFloat *result=m_cEvaluator.NumericResult();
   int          row=m_iTokenCount;
   int          maximum=(int)m_fConfig[AG_CFG_MAX_TOKENS];
   int          capacity=m_cTokenSamples.Total();

   if(row<0||row>=capacity||row>=ArraySize(m_iTokenLengths)||
      m_iEpisodeStart<0||m_iEpisodeStart>row||m_iPrefixLength<0||m_iPrefixLength>=maximum||
      m_cPrefix.Total()<maximum+1||m_cFormula.Total()<maximum||
      m_cSearch.Sample().Total()<1||m_cSearch.Metadata().Total()<7||
      m_cEvaluator.MaskBuffer().Total()<AG_VOCABULARY||result.Total()<2||
      (long)m_cTokenPrefixes.Total()<(long)capacity*(maximum+1)||
      (long)m_cTokenMasks.Total()<(long)capacity*AG_VOCABULARY||
      (long)m_cTokenMetadata.Total()<(long)capacity*7||
      (long)m_cTokenGAE.Total()<(long)capacity*AG_GAE_FIELDS)
      ReturnFalse;

   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_sample,m_cSearch.Sample().GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_metadata,m_cSearch.Metadata().GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_mask,m_cEvaluator.MaskBuffer().GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_prefix,m_cPrefix.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_formula,m_cFormula.GetIndex())
   setArgument(def_k_AGSearchRecord,def_ag_AGSearchRecord_prefix_length,m_iPrefixLength)
   setArgument(def_k_AGSearchRecord,def_ag_AGSearchRecord_maximum,maximum)
   setArgument(def_k_AGSearchRecord,def_ag_AGSearchRecord_row,row)
   setArgument(def_k_AGSearchRecord,def_ag_AGSearchRecord_episode_start,m_iEpisodeStart)
   setArgument(def_k_AGSearchRecord,def_ag_AGSearchRecord_capacity,capacity)
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_prefixes,m_cTokenPrefixes.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_masks,m_cTokenMasks.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_samples,m_cTokenSamples.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_rollout_metadata,m_cTokenMetadata.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_gae,m_cTokenGAE.GetIndex())
   setBuffer(def_k_AGSearchRecord,def_ag_AGSearchRecord_result,result.GetIndex())
   uint record_offset[]={0};
   uint record_global[]={1};

   kernelExecute(def_k_AGSearchRecord,record_offset,record_global)
#ifdef ALPHAGEN_OPENCL_QUEUE_PROBES
   if(!OpenCL.WaitAlphaGenKernel(def_k_AGSearchRecord))
      ReturnFalse;

#endif
   if(!result.BufferRead()||result.At(0)!=AG_OK)
     {
      PrintFormat("AlphaGen token record rejected: prefix=%d trial=%d row=%d",m_iPrefixLength,m_iTrials,row);
      m_iStatus=AG_TECHNICAL;
      ReturnFalse;
     }
   int token=(int)result.At(1);

   m_iTokenLengths[row]=m_iPrefixLength+1;
   m_iTokenCount++;
   m_iPrefixLength++;

   if(token!=AG_STOP)
     {
      m_iStatus=AG_WORKING;

      return(true);
     }

   if(!m_cPool.Candidate(GetPointer(m_cSearchMarket),GetPointer(m_cFormula),m_iPrefixLength)||!PoolControl())
      ReturnFalse;

   if(m_cPool.Control().At(AG_POOL_STATUS)==AG_TECHNICAL)
      Print("AlphaGen pool operation failed technically; candidate excluded without reward");

   if(m_cPool.Control().At(AG_POOL_REWARD_READY)==1)
     {
      if(!Scalar(GetPointer(m_cTokenGAE),row*AG_GAE_FIELDS+AG_GAE_REWARD,m_cPool.Control().At(AG_POOL_REWARD))||
         !Scalar(GetPointer(m_cTokenGAE),row*AG_GAE_FIELDS+AG_GAE_BOOTSTRAP,0)||
         !Scalar(GetPointer(m_cTokenGAE),row*AG_GAE_FIELDS+AG_GAE_CONTINUE,0)||
         !Scalar(GetPointer(m_cTokenGAE),row*AG_GAE_FIELDS+AG_GAE_NEXT,0))
         ReturnFalse;
     }
   else
      m_iTokenCount=m_iEpisodeStart;
   m_iPrefixLength=0;
   m_iTrials++;

   if(m_iTokenCount>=(int)m_fConfig[AG_CFG_TOKEN_ROLLOUT]&&!BeginTraining(1))
      ReturnFalse;

   m_iStatus=AG_WORKING;

   return(true);
  }
//+------------------------------------------------------------------+
//| ResetWorking - Reset the working recurrent stream                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::ResetWorking(void)
  {
   if(m_bFrame||m_bCorrupted)
      ReturnFalse;

   if(!m_cPool.ResetContext()||!m_cWorkAnchor.BufferInit(m_cWorkAnchor.Total(),0)||
      !m_cWorkAnchor.BufferWrite())
      ReturnFalse;

   m_bWorkReady=false;
   m_bNextReady=false;
   m_uObserved=0;
   m_uActionBoundary=0;
   m_uBoundary=0;

   return(true);
  }
//+------------------------------------------------------------------+
//| CancelTraining - Discard retained training examples              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::CancelTraining(void)
  {
   if(m_bFrame||m_bCorrupted||m_iPendingTrade>=0)
      ReturnFalse;

   PrintFormat("AlphaGen Stop discards retained token rows=%d trade rows=%d kind=%d pass=%d example=%d",
               m_iTokenCount,m_iTradeCount,m_iTrainingKind,m_iPass,m_iExample);
   bool changed_trade=m_iTrainingKind==2;

   m_iTrainingKind=0;
   m_iTokenCount=0;
   m_iTradeCount=0;
   m_iPrefixLength=0;
   m_iExample=0;
   m_iPass=0;

   if(changed_trade&&!Rewarm())
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| RestartPoolWarmup - Restart causal pool preparation              |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::RestartPoolWarmup(void)
  {
   if(!m_bPoolWarmup||!m_bPublicationPermit||m_bFrame)
      ReturnFalse;

   return(m_cPool.RestartPrepare());
  }
//+------------------------------------------------------------------+
//| AbortPoolWarmup - Cancel pending pool preparation                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::AbortPoolWarmup(void)
  {
   if(!m_bPoolWarmup||m_bFrame||!m_cPool.AbortPrepare()||!PoolControl())
      ReturnFalse;

   m_bPoolWarmup=false;
   m_iStatus=AG_OK;

   return(true);
  }
//+------------------------------------------------------------------+
//| AdjustReward - Correct an untrained sealed reward                |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::AdjustReward(const ulong decision,const float adjustment)
  {
   if(!MathIsValidNumber(adjustment)||m_bFrame||m_bCorrupted||
      (m_iTrainingKind==2&&(m_iPass>0||m_iExample>0)))
      ReturnFalse;

   for(int row=0;row<m_iTradeCount;row++)
     {
      if(m_uTradeDecisions[row]!=decision)
         continue;
      float corrected=m_fTradeRewards[row]+adjustment;

      if(!MathIsValidNumber(corrected)||
         !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_REWARD,corrected))
         ReturnFalse;

      m_fTradeRewards[row]=corrected;

      if(m_iTrainingKind==2&&!m_cEvaluator.GAE(GetPointer(m_cTradeGAE),m_iTradeCount,false))
         ReturnFalse;

      return(true);
     }
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| PoolWarmup - Prepare the next pool publication                   |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::PoolWarmup(CNeuronBaseOCL*source,const int flags)
  {
   if(!m_bPublicationPermit||m_iPendingTrade>=0||HasTraining())
      ReturnFalse;

   if(!m_cPool.PrepareAppend(source.getOutput(),(flags&AG_FLAG_RESET_WORK)!=0)||!PoolControl())
      ReturnFalse;

   if(m_cPool.Control().At(AG_POOL_STATUS)==AG_TECHNICAL)
     {
      m_cPool.AbortPrepare();
      m_bPoolWarmup=false;
      m_iStatus=AG_WAIT;

      return(true);
     }

   if((flags&AG_FLAG_END)==0)
     {
      m_iStatus=AG_PREPARING_POOL;

      return(true);
     }

   if(m_cPool.Control().At(AG_POOL_SIGNAL_READY)!=1)
     {
      m_cPool.AbortPrepare();
      m_bPoolWarmup=false;
      m_iStatus=AG_WAIT;

      return(true);
     }

//--- preparation uses a separate ring. No trade is permitted at publication.
   if(!m_cStageAnchor.BufferInit(m_cStageAnchor.Total(),0)||!m_cStageAnchor.BufferWrite()||
      !m_cTrading.Encode(m_cPool.StagedContext(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cStageAnchor)))
      ReturnFalse;

   if(!m_cPool.Commit()||!PoolControl())
      ReturnFalse;

   if(m_cPool.Control().At(AG_POOL_PUBLISHED)==1)
     {
      if(!Slice(GetPointer(m_cStageAnchor),0,GetPointer(m_cWorkAnchor),0,m_cWorkAnchor.Total()))
         ReturnFalse;

      if(!Metadata(true))
         ReturnFalse;

      m_bPublicationPermit=false;
      m_bPoolWarmup=false;
      m_bWorkReady=true;
      m_uObserved=Identifier(m_cService,AG_CTX_BOUNDARY);
      m_uActionBoundary=m_uObserved;
      m_iStatus=AG_PUBLISHED;
     }
   else
     {
      m_bPoolWarmup=false;
      m_iStatus=AG_WAIT;
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| Flush - Process a financial transition                           |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Flush(void)
  {
   if(!bTrain)
      return(true);

   if(((int)m_cService.At(AG_CTX_FLAGS)&AG_FLAG_END)!=0&&m_iPrefixLength>0)
     {
      m_iTokenCount=m_iEpisodeStart;
      m_iPrefixLength=0;
     }

   if(((int)m_cService.At(AG_CTX_FLAGS)&AG_FLAG_DISCARD)!=0)
     {
      m_iPendingTrade=-1;

      if(!HasTraining())
        {
         if(m_iTradeCount>0)
            return(BeginTraining(2));

         if(m_iTokenCount>0&&m_iPrefixLength==0)
            return(BeginTraining(1));
        }

      return(true);
     }

   if(m_iPendingTrade<0)
     {
      if(m_cService.At(AG_CTX_TERMINAL)==1&&m_cService.At(AG_CTX_MONEY_CONFIRMED)==1)
         m_bPublicationPermit=true;

      if(m_cService.At(AG_CTX_TERMINAL)==1&&!HasTraining())
        {
         if(m_iTradeCount>0)
            return(BeginTraining(2));

         if(m_iTokenCount>0&&m_iPrefixLength==0)
            return(BeginTraining(1));
        }

      return(true);
     }
   ulong event=Identifier(m_cService,AG_CTX_EVENT);
   ulong decision=Identifier(m_cService,AG_CTX_DECISION);

   if(event<=m_uEvent)
      return(true);

   if(decision!=m_uDecision)
     {
      m_iStatus=AG_WAIT;

      return(true);
     }
   bool terminal=m_cService.At(AG_CTX_TERMINAL)==1;
   bool confirmed=m_cService.At(AG_CTX_MONEY_CONFIRMED)==1;

   bool next=
      m_cService.At(AG_CTX_NEXT_READY)==1&&m_bWorkReady&&m_bNextReady&&Identifier(m_cService,AG_CTX_BOUNDARY)==m_uObserved;

   if(!confirmed||(!terminal&&!next))
     {
      m_iStatus=AG_WAIT;

      return(true);
     }
   int   row=m_iPendingTrade;
   float reward=m_cService.At(AG_CTX_REWARD);
   float delta=m_cService.At(AG_CTX_DELTA);

   if(!MathIsValidNumber(reward)||!MathIsValidNumber(delta)||delta<1)
      ReturnFalse;

   if(!Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_REWARD,reward)||
      !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_DELTA,delta)||
      !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_BOOTSTRAP,terminal?0:1)||
      !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_CONTINUE,terminal?0:1)||
      !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_READY,1))
      ReturnFalse;

   if(terminal)
     {
      if(!Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_NEXT,0))
         ReturnFalse;
     }
   else
      if(!Slice(m_cTrading.Metadata(),2,GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_NEXT,1))
         ReturnFalse;

   m_fTradeRewards[row]=reward;
   m_uEvent=event;
   m_iPendingTrade=-1;
   m_iTradeCount++;

   if(terminal||m_iTradeCount>=(int)m_fConfig[AG_CFG_TRADE_ROLLOUT])
     {
      if(!BeginTraining(2))
         ReturnFalse;
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| Act - Sample and retain a trading action                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Act(CNeuronBaseOCL*source,const ulong boundary)
  {
   if(!m_bWorkReady||boundary!=m_uObserved||m_bSearchBudgetPending||m_bPoolWarmup||
      HasTraining()||m_iPendingTrade>=0||boundary==m_uActionBoundary)
     {
      m_iStatus=AG_WAIT;

      return(true);
     }

   if(!Slice(source.getOutput(),(int)m_fConfig[AG_CFG_W]*5,GetPointer(m_cAccount),0,AG_ACCOUNT))
      ReturnFalse;

   if(!m_cTrading.Forward(m_cPool.Context(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cWorkAnchor),
                          GetPointer(m_cAccount),bTrain&&m_iPhase==AG_ACT?0:1))
      ReturnFalse;

   if(!m_cEvaluator.Numeric(m_cTrading.Action(),true)||!m_cEvaluator.Numeric(m_cTrading.Metadata(),false)||
      !m_cEvaluator.NumericOK())
     {
      m_iStatus=AG_INVALID;
      m_bWorkReady=false;

      return(true);
     }

   if(!Slice(m_cTrading.Action(),0,Output,0,AG_ACTIONS))
      ReturnFalse;

   m_uActionBoundary=boundary;

   if(bTrain&&m_iPhase==AG_ACT)
     {
      int row=m_iTradeCount;
      int context=(int)m_fConfig[AG_CFG_CONTEXT];
      int hidden=(int)m_fConfig[AG_CFG_TRADE_HIDDEN];

      if(row>=(int)m_fConfig[AG_CFG_TRADE_ROLLOUT])
         ReturnFalse;

      int version_offset=10+((int)m_fConfig[AG_CFG_POOL]+1)*(13+(int)m_fConfig[AG_CFG_MAX_TOKENS])+5;

      if(!Slice(GetPointer(m_cWorkMetadata),0,GetPointer(m_cTradeContextMetadata),row*context*8,context*8)||
         !Slice(m_cService,AG_CTX_DECISION,GetPointer(m_cTradeIdentifiers),row*8,4)||
         !Slice(m_cPool.Banks(),version_offset,GetPointer(m_cTradeIdentifiers),row*8+4,4)||
         !Slice(m_cPool.Context(),0,GetPointer(m_cTradeContexts),row*context,context)||
         !Slice(GetPointer(m_cWorkAnchor),0,GetPointer(m_cTradeAnchors),row*hidden*2,hidden*2)||
         !Slice(GetPointer(m_cAccount),0,GetPointer(m_cTradeAccounts),row*AG_ACCOUNT,AG_ACCOUNT)||
         !Slice(m_cTrading.Sample(),0,GetPointer(m_cTradeSamples),row*AG_ACTIONS,AG_ACTIONS)||
         !Slice(m_cTrading.Action(),0,GetPointer(m_cTradeActions),row*AG_ACTIONS,AG_ACTIONS)||
         !Slice(m_cTrading.Metadata(),0,GetPointer(m_cTradeMetadata),row*7,7)||
         !Slice(m_cTrading.Metadata(),0,GetPointer(m_cTradeMetadata),row*7+3,1)||
         !Slice(m_cTrading.Metadata(),2,GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_VALUE,1)||
         !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_READY,0))
         ReturnFalse;

      m_bNextReady=false;
      m_iPendingTrade=row;
      m_uDecision=Identifier(m_cService,AG_CTX_DECISION);
      m_uTradeDecisions[row]=m_uDecision;
      m_bPublicationPermit=false;
     }
   m_iStatus=AG_ACTION_READY;

   return(true);
  }
//+------------------------------------------------------------------+
//| BeginTraining - Freeze advantages for PPO passes                 |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::BeginTraining(const int kind)
  {
   int count=kind==1?m_iTokenCount:m_iTradeCount;

   if(count<1)
      return(true);

   if(m_iTrainingKind!=0||m_iPendingTrade>=0)
      ReturnFalse;

   CBufferFloat *gae=kind==1?GetPointer(m_cTokenGAE):GetPointer(m_cTradeGAE);

   if(!Scalar(gae,(count-1)*AG_GAE_FIELDS+AG_GAE_CONTINUE,0)||!m_cEvaluator.GAE(gae,count,kind==1))
      ReturnFalse;

   m_iTrainingKind=kind;
   m_iExample=0;
   m_iPass=0;

   return(true);
  }
//+------------------------------------------------------------------+
//| PrepareUpdate - Select the next training example                 |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::PrepareUpdate(void)
  {
   if(!HasTraining()||m_bFrame||m_bCorrupted)
      ReturnFalse;

   m_iPhase=AG_BACKWARD;

   return(true);
  }
//+------------------------------------------------------------------+
//| Replay - Restore and replay a saved example                      |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Replay(void)
  {
   if(!HasTraining())
      ReturnFalse;

   int          row=m_iExample;
   int          max_tokens=(int)m_fConfig[AG_CFG_MAX_TOKENS];
   CBufferFloat *meta=m_iTrainingKind==1?m_cSearch.Metadata():m_cTrading.Metadata();
   CBufferFloat *gae=m_iTrainingKind==1?GetPointer(m_cTokenGAE):GetPointer(m_cTradeGAE);
   CBufferFloat *stored_meta=m_iTrainingKind==1?GetPointer(m_cTokenMetadata):GetPointer(m_cTradeMetadata);
   bool         search=m_iTrainingKind==1;
   int          length=search?m_iTokenLengths[row]:(int)m_fConfig[AG_CFG_CONTEXT];
   int          counts[4];

   counts[0]=length;
   counts[1]=search?AG_VOCABULARY:(int)m_fConfig[AG_CFG_TRADE_HIDDEN]*2;
   counts[2]=search?1:AG_ACCOUNT;
   counts[3]=search?0:AG_ACTIONS;
   int          stride=search?max_tokens+1:length;
   CBufferFloat *src[4];
   CBufferFloat *dst[4];

   src[0]=search?GetPointer(m_cTokenPrefixes):GetPointer(m_cTradeContexts);
   src[1]=search?GetPointer(m_cTokenMasks):GetPointer(m_cTradeAnchors);
   src[2]=search?GetPointer(m_cTokenSamples):GetPointer(m_cTradeAccounts);
   src[3]=GetPointer(m_cTradeSamples);
   dst[0]=search?GetPointer(m_cReplayPrefix):GetPointer(m_cReplayContext);
   dst[1]=search?GetPointer(m_cReplayMask):GetPointer(m_cReplayAnchor);
   dst[2]=search?m_cSearch.Sample():GetPointer(m_cReplayAccount);
   dst[3]=m_cTrading.Sample();

   if(row<0||length<1||length>stride||meta.Total()<7||
      (long)(row+1)*7>stored_meta.Total()||(long)(row+1)*AG_GAE_FIELDS>gae.Total())
      ReturnFalse;

   if(!OpenCL.Profile().Begin(OpenCL,meta.GetIndex(),m_iTrainingKind,m_iPass,m_iExample))
      ReturnFalse;

   int work=7;

   for(int i=0;i<4;i++)
     {
      int step=i==0?stride:counts[i];

      if(counts[i]<0||(long)row*step+counts[i]>src[i].Total()||counts[i]>dst[i].Total())
         ReturnFalse;

      work=MathMax(work,counts[i]);
     }
   setBuffer(def_k_AGReplayPrepare,0,stored_meta.GetIndex())
   setBuffer(def_k_AGReplayPrepare,1,gae.GetIndex())
   setBuffer(def_k_AGReplayPrepare,2,meta.GetIndex())
   setArgument(def_k_AGReplayPrepare,3,row)
   setBuffer(def_k_AGReplayPrepare,4,src[0].GetIndex())
   setBuffer(def_k_AGReplayPrepare,5,dst[0].GetIndex())
   setArgument(def_k_AGReplayPrepare,6,stride)
   setArgument(def_k_AGReplayPrepare,7,counts[0])

   for(int i=1;i<4;i++)
     {
      int arg=8+(i-1)*3;

      setBuffer(def_k_AGReplayPrepare,arg,src[i].GetIndex())
      setBuffer(def_k_AGReplayPrepare,arg+1,dst[i].GetIndex())
      setArgument(def_k_AGReplayPrepare,arg+2,counts[i])
     }
   uint offset[]={0};
   uint global[]={(uint)work};
   //+------------------------------------------------------------------+
   //| kernelExecute - Implement kernelExecute                          |
   //+------------------------------------------------------------------+
   kernelExecute(def_k_AGReplayPrepare,offset,global)

   if(!OpenCL.Profile().Mark(1))
      ReturnFalse;

   if(search)
     {
      if(!m_cSearch.ForwardPrefix(GetPointer(m_cReplayPrefix),length,GetPointer(m_cReplayMask),2))
         ReturnFalse;
     }
   else
      if(!m_cTrading.Forward(GetPointer(m_cReplayContext),length,GetPointer(m_cReplayAnchor),
                             GetPointer(m_cReplayAccount),2))
         ReturnFalse;

   if(!ZeroOutput()||!OpenCL.Profile().Mark(2))
      ReturnFalse;

   m_bFrame=true;
   m_bGradientReady=false;
   m_bRejected=false;

   return(true);
  }
//+------------------------------------------------------------------+
//| calcOutputGradients - Calculate policy output gradients          |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::calcOutputGradients(CBufferFloat*target,float&error)
  {
   if(!m_bFrame||!HasTraining()||!target||target.Total()!=AG_ACTIONS)
      ReturnFalse;

   if(!m_cEvaluator.Numeric(Output,true))
      ReturnFalse;

   m_bGradientReady=m_iTrainingKind==1?m_cSearch.Backward(GetPointer(m_cEvaluator))
                      :m_cTrading.Backward(GetPointer(m_cEvaluator));

   if(!m_bGradientReady)
     {
      if(m_cEvaluator.NumericOK())
         ReturnFalse;

      m_bRejected=true;
      error=0;
      if(!OpenCL.Profile().Mark(5))
         ReturnFalse;
      PrintFormat("AlphaGen rejected nonfinite PPO example kind=%d pass=%d example=%d",m_iTrainingKind,m_iPass,
                  m_iExample);

      return(true);
     }
   CBufferFloat *meta=m_iTrainingKind==1?m_cSearch.Metadata():m_cTrading.Metadata();

   if(!meta.BufferRead())
      ReturnFalse;

   error=meta.At(6);
   if(!OpenCL.Profile().Mark(5))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| updateInputWeights - Apply the native backward update            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::updateInputWeights(CNeuronBaseOCL*source)
  {
   if(!m_bFrame||(!m_bGradientReady&&!m_bRejected))
      ReturnFalse;

   if(m_bGradientReady)
     {
      bool ok=m_iTrainingKind==1?m_cSearch.Apply():m_cTrading.Apply();

      if(!ok)
        {
         m_bCorrupted=true;
         m_iStatus=AG_CORRUPTED;
         ReturnFalse;
        }

      if(m_iTrainingKind==1)
         m_uTokenUpdates++;
      else
         m_uTradeUpdates++;
     }
   m_bFrame=false;
   m_bGradientReady=false;

   if(!OpenCL.Profile().Mark(6))
      ReturnFalse;

   bool completed=FinishUpdate();
   if(!completed||!OpenCL.Profile().End())
      ReturnFalse;

   if(!HasTraining())
      OpenCL.Profile().Report("cycle");

   return(true);
  }
//+------------------------------------------------------------------+
//| FinishUpdate - Advance PPO pass counters                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::FinishUpdate(void)
  {
   m_iExample++;
   int count=m_iTrainingKind==1?m_iTokenCount:m_iTradeCount;

   if(m_iExample<count)
      return(true);

   m_iExample=0;
   m_iPass++;

   if(m_iPass<(int)m_fConfig[AG_CFG_PASSES])
      return(true);

   int kind=m_iTrainingKind;

   m_iTrainingKind=0;
   m_iPass=0;

   if(kind==1)
      m_iTokenCount=0;
   else
     {
      m_iTradeCount=0;
      m_bPublicationPermit=true;

      if(!Rewarm())
         ReturnFalse;
     }
   m_iStatus=AG_OK;

   return(true);
  }
//+------------------------------------------------------------------+
//| Rewarm - Replay context after weight updates                     |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Rewarm(void)
  {
   if(!m_bWorkReady)
      return(true);

   if(!m_cWorkAnchor.BufferInit(m_cWorkAnchor.Total(),0)||!m_cWorkAnchor.BufferWrite()||
      !m_cTrading.Encode(m_cPool.Context(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cWorkAnchor)))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| PreparedCount - Read the prepared factor count                   |
//+------------------------------------------------------------------+
int CNeuronAlphaGenOCL::PreparedCount(void)
  {
   return((int)m_cPool.Control().At(AG_POOL_PREPARED_COUNT));
  }
//+------------------------------------------------------------------+
//| ActiveCount - Read the active factor count                       |
//+------------------------------------------------------------------+
int CNeuronAlphaGenOCL::ActiveCount(void)
  {
   return((int)m_cPool.Control().At(AG_POOL_ACTIVE_COUNT));
  }
//+------------------------------------------------------------------+
//| FinancialUniform - Advance the financial random stream           |
//+------------------------------------------------------------------+
double CNeuronAlphaGenOCL::FinancialUniform(void)
  {
   m_iFinanceRandom^=m_iFinanceRandom<<13;
   m_iFinanceRandom^=m_iFinanceRandom>>17;
   m_iFinanceRandom^=m_iFinanceRandom<<5;

   return(((double)m_iFinanceRandom+0.5)/4294967296.0);
  }
//+------------------------------------------------------------------+
//| Save - Serialize coherent model state; another run's rollouts... |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Save(const int handle)
  {
   if(!SafeToSave()||!Output.BufferRead()||!PrevOutput.BufferRead()||!Gradient.BufferRead()||
      !CNeuronBaseOCL::Save(handle))
      ReturnFalse;

   dFileWriteUInt(handle,AG_CFG_SIZE)

   for(int i=0;i<AG_CFG_SIZE;i++)
     {
      dFileWriteFloat(handle,m_fConfig[i])
     }
   uint dictionary[]=AG_VOCABULARY_ORDER;

   dFileWriteUInt(handle,AG_VOCABULARY)

   for(int i=0;i<AG_VOCABULARY;i++)
     {
      dFileWriteUInt(handle,dictionary[i])
     }
   dFileWriteUInt(handle,m_iFinanceRandom)

   if(FileWriteLong(handle,(long)m_uSeed)!=sizeof(long))
      ReturnFalse;

   if(FileWriteLong(handle,(long)m_uTokenUpdates)!=sizeof(long)||
      FileWriteLong(handle,(long)m_uTradeUpdates)!=sizeof(long)||!m_cPool.Save(handle)||
      !m_cSearch.Save(handle)||!m_cTrading.Save(handle)||!m_cWorkAnchor.BufferRead()||
      !m_cWorkAnchor.Save(handle)||!m_cWorkMetadata.BufferRead()||!m_cWorkMetadata.Save(handle))
      ReturnFalse;

   return(true);
  }
//+------------------------------------------------------------------+
//| Load - Restore persistent model state                            |
//+------------------------------------------------------------------+
bool CNeuronAlphaGenOCL::Load(const int handle)
  {
   if(!CNeuronBaseOCL::Load(handle)||Neurons()!=AG_ACTIONS)
      ReturnFalse;

   uint size=0;

   dFileReadUInt(handle,size)

   if(size!=AG_CFG_SIZE||ArrayResize(m_fConfig,AG_CFG_SIZE)!=AG_CFG_SIZE)
      ReturnFalse;

   for(int i=0;i<AG_CFG_SIZE;i++)
     {
      dFileReadUFloat(handle,m_fConfig[i])
     }
   uint dictionary[]=AG_VOCABULARY_ORDER;
   uint vocabulary_size=0;
   uint token=0;

   dFileReadUInt(handle,vocabulary_size)

   if(vocabulary_size!=AG_VOCABULARY)
      ReturnFalse;

   for(int i=0;i<AG_VOCABULARY;i++)
     {
      dFileReadUInt(handle,token)

      if(token!=dictionary[i])
         ReturnFalse;
     }
   dFileReadUInt(handle,m_iFinanceRandom)
   m_uSeed=(ulong)FileReadLong(handle);
   m_uTokenUpdates=(ulong)FileReadLong(handle);
   m_uTradeUpdates=(ulong)FileReadLong(handle);

   if(!m_cEvaluator.Configure(OpenCL,m_fConfig)||!m_cPool.Configure(OpenCL,m_fConfig)||!m_cPool.Load(handle)||
      !m_cSearch.Load(handle,OpenCL)||!m_cTrading.Load(handle,OpenCL)||!Workspace())
      ReturnFalse;

   CBufferFloat saved_anchor;
   CBufferFloat saved_metadata;

   if(!saved_anchor.Load(handle)||saved_anchor.Total()!=m_cWorkAnchor.Total()||!saved_metadata.Load(handle)||
      saved_metadata.Total()!=m_cWorkMetadata.Total())
      ReturnFalse;

   if(!PoolControl())
      ReturnFalse;

   m_bPublicationPermit=m_cPool.Control().At(AG_POOL_ACTIVE_READY)!=1;
   m_bConfigured=true;
   m_bWorkReady=false;
   m_iPendingTrade=-1;
   m_iTokenCount=0;
   m_iTradeCount=0;
   m_iTrainingKind=0;
   m_iPrefixLength=0;

   return(true);
  }
#endif
//+------------------------------------------------------------------+
