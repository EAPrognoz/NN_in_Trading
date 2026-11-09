#ifndef NEURONET_ALPHAGENPOOL_DECLARATIONS_MQH
#define NEURONET_ALPHAGENPOOL_DECLARATIONS_MQH
class CAlphaGenPool
  {
protected:
   COpenCLMy         *OpenCL;
   CBufferFloat      m_cConfig,m_cBanks,m_cWork,m_cRings,m_cControl;
   // Transient per-batch formula cache; deliberately not serialized.
   CBufferFloat      m_cEvaluationWork,m_cEvaluated;
   CBufferFloat      m_cSignal,m_cContext,m_cPreparedSignal,m_cPreparedContext,m_cDummy;
   int               m_iWindow,m_iTokens,m_iCapacity,m_iIC,m_iNorm,m_iContext;
   int               m_iBankSize,m_iRingSize,m_iWorkSize,m_iLookback;
   bool              Allocate(CBufferFloat &buffer,const int count)
     {
      if(count<1 || !buffer.BufferInit(count,0) || !buffer.BufferCreate(OpenCL))
         ReturnFalse;
      return(true);
     }
   bool              Transaction(CBufferFloat *market,CBufferFloat *tokens,const int length,const int mode);
   bool              SignalStep(CBufferFloat *market,const int mode,const bool first=false);
public:
                     CAlphaGenPool(void) {OpenCL=NULL;m_iWindow=0;}
   bool              Configure(COpenCLMy *context,const float &config[]);
   bool              Candidate(CBufferFloat *market,CBufferFloat *tokens,const int length)
     {return(Transaction(market,tokens,length,0));}
   bool              CommitSearchBatch(CBufferFloat *market,CBufferFloat *candidates,const int count,
                                       const int first_row,CBufferFloat *prefixes,CBufferFloat *masks,
                                       CBufferFloat *samples,CBufferFloat *metadata,CBufferFloat *gae,
                                       CBufferFloat *summary);
   bool              Prepare(CBufferFloat *market)
     {return(Transaction(market,GetPointer(m_cDummy),0,1));}
   bool              PrepareAppend(CBufferFloat *market,const bool first=false) {return(SignalStep(market,2,first));}
   bool              Commit(void)
     {return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,2));}
   bool              RestartPrepare(void)
     {return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,AG_POOL_RESTART));}
   bool              AbortPrepare(void)
     {return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,3));}
   bool              ResetContext(void)
     {return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,4));}
   bool              Observe(CBufferFloat *market) {return(SignalStep(market,0));}
   bool              Warmup(CBufferFloat *market,const bool first=false) {return(SignalStep(market,1,first));}
   CBufferFloat      *Control(void) {return(GetPointer(m_cControl));}
   CBufferFloat      *Signal(void) {return(GetPointer(m_cSignal));}
   CBufferFloat      *Context(void) {return(GetPointer(m_cContext));}
   CBufferFloat      *StagedContext(void) {return(GetPointer(m_cPreparedContext));}
   CBufferFloat      *StagedSignal(void) {return(GetPointer(m_cPreparedSignal));}
   CBufferFloat      *Banks(void) {return(GetPointer(m_cBanks));}
   CBufferFloat      *Rings(void) {return(GetPointer(m_cRings));}
   bool              Save(const int file_handle);
   bool              Load(const int file_handle);
  };

#endif
