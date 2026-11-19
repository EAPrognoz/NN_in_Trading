//+------------------------------------------------------------------+
//|                            NeuroNet_AlphaGenPoolDeclarations.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/25939 |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGENPOOL_DECLARATIONS_MQH
#define NEURONET_ALPHAGENPOOL_DECLARATIONS_MQH
//+------------------------------------------------------------------+
//| CAlphaGenPool - Define class interface                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| CAlphaGenPool                                                    |
//+------------------------------------------------------------------+
class CAlphaGenPool
  {
protected:
   COpenCLMy    *OpenCL;
   CBufferFloat m_cConfig;
   CBufferFloat m_cBanks;
   CBufferFloat m_cWork;
   CBufferFloat m_cRings;
   CBufferFloat m_cControl;

//--- transient per-batch formula cache; deliberately not serialized.
   CBufferFloat m_cEvaluationWork;
   CBufferFloat m_cEvaluated;
   CBufferFloat m_cSignal;
   CBufferFloat m_cContext;
   CBufferFloat m_cPreparedSignal;
   CBufferFloat m_cPreparedContext;
   CBufferFloat m_cDummy;
   CBufferFloat m_cHistory;
   CBufferFloat m_cHistoryValues;
   int          m_iHistoryCount;
   int          m_iWindow;
   int          m_iTokens;
   int          m_iCapacity;
   int          m_iIC;
   int          m_iNorm;
   int          m_iContext;
   int          m_iBankSize;
   int          m_iRingSize;
   int          m_iWorkSize;
   int          m_iLookback;
   //+------------------------------------------------------------------+
   //| Allocate - Implement Allocate                                    |
   //+------------------------------------------------------------------+
   bool Allocate(CBufferFloat&buffer,const int count)
     {
      if(count<1||!buffer.BufferInit(count,0)||!buffer.BufferCreate(OpenCL))
         ReturnFalse;

      return(true);
     }
   bool Transaction(CBufferFloat*market,CBufferFloat*tokens,const int length,const int mode);
   bool SignalStep(CBufferFloat*market,const int mode,const bool first=false,const int count=0);
public:
                     CAlphaGenPool(void){OpenCL=NULL;m_iWindow=0;m_iHistoryCount=0;}
   bool Configure(COpenCLMy*context,const float&config[]);
   //+------------------------------------------------------------------+
   //| Candidate - Implement Candidate                                  |
   //+------------------------------------------------------------------+
   bool Candidate(CBufferFloat*market,CBufferFloat*tokens,const int length)
     {
      return(Transaction(market,tokens,length,0));
     }
   bool CommitSearchBatch(CBufferFloat*market,CBufferFloat*candidates,const int count,
                          const int    first_row,CBufferFloat*prefixes,CBufferFloat*masks,
                          CBufferFloat *samples,CBufferFloat*metadata,CBufferFloat*gae,
                          CBufferFloat *summary);
   //+------------------------------------------------------------------+
   //| Prepare - Implement Prepare                                      |
   //+------------------------------------------------------------------+
   bool Prepare(CBufferFloat*market)
     {
      return(Transaction(market,GetPointer(m_cDummy),0,1));
     }
   bool SetWarmupHistory(const float&history[],const int count);
//+------------------------------------------------------------------+
//| PrepareAppend                                                    |
//+------------------------------------------------------------------+
   bool PrepareAppend(CBufferFloat*market,const bool first=false)
     {
      if(first&&m_iHistoryCount>0)
        {
         int count=m_iHistoryCount;
         m_iHistoryCount=0;

         return(SignalStep(GetPointer(m_cHistory),2,true,count));
        }

      return(SignalStep(market,2,first));
     }
   //+------------------------------------------------------------------+
   //| Commit - Implement Commit                                        |
   //+------------------------------------------------------------------+
   bool Commit(void)
     {
      return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,2));
     }
   //+------------------------------------------------------------------+
   //| RestartPrepare - Implement RestartPrepare                        |
   //+------------------------------------------------------------------+
   bool RestartPrepare(void)
     {
      return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,AG_POOL_RESTART));
     }
   //+------------------------------------------------------------------+
   //| AbortPrepare - Implement AbortPrepare                            |
   //+------------------------------------------------------------------+
   bool AbortPrepare(void)
     {
      return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,3));
     }
   //+------------------------------------------------------------------+
   //| ResetContext - Implement ResetContext                            |
   //+------------------------------------------------------------------+
   bool ResetContext(void)
     {
      return(Transaction(GetPointer(m_cDummy),GetPointer(m_cDummy),0,4));
     }
//+------------------------------------------------------------------+
//| Observe                                                          |
//+------------------------------------------------------------------+
   bool         Observe(CBufferFloat*market)
     {
      return(SignalStep(market,0));
     }
//+------------------------------------------------------------------+
//| Warmup                                                           |
//+------------------------------------------------------------------+
   bool         Warmup(CBufferFloat*market,const bool first=false)
     {
      return(SignalStep(market,1,first));
     }
//+------------------------------------------------------------------+
//| Control                                                          |
//+------------------------------------------------------------------+
   CBufferFloat *Control(void)
     {
      return(GetPointer(m_cControl));
     }
//+------------------------------------------------------------------+
//| Signal                                                           |
//+------------------------------------------------------------------+
   CBufferFloat *Signal(void)
     {
      return(GetPointer(m_cSignal));
     }
//+------------------------------------------------------------------+
//| Context                                                          |
//+------------------------------------------------------------------+
   CBufferFloat *Context(void)
     {
      return(GetPointer(m_cContext));
     }
//+------------------------------------------------------------------+
//| StagedContext                                                    |
//+------------------------------------------------------------------+
   CBufferFloat *StagedContext(void)
     {
      return(GetPointer(m_cPreparedContext));
     }
//+------------------------------------------------------------------+
//| StagedSignal                                                     |
//+------------------------------------------------------------------+
   CBufferFloat *StagedSignal(void)
     {
      return(GetPointer(m_cPreparedSignal));
     }
//+------------------------------------------------------------------+
//| Banks                                                            |
//+------------------------------------------------------------------+
   CBufferFloat *Banks(void)
     {
      return(GetPointer(m_cBanks));
     }
//+------------------------------------------------------------------+
//| Rings                                                            |
//+------------------------------------------------------------------+
   CBufferFloat *Rings(void)
     {
      return(GetPointer(m_cRings));
     }
   bool         Save(const int file_handle);
   bool         Load(const int file_handle);
  };

#endif
