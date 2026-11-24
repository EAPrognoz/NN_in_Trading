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
#ifdef NN_ALPHAQCM
   CBufferFloat m_cQCMFit;
   CBufferFloat m_cQCMScores;
   bool m_bQCMUncertain;
   bool m_bQCMBaseline;
   ulong m_uQCMWindow;
   int m_iQCMMarket;
   int m_iQCMPacked;
   int m_iQCMBatch;
   int m_iQCMNext;
   int m_iQCMStatus;

public:
   bool QCMRefresh(CBufferFloat *market,const bool pin=false,const ulong window=0);
   bool QCMEvaluateBatch(CBufferFloat *market,CBufferFloat *packed,const int count);
   bool QCMCommitCandidate(CBufferFloat *market,CBufferFloat *packed,const int count,const int index);
   bool QCMPrepareAppend(CBufferFloat *market,const bool first=false);
   bool QCMObserve(CBufferFloat *market){return(QCMSignalStep(market,0,false,0));}
   bool QCMWarmup(CBufferFloat *market,const bool first=false){return(QCMSignalStep(market,1,first,0));}
   bool QCMRewarmActive(CBufferFloat *market);
   bool QCMValidateWorkspace(const int candidates);
   bool QCMValidLoadedActive(void) const;
   int QCMStatus(void) const;
   bool QCMChanged(void){return(m_cQCMFit.Total()==3&&m_cQCMFit.At(2)==1);}
   float QCMFitIC(void){return(m_cQCMFit.Total()==3?m_cQCMFit.At(0):0);}
   bool QCMAggregateUsable(void){return(m_cQCMFit.Total()==3&&m_cQCMFit.At(1)==1);}
   bool QCMUncertain(void){return(m_bQCMUncertain);}
                     CAlphaGenPool(void);
#else
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
#endif
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
#ifdef NN_ALPHAQCM
   bool PrepareAppend(CBufferFloat*market,const bool first=false);
#else
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
#endif
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

#ifdef NN_ALPHAQCM
protected:
   bool Allocate(CBufferFloat&buffer,const int count);
   bool Transaction(CBufferFloat*market,CBufferFloat*tokens,const int length,const int mode);
   bool SignalStep(CBufferFloat*market,const int mode,const bool first=false,const int count=0);
   bool QCMEnsure(CBufferFloat &buffer,const long count);
   bool QCMRun(const int kernel,const int groups);
   bool QCMBindAdvance(const int begin,const int end);
   bool QCMReadStatus(const int changes,const uint offset=0);
   bool QCMCache(CBufferFloat *market,CBufferFloat *source,CBufferFloat *result,
                 const int count,const bool packed);
   bool QCMOrdered(CBufferFloat *market,CBufferFloat *packed,CBufferFloat *values,
                   const int index,const int mode);
   bool QCMSignalStep(CBufferFloat *market,const int mode,const bool first,const int count);
#endif
  };

#ifdef NN_ALPHAQCM
//+------------------------------------------------------------------+
//| CAlphaGenPool - Initialize the AlphaQCM pool state                |
//+------------------------------------------------------------------+
CAlphaGenPool::CAlphaGenPool(void)
  {
   OpenCL=NULL;
   m_iWindow=0;
   m_iHistoryCount=0;
   m_bQCMUncertain=false;
   m_bQCMBaseline=false;
   m_iQCMBatch=0;
   m_iQCMNext=0;
   m_iQCMStatus=AG_OK;
   m_uQCMWindow=0;
   m_iQCMMarket=-1;
   m_iQCMPacked=-1;
  }

//+------------------------------------------------------------------+
//| Allocate - Allocate a pool buffer                                |
//+------------------------------------------------------------------+
bool CAlphaGenPool::Allocate(CBufferFloat&buffer,const int count)
  {
   if(count<1||!buffer.BufferInit(count,0)||!buffer.BufferCreate(OpenCL))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| PrepareAppend - Append history or the current market window      |
//+------------------------------------------------------------------+
bool CAlphaGenPool::PrepareAppend(CBufferFloat*market,const bool first)
  {
   if(first&&m_iHistoryCount>0)
     {
      int count=m_iHistoryCount;
      m_iHistoryCount=0;

      return(SignalStep(GetPointer(m_cHistory),2,true,count));
     }

   return(SignalStep(market,2,first));
  }
#endif

#endif
