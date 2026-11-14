//+------------------------------------------------------------------+
//| NeuroNet_AlphaGenProfile.mqh                                      |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGENPROFILE_MQH
#define NEURONET_ALPHAGENPROFILE_MQH
#include <OpenCL/OpenCL.mqh>
//+------------------------------------------------------------------+
//| CAlphaGenPPOProfile - Measure synchronized PPO stage boundaries    |
//+------------------------------------------------------------------+
class CAlphaGenPPOProfile
  {
private:
   COpenCL *m_pContext;
   int      m_iBuffer;
   int      m_iKind;
   int      m_iStage;
   int      m_iPass;
   int      m_iExample;
   ulong    m_uStart;
   ulong    m_uBegin;
   ulong    m_uBody[14];
   ulong    m_uWait[14];
   ulong    m_uMax[14];
   ulong    m_uCount[14];
   int      m_iMaxPass[14];
   int      m_iMaxExample[14];
   ulong    m_uExamples[2];
   ulong    m_uTotal[2];
   ulong    m_uEntry[2];

   //+------------------------------------------------------------------+
   //| Enabled - Read the optional diagnostic input                     |
   //+------------------------------------------------------------------+
   bool Enabled(void)const
     {
#ifdef ALPHAGEN_PPO_PROFILE
      return(ALPHAGEN_PPO_PROFILE);
#else
      return(false);
#endif
     }
   //+------------------------------------------------------------------+
   //| Fence - Drain prior queue work through a blocking scalar read    |
   //+------------------------------------------------------------------+
   bool Fence(void)
     {
      float value[1];

      ResetLastError();
      if(!m_pContext||!m_pContext.BufferRead(m_iBuffer,value,0,0,1))
        {
         PrintFormat("AlphaGen PPO profile fence failed: kind=%d stage=%d error=%d",
                     m_iKind+1,m_iStage,GetLastError());
         return(false);
        }
      return(true);
     }
public:
   //+------------------------------------------------------------------+
   //| CAlphaGenPPOProfile - Initialize transient diagnostic counters    |
   //+------------------------------------------------------------------+
                     CAlphaGenPPOProfile(void):m_pContext(NULL),m_iBuffer(-1),m_iKind(0),m_iStage(-1),
                                               m_iPass(0),m_iExample(0),m_uStart(0),m_uBegin(0)
     {
      ArrayInitialize(m_uBody,0);
      ArrayInitialize(m_uWait,0);
      ArrayInitialize(m_uMax,0);
      ArrayInitialize(m_uCount,0);
      ArrayInitialize(m_iMaxPass,0);
      ArrayInitialize(m_iMaxExample,0);
      ArrayInitialize(m_uExamples,0);
      ArrayInitialize(m_uTotal,0);
      ArrayInitialize(m_uEntry,0);
     }
   //+------------------------------------------------------------------+
   //| ~CAlphaGenPPOProfile - Report accumulated diagnostic samples      |
   //+------------------------------------------------------------------+
                    ~CAlphaGenPPOProfile(void)
     {
      Report("release");
     }
   //+------------------------------------------------------------------+
   //| Begin - Start one example after draining the preceding work      |
   //+------------------------------------------------------------------+
   bool Begin(COpenCL*context,const int buffer,const int kind,const int pass,const int example)
     {
      if(!Enabled())
         return(true);

      if(kind<1||kind>2)
         return(false);

      m_pContext=context;
      m_iBuffer=buffer;
      m_iKind=kind-1;
      m_iPass=pass;
      m_iExample=example;
      m_iStage=0;
      m_uBegin=GetMicrosecondCount();
      if(!Fence())
         return(false);

      m_uStart=GetMicrosecondCount();
      m_uEntry[m_iKind]+=m_uStart-m_uBegin;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Mark - Attribute body and additional fence time separately       |
   //+------------------------------------------------------------------+
   bool Mark(const int next)
     {
      if(m_iStage<0)
         return(true);

      if(next<0||next>7)
         return(false);

      ulong submitted=GetMicrosecondCount();
      if(!Fence())
         return(false);

      ulong done=GetMicrosecondCount();
      ulong elapsed=done-m_uStart;
      int   slot=m_iKind*7+m_iStage;

      m_uBody[slot]+=submitted-m_uStart;
      m_uWait[slot]+=done-submitted;
      m_uCount[slot]++;
      if(elapsed>m_uMax[slot])
        {
         m_uMax[slot]=elapsed;
         m_iMaxPass[slot]=m_iPass;
         m_iMaxExample[slot]=m_iExample;
        }
      m_iStage=next<7?next:-1;
      m_uStart=done;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| End - Complete one example and publish periodic totals           |
   //+------------------------------------------------------------------+
   bool End(void)
     {
      if(m_iStage<0)
         return(true);

      if(!Mark(7))
         return(false);

      m_uExamples[m_iKind]++;
      m_uTotal[m_iKind]+=GetMicrosecondCount()-m_uBegin;
      if(m_uExamples[m_iKind]%1024==0)
         Report("interval");

      return(true);
     }
   //+------------------------------------------------------------------+
   //| Report - Print cumulative host wall time rather than GPU events  |
   //+------------------------------------------------------------------+
   void Report(const string reason)
     {
      string names[7]={"replay","forward","backward","numeric","metadata","adam","finish"};

      for(int kind=0;kind<2;kind++)
        {
         if(m_uExamples[kind]==0&&m_uCount[kind*7]==0)
            continue;

         PrintFormat("AlphaGen PPO profile summary reason=%s kind=%d examples=%I64u total_us=%I64u entry_fence_us=%I64u incomplete=%d",
                     reason,kind+1,m_uExamples[kind],m_uTotal[kind],m_uEntry[kind],m_iStage>=0&&m_iKind==kind?1:0);
         for(int stage=0;stage<7;stage++)
           {
            int slot=kind*7+stage;
            if(m_uCount[slot]==0)
               continue;

            PrintFormat("AlphaGen PPO profile kind=%d stage=%s count=%I64u body_mean_us=%.1f fence_mean_us=%.1f total_us=%I64u max_us=%I64u pass=%d example=%d",
                        kind+1,names[stage],m_uCount[slot],(double)m_uBody[slot]/m_uCount[slot],
                        (double)m_uWait[slot]/m_uCount[slot],m_uBody[slot]+m_uWait[slot],m_uMax[slot],
                        m_iMaxPass[slot],m_iMaxExample[slot]);
           }
        }
     }
  };
#endif // NEURONET_ALPHAGENPROFILE_MQH
