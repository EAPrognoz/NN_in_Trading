//+------------------------------------------------------------------+
//|                                       NeuroNet_AlphaQCMReplay.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_REPLAY_MQH
#define NEURONET_ALPHAQCM_REPLAY_MQH

#define def_k_AQReplayTreeLeaves       530
#define def_k_AQReplayTreeLevel        531
#define def_k_AQReplayStageEpisode     532
#define def_k_AQReplayPublishEpisode   533
#define def_k_AQReplaySample           534
#define def_k_AQReplayGather           535
#define def_k_AQReplayTau              536
#define def_k_AQReplayStagePriorities  537
#define def_k_AQReplayCommitPriorities 538
#define def_k_AQReplayMirrorTree       539
#define def_k_AQReplayCheckIntervals   540
#define def_k_AQReplayStatus           541

#define AQC_REPLAY_MAGIC        0x41515250
#define AQC_REPLAY_VERSION      1
#define AQC_REPLAY_ROW_HEADER   6
#define AQC_REPLAY_META_WIDTH   6
#define AQC_REPLAY_FILE_ROWS    256

//+------------------------------------------------------------------+
//| ENUM_AQC_REPLAY_STATUS - Replay transaction result                |
//+------------------------------------------------------------------+
enum ENUM_AQC_REPLAY_STATUS
  {
   AQC_REPLAY_OK=0,
   AQC_REPLAY_NUMERIC=1,
   AQC_REPLAY_PAYLOAD=2,
   AQC_REPLAY_STALE=3,
   AQC_REPLAY_RESOURCE=4,
   AQC_REPLAY_INCONSISTENT=5,
   AQC_REPLAY_OPENCL_ERROR=6
  };

//+------------------------------------------------------------------+
//| AQCReplayBatch - Borrowed device views of one sampled minibatch   |
//+------------------------------------------------------------------+
struct AQCReplayBatch
  {
   int tokens;       // int[B,L]
   int lengths;      // int[B]
   int next_tokens;  // int[B,L]
   int next_lengths; // int[B]
   int next_legal;   // int[B,A]
   int actions;      // int[B]
   int rewards;      // float[B]
   int done;         // int[B]
   int weights;      // float[B]
   int tau;          // float[B,N]
   int next_tau;     // float[B,N']
  };

//+------------------------------------------------------------------+
//| ENUM_AQC_REPLAY_BUFFER - Replay-owned typed OpenCL allocations    |
//+------------------------------------------------------------------+
enum ENUM_AQC_REPLAY_BUFFER
  {
   AQC_RB_STATE,
   AQC_RB_METADATA,
   AQC_RB_CHANGES,
   AQC_RB_STATUS,
   AQC_RB_PARTIAL,
   AQC_RB_SCALAR,
   AQC_RB_SELECTED,
   AQC_RB_GENERATIONS,
   AQC_RB_TOKENS,
   AQC_RB_LENGTHS,
   AQC_RB_NEXT_TOKENS,
   AQC_RB_NEXT_LENGTHS,
   AQC_RB_NEXT_LEGAL,
   AQC_RB_ACTIONS,
   AQC_RB_DONE,
   AQC_RB_COUNT
  };

//+------------------------------------------------------------------+
//| CAlphaQCMReplay - GPU transition ring and prioritized sampling    |
//+------------------------------------------------------------------+
class CAlphaQCMReplay
  {
private:
   COpenCLMy *OpenCL;
   int        m_iBuffer[AQC_RB_COUNT];
   int        m_iCapacity;
   int        m_iLength;
   int        m_iActions;
   int        m_iBatch;
   int        m_iQuantiles;
   int        m_iTargetQuantiles;
   int        m_iLeaves;
   int        m_iDepth;
   uint       m_uTreeLanes;
   uint       m_uStatusLanes;
   int        m_iSize;
   int        m_iHead;
   ulong      m_uVerified;
   float      m_fPriorityEpsilon;
   float      m_fBetaInitial;
   ulong      m_uBetaUpdates;
   bool       m_bReady;
   bool       m_bConsistent;
   bool       m_bSnapshot;
   bool       m_bRestoring;
   int        m_iDirty;
   ENUM_AQC_REPLAY_STATUS m_eStatus;

//--- Persistent float data and the fixed association CDF mirrors.
   CBufferFloat m_cRewards;
   CBufferFloat m_cPriorities;
   CBufferFloat m_cLiveTree;
   CBufferFloat m_cStagedTree;

//--- Reload staging is released once the post-Build device restore completes.
   int   m_iSavedState[];
   ulong m_uSavedMetadata[];
   float m_fSavedRewards[];
   float m_fSavedPriorities[];

//--- Batch float buffers share the lifetime of the typed batch views.
   CBufferFloat m_cBatchRewards;
   CBufferFloat m_cBatchWeights;
   CBufferFloat m_cTau;
   CBufferFloat m_cNextTau;

   void Clear(void);
   void ClearRestore(void);
   bool AllocateInteger(const ENUM_AQC_REPLAY_BUFFER slot,const long count,const int width);
   bool AllocateFloat(CBufferFloat &buffer,const long count);
   bool Fail(const ENUM_AQC_REPLAY_STATUS status,const string message);
   bool Run(const int kernel,const uint first,const uint second=0,const uint local=0);
   bool BuildTree(const int size,const int changes);
   bool MirrorTree(const bool publish,const int changes);
   bool CheckTree(const int size);
   bool CheckStatus(const int count);
   bool Discard(void);

public:
   CAlphaQCMReplay(void);
   ~CAlphaQCMReplay(void);
   bool Configure(COpenCLMy *context,const int capacity,const int length,const int actions,
                  const int batch,const int current_quantiles,const int target_quantiles,
                  const float priority_epsilon,const float beta_initial,
                  const ulong beta_updates,const float alpha);
   bool Initialize(void);
   bool PublishEpisode(const int results,const int row,const int steps,const int rewards,
                       const int reward_index,const ulong candidate_id,const ulong window_revision,
                       const ulong prepared_before,const ulong prepared_after,const ulong active_generation);
   bool Sample(const ulong seed,const ulong logical_id,const ulong accepted,AQCReplayBatch &batch);
   bool PreparePriorities(const int loss);
   bool CommitPriorities(void);
   bool ReleaseBatch(void);
   bool Save(const int file);
   bool Load(const int file,const int capacity,const int length,const int actions,
             const ulong next_candidate,const bool restore);
   bool Restore(void);
   int Size(void) const { return(m_iSize); }
   ulong Verified(void) const { return(m_uVerified); }
   bool Consistent(void) const { return(m_bConsistent); }
   ENUM_AQC_REPLAY_STATUS LastStatus(void) const { return(m_eStatus); }
  };

//+------------------------------------------------------------------+
//| CAlphaQCMReplay - Start with no device allocations                 |
//+------------------------------------------------------------------+
CAlphaQCMReplay::CAlphaQCMReplay(void)
  {
   OpenCL=NULL;
   ArrayInitialize(m_iBuffer,INVALID_HANDLE);
   Clear();
  }

//+------------------------------------------------------------------+
//| ~CAlphaQCMReplay - Release buffers before the borrowed runtime     |
//+------------------------------------------------------------------+
CAlphaQCMReplay::~CAlphaQCMReplay(void)
  {
   Clear();
  }

//+------------------------------------------------------------------+
//| ClearRestore - Release the checkpoint upload arrays               |
//+------------------------------------------------------------------+
void CAlphaQCMReplay::ClearRestore(void)
  {
   ArrayFree(m_iSavedState);
   ArrayFree(m_uSavedMetadata);
   ArrayFree(m_fSavedRewards);
   ArrayFree(m_fSavedPriorities);
   m_bRestoring=false;
  }

//+------------------------------------------------------------------+
//| Clear - Revoke views and release resources                         |
//+------------------------------------------------------------------+
void CAlphaQCMReplay::Clear(void)
  {
   if(!!OpenCL)
      for(int i=0;i<AQC_RB_COUNT;++i)
         if(m_iBuffer[i]>=0)
            OpenCL.BufferFree(m_iBuffer[i]);

   ArrayInitialize(m_iBuffer,INVALID_HANDLE);
   m_cRewards.Clear();
   m_cPriorities.Clear();
   m_cLiveTree.Clear();
   m_cStagedTree.Clear();
   m_cBatchRewards.Clear();
   m_cBatchWeights.Clear();
   m_cTau.Clear();
   m_cNextTau.Clear();
   ClearRestore();
   OpenCL=NULL;
   m_iCapacity=0;
   m_iLength=0;
   m_iActions=0;
   m_iBatch=0;
   m_iSize=0;
   m_iHead=0;
   m_uVerified=0;
   m_iDirty=0;
   m_bReady=false;
   m_bConsistent=true;
   m_bSnapshot=false;
   m_eStatus=AQC_REPLAY_OK;
  }

//+------------------------------------------------------------------+
//| AllocateInteger - Allocate a checked typed buffer                |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::AllocateInteger(const ENUM_AQC_REPLAY_BUFFER slot,const long count,const int width)
  {
   if(count<1||count>INT_MAX/width)
      ReturnFalse;

   m_iBuffer[slot]=OpenCL.AddBuffer((uint)(count*width),CL_MEM_READ_WRITE);
   if(m_iBuffer[slot]<0)
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| AllocateFloat - Allocate owned float storage                       |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::AllocateFloat(CBufferFloat &buffer,const long count)
  {
   if(count<1||count>INT_MAX/(int)sizeof(float)||
      !buffer.BufferInit((uint)count,0.0f)||!buffer.BufferCreate(OpenCL))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Fail - Keep the failure available to the application                |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Fail(const ENUM_AQC_REPLAY_STATUS status,const string message)
  {
   m_eStatus=status;
   ReturnFalseEx(message);
  }

//+------------------------------------------------------------------+
//| Configure - Reserve the fixed ring and minibatch geometry           |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Configure(COpenCLMy *context,const int capacity,const int length,const int actions,
                                const int batch,const int current_quantiles,const int target_quantiles,
                                const float priority_epsilon,const float beta_initial,
                                const ulong beta_updates,const float alpha)
  {
   Clear();
   if(!context||length<2||capacity<length||capacity>INT_MAX/16||actions<3||actions>64||
      batch<2||batch>capacity||current_quantiles<1||target_quantiles<1||alpha!=0.5f||
      !MathIsValidNumber(priority_epsilon)||priority_epsilon<=0.0f||beta_updates==0||
      !MathIsValidNumber(beta_initial)||beta_initial<=0.0f||beta_initial>1.0f)
      return(Fail(AQC_REPLAY_RESOURCE,"Unsupported replay profile or geometry"));

   OpenCL=context;
   m_iCapacity=capacity;
   m_iLength=length;
   m_iActions=actions;
   m_iBatch=batch;
   m_iQuantiles=current_quantiles;
   m_iTargetQuantiles=target_quantiles;
   m_fPriorityEpsilon=priority_epsilon;
   m_fBetaInitial=beta_initial;
   m_uBetaUpdates=beta_updates;
   m_iLeaves=1;
   m_iDepth=0;
   while(m_iLeaves<capacity)
     {
      m_iLeaves*=2;
      ++m_iDepth;
     }

//--- Persistent, per slot: int[L+6], ulong[6], and two floats. Two
//--- float4 trees have 2*next_power_of_two(capacity) nodes each.
   const long state_width=(long)length+AQC_REPLAY_ROW_HEADER;
   const long state_bytes=capacity*state_width*sizeof(int);
   const long metadata_bytes=(long)capacity*AQC_REPLAY_META_WIDTH*sizeof(ulong);
   const long tree_bytes=(long)m_iLeaves*8*sizeof(float);
   const long batch_largest=(long)batch*MathMax(MathMax(length,actions),
                                               MathMax(current_quantiles,target_quantiles))*sizeof(float);
   const long largest=MathMax(MathMax(state_bytes,metadata_bytes),MathMax(tree_bytes,batch_largest));
   const long batch_ints=5L*batch+2L*batch*length+(long)batch*actions;
   const long status_ints=MathMax(batch,length)+(long)capacity+(capacity+7L)/8+1;
   const long batch_floats=(long)batch*(2L+current_quantiles+target_quantiles);
   const long bytes=state_bytes+metadata_bytes+2*tree_bytes+(batch_ints+status_ints)*sizeof(int)+
                    (2L*capacity+batch_floats)*sizeof(float)+(long)batch*sizeof(ulong);
   const long limit=OpenCL.GetDeviceInfo(0x1010); // CL_DEVICE_MAX_MEM_ALLOC_SIZE
   const long memory=OpenCL.GetDeviceInfoInteger(CL_DEVICE_GLOBAL_MEM_SIZE);
   if(limit<largest||memory<bytes)
      return(Fail(AQC_REPLAY_RESOURCE,StringFormat("RESOURCE_UNSUPPORTED: replay needs %I64d total bytes and a %I64d-byte buffer",
                                                 bytes,largest)));
   PrintFormat("AlphaQCM replay floor: bytes=%I64d largest_buffer=%I64d, two parallel lanes, logical B=%d",
               bytes,largest,batch);

   if(!AllocateInteger(AQC_RB_STATE,capacity*state_width,sizeof(int))||
      !AllocateInteger(AQC_RB_METADATA,(long)capacity*AQC_REPLAY_META_WIDTH,sizeof(ulong))||
      !AllocateFloat(m_cRewards,capacity)||!AllocateFloat(m_cPriorities,capacity)||
      !AllocateFloat(m_cLiveTree,(long)m_iLeaves*8)||!AllocateFloat(m_cStagedTree,(long)m_iLeaves*8))
      return(Fail(AQC_REPLAY_RESOURCE,"Replay ring or CDF allocation failed"));

//--- Transient: max(B,L) changed slots, one status per physical slot, one
//--- reduction partial per eight statuses, and the gathered minibatch.
   if(!AllocateInteger(AQC_RB_CHANGES,MathMax(batch,length),sizeof(int))||
      !AllocateInteger(AQC_RB_STATUS,capacity,sizeof(int))||
      !AllocateInteger(AQC_RB_PARTIAL,(capacity+7)/8,sizeof(int))||
      !AllocateInteger(AQC_RB_SCALAR,1,sizeof(int))||
      !AllocateInteger(AQC_RB_SELECTED,batch,sizeof(int))||
      !AllocateInteger(AQC_RB_GENERATIONS,batch,sizeof(ulong))||
      !AllocateInteger(AQC_RB_TOKENS,(long)batch*length,sizeof(int))||
      !AllocateInteger(AQC_RB_LENGTHS,batch,sizeof(int))||
      !AllocateInteger(AQC_RB_NEXT_TOKENS,(long)batch*length,sizeof(int))||
      !AllocateInteger(AQC_RB_NEXT_LENGTHS,batch,sizeof(int))||
      !AllocateInteger(AQC_RB_NEXT_LEGAL,(long)batch*actions,sizeof(int))||
      !AllocateInteger(AQC_RB_ACTIONS,batch,sizeof(int))||
      !AllocateInteger(AQC_RB_DONE,batch,sizeof(int))||
      !AllocateFloat(m_cBatchRewards,batch)||!AllocateFloat(m_cBatchWeights,batch)||
      !AllocateFloat(m_cTau,(long)batch*current_quantiles)||
      !AllocateFloat(m_cNextTau,(long)batch*target_quantiles))
      return(Fail(AQC_REPLAY_RESOURCE,"Replay minibatch or scratch allocation failed"));

   return(true);
  }

//+------------------------------------------------------------------+
//| Run - Complete dependent dispatches before binding or reuse        |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Run(const int kernel,const uint first,const uint second,const uint local)
  {
   if(second>0)
     {
      uint work[2]={first,second};
      uint offset[2]={0,0};
      kernelExecute(kernel,offset,work)
     }
   else
     {
      uint work[1]={first};
      uint offset[1]={0};
      if(local==0)
        {
         kernelExecute(kernel,offset,work)
        }
      else
        {
         uint local_work[1]={local};
         kernelExecuteLoc(kernel,offset,work,local_work)
        }
     }

   if(!OpenCL.WaitAlphaGenKernel(kernel))
     {
      m_bConsistent=false;
      return(Fail(AQC_REPLAY_INCONSISTENT,"Replay kernel completion could not be established"));
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| BuildTree - Rebuild in fixed order or update unique touched nodes  |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::BuildTree(const int size,const int changes)
  {
   const int leaves=def_k_AQReplayTreeLeaves;
   const int level=def_k_AQReplayTreeLevel;
   if(changes==0)
     {
      setBuffer(leaves,0,m_cPriorities.GetIndex())
      setBuffer(leaves,1,m_cStagedTree.GetIndex())
      setArgument(leaves,2,size)
      setArgument(leaves,3,m_iLeaves)
      if(!Run(leaves,m_iLeaves))
         ReturnFalse;
     }

   setBuffer(level,0,m_cStagedTree.GetIndex())
   setBuffer(level,1,m_iBuffer[AQC_RB_CHANGES])
   setArgument(level,2,m_iLeaves)
   setArgument(level,4,changes)
   if(changes>0)
     {
      setArgument(level,3,m_iDepth)
      return(Run(level,m_uTreeLanes,0,m_uTreeLanes));
     }

   for(int height=1;height<=m_iDepth;++height)
     {
      setArgument(level,3,height)
      if(!Run(level,m_iLeaves>>height))
         ReturnFalse;
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| MirrorTree - Publish staged nodes or discard tentative writes      |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::MirrorTree(const bool publish,const int changes)
  {
   const int k=def_k_AQReplayMirrorTree;
   const int sparse=(changes>0)?1:0;
   const int source=publish?m_cStagedTree.GetIndex():m_cLiveTree.GetIndex();
   const int destination=publish?m_cLiveTree.GetIndex():m_cStagedTree.GetIndex();
   setBuffer(k,0,source)
   setBuffer(k,1,destination)
   setBuffer(k,2,m_iBuffer[AQC_RB_CHANGES])
   setArgument(k,3,m_iLeaves)
   setArgument(k,4,sparse)
   if(changes>0)
      return(Run(k,changes,m_iDepth+1));

   return(Run(k,2*m_iLeaves-1));
  }

//+------------------------------------------------------------------+
//| CheckStatus - Reduce per-item statuses without a payload readback  |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::CheckStatus(const int count)
  {
   const int k=def_k_AQReplayStatus;
   const int group_items=(int)m_uStatusLanes*4;
   int source=m_iBuffer[AQC_RB_STATUS];
   int remaining=count;
   while(remaining>0)
     {
      const int groups=(remaining+group_items-1)/group_items;
      const int output=(groups==1)?m_iBuffer[AQC_RB_SCALAR]:
                       ((source==m_iBuffer[AQC_RB_STATUS])?m_iBuffer[AQC_RB_PARTIAL]:m_iBuffer[AQC_RB_STATUS]);
      setBuffer(k,0,source)
      setBuffer(k,1,output)
      setArgument(k,2,remaining)
      if(!Run(k,(uint)groups*m_uStatusLanes,0,m_uStatusLanes))
         ReturnFalse;
      if(groups==1)
         break;
      source=output;
      remaining=groups;
     }

   int status[1];
   if(!OpenCL.BufferRead(m_iBuffer[AQC_RB_SCALAR],status,0,0,1))
      return(Fail(AQC_REPLAY_OPENCL_ERROR,"Replay status read failed"));
   if(status[0]!=AQC_REPLAY_OK)
      return(Fail((ENUM_AQC_REPLAY_STATUS)status[0],StringFormat("Replay device status %d",status[0])));

   return(true);
  }

//+------------------------------------------------------------------+
//| CheckTree - Prove every live CDF interval has positive width       |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::CheckTree(const int size)
  {
   float root[4];
   if(!OpenCL.BufferRead(m_cStagedTree.GetIndex(),root,4*sizeof(float),0,4))
      return(Fail(AQC_REPLAY_OPENCL_ERROR,"Replay CDF root read failed"));
   if(size==0&&root[0]==0.0f&&root[3]==0.0f)
      return(true);
   const double total=(double)root[0]+root[3];
   if(!MathIsValidNumber(total)||total<=0.0||!MathIsValidNumber(root[1])||
      root[1]<=0.0f||!MathIsValidNumber(root[2])||root[2]<=0.0f||
      (float)(root[1]/total)<=0.0f)
      return(Fail(AQC_REPLAY_NUMERIC,"NUMERIC_ERROR: nonpositive or nonfinite PER distribution"));

//--- Each positive prefix uses at most twice the tree depth in additions.
//--- The error per compensated addition is at most 3*u^2 relative (u=2^-24).
//--- Allow both endpoints' errors and rounding in their accumulated bound.
   const double unit_roundoff=1.0/16777216.0;
   const double interval_error=16.0*(m_iDepth+2)*unit_roundoff*unit_roundoff*total;
   if((double)root[1]>interval_error)
      return(true);

   const int k=def_k_AQReplayCheckIntervals;
   setBuffer(k,0,m_cStagedTree.GetIndex())
   setBuffer(k,1,m_iBuffer[AQC_RB_STATUS])
   setArgument(k,2,m_iLeaves)
   if(!Run(k,size)||!CheckStatus(size))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Discard - Restore the staging mirror after a completed preflight   |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Discard(void)
  {
   if(!m_bConsistent)
      return(Fail(AQC_REPLAY_INCONSISTENT,"Replay has pending or inconsistent device state"));
   if(m_iDirty>0)
     {
      if(!MirrorTree(false,m_iDirty))
        {
         m_bConsistent=false;
         return(Fail(AQC_REPLAY_INCONSISTENT,"Unable to discard staged replay writes"));
        }
      m_iDirty=0;
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Initialize - Build fresh or restored mirrors after OpenCL Build   |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Initialize(void)
  {
   if(!OpenCL||m_iCapacity<1||m_bReady||m_bRestoring)
      return(Fail(AQC_REPLAY_PAYLOAD,"Replay initialization state is invalid"));

   m_uTreeLanes=OpenCL.AlphaGenGroupSize(def_k_AQReplayTreeLevel,128,2);
   m_uStatusLanes=OpenCL.AlphaGenGroupSize(def_k_AQReplayStatus,128,2);
   if(m_uTreeLanes==0||m_uStatusLanes==0)
      return(Fail(AQC_REPLAY_RESOURCE,"Replay kernels require at least two lanes"));

   if(!BuildTree(m_iSize,0)||!CheckTree(m_iSize)||!MirrorTree(true,0))
      ReturnFalse;

   m_bReady=true;
   return(true);
  }

//+------------------------------------------------------------------+
//| PublishEpisode - Publish one complete, confirmed FIFO episode     |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::PublishEpisode(const int results,const int row,const int steps,const int rewards,
                                     const int reward_index,const ulong candidate_id,const ulong window_revision,
                                     const ulong prepared_before,const ulong prepared_after,const ulong active_generation)
  {
   m_eStatus=AQC_REPLAY_OPENCL_ERROR;
   const ulong stride=1+(ulong)m_iLength*(4+(ulong)m_iActions);
   if(!m_bReady||!m_bConsistent||m_bSnapshot||m_iDirty!=0||results<0||rewards<0||
      row<0||steps<2||steps>m_iLength||reward_index<0||m_uVerified>ULONG_MAX-(ulong)steps||
      prepared_after<prepared_before||((ulong)row+1)*stride>INT_MAX/sizeof(float)||
      OpenCL.GetBufferSize(results)<((ulong)row+1)*stride*sizeof(float)||
      OpenCL.GetBufferSize(rewards)<((ulong)reward_index+1)*sizeof(float))
      return(Fail(AQC_REPLAY_PAYLOAD,"Invalid replay publication geometry, lifetime, or revisions"));

   int k=def_k_AQReplayStageEpisode;
   setBuffer(k,0,results)
   setBuffer(k,1,rewards)
   setBuffer(k,2,m_cLiveTree.GetIndex())
   setBuffer(k,3,m_cStagedTree.GetIndex())
   setBuffer(k,4,m_iBuffer[AQC_RB_CHANGES])
   setBuffer(k,5,m_iBuffer[AQC_RB_STATUS])
   setArgument(k,6,row)
   setArgument(k,7,steps)
   setArgument(k,8,reward_index)
   setArgument(k,9,m_iHead)
   setArgument(k,10,m_iCapacity)
   setArgument(k,11,m_iLeaves)
   setArgument(k,12,m_iSize)
   setArgument(k,13,m_iLength)
   setArgument(k,14,m_iActions)
   if(!Run(k,steps))
      ReturnFalse;

   m_iDirty=steps;
   const int size=(int)MathMin(m_iSize+steps,m_iCapacity);
   if(!CheckStatus(steps)||!BuildTree(size,steps)||!CheckTree(size))
     {
      if(!Discard())
         ReturnFalse;
      ReturnFalse;
     }

   k=def_k_AQReplayPublishEpisode;
   setBuffer(k,0,results)
   setBuffer(k,1,rewards)
   setBuffer(k,2,m_cStagedTree.GetIndex())
   setBuffer(k,3,m_iBuffer[AQC_RB_STATE])
   setBuffer(k,4,m_iBuffer[AQC_RB_METADATA])
   setBuffer(k,5,m_cRewards.GetIndex())
   setBuffer(k,6,m_cPriorities.GetIndex())
   setBuffer(k,7,m_iBuffer[AQC_RB_CHANGES])
   setArgument(k,8,row)
   setArgument(k,9,steps)
   setArgument(k,10,reward_index)
   setArgument(k,11,m_iLeaves)
   setArgument(k,12,m_iLength)
   setArgument(k,13,m_iActions)
   setArgument(k,14,m_uVerified)
   setArgument(k,15,candidate_id)
   setArgument(k,16,window_revision)
   setArgument(k,17,prepared_before)
   setArgument(k,18,prepared_after)
   setArgument(k,19,active_generation)

   m_bConsistent=false;
   m_eStatus=AQC_REPLAY_INCONSISTENT;
   if(!Run(k,steps,m_iLength+AQC_REPLAY_ROW_HEADER)||!MirrorTree(true,steps))
      ReturnFalse;

   m_iSize=size;
   m_iHead=(m_iHead+steps)%m_iCapacity;
   m_uVerified+=(ulong)steps;
   m_iDirty=0;
   m_bConsistent=true;
   m_eStatus=AQC_REPLAY_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| Sample - Freeze a minibatch, gather state and generate tau        |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Sample(const ulong seed,const ulong logical_id,const ulong accepted,AQCReplayBatch &batch)
  {
   m_eStatus=AQC_REPLAY_OPENCL_ERROR;
   if(!m_bReady||!m_bConsistent||m_iSize<1||m_bSnapshot||m_iDirty!=0)
      return(Fail(AQC_REPLAY_PAYLOAD,"Replay sampling boundary is invalid"));

   const float progress=(accepted>=m_uBetaUpdates)?1.0f:(float)((double)accepted/(double)m_uBetaUpdates);
   const float beta=m_fBetaInitial+(1.0f-m_fBetaInitial)*progress;
   int k=def_k_AQReplaySample;
   setBuffer(k,0,m_cLiveTree.GetIndex())
   setBuffer(k,1,m_iBuffer[AQC_RB_METADATA])
   setBuffer(k,2,m_iBuffer[AQC_RB_SELECTED])
   setBuffer(k,3,m_iBuffer[AQC_RB_GENERATIONS])
   setBuffer(k,4,m_cBatchWeights.GetIndex())
   setBuffer(k,5,m_iBuffer[AQC_RB_STATUS])
   setArgument(k,6,m_iLeaves)
   setArgument(k,7,m_iSize)
   setArgument(k,8,beta)
   setArgument(k,9,seed)
   setArgument(k,10,logical_id)
   if(!Run(k,m_iBatch)||!CheckStatus(m_iBatch))
      ReturnFalse;

   k=def_k_AQReplayGather;
   setBuffer(k,0,m_iBuffer[AQC_RB_STATE])
   setBuffer(k,1,m_cRewards.GetIndex())
   setBuffer(k,2,m_iBuffer[AQC_RB_SELECTED])
   setBuffer(k,3,m_iBuffer[AQC_RB_TOKENS])
   setBuffer(k,4,m_iBuffer[AQC_RB_LENGTHS])
   setBuffer(k,5,m_iBuffer[AQC_RB_NEXT_TOKENS])
   setBuffer(k,6,m_iBuffer[AQC_RB_NEXT_LENGTHS])
   setBuffer(k,7,m_iBuffer[AQC_RB_NEXT_LEGAL])
   setBuffer(k,8,m_iBuffer[AQC_RB_ACTIONS])
   setBuffer(k,9,m_cBatchRewards.GetIndex())
   setBuffer(k,10,m_iBuffer[AQC_RB_DONE])
   setArgument(k,11,m_iLength)
   setArgument(k,12,m_iActions)
   if(!Run(k,m_iBatch,(uint)MathMax(m_iLength,m_iActions)))
      ReturnFalse;

   k=def_k_AQReplayTau;
   setBuffer(k,0,m_cTau.GetIndex())
   setBuffer(k,1,m_cNextTau.GetIndex())
   setArgument(k,2,m_iQuantiles)
   setArgument(k,3,m_iTargetQuantiles)
   setArgument(k,4,seed)
   setArgument(k,5,logical_id)
   if(!Run(k,m_iBatch,(uint)MathMax(m_iQuantiles,m_iTargetQuantiles)))
      ReturnFalse;

   batch.tokens=m_iBuffer[AQC_RB_TOKENS];
   batch.lengths=m_iBuffer[AQC_RB_LENGTHS];
   batch.next_tokens=m_iBuffer[AQC_RB_NEXT_TOKENS];
   batch.next_lengths=m_iBuffer[AQC_RB_NEXT_LENGTHS];
   batch.next_legal=m_iBuffer[AQC_RB_NEXT_LEGAL];
   batch.actions=m_iBuffer[AQC_RB_ACTIONS];
   batch.rewards=m_cBatchRewards.GetIndex();
   batch.done=m_iBuffer[AQC_RB_DONE];
   batch.weights=m_cBatchWeights.GetIndex();
   batch.tau=m_cTau.GetIndex();
   batch.next_tau=m_cNextTau.GetIndex();
   m_bSnapshot=true;
   m_eStatus=AQC_REPLAY_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| PreparePriorities - Preflight stale slots and prospective CDF     |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::PreparePriorities(const int loss)
  {
   m_eStatus=AQC_REPLAY_OPENCL_ERROR;
   if(!m_bReady||!m_bSnapshot||!m_bConsistent||loss<0||
      OpenCL.GetBufferSize(loss)<(ulong)m_iBatch*sizeof(float))
      return(Fail(AQC_REPLAY_PAYLOAD,"Replay priority preflight requires a frozen minibatch"));
   if(!Discard())
      ReturnFalse;

   const int k=def_k_AQReplayStagePriorities;
   setBuffer(k,0,m_iBuffer[AQC_RB_SELECTED])
   setBuffer(k,1,m_iBuffer[AQC_RB_GENERATIONS])
   setBuffer(k,2,m_iBuffer[AQC_RB_METADATA])
   setBuffer(k,3,loss)
   setBuffer(k,4,m_cStagedTree.GetIndex())
   setBuffer(k,5,m_iBuffer[AQC_RB_CHANGES])
   setBuffer(k,6,m_iBuffer[AQC_RB_STATUS])
   setArgument(k,7,m_iBatch)
   setArgument(k,8,m_iSize)
   setArgument(k,9,m_iLeaves)
   setArgument(k,10,m_fPriorityEpsilon)
   if(!Run(k,m_iBatch))
      ReturnFalse;

   m_iDirty=m_iBatch;
   if(!CheckStatus(m_iBatch)||!BuildTree(m_iSize,m_iBatch)||!CheckTree(m_iSize))
     {
      if(!Discard())
         ReturnFalse;
      ReturnFalse;
     }

   m_eStatus=AQC_REPLAY_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| CommitPriorities - Publish one priority per unique frozen slot    |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::CommitPriorities(void)
  {
   if(!m_bReady||!m_bConsistent||!m_bSnapshot||m_iDirty!=m_iBatch)
      return(Fail(AQC_REPLAY_INCONSISTENT,"Replay priority publication has no successful preflight"));

   m_bConsistent=false;
   m_eStatus=AQC_REPLAY_INCONSISTENT;
   const int k=def_k_AQReplayCommitPriorities;
   setBuffer(k,0,m_iBuffer[AQC_RB_CHANGES])
   setBuffer(k,1,m_cStagedTree.GetIndex())
   setBuffer(k,2,m_cPriorities.GetIndex())
   setArgument(k,3,m_iLeaves)
   if(!Run(k,m_iBatch)||!MirrorTree(true,m_iBatch))
      ReturnFalse;

   m_iDirty=0;
   m_bSnapshot=false;
   m_bConsistent=true;
   m_eStatus=AQC_REPLAY_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| ReleaseBatch - Revoke views at a clean abort boundary             |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::ReleaseBatch(void)
  {
   if(!Discard())
      ReturnFalse;

   m_bSnapshot=false;
   return(true);
  }

//+------------------------------------------------------------------+
//| AQCReplayFileBytes - Size of the framed live-slot payload         |
//+------------------------------------------------------------------+
ulong AQCReplayFileBytes(const int size,const int length)
  {
   return((ulong)size*((ulong)(length+AQC_REPLAY_ROW_HEADER)*sizeof(int)+
                       AQC_REPLAY_META_WIDTH*sizeof(ulong)+2*sizeof(float)));
  }

//+------------------------------------------------------------------+
//| AQCReplayFileMask - Validate only representable vocabulary bits  |
//+------------------------------------------------------------------+
bool AQCReplayFileMask(const uint low,const uint high,const int actions)
  {
   if((low&3)!=0)
      return(false);
   if(actions<32)
      return(high==0&&(low>>actions)==0);
   if(actions<64)
      return((high>>(actions-32))==0);
   return(true);
  }

//+------------------------------------------------------------------+
//| AQCReplayFileRows - Validate a bounded checkpoint block          |
//+------------------------------------------------------------------+
bool AQCReplayFileRows(const int &payload[],const ulong &metadata[],const float &rewards[],
                       const float &priorities[],const int first,const int rows,const int capacity,
                       const int length,const int actions,const int size,const int head,
                       const ulong verified,const ulong next_candidate)
  {
   const int stride=length+AQC_REPLAY_ROW_HEADER;
   for(int row=0;row<rows;++row)
     {
      const int state=row*stride;
      const int meta=row*AQC_REPLAY_META_WIDTH;
      const int slot=first+row;
      const int age=(size==capacity)?(slot+capacity-head)%capacity:slot;
      const ulong generation=verified-(ulong)size+1+(ulong)age;
      const int used=payload[state];
      const int action=payload[state+1];
      const uint low=(uint)payload[state+2];
      const uint high=(uint)payload[state+3];
      const uint next_low=(uint)payload[state+4];
      const uint next_high=(uint)payload[state+5];
      if(metadata[meta]!=generation||metadata[meta+4]<metadata[meta+3]||
         used<1||used>length||action<2||action>=actions||!MathIsValidNumber(rewards[row])||
         !MathIsValidNumber(priorities[row])||priorities[row]<=0.0f||
         !AQCReplayFileMask(low,high,actions)||!AQCReplayFileMask(next_low,next_high,actions)||
         (((action<32)?low:high)&((uint)1<<(action%32)))==0)
         ReturnFalseEx("Invalid replay checkpoint record");

      const ulong candidate_id=metadata[meta+1];
      if(candidate_id>=next_candidate)
         ReturnFalseEx(StringFormat("Replay checkpoint slot %d uses candidate %I64u at or beyond next ID %I64u",
                                   slot,candidate_id,next_candidate));

      if(action==2)
        {
         if(used<2||next_low!=0||next_high!=0)
            ReturnFalseEx("Invalid terminal replay successor");
        }
      else if(used>=length||rewards[row]!=0.0f||(next_low|next_high)==0)
         ReturnFalseEx("Invalid nonterminal replay successor");

      if(payload[state+AQC_REPLAY_ROW_HEADER]!=0)
         ReturnFalseEx("Replay prefix must begin at BOS");
      for(int t=1;t<length;++t)
        {
         const int token=payload[state+AQC_REPLAY_ROW_HEADER+t];
         if((t<used&&(token<=2||token>=actions))||(t>=used&&token!=1))
            ReturnFalseEx("Invalid replay prefix token");
        }
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Save - Serialize live rows in bounded portable blocks            |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Save(const int file)
  {
   if(file==INVALID_HANDLE||!m_bReady||!m_bConsistent||m_bSnapshot||m_iDirty!=0)
      ReturnFalseEx("Replay is outside a checkpoint boundary");

   const int stride=m_iLength+AQC_REPLAY_ROW_HEADER;
   const ulong bytes=AQCReplayFileBytes(m_iSize,m_iLength);
   uint header[8]={AQC_REPLAY_MAGIC,AQC_REPLAY_VERSION,(uint)m_iCapacity,(uint)m_iLength,
                   (uint)m_iActions,(uint)m_iSize,(uint)m_iHead,0};
   if(FileWriteArray(file,header,0,8)!=8||FileWriteLong(file,(long)m_uVerified)!=sizeof(ulong)||
      FileWriteLong(file,(long)bytes)!=sizeof(ulong))
      ReturnFalseEx("Replay checkpoint header write failed");

   int payload[];
   ulong metadata[];
   float rewards[];
   float priorities[];
   if(ArrayResize(payload,AQC_REPLAY_FILE_ROWS*stride)!=AQC_REPLAY_FILE_ROWS*stride||
      ArrayResize(metadata,AQC_REPLAY_FILE_ROWS*AQC_REPLAY_META_WIDTH)!=AQC_REPLAY_FILE_ROWS*AQC_REPLAY_META_WIDTH||
      ArrayResize(rewards,AQC_REPLAY_FILE_ROWS)!=AQC_REPLAY_FILE_ROWS||
      ArrayResize(priorities,AQC_REPLAY_FILE_ROWS)!=AQC_REPLAY_FILE_ROWS)
      ReturnFalseEx("Replay save staging allocation failed");

   for(int first=0;first<m_iSize;first+=AQC_REPLAY_FILE_ROWS)
     {
      const int rows=(int)MathMin(AQC_REPLAY_FILE_ROWS,m_iSize-first);
      const int n=rows*stride;
      const int m=rows*AQC_REPLAY_META_WIDTH;
      const uint base=(uint)first;
      if(!OpenCL.BufferRead(m_iBuffer[AQC_RB_STATE],payload,base*stride*sizeof(int),0,n)||
         !OpenCL.BufferRead(m_iBuffer[AQC_RB_METADATA],metadata,base*AQC_REPLAY_META_WIDTH*sizeof(ulong),0,m)||
         !OpenCL.BufferRead(m_cRewards.GetIndex(),rewards,base*sizeof(float),0,rows)||
         !OpenCL.BufferRead(m_cPriorities.GetIndex(),priorities,base*sizeof(float),0,rows)||
         FileWriteArray(file,payload,0,n)!=n||FileWriteArray(file,metadata,0,m)!=m||
         FileWriteArray(file,rewards,0,rows)!=rows||FileWriteArray(file,priorities,0,rows)!=rows)
         ReturnFalseEx("Replay checkpoint block write failed");
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Load - Read the same bounded blocks in all three load modes       |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Load(const int file,const int capacity,const int length,const int actions,
                          const ulong next_candidate,const bool restore)
  {
   if(file==INVALID_HANDLE||capacity<length||length<2||actions<3||actions>64||
      (long)capacity*((long)length+AQC_REPLAY_ROW_HEADER)>INT_MAX/(int)sizeof(int)||
      (long)capacity*AQC_REPLAY_META_WIDTH>INT_MAX/(int)sizeof(ulong)||
      (restore&&(!OpenCL||m_iCapacity!=capacity||m_iLength!=length||m_iActions!=actions||m_bReady)))
      ReturnFalseEx("Replay checkpoint geometry is incompatible");

   ulong cursor=FileTell(file);
   const ulong total=FileSize(file);
   uint header[8];
   const ulong header_bytes=8*sizeof(uint)+2*sizeof(ulong);
   if(cursor==ULONG_MAX||total==ULONG_MAX||cursor>total||total-cursor<header_bytes||
      FileReadArray(file,header,0,8)!=8||header[0]!=AQC_REPLAY_MAGIC||header[1]!=AQC_REPLAY_VERSION||
      header[2]!=(uint)capacity||header[3]!=(uint)length||header[4]!=(uint)actions||
      header[5]>(uint)capacity||header[6]>=(uint)capacity||header[7]!=0)
      ReturnFalseEx("Invalid replay checkpoint header");

   const ulong verified=(ulong)FileReadLong(file);
   const ulong bytes=(ulong)FileReadLong(file);
   cursor=FileTell(file);
   const int size=(int)header[5];
   const int head=(int)header[6];
   const int expected_size=(verified>=(ulong)capacity)?capacity:(int)verified;
   if(size!=expected_size||head!=(int)(verified%(ulong)capacity)||
      bytes!=AQCReplayFileBytes(size,length)||cursor>total||bytes>total-cursor)
      ReturnFalseEx("Replay checkpoint payload is truncated or inconsistent");

   const int stride=length+AQC_REPLAY_ROW_HEADER;
   int payload[];
   ulong metadata[];
   float rewards[];
   float priorities[];
   if(restore)
     {
      if(ArrayResize(m_iSavedState,size*stride)!=size*stride||
         ArrayResize(m_uSavedMetadata,size*AQC_REPLAY_META_WIDTH)!=size*AQC_REPLAY_META_WIDTH||
         ArrayResize(m_fSavedRewards,size)!=size||ArrayResize(m_fSavedPriorities,size)!=size)
         ReturnFalseEx("Replay restore staging allocation failed");
     }

   for(int first=0;first<size;first+=AQC_REPLAY_FILE_ROWS)
     {
      const int rows=(int)MathMin(AQC_REPLAY_FILE_ROWS,size-first);
      const int n=rows*stride;
      const int m=rows*AQC_REPLAY_META_WIDTH;
      if(FileReadArray(file,payload,0,n)!=n||FileReadArray(file,metadata,0,m)!=m||
         FileReadArray(file,rewards,0,rows)!=rows||FileReadArray(file,priorities,0,rows)!=rows||
         !AQCReplayFileRows(payload,metadata,rewards,priorities,first,rows,capacity,
                            length,actions,size,head,verified,next_candidate))
         ReturnFalseEx("Invalid replay checkpoint block");

      if(restore&&(ArrayCopy(m_iSavedState,payload,first*stride,0,n)!=n||
                   ArrayCopy(m_uSavedMetadata,metadata,first*AQC_REPLAY_META_WIDTH,0,m)!=m||
                   ArrayCopy(m_fSavedRewards,rewards,first,0,rows)!=rows||
                   ArrayCopy(m_fSavedPriorities,priorities,first,0,rows)!=rows))
         ReturnFalseEx("Replay restore staging copy failed");
     }

   if(FileTell(file)!=cursor+bytes)
      ReturnFalseEx("Replay checkpoint framing mismatch");

   m_iSize=size;
   m_iHead=head;
   m_uVerified=verified;
   m_bRestoring=restore;
   return(true);
  }

//+------------------------------------------------------------------+
//| Restore - Upload TRAIN payload after Build then rebuild the CDF   |
//+------------------------------------------------------------------+
bool CAlphaQCMReplay::Restore(void)
  {
   if(!OpenCL||!m_bRestoring||m_bReady)
      ReturnFalseEx("Replay restore state is invalid");

   if(m_iSize>0)
     {
      const int state=m_iSize*(m_iLength+AQC_REPLAY_ROW_HEADER);
      const int metadata=m_iSize*AQC_REPLAY_META_WIDTH;
      if(!OpenCL.BufferWrite(m_iBuffer[AQC_RB_STATE],m_iSavedState,0,0,state)||
         !OpenCL.BufferWrite(m_iBuffer[AQC_RB_METADATA],m_uSavedMetadata,0,0,metadata)||
         !OpenCL.BufferWrite(m_cRewards.GetIndex(),m_fSavedRewards,0,0,m_iSize)||
         !OpenCL.BufferWrite(m_cPriorities.GetIndex(),m_fSavedPriorities,0,0,m_iSize))
         ReturnFalseEx("Replay restore upload failed");
     }

   ClearRestore();
   return(Initialize());
  }

#endif // NEURONET_ALPHAQCM_REPLAY_MQH
