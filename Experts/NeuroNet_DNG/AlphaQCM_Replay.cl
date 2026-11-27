//+------------------------------------------------------------------+
//|                                               AlphaQCM_Replay.cl |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+

// Philox4x32-10 from Salmon et al., Parallel Random Numbers: As Easy
// as 1, 2, 3 (SC11), doi:10.1145/2063384.2063405. The two counter
// words at each end encode a little-endian uint64 logical ID and ordinal.
#define AQC_STREAM_Q_INIT      0UL
#define AQC_STREAM_IQN_INIT    1UL
#define AQC_STREAM_REPLAY      2UL
#define AQC_STREAM_TAU_CURRENT 3UL
#define AQC_STREAM_TAU_TARGET  4UL
#define AQC_STREAM_WARMUP      5UL
#define AQC_STREAM_EPSILON     6UL
#define AQC_STREAM_LEGAL       7UL

inline uint AQCPhiloxWord(const ulong seed,const ulong stream,const ulong logical_id,const ulong ordinal)
  {
   const ulong stream_key=seed+stream;
   uint2 key=(uint2)((uint)stream_key,(uint)(stream_key>>32));
   uint4 counter=(uint4)((uint)logical_id,(uint)(logical_id>>32),(uint)ordinal,(uint)(ordinal>>32));

   for(int round=0;round<10;++round)
     {
      const uint left=0xd2511f53U;
      const uint right=0xcd9e8d57U;
      counter=(uint4)(mul_hi(right,counter.z)^counter.y^key.x,right*counter.z,
                      mul_hi(left,counter.x)^counter.w^key.y,left*counter.x);
      key+=(uint2)(0x9e3779b9U,0xbb67ae85U);
     }

   return(counter.x);
  }

inline float AQCPhiloxUnit(const uint word)
  {
   return(((float)(word>>9)+0.5f)*0x1p-23f);
  }

#define AQC_R_OK           0
#define AQC_R_NUMERIC      1
#define AQC_R_PAYLOAD      2
#define AQC_R_STALE        3
#define AQC_R_META_WIDTH   6
#define AQC_R_STATE_HEADER 6
#define AQC_R_BOS          0
#define AQC_R_PAD          1
#define AQC_R_STOP         2

// A CDF mass uses a rounded FP32 sum and a separate FP32 residual. This
// preserves small intervals next to a large prefix on devices without FP64.
inline float2 AQCReplayTwoSum(const float a,const float b)
  {
   const float sum=a+b;
   const float b_virtual=sum-a;
   return((float2)(sum,(a-(sum-b_virtual))+(b-b_virtual)));
  }

// Fast2Sum with the larger component first.
inline float2 AQCReplayNormalize(const float high,const float low)
  {
   const float sum=high+low;
   return((float2)(sum,low-(sum-high)));
  }

// AccurateDWPlusDW (Muller and Rideau, doi:10.1145/3484514).
inline float2 AQCReplayAddMass(const float2 left,const float2 right)
  {
   const float2 high=AQCReplayTwoSum(left.x,right.x);
   const float2 low=AQCReplayTwoSum(left.y,right.y);
   const float2 partial=AQCReplayNormalize(high.x,high.y+low.x);
   return(AQCReplayNormalize(partial.x,partial.y+low.y));
  }

inline bool AQCReplayMassLess(const float2 left,const float2 right)
  {
   return(left.x<right.x||(left.x==right.x&&left.y<right.y));
  }

// Nodes store sum high, minimum positive mass, maximum priority and sum low.
// Invalid leaves propagate a nonfinite sum through the tree.
inline float4 AQCReplayLeaf(const float priority)
  {
   const float mass=sqrt(priority);
   const float sum=(isfinite(priority)&&isfinite(mass)&&mass>0.0f)?mass:NAN;
   return((float4)(sum,mass,priority,0.0f));
  }

inline bool AQCReplayOwner(__global const int *changes,const int position,const int level)
  {
   const int slot=changes[position];
   if(slot<0)
      return(false);

   const int node=slot>>level;
   for(int i=0;i<position;++i)
      if(changes[i]>=0&&(changes[i]>>level)==node)
         return(false);

   return(true);
  }

__kernel void AQReplayTreeLeaves(__global const float *priorities,__global float4 *tree,
                                 const int size,const int leaves)
  {
   const int slot=get_global_id(0);
   tree[leaves+slot]=(slot<size)?AQCReplayLeaf(priorities[slot]):(float4)(0.0f,FLT_MAX,0.0f,0.0f);
  }

inline void AQCReplayTreeNode(__global float4 *tree,const int node)
  {
   const float4 left=tree[2*node];
   const float4 right=tree[2*node+1];
   const float2 sum=AQCReplayAddMass(left.xw,right.xw);
   tree[node]=(float4)(sum.x,fmin(left.y,right.y),fmax(left.z,right.z),sum.y);
  }

__kernel void AQReplayTreeLevel(__global float4 *tree,__global const int *changes,
                                const int leaves,const int level,const int count)
  {
   if(count==0)
     {
      AQCReplayTreeNode(tree,(leaves>>level)+(int)get_global_id(0));
      return;
     }

//--- A sparse transaction uses one work-group for the dependency chain.
   const int lane=get_local_id(0);
   const int lanes=get_local_size(0);
   for(int height=1;height<=level;++height)
     {
      for(int i=lane;i<count;i+=lanes)
         if(AQCReplayOwner(changes,i,height))
            AQCReplayTreeNode(tree,(leaves+changes[i])>>height);

      barrier(CLK_GLOBAL_MEM_FENCE);
     }
  }

inline uint AQCReplayPackMask(__global const float *mask,const int begin,const int end)
  {
   uint bits=0;
   for(int a=begin;a<end;++a)
      if(mask[a]==1.0f)
         bits|=1U<<(a-begin);
   return(bits);
  }

__kernel void AQReplayStageEpisode(__global const float *generated,__global const float *reward,
                                   __global const float4 *live,__global float4 *staged,
                                   __global int *changes,__global int *status,
                                   const int row,const int steps,const int reward_index,
                                   const int head,const int capacity,const int leaves,const int size,
                                   const int length,const int actions)
  {
   const int step=get_global_id(0);
   const int stride=4+actions;
   const int slot=(head+step)%capacity;
   __global const float *formula=generated+row*(1+length*stride);
   __global const float *decision=formula+1+step*stride;
   const float a=decision[0];
   changes[step]=slot;
   status[step]=AQC_R_PAYLOAD;

   if(formula[0]!=(float)steps||!isfinite(a)||a<(float)AQC_R_STOP||a>=(float)actions)
      return;

   const int action=(int)a;
   if(a!=(float)action||(action==AQC_R_STOP)!=(step==steps-1)||decision[4+action]!=1.0f||
      decision[4+AQC_R_BOS]!=0.0f||decision[4+AQC_R_PAD]!=0.0f)
      return;

   for(int k=AQC_R_STOP;k<actions;++k)
      if(decision[4+k]!=0.0f&&decision[4+k]!=1.0f)
         return;

   const float priority=(size==0)?1.0f:live[1].z;
   const float4 leaf=AQCReplayLeaf(priority);
   if(!isfinite(leaf.x)||!isfinite(reward[reward_index]))
     {
      status[step]=AQC_R_NUMERIC;
      return;
     }

   staged[leaves+slot]=leaf;
   status[step]=AQC_R_OK;
  }

__kernel void AQReplayPublishEpisode(__global const float *generated,__global const float *reward,
                                     __global const float4 *staged,__global int *payload,
                                     __global ulong *metadata,__global float *rewards,
                                     __global float *priorities,__global const int *changes,
                                     const int row,const int steps,const int reward_index,
                                     const int leaves,const int length,const int actions,
                                     const ulong verified,const ulong candidate,const ulong window_revision,
                                     const ulong prepared_before,const ulong prepared_after,const ulong active)
  {
   const int step=get_global_id(0);
   const int column=get_global_id(1);
   const int stride=4+actions;
   const int slot=changes[step];
   const bool terminal=step==steps-1;
   __global const float *formula=generated+row*(1+length*stride)+1;
   __global const float *decision=formula+step*stride;
   __global int *state=payload+slot*(length+AQC_R_STATE_HEADER);

   if(column==0)
     {
      state[0]=step+1;
      rewards[slot]=terminal?reward[reward_index]:0.0f;
      priorities[slot]=staged[leaves+slot].z;
      __global ulong *meta=metadata+slot*AQC_R_META_WIDTH;
      meta[0]=verified+(ulong)step+1UL;
      meta[1]=candidate;
      meta[2]=window_revision;
      meta[3]=prepared_before;
      meta[4]=prepared_after;
      meta[5]=active;
     }
   else if(column==1)
      state[1]=(int)decision[0];
   else if(column<AQC_R_STATE_HEADER)
     {
      const bool successor=column>=4;
      const int start=(column%2)*32;
      const int end=min(start+32,actions);
      const uint bits=(terminal&&successor)?0U:AQCReplayPackMask(decision+4+(successor?stride:0),start,end);
      state[column]=as_int(bits);
     }
   else
     {
      const int token=column-AQC_R_STATE_HEADER;
      state[column]=(token==0)?AQC_R_BOS:((token<=step)?(int)formula[(token-1)*stride]:AQC_R_PAD);
     }
  }

// Prefixes are truncated sums using the fixed tree association over ascending
// physical slot indices. Including the last slot reproduces tree[1].xw.
inline float2 AQCReplayPrefix(__global const float4 *tree,const int leaves,const int end)
  {
   if(end==0)
      return((float2)(0.0f));

   int node=leaves+end-1;
   float2 sum=tree[node].xw;
   while(node>1)
     {
      if(node&1)
         sum=AQCReplayAddMass(tree[node-1].xw,sum);
      node>>=1;
     }

   return(sum);
  }

__kernel void AQReplaySample(__global const float4 *tree,__global const ulong *metadata,
                             __global int *selected,__global ulong *generations,
                             __global float *weights,__global int *status,
                             const int leaves,const int size,const float beta,
                             const ulong seed,const ulong logical_id)
  {
   const int row=get_global_id(0);
   const float2 total=tree[1].xw;
   const float minimum=tree[1].y;
   selected[row]=-1;
   status[row]=AQC_R_NUMERIC;
   if(!all(isfinite(total))||total.x<=0.0f||!isfinite(minimum)||minimum<=0.0f)
      return;

   const float uniform=AQCPhiloxUnit(AQCPhiloxWord(seed,AQC_STREAM_REPLAY,logical_id,(ulong)row));
   const float high=uniform*total.x;
   const float low=fma(uniform,total.y,fma(uniform,total.x,-high));
   const float2 product=AQCReplayNormalize(high,low);
   const float2 x=AQCReplayMassLess(product,total)?product:
                  (float2)(total.x,nextafter(total.y,-INFINITY));
   int lower=0;
   int upper=size;
   while(upper-lower>1)
     {
      const int middle=lower+(upper-lower)/2;
      if(AQCReplayMassLess(x,AQCReplayPrefix(tree,leaves,middle)))
         upper=middle;
      else
         lower=middle;
     }

   const float mass=tree[leaves+lower].x;
   const float weight=pow(minimum/mass,beta);
   if(AQCReplayMassLess(x,AQCReplayPrefix(tree,leaves,lower))||
      !AQCReplayMassLess(x,AQCReplayPrefix(tree,leaves,lower+1))||
      mass<=0.0f||!isfinite(weight)||weight<=0.0f||weight>1.0f)
      return;

   selected[row]=lower;
   generations[row]=metadata[lower*AQC_R_META_WIDTH];
   weights[row]=weight;
   status[row]=AQC_R_OK;
  }

__kernel void AQReplayGather(__global const int *payload,__global const float *rewards,
                             __global const int *selected,__global int *tokens,
                             __global int *lengths,__global int *next_tokens,
                             __global int *next_lengths,__global int *next_legal,
                             __global int *actions_out,__global float *rewards_out,
                             __global int *done,const int length,const int actions)
  {
   const int row=get_global_id(0);
   const int column=get_global_id(1);
   const int slot=selected[row];
   __global const int *state=payload+slot*(length+AQC_R_STATE_HEADER);
   const int n=state[0];
   const int action=state[1];
   const bool terminal=action==AQC_R_STOP;

   if(column==0)
     {
      lengths[row]=n;
      next_lengths[row]=terminal?0:n+1;
      actions_out[row]=action;
      rewards_out[row]=rewards[slot];
      done[row]=terminal?1:0;
     }

   if(column<length)
     {
      const int token=state[AQC_R_STATE_HEADER+column];
      tokens[row*length+column]=token;
      next_tokens[row*length+column]=terminal?AQC_R_PAD:((column==n)?action:token);
     }

   if(column<actions)
     {
      int legal=0;
      if(!terminal)
         legal=(int)((as_uint(state[4+column/32])>>(column%32))&1U);
      next_legal[row*actions+column]=legal;
     }
  }

__kernel void AQReplayTau(__global float *current,__global float *target,
                          const int n,const int next_n,const ulong seed,const ulong logical_id)
  {
   const ulong ordinal=get_global_id(0);
   const int quantile=get_global_id(1);
   if(quantile<n)
     {
      const ulong i=ordinal*(ulong)n+(ulong)quantile;
      current[i]=AQCPhiloxUnit(AQCPhiloxWord(seed,AQC_STREAM_TAU_CURRENT,logical_id,i));
     }

   if(quantile<next_n)
     {
      const ulong i=ordinal*(ulong)next_n+(ulong)quantile;
      target[i]=AQCPhiloxUnit(AQCPhiloxWord(seed,AQC_STREAM_TAU_TARGET,logical_id,i));
     }
  }

__kernel void AQReplayStagePriorities(__global const int *selected,__global const ulong *generations,
                                      __global const ulong *metadata,__global const float *loss,
                                      __global float4 *staged,__global int *changes,__global int *status,
                                      const int count,const int size,const int leaves,const float epsilon)
  {
   const int row=get_global_id(0);
   const int slot=selected[row];
   changes[row]=-1;
   status[row]=AQC_R_STALE;
   if(slot<0||slot>=size||generations[row]!=metadata[slot*AQC_R_META_WIDTH])
      return;

   status[row]=AQC_R_NUMERIC;
   if(!isfinite(loss[row])||loss[row]<0.0f)
      return;

   status[row]=AQC_R_OK;
//--- The earliest occurrence owns the slot and reduces all duplicate losses.
   for(int b=0;b<row;++b)
      if(selected[b]==slot)
         return;

   float maximum=loss[row];
   for(int b=row+1;b<count;++b)
      if(selected[b]==slot)
         maximum=fmax(maximum,loss[b]);

   const float4 leaf=AQCReplayLeaf(epsilon+maximum);
   if(!isfinite(leaf.x))
     {
      status[row]=AQC_R_NUMERIC;
      return;
     }

   changes[row]=slot;
   staged[leaves+slot]=leaf;
  }

__kernel void AQReplayCommitPriorities(__global const int *changes,__global const float4 *staged,
                                       __global float *priorities,const int leaves)
  {
   const int slot=changes[get_global_id(0)];
   if(slot>=0)
      priorities[slot]=staged[leaves+slot].z;
  }

__kernel void AQReplayMirrorTree(__global const float4 *source,__global float4 *destination,
                                 __global const int *changes,const int leaves,const int sparse)
  {
   const int position=get_global_id(0);
   int node=position+1;
   if(sparse)
     {
      const int level=get_global_id(1);
      if(!AQCReplayOwner(changes,position,level))
         return;
      node=(leaves+changes[position])>>level;
     }

   destination[node]=source[node];
  }

__kernel void AQReplayCheckIntervals(__global const float4 *tree,__global int *status,const int leaves)
  {
   const int slot=get_global_id(0);
   const float2 lower=AQCReplayPrefix(tree,leaves,slot);
   const float2 upper=AQCReplayPrefix(tree,leaves,slot+1);
   status[slot]=AQCReplayMassLess(lower,upper)?AQC_R_OK:AQC_R_NUMERIC;
  }

__kernel void AQReplayStatus(__global const int *entries,__global int *result,const int count)
  {
   __local int reduction[128];
   const int lane=get_local_id(0);
   const int lanes=get_local_size(0);
   const int first=(int)get_group_id(0)*lanes*4;
   int error=AQC_R_OK;
   for(int i=first+lane;i<min(first+lanes*4,count);i+=lanes)
      error=max(error,entries[i]);

   reduction[lane]=error;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int width=lanes/2;width>0;width/=2)
     {
      if(lane<width)
         reduction[lane]=max(reduction[lane],reduction[lane+width]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }

   if(lane==0)
      result[get_group_id(0)]=reduction[0];
  }
