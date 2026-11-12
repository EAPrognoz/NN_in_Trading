//+------------------------------------------------------------------+
//| AlphaGen transactional pools. Appended after AlphaGen.cl.        |
//+------------------------------------------------------------------+
inline void AGPoolEvaluate(__global const float *market,
                           __global const float *tokens, const int length,
                           __global const float *cfg, __global float *stack,
                           __global float *validity, __global float *result,
                           __global float *status)
  {
// One work item owns the complete causal recurrence, including rolling
// nodes. Parallel work items must not read nodes still being produced.

   int w=(int)cfg[AG_CFG_W],depth=0;
   int capacity=(int)cfg[AG_CFG_MAX_TOKENS]+1;
   __global int *types=(__global int *)validity;
   __global int *lookback=types+capacity;
   __global int *dependent=lookback+capacity;
   int syntax=length>=2 && length<capacity;
   for(int k=0;k<length && syntax;k++)
     {
      float raw_token=tokens[k];
      if(!isfinite(raw_token) || raw_token<0.0f || raw_token>=(float)AG_VOCABULARY ||
         raw_token!=floor(raw_token))
        {
         syntax=0;
         break;
        }
      int token=(int)raw_token;
      if(k==length-1)
         syntax=token==AG_STOP && depth==1 && types[0]==0 && dependent[0];
      else
        {
         depth=AGApplyToken(token,cfg,types,lookback,dependent,depth);
         if(depth<0)
            syntax=0;
        }
     }
   if(!syntax)
     {
      for(int i=0;i<w;i++)
        {
         result[i]=0.0f;
         status[i]=AG_TECHNICAL;
        }
      return;
     }
   depth=0;
   for(int k=0;k<length;k++)
     {
      int token=(int)tokens[k];
      if(token==AG_STOP)
         break;
      if(token>=AG_OPEN && token<AG_ABS)
        {
         for(int i=0;i<w;i++)
           {
            float x=token<=AG_VOLUME ? market[(token-AG_OPEN)*w+i] :
                    token<AG_INTERVAL_FIRST ? cfg[AG_CFG_CONSTANTS+token-AG_CONSTANT_FIRST] :
                    cfg[AG_CFG_INTERVALS+token-AG_INTERVAL_FIRST];
            stack[depth*w+i]=x;
            validity[depth*w+i]=isfinite(x)?AG_OK:AG_INVALID;
           }
         depth++;
         continue;
        }
      if(token>=AG_ABS && token<=AG_LOG)
        {
         for(int i=0;i<w;i++)
           {
            int p=(depth-1)*w+i;
            float x=stack[p];
            if(validity[p]!=AG_OK)
               continue;
            if(token==AG_LOG && !(x>0.0f))
               validity[p]=AG_INVALID;
            else
               stack[p]=token==AG_ABS?fabs(x):token==AG_SIGN?sign(x):log(x);
            if(!isfinite(stack[p]))
               validity[p]=AG_INVALID;
           }
         continue;
        }
      if(token>=AG_ADD && token<=AG_DIV)
        {
         for(int i=0;i<w;i++)
           {
            int l=(depth-2)*w+i,r=(depth-1)*w+i;
            int s=max((int)validity[l],(int)validity[r]);
            float a=stack[l],b=stack[r],x=0.0f;
            if(s==AG_OK)
              {
               if(token==AG_DIV && b==0.0f)
                  s=AG_INVALID;
               else
                  x=token==AG_ADD?a+b:token==AG_SUB?a-b:token==AG_MUL?a*b:a/b;
               if(!isfinite(x))
                  s=AG_INVALID;
              }
            stack[l]=x;
            validity[l]=(float)s;
           }
         depth--;
         continue;
        }
      int pair=(token==AG_CORR || token==AG_COV);
      int slot=depth-(pair?3:2),n=(int)stack[(depth-1)*w];
      // Descending evaluation preserves the original operand rows.
      for(int i=w-1;i>=0;i--)
        {
         int begin=i-n+((token==AG_REF || token==AG_DELTA)?0:1);
         int p=slot*w+i,s=AG_OK;
         float x=0.0f;
         if(begin<0)
           {
            validity[p]=AG_SKIP;
            stack[p]=x;
            continue;
           }
         if(token==AG_REF || token==AG_DELTA)
           {
            s=(int)validity[slot*w+begin];
            if(token==AG_DELTA)
               s=max(s,(int)validity[p]);
            x=token==AG_REF?stack[slot*w+begin]:stack[p]-stack[slot*w+begin];
           }
         else
           {
            float a=0.0f,b=0.0f,mx=0.0f,my=0.0f,xx=0.0f,yy=0.0f,xy=0.0f;
            float lo=stack[slot*w+begin],hi=lo;
            for(int j=begin;j<=i;j++)
              {
               s=max(s,(int)validity[slot*w+j]);
               a=fmax(a,fabs(stack[slot*w+j]));
               if(pair)
                 {
                  s=max(s,(int)validity[(slot+1)*w+j]);
                  b=fmax(b,fabs(stack[(slot+1)*w+j]));
                 }
               lo=fmin(lo,stack[slot*w+j]);
               hi=fmax(hi,stack[slot*w+j]);
              }
            if(token==AG_MIN)
               x=lo;
            else
               if(token==AG_MAX)
                  x=hi;
               else
                 {
                  if(a==0.0f)
                     a=1.0f;
                  if(b==0.0f)
                     b=1.0f;
                  for(int j=begin;j<=i;j++)
                    {
                     mx+=stack[slot*w+j]/a/(float)n;
                     if(pair)
                        my+=stack[(slot+1)*w+j]/b/(float)n;
                    }
                  for(int j=begin;j<=i;j++)
                    {
                     float dx=stack[slot*w+j]/a-mx;
                     float dy=pair?stack[(slot+1)*w+j]/b-my:0.0f;
                     xx+=dx*dx/(float)n;
                     yy+=dy*dy/(float)n;
                     xy+=dx*dy/(float)n;
                    }
                  if(token==AG_MEAN)
                     x=mx*a;
                  else
                     if(token==AG_STD)
                        x=sqrt(xx)*a;
                     else
                        if(token==AG_COV)
                           x=(xy*a)*b;
                        else
                           if(xx>0.0f && yy>0.0f)
                              x=xy/sqrt(xx)/sqrt(yy);
                           else
                              s=max(s,AG_SKIP);
                 }
           }
         if(s==AG_OK && !isfinite(x))
            s=AG_INVALID;
         stack[p]=x;
         validity[p]=(float)s;
        }
      depth-=pair?2:1;
     }
   for(int i=0;i<w;i++)
     {
      result[i]=stack[i];
      status[i]=validity[i];
     }
  }
//+------------------------------------------------------------------+
//| One independent formula per workgroup; lanes cover historical    |
//| rows. Only the syntax flag is shared in local memory. No atomics |
//| or intergroup writes.                                            |
//+------------------------------------------------------------------+
__kernel void AGPoolEvaluateBatch(__global const float *market,
                                  __global const float *candidates,const int batch,
                                  __global const float *cfg,__global float *workspace,
                                  __global float *evaluated)
  {
   int candidate=(int)get_group_id(0);
   if(candidate>=batch)
      return; // Uniform for the whole group, before barriers.
   int w=(int)cfg[AG_CFG_W],t=(int)cfg[AG_CFG_MAX_TOKENS];
   int stride=4+AG_VOCABULARY;
   size_t work_stride=(size_t)w*t+max(w*t,3*(t+1));
   __global const float *packed=candidates+(size_t)candidate*(1+(size_t)t*stride);
   __global float *stack=workspace+(size_t)candidate*work_stride;
   __global float *result=evaluated+(size_t)candidate*2*w;
   __local int syntax;
   AGFormulaEvaluate(market,packed+1,(int)packed[0],cfg,stack,stack+w*t,
                     result,result+w,stride,&syntax);
  }
//+------------------------------------------------------------------+
//| Stable affine FP32 fit. Small raw amplitude is not degeneracy.   |
//+------------------------------------------------------------------+
inline void AGPoolStandardize(__global const float *input,
                              __global const float *validity,
                              const int offset,const int count,
                              __global const float *cfg,
                              __global float *normalized,
                              __global float *statistics,const int stats_offset)
  {

   __global float *s=statistics+stats_offset;
   s[AG_STAT_STATUS]=AG_SKIP;
   s[AG_STAT_VALID]=0.0f;
   if(count<1)
      return;
   float a=0.0f;
   for(int i=0;i<count;i++)
     {
      if(validity[offset+i]!=AG_OK)
        {
         s[AG_STAT_STATUS]=validity[offset+i];
         return;
        }
      if(!isfinite(input[offset+i]))
        {
         s[AG_STAT_STATUS]=AG_INVALID;
         return;
        }
      a=fmax(a,fabs(input[offset+i]));
     }
   if(!(a>0.0f))
      return;
   float origin=input[offset]/a,b=0.0f;
   for(int i=0;i<count;i++)
      b=fmax(b,fabs(input[offset+i]/a-origin));
   if(!(b>0.0f))
      return;
   float mean=0.0f;
   for(int i=0;i<count;i++)
      mean+=(input[offset+i]/a-origin)/b/(float)count;
   float variance=0.0f;
   for(int i=0;i<count;i++)
     {
      float d=(input[offset+i]/a-origin)/b-mean;
      variance+=d*d/(float)count;
     }
   if(!isfinite(variance) || variance<=cfg[AG_CFG_EPS_VAR])
      return;
   float sd=sqrt(variance);
   for(int i=0;i<count;i++)
      normalized[i]=((input[offset+i]/a-origin)/b-mean)/sd;
   s[AG_STAT_SCALE]=a;
   s[AG_STAT_ORIGIN]=origin;
   s[AG_STAT_DIFF_SCALE]=b;
   s[AG_STAT_MEAN]=mean;
   s[AG_STAT_STD]=sd;
   s[AG_STAT_STATUS]=AG_OK;
   s[AG_STAT_VALID]=(float)count;
  }
//+------------------------------------------------------------------+
//| Ridge fit on standardized columns/target using Cholesky FP32.    |
//| scratch = (K+1)^2 + (K+1); no CPU matrix transfer is required.   |
//+------------------------------------------------------------------+
inline void AGPoolRidge(__global const float *z,__global const float *target,
                        const int count,const int columns,
                        __global const float *cfg,__global float *scratch,
                        __global float *weights,__global float *metrics)
  {

   metrics[0]=AG_TECHNICAL;
   metrics[1]=0.0f;
   metrics[2]=0.0f;
   if(count<1 || columns<1)
      return;
   __global float *rhs=scratch+columns*columns;
   for(int i=0;i<columns;i++)
     {
      float c=0.0f;
      for(int t=0;t<count;t++)
         c+=z[i*count+t]*target[t]/(float)count;
      rhs[i]=c;
      for(int j=0;j<=i;j++)
        {
         float r=0.0f;
         for(int t=0;t<count;t++)
            r+=z[i*count+t]*z[j*count+t]/(float)count;
         if(i==j)
            r+=cfg[AG_CFG_RIDGE];
         for(int k=0;k<j;k++)
            r-=scratch[i*columns+k]*scratch[j*columns+k];
         if(i==j)
           {
            if(!(r>0.0f) || !isfinite(r))
               return;
            scratch[i*columns+j]=sqrt(r);
           }
         else
            scratch[i*columns+j]=r/scratch[j*columns+j];
        }
     }
   for(int i=0;i<columns;i++)
     {
      float x=rhs[i];
      for(int j=0;j<i;j++)
         x-=scratch[i*columns+j]*weights[j];
      weights[i]=x/scratch[i*columns+i];
     }
   for(int i=columns-1;i>=0;i--)
     {
      float x=weights[i];
      for(int j=i+1;j<columns;j++)
         x-=scratch[j*columns+i]*weights[j];
      weights[i]=x/scratch[i*columns+i];
      if(!isfinite(weights[i]))
         return;
     }
   float mean=0.0f;
   for(int t=0;t<count;t++)
     {
      float a=0.0f;
      for(int j=0;j<columns;j++)
         a+=weights[j]*z[j*count+t];
      mean+=a/(float)count;
     }
   float covariance=0.0f,variance=0.0f;
   for(int t=0;t<count;t++)
     {
      float a=0.0f;
      for(int j=0;j<columns;j++)
         a+=weights[j]*z[j*count+t];
      a-=mean;
      covariance+=a*target[t]/(float)count;
      variance+=a*a/(float)count;
     }
   float ic=variance>cfg[AG_CFG_EPS_VAR]?covariance/sqrt(variance):0.0f;
   if(!isfinite(ic))
      return;
   metrics[0]=AG_OK;
   metrics[1]=ic;
   metrics[2]=variance>cfg[AG_CFG_EPS_VAR]?1.0f:0.0f;
  }
//+------------------------------------------------------------------+
//| Duration-aware GAE, with immutable value targets across passes.  |
//| Banks: header [count,next_id limbs(4),version limbs(4),ready],   |
//| then records [length,id limbs(4),tokens[max],                    |
//|                 affine statistics[7],weight].                    |
//+------------------------------------------------------------------+
inline int AGPoolStride(__global const float *cfg) {return 13+(int)cfg[AG_CFG_MAX_TOKENS];}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline int AGPoolBankSize(__global const float *cfg) {return 10+((int)cfg[AG_CFG_POOL]+1)*AGPoolStride(cfg);}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline int AGPoolRingSize(__global const float *cfg) {return 11+(int)cfg[AG_CFG_NORM_WINDOW]+(int)cfg[AG_CFG_CONTEXT];}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline int AGPoolIncrement(__global float *id)
  {
   for(int i=0;i<4;i++)
     {
      if(id[i]<65535.0f)
        {
         id[i]+=1.0f;
         return 1;
        }
      id[i]=0.0f;
     }
   return 0;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline int AGPoolIDGreater(__global const float *a,__global const float *b)
  {for(int i=3;i>=0;i--) {if(a[i]!=b[i]) return a[i]>b[i];} return 0;}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline void AGPoolCopyValues(__global const float *a,__global float *b,int n)
  {for(int i=0;i<n;i++) b[i]=a[i];}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline float AGPoolAffine(float x,__global const float *s)
  {return ((x/s[0]-s[1])/s[2]-s[3])/s[4];}
// Scratch layout: stack W*T, validity max(W*T,3*(T+1)), result W,
// status W, target W, target-valid W, standardized-target N, target-stat 7,
// standardized columns (K+1)*N, solve (K+1)*(K+2), weights K+1, metrics 3.
inline int AGPoolFitCached(__global const float *market,__global const float *cfg,
                           __global float *bank,__global float *work,
                           __global const float *candidates,const int batch,__global const float *evaluated)
  {
   int w=(int)cfg[AG_CFG_W],t=(int)cfg[AG_CFG_MAX_TOKENS];
   int n=(int)cfg[AG_CFG_IC_WINDOW],h=(int)cfg[AG_CFG_HORIZON];
   int cap=(int)cfg[AG_CFG_POOL]+1,stride=AGPoolStride(cfg),offset=w-h-n;
   __global float *stack=work,*valid=stack+w*t;
   __global float *raw=valid+max(w*t,3*(t+1)),*status=raw+w;
   __global float *target=status+w,*tv=target+w,*y=tv+w,*yst=y+n;
   __global float *z=yst+7,*solve=z+cap*n,*weights=solve+cap*(cap+1),*metric=weights+cap;
   int count=(int)bank[0];
   metric[0]=AG_SKIP;
   metric[1]=0.0f;
   metric[2]=0.0f;
   for(int i=0;i<w;i++)
     {
      target[i]=0.0f;
      tv[i]=AG_SKIP;
     }
   for(int i=offset;i<w-h;i++)
     {
      float p=market[3*w+i],q=market[3*w+i+h];
      if(!(p>0.0f) || !(q>0.0f) || !isfinite(p) || !isfinite(q))
         return AG_TECHNICAL;
      target[i]=q/p-1.0f;
      tv[i]=isfinite(target[i])?AG_OK:AG_TECHNICAL;
     }
   AGPoolStandardize(target,tv,offset,n,cfg,y,yst,0);
   if(yst[5]!=AG_OK)
      return AG_SKIP;
   if(count==0)
     {
      metric[0]=AG_OK;
      return AG_OK;
     }
   for(int j=0;j<count;j++)
     {
      __global float *record=bank+10+j*stride;
      // Match by tokens, never pool position: earlier commits can evict/reorder records.
      int cached=-1,token_stride=4+AG_VOCABULARY;
      for(int c=0;c<batch && cached<0;c++)
        {
         __global const float *packed=candidates+(size_t)c*(1+(size_t)t*token_stride);
         int same=record[0]==packed[0];
         for(int k=0;k<(int)record[0] && same;k++)
            same=record[5+k]==packed[1+k*token_stride];
         if(same)
            cached=c;
        }
      if(cached>=0)
        {
         __global const float *values=evaluated+(size_t)cached*2*w;
         for(int i=0;i<w;i++)
           {
            raw[i]=values[i];
            status[i]=values[w+i];
           }
        }
      else
         AGPoolEvaluate(market,record+5,(int)record[0],cfg,stack,valid,raw,status);
      AGPoolStandardize(raw,status,offset,n,cfg,z+j*n,record+5+t,0);
      if(record[10+t]!=AG_OK)
         return (int)record[10+t];
     }
   AGPoolRidge(z,y,n,count,cfg,solve,weights,metric);
   if(metric[0]!=AG_OK)
      return AG_TECHNICAL;
// Re-standardize the aggregate before deciding degeneracy. Tiny signed
// ridge weights must not make a distinguishable aggregate untradable.
   for(int i=0;i<n;i++)
     {
      float a=0.0f;
      for(int j=0;j<count;j++)
         a+=weights[j]*z[j*n+i];
      raw[i]=a;
      status[i]=AG_OK;
     }
   AGPoolStandardize(raw,status,0,n,cfg,target,yst,0);
   metric[1]=0.0f;
   metric[2]=yst[5]==AG_OK?1.0f:0.0f;
   if(metric[2]>0.0f)
      for(int i=0;i<n;i++)
         metric[1]+=target[i]*y[i]/(float)n;
   if(!isfinite(metric[1]))
      return AG_TECHNICAL;
   for(int j=0;j<count;j++)
      bank[10+j*stride+12+t]=weights[j];
   bank[9]=1.0f;
   return AG_OK;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
inline int AGPoolFit(__global const float *market,__global const float *cfg,
                     __global float *bank,__global float *work)
  {return AGPoolFitCached(market,cfg,bank,work,work,0,work);}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AGPoolCandidateCached(__global const float *market,__global const float *tokens,
                           const int length,const int token_stride,__global const float *cfg,
                           __global float *banks,__global float *work,__global float *control,
                           __global const float *candidates,const int batch,__global const float *evaluated)
  {
   int size=AGPoolBankSize(cfg),stride=AGPoolStride(cfg);
   int t=(int)cfg[AG_CFG_MAX_TOKENS],cap=(int)cfg[AG_CFG_POOL]+1;
   int w=(int)cfg[AG_CFG_W],n=(int)cfg[AG_CFG_IC_WINDOW];
   int mi=w*t+max(w*t,3*(t+1))+4*w+n+7+cap*n+cap*(cap+1)+cap;
   __global float *prepared=banks,*scratch=banks+2*size;
   control[AG_POOL_PREPARED_COUNT]=prepared[0];
   control[AG_POOL_ACTIVE_COUNT]=banks[size];
   control[AG_POOL_STATUS]=AG_SKIP;
   control[AG_POOL_REWARD_READY]=0.0f;
   control[AG_POOL_PUBLISHED]=0.0f;
   control[AG_POOL_SIGNAL_READY]=0.0f;
   if(length<2 || length>t)
     {
      control[0]=AG_TECHNICAL;
      return;
     }
// Scratch is persisted and validated alongside Prepared/Active. Every
// rejected transaction must restore a complete valid snapshot, including
// count, ready flag and records partially updated by AGPoolFit.
   AGPoolCopyValues(prepared,scratch,size);
   int baseline=AGPoolFitCached(market,cfg,scratch,work,candidates,batch,evaluated);
   if(baseline!=AG_OK)
     {
      AGPoolCopyValues(prepared,scratch,size);
      control[0]=baseline==AG_INVALID?AG_TECHNICAL:baseline;
      return;
     }
   float baseline_ic=work[mi+1];
   int count=(int)scratch[0],duplicate=0;
   for(int j=0;j<count;j++)
     {
      __global float *r=scratch+10+j*stride;
      int same=(int)r[0]==length;
      for(int k=0;k<length && same;k++)
         same=r[5+k]==tokens[k*token_stride];
      if(same)
         duplicate=1;
     }
   if(duplicate)
     {
      AGPoolCopyValues(scratch,prepared,size);
      control[0]=AG_OK;
      control[1]=baseline_ic;
      control[2]=1.0f;
      return;
     }
   __global float *candidate=scratch+10+count*stride;
   candidate[0]=(float)length;
   AGPoolCopyValues(scratch+1,candidate+1,4);
   for(int k=0;k<t;k++)
      candidate[5+k]=k<length?tokens[k*token_stride]:AG_PAD;
   scratch[0]=(float)(count+1);
   int result=AGPoolFitCached(market,cfg,scratch,work,candidates,batch,evaluated);
   if(result!=AG_OK)
     {
      AGPoolCopyValues(prepared,scratch,size);
      control[0]=result;
      if(result==AG_INVALID)
        {
         control[1]=cfg[AG_CFG_INVALID_TOKEN_REWARD];
         control[2]=1.0f;
        }
      return;
     }
   if(count+1>cap-1)
     {
      int remove=count;
      for(int j=0;j<count;j++)
        {
         float a=fabs(scratch[10+j*stride+12+t]),b=fabs(scratch[10+remove*stride+12+t]);
         if(a<b || (a==b && remove!=count && AGPoolIDGreater(scratch+11+j*stride,scratch+11+remove*stride)))
            remove=j;
        }
      if(remove==count)
        {
         AGPoolCopyValues(prepared,scratch,size);
         control[0]=AG_OK;
         control[1]=baseline_ic;
         control[2]=1.0f;
         return;
        }
      for(int j=remove;j<count;j++)
         AGPoolCopyValues(scratch+10+(j+1)*stride,scratch+10+j*stride,stride);
      scratch[0]=(float)count;
      result=AGPoolFitCached(market,cfg,scratch,work,candidates,batch,evaluated);
      if(result!=AG_OK)
        {
         AGPoolCopyValues(prepared,scratch,size);
         control[0]=AG_TECHNICAL;
         return;
        }
     }
   if(!AGPoolIncrement(scratch+1))
     {
      AGPoolCopyValues(prepared,scratch,size);
      control[0]=AG_TECHNICAL;
      return;
     }
   AGPoolCopyValues(scratch,prepared,size);
   control[6]=prepared[0];
   control[0]=AG_OK;
   control[1]=work[mi+1];
   control[2]=1.0f;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AGPoolCandidate(__global const float *market,__global const float *tokens,
                     const int length,const int token_stride,__global const float *cfg,
                     __global float *banks,__global float *work,__global float *control)
  {AGPoolCandidateCached(market,tokens,length,token_stride,cfg,banks,work,control,work,0,work);}

// Cooperative commit helpers. Every caller participates with the complete group.
// Reductions use local memory only; local size is a power of two, at most 64.
//+------------------------------------------------------------------+
//| Reduce group values in local memory without atomic operations.   |
//+------------------------------------------------------------------+
inline float AGPoolReduce(float value,__local float *scratch,const int operation)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   scratch[lane]=value;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int step=lanes/2;step>0;step/=2)
     {
      if(lane<step)
         scratch[lane]=operation==0?scratch[lane]+scratch[lane+step]:
                       operation==1?fmax(scratch[lane],scratch[lane+step]):
                       fmin(scratch[lane],scratch[lane+step]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   float result=scratch[0];
   barrier(CLK_LOCAL_MEM_FENCE);
// No lane may overwrite scratch before all read.
   return result;
  }

//+------------------------------------------------------------------+
//| Copy independent values cooperatively and synchronize the group. |
//+------------------------------------------------------------------+
inline void AGPoolCopyGroup(__global const float *source,__global float *destination,const int count)
  {
   for(int i=get_local_id(0);i<count;i+=get_local_size(0))
      destination[i]=source[i];
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

//+------------------------------------------------------------------+
//| Normalize rows with parallel terms and ordered FP32 sums.        |
//+------------------------------------------------------------------+
inline void AGPoolStandardizeGroup(__global const float *input,__global const float *validity,
                                   const int offset,const int count,__global const float *cfg,
                                   __global float *normalized,__global float *statistics,
                                   __local float *reduce)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   if(lane==0)
     {
      statistics[AG_STAT_STATUS]=AG_SKIP;
      statistics[AG_STAT_VALID]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(count<1)
      return;
   float a=0.0f,bad=(float)count;
   for(int i=lane;i<count;i+=lanes)
     {
      if(validity[offset+i]!=AG_OK || !isfinite(input[offset+i]))
         bad=fmin(bad,(float)i);
      a=fmax(a,fabs(input[offset+i]));
     }
   int first_bad=(int)AGPoolReduce(bad,reduce,2);
   if(first_bad<count)
     {
      if(lane==0)
         statistics[AG_STAT_STATUS]=validity[offset+first_bad]!=AG_OK?validity[offset+first_bad]:AG_INVALID;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   a=AGPoolReduce(a,reduce,1);
   if(!(a>0.0f))
      return;
   float origin=input[offset]/a,b=0.0f;
   for(int i=lane;i<count;i+=lanes)
      b=fmax(b,fabs(input[offset+i]/a-origin));
   b=AGPoolReduce(b,reduce,1);
   if(!(b>0.0f))
      return;
// Compute terms in parallel, then preserve the scalar FP32 addition order.
// Ill-conditioned ridge columns amplify changes in mean/variance rounding.
   float mean=0.0f;
   for(int base=0;base<count;base+=lanes)
     {
      int i=base+lane;
      reduce[lane]=i<count?(input[offset+i]/a-origin)/b/(float)count:0.0f;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lane==0)
         for(int j=0;j<min(lanes,count-base);j++)
            mean+=reduce[j];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   mean=AGPoolReduce(mean,reduce,0);
   float variance=0.0f;
   for(int base=0;base<count;base+=lanes)
     {
      int i=base+lane;
      float d=i<count?(input[offset+i]/a-origin)/b-mean:0.0f;
      reduce[lane]=d*d/(float)count;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lane==0)
         for(int j=0;j<min(lanes,count-base);j++)
            variance+=reduce[j];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   variance=AGPoolReduce(variance,reduce,0);
   if(!isfinite(variance) || variance<=cfg[AG_CFG_EPS_VAR])
      return;
   float sd=sqrt(variance);
   for(int i=lane;i<count;i+=lanes)
      normalized[i]=((input[offset+i]/a-origin)/b-mean)/sd;
   if(lane==0)
     {
      statistics[AG_STAT_SCALE]=a;
      statistics[AG_STAT_ORIGIN]=origin;
      statistics[AG_STAT_DIFF_SCALE]=b;
      statistics[AG_STAT_MEAN]=mean;
      statistics[AG_STAT_STD]=sd;
      statistics[AG_STAT_STATUS]=AG_OK;
      statistics[AG_STAT_VALID]=(float)count;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

// Ring-only parallel sum variant; ridge normalization retains its addition order.
inline void AGPoolStandardizeRingGroup(__global const float *input,__global const float *validity,
                                   const int offset,const int count,__global const float *cfg,
                                   __global float *normalized,__global float *statistics,
                                   __local float *reduce)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   if(lane==0)
     {
      statistics[AG_STAT_STATUS]=AG_SKIP;
      statistics[AG_STAT_VALID]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(count<1)
      return;
   float a=0.0f,bad=(float)count;
   for(int i=lane;i<count;i+=lanes)
     {
      if(validity[offset+i]!=AG_OK || !isfinite(input[offset+i]))
         bad=fmin(bad,(float)i);
      a=fmax(a,fabs(input[offset+i]));
     }
   int first_bad=(int)AGPoolReduce(bad,reduce,2);
   if(first_bad<count)
     {
      if(lane==0)
         statistics[AG_STAT_STATUS]=validity[offset+first_bad]!=AG_OK?validity[offset+first_bad]:AG_INVALID;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   a=AGPoolReduce(a,reduce,1);
   if(!(a>0.0f))
      return;
   float origin=input[offset]/a,b=0.0f;
   for(int i=lane;i<count;i+=lanes)
      b=fmax(b,fabs(input[offset+i]/a-origin));
   b=AGPoolReduce(b,reduce,1);
   if(!(b>0.0f))
      return;
// Ring normalization has no ridge-system sensitivity: reduce sums locally.
   float mean=0.0f;
   for(int i=lane;i<count;i+=lanes)
      mean+=(input[offset+i]/a-origin)/b/(float)count;
   mean=AGPoolReduce(mean,reduce,0);
   float variance=0.0f;
   for(int i=lane;i<count;i+=lanes)
     {
      float d=(input[offset+i]/a-origin)/b-mean;
      variance+=d*d/(float)count;
     }
   variance=AGPoolReduce(variance,reduce,0);
   if(!isfinite(variance) || variance<=cfg[AG_CFG_EPS_VAR])
      return;
   float sd=sqrt(variance);
   for(int i=lane;i<count;i+=lanes)
      normalized[i]=((input[offset+i]/a-origin)/b-mean)/sd;
   if(lane==0)
     {
      statistics[AG_STAT_SCALE]=a;
      statistics[AG_STAT_ORIGIN]=origin;
      statistics[AG_STAT_DIFF_SCALE]=b;
      statistics[AG_STAT_MEAN]=mean;
      statistics[AG_STAT_STD]=sd;
      statistics[AG_STAT_STATUS]=AG_OK;
      statistics[AG_STAT_VALID]=(float)count;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

// The small triangular dependencies stay on the leader after parallel Gram/RHS construction.
//+------------------------------------------------------------------+
//| Resolve the small Cholesky dependencies on the group leader.     |
//+------------------------------------------------------------------+
inline int AGPoolSolveGroupLeader(__global float *matrix,__global float *rhs,
                                  __global float *weights,const int columns)
  {
   for(int i=0;i<columns;i++)
     {
      for(int j=0;j<=i;j++)
        {
         float r=matrix[i*columns+j];
         for(int k=0;k<j;k++)
            r-=matrix[i*columns+k]*matrix[j*columns+k];
         if(i==j)
           {
            if(!(r>0.0f) || !isfinite(r))
               return AG_TECHNICAL;
            matrix[i*columns+j]=sqrt(r);
           }
         else
            matrix[i*columns+j]=r/matrix[j*columns+j];
        }
     }
   for(int i=0;i<columns;i++)
     {
      float x=rhs[i];
      for(int j=0;j<i;j++)
         x-=matrix[i*columns+j]*weights[j];
      weights[i]=x/matrix[i*columns+i];
     }
   for(int i=columns-1;i>=0;i--)
     {
      float x=weights[i];
      for(int j=i+1;j<columns;j++)
         x-=matrix[j*columns+i]*weights[j];
      weights[i]=x/matrix[i*columns+i];
      if(!isfinite(weights[i]))
         return AG_TECHNICAL;
     }
   return AG_OK;
  }

//+------------------------------------------------------------------+
//| Build Gram/RHS in parallel and reduce metrics in local memory.   |
//+------------------------------------------------------------------+
inline void AGPoolRidgeGroup(__global const float *z,__global const float *target,
                             const int count,const int columns,__global const float *cfg,
                             __global float *scratch,__global float *weights,__global float *metrics,
                             __local float *reduce,__local int *shared)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   if(lane==0)
     {
      metrics[0]=AG_TECHNICAL;
      metrics[1]=0.0f;
      metrics[2]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(count<1 || columns<1)
      return;
   __global float *rhs=scratch+columns*columns;
// Independent matrix cells and RHS entries run across the group.
   for(int cell=lane;cell<columns*columns+columns;cell+=lanes)
     {
      if(cell>=columns*columns)
        {
         int i=cell-columns*columns;
         float c=0.0f;
         for(int t=0;t<count;t++)
            c+=z[i*count+t]*target[t]/(float)count;
         rhs[i]=c;
        }
      else
        {
         int i=cell/columns,j=cell%columns;
         if(j>i)
            continue;
         float r=0.0f;
         for(int t=0;t<count;t++)
            r+=z[i*count+t]*z[j*count+t]/(float)count;
         scratch[cell]=r+(i==j?cfg[AG_CFG_RIDGE]:0.0f);
        }
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(lane==0)
      shared[0]=AGPoolSolveGroupLeader(scratch,rhs,weights,columns);
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   if(shared[0]!=AG_OK)
      return;
   float mean=0.0f;
   for(int t=lane;t<count;t+=lanes)
     {
      float a=0.0f;
      for(int j=0;j<columns;j++)
         a+=weights[j]*z[j*count+t];
      mean+=a/(float)count;
     }
   mean=AGPoolReduce(mean,reduce,0);
   float covariance=0.0f,variance=0.0f;
   for(int t=lane;t<count;t+=lanes)
     {
      float a=0.0f;
      for(int j=0;j<columns;j++)
         a+=weights[j]*z[j*count+t];
      a-=mean;
      covariance+=a*target[t]/(float)count;
      variance+=a*a/(float)count;
     }
   covariance=AGPoolReduce(covariance,reduce,0);
   variance=AGPoolReduce(variance,reduce,0);
   float ic=variance>cfg[AG_CFG_EPS_VAR]?covariance/sqrt(variance):0.0f;
   if(lane==0 && isfinite(ic))
     {
      metrics[0]=AG_OK;
      metrics[1]=ic;
      metrics[2]=variance>cfg[AG_CFG_EPS_VAR]?1.0f:0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

//+------------------------------------------------------------------+
//| Fit the pool cooperatively, reusing cached candidate formulas.   |
//+------------------------------------------------------------------+
inline int AGPoolFitGroup(__global const float *market,__global const float *cfg,
                          __global float *bank,__global float *work,
                          __global const float *candidates,const int batch,__global const float *evaluated,
                          __local float *reduce,__local int *shared)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   int w=(int)cfg[AG_CFG_W],t=(int)cfg[AG_CFG_MAX_TOKENS];
   int n=(int)cfg[AG_CFG_IC_WINDOW],h=(int)cfg[AG_CFG_HORIZON];
   int cap=(int)cfg[AG_CFG_POOL]+1,stride=AGPoolStride(cfg),offset=w-h-n;
   __global float *stack=work,*valid=stack+w*t;
   __global float *raw=valid+max(w*t,3*(t+1)),*status=raw+w;
   __global float *target=status+w,*tv=target+w,*y=tv+w,*yst=y+n;
   __global float *z=yst+7,*solve=z+cap*n,*weights=solve+cap*(cap+1),*metric=weights+cap;
   int count=(int)bank[0];
   if(lane==0)
     {
      metric[0]=AG_SKIP;
      metric[1]=0.0f;
      metric[2]=0.0f;
     }
   for(int i=lane;i<w;i+=lanes)
     {
      target[i]=0.0f;
      tv[i]=AG_SKIP;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   float invalid=0.0f;
   for(int i=offset+lane;i<w-h;i+=lanes)
     {
      float p=market[3*w+i],q=market[3*w+i+h];
      if(!(p>0.0f) || !(q>0.0f) || !isfinite(p) || !isfinite(q))
         invalid=1.0f;
      target[i]=q/p-1.0f;
      tv[i]=isfinite(target[i])?AG_OK:AG_TECHNICAL;
     }
   invalid=AGPoolReduce(invalid,reduce,1);
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(invalid>0.0f)
      return AG_TECHNICAL;
   AGPoolStandardizeGroup(target,tv,offset,n,cfg,y,yst,reduce);
   if(yst[5]!=AG_OK)
      return AG_SKIP;
   if(count==0)
     {
      if(lane==0)
         metric[0]=AG_OK;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return AG_OK;
     }
   for(int j=0;j<count;j++)
     {
      __global float *record=bank+10+j*stride;
      if(lane==0)
        {
         shared[1]=-1;
         int token_stride=4+AG_VOCABULARY;
         for(int c=0;c<batch && shared[1]<0;c++)
           {
            __global const float *packed=candidates+(size_t)c*(1+(size_t)t*token_stride);
            int same=record[0]==packed[0];
            for(int k=0;k<(int)record[0] && same;k++)
               same=record[5+k]==packed[1+k*token_stride];
            if(same)
               shared[1]=c;
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      int cached=shared[1];
      if(cached>=0)
        {
         __global const float *values=evaluated+(size_t)cached*2*w;
         for(int i=lane;i<w;i+=lanes)
           {
            raw[i]=values[i];
            status[i]=values[w+i];
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
        }
      else
        {
         AGFormulaEvaluate(market,record+5,(int)record[0],cfg,stack,valid,raw,status,1,shared);
         barrier(CLK_GLOBAL_MEM_FENCE);
        }
      AGPoolStandardizeGroup(raw,status,offset,n,cfg,z+j*n,record+5+t,reduce);
      // Snapshot the branch result before a rejected transaction restores bank records.
      int column_status=(int)record[10+t];
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(column_status!=AG_OK)
         return column_status;
     }
   AGPoolRidgeGroup(z,y,n,count,cfg,solve,weights,metric,reduce,shared);
   if(metric[0]!=AG_OK)
      return AG_TECHNICAL;
   for(int i=lane;i<n;i+=lanes)
     {
      float a=0.0f;
      for(int j=0;j<count;j++)
         a+=weights[j]*z[j*n+i];
      raw[i]=a;
      status[i]=AG_OK;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   AGPoolStandardizeGroup(raw,status,0,n,cfg,target,yst,reduce);
   float ic=0.0f;
   if(yst[5]==AG_OK)
      for(int i=lane;i<n;i+=lanes)
         ic+=target[i]*y[i]/(float)n;
   ic=AGPoolReduce(ic,reduce,0);
   if(lane==0)
     {
      metric[1]=ic;
      metric[2]=yst[5]==AG_OK?1.0f:0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(!isfinite(ic))
      return AG_TECHNICAL;
   for(int j=lane;j<count;j+=lanes)
      bank[10+j*stride+12+t]=weights[j];
   if(lane==0)
      bank[9]=1.0f;
   barrier(CLK_GLOBAL_MEM_FENCE);
   return AG_OK;
  }

//+------------------------------------------------------------------+
//| Apply one ordered candidate transaction with cooperative fitting.|
//+------------------------------------------------------------------+
inline void AGPoolCandidateGroup(__global const float *market,__global const float *tokens,
                                 const int length,const int token_stride,__global const float *cfg,
                                 __global float *banks,__global float *work,__global float *control,
                                 __global const float *candidates,const int batch,__global const float *evaluated,
                                 __local float *reduce,__local int *shared)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   int size=AGPoolBankSize(cfg),stride=AGPoolStride(cfg);
   int t=(int)cfg[AG_CFG_MAX_TOKENS],cap=(int)cfg[AG_CFG_POOL]+1;
   int w=(int)cfg[AG_CFG_W],n=(int)cfg[AG_CFG_IC_WINDOW];
   int mi=w*t+max(w*t,3*(t+1))+4*w+n+7+cap*n+cap*(cap+1)+cap;
   __global float *prepared=banks,*scratch=banks+2*size;
   if(lane==0)
     {
      control[AG_POOL_PREPARED_COUNT]=prepared[0];
      control[AG_POOL_ACTIVE_COUNT]=banks[size];
      control[AG_POOL_STATUS]=AG_SKIP;
      control[AG_POOL_REWARD_READY]=0.0f;
      control[AG_POOL_PUBLISHED]=0.0f;
      control[AG_POOL_SIGNAL_READY]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(length<2 || length>t)
     {
      if(lane==0)
         control[0]=AG_TECHNICAL;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   AGPoolCopyGroup(prepared,scratch,size);
   int baseline=AGPoolFitGroup(market,cfg,scratch,work,candidates,batch,evaluated,reduce,shared);
   if(baseline!=AG_OK)
     {
      AGPoolCopyGroup(prepared,scratch,size);
      if(lane==0)
         control[0]=baseline==AG_INVALID?AG_TECHNICAL:baseline;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   float baseline_ic=work[mi+1];
   int count=(int)scratch[0];
   if(lane==0)
     {
      shared[1]=0;
      for(int j=0;j<count;j++)
        {
         __global float *r=scratch+10+j*stride;
         int same=(int)r[0]==length;
         for(int k=0;k<length && same;k++)
            same=r[5+k]==tokens[k*token_stride];
         if(same)
            shared[1]=1;
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   if(shared[1])
     {
      AGPoolCopyGroup(scratch,prepared,size);
      if(lane==0)
        {
         control[0]=AG_OK;
         control[1]=baseline_ic;
         control[2]=1.0f;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   __global float *candidate=scratch+10+count*stride;
   if(lane==0)
     {
      candidate[0]=(float)length;
      AGPoolCopyValues(scratch+1,candidate+1,4);
      scratch[0]=(float)(count+1);
     }
   for(int k=lane;k<t;k+=lanes)
      candidate[5+k]=k<length?tokens[k*token_stride]:AG_PAD;
   barrier(CLK_GLOBAL_MEM_FENCE);
   int result=AGPoolFitGroup(market,cfg,scratch,work,candidates,batch,evaluated,reduce,shared);
   if(result!=AG_OK)
     {
      AGPoolCopyGroup(prepared,scratch,size);
      if(lane==0)
        {
         control[0]=result;
         if(result==AG_INVALID)
           {
            control[1]=cfg[AG_CFG_INVALID_TOKEN_REWARD];
            control[2]=1.0f;
           }
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   if(count+1>cap-1)
     {
      if(lane==0)
        {
         int remove=count;
         for(int j=0;j<count;j++)
           {
            float a=fabs(scratch[10+j*stride+12+t]),b=fabs(scratch[10+remove*stride+12+t]);
            if(a<b || (a==b && remove!=count && AGPoolIDGreater(scratch+11+j*stride,scratch+11+remove*stride)))
               remove=j;
           }
         shared[2]=remove;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      int remove=shared[2];
      if(remove==count)
        {
         AGPoolCopyGroup(prepared,scratch,size);
         if(lane==0)
           {
            control[0]=AG_OK;
            control[1]=baseline_ic;
            control[2]=1.0f;
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         return;
        }
      // Overlapping records must move in order, with a barrier between records.
      for(int j=remove;j<count;j++)
         AGPoolCopyGroup(scratch+10+(j+1)*stride,scratch+10+j*stride,stride);
      if(lane==0)
         scratch[0]=(float)count;
      barrier(CLK_GLOBAL_MEM_FENCE);
      result=AGPoolFitGroup(market,cfg,scratch,work,candidates,batch,evaluated,reduce,shared);
      if(result!=AG_OK)
        {
         AGPoolCopyGroup(prepared,scratch,size);
         if(lane==0)
            control[0]=AG_TECHNICAL;
         barrier(CLK_GLOBAL_MEM_FENCE);
         return;
        }
     }
   if(lane==0)
      shared[0]=AGPoolIncrement(scratch+1);
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   if(!shared[0])
     {
      AGPoolCopyGroup(prepared,scratch,size);
      if(lane==0)
         control[0]=AG_TECHNICAL;
      barrier(CLK_GLOBAL_MEM_FENCE);
      return;
     }
   AGPoolCopyGroup(scratch,prepared,size);
   if(lane==0)
     {
      control[6]=prepared[0];
      control[0]=AG_OK;
      control[1]=work[mi+1];
      control[2]=1.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

// Commit in candidate order because a candidate is rewarded against the pool
// resulting from all previous candidates. Unconsumed candidates never enter PPO.
__kernel void AGPoolCommitBatch(__global const float *market,__global const float *candidates,
                                const int batch,__global const float *cfg,
                                __global float *banks,__global float *work,
                                __global float *rings,__global float *control,
                                const int first_row,const int capacity,
                                __global float *prefixes,__global float *masks,
                                __global float *samples,__global float *metadata,
                                __global float *gae,__global float *summary,__global const float *evaluated)
  {
   int lane=get_local_id(0);
   int i=(int)(get_global_offset(0)/get_local_size(0));
   __local float reduce[64];
   __local int shared[4];
   int maximum=(int)cfg[AG_CFG_MAX_TOKENS];
   int token_stride=4+AG_VOCABULARY;
   int target=(int)cfg[AG_CFG_TOKEN_ROLLOUT];
   int row=i==0?first_row:(int)summary[2];
   int processed=i==0?0:(int)summary[1];
   int rejected=i==0?0:(int)summary[3];
   int status=i==0?AG_OK:(int)summary[0];
// Each dispatch applies one candidate against the preceding pool state.
// Keep PPO rows and counters on the device between dispatches.
   do
     {
      if(i>=batch || row>=target || status!=AG_OK)
         break;
      __global const float *candidate=candidates+(size_t)i*(1+(size_t)maximum*token_stride);
      int length=(int)candidate[0];
      __global const float *tokens=candidate+1;
      if(length<2 || length>maximum || tokens[(length-1)*token_stride]!=AG_STOP ||
         length>capacity-row)
        {
         status=AG_TECHNICAL;
         break;
        }

      AGPoolCandidateGroup(market,tokens,length,token_stride,cfg,banks,work,control,candidates,batch,evaluated,reduce,shared);
      // No further barriers: the leader serializes PPO metadata and summary.
      if(lane!=0)
         return;
      processed++;
      if(control[AG_POOL_STATUS]==AG_TECHNICAL)
         rejected++;
      if(control[AG_POOL_REWARD_READY]!=1.0f)
         continue;

      for(int k=0;k<length;k++,row++)
        {
         __global const float *token=tokens+k*token_stride;
         __global float *prefix=prefixes+(size_t)row*(maximum+1);
         __global float *meta=metadata+(size_t)row*7;
         __global float *transition=gae+(size_t)row*AG_GAE_FIELDS;
         int terminal=k==length-1;
         prefix[0]=AG_BOS;
         for(int j=0;j<k;j++)
            prefix[j+1]=tokens[j*token_stride];
         for(int j=0;j<AG_VOCABULARY;j++)
            masks[(size_t)row*AG_VOCABULARY+j]=token[4+j];
         samples[row]=token[0];
         meta[0]=token[1];
         meta[1]=token[2];
         meta[2]=token[3];
         meta[3]=token[1];
         meta[4]=0.0f;
         meta[5]=0.0f;
         meta[6]=0.0f;
         transition[AG_GAE_REWARD]=terminal?control[AG_POOL_REWARD]:0.0f;
         transition[AG_GAE_DELTA]=1.0f;
         transition[AG_GAE_VALUE]=token[3];
         transition[AG_GAE_NEXT]=terminal?0.0f:token[token_stride+3];
         transition[AG_GAE_BOOTSTRAP]=terminal?0.0f:1.0f;
         transition[AG_GAE_CONTINUE]=terminal?0.0f:1.0f;
         transition[AG_GAE_READY]=1.0f;
         summary[4+AG_POOL_CONTROL_SIZE+row-first_row]=(float)(k+1);
        }
     }
   while(false);
   if(lane!=0)
      return;
   summary[0]=(float)status;
   summary[1]=(float)processed;
   summary[2]=(float)row;
   summary[3]=(float)rejected;
   for(int j=0;j<AG_POOL_CONTROL_SIZE;j++)
      summary[4+j]=control[j];
  }

// modes: 0 candidate transaction; 1 pin Prepared for publication;
// 2 commit pinned pool/ring after external LSTM preparation; 3 abort pin.
// Control: status,reward,reward_ready,published,signal_ready,active_ready,
// prepared_count,active_count.
__kernel void AGPoolTransaction(__global const float *market,__global const float *tokens,
                                const int length,__global const float *cfg,
                                __global float *banks,__global float *work,
                                __global float *rings,__global float *control,const int mode,
                                __global float *signal,__global float *context)
  {
   if(get_group_id(0)!=0)
      return;
   int lane=get_local_id(0);
   __local float reduce[64];
   __local int shared[4];
   if(mode==0)
     {
      AGPoolCandidateGroup(market,tokens,length,1,cfg,banks,work,control,work,0,work,reduce,shared);
      return;
     }
   int size=AGPoolBankSize(cfg),rs=AGPoolRingSize(cfg);
   if(mode==1)
     {
      __global float *prepared=banks,*scratch=banks+2*size,*pinned=banks+3*size;
      if(lane==0)
        {
         control[6]=prepared[0];control[7]=banks[size];control[0]=AG_SKIP;
         control[2]=0.0f;control[3]=0.0f;control[4]=0.0f;pinned[9]=0.0f;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(prepared[0]<1.0f || prepared[9]!=1.0f)
         return;
      AGPoolCopyGroup(prepared,scratch,size);
      int refresh=AGPoolFitGroup(market,cfg,scratch,work,work,0,work,reduce,shared);
      if(refresh!=AG_OK)
        {
         AGPoolCopyGroup(prepared,scratch,size);
         if(lane==0) control[0]=refresh==AG_INVALID?AG_TECHNICAL:refresh;
         return;
        }
      AGPoolCopyGroup(scratch,pinned,size);
      AGPoolCopyGroup(scratch,prepared,size);
      for(int i=lane;i<rs;i+=get_local_size(0)) rings[rs+i]=0.0f;
      if(lane==0) control[0]=AG_OK;
      return;
     }
   __global float *prepared=banks,*active=banks+size,*pinned=banks+3*size;
   if(lane==0)
     {
      control[6]=prepared[0];
      control[7]=active[0];
      control[0]=AG_SKIP;
      control[2]=0.0f;
      control[3]=0.0f;
      control[4]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(mode==2)
     {
      if(pinned[9]!=1.0f || rings[rs+2]<(float)((int)cfg[AG_CFG_CONTEXT]) || rings[rs+10]!=AG_OK)
         return;
      if(lane==0)
        {
         AGPoolCopyValues(active+5,pinned+5,4);
         shared[0]=AGPoolIncrement(pinned+5);
         if(!shared[0])
            control[0]=AG_TECHNICAL;
        }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if(!shared[0])
         return;
      AGPoolCopyGroup(pinned,active,size);
      AGPoolCopyGroup(rings+rs,rings,rs);
      int nw=(int)cfg[AG_CFG_NORM_WINDOW],nc=(int)cfg[AG_CFG_CONTEXT];
      AGPoolCopyGroup(rings+11+nw,context,nc);
      if(lane==0)
        {
         signal[0]=context[nc-1];
         control[7]=active[0];
         pinned[9]=0.0f;
         control[0]=AG_PUBLISHED;
         control[3]=1.0f;
         control[5]=1.0f;
        }
      return;
     }
   if(mode==AG_POOL_RESTART)
     {
      if(pinned[9]!=1.0f)
         return;
      for(int i=lane;i<rs;i+=get_local_size(0))
         rings[rs+i]=0.0f;
      if(lane==0)
         control[0]=AG_OK;
      return;
     }
   if(lane!=0)
      return;
   if(mode==3)
     {
      pinned[9]=0.0f;
      control[0]=AG_OK;
      return;
     }
   if(mode==4)
     {
      rings[2]=0.0f;
      control[0]=AG_OK;
      return;
     }
   control[0]=AG_TECHNICAL;
  }
// mode 0: advance active ring; 1: use frozen active statistics (warmup);
// 2: append publication score and one causal g when its window is full.
__kernel void AGPoolSignal(__global const float *market,__global const float *cfg,
                           __global float *banks,__global float *work,
                           __global float *rings,__global float *control,
                           __global float *signal,__global float *context,const int mode,const int start)
  {
   if(get_group_id(0)!=0)
      return;
   int lane=get_local_id(0),lanes=get_local_size(0);
   __local int syntax;
   __local float reduce[64];
   __local float shifted[64];
   int size=AGPoolBankSize(cfg),rs=AGPoolRingSize(cfg),stride=AGPoolStride(cfg);
   int nw=(int)cfg[AG_CFG_NORM_WINDOW],nc=(int)cfg[AG_CFG_CONTEXT];
   int w=(int)cfg[AG_CFG_W],t=(int)cfg[AG_CFG_MAX_TOKENS];
   __global float *bank=banks+(mode==2?3:1)*size,*ring=rings+(mode==2?rs:0);
   __global float *stack=work,*valid=stack+w*t,*raw=valid+max(w*t,3*(t+1)),*status=raw+w;
   __global float *tmp=status+w,*tv=tmp+w;
   if(lane==0)
     {
      control[6]=banks[0];control[7]=banks[size];
      control[0]=AG_SKIP;control[4]=0.0f;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(bank[0]<1.0f || bank[9]!=1.0f)
      return;
   int cap=(int)cfg[AG_CFG_POOL]+1,n=(int)cfg[AG_CFG_IC_WINDOW];
   int base=w*t+max(w*t,3*(t+1))+4*w+n+7+cap*n+cap*(cap+1)+cap+3;
   __global float *scores=work+base;
   for(int row=start+lane;row<w;row+=lanes)
      scores[row]=0.0f;
   barrier(CLK_GLOBAL_MEM_FENCE);
// Evaluate each formula once for all rows of the first history window.
   for(int j=0;j<(int)bank[0];j++)
     {
      __global float *record=bank+10+j*stride;
      AGFormulaEvaluate(market,record+5,(int)record[0],cfg,stack,valid,raw,status,1,&syntax);
      barrier(CLK_GLOBAL_MEM_FENCE);
      float first_bad=(float)w;
      for(int row=start+lane;row<w;row+=lanes)
         if(status[row]!=AG_OK)
            first_bad=fmin(first_bad,(float)row);
      int invalid=(int)AGPoolReduce(first_bad,reduce,2);
      if(invalid<w)
        {
         if(lane==0)
           {
            control[0]=status[invalid];
            if(mode==2)
               bank[9]=0.0f;
           }
         return;
        }
      for(int row=start+lane;row<w;row+=lanes)
         scores[row]+=record[12+t]*AGPoolAffine(raw[row],record+5+t);
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
   // Rows remain chronological; all lanes cooperate within each row.
   for(int row=start;row<w;row++)
     {
      float score=scores[row];
      if(!isfinite(score))
        {
         if(lane==0)
           {
            control[0]=AG_SKIP;
            if(mode==2)
               bank[9]=0.0f;
           }
         return;
        }
      if(mode!=1)
        {
         if(lane==0)
           {
            int pos=(int)ring[1];
            ring[11+pos]=score;
            ring[1]=(float)((pos+1)%nw);
            ring[0]=fmin((float)nw,ring[0]+1.0f);
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         int full=ring[0]>=(float)nw;
         // Snapshot before any lane advances to the next row's write.
         barrier(CLK_GLOBAL_MEM_FENCE);
         if(!full)
            continue;
         for(int i=lane;i<nw;i+=lanes)
            tv[i]=AG_OK;
         barrier(CLK_GLOBAL_MEM_FENCE);
         AGPoolStandardizeRingGroup(ring+11,tv,0,nw,cfg,tmp,ring+3,reduce);
         if(lane==0)
            ring[10]=ring[8];
         barrier(CLK_GLOBAL_MEM_FENCE);
        }
      int ready=ring[0]>=(float)nw && ring[10]==AG_OK;
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(!ready)
        {
         if(lane==0)
           {
            control[6]=banks[0];
            control[7]=banks[size];
            control[0]=AG_SKIP;
            control[4]=0.0f;
            if(mode==2)
               ring[2]=0.0f;
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         continue;
        }
      float g=AGPoolAffine(score,ring+3),clip=cfg[AG_CFG_SIGNAL_CLIP];
      if(!isfinite(g) || !(clip>0.0f))
         return;
      g=clamp(g,-clip,clip)/clip;
      int count=(int)ring[2];
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(count>=nc)
        {
         // Read each source tile completely before overlapping destination writes.
         // Tiling avoids any assumption that context length fits W or local memory.
         for(int base=0;base<nc-1;base+=lanes)
           {
            int i=base+lane;
            if(i<nc-1)
               shifted[lane]=ring[11+nw+i+1];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if(i<nc-1)
               ring[11+nw+i]=shifted[lane];
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
           }
         count=nc-1;
        }
      if(lane==0)
        {
         signal[0]=g;
         ring[11+nw+count]=g;
         ring[2]=(float)(count+1);
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      AGPoolCopyGroup(ring+11+nw,context,count+1);
      if(lane==0)
        {
         control[0]=AG_OK;
         control[4]=(mode==2 && count+1<nc)?0.0f:1.0f;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
  }
// Export exact selected tokens without host numerical processing.
__kernel void AGPoolCopy(__global const float *source,__global float *destination,const int count)
  {int i=get_global_id(0);if(i<count) destination[i]=source[i];}
//+------------------------------------------------------------------+
