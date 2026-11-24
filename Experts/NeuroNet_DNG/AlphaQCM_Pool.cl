//+------------------------------------------------------------------+
//|                                                 AlphaQCM_Pool.cl |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#define AQ_POOL_NUMERIC_ERROR (-1)

//+------------------------------------------------------------------+
//| AQTileReduce - Reduce adjacent independent lane tiles            |
//+------------------------------------------------------------------+
inline float AQTileReduce(const float value,const int width,const int operation,
                          __local float *reduce)
  {
   const int lane=get_local_id(0);
   const int part=lane%width;
   reduce[lane]=value;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int step=width/2;step>0;step/=2)
     {
      if(part<step)
         reduce[lane]=operation==0?reduce[lane]+reduce[lane+step]:
                      operation==1?fmax(reduce[lane],reduce[lane+step]):
                      fmin(reduce[lane],reduce[lane+step]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   const float result=reduce[lane-part];
   barrier(CLK_LOCAL_MEM_FENCE);
   return(result);
  }

//+------------------------------------------------------------------+
//| AQNormalize - Preserve donor statistics for each active tile     |
//+------------------------------------------------------------------+
inline int AQNormalize(__global const float *x,__global const float *status,const int n,
                       __global const float *cfg,__global float *z,__global float *stats,
                       const int width,const bool active,__local float *reduce)
  {
   const int part=get_local_id(0)%width;
   float bad=n;
   float scale=0.0f;
   if(active)
      for(int i=part;i<n;i+=width)
        {
         if(status[i]!=AG_OK||!isfinite(x[i]))
            bad=fmin(bad,(float)i);
         scale=fmax(scale,fabs(x[i]));
        }
   const int first=(int)AQTileReduce(bad,width,2,reduce);
   int result=first<n?(status[first]==AG_OK?AQ_POOL_NUMERIC_ERROR:(int)status[first]):AG_OK;
   scale=AQTileReduce(scale,width,1,reduce);
   if(active&&result==AG_OK&&scale==0.0f)
      result=AG_SKIP;

   const float origin=active&&result==AG_OK?x[0]/scale:0.0f;
   float amplitude=0.0f;
   if(active&&result==AG_OK)
      for(int i=part;i<n;i+=width)
         amplitude=fmax(amplitude,fabs(x[i]/scale-origin));
   amplitude=AQTileReduce(amplitude,width,1,reduce);
   if(active&&result==AG_OK&&amplitude==0.0f)
      result=AG_SKIP;

   float mean=0.0f;
   if(active&&result==AG_OK)
      for(int i=part;i<n;i+=width)
         mean+=(x[i]/scale-origin)/amplitude;
   mean=AQTileReduce(mean,width,0,reduce)/n;
   float variance=0.0f;
   if(active&&result==AG_OK)
      for(int i=part;i<n;i+=width)
        {
         const float delta=(x[i]/scale-origin)/amplitude-mean;
         variance+=delta*delta;
        }
   variance=AQTileReduce(variance,width,0,reduce)/n;
   if(active&&result==AG_OK)
     {
      if(!isfinite(variance))
         result=AQ_POOL_NUMERIC_ERROR;
      else if(variance<=cfg[AG_CFG_EPS_VAR])
         result=AG_SKIP;
     }
   if(active&&result==AG_OK)
     {
      const float sd=sqrt(variance);
      for(int i=part;i<n;i+=width)
         z[i]=((x[i]/scale-origin)/amplitude-mean)/sd;
      if(part==0)
        {
         stats[0]=scale;
         stats[1]=origin;
         stats[2]=amplitude;
         stats[3]=mean;
         stats[4]=sd;
         stats[5]=AG_OK;
         stats[6]=n;
        }
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   return(result);
  }

// skip remaps columns during eviction. The committed Z cache is compacted
// only after both ridge solves have succeeded.
inline int AQRidge(__global const float *z,__global const float *target,int n,int p,int skip,
                   __global const float *cfg,__global float *matrix,__global float *weights,
                   __global float *raw,__global float *valid,__global float *normalized,
                   __global float *stats,__global float *metric,__local float *reduce)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   int width=min(lanes,8),tiles=lanes/width;
   __global float *rhs=matrix+p*p;
   for(int base=0;base<p*(p+1);base+=tiles)
     {
      int cell=base+lane/width,i=cell/(p+1),j=cell%(p+1);
      float dot=0;
      if(i<p&&(j<=i||j==p))
        {
         __global const float *a=z+(i+(skip>=0&&i>=skip))*n;
         __global const float *b=j==p?target:z+(j+(skip>=0&&j>=skip))*n;
         for(int k=lane%width;k<n;k+=width)dot+=a[k]*b[k];
        }
       dot=AQTileReduce(dot,width,0,reduce)/n;
      if(i<p&&lane%width==0)
        {
         if(j==p)rhs[i]=dot;
         else if(j<=i)matrix[i*p+j]=dot+(i==j?cfg[AG_CFG_RIDGE]:0.0f);
        }
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   // Right-looking Cholesky. Only the pivot order is serial.
   for(int k=0;k<p;k++)
     {
      float diag=matrix[k*p+k];
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(!isfinite(diag)||diag<=0)return(AQ_POOL_NUMERIC_ERROR);
      diag=sqrt(diag);
      if(lane==0)matrix[k*p+k]=diag;
      for(int i=k+1+lane;i<p;i+=lanes)matrix[i*p+k]/=diag;
      barrier(CLK_GLOBAL_MEM_FENCE);
      int tail=p-k-1;
      for(int cell=lane;cell<tail*tail;cell+=lanes)
        {
         int i=k+1+cell/tail,j=k+1+cell%tail;
         if(j<=i)matrix[i*p+j]-=matrix[i*p+k]*matrix[j*p+k];
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
   for(int k=0;k<p;k++)
     {
      float x=rhs[k]/matrix[k*p+k];
      if(!isfinite(x))return(AQ_POOL_NUMERIC_ERROR);
      if(lane==0)weights[k]=x;
      for(int i=k+1+lane;i<p;i+=lanes)rhs[i]-=matrix[i*p+k]*x;
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
   for(int k=p-1;k>=0;k--)
     {
      float x=weights[k]/matrix[k*p+k];
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(!isfinite(x))return(AQ_POOL_NUMERIC_ERROR);
      if(lane==0)weights[k]=x;
      for(int i=lane;i<k;i+=lanes)weights[i]-=matrix[k*p+i]*x;
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
   for(int base=0;base<n;base+=tiles)
     {
      int row=base+lane/width;
      float sum=0;
      if(row<n)
         for(int j=lane%width;j<p;j+=width)
            sum+=weights[j]*z[(j+(skip>=0&&j>=skip))*n+row];
      sum=AQTileReduce(sum,width,0,reduce);
      if(row<n&&lane%width==0)
        {
          raw[row]=sum;valid[row]=isfinite(sum)?AG_OK:AQ_POOL_NUMERIC_ERROR;
        }
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   int status=AQNormalize(raw,valid,n,cfg,normalized,stats,lanes,true,reduce);
   if(status!=AG_OK&&status!=AG_SKIP)return(status);
   float dot=0;
   if(status==AG_OK)
      for(int i=lane;i<n;i+=lanes)dot+=normalized[i]*target[i];
   dot=AGPoolReduce(dot,reduce,0)/n;
   if(!isfinite(dot))return(AQ_POOL_NUMERIC_ERROR);
   if(lane==0)
     {
      metric[0]=AG_OK;metric[1]=dot;metric[2]=status==AG_OK;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   return(AG_OK);
  }

__kernel void AQCMPoolEvaluatePrepared(__global const float *market,__global const float *cfg,
                                       __global const float *banks,__global float *work,
                                       __global float *values,int first,int packed)
  {
   int group=get_group_id(0),column=first+group;
   if(!packed&&column>=(int)banks[0])return;
   int w=cfg[AG_CFG_W],t=cfg[AG_CFG_MAX_TOKENS];
   int stride=packed?4+AG_VOCABULARY:1;
   __global const float *record=packed?banks+(size_t)column*(1+(size_t)t*stride):
                                      banks+10+column*AGPoolStride(cfg);
   __global float *stack=work+(size_t)group*(w*t+max(w*t,3*(t+1)));
   __global float *raw=values+(size_t)column*2*w;
   __local int syntax;
   AGFormulaEvaluate(market,record+(packed?1:5),(int)record[0],cfg,stack,stack+w*t,raw,raw+w,stride,&syntax);
  }

//+------------------------------------------------------------------+
//| AQEvictBefore - Order exact ties by candidate and formula ID      |
//+------------------------------------------------------------------+
inline bool AQEvictBefore(const int first,const int second,const int candidate,const int stride,
                          __global const float *weights,__global const float *bank)
  {
   if(first<0)
      return(false);
   if(second<0)
      return(true);
   const float a=fabs(weights[first]);
   const float b=fabs(weights[second]);
   if(a!=b)
      return(a<b);
   if(first==candidate||second==candidate)
      return(first==candidate);
   return(AGPoolIDGreater(bank+11+first*stride,bank+11+second*stride));
  }

//+------------------------------------------------------------------+
//| AQCompactRows - Move overlapping source tiles left in order      |
//+------------------------------------------------------------------+
inline void AQCompactRows(__global float *data,const int width,const int rows)
  {
   const int lane=get_local_id(0);
   const int lanes=get_local_size(0);
   const int elements=width*rows;
   for(int base=0;base<elements;base+=lanes)
     {
      const int index=base+lane;
      const float value=index<elements?data[index+width]:0.0f;
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(index<elements)
         data[index]=value;
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
  }

// Modes 0/2 refresh, 1 commits one candidate. Modes 3/4 reuse a
// confirmed fit on the same logical window, with/without a publication pin.
// fit = confirmed IC, aggregate usable, candidate changed.
__kernel void AQCMPoolOrdered(__global const float *market,__global const float *cfg,
                              __global float *banks,__global float *work,__global float *rings,
                              __global float *control,__global float *fit,
                              __global const float *packed,__global const float *values,
                              int index,int mode)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   __local float reduce[64];
   __local int entries[64];
   __local int shared;
   int w=cfg[AG_CFG_W],t=cfg[AG_CFG_MAX_TOKENS],cap=(int)cfg[AG_CFG_POOL]+1;
   int n=cfg[AG_CFG_IC_WINDOW],h=cfg[AG_CFG_HORIZON],offset=w-h-n;
   int size=AGPoolBankSize(cfg),stride=AGPoolStride(cfg),count=banks[0];
   __global float *scratch=banks+2*size;
   __global float *raw=work+w*t+max(w*t,3*(t+1));
   __global float *valid=raw+w,*normalized=valid+w,*y=normalized+2*w;
   __global float *stats=y+n,*z=stats+7,*matrix=z+cap*n;
   __global float *weights=matrix+cap*(cap+1),*metric=weights+cap;
   int token_stride=4+AG_VOCABULARY,p=count,remove=-1,status=AG_OK;
   if(lane==0)
     {
      control[0]=AG_SKIP;control[1]=0;control[2]=0;control[3]=0;control[4]=0;
      control[6]=count;control[7]=banks[size];fit[2]=0;
      if(mode==2||mode==3)banks[3*size+9]=0;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(mode==3||mode==4)
     {
      if(mode==3)
        {
         if(fit[1]!=1||count<1||banks[9]!=1)return;
         AGPoolCopyGroup(banks,banks+3*size,size);
         int rs=AGPoolRingSize(cfg);
         for(int i=lane;i<rs;i+=lanes)rings[rs+i]=0;
        }
      if(lane==0)control[0]=AG_OK;
      return;
     }
   AGPoolCopyGroup(banks,scratch,size);
   if(mode!=1)
     {
      for(int i=lane;i<n;i+=lanes)
        {
         float a=market[3*w+offset+i],b=market[3*w+offset+i+h];
         raw[i]=b/a-1;
          valid[i]=(a>0&&b>0&&isfinite(a)&&isfinite(b))?
                   (isfinite(raw[i])?AG_OK:AQ_POOL_NUMERIC_ERROR):AG_TECHNICAL;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
       status=AQNormalize(raw,valid,n,cfg,y,stats,lanes,true,reduce);
       const int width=min(8,lanes/2);
       const int columns=lanes/width;
       for(int base=0;base<count&&status==AG_OK;base+=columns)
        {
          const int tile=lane/width;
          const bool active=base+tile<count;
          const int column=min(base+tile,count-1);
          __global const float *x=values+(size_t)column*2*w+offset;
          const int result=AQNormalize(x,x+w,n,cfg,z+column*n,scratch+10+column*stride+5+t,
                                       width,active,reduce);
          if(lane%width==0)
             entries[tile]=result;
          const float bad=active&&result!=AG_OK?(float)tile:(float)columns;
          const int first=(int)AGPoolReduce(bad,reduce,2);
          if(first<columns)
             status=entries[first];
        }
      if(status==AG_INVALID)status=AG_TECHNICAL;
     }
   else
     {
      __global const float *candidate=packed+(size_t)index*(1+(size_t)t*token_stride);
      float length=candidate[0];
      if(!isfinite(length)||length!=floor(length)||length<2||length>t||
         candidate[1+((int)length-1)*token_stride]!=AG_STOP)status=AG_TECHNICAL;
      if(status==AG_OK)
        {
         float duplicate=0;
         for(int j=lane;j<count;j+=lanes)
           {
            __global float *r=scratch+10+j*stride;
            int same=r[0]==length;
            for(int k=0;same&&k<(int)length;k++)same=r[5+k]==candidate[1+k*token_stride];
            duplicate=fmax(duplicate,(float)same);
           }
         if(AGPoolReduce(duplicate,reduce,1)>0)
           {
            if(lane==0){control[0]=AG_OK;control[1]=fit[0];control[2]=1;}
            return;
           }
         __global float *record=scratch+10+count*stride;
         __global const float *x=values+(size_t)index*2*w+offset;
          status=AQNormalize(x,x+w,n,cfg,z+count*n,record+5+t,lanes,true,reduce);
         if(status==AG_OK)
           {
            if(lane==0){record[0]=length;AGPoolCopyValues(scratch+1,record+1,4);}
            for(int k=lane;k<t;k+=lanes)record[5+k]=k<(int)length?candidate[1+k*token_stride]:AG_PAD;
            p++;
            barrier(CLK_GLOBAL_MEM_FENCE);
           }
        }
     }
   if(status==AG_OK&&p>0)
      status=AQRidge(z,y,n,p,-1,cfg,matrix,weights,raw,valid,normalized,stats,metric,reduce);
   if(status==AG_OK&&p==cap)
     {
       int selected=-1;
       for(int column=lane;column<p;column+=lanes)
          if(AQEvictBefore(column,selected,p-1,stride,weights,scratch))
             selected=column;
       entries[lane]=selected;
       barrier(CLK_LOCAL_MEM_FENCE);
       for(int step=lanes/2;step>0;step/=2)
        {
          if(lane<step&&AQEvictBefore(entries[lane+step],entries[lane],p-1,stride,weights,scratch))
             entries[lane]=entries[lane+step];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
       remove=entries[0];
      if(remove==p-1)
        {
         AGPoolCopyGroup(banks,scratch,size);
         if(lane==0){control[0]=AG_OK;control[1]=fit[0];control[2]=1;}
         return;
        }
      p--;
      status=AQRidge(z,y,n,p,remove,cfg,matrix,weights,raw,valid,normalized,stats,metric,reduce);
     }
   if(status==AG_OK&&mode==1)
     {
      if(lane==0)shared=AGPoolIncrement(scratch+1);
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if(!shared)status=AG_TECHNICAL;
     }
   if(status!=AG_OK)
     {
      AGPoolCopyGroup(banks,scratch,size);
      if(lane==0)
        {
         control[0]=status;
         if(mode==1&&status==AG_INVALID){control[1]=cfg[AG_CFG_INVALID_TOKEN_REWARD];control[2]=1;}
        }
      return;
     }
   if(remove>=0)
     {
      AQCompactRows(scratch+10+remove*stride,stride,p-remove);
      AQCompactRows(z+remove*n,n,p-remove);
     }
   for(int j=lane;j<p;j+=lanes)scratch[10+j*stride+12+t]=weights[j];
   if(lane==0){scratch[0]=p;scratch[9]=1;}
   barrier(CLK_GLOBAL_MEM_FENCE);
   AGPoolCopyGroup(scratch,banks,size);
   if(lane==0)
     {
      fit[0]=p>0?metric[1]:0;fit[1]=p>0?metric[2]:0;fit[2]=(mode==1);
      control[0]=AG_OK;control[6]=p;
      if(mode==1){control[1]=fit[0];control[2]=1;}
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   if(mode==2)
     {
      if(fit[1]!=1)
         return;
      AGPoolCopyGroup(scratch,banks+3*size,size);
      int rs=AGPoolRingSize(cfg);
      for(int i=lane;i<rs;i+=lanes)rings[rs+i]=0;
     }
  }

// One group per bar. Validate each weighted factor, retaining the first
// failing factor's status in stable bank order.
__kernel void AQCMPoolScore(__global const float *cfg,__global const float *banks,
                            __global const float *values,__global float *scores,
                            int start,int span,int mode)
  {
   int row=start+get_group_id(0),lane=get_local_id(0),lanes=get_local_size(0);
   __local float reduce[64];
   __global const float *bank=banks+(mode==2?3:1)*AGPoolBankSize(cfg);
   int count=(int)bank[0],t=(int)cfg[AG_CFG_MAX_TOKENS];
   if(!count||bank[9]!=1)
     {
      if(lane==0){scores[row]=0;scores[span+row]=AG_SKIP;}
      return;
     }
   float sum=0,bad=count;
   int stride=AGPoolStride(cfg);
   for(int j=lane;j<count;j+=lanes)
     {
      __global const float *x=values+(size_t)j*span*2+row;
      __global const float *r=bank+10+j*stride;
      if(x[span]!=AG_OK)bad=fmin(bad,(float)j);
      else sum+=r[12+t]*AGPoolAffine(x[0],r+5+t);
     }
   int first=(int)AGPoolReduce(bad,reduce,2);
   sum=AGPoolReduce(sum,reduce,0);
   if(lane==0)
     {
      scores[row]=sum;
      scores[span+row]=first<count?values[((size_t)first*2+1)*span+row]:
                       isfinite(sum)?AG_OK:AQ_POOL_NUMERIC_ERROR;
     }
  }

__kernel void AQCMPoolAdvance(__global const float *cfg,__global float *banks,
                              __global float *rings,__global float *control,
                              __global float *signal,__global float *context,
                              __global float *scores,int start,int span,int mode,int begin,int end)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   __local float reduce[64],shifted[64];
   int rs=AGPoolRingSize(cfg),nw=cfg[AG_CFG_NORM_WINDOW],nc=cfg[AG_CFG_CONTEXT];
   __global float *ring=rings+(mode==2?rs:0);
   if(begin==start)
     {
      if(lane==0)
        {
         scores[2*span]=1;
         control[0]=AG_SKIP;control[4]=0;control[6]=banks[0];control[7]=banks[AGPoolBankSize(cfg)];
          if(mode==3)
             control[5]=0;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      float bad=span;
      for(int i=start+lane;i<span;i+=lanes)
         if(scores[span+i]!=AG_OK)bad=fmin(bad,(float)i);
      int first=(int)AGPoolReduce(bad,reduce,2);
      if(first<span)
        {
         if(lane==0)
           {
             scores[2*span]=0;
            control[0]=scores[span+first];
            if(mode==2)banks[3*AGPoolBankSize(cfg)+9]=0;
           }
         return;
        }
       if(mode==3)
         {
          if(lane==0)
             signal[0]=0;
          for(int i=lane;i<rs;i+=lanes)
             ring[i]=0;
          for(int i=lane;i<nc;i+=lanes)
             context[i]=0;
          barrier(CLK_GLOBAL_MEM_FENCE);
         }
     }
   if(scores[2*span]!=1)return;
   for(int row=begin;row<end;row++)
     {
      float score=scores[row];
      if(mode!=1)
        {
         if(lane==0)
           {
            int pos=(int)ring[1];
            ring[11+pos]=score;
            ring[1]=(pos+1)%nw;
            ring[0]=fmin((float)nw,ring[0]+1);
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         int full=ring[0]>=nw;
         barrier(CLK_GLOBAL_MEM_FENCE);
         if(!full)continue;
         AGPoolStandardizeRingGroup(ring+11,0,nw,cfg,ring+3,reduce);
          if(lane==0)ring[10]=ring[8];
         barrier(CLK_GLOBAL_MEM_FENCE);
          if(ring[10]==AG_TECHNICAL)
            {
             if(lane==0)
               {
                scores[2*span]=-1;
                control[0]=AQ_POOL_NUMERIC_ERROR;
                control[4]=0;
                if(mode==3)
                   control[5]=0;
               }
             return;
            }
        }
      int ready=ring[0]>=nw&&ring[10]==AG_OK;
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(!ready)
        {
         if(lane==0)
           {
            control[0]=AG_SKIP;control[4]=0;
            if(mode>=2)ring[2]=0;
            if(mode==3)control[5]=0;
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         continue;
        }
      float g=AGPoolAffine(score,ring+3),clip=cfg[AG_CFG_SIGNAL_CLIP];
      if(!isfinite(g)||clip<=0)
        {
         if(lane==0)
              {
               scores[2*span]=mode==1&&row==start?0:-1;
                control[0]=isfinite(g)?AG_TECHNICAL:AQ_POOL_NUMERIC_ERROR;
                control[4]=0;
               if(mode==3)control[5]=0;
              }
         return;
        }
      g=clamp(g,-clip,clip)/clip;
      int count=(int)ring[2];
      barrier(CLK_GLOBAL_MEM_FENCE);
      if(count>=nc)
        {
         for(int base=0;base<nc-1;base+=lanes)
           {
            int i=base+lane;
            if(i<nc-1)shifted[lane]=ring[11+nw+i+1];
            barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
            if(i<nc-1)ring[11+nw+i]=shifted[lane];
            barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
           }
         count=nc-1;
        }
      if(lane==0)
        {
         ring[11+nw+count]=g;signal[0]=g;ring[2]=count+1;
         control[0]=AG_OK;control[4]=mode>=2?(count+1>=nc):1;
         if(mode==3)control[5]=control[4];
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      AGPoolCopyGroup(ring+11+nw,context,count+1);
     }
  }

// The same evaluator can use a global stack when the device cannot hold
// the history kernel's minimum local tile (lookback + one new bar).
__kernel void AQCMPoolHistoryGlobal(__global const float *market,__global const float *cfg,
                                    __global const float *banks,__global float *values,
                                    __global float *workspace,int first_formula,int mode,
                                    int start,int span)
  {
   int lane=get_local_id(0),lanes=get_local_size(0),group=get_group_id(0);
   int formula=first_formula+group;
   __global const float *bank=banks+(mode==2?3:1)*AGPoolBankSize(cfg);
   if(formula>=(int)bank[0]||bank[9]!=1)return;
   int w=cfg[AG_CFG_W],t=cfg[AG_CFG_MAX_TOKENS],halo=cfg[AG_CFG_LOOKBACK];
   int stride=w*t+max(w*t,3*(t+1))+7*w;
   __global float *frame=workspace+(size_t)group*stride;
   __global float *raw=frame+5*w,*status=raw+w,*stack=status+w,*valid=stack+w*t;
   __global float *out=values+(size_t)formula*span*2;
   __global const float *record=bank+10+formula*AGPoolStride(cfg);
   __local int syntax;
   for(int first=start;first<span;first+=w-halo)
     {
      int offset=max(0,first-halo),end=min(span,first+w-halo);
      for(int i=lane;i<5*w;i+=lanes)
        {
         int row=offset+i%w;
         frame[i]=row<span?market[(size_t)(i/w)*span+row]:0;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      AGFormulaEvaluate(frame,record+5,(int)record[0],cfg,stack,valid,raw,status,1,&syntax);
      barrier(CLK_GLOBAL_MEM_FENCE);
      for(int i=first+lane;i<end;i+=lanes)
        {
         out[i]=raw[i-offset];out[span+i]=status[i-offset];
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
  }
