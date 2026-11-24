//+------------------------------------------------------------------+
//|                                                      AlphaQCM.cl |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
// One group owns each causal prefix. A token chunk ends before rebinding
// the kernel. Results use AGPoolEvaluateBatch's immutable packed layout.

//+------------------------------------------------------------------+
//| AQCMGridCosines - Materialize the fixed QCM cosine basis once.   |
//+------------------------------------------------------------------+
__kernel void AQCMGridCosines(__global float *basis,int points,int frequencies)
  {
   const int m=get_global_id(0);
   const int k=get_global_id(1);
   basis[k*frequencies+m]=cospi((float)m*((float)k+0.5f)/(float)points);
  }

//+------------------------------------------------------------------+
//| AQCMGridFeatures - Apply the current frozen fraction embedding.  |
//+------------------------------------------------------------------+
__kernel void AQCMGridFeatures(__global const float *basis,
                               __global const float *weights,
                               __global float *phi,int frequencies)
  {
   const int j=get_group_id(0),k=get_group_id(1);
   const int hidden=get_num_groups(0);
   const int lane=get_local_id(0),lanes=get_local_size(0);
   __local float partial[64];
   float sum=0.0f;
   for(int m=lane;m<frequencies;m+=lanes)
      sum+=basis[k*frequencies+m]*weights[j*(frequencies+1)+m];
   partial[lane]=sum;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int span=lanes/2;span>0;span/=2)
     {
      if(lane<span)
         partial[lane]+=partial[lane+span];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   if(lane==0)
     {
      float value=partial[0]+weights[j*(frequencies+1)+frequencies];
      phi[k*hidden+j]=isfinite(value)?fmax(value,0.0f):NAN;
     }
  }

//+------------------------------------------------------------------+
//| AQCMGenDot - Sum the private input tiles of a row.               |
//+------------------------------------------------------------------+
inline float AQCMGenDot(float sum,__local float *partial,int parts)
  {
   int lane=get_local_id(0),base=(lane/parts)*parts;
   partial[lane]=sum;
   barrier(CLK_LOCAL_MEM_FENCE);
   float result=partial[base]+partial[base+1];
   if(parts==4)
      result+=partial[base+2]+partial[base+3];
   // All peers finish reading this tile before reuse.
   barrier(CLK_LOCAL_MEM_FENCE);
   return result;
  }

//+------------------------------------------------------------------+
//| AQCMGenDense - Tiled, shared dense head for a batch of rows.     |
//+------------------------------------------------------------------+
inline void AQCMGenDense(__global const float *input,__global const float *weights,
                         int inputs,int outputs,int batch,int relu,float bias_scale,
                         __global float *output,__local float *partial)
  {
   int lane=get_local_id(0),lanes=get_local_size(0),parts=min(lanes,4);
   int rows=lanes/parts,part=lane%parts;
   for(int start=0;start<batch*outputs;start+=rows)
     {
      int index=start+lane/parts;
      int w=(index%outputs)*(inputs+1);
      float sum=0.0f;
      if(index<batch*outputs)
         for(int d=part;d<inputs;d+=parts)
            sum+=input[(index/outputs)*inputs+d]*weights[w+d];
      sum=AQCMGenDot(sum,partial,parts);
      if(part==0&&index<batch*outputs)
        {
         sum+=bias_scale*weights[w+inputs];
         output[index]=isfinite(sum)?(relu?fmax(0.0f,sum):sum):NAN;
        }
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

//+------------------------------------------------------------------+
//| AQCMGenProject - Project all quantiles onto the scale row.       |
//+------------------------------------------------------------------+
inline void AQCMGenProject(__global const float *hidden_values,__global const float *w,
                           int hidden,int points,__global float *projected,
                           __local float *partial)
  {
   int lane=get_local_id(0),lanes=get_local_size(0),parts=min(lanes,4);
   int rows=lanes/parts,part=lane%parts;
   for(int start=0;start<=hidden;start+=rows)
     {
      int d=start+lane/parts;
      float sum=0.0f;
      if(d<=hidden)
         for(int k=part;k<points;k+=parts)
            sum+=w[k]*(d==hidden?1.0f:hidden_values[k*hidden+d]);
      sum=AQCMGenDot(sum,partial,parts);
      if(part==0&&d<=hidden)
         projected[d]=sum;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

//+------------------------------------------------------------------+
//| AQCMGenEncode - Extend one branch's masked-mean prefix cache.    |
//+------------------------------------------------------------------+
inline void AQCMGenEncode(int token,int t,int embed,int hidden,float position_scale,
                          __global const float *embedding,__global const float *lower,
                          __global const float *upper,__global float *state,
                          __global float *input,__global float *gates,
                          __local float *partial)
  {
   int lane=get_local_id(0),lanes=get_local_size(0);
   for(int j=lane;j<embed;j+=lanes)
     {
      float angle=(float)t/pow(position_scale,(float)(2*(j/2))/(float)embed);
      input[j]=embedding[token*embed+j]+((j&1)?cos(angle):sin(angle));
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
   AGSearchRecur(input,embed,lower,hidden,state,gates,partial);
   AGSearchRecur(state,hidden,upper,hidden,state+2*hidden,gates,partial);
   for(int j=lane;j<hidden;j+=lanes)
      state[4*hidden+j]+=state[2*hidden+j];
   barrier(CLK_GLOBAL_MEM_FENCE);
  }

//+------------------------------------------------------------------+
//| AQCMGenRank - Unbiased ordinal among the legal actions.          |
//+------------------------------------------------------------------+
inline int AQCMGenRank(ulong seed,ulong stream,ulong candidate,int t,uint count)
  {
   uint reject=(0u-count)%count;
   ulong base=(ulong)(uint)t<<32;
   for(uint attempt=0;;attempt++)
     {
      uint word=AQCPhiloxWord(seed,stream,candidate,base|(ulong)attempt);
      if(word>=reject)
         return (int)(word%count);
      if(attempt==0xffffffffu)
         return -1;
     }
  }

//+------------------------------------------------------------------+
//| AQCMGenerate - Generate a bounded chunk with frozen weights.     |
//+------------------------------------------------------------------+
__kernel void AQCMGenerate(__global const float *cfg,
                           __global const float *q_embed,__global const float *q_lower,
                           __global const float *q_upper,__global const float *q_hidden,
                           __global const float *q_out,__global const float *z_embed,
                           __global const float *z_lower,__global const float *z_upper,
                           __global const float *z_hidden,__global const float *z_out,
                           __global const float *phi,__global const float *projection,
                           __global float *packed,__global float *workspace,__global float *status,
                           ulong seed,ulong first_id,int base,int length,int embed,int hidden,
                           int points,int warmup,float epsilon,float bonus,int begin,int end)
  {
   int lane=get_local_id(0),lanes=get_local_size(0),group=get_group_id(0);
   int candidate=base+group;
   ulong id=first_id+(ulong)candidate;
   const int A=AG_VOCABULARY,stride=4+A;
   // Two recurrent caches, shared embed/gates, dense/product/projection scratch,
   // two action rows, and the independent typed stacks used by AGMaskToken.
   int work_stride=15*hidden+embed+1+2*A+2*points*hidden+3*(length+1)*A;
   __global float *q_state=workspace+(size_t)group*work_stride;
   __global float *z_state=q_state+5*hidden;
   __global float *input=z_state+5*hidden;
   __global float *gates=input+embed;
   __global float *projected=gates+4*hidden;
   __global float *q_values=projected+hidden+1;
   __global float *scale=q_values+A;
   __global float *product=scale+A;
   __global float *features=product+points*hidden;
   __global int *mask_work=(__global int *)(features+points*hidden);
   __global float *result=packed+(size_t)candidate*(1+length*stride);

   __local float partial[64],best[128];
   __local int counts[128],winner[128],choice;
   if(begin==0)
     {
      if(lane==0)
        {
         result[0]=0.0f;
         status[candidate]=0.0f;
         choice=AG_BOS;
        }
      if(!warmup)
         for(int d=lane;d<10*hidden;d+=lanes)
            q_state[d]=0.0f;
     }
   else if(lane==0)
     {
      choice=AG_STOP;
      if(result[0]==0.0f&&status[candidate]==0.0f)
        {
         float token=result[1+(begin-1)*stride];
         if(isfinite(token)&&token>=AG_STOP&&token<(float)A&&token==trunc(token))
            choice=(int)token;
         else
            status[candidate]=(float)(4*begin+1);
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
   if(cfg[AG_CFG_MAX_TOKENS]!=(float)length||cfg[AG_CFG_EMBEDDING]!=(float)embed||
      cfg[AG_CFG_SEARCH_HIDDEN]!=(float)hidden||!(cfg[AG_CFG_POSITION_SCALE]>0.0f)||
      !isfinite(cfg[AG_CFG_POSITION_SCALE]))
     {
      if(lane==0)
         status[candidate]=1.0f;
      return;
     }

   for(int t=begin;t<end;t++)
     {
      int token=choice;
      if(token==AG_STOP||token<0)
         break;
      __global float *row=result+1+t*stride;
      for(int a=lane;a<A;a+=lanes)
         row[4+a]=AGMaskToken(result+1,t,stride,cfg,mask_work,a);
      barrier(CLK_GLOBAL_MEM_FENCE);

      if(!warmup)
        {
         AQCMGenEncode(token,t,embed,hidden,cfg[AG_CFG_POSITION_SCALE],q_embed,q_lower,q_upper,
                       q_state,input,gates,partial);
         AQCMGenEncode(token,t,embed,hidden,cfg[AG_CFG_POSITION_SCALE],z_embed,z_lower,z_upper,
                       z_state,input,gates,partial);
         for(int d=lane;d<hidden;d+=lanes)
            product[d]=q_state[4*hidden+d]/(float)(t+1);
         barrier(CLK_GLOBAL_MEM_FENCE);
         AQCMGenDense(product,q_hidden,hidden,hidden,1,1,1.0f,projected,partial);
         AQCMGenDense(projected,q_out,hidden,A,1,0,1.0f,q_values,partial);
         for(int x=lane;x<points*hidden;x+=lanes)
            product[x]=(z_state[4*hidden+x%hidden]/(float)(t+1))*phi[x];
         barrier(CLK_GLOBAL_MEM_FENCE);
         AQCMGenDense(product,z_hidden,hidden,hidden,points,1,1.0f,features,partial);
         AQCMGenProject(features,projection,hidden,points,projected,partial);
         AQCMGenDense(projected,z_out,hidden,A,1,0,projected[hidden],scale,partial);
        }

      int lo=lane*A/lanes,hi=(lane+1)*A/lanes;
      int owner=lanes+lane,action=-1,count=0;
      float top=-INFINITY;
      for(int a=lo;a<hi;a++)
        {
         if(row[4+a]==0.0f)
            continue;
         count++;
         float score=warmup?0.0f:q_values[a]+bonus*fabs(scale[a]);
         if(a<=AG_PAD||!isfinite(score))
           {
            top=NAN;
            continue;
           }
         if(score>top)
           {
            action=a;
            top=score;
           }
        }
      best[owner]=top;
      counts[owner]=count;
      winner[owner]=action;
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int span=lanes/2;span>0;span/=2)
        {
         if(lane<span)
           {
            int node=span+lane,left=node*2,right=left+1;
            int pick=best[right]>best[left]?right:left;
            best[node]=(isnan(best[left])||isnan(best[right]))?NAN:best[pick];
            counts[node]=counts[left]+counts[right];
            winner[node]=winner[pick];
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      if(lane==0)
        {
         choice=-1;
         if(counts[1]==0||row[4+AG_BOS]!=0.0f||row[4+AG_PAD]!=0.0f||
            (t==length-1&&(counts[1]!=1||row[4+AG_STOP]!=1.0f)))
            status[candidate]=(float)(4*t+1);
         else if(!isfinite(best[1]))
            status[candidate]=(float)(4*t+2);
         else
           {
            int selected=winner[1];
            if(warmup||AQCPhiloxUnit(AQCPhiloxWord(seed,AQC_STREAM_EPSILON,id,(ulong)(uint)t<<32))<epsilon)
              {
               int rank=AQCMGenRank(seed,warmup?AQC_STREAM_WARMUP:AQC_STREAM_LEGAL,id,t,(uint)counts[1]);
               selected=-1;
               if(rank>=0)
                 {
                  int node=1;
                  while(node<lanes)
                    {
                     node*=2;
                     if(rank>=counts[node])
                       {
                        rank-=counts[node];
                        node++;
                       }
                    }
                  int first=(node-lanes)*A/lanes,last=(node-lanes+1)*A/lanes;
                  for(int a=first;a<last;a++)
                     if(row[4+a]!=0.0f&&rank--==0)
                       {
                        selected=a;
                        break;
                       }
                 }
              }
            if(selected<AG_STOP||selected>=A||row[4+selected]!=1.0f)
               status[candidate]=(float)(4*t+1);
            else
              {
               choice=selected;
               row[0]=(float)selected;
               if(selected==AG_STOP)
                  result[0]=(float)(t+1);
              }
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
     }
  }
