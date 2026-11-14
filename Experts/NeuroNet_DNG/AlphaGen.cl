//+------------------------------------------------------------------+
//| AlphaGen.cl                                                      |
//+------------------------------------------------------------------+
//--- a typed stack admits DeltaT only at its top. Once covered by Expr,
//--- deltaT could never be consumed by this grammar. This invariant makes
//--- exact completion possible without an exponential prefix search.
//+------------------------------------------------------------------+
//| AGApplyToken - Implement AGApplyToken                            |
//+------------------------------------------------------------------+
inline int AGApplyToken(const int token,__global const float*cfg,
                        __global int *types,__global int*lookback,
                        __global int *dependent,int depth)
  {
   const int limit=(int)cfg[AG_CFG_LOOKBACK];

   if(token>=AG_OPEN&&token<=AG_VOLUME)
     {
      if(depth>0&&types[depth-1]!=0)
         return(-1);

      types[depth]=0;
      lookback[depth]=0;
      dependent[depth]=1;

      return(depth+1);
     }

   if(token>=AG_CONSTANT_FIRST&&token<AG_INTERVAL_FIRST)
     {
      if(depth>0&&types[depth-1]!=0)
         return(-1);

      types[depth]=0;
      lookback[depth]=0;
      dependent[depth]=0;

      return(depth+1);
     }

   if(token>=AG_INTERVAL_FIRST&&token<AG_ABS)
     {
      if(depth<1||types[depth-1]!=0)
         return(-1);

      int n=(int)cfg[AG_CFG_INTERVALS+token-AG_INTERVAL_FIRST];

      if(n<=0||lookback[depth-1]+n-1>limit)
         return(-1);

      types[depth]=n;
      lookback[depth]=0;
      dependent[depth]=0;

      return(depth+1);
     }

   if(token>=AG_ABS&&token<=AG_LOG)
      return(depth>0&&types[depth-1]==0?depth:-1);

   if(token>=AG_ADD&&token<=AG_DIV)
     {
      if(depth<2||types[depth-1]!=0||types[depth-2]!=0)
         return(-1);

      lookback[depth-2]=max(lookback[depth-1],lookback[depth-2]);
      dependent[depth-2]|=dependent[depth-1];

      return(depth-1);
     }

   if(token>=AG_REF&&token<=AG_MAX)
     {
      if(depth<2||types[depth-1]<=0||types[depth-2]!=0)
         return(-1);

      int n=types[depth-1];
      int lb=lookback[depth-2]+n-((token==AG_REF||token==AG_DELTA)?0:1);

      if(lb>limit)
         return(-1);

      lookback[depth-2]=lb;

      return(depth-1);
     }

   if(token==AG_CORR||token==AG_COV)
     {
      if(depth<3||types[depth-1]<=0||types[depth-2]!=0||types[depth-3]!=0)
         return(-1);

      int lb=max(lookback[depth-2],lookback[depth-3])+types[depth-1]-1;

      if(lb>limit)
         return(-1);

      lookback[depth-3]=lb;
      dependent[depth-3]|=dependent[depth-2];

      return(depth-2);
     }

   return(-1);
  }
//+------------------------------------------------------------------+
//| AGMaskToken - Build each candidate mask from the same immutable  |
//+------------------------------------------------------------------+
inline float AGMaskToken(__global const float*tokens,const int length,const int stride,
                         __global const float *cfg,__global int*workspace,const int candidate)
  {
   const int capacity=(int)cfg[AG_CFG_MAX_TOKENS]+1;

   __global int *types=workspace+candidate*capacity*3;
   __global int *lookback=types+capacity;
   __global int *dependent=lookback+capacity;
   int          depth=0;

   if(length<0||length>=(int)cfg[AG_CFG_MAX_TOKENS])
      return(0.0f);

   for(int i=0;i<length;i++)
     {
      depth=AGApplyToken((int)tokens[i*stride],cfg,types,lookback,dependent,depth);

      if(depth<0)
         return(0.0f);
     }

   if(candidate==AG_STOP)
     {
      return(depth==1&&types[0]==0&&dependent[0])?1.0f:0.0f;

     }
   depth=AGApplyToken(candidate,cfg,types,lookback,dependent,depth);

   if(depth<0)
      return(0.0f);

   int remaining=(int)cfg[AG_CFG_MAX_TOKENS]-length-1;
   int has_market=0;

   for(int i=0;i<depth;i++)
      has_market|=dependent[i];

//--- d Expr -> d-1 binary operations + STOP. Top Delta adds one rolling.
   int required=depth+(has_market?0:2);

   if(depth>0&&types[depth-1]>0)
     {
      //--- corr/Cov can consume two Exprs and the interval in one step.
      //--- otherwise a one-Expr rolling operator precedes binary reduction.
      int pair=depth>=3&&types[depth-2]==0&&types[depth-3]==0&&
               max(lookback[depth-2],lookback[depth-3])+types[depth-1]-1<=(int)cfg[AG_CFG_LOOKBACK];
      required=depth-(pair?1:0)+(has_market?0:2);
     }

   return(remaining>=required?1.0f:0.0f);
  }
//+------------------------------------------------------------------+
//| AGMask - AGMask -                                                |
//+------------------------------------------------------------------+
__kernel void AGMask(__global const float*tokens,const int length,
                     __global const float *cfg,__global float*mask,
                     __global int         *workspace)
  {
   int candidate=get_global_id(0);

   if(candidate<AG_VOCABULARY)
      mask[candidate]=AGMaskToken(tokens,length,1,cfg,workspace,candidate);
  }
//+------------------------------------------------------------------+
//| AGFormulaEvaluate - Evaluate a complete RPN expression over...   |
//+------------------------------------------------------------------+
inline void AGFormulaEvaluate(__global const float*market,
                              __global const float *tokens,const int length,
                              __global const float *cfg,__global float*stack,
                              __global float       *validity,__global float*result,
                              __global float       *status,const int token_stride,__local int*syntax_state)
  {
   int lane=get_local_id(0);
   int lanes=get_local_size(0);
   int w=(int)cfg[AG_CFG_W];
   int depth=0;
   int capacity=(int)cfg[AG_CFG_MAX_TOKENS]+1;

   if(lane==0)
     {
      __global int *types=(__global int*)validity;
      __global int *lookback=types+capacity;
      __global int *dependent=lookback+capacity;
      (*syntax_state)=length>=2&&length<capacity;

      for(int k=0;k<length&&(*syntax_state);k++)
        {
         float raw_token=tokens[k*token_stride];

         if(!isfinite(raw_token)||raw_token<0.0f||raw_token>=(float)AG_VOCABULARY||
            raw_token!=floor(raw_token))
           {
            (*syntax_state)=0;
            break;
           }
         int token=(int)raw_token;

         if(k==length-1)
            (*syntax_state)=token==AG_STOP&&depth==1&&types[0]==0&&dependent[0];
         else
           {
            depth=AGApplyToken(token,cfg,types,lookback,dependent,depth);

            if(depth<0)
               (*syntax_state)=0;
           }
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

   if(!(*syntax_state))
     {
      for(int i=lane;i<w;i+=lanes)
        {
         result[i]=0.0f;
         status[i]=AG_TECHNICAL;
        }
      return;
     }
   depth=0;

   for(int k=0;k<length;k++)
     {
      int token=(int)tokens[k*token_stride];

      if(token==AG_STOP)
         break;

      if(token>=AG_OPEN&&token<AG_ABS)
        {
         for(int i=lane;i<w;i+=lanes)
           {
            float x=token<=AG_VOLUME?market[(token-AG_OPEN)*w+i]:
                    token<AG_INTERVAL_FIRST?cfg[AG_CFG_CONSTANTS+token-AG_CONSTANT_FIRST]:
                    cfg[AG_CFG_INTERVALS+token-AG_INTERVAL_FIRST];
            stack[depth*w+i]=x;
            validity[depth*w+i]=isfinite(x)?AG_OK:AG_INVALID;
           }
         depth++;
         barrier(CLK_GLOBAL_MEM_FENCE);
         continue;
        }

      if(token>=AG_ABS&&token<=AG_LOG)
        {
         for(int i=lane;i<w;i+=lanes)
           {
            int   p=(depth-1)*w+i;
            float x=stack[p];

            if(validity[p]!=AG_OK)
               continue;

            if(token==AG_LOG&&!(x>0.0f))
               validity[p]=AG_INVALID;
            else
               stack[p]=token==AG_ABS?fabs(x):token==AG_SIGN?sign(x):log(x);

            if(!isfinite(stack[p]))
               validity[p]=AG_INVALID;
           }
         barrier(CLK_GLOBAL_MEM_FENCE);
         continue;
        }

      if(token>=AG_ADD&&token<=AG_DIV)
        {
         for(int i=lane;i<w;i+=lanes)
           {
            int   l=(depth-2)*w+i;
            int   r=(depth-1)*w+i;
            int   s=max((int)validity[l],(int)validity[r]);
            float a=stack[l];
            float b=stack[r];
            float x=0.0f;

            if(s==AG_OK)
              {
               if(token==AG_DIV&&b==0.0f)
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
         barrier(CLK_GLOBAL_MEM_FENCE);
         continue;
        }
      int pair=(token==AG_CORR||token==AG_COV);
      int slot=depth-(pair?3:2);
      int n=(int)stack[(depth-1)*w];

      //--- reuse the interval row after every lane has read its value.
      int staged=(depth-1)*w;

      barrier(CLK_GLOBAL_MEM_FENCE);

      for(int i=lane;i<w;i+=lanes)
        {
         int   begin=i-n+((token==AG_REF||token==AG_DELTA)?0:1);
         int   p=slot*w+i;
         int   s=AG_OK;
         float x=0.0f;

         if(begin<0)
           {
            validity[staged+i]=AG_SKIP;
            stack[staged+i]=x;
            continue;
           }

         if(token==AG_REF||token==AG_DELTA)
           {
            s=(int)validity[slot*w+begin];

            if(token==AG_DELTA)
               s=max(s,(int)validity[p]);
            x=token==AG_REF?stack[slot*w+begin]:stack[p]-stack[slot*w+begin];
           }
         else
           {
            float a=0.0f;
            float b=0.0f;
            float mx=0.0f;
            float my=0.0f;
            float xx=0.0f;
            float yy=0.0f;
            float xy=0.0f;
            float lo=stack[slot*w+begin];
            float hi=lo;

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
                           if(xx>0.0f&&yy>0.0f)
                              x=xy/sqrt(xx)/sqrt(yy);
                           else
                              s=max(s,AG_SKIP);
                 }
           }

         if(s==AG_OK&&!isfinite(x))
            s=AG_INVALID;
         stack[staged+i]=x;
         validity[staged+i]=(float)s;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);

      for(int i=lane;i<w;i+=lanes)
        {
         stack[slot*w+i]=stack[staged+i];
         validity[slot*w+i]=validity[staged+i];
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      depth-=pair?2:1;
     }

   for(int i=lane;i<w;i+=lanes)
     {
      result[i]=stack[i];
      status[i]=validity[i];
     }
  }
//+------------------------------------------------------------------+
//| AGEvaluate - AGEvaluate -                                        |
//+------------------------------------------------------------------+
__kernel void AGEvaluate(__global const float*market,
                         __global const float *tokens,const int length,
                         __global const float *cfg,__global float*stack,
                         __global float       *validity,__global float*result,
                         __global float       *status)
  {
   __local int syntax;

   AGFormulaEvaluate(market,tokens,length,cfg,stack,validity,result,status,1,&syntax);
  }
//+------------------------------------------------------------------+
//| AGStandardize - Stable affine FP32 fit. Small raw amplitude is   |
//+------------------------------------------------------------------+
__kernel void AGStandardize(__global const float*input,
                            __global const float *validity,
                            const int            offset,const int count,
                            __global const float *cfg,
                            __global float       *normalized,
                            __global float       *statistics,const int stats_offset)
  {
   if(get_global_id(0)!=0)
      return;

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

   float origin=input[offset]/a;
   float b=0.0f;

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

   if(!isfinite(variance)||variance<=cfg[AG_CFG_EPS_VAR])
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
//| AGRidge - Ridge fit on standardized columns/target using...      |
//+------------------------------------------------------------------+
__kernel void AGRidge(__global const float*z,__global const float*target,
                      const int            count,const int columns,
                      __global const float *cfg,__global float*scratch,
                      __global float       *weights,__global float*metrics)
  {
   if(get_global_id(0)!=0)
      return;

   metrics[0]=AG_TECHNICAL;
   metrics[1]=0.0f;
   metrics[2]=0.0f;

   if(count<1||columns<1)
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
            if(!(r>0.0f)||!isfinite(r))
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

//--- normalize the aggregate safely before its degeneracy decision.
   float scale=0.0f;
   float origin=0.0f;
   float diff_scale=0.0f;

   for(int t=0;t<count;t++)
     {
      float score=0.0f;

      for(int j=0;j<columns;j++)
         score+=weights[j]*z[j*count+t];
      scale=fmax(scale,fabs(score));

      if(t==0)
         origin=score;
     }

   if(!(scale>0.0f))
     {
      metrics[0]=AG_OK;
      return;
     }
   origin/=scale;

   for(int t=0;t<count;t++)
     {
      float score=0.0f;

      for(int j=0;j<columns;j++)
         score+=weights[j]*z[j*count+t];
      diff_scale=fmax(diff_scale,fabs(score/scale-origin));
     }

   if(!(diff_scale>0.0f))
     {
      metrics[0]=AG_OK;
      return;
     }
   float mean=0.0f;

   for(int t=0;t<count;t++)
     {
      float score=0.0f;

      for(int j=0;j<columns;j++)
         score+=weights[j]*z[j*count+t];
      mean+=(score/scale-origin)/diff_scale/(float)count;
     }
   float covariance=0.0f;
   float variance=0.0f;

   for(int t=0;t<count;t++)
     {
      float score=0.0f;

      for(int j=0;j<columns;j++)
         score+=weights[j]*z[j*count+t];
      score=(score/scale-origin)/diff_scale-mean;
      covariance+=score*target[t]/(float)count;
      variance+=score*score/(float)count;
     }
   float ic=variance>cfg[AG_CFG_EPS_VAR]?covariance/sqrt(variance):0.0f;

   if(!isfinite(ic))
      return;

   metrics[0]=AG_OK;
   metrics[1]=ic;
   metrics[2]=variance>cfg[AG_CFG_EPS_VAR]?1.0f:0.0f;
  }
//+------------------------------------------------------------------+
//| AGGAE - Duration-aware GAE, with immutable value targets across  |
//+------------------------------------------------------------------+
__kernel void AGGAE(__global float*rollout,const int count,
                    __global const float *cfg,const int token_policy)
  {
   if(get_global_id(0)!=0||count<1)
      return;

   for(int i=0;i<count;i++)
      if(rollout[i*AG_GAE_FIELDS+AG_GAE_READY]!=1.0f)
        {
         for(int j=0;j<count;j++)
           {
            rollout[j*AG_GAE_FIELDS+AG_GAE_ADVANTAGE]=NAN;
            rollout[j*AG_GAE_FIELDS+AG_GAE_RETURN]=NAN;
            rollout[j*AG_GAE_FIELDS+AG_GAE_ACTOR]=NAN;
           }
         return;
        }
   float gamma=cfg[token_policy?AG_CFG_TOKEN_GAMMA:AG_CFG_TRADE_GAMMA];
   float carry=0.0f;
   float mean=0.0f;

   for(int i=count-1;i>=0;i--)
     {
      __global float *r=rollout+i*AG_GAE_FIELDS;

      if(r[AG_GAE_READY]!=1.0f)
        {
         carry=0.0f;
         continue;
        }
      float discount=pow(gamma,r[AG_GAE_DELTA]);
      float delta=r[AG_GAE_REWARD]+discount*r[AG_GAE_BOOTSTRAP]*r[AG_GAE_NEXT]-r[AG_GAE_VALUE];

      carry=delta+discount*cfg[AG_CFG_LAMBDA]*r[AG_GAE_CONTINUE]*carry;
      r[AG_GAE_ADVANTAGE]=carry;
      r[AG_GAE_RETURN]=carry+r[AG_GAE_VALUE];
      mean+=carry/(float)count;
     }
   float variance=0.0f;

   for(int i=0;i<count;i++)
     {
      float d=rollout[i*AG_GAE_FIELDS+AG_GAE_ADVANTAGE]-mean;

      variance+=d*d/(float)count;
     }

   for(int i=0;i<count;i++)
     {
      __global float *r=rollout+i*AG_GAE_FIELDS;
      r[AG_GAE_ACTOR]=count<2||variance<=cfg[AG_CFG_EPS_VAR]?r[AG_GAE_ADVANTAGE]:
                      (r[AG_GAE_ADVANTAGE]-mean)/(sqrt(variance)+sqrt(cfg[AG_CFG_EPS_VAR]));
     }
  }
//+------------------------------------------------------------------+
//| AGEmbedding - Embedding lookup: independent sparse rows, then    |
//+------------------------------------------------------------------+
__kernel void AGEmbedding(__global const float*tokens,const int length,
                          __global const float *weights,const int dimension,
                          __global float       *output,const float position_scale)
  {
   int j=get_global_id(0);
   int t=get_global_id(1);

   if(t>=length||j>=dimension)
      return;

   float raw_token=tokens[t];

   if(!isfinite(raw_token)||raw_token<0.0f||raw_token>=(float)AG_VOCABULARY||
      raw_token!=floor(raw_token))
     {
      output[t*dimension+j]=NAN;
      return;
     }
   int token=(int)raw_token;

   if(token==AG_PAD)
     {
      output[t*dimension+j]=0.0f;
      return;
     }
   float angle=(float)t/pow(position_scale,(float)(2*(j/2))/(float)dimension);

   output[t*dimension+j]=weights[token*dimension+j]+((j%2)==0?sin(angle):cos(angle));
  }
//+------------------------------------------------------------------+
//| AGEmbeddingBackward - Sum only the token occurrences of this     |
//+------------------------------------------------------------------+
__kernel void AGEmbeddingBackward(__global const float*tokens,const int length,
                                  __global const float *gradient,const int dimension,
                                  __global float       *weights_gradient)
  {
   int j=get_global_id(0);
   int token=get_global_id(1);

   if(j>=dimension||token>=AG_VOCABULARY)
      return;

   float sum=0.0f;

   for(int t=0;t<length;t++)
     {
      float raw_token=tokens[t];

      if(!isfinite(raw_token)||raw_token<0.0f||raw_token>=(float)AG_VOCABULARY||
         raw_token!=floor(raw_token))
        {
         weights_gradient[token*dimension+j]=NAN;
         return;
        }

      if(token!=AG_PAD&&(int)raw_token==token)
         sum+=gradient[t*dimension+j];
     }
   weights_gradient[token*dimension+j]=sum;
  }
//+------------------------------------------------------------------+
//| AGLSTMForward - Complete recurrent tape, with a constant...      |
//+------------------------------------------------------------------+
__kernel void AGLSTMForward(__global const float*input,const int length,
                            const int            inputs,const int hidden,
                            __global const float *weights,
                            __global const float *anchor,
                            __global float       *tape,__global float*output,
                            __local float        *state)
  {
   int lane=get_local_id(0);
   int lanes=get_local_size(0);

//--- dimension zero holds a single work group per sequence.
//--- dimension one selects its packed input, anchor, tape and output.
   const size_t sequence=get_group_id(1);

   input+=sequence*(size_t)length*inputs;
   anchor+=sequence*(size_t)hidden*2;
   tape+=sequence*(size_t)length*hidden*6;
   output+=sequence*(size_t)length*hidden;
   int           width=inputs+hidden+1;
   __local float *partial=state+6*hidden;

   //--- keep recurrent state and gates within this work group.
   for(int j=lane;j<2*hidden;j+=lanes)
      state[j]=anchor[j];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int t=0;t<length;t++)
     {
      int base=t*hidden*6;

      //--- accumulate each row partition before one local reduction.
      for(int index=lane;index<16*hidden;index+=lanes)
        {
         int   row=index/4;
         int   part=index%4;
         float sum=0.0f;

         int k=part;

         for(;k<inputs;k+=4)
            sum+=weights[row*width+k]*input[t*inputs+k];

         for(;k<width-1;k+=4)
            sum+=weights[row*width+k]*state[k-inputs];
         partial[index]=sum;
        }
      barrier(CLK_LOCAL_MEM_FENCE);

      for(int row=lane;row<4*hidden;row+=lanes)
        {
         int   index=row*4;
         float a=weights[row*width+width-1]+
                 ((partial[index]+partial[index+1])+(partial[index+2]+partial[index+3]));
         float gate=row/hidden==3?tanh(a):1.0f/(1.0f+exp(-a));

         state[2*hidden+row]=gate;
         tape[base+2*hidden+row]=gate;
        }
      barrier(CLK_LOCAL_MEM_FENCE);

      for(int j=lane;j<hidden;j+=lanes)
        {
         float previous=state[hidden+j];
         float cell=state[hidden*2+j]*previous+state[hidden*3+j]*state[hidden*5+j];
         float h=state[hidden*4+j]*tanh(cell);

         state[hidden+j]=cell;
         state[j]=h;
         tape[base+hidden+j]=cell;
         tape[base+j]=h;
         output[t*hidden+j]=h;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
  }
//+------------------------------------------------------------------+
//| AGLSTMBackward - Compute recurrent sequence gradients            |
//+------------------------------------------------------------------+
__kernel void AGLSTMBackward(__global const float*input,const int length,
                             const int            inputs,const int hidden,
                             __global const float *weights,
                             __global const float *anchor,
                             __global const float *tape,
                             __global const float *gradient,
                             __global float       *input_gradient,
                             __global float       *gate_gradient,
                             __local float        *scratch,const int tile_columns)
  {
   int           lane=get_local_id(0);
   int           lanes=get_local_size(0);
   int           width=inputs+hidden+1;
   __local float *partial=scratch+6*hidden;

   for(int i=lane;i<hidden*6;i+=lanes)
      scratch[i]=0.0f;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int t=length-1;t>=0;t--)
     {
      int base=t*hidden*6;

      for(int j=lane;j<hidden;j+=lanes)
        {
         float dh=gradient[t*hidden+j]+scratch[j];
         float cell=tape[base+hidden+j];
         float tc=tanh(cell);
         float f=tape[base+2*hidden+j];
         float i=tape[base+3*hidden+j];
         float o=tape[base+4*hidden+j];
         float g=tape[base+5*hidden+j];
         float previous=t>0?tape[(t-1)*hidden*6+hidden+j]:anchor[hidden+j];
         float dc=scratch[hidden+j]+dh*o*(1.0f-tc*tc);

         scratch[2*hidden+j]=dc*previous*f*(1.0f-f);
         scratch[3*hidden+j]=dc*g*i*(1.0f-i);
         scratch[4*hidden+j]=dh*tc*o*(1.0f-o);
         scratch[5*hidden+j]=dc*i*(1.0f-g*g);
         scratch[hidden+j]=dc*f;
        }
      barrier(CLK_LOCAL_MEM_FENCE);

      //--- contiguous columns cooperate within one local tile.
      for(int start=0;start<inputs+hidden;start+=tile_columns)
        {
         int columns=min(tile_columns,inputs+hidden-start);

         for(int index=lane;index<4*columns;index+=lanes)
           {
            int   k=start+index%columns;
            int   part=index/columns;
            float sum=0.0f;

            for(int row=part;row<4*hidden;row+=4)
               sum+=scratch[2*hidden+row]*weights[row*width+k];
            partial[index]=sum;
           }
         barrier(CLK_LOCAL_MEM_FENCE);

         for(int column=lane;column<columns;column+=lanes)
           {
            int   k=start+column;
            float d=(partial[column]+partial[column+columns])+
                    (partial[column+2*columns]+partial[column+3*columns]);

            if(k<inputs)
               input_gradient[t*inputs+k]=d;
            else
               scratch[k-inputs]=d;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }

      for(int row=lane;row<4*hidden;row+=lanes)
         gate_gradient[(size_t)t*4*hidden+row]=scratch[2*hidden+row];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
  }
//+------------------------------------------------------------------+
//| AGLSTMWeightGradient - Compute independent weight gradients      |
//+------------------------------------------------------------------+
__kernel void AGLSTMWeightGradient(__global const float*input,const int length,
                                   const int            inputs,const int hidden,
                                   __global const float *anchor,__global const float*tape,
                                   __global const float *gate_gradient,__global float*weights_gradient)
  {
   const size_t index=get_global_id(0);
   const int    width=inputs+hidden+1;

   if(index>=(size_t)width*hidden*4)
      return;

   const int k=index%width;
   const int row=index/width;
   float     sum=0.0f;

   for(int t=length-1;t>=0;t--)
     {
      float x=k<inputs?input[(size_t)t*inputs+k]:k==width-1?1.0f:
              t>0?tape[(size_t)(t-1)*hidden*6+k-inputs]:anchor[k-inputs];
      sum+=gate_gradient[(size_t)t*hidden*4+row]*x;
     }
   weights_gradient[index]=sum;
  }
//+------------------------------------------------------------------+
//| AGSequence - Pool a true prefix, append account, or distribute   |
//+------------------------------------------------------------------+
__kernel void AGSequence(__global const float*input,__global float*output,
                         const int            length,const int hidden,const int mode,
                         __global const float *account)
  {
   int j=get_global_id(0);

   if(mode==0&&j<hidden)
     {
      float sum=0.0f;

      for(int t=0;t<length;t++)
         sum+=input[t*hidden+j]/(float)length;
      output[j]=sum;
     }
   else
      if(mode==1&&j<length*hidden)
         output[j]=input[j%hidden]/(float)length;
      else
         if(mode==2&&j<hidden+AG_ACCOUNT)
            output[j]=j<hidden?input[(length-1)*hidden+j]:account[j-hidden];
         else
            if(mode==3&&j<length*hidden)
               output[j]=j/hidden==length-1?input[j%hidden]:0.0f;
            else
               if(mode==4&&j<hidden*2)
                  output[j]=input[(length-1)*hidden*6+j];
  }
//+------------------------------------------------------------------+
//| AGDistribution - Sampling, log-density, entropy and clipped...   |
//+------------------------------------------------------------------+
__kernel void AGDistribution(__global const float*head,
                             __global const float *mask,
                             __global const float *noise,
                             __global const float *cfg,
                             const int            categorical,const int mode,
                             __global float       *sample,__global float*action,
                             __global float       *metadata,__global float*gradient)
  {
   if(get_global_id(0)!=0)
      return;

   float logp=0.0f;
   float entropy=0.0f;
   int   heads=categorical?AG_VOCABULARY:AG_ACTIONS*2;
   float value=head[heads];

   if(categorical)
     {
      float maximum=-INFINITY;
      float total=0.0f;

      for(int j=0;j<AG_VOCABULARY;j++)
         if(mask[j]>0.0f)
            maximum=fmax(maximum,head[j]);

      for(int j=0;j<AG_VOCABULARY;j++)
         if(mask[j]>0.0f)
            total+=exp(head[j]-maximum);

      if(!(total>0.0f)||!isfinite(total))
        {
         metadata[6]=NAN;
         return;
        }
      float normalizer=maximum+log(total);
      float cumulative=0.0f;
      int   choice=-1;
      int   last=-1;

      for(int j=0;j<AG_VOCABULARY;j++)
         if(mask[j]>0.0f)
           {
            float lp=head[j]-normalizer;
            float p=exp(lp);

            entropy-=p*lp;
            cumulative+=p;
            last=j;

            if(choice<0&&noise[0]<cumulative)
               choice=j;
           }

      if(mode!=2)
         sample[0]=(float)(choice<0?last:choice);
      int selected=(int)sample[0];

      if(selected<0||selected>=AG_VOCABULARY||mask[selected]<=0.0f)
        {
         metadata[6]=NAN;
         return;
        }
      logp=head[selected]-normalizer;

      if(mode==2)
        {
         float ratio=exp(logp-metadata[3]);
         float adv=metadata[4];
         float clipped=clamp(ratio,1.0f-cfg[AG_CFG_PPO_CLIP],1.0f+cfg[AG_CFG_PPO_CLIP]);
         float score=(ratio*adv<=clipped*adv)?ratio*adv:0.0f;

         for(int j=0;j<AG_VOCABULARY;j++)
           {
            float lp=head[j]-normalizer;
            float p=mask[j]>0.0f?exp(lp):0.0f;

            gradient[j]=mask[j]>0.0f?score*((j==selected?1.0f:0.0f)-p)-cfg[AG_CFG_ENTROPY_COEF]*p*(lp+entropy):0.0f;
           }
        }
     }
   else
     {
      for(int j=0;j<AG_ACTIONS;j++)
        {
         float mu=head[j];
         float xi=head[AG_ACTIONS+j];
         float sigma=fmax(xi,0.0f)+log1p(exp(-fabs(xi)))+cfg[AG_CFG_SIGMA_FLOOR];

         if(mode!=2)
            sample[j]=mode==1?mu:mu+sigma*noise[j];
         float u=sample[j];
         float standard=(u-mu)/sigma;

         action[j]=u>=0.0f?1.0f/(1.0f+exp(-u)):exp(u)/(1.0f+exp(u));
         logp-=0.5f*(standard*standard+2.0f*log(sigma)+log(2.0f*M_PI_F));
         entropy+=0.5f*(log(2.0f*M_PI_F*M_E_F)+2.0f*log(sigma));
        }

      if(mode==2)
        {
         float ratio=exp(logp-metadata[3]);
         float adv=metadata[4];
         float clipped=clamp(ratio,1.0f-cfg[AG_CFG_PPO_CLIP],1.0f+cfg[AG_CFG_PPO_CLIP]);
         float score=ratio*adv<=clipped*adv?ratio*adv:0.0f;

         for(int j=0;j<AG_ACTIONS;j++)
           {
            float mu=head[j];
            float xi=head[AG_ACTIONS+j];
            float sigma=fmax(xi,0.0f)+log1p(exp(-fabs(xi)))+cfg[AG_CFG_SIGMA_FLOOR];
            float d=sample[j]-mu;
            float dxi=1.0f/(1.0f+exp(-xi));

            gradient[j]=score*d/(sigma*sigma);
            gradient[AG_ACTIONS+j]=(score*((d/sigma)*(d/sigma)-1.0f)+cfg[AG_CFG_ENTROPY_COEF])*dxi/sigma;
           }
        }
     }
   metadata[0]=logp;
   metadata[1]=entropy;
   metadata[2]=value;

   if(mode==2)
     {
      float ratio=exp(logp-metadata[3]);
      float adv=metadata[4];
      float error=value-metadata[5];
      float surrogate=fmin(ratio*adv,clamp(ratio,1.0f-cfg[AG_CFG_PPO_CLIP],1.0f+cfg[AG_CFG_PPO_CLIP])*adv);

      metadata[6]=-surrogate+cfg[AG_CFG_VALUE_COEF]*error*error-cfg[AG_CFG_ENTROPY_COEF]*entropy;
      gradient[heads]=-2.0f*cfg[AG_CFG_VALUE_COEF]*error;
     }
   else
      metadata[6]=0.0f;
  }
//+------------------------------------------------------------------+
//| AGNumeric - Validate numeric values within groups                |
//+------------------------------------------------------------------+
__kernel void AGNumeric(__global const float*input,const int count,
                        __global float *status,const int reset,
                        __global const float*b,const int count_b,
                        __global const float*c,const int count_c,
                        __global const float*d,const int count_d)
  {
   const int   lane=get_local_id(0);
   const int   lanes=get_local_size(0);
   const int   group=get_group_id(0);
   __local int invalid[64];

   if(lanes>64||get_num_groups(0)!=AG_NUMERIC_GROUPS)
      return;

   int bad=0;

   for(size_t i=get_global_id(0);i<(size_t)count;i+=get_global_size(0))
      bad|=!isfinite(input[i]);
   for(size_t i=get_global_id(0);i<(size_t)count_b;i+=get_global_size(0))
      bad|=!isfinite(b[i]);
   for(size_t i=get_global_id(0);i<(size_t)count_c;i+=get_global_size(0))
      bad|=!isfinite(c[i]);
   for(size_t i=get_global_id(0);i<(size_t)count_d;i+=get_global_size(0))
      bad|=!isfinite(d[i]);
   invalid[lane]=bad;
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int active=lanes;active>1;active=(active+1)/2)
     {
      int split=(active+1)/2;

      if(lane<active-split)
         invalid[lane]|=invalid[lane+split];
      barrier(CLK_LOCAL_MEM_FENCE);
     }

   if(lane==0)
      status[2+group]=(invalid[0]||(!reset&&status[2+group]!=AG_OK))?AG_INVALID:AG_OK;
  }
//+------------------------------------------------------------------+
//| AGNumericFinish - Reduce persistent numeric flags                |
//+------------------------------------------------------------------+
__kernel void AGNumericFinish(__global float*status)
  {
   if(get_global_id(0)!=0)
      return;

   int bad=0;

   for(int i=0;i<AG_NUMERIC_GROUPS;i++)
      bad|=(status[2+i]!=AG_OK);
   status[0]=bad?AG_INVALID:AG_OK;
  }
//+------------------------------------------------------------------+
//| AGReplayPrepare - Copy replay data and overlay GAE               |
//+------------------------------------------------------------------+
__kernel void AGReplayPrepare(__global const float*metadata,__global const float*gae,
                              __global float       *meta,const int row,
                              __global const float *a,__global float*out_a,const int stride_a,const int count_a,
                              __global const float *b,__global float*out_b,const int count_b,
                              __global const float *c,__global float*out_c,const int count_c,
                              __global const float *d,__global float*out_d,const int count_d)
  {
   const int i=get_global_id(0);

   if(i<7)
      meta[i]=i==4?gae[(size_t)row*AG_GAE_FIELDS+AG_GAE_ACTOR]:
              (i==5?gae[(size_t)row*AG_GAE_FIELDS+AG_GAE_RETURN]:metadata[(size_t)row*7+i]);

   if(i<count_a)
      out_a[i]=a[(size_t)row*stride_a+i];

   if(i<count_b)
      out_b[i]=b[(size_t)row*count_b+i];

   if(i<count_c)
      out_c[i]=c[(size_t)row*count_c+i];

   if(i<count_d)
      out_d[i]=d[(size_t)row*count_d+i];
  }
//+------------------------------------------------------------------+
//| AGDenseGradient - Materialize the native dense outer product     |
//+------------------------------------------------------------------+
__kernel void AGDenseGradient(__global const float*input,
                              __global const float *gradient,
                              const int            inputs,const int outputs,
                              __global float       *weights_gradient)
  {
   int i=get_global_id(0);
   int j=get_global_id(1);

   if(i>inputs||j>=outputs)
      return;

   weights_gradient[j*(inputs+1)+i]=gradient[j]*(i<inputs?input[i]:1.0f);
  }
//+------------------------------------------------------------------+
//| AGReLU - Strict ReLU on hidden projections, and its saved-output |
//+------------------------------------------------------------------+
__kernel void AGReLU(__global float*input,__global float*gradient,
                     const int count,const int mode)
  {
   int i=get_global_id(0);

   if(i>=count)
      return;

   if(mode==0)
      input[i]=fmax(input[i],0.0f);
   else
      gradient[i]=input[i]>0.0f?gradient[i]:0.0f;
  }
//+------------------------------------------------------------------+
//| AGCopySlice - Copy record slices on device. All bounds are       |
//+------------------------------------------------------------------+
__kernel void AGCopySlice(__global const float*source,const int source_offset,
                          __global float *destination,const int destination_offset,
                          const int      count)
  {
   int i=get_global_id(0);

   if(i<count)
      destination[destination_offset+i]=source[source_offset+i];
  }
//+------------------------------------------------------------------+
//| AGContextMetadata - Exact time/version ring accompanies the      |
//+------------------------------------------------------------------+
__kernel void AGContextMetadata(__global const float*service,const int window,
                                __global const float *banks,const int version_offset,
                                __global float       *metadata,const int count,const int fresh)
  {
   if(get_global_id(0)!=0)
      return;

   if(!fresh)
      for(int i=0;i<(count-1)*8;i++)
         metadata[i]=metadata[i+8];
   int first=fresh?0:count-1;

   for(int i=first;i<count;i++)
      for(int j=0;j<4;j++)
        {
         metadata[i*8+j]=service[AG_CTX_SIZE+(window-count+i)*5+j];
         metadata[i*8+4+j]=banks[version_offset+j];
        }
  }

//--- one group owns one formula. LSTM states and gate storage are disjoint
//--- across groups. Partial products are reduced in group-local memory.
//+------------------------------------------------------------------+
//| AGSearchRecur - Implement AGSearchRecur                          |
//+------------------------------------------------------------------+
inline void AGSearchRecur(__global const float*input,const int inputs,
                          __global const float *weights,const int hidden,
                          __global float       *state,__global float*gates,
                          __local float        *partial)
  {
   const int lane=get_local_id(0);
   const int lanes=get_local_size(0);
   const int parts=min(lanes,4);
   const int part=lane%parts;
   const int rows=lanes/parts;
   const int width=inputs+hidden+1;

   for(int start=0;start<4*hidden;start+=rows)
     {
      int   row=start+lane/parts;
      float sum=0.0f;

      if(row<4*hidden)
         for(int k=part;k<width-1;k+=parts)
            sum+=weights[row*width+k]*(k<inputs?input[k]:state[k-inputs]);
      partial[lane]=sum;

      if(lanes>1)
         barrier(CLK_LOCAL_MEM_FENCE);

      if(part==0&&row<4*hidden)
        {
         float dot=partial[lane];

         if(parts==4)
            dot=(dot+partial[lane+1])+(partial[lane+2]+partial[lane+3]);
         else
            if(parts==2)
               dot+=partial[lane+1];
         float a=weights[row*width+width-1]+dot;

         gates[row]=row/hidden==3?tanh(a):1.0f/(1.0f+exp(-a));
        }

      if(lanes>1)
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
     }

   for(int j=lane;j<hidden;j+=lanes)
     {
      float cell=gates[j]*state[hidden+j]+gates[hidden+j]*gates[3*hidden+j];

      state[hidden+j]=cell;
      state[j]=gates[2*hidden+j]*tanh(cell);
     }

   if(lanes>1)
      barrier(CLK_GLOBAL_MEM_FENCE);
  }
//+------------------------------------------------------------------+
//| AGSearchGenerate - AGSearchGenerate -                            |
//+------------------------------------------------------------------+
__kernel void AGSearchGenerate(__global const float*cfg,
                               __global const float *embedding,
                               __global const float *first_lstm,
                               __global const float *second_lstm,
                               __global const float *mean,
                               const uint           seed,
                               __global float       *results,
                               __global float       *workspace)
  {
   const int    lane=get_local_id(0);
   const int    lanes=get_local_size(0);
   const size_t candidate=get_group_id(0);
   const int    maximum=(int)cfg[AG_CFG_MAX_TOKENS];
   const int    hidden=(int)cfg[AG_CFG_SEARCH_HIDDEN];
   const int    dimension=(int)cfg[AG_CFG_EMBEDDING];
   const int    stride=4+AG_VOCABULARY;
   const int    work_stride=dimension+9*hidden+3*(maximum+1)*AG_VOCABULARY;

   __global float *result=results+candidate*(1+maximum*stride);

//--- match GenerateBatch: power-of-two groups, bounded by partial[64].
//--- the rejection is uniform and precedes all barriers and scratch access.
   if(lanes>64||(lanes&(lanes-1))!=0)
     {
      if(lane==0)
         result[0]=-1.0f;
      return;
     }
   __global float *embedded=workspace+candidate*work_stride;
   __global float *first=embedded+dimension;
   __global float *second=first+2*hidden;
   __global float *gates=second+2*hidden;
   __global float *sum=gates+4*hidden;
   __global int   *mask_work=(__global int*)(sum+hidden);
   __local float  partial[64];
   __local float  logits[AG_VOCABULARY+1];
   __local int    choice;
   uint           random=seed^((uint)(candidate+1)*0x9e3779b9U);

   if(!random)
      random=1;

   if(lane==0)
     {
      result[0]=-1.0f;
      choice=AG_BOS;
     }

   for(int j=lane;j<5*hidden;j+=lanes)
      if(j<4*hidden)
         first[j]=0.0f;
      else
         sum[j-4*hidden]=0.0f;

   if(lanes>1)
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

   for(int t=0;t<maximum;t++)
     {
      __global float *row=result+1+t*stride;

      //--- snapshot the shared token before any lane can publish a new one.
      const int  selected_token=choice;
      const bool active=selected_token!=AG_STOP&&selected_token>=0;
      const int  input_token=active?selected_token:AG_BOS;

      if(lanes>1)
         barrier(CLK_LOCAL_MEM_FENCE);

      for(int j=lane;j<dimension;j+=lanes)
        {
         float angle=(float)t/pow(cfg[AG_CFG_POSITION_SCALE],(float)(2*(j/2))/(float)dimension);

         embedded[j]=embedding[input_token*dimension+j]+(j%2==0?sin(angle):cos(angle));
        }

      if(active)
         for(int token=lane;token<AG_VOCABULARY;token+=lanes)
            row[4+token]=AGMaskToken(result+1,t,stride,cfg,mask_work,token);

      if(lanes>1)
         barrier(CLK_GLOBAL_MEM_FENCE);
      AGSearchRecur(embedded,dimension,first_lstm,hidden,first,gates,partial);
      AGSearchRecur(first,hidden,second_lstm,hidden,second,gates,partial);

      for(int j=lane;j<hidden;j+=lanes)
         sum[j]+=second[j];

      if(lanes>1)
         barrier(CLK_GLOBAL_MEM_FENCE);

      for(int unit=lane;unit<=AG_VOCABULARY;unit+=lanes)
        {
         const int offset=unit*(hidden+1);
         float     score=0.0f;

         for(int j=0;j<hidden;j++)
            score+=mean[offset+j]*(sum[j]/(float)(t+1));
         logits[unit]=score+mean[offset+hidden];
        }

      if(lanes>1)
         barrier(CLK_LOCAL_MEM_FENCE);

      if(active&&lane==0)
        {
         float top=-INFINITY;
         float total=0.0f;
         bool  valid=true;

         for(int j=0;j<=AG_VOCABULARY;j++)
            valid=valid&&isfinite(logits[j]);

         for(int j=0;j<AG_VOCABULARY;j++)
            if(row[4+j]>0.0f)
               top=fmax(top,logits[j]);

         for(int j=0;j<AG_VOCABULARY;j++)
            if(row[4+j]>0.0f)
               total+=exp(logits[j]-top);
         choice=-1;

         if(valid&&total>0.0f&&isfinite(total))
           {
            random^=random<<13;
            random^=random>>17;
            random^=random<<5;
            float uniform=convert_float(random)*0x1p-32f;
            float normalizer=top+log(total);
            float cumulative=0.0f;
            float entropy=0.0f;
            int   last=-1;

            for(int j=0;j<AG_VOCABULARY;j++)
               if(row[4+j]>0.0f)
                 {
                  float lp=logits[j]-normalizer;
                  float probability=exp(lp);

                  entropy-=probability*lp;
                  cumulative+=probability;
                  last=j;

                  if(choice<0&&uniform<cumulative)
                     choice=j;
                 }

            if(choice<0)
               choice=last;
            row[0]=(float)choice;
            row[1]=logits[choice]-normalizer;
            row[2]=entropy;
            row[3]=logits[AG_VOCABULARY];

            if(choice==AG_STOP)
               result[0]=(float)(t+1);
           }
        }

      if(lanes>1)
         barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      //--- keep completed outputs while all lanes finish the fixed barrier sequence.
     }
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| AGSearchRecord - Validate and record one search transition       |
//+------------------------------------------------------------------+
__kernel void AGSearchRecord(__global const float*sample,__global const float*metadata,
                             __global const float *mask,__global float*prefix,
                             __global float       *formula,const int prefix_length,
                             const int            maximum,const int row,const int episode_start,
                             const int            capacity,__global float*prefixes,
                             __global float       *masks,__global float*samples,
                             __global float       *rollout_metadata,__global float*gae,
                             __global float       *result)
  {
   if(get_global_id(0)!=0)
      return;

   result[0]=AG_INVALID;
   result[1]=sample[0];

   if(maximum<1||prefix_length<0||prefix_length>=maximum||row<0||
      row>=capacity||episode_start<0||episode_start>row)
      return;

   float raw=sample[0];

   if(!isfinite(raw)||raw<0.0f||raw>=(float)AG_VOCABULARY||raw!=floor(raw))
      return;

   int token=(int)raw;

   if(mask[token]!=1.0f||(token==AG_STOP&&prefix_length<1))
      return;

   for(int j=0;j<7;j++)
      if(!isfinite(metadata[j]))
         return;

//--- host validates complete backing capacities. Use wide offsets here.
   const size_t p=(size_t)row*((size_t)maximum+1);
   const size_t m=(size_t)row*AG_VOCABULARY;
   const size_t g=(size_t)row*AG_GAE_FIELDS;
   const size_t md=(size_t)row*7;

   for(int j=0;j<=prefix_length;j++)
      prefixes[p+j]=prefix[j];

   for(int j=0;j<AG_VOCABULARY;j++)
      masks[m+j]=mask[j];
   samples[row]=raw;

   for(int j=0;j<7;j++)
      rollout_metadata[md+j]=metadata[j];
   rollout_metadata[md+3]=metadata[0];
   gae[g+AG_GAE_VALUE]=metadata[2];
   gae[g+AG_GAE_REWARD]=0.0f;
   gae[g+AG_GAE_DELTA]=1.0f;
   gae[g+AG_GAE_BOOTSTRAP]=1.0f;
   gae[g+AG_GAE_CONTINUE]=1.0f;
   gae[g+AG_GAE_READY]=1.0f;

   if(row>episode_start)
      gae[(size_t)(row-1)*AG_GAE_FIELDS+AG_GAE_NEXT]=metadata[2];
   formula[prefix_length]=raw;
   prefix[prefix_length+1]=raw;
   result[0]=AG_OK;
  }
//+------------------------------------------------------------------+
