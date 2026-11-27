//+------------------------------------------------------------------+
//|                                              AlphaQCM_Network.cl |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
// AlphaQCM batch network. Tape rows contain h, c, then f, i, o and g.
// Each group reduction uses a power-of-two local size chosen for its kernel.
inline float AQCNetSum(float value,__local float *scratch)
  {
   int lane=get_local_id(0);
   scratch[lane]=value;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int step=get_local_size(0)/2;step>0;step/=2)
     {
      if(lane<step)
         scratch[lane]+=scratch[lane+step];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   value=scratch[0];
   barrier(CLK_LOCAL_MEM_FENCE);
   return value;
  }

__kernel void AQCNetInit(__global float *w,ulong seed,ulong stream,ulong start,float scale)
  {
   uint i=get_global_id(0);
   w[i]=(2.0f*AQCPhiloxUnit(AQCPhiloxWord(seed,stream,0,start+i))-1.0f)*scale;
  }

//+------------------------------------------------------------------+
//| AQCStageBatch - Copy one logical batch slice into encoder input. |
//+------------------------------------------------------------------+
__kernel void AQCStageBatch(__global const int *source_tokens,__global const int *source_lengths,
                            __global const float *source_tau,__global int *tokens,
                            __global int *lengths,__global float *tau,
                            const int first,const int length,const int quantiles)
  {
   const int column=get_global_id(0);
   const int row=get_global_id(1);
   const int source=first+row;
   const int used=source_lengths[source];
   if(column==0)
      lengths[row]=used;
   if(column<length)
      tokens[row*length+column]=used==0?AG_PAD:source_tokens[source*length+column];
   if(column<quantiles)
      tau[row*quantiles+column]=source_tau[source*quantiles+column];
  }

__kernel void AQCEmbed(__global const int *token,__global const int *length,
                       __global const float *weight,__global float *x,int L,int E,int A,float position_scale)
  {
   int j=get_global_id(0),p=get_global_id(1),t=p%L,b=p/L;
   if(t>=length[b])
     {
      x[p*E+j]=0;
      return;
     }
   int id=token[p];
   if(length[b]>L || id<0 || id>=A || (t==0 && id!=0) || (t!=0 && id<=1))
     {
      x[p*E+j]=NAN;
      return;
     }
   float phase=t/pow(position_scale,(float)(2*(j/2))/E);
   x[p*E+j]=weight[id*E+j]+((j&1)?cos(phase):sin(phase));
  }

__kernel void AQCRecur(__global const float *x,__global const int *len,__global const float *w,
                       __global float *tape,__global float *state,__global float *mean,
                       int L,int I,int H,int stride,int mode,__global float *scratch)
  {
   __local float partial[128];
   int b=get_group_id(0),lane=get_local_id(0),lanes=get_local_size(0);
   __global float *s=state+b*2*H;
   for(int j=lane;j<2*H;j+=lanes) s[j]=0;
   for(int j=lane;j<H;j+=lanes) mean[b*H+j]=0;
   barrier(CLK_GLOBAL_MEM_FENCE);
   for(int t=0;t<L;t++)
     {
      int p=b*L+t;
      __global float *row=mode?scratch+b*6*H:tape+p*6*H;
      if(t>=len[b])
        {
         if(mode==0) for(int j=lane;j<6*H;j+=lanes) row[j]=0;
         if(mode==1) for(int j=lane;j<H;j+=lanes) tape[p*H+j]=0;
         continue;
        }
      AGSearchRecur(x+p*stride,I,w,H,s,row+2*H,partial);
      for(int j=lane;j<H;j+=lanes)
        {
         if(mode==0)
           {
            row[j]=s[j];
            row[H+j]=s[H+j];
           }
         if(mode==1) tape[p*H+j]=s[j];
         mean[b*H+j]+=s[j];
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
   for(int j=lane;j<H;j+=lanes)
      mean[b*H+j]=len[b]>L||len[b]<0?NAN:mean[b*H+j]/max(1,len[b]);
  }

__kernel void AQCBPTT(__global const float *tape,__global const int *len,__global const float *w,
                      __global const float *dy,__global float *dx,__global float *da,
                      __global float *state,int L,int I,int H,int pooled)
  {
   int b=get_group_id(0),lane=get_local_id(0),lanes=get_local_size(0),D=I+H+1;
   __global float *dh=state+b*2*H,*dc=dh+H;
   for(int j=lane;j<2*H;j+=lanes) dh[j]=0;
   barrier(CLK_GLOBAL_MEM_FENCE);
   for(int t=L-1;t>=0;t--)
     {
      int p=b*L+t;
      __global const float *row=tape+p*6*H;
      __global float *a=da+p*4*H;
      if(t>=len[b])
        {
         for(int j=lane;j<4*H;j+=lanes) a[j]=0;
         for(int j=lane;j<I;j+=lanes) dx[p*I+j]=0;
         continue;
        }
      for(int j=lane;j<H;j+=lanes)
        {
         float g=dh[j]+(pooled?dy[b*H+j]/len[b]:dy[p*H+j]);
         float prev=t?row[-5*H+j]:0;
         float f=row[2*H+j],i=row[3*H+j],o=row[4*H+j],v=row[5*H+j];
         float c=tanh(row[H+j]),d=dc[j]+g*o*(1-c*c);
         a[j]=d*prev*f*(1-f); a[H+j]=d*v*i*(1-i);
         a[2*H+j]=g*c*o*(1-o); a[3*H+j]=d*i*(1-v*v);
         dc[j]=d*f;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
      for(int j=lane;j<I+H;j+=lanes)
        {
         float sum=0;
         for(int k=0;k<4*H;k++) sum+=a[k]*w[k*D+j];
         if(j<I) dx[p*I+j]=sum; else dh[j-I]=sum;
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
     }
  }

__kernel void AQCRecurOuter(__global const float *x,__global const float *tape,
                            __global const float *da,__global float *grad,
                            int rows,int L,int I,int H,int stride)
  {
   __local float part[128];
   int at=get_group_id(0),D=I+H+1,j=at%D,k=at/D;
   float sum=0;
   for(int p=get_local_id(0);p<rows;p+=get_local_size(0))
     {
      float v=j<I?x[p*stride+j]:(j==D-1?1.0f:(p%L?tape[(p-1)*6*H+j-I]:0.0f));
      sum+=v*da[p*4*H+k];
     }
   sum=AQCNetSum(sum,part);
   if(!get_local_id(0)) grad[at]+=sum;
  }

__kernel void AQCEmbedOuter(__global const int *tokens,__global const int *length,
                            __global const float *dx,__global float *grad,int rows,int L,int E)
  {
   __local float part[128];
   int at=get_group_id(0),id=at/E,j=at%E;
   float sum=0;
   for(int p=get_local_id(0);p<rows;p+=get_local_size(0))
      if(p%L<length[p/L] && tokens[p]==id) sum+=dx[p*E+j];
   sum=AQCNetSum(sum,part);
   if(!get_local_id(0)) grad[at]+=sum;
  }

__kernel void AQCCosine(__global const float *tau,__global float *x,int M)
  {
   int j=get_global_id(0),r=get_global_id(1);
   float t=tau[r];
   x[r*M+j]=t>0 && t<1?cospi(j*t):NAN;
  }

// Four lanes cooperate on each dot. A group owns up to 32 features of one row.
inline float AQCDotParts(float dot,int parts,__local float *scratch)
  {
   int lane=get_local_id(0);
   scratch[lane]=dot;
   barrier(CLK_LOCAL_MEM_FENCE);
   if(lane%parts==0)
     {
      if(parts>=2) dot+=scratch[lane+1];
      if(parts==4) dot+=scratch[lane+2]+scratch[lane+3];
     }
   return dot;
  }

__kernel void AQCDenseRows(__global const float *x,__global const float *w,__global float *y,
                           int I,int O,int relu)
  {
   __local float scratch[128];
   int lane=get_local_id(0),parts=min(4,(int)get_local_size(0)),columns=get_local_size(0)/parts;
   int j=get_group_id(0)*columns+lane/parts,r=get_global_id(1),part=lane%parts;
   float dot=0;
   if(j<O)
      for(int k=part;k<I;k+=parts) dot+=x[r*I+k]*w[j*(I+1)+k];
   dot=AQCDotParts(dot,parts,scratch);
   if(part==0&&j<O)
     {
      dot+=w[j*(I+1)+I];
      y[r*O+j]=relu&&isfinite(dot)?fmax(0.0f,dot):dot;
     }
  }

__kernel void AQCProduct(__global const float *psi,__global const float *phi,
                         __global float *v,int N,int H)
  {
   int j=get_global_id(0),r=get_global_id(1);
   v[r*H+j]=psi[(r/N)*H+j]*phi[r*H+j];
  }

__kernel void AQCDenseInput(__global const float *dy,__global const float *w,
                            __global const float *x,__global float *dx,int I,int O,int relu)
  {
   __local float scratch[128];
   int lane=get_local_id(0),parts=min(4,(int)get_local_size(0)),columns=get_local_size(0)/parts;
   int j=get_group_id(0)*columns+lane/parts,r=get_global_id(1),part=lane%parts;
   float dot=0;
   if(j<I)
      for(int k=part;k<O;k+=parts) dot+=dy[r*O+k]*w[k*(I+1)+j];
   dot=AQCDotParts(dot,parts,scratch);
   if(part==0&&j<I) dx[r*I+j]=relu&&x[r*I+j]<=0?0.0f:dot;
  }

__kernel void AQCDenseOuter(__global const float *x,__global const float *dy,__global float *grad,
                            int rows,int I,int O)
  {
   __local float part[128];
   int at=get_group_id(0),j=at%(I+1),k=at/(I+1);
   float sum=0;
   for(int p=get_local_id(0);p<rows;p+=get_local_size(0))
      sum+=dy[p*O+k]*(j==I?1.0f:x[p*I+j]);
   sum=AQCNetSum(sum,part);
   if(!get_local_id(0)) grad[at]+=sum;
  }

__kernel void AQCProductGrad(__global const float *dv,__global const float *psi,
                             __global const float *phi,__global float *dpsi,
                             __global float *dphi,int N,int H)
  {
   __local float part[128];
   int at=get_group_id(0),b=at/H,j=at%H;
   float sum=0;
   for(int i=get_local_id(0);i<N;i+=get_local_size(0))
     {
      int k=(b*N+i)*H+j;
      sum+=dv[k]*phi[k];
      dphi[k]=phi[k]>0?dv[k]*psi[at]:0;
     }
   sum=AQCNetSum(sum,part);
   if(!get_local_id(0)) dpsi[at]=sum;
  }

// Prospective native Adam preflight. One group per parameter owner.
__kernel void AQCParameterCheck(__global const float *w,__global const float *g,
                                __global const float *m,__global const float *v,__global float *flags,
                                int offset,int count,int D,float learning,float beta1,float beta2)
  {
   __local float part[128];
   int bad=0;
   for(int k=get_local_id(0);k<count;k+=get_local_size(0))
     {
      float grad=clamp(g[k],-MAX_GRAD,MAX_GRAD);
      float mt=beta1*m[k]+(1-beta1)*grad,vt=beta2*v[k]+(1-beta2)*(grad*grad);
      float delta=learning*(mt/(sqrt(vt)+1.e-37f)-(l1*sign(w[k])+l2*w[k]/D));
      bad|=!isfinite(w[k])||!isfinite(g[k])||!isfinite(m[k])||!isfinite(v[k])||v[k]<0||
           !isfinite(mt)||!isfinite(vt)||vt<0||!isfinite(delta)||!isfinite(w[k]+delta);
     }
   float fail=AQCNetSum((float)bad,part);
   if(!get_local_id(0)) flags[offset]=fail;
  }

__kernel void AQCSpanCheck(__global const float *x,__global float *flags,int offset,int count,int require_zero)
  {
   __local float part[128];
   int bad=0;
   for(int k=get_local_id(0);k<count;k+=get_local_size(0))
      bad|=!isfinite(x[k])||(require_zero&&x[k]!=0);
   float fail=AQCNetSum((float)bad,part);
   if(!get_local_id(0)) flags[offset]=fail;
  }

__kernel void AQCTargetSelect(__global const int *done,__global const int *legal,
                              __global const float *qt,__global float *next,int A,int base)
  {
   __local float scores[128];
   __local int ids[128],bad[128];
   int b=get_group_id(0),lane=get_local_id(0),lanes=get_local_size(0);
   if(done[base+b])
     {
      if(!lane) next[b]=(done[base+b]==1?-1.0f:NAN);
      return;
     }
   float best=-INFINITY;
   int id=A,failed=0;
   for(int a=lane;a<A;a+=lanes)
     {
      int allowed=legal[(base+b)*A+a];
      if(allowed!=0 && (allowed!=1||a<2)) failed=1;
      if(!allowed) continue;
      float q=qt[b*A+a];
      if(!isfinite(q)) failed=1;
      if(q>best)
        {
         best=q;
         id=a;
        }
     }
   scores[lane]=best; ids[lane]=id; bad[lane]=failed;
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int n=lanes/2;n;n/=2)
     {
      if(lane<n)
        {
         int j=lane+n;
         if(scores[j]>scores[lane]||(scores[j]==scores[lane]&&ids[j]<ids[lane]))
           {
            scores[lane]=scores[j];
            ids[lane]=ids[j];
           }
         bad[lane]|=bad[j];
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   if(!lane) next[b]=bad[0]||ids[0]>=A?NAN:(float)ids[0];
  }

__kernel void AQCQResidual(__global const float *q,__global const float *qt,
                           __global const int *actions,__global const float *reward,
                           __global const float *next,__global const float *is_weight,
                           __global float *gradient,__global float *loss,float gamma,int A,int B,int base)
  {
   int a=get_global_id(0),b=get_global_id(1);
   if(actions[base+b]<2||actions[base+b]>=A)
     {
      gradient[b*A+a]=0;
      if(!a) loss[b]=NAN;
      return;
     }
   if(a!=actions[base+b])
     {
      gradient[b*A+a]=0;
      return;
     }
   float v=reward[base+b],weight=is_weight[base+b],id=next[b];
   if(!isfinite(id)||!(weight>0)||!isfinite(weight)) v=NAN;
   else if(id>=0) v+=gamma*qt[b*A+(int)id];
   float delta=v-q[b*A+a];
   gradient[b*A+a]=weight*delta/B;
   loss[b]=weight*(0.5f*delta*delta)/B;
  }

__kernel void AQCQuantileTarget(__global const float *zt,__global const float *reward,
                                __global const float *next,__global float *target,float gamma,int A,int nt,int base)
  {
   int j=get_global_id(0),b=get_global_id(1);
   float id=next[b],y=reward[base+b];
   if(!isfinite(id)) y=NAN;
   else if(id>=0) y+=gamma*zt[(b*nt+j)*A+(int)id];
   target[b*nt+j]=y;
  }

__kernel void AQCPairLoss(__global const float *z,__global const float *target,__global const int *action,
                          __global const float *tau,__global const float *is_weight,
                          __global float *row_loss,__global float *selected,int A,int N,int nt,int B,float kappa,int base)
  {
   __local float part[128];
   int r=get_group_id(0),b=r/N,id=action[base+b];
   float estimate=id>=2&&id<A?z[r*A+id]:NAN,t=tau[base*N+r],loss=0,grad=0;
   for(int j=get_local_id(0);j<nt;j+=get_local_size(0))
     {
      float d=target[b*nt+j]-estimate,mag=fabs(d),w=fabs(t-(float)(d<0));
      float huber=mag<=kappa?0.5f*d*d:kappa*(mag-0.5f*kappa);
      loss+=w*huber/kappa;
      grad+=w*clamp(d,-kappa,kappa)/kappa;
     }
   loss=AQCNetSum(loss,part);
   grad=AQCNetSum(grad,part);
   if(!get_local_id(0))
     {
      row_loss[r]=t>0&&t<1?loss/(N*(float)nt):NAN;
      selected[r]=is_weight[base+b]*grad/(B*(float)N*nt);
     }
  }

__kernel void AQCExpandGrad(__global const float *selected,__global const int *action,
                            __global float *gradient,int A,int N,int base)
  {
   int a=get_global_id(0),r=get_global_id(1);
   gradient[r*A+a]=action[base+r/N]==a?selected[r]:0;
  }

__kernel void AQCPriorityLoss(__global const float *row_loss,__global const float *is_weight,
                              __global float *loss,__global float *weighted,int N,int B,int base)
  {
   __local float part[128];
   int b=get_group_id(0);
   float v=0;
   for(int i=get_local_id(0);i<N;i+=get_local_size(0)) v+=row_loss[b*N+i];
   v=AQCNetSum(v,part);
   if(!get_local_id(0))
     {
      loss[base+b]=v;
      weighted[b]=is_weight[base+b]*v/B;
     }
  }

__kernel void AQCZeroGradient(__global float *gradient)
  {
   gradient[get_global_id(0)]=0;
  }
