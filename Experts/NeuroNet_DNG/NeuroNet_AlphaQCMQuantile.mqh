//+------------------------------------------------------------------+
//|                                    NeuroNet_AlphaQCMQuantile.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_QUANTILE_MQH
#define NEURONET_ALPHAQCM_QUANTILE_MQH

// The design is fixed at creation. SVD supplies the condition estimate.
// A second pass of modified Gram-Schmidt orthogonalizes the four columns.
//+------------------------------------------------------------------+
//| AlphaQCMProjection - Build the orthogonal quantile projection    |
//+------------------------------------------------------------------+
bool AlphaQCMProjection(CBufferFloat &projection,COpenCLMy *context,const int points,int &status)
  {
   status=AQCM_NUMERIC_ERROR;
   if(!context||points<4)
     {
      ReturnFalse
     }
   matrix design(points,4),q(points,4),r=matrix::Zeros(4,4);
   for(int k=0;k<points;k++)
     {
      int error=0;
      double z=MathQuantileNormal((k+0.5)/points,0.0,1.0,error);
      if(error||!MathIsValidNumber(z))
        {
         ReturnFalse
        }
      design[k][0]=1; design[k][1]=z; design[k][2]=z*z-1; design[k][3]=z*(z*z-3);
     }
   matrix u;
   matrix v;
   vector singular;
   if(!design.SVD(u,v,singular)||singular.Size()!=4)
     {
      ReturnFalse
     }
   double top=singular.Max(),bottom=singular.Min();
   if(!MathIsValidNumber(top)||!MathIsValidNumber(bottom)||bottom<=points*DBL_EPSILON*top||
      top/bottom>1.0e6)
     {
      ReturnFalse
     }
   q=design;
   for(int j=0;j<4;j++)
     {
      for(int pass=0;pass<2;pass++)
         for(int i=0;i<j;i++)
           {
            double dot=0;
            for(int k=0;k<points;k++) dot+=q[k][i]*q[k][j];
            r[i][j]+=dot;
            for(int k=0;k<points;k++) q[k][j]-=dot*q[k][i];
           }
      double norm=0;
      for(int k=0;k<points;k++) norm+=q[k][j]*q[k][j];
      norm=MathSqrt(norm);
      if(!(norm>0)||!MathIsValidNumber(norm))
        {
         ReturnFalse
        }
      r[j][j]=norm;
      for(int k=0;k<points;k++) q[k][j]/=norm;
     }
   double dual[4];
   for(int j=0;j<4;j++)
     {
      double rhs=(j==1?1.0:0.0);
      for(int i=0;i<j;i++) rhs-=r[i][j]*dual[i];
      dual[j]=rhs/r[j][j];
     }
   if(!projection.BufferInit(points,0))
     {
      status=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalse
     }
   double check[4]={0,0,0,0};
   for(int k=0;k<points;k++)
     {
      double weight=0;
      for(int j=0;j<4;j++) weight+=q[k][j]*dual[j];
      float stored=(float)weight;
      if(!MathIsValidNumber(stored)||!projection.Update(k,stored))
        {
         ReturnFalse
        }
      for(int j=0;j<4;j++) check[j]+=(double)stored*(double)(float)design[k][j];
     }
   for(int j=0;j<4;j++)
      if(!MathIsValidNumber(check[j])||MathAbs(check[j]-(j==1?1.0:0.0))>1e-5)
        {
         ReturnFalse
        }
   if(!projection.BufferCreate(context))
     {
      status=AQCM_RESOURCE_UNSUPPORTED;
      ReturnFalse
     }
   status=AG_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| AlphaQCMProjection - Build the orthogonal quantile projection    |
//+------------------------------------------------------------------+
bool AlphaQCMProjection(CBufferFloat &projection,COpenCLMy *context,const int points)
  {
   int status;
   return(AlphaQCMProjection(projection,context,points,status));
  }

class CAlphaQCMObjective
  {
protected:
   COpenCLMy *OpenCL;
   CBufferFloat m_next;
   CBufferFloat m_qGradient;
   CBufferFloat m_zGradient;
   CBufferFloat m_qLoss;
   CBufferFloat m_zLoss;
   CBufferFloat m_target;
   CBufferFloat m_rows;
   CBufferFloat m_selected;
   CBufferFloat m_priority;
   CBufferFloat m_flags;
   CBufferFloat m_tileFlags;
   int          m_chunk;
   int          m_status;
   bool         m_ready;

public:
//+------------------------------------------------------------------+
//| CAlphaQCMObjective - Initialize the object state                 |
//+------------------------------------------------------------------+
   CAlphaQCMObjective(void):OpenCL(NULL),m_chunk(0),m_status(AG_OK),m_ready(false) {}
//+------------------------------------------------------------------+
//| Status - Return the status                                       |
//+------------------------------------------------------------------+
   int  Status(void) const { return(m_status); }
//+------------------------------------------------------------------+
//| QGradient - Return the q gradient                                |
//+------------------------------------------------------------------+
   int  QGradient(void) const { return(m_qGradient.GetIndex()); }
//+------------------------------------------------------------------+
//| ZGradient - Return the z gradient                                |
//+------------------------------------------------------------------+
   int  ZGradient(void) const { return(m_zGradient.GetIndex()); }
//+------------------------------------------------------------------+
//| PriorityLoss - Return the priority loss                          |
//+------------------------------------------------------------------+
   int  PriorityLoss(void) const { return(m_priority.GetIndex()); }
   bool Prepare(COpenCLMy *context,AQCReplayBatch &batch,const int q,const int z,const int qt,const int zt,
                const int rows,const int actions,const int n,const int nt,const float gamma,const float kappa,
                const int logical,const int base);
   bool Accumulate(COpenCLMy *context,AQCReplayBatch &batch,CNet *qnet,CNet *znet,CNet *qtarget,CNet *ztarget,
                   const int rows,const int actions,const int n,const int nt,const float gamma,const float kappa,
                   const float position_scale,const int flags);
   bool Preflight(const int flags,const int offset);

protected:
   bool         Reserve(const int rows,const int actions,const int n,const int nt,const int logical);
  };

//+------------------------------------------------------------------+
//| Reserve - Reserve target, loss and gradient buffers              |
//+------------------------------------------------------------------+
bool CAlphaQCMObjective::Reserve(const int rows,const int actions,const int n,const int nt,const int logical)
  {
   const long r=(long)rows*n;
   if(r>INT_MAX/actions||!AQCNetStorage(OpenCL,m_next,rows)||!AQCNetStorage(OpenCL,m_qGradient,(long)rows*actions)||
      !AQCNetStorage(OpenCL,m_zGradient,r*actions)||!AQCNetStorage(OpenCL,m_qLoss,rows)||
      !AQCNetStorage(OpenCL,m_zLoss,rows)||!AQCNetStorage(OpenCL,m_priority,logical)||
      !AQCNetStorage(OpenCL,m_target,(long)rows*nt)||!AQCNetStorage(OpenCL,m_rows,r)||
      !AQCNetStorage(OpenCL,m_selected,r)||!AQCNetStorage(OpenCL,m_flags,2)||
      !AQCNetStorage(OpenCL,m_tileFlags,3L*(logical/rows)))
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Prepare - Build targets and loss gradients                       |
//+------------------------------------------------------------------+
bool CAlphaQCMObjective::Prepare(COpenCLMy *context,AQCReplayBatch &batch,const int q,const int z,
                                 const int qt,const int zt,const int rows,const int actions,const int n,const int nt,
                                 const float gamma,const float kappa,const int logical,const int base)
  {
   m_ready=false;
   if(!context||rows<1||n<1||nt<1||actions<3||!MathIsValidNumber(kappa)||kappa<=0||!MathIsValidNumber(gamma))
     {
      ReturnFalse
     }
   OpenCL=context;
   const long r=(long)rows*n;
   int k=def_k_AQCTargetSelect;
   setBuffer(k,0,batch.done)
   setBuffer(k,1,batch.next_legal)
   setBuffer(k,2,qt)
   setBuffer(k,3,m_next.GetIndex())
   setArgument(k,4,actions)
   setArgument(k,5,base)
   if(!AQCNetGroups(OpenCL,k,rows))
     {
      ReturnFalse
     }

   k=def_k_AQCQResidual;
   setBuffer(k,0,q)
   setBuffer(k,1,qt)
   setBuffer(k,2,batch.actions)
   setBuffer(k,3,batch.rewards)
   setBuffer(k,4,m_next.GetIndex())
   setBuffer(k,5,batch.weights)
   setBuffer(k,6,m_qGradient.GetIndex())
   setBuffer(k,7,m_qLoss.GetIndex())
   setArgument(k,8,gamma)
   setArgument(k,9,actions)
   setArgument(k,10,logical)
   setArgument(k,11,base)
   if(!AQCNetRange(OpenCL,k,actions,rows))
     {
      ReturnFalse
     }

   k=def_k_AQCQuantileTarget;
   setBuffer(k,0,zt)
   setBuffer(k,1,batch.rewards)
   setBuffer(k,2,m_next.GetIndex())
   setBuffer(k,3,m_target.GetIndex())
   setArgument(k,4,gamma)
   setArgument(k,5,actions)
   setArgument(k,6,nt)
   setArgument(k,7,base)
   if(!AQCNetRange(OpenCL,k,nt,rows))
     {
      ReturnFalse
     }

   k=def_k_AQCPairLoss;
   setBuffer(k,0,z)
   setBuffer(k,1,m_target.GetIndex())
   setBuffer(k,2,batch.actions)
   setBuffer(k,3,batch.tau)
   setBuffer(k,4,batch.weights)
   setBuffer(k,5,m_rows.GetIndex())
   setBuffer(k,6,m_selected.GetIndex())
   setArgument(k,7,actions)
   setArgument(k,8,n)
   setArgument(k,9,nt)
   setArgument(k,10,logical)
   setArgument(k,11,kappa)
   setArgument(k,12,base)
   if(!AQCNetGroups(OpenCL,k,(int)r))
     {
      ReturnFalse
     }

   k=def_k_AQCExpandGrad;
   setBuffer(k,0,m_selected.GetIndex())
   setBuffer(k,1,batch.actions)
   setBuffer(k,2,m_zGradient.GetIndex())
   setArgument(k,3,actions)
   setArgument(k,4,n)
   setArgument(k,5,base)
   if(!AQCNetRange(OpenCL,k,actions,(int)r))
     {
      ReturnFalse
     }

   k=def_k_AQCPriorityLoss;
   setBuffer(k,0,m_rows.GetIndex())
   setBuffer(k,1,batch.weights)
   setBuffer(k,2,m_priority.GetIndex())
   setBuffer(k,3,m_zLoss.GetIndex())
   setArgument(k,4,n)
   setArgument(k,5,logical)
   setArgument(k,6,base)
   if(!AQCNetGroups(OpenCL,k,rows))
     {
      ReturnFalse
     }
   m_ready=true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Preflight - Check the prepared loss values                       |
//+------------------------------------------------------------------+
bool CAlphaQCMObjective::Preflight(const int flags,const int offset)
  {
   if(!m_ready||flags<0||offset<0)
     {
      ReturnFalse
     }
   int k=def_k_AQCSpanCheck;
   for(int i=0;i<2;i++)
     {
      setBuffer(k,0,(i?m_zLoss.GetIndex():m_qLoss.GetIndex()))
      setBuffer(k,1,m_flags.GetIndex())
      setArgument(k,2,i)
      setArgument(k,3,m_qLoss.Total())
      setArgument(k,4,0)
      if(!AQCNetGroups(OpenCL,k,1))
        {
         ReturnFalse
        }
     }
   setBuffer(k,0,m_flags.GetIndex())
   setBuffer(k,1,flags)
   setArgument(k,2,offset)
   setArgument(k,3,2)
   setArgument(k,4,1)
   return(AQCNetGroups(OpenCL,k,1));
  }
//+------------------------------------------------------------------+
//| Accumulate - Accumulate the replay minibatch gradients           |
//+------------------------------------------------------------------+
bool CAlphaQCMObjective::Accumulate(COpenCLMy *context,AQCReplayBatch &batch,CNet *qnet,CNet *znet,
                                    CNet *qtarget,CNet *ztarget,const int rows,const int actions,
                                    const int n,const int nt,const float gamma,const float kappa,
                                    const float position_scale,const int flags)
  {
   m_status=AG_TECHNICAL;
   if(!context||!qnet||!znet||!qtarget||!ztarget||rows<1||actions<3||n<1||nt<1||
      (long)rows*n>INT_MAX/actions||(long)rows*nt>INT_MAX/actions)
     {
      ReturnFalse
     }
   OpenCL=context;
   CNeuronAlphaQCMEncoderOCL *q=(CNeuronAlphaQCMEncoderOCL *)qnet.Layer(1);
   CNeuronAlphaQCMEncoderOCL *z=(CNeuronAlphaQCMEncoderOCL *)znet.Layer(1);
   CNeuronAlphaQCMEncoderOCL *qt=(CNeuronAlphaQCMEncoderOCL *)qtarget.Layer(1);
   CNeuronAlphaQCMEncoderOCL *zt=(CNeuronAlphaQCMEncoderOCL *)ztarget.Layer(1);
   if(!q||!z||!qt||!zt)
     {
      ReturnFalse
     }

   // Try the largest supported physical divisor. A successful reservation covers
   // all four graphs, both objectives and every backward buffer simultaneously.
   int chunk=m_chunk>0&&rows%m_chunk==0?m_chunk:rows;
   while(!q.ReserveBatch(chunk,1)||!z.ReserveBatch(chunk,n)||!qt.ReserveBatch(chunk,1)||
         !zt.ReserveBatch(chunk,nt)||!Reserve(chunk,actions,n,nt,rows))
     {
      q.ReleaseWorkspace(); z.ReleaseWorkspace(); qt.ReleaseWorkspace(); zt.ReleaseWorkspace();
      if(chunk==1)
        {
         m_status=AQCM_RESOURCE_UNSUPPORTED;
         ReturnFalseEx("RESOURCE_UNSUPPORTED: training workspace for one replay row")
        }
      chunk/=2;
      while(rows%chunk!=0) chunk--;
     }
   if(chunk!=m_chunk)
     {
      PrintFormat("AlphaQCM learner physical rows=%d logical rows=%d current=%d target=%d",chunk,rows,n,nt);
      m_chunk=chunk;
     }

   if(!q.BeginUpdate()||!z.BeginUpdate())
     {
      ReturnFalse
     }
   const int tiles=rows/chunk;
   for(int tile=0;tile<tiles;tile++)
     {
      const int first=tile*chunk;
      const bool last=(tile+1==tiles);
      if(!q.BindBatch(batch.tokens,batch.lengths,-1,chunk,1,first,position_scale)||
         !qnet.FeedForwardLayer(qnet.Layer(0))||
         !z.BindBatch(batch.tokens,batch.lengths,batch.tau,chunk,n,first,position_scale)||
         !znet.FeedForwardLayer(znet.Layer(0))||
         !qt.BindBatch(batch.next_tokens,batch.next_lengths,-1,chunk,1,first,position_scale)||
         !qtarget.FeedForwardLayer(qtarget.Layer(0))||
         !zt.BindBatch(batch.next_tokens,batch.next_lengths,batch.next_tau,chunk,nt,first,position_scale)||
         !ztarget.FeedForwardLayer(ztarget.Layer(0))||
         !Prepare(context,batch,q.Result(),z.Result(),qt.Result(),zt.Result(),chunk,actions,n,nt,gamma,kappa,rows,first)||
         !q.PrepareGradient(QGradient())||!qnet.backPropGradient((CBufferFloat *)NULL,(CBufferFloat *)NULL,-1,false)||
         !z.PrepareGradient(ZGradient())||!znet.backPropGradient((CBufferFloat *)NULL,(CBufferFloat *)NULL,-1,false)||
         !q.Preflight(m_tileFlags.GetIndex(),3*tile,last)||!z.Preflight(m_tileFlags.GetIndex(),3*tile+1,last)||
         !Preflight(m_tileFlags.GetIndex(),3*tile+2))
        {
         ReturnFalse
        }
     }
   int k=def_k_AQCSpanCheck;
   setBuffer(k,0,m_tileFlags.GetIndex())
   setBuffer(k,1,flags)
   setArgument(k,2,0)
   setArgument(k,3,3*tiles)
   setArgument(k,4,1)
   if(!AQCNetGroups(context,k,1))
     {
      ReturnFalse
     }
   m_status=AG_OK;

   return(true);
  }
#endif
