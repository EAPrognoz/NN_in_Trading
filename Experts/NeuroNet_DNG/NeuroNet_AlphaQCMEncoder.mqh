//+------------------------------------------------------------------+
//|                                     NeuroNet_AlphaQCMEncoder.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_ENCODER_MQH
#define NEURONET_ALPHAQCM_ENCODER_MQH

//+------------------------------------------------------------------+
//| AQCNetStorage - Reserve an exact-size float buffer               |
//+------------------------------------------------------------------+
bool AQCNetStorage(COpenCLMy *context,CBufferFloat &buffer,const long count)
  {
   if(!context||count<1||count>INT_MAX/(int)sizeof(float)||
      !((CAlphaQCMOpenCL *)context).CanAllocate((ulong)count,sizeof(float)))
     {
      ReturnFalse
     }
   if(buffer.GetIndex()>=0&&buffer.Total()==count)
      return(true);
   if(!buffer.BufferInit((uint)count,0)||!buffer.BufferCreate(context))
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| AQCNetRange - Run a kernel over columns and rows                 |
//+------------------------------------------------------------------+
bool AQCNetRange(COpenCLMy *OpenCL,const int kernel,const int columns,const int rows)
  {
   if(columns<1||rows<1)
     {
      ReturnFalse
     }
   uint offset[]={0,0},global[]={(uint)columns,(uint)rows};
   kernelExecute(kernel,offset,global)
   return(OpenCL.WaitAlphaGenKernel(kernel));
  }

//+------------------------------------------------------------------+
//| AQCNetGroups - Run a fixed number of workgroups                  |
//+------------------------------------------------------------------+
bool AQCNetGroups(COpenCLMy *OpenCL,const int kernel,const int groups)
  {
   uint lanes=((CAlphaQCMOpenCL *)OpenCL).GroupSize(kernel,128,AQC_NET_MIN_GROUP);
   if(lanes<1||groups<1||(ulong)groups*lanes>UINT_MAX)
     {
      ReturnFalse
     }
   uint offset[]={0},local[]={lanes},global[]={(uint)groups*lanes};
   kernelExecuteLoc(kernel,offset,global,local)
   return(OpenCL.WaitAlphaGenKernel(kernel));
  }

//+------------------------------------------------------------------+
//| AQCNetDots - Run tiled row-wise dot products                     |
//+------------------------------------------------------------------+
bool AQCNetDots(COpenCLMy *OpenCL,const int kernel,const int features,const int rows)
  {
   uint lanes=((CAlphaQCMOpenCL *)OpenCL).GroupSize(kernel,128,AQC_NET_MIN_GROUP);
   if(!lanes||features<1||rows<1)
     {
      ReturnFalse
     }
   uint columns=lanes/(lanes<4?lanes:4);
   uint offset[]={0,0},local[]={lanes,1},global[]={((uint)features+columns-1)/columns*lanes,(uint)rows};
   kernelExecuteLoc(kernel,offset,global,local)
   return(OpenCL.WaitAlphaGenKernel(kernel));
  }

//+------------------------------------------------------------------+
//| CNeuronAlphaQCMEncoderOCL - Initialize the unconfigured encoder  |
//+------------------------------------------------------------------+
CNeuronAlphaQCMEncoderOCL::CNeuronAlphaQCMEncoderOCL(void):
   m_L(0),m_A(0),m_E(0),m_H(0),m_M(0),m_B(0),m_N(0),m_tokens(-1),m_lengths(-1),m_taus(-1),
   m_prepared(-1),m_position_scale(0),m_iqn(false),m_forward_ready(false),m_gradient_ready(false),
   m_bound(false),m_revision(0),m_viewLifetime(0)
  {
  }

//+------------------------------------------------------------------+
//| ~CNeuronAlphaQCMEncoderOCL - Release the owned device input.     |
//+------------------------------------------------------------------+
CNeuronAlphaQCMEncoderOCL::~CNeuronAlphaQCMEncoderOCL(void)
  {
   ReleaseInput(m_tokens);
   ReleaseInput(m_lengths);
   ReleaseInput(m_taus);
  }

//+------------------------------------------------------------------+
//| Geometry - Validate encoder dimensions and Adam settings         |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Geometry(void)
  {
   if(lr!=3.0e-4f||b1!=0.9f||b2!=0.999f||m_L<1||m_L>1024||m_A<3||m_A>4096||
      m_E<2||m_E>1024||m_H<1||m_H>1024||m_M<1||m_M>1024)
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Configure - Configure encoder geometry and weights               |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Configure(COpenCLMy *context,const bool iqn,const int length,
                                          const int actions,const int embedding,const int hidden,const int frequencies)
  {
   if(context!=OpenCL||m_L!=0)
     {
      ReturnFalse
     }
   m_iqn=iqn; m_L=length; m_A=actions; m_E=embedding; m_H=hidden; m_M=frequencies;
   if(!Geometry()||!m_embedding.Configure(context,m_E-1,m_A,1)||
      !m_first.Configure(context,m_E,m_H,m_L)||!m_second.Configure(context,m_H,m_H,m_L)||
      (iqn&&!m_tau.Configure(context,m_M,m_H,1))||!m_hidden.Configure(context,m_H,m_H,1)||
      !m_head.Configure(context,m_H,m_A,1))
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| MatrixView - Expose a parameter and gradient matrix              |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::MatrixView(const int id,SAQCMatrix &v)
  {
   if(id<0||id>=MatrixCount())
     {
      ReturnFalse
     }
   if(id==1||id==2)
     {
      CAlphaQCMRecurrentOwner *owner=(id==1?GetPointer(m_first):GetPointer(m_second));
      v.weights=owner.RecurrentWeightsIndex();
      v.gradient=owner.WeightGradient();
      v.first=owner.FirstIndex(); v.second=owner.SecondIndex();
      v.width=(id==1?m_E:m_H)+m_H+1;
      v.rows=4*m_H;
      v.layout=AQC_LSTM_FIOG_ROWS;
     }
   else
     {
      CNeuronAlphaGenDenseOCL *owner=GetPointer(m_embedding);
      v.width=m_E;
      v.rows=m_A;
      v.layout=(id==0?AQC_EMBED_ROWS:AQC_DENSE_BIAS_LAST_ROWS);
      if(id==MatrixCount()-1)
        {
         owner=GetPointer(m_head);
         v.width=m_H+1;
        }
      else if(id==MatrixCount()-2)
        {
         owner=GetPointer(m_hidden);
         v.width=m_H+1;
         v.rows=m_H;
        }
      else if(id==3)
        {
         owner=GetPointer(m_tau);
         v.width=m_M+1;
         v.rows=m_H;
        }
      v.weights=owner.getWeightsIndex(); v.gradient=owner.WeightGradient().GetIndex();
      v.first=owner.getFirstMomentumIndex(); v.second=owner.getSecondMomentumIndex();
     }
   if(v.weights<0||v.gradient<0||v.first<0||v.second<0)
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| InitializeWeights - Seed encoder weights                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::InitializeWeights(const ulong seed,const uint stream)
  {
   if(m_revision!=0||stream!=(uint)m_iqn)
     {
      ReturnFalse
     }
   ulong ordinal=0;
   for(int i=0;i<MatrixCount();i++)
     {
      SAQCMatrix v;
      if(!MatrixView(i,v))
        {
         ReturnFalse
        }
      int kernel=def_k_AQCNetInit;
      float scale=(float)(WeightsMultiplier/MathSqrt(i==1||i==2?m_H+1:v.width));
      setBuffer(kernel,0,v.weights)
      setArgument(kernel,1,seed)
      setArgument(kernel,2,(ulong)stream)
      setArgument(kernel,3,ordinal)
      setArgument(kernel,4,scale)
      uint offset[]={0},global[]={(uint)(v.rows*v.width)};
      kernelExecute(kernel,offset,global)
      if(!OpenCL.WaitAlphaGenKernel(kernel))
        {
         ReturnFalse
        }
      ordinal+=global[0];
     }
   m_revision=1;
   return(true);
  }

//+------------------------------------------------------------------+
//| ExportInferenceViews - Publish a read-only view of live weights. |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::ExportInferenceViews(SAQCInferenceView &view)
  {
   view.owner=NULL;
   if(!m_revision||!OpenCL)
     {
      ReturnFalse
     }

   const int count=MatrixCount();
   for(int i=0;i<count;i++)
     {
      SAQCMatrix v;
      if(!MatrixView(i,v))
        {
         ReturnFalse
        }
      view.matrices[i].buffer=v.weights;
      view.matrices[i].width=v.width;
      view.matrices[i].rows=v.rows;
      view.matrices[i].layout=v.layout;
     }

   view.context=OpenCL;
   view.revision=m_revision;
   view.lifetime=m_viewLifetime;
   view.length=m_L;
   view.matrix_count=count;
   view.iqn=m_iqn;
   view.owner=GetPointer(this);
   return(true);
  }

//+------------------------------------------------------------------+
//| ValidateInferenceView - Check the producing owner and buffers.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::ValidateInferenceView(const SAQCInferenceView &view)
  {
   if(!m_revision||!OpenCL||view.owner!=GetPointer(this)||view.context!=OpenCL||
      view.revision!=m_revision||view.lifetime!=m_viewLifetime||view.length!=m_L||
      view.matrix_count!=MatrixCount()||view.iqn!=m_iqn)
     {
      ReturnFalseEx("AlphaQCM inference view no longer identifies the current weights")
     }

   for(int i=0;i<view.matrix_count;i++)
     {
      SAQCMatrix v;
      if(!MatrixView(i,v)||view.matrices[i].buffer!=v.weights||view.matrices[i].width!=v.width||
         view.matrices[i].rows!=v.rows||view.matrices[i].layout!=v.layout||
         OpenCL.GetBufferSize(v.weights)!=(ulong)v.width*v.rows*sizeof(float))
        {
         ReturnFalseEx(StringFormat("AlphaQCM inference matrix %d has an incompatible handle or layout",i))
        }
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Workspace - Allocate encoder work buffers                        |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Workspace(const bool backward)
  {
   const long B=m_B,L=m_L,H=m_H,E=m_E,R=B*m_N;
   long size[AQX_COUNT];
   ArrayInitialize(size,0);
   size[AQX_EMBED]=B*L*E;
   size[AQX_TAPE1]=B*L*H*(bTrain?6:1); size[AQX_TAPE2]=B*6*H*(bTrain?L:1);
   size[AQX_STATE]=B*2*H; size[AQX_MEAN]=B*H;
   size[AQX_HIDDEN]=R*H; size[AQX_RESULT]=R*m_A;
   if(m_iqn)
     {
      size[AQX_COS]=R*m_M;
      size[AQX_PHI]=R*H;
      size[AQX_PRODUCT]=R*H;
     }
   if(backward)
     {
      size[AQX_GHIDDEN]=R*H; size[AQX_GMEAN]=B*H;
      if(m_iqn)
        {
         size[AQX_GPRODUCT]=R*H;
         size[AQX_GPHI]=R*H;
        }
      size[AQX_GINTER]=B*L*H; size[AQX_GEMBED]=B*L*E; size[AQX_GATES]=B*L*4*H;
      size[AQX_FLAGS]=AQX_COUNT+MatrixCount();
     }
   for(int i=0;i<AQX_COUNT;i++)
      if(size[i]>0&&!AQCNetStorage(OpenCL,m_work[i],size[i]))
        {
         ReturnFalse
        }
   return(true);
  }

//+------------------------------------------------------------------+
//| ReserveBatch - Reserve batch and quantile storage                |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::ReserveBatch(const int rows,const int quantiles)
  {
   if(m_L<1||m_H<1||rows<1||quantiles<1||(!m_iqn&&quantiles!=1)||
      (long)rows*quantiles>INT_MAX/(long)MathMax(m_A,MathMax(m_H,m_M))||
      (long)rows*m_L>INT_MAX/(6L*m_H))
     {
      ReturnFalse
     }

   m_B=rows;
   m_N=quantiles;
   return(ReserveInput(m_tokens,(long)rows*m_L)&&ReserveInput(m_lengths,rows)&&
          (!m_iqn||ReserveInput(m_taus,(long)rows*quantiles))&&Workspace(bTrain));
  }

//+------------------------------------------------------------------+
//| ReserveInput - Allocate a 32-bit token, length or quantile span. |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::ReserveInput(int &buffer,const long count)
  {
   if(count<1||!OpenCL||!((CAlphaQCMOpenCL *)OpenCL).CanAllocate((ulong)count,sizeof(uint)))
     {
      ReturnFalse
     }

   const ulong bytes=(ulong)count*sizeof(uint);
   if(buffer>=0&&OpenCL.GetBufferSize(buffer)==bytes)
      return(true);

   ReleaseInput(buffer);
   buffer=OpenCL.AddBuffer((uint)bytes,CL_MEM_READ_WRITE);
   if(buffer<0)
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| ReleaseInput - Release only this encoder's allocation.           |
//+------------------------------------------------------------------+
void CNeuronAlphaQCMEncoderOCL::ReleaseInput(int &buffer)
  {
   if(OpenCL&&buffer>=0)
      OpenCL.BufferFree(buffer);
   buffer=-1;
  }

//+------------------------------------------------------------------+
//| ReleaseWorkspace - Release encoder inputs and work buffers       |
//+------------------------------------------------------------------+
void CNeuronAlphaQCMEncoderOCL::ReleaseWorkspace(void)
  {
   ReleaseInput(m_tokens);
   ReleaseInput(m_lengths);
   ReleaseInput(m_taus);
   for(int i=0;i<AQX_COUNT;i++)
     {
      if(m_work[i].GetIndex()>=0) m_work[i].BufferFree();
      m_work[i].Shutdown();
     }
   m_bound=false; m_prepared=-1; m_forward_ready=false; m_gradient_ready=false;
  }

//+------------------------------------------------------------------+
//| BindBatch - Stage a batch slice for both forward and backward.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::BindBatch(const int tokens,const int lengths,const int tau,
                                          const int rows,const int quantiles,const int first,
                                          const float position_scale)
  {
   m_bound=false;
   m_forward_ready=false;
   m_gradient_ready=false;
   m_prepared=-1;
   const long end=(long)first+rows;
   if(!OpenCL||!m_revision||m_L<1||!MathIsValidNumber(position_scale)||position_scale<=0||
      first<0||rows<1||quantiles<1||rows!=m_B||quantiles!=m_N||
      end>INT_MAX/m_L||end>INT_MAX/quantiles||m_tokens<0||m_lengths<0||
      tokens<0||lengths<0||(!m_iqn&&quantiles!=1)||(m_iqn&&(tau<0||m_taus<0)))
     {
      ReturnFalse
     }

   if(OpenCL.GetBufferSize(tokens)<(ulong)end*m_L*sizeof(int)||
      OpenCL.GetBufferSize(lengths)<(ulong)end*sizeof(int)||
      (m_iqn&&OpenCL.GetBufferSize(tau)<(ulong)end*quantiles*sizeof(float)))
     {
      ReturnFalseEx("AlphaQCM batch slice exceeds its replay buffers")
     }

//--- A Q batch has no tau. Bind valid buffers for the inactive arguments.
   const int kernel=def_k_AQCStageBatch;
   const int input_quantiles=(m_iqn?quantiles:0);
   setBuffer(kernel,0,tokens)
   setBuffer(kernel,1,lengths)
   setBuffer(kernel,2,(m_iqn?tau:lengths))
   setBuffer(kernel,3,m_tokens)
   setBuffer(kernel,4,m_lengths)
   setBuffer(kernel,5,(m_iqn?m_taus:m_lengths))
   setArgument(kernel,6,first)
   setArgument(kernel,7,m_L)
   setArgument(kernel,8,input_quantiles)
   if(!AQCNetRange(OpenCL,kernel,(int)MathMax(m_L,input_quantiles),rows))
     {
      ReturnFalse
     }

   m_position_scale=position_scale;
   m_bound=true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Recur - Run the selected recurrent layer                         |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Recur(const bool second)
  {
   int k=def_k_AQCRecur;
   setBuffer(k,0,m_work[second?AQX_TAPE1:AQX_EMBED].GetIndex())
   setBuffer(k,1,m_lengths)
   setBuffer(k,2,(second?m_second.RecurrentWeightsIndex():m_first.RecurrentWeightsIndex()))
   setBuffer(k,3,m_work[second?AQX_TAPE2:AQX_TAPE1].GetIndex())
   setBuffer(k,4,m_work[AQX_STATE].GetIndex())
   setBuffer(k,5,m_work[AQX_MEAN].GetIndex())
   setArgument(k,6,m_L)
   setArgument(k,7,(second?m_H:m_E))
   setArgument(k,8,m_H)
   setArgument(k,9,(second?(bTrain?6*m_H:m_H):m_E))
   setArgument(k,10,(bTrain?0:(second?2:1)))
   setBuffer(k,11,m_work[AQX_TAPE2].GetIndex())
   return(AQCNetGroups(OpenCL,k,m_B));
  }

//+------------------------------------------------------------------+
//| DenseRows - Apply a dense transform to each row                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::DenseRows(const int x,const int w,const int y,const int rows,
                                          const int inputs,const int outputs,const int relu)
  {
   int k=def_k_AQCDenseRows;
   setBuffer(k,0,x)
   setBuffer(k,1,w)
   setBuffer(k,2,y)
   setArgument(k,3,inputs)
   setArgument(k,4,outputs)
   setArgument(k,5,relu)
   return(AQCNetDots(OpenCL,k,outputs,rows));
  }

//+------------------------------------------------------------------+
//| feedForward - Encode the bound token batch                       |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::feedForward(CNeuronBaseOCL *previous)
  {
   if(!m_bound||!previous||previous.GetOpenCL()!=OpenCL||previous.Neurons()!=(uint)m_L)
     {
      ReturnFalse
     }
   m_bound=false;
   int k=def_k_AQCEmbed;
   setBuffer(k,0,m_tokens)
   setBuffer(k,1,m_lengths)
   setBuffer(k,2,m_embedding.getWeightsIndex())
   setBuffer(k,3,m_work[AQX_EMBED].GetIndex())
   setArgument(k,4,m_L)
   setArgument(k,5,m_E)
   setArgument(k,6,m_A)
   setArgument(k,7,m_position_scale)
   if(!AQCNetRange(OpenCL,k,m_E,m_B*m_L)||!Recur(false)||!Recur(true))
     {
      ReturnFalse
     }
   int state=m_work[AQX_MEAN].GetIndex(),rows=m_B*m_N;
   if(m_iqn)
     {
      k=def_k_AQCCosine;
      setBuffer(k,0,m_taus)
      setBuffer(k,1,m_work[AQX_COS].GetIndex())
      setArgument(k,2,m_M)
      if(!AQCNetRange(OpenCL,k,m_M,rows)||
         !DenseRows(m_work[AQX_COS].GetIndex(),m_tau.getWeightsIndex(),m_work[AQX_PHI].GetIndex(),rows,m_M,m_H,1))
        {
         ReturnFalse
        }
      k=def_k_AQCProduct;
      setBuffer(k,0,state)
      setBuffer(k,1,m_work[AQX_PHI].GetIndex())
      setBuffer(k,2,m_work[AQX_PRODUCT].GetIndex())
      setArgument(k,3,m_N)
      setArgument(k,4,m_H)
      if(!AQCNetRange(OpenCL,k,m_H,rows))
        {
         ReturnFalse
        }
      state=m_work[AQX_PRODUCT].GetIndex();
     }
   if(!DenseRows(state,m_hidden.getWeightsIndex(),m_work[AQX_HIDDEN].GetIndex(),rows,m_H,m_H,1)||
      !DenseRows(m_work[AQX_HIDDEN].GetIndex(),m_head.getWeightsIndex(),Result(),rows,m_H,m_A,0))
     {
      ReturnFalse
     }
   m_forward_ready=true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Reverse - Backpropagate through one recurrent layer              |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Reverse(const bool second)
  {
   int I=(second?m_H:m_E),stride=(second?6*m_H:m_E),k=def_k_AQCBPTT;
   int tape=m_work[second?AQX_TAPE2:AQX_TAPE1].GetIndex();
   SAQCMatrix v;
   if(!MatrixView(second?2:1,v))
     {
      ReturnFalse
     }
   setBuffer(k,0,tape)
   setBuffer(k,1,m_lengths)
   setBuffer(k,2,v.weights)
   setBuffer(k,3,m_work[second?AQX_GMEAN:AQX_GINTER].GetIndex())
   setBuffer(k,4,m_work[second?AQX_GINTER:AQX_GEMBED].GetIndex())
   setBuffer(k,5,m_work[AQX_GATES].GetIndex())
   setBuffer(k,6,m_work[AQX_STATE].GetIndex())
   setArgument(k,7,m_L)
   setArgument(k,8,I)
   setArgument(k,9,m_H)
   setArgument(k,10,(int)second)
   if(!AQCNetGroups(OpenCL,k,m_B))
     {
      ReturnFalse
     }
   k=def_k_AQCRecurOuter;
   setBuffer(k,0,m_work[second?AQX_TAPE1:AQX_EMBED].GetIndex())
   setBuffer(k,1,tape)
   setBuffer(k,2,m_work[AQX_GATES].GetIndex())
   setBuffer(k,3,v.gradient)
   setArgument(k,4,m_B*m_L)
   setArgument(k,5,m_L)
   setArgument(k,6,I)
   setArgument(k,7,m_H)
   setArgument(k,8,stride)
   return(AQCNetGroups(OpenCL,k,v.width*v.rows));
  }

//+------------------------------------------------------------------+
//| DenseBackward - Accumulate dense gradients                       |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::DenseBackward(const int x,const int dy,const int dx,const int parameter,
                                              const int rows,const bool relu)
  {
   SAQCMatrix v;
   if(!MatrixView(parameter,v))
     {
      ReturnFalse
     }
   int k=def_k_AQCDenseOuter,I=v.width-1;
   setBuffer(k,0,x)
   setBuffer(k,1,dy)
   setBuffer(k,2,v.gradient)
   setArgument(k,3,rows)
   setArgument(k,4,I)
   setArgument(k,5,v.rows)
   if(!AQCNetGroups(OpenCL,k,v.width*v.rows))
     {
      ReturnFalse
     }
   if(dx<0)
      return(true);
   k=def_k_AQCDenseInput;
   setBuffer(k,0,dy)
   setBuffer(k,1,v.weights)
   setBuffer(k,2,x)
   setBuffer(k,3,dx)
   setArgument(k,4,I)
   setArgument(k,5,v.rows)
   setArgument(k,6,(int)relu)
   return(AQCNetDots(OpenCL,k,I,rows));
  }

//+------------------------------------------------------------------+
//| BeginUpdate - Reset gradient state for an update                 |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::BeginUpdate(void)
  {
   if(!bTrain)
     {
      ReturnFalse
     }
   m_gradient_ready=false; m_prepared=-1;
   for(int i=0;i<MatrixCount();i++)
     {
      SAQCMatrix v;
      if(!MatrixView(i,v))
        {
         ReturnFalse
        }
      int k=def_k_AQCZeroGradient;
      setBuffer(k,0,v.gradient)
      uint offset[]={0},global[]={(uint)(v.width*v.rows)};
      kernelExecute(k,offset,global)
      if(!OpenCL.WaitAlphaGenKernel(k))
        {
         ReturnFalse
        }
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| PrepareGradient - Bind the output gradient                       |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::PrepareGradient(const int gradient)
  {
   if(!bTrain||!m_forward_ready||m_gradient_ready||gradient<0)
     {
      ReturnFalse
     }
   m_prepared=gradient;
   return(true);
  }

//+------------------------------------------------------------------+
//| RunBackward - Backpropagate through the encoder                  |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::RunBackward(void)
  {
   if(!m_forward_ready||m_prepared<0||!Workspace(true))
     {
      ReturnFalse
     }
   int rows=m_N*m_B,state=m_work[m_iqn?AQX_PRODUCT:AQX_MEAN].GetIndex();
   int dv=m_work[m_iqn?AQX_GPRODUCT:AQX_GMEAN].GetIndex();
   if(!DenseBackward(m_work[AQX_HIDDEN].GetIndex(),m_prepared,m_work[AQX_GHIDDEN].GetIndex(),
                     MatrixCount()-1,rows,true)||
      !DenseBackward(state,m_work[AQX_GHIDDEN].GetIndex(),dv,MatrixCount()-2,rows,false))
     {
      ReturnFalse
     }
   if(m_iqn)
     {
      int k=def_k_AQCProductGrad;
      setBuffer(k,0,dv)
      setBuffer(k,1,m_work[AQX_MEAN].GetIndex())
      setBuffer(k,2,m_work[AQX_PHI].GetIndex())
      setBuffer(k,3,m_work[AQX_GMEAN].GetIndex())
      setBuffer(k,4,m_work[AQX_GPHI].GetIndex())
      setArgument(k,5,m_N)
      setArgument(k,6,m_H)
      if(!AQCNetGroups(OpenCL,k,m_B*m_H)||
         !DenseBackward(m_work[AQX_COS].GetIndex(),m_work[AQX_GPHI].GetIndex(),-1,3,rows,false))
        {
         ReturnFalse
        }
     }
   if(!Reverse(true)||!Reverse(false))
     {
      ReturnFalse
     }
   int k=def_k_AQCEmbedOuter;
   setBuffer(k,0,m_tokens)
   setBuffer(k,1,m_lengths)
   setBuffer(k,2,m_work[AQX_GEMBED].GetIndex())
   setBuffer(k,3,m_embedding.WeightGradient().GetIndex())
   setArgument(k,4,m_B*m_L)
   setArgument(k,5,m_L)
   setArgument(k,6,m_E)
   if(!AQCNetGroups(OpenCL,k,m_A*m_E))
     {
      ReturnFalse
     }
   m_forward_ready=false; m_gradient_ready=true; m_prepared=-1;
   return(true);
  }

// Both CNet reverse passes visit the layer. Only calcInputGradients accumulates.
// Discrete token input has no upstream derivative.
//+------------------------------------------------------------------+
//| calcInputGradients - Accumulate encoder gradients                |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::calcInputGradients(CNeuronBaseOCL *previous)
  {
   return(RunBackward());
  }

//+------------------------------------------------------------------+
//| updateInputWeights - Check encoder gradient readiness            |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::updateInputWeights(CNeuronBaseOCL *previous)
  {
   if(!m_gradient_ready)
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Backward - Bind gradients and backpropagate                      |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Backward(const int gradient)
  {
   if(!PrepareGradient(gradient))
     {
      ReturnFalse
     }
   return(RunBackward());
  }

//+------------------------------------------------------------------+
//| Preflight - Check the pending optimizer update                   |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Preflight(const int flags,const int offset,const bool parameters)
  {
   if(!m_gradient_ready||flags<0||offset<0||m_revision==ULONG_MAX)
     {
      ReturnFalse
     }
   int out=m_work[AQX_FLAGS].GetIndex(),count=0;
   for(int i=0;parameters&&i<MatrixCount();i++)
     {
      SAQCMatrix v;
      if(!MatrixView(i,v))
        {
         ReturnFalse
        }
      int k=def_k_AQCParameterCheck;
      setBuffer(k,0,v.weights)
      setBuffer(k,1,v.gradient)
      setBuffer(k,2,v.first)
      setBuffer(k,3,v.second)
      setBuffer(k,4,out)
      setArgument(k,5,count)
      setArgument(k,6,v.width*v.rows)
      setArgument(k,7,v.width)
      setArgument(k,8,lr)
      setArgument(k,9,b1)
      setArgument(k,10,b2)
      if(!AQCNetGroups(OpenCL,k,1))
        {
         ReturnFalse
        }
      count++;
     }
   int k=def_k_AQCSpanCheck;
   for(int i=0;i<AQX_FLAGS;i++)
     {
      if(m_work[i].GetIndex()<0) continue;
      setBuffer(k,0,m_work[i].GetIndex())
      setBuffer(k,1,out)
      setArgument(k,2,count)
      setArgument(k,3,m_work[i].Total())
      setArgument(k,4,0)
      if(!AQCNetGroups(OpenCL,k,1))
        {
         ReturnFalse
        }
      count++;
     }
   setBuffer(k,0,out)
   setBuffer(k,1,flags)
   setArgument(k,2,offset)
   setArgument(k,3,count)
   setArgument(k,4,1)
   return(AQCNetGroups(OpenCL,k,1));
  }

//+------------------------------------------------------------------+
//| Apply - Apply the prepared optimizer update                      |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Apply(void)
  {
   if(!m_gradient_ready||m_revision==ULONG_MAX)
     {
      ReturnFalse
     }
   m_gradient_ready=false;
   const int kernel=def_k_LSTM_UpdateWeightsAdam;
   if(!m_embedding.Apply()||!OpenCL.WaitAlphaGenKernel(kernel)||
      !m_first.Apply()||!OpenCL.WaitAlphaGenKernel(kernel)||
      !m_second.Apply()||!OpenCL.WaitAlphaGenKernel(kernel)||
      (m_iqn&&(!m_tau.Apply()||!OpenCL.WaitAlphaGenKernel(kernel)))||
      !m_hidden.Apply()||!OpenCL.WaitAlphaGenKernel(kernel)||
      !m_head.Apply()||!OpenCL.WaitAlphaGenKernel(kernel))
     {
      ReturnFalse
     }
   m_revision++;
   return(true);
  }

//+------------------------------------------------------------------+
//| WeightsUpdate - Copy weights from a matching encoder             |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::WeightsUpdate(CNeuronBaseOCL *source,float fraction)
  {
   if(!source||source.Type()!=Type()||source.GetOpenCL()!=OpenCL||fraction!=1.0f)
     {
      ReturnFalse
     }
   CNeuronAlphaQCMEncoderOCL *net=source;
   if(net.m_L!=m_L||net.m_A!=m_A||net.m_E!=m_E||net.m_H!=m_H||net.m_M!=m_M||
      net.m_iqn!=m_iqn||net.m_revision==0||net==GetPointer(this))
     {
      ReturnFalse
     }
   for(int i=0;i<MatrixCount();i++)
     {
      SAQCMatrix a;
      SAQCMatrix b;
      if(!net.MatrixView(i,a)||!MatrixView(i,b)||a.weights==b.weights)
        {
         ReturnFalse
        }
      int k=def_k_AGCopySlice,count=b.rows*b.width;
      setBuffer(k,0,a.weights)
      setArgument(k,1,0)
      setBuffer(k,2,b.weights)
      setArgument(k,3,0)
      setArgument(k,4,count)
      uint offset[]={0},global[]={(uint)count};
      kernelExecute(k,offset,global)
      if(!OpenCL.WaitAlphaGenKernel(k))
        {
         ReturnFalse
        }
     }
   m_revision=net.m_revision; m_forward_ready=false; m_gradient_ready=false; m_bound=false;
   return(true);
  }

//+------------------------------------------------------------------+
//| Save - Write the encoder configuration and weights               |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Save(const int handle)
  {
   if(!m_revision||!CNeuronBaseOCL::Save(handle))
     {
      ReturnFalse
     }
   dFileWriteUInt(handle,1)
   dFileWriteUInt(handle,m_iqn)
   dFileWriteUInt(handle,m_L)
   dFileWriteUInt(handle,m_A)
   dFileWriteUInt(handle,m_E)
   dFileWriteUInt(handle,m_H)
   dFileWriteUInt(handle,m_M)
   if(FileWriteLong(handle,(long)m_revision)!=8||!m_embedding.SaveState(handle)||
      !m_first.SaveState(handle)||!m_second.SaveState(handle)||(m_iqn&&!m_tau.SaveState(handle))||
      !m_hidden.SaveState(handle)||!m_head.SaveState(handle))
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Load - Restore encoder configuration and weights                 |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMEncoderOCL::Load(const int handle)
  {
   if(m_viewLifetime==ULONG_MAX)
     {
      ReturnFalse
     }
   m_viewLifetime++;
   if(!CNeuronBaseOCL::Load(handle))
     {
      ReturnFalse
     }
   uint version;
   uint iqn;
   dFileReadUInt(handle,version)
   dFileReadUInt(handle,iqn)
   if(version!=1||iqn>1)
     {
      ReturnFalse
     }
   m_iqn=(iqn==1);
   m_L=FileReadInteger(handle);
   m_A=FileReadInteger(handle);
   m_E=FileReadInteger(handle);
   m_H=FileReadInteger(handle);
   m_M=FileReadInteger(handle);
   m_revision=(ulong)FileReadLong(handle);
   if(!m_revision||!Geometry()||!m_embedding.LoadState(handle,OpenCL)||!m_embedding.Geometry(m_E-1,m_A)||
      !m_first.LoadState(handle,OpenCL)||!m_first.Geometry(m_E,m_H,m_L)||
      !m_second.LoadState(handle,OpenCL)||!m_second.Geometry(m_H,m_H,m_L)||
      (m_iqn&&(!m_tau.LoadState(handle,OpenCL)||!m_tau.Geometry(m_M,m_H)))||
      !m_hidden.LoadState(handle,OpenCL)||!m_hidden.Geometry(m_H,m_H)||
      !m_head.LoadState(handle,OpenCL)||!m_head.Geometry(m_H,m_A))
     {
      ReturnFalse
     }
   m_forward_ready=false; m_gradient_ready=false; m_bound=false;
   return(true);
  }

// Validate the exact CArrayFloat/CBufferFloat framing emitted by the donor.
// Inspection and TRAIN restore share this bounded streaming validation.
//+------------------------------------------------------------------+
//| AQCReadNetworkSpan - Validate a serialized float buffer          |
//+------------------------------------------------------------------+
bool AQCReadNetworkSpan(const int handle,const int expected,const bool nonnegative)
  {
   ulong at=FileTell(handle),size=FileSize(handle);
   if(expected<0||at>size||size-at<16+(ulong)expected*sizeof(float)||
      FileReadLong(handle)!=-1||FileReadInteger(handle)!=defBufferDouble||
      FileReadInteger(handle)!=expected)
     {
      ReturnFalse
     }
   float values[512];
   for(int base=0;base<expected;base+=512)
     {
      uint count=(uint)MathMin(512,expected-base);
      if(FileReadArray(handle,values,0,count)!=count)
        {
         ReturnFalse
        }
      for(uint i=0;i<count;i++)
         if(!MathIsValidNumber(values[i])||(nonnegative&&values[i]<0))
           {
            ReturnFalse
           }
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| AQCReadNetworkBase - Validate the serialized base layer fields   |
//+------------------------------------------------------------------+
bool AQCReadNetworkBase(const int handle,const int type,const int neurons,const int parameters)
  {
   if(FileReadInteger(handle)!=type||FileReadInteger(handle)!=None||FileReadInteger(handle)!=ADAM)
     {
      ReturnFalse
     }
   uint batch=(uint)FileReadInteger(handle),step=(uint)FileReadInteger(handle);
   int train=FileReadInteger(handle);
   if(batch==0||step==0||(train!=0&&train!=1))
     {
      ReturnFalse
     }
   for(int i=0;i<3;i++)
      if(!AQCReadNetworkSpan(handle,neurons,false))
        {
         ReturnFalse
        }
   int has_weights=FileReadInteger(handle);
   if(has_weights!=(int)(parameters>0))
     {
      ReturnFalse
     }
   if(parameters>0)
      for(int i=0;i<3;i++)
         if(!AQCReadNetworkSpan(handle,parameters,i==2))
           {
            ReturnFalse
           }
   return(true);
  }

//+------------------------------------------------------------------+
//| AlphaQCMSkipNetwork - Validate and skip a frozen search network  |
//+------------------------------------------------------------------+
bool AlphaQCMSkipNetwork(const int handle,const bool iqn,const int length,const int actions,
                         const int embedding,const int hidden,const int frequencies)
  {
   if(!AQCReadNetworkBase(handle,defNeuronAlphaQCMEncoderOCL,actions,0)||
      FileReadInteger(handle)!=1||FileReadInteger(handle)!=(int)iqn||
      FileReadInteger(handle)!=length||FileReadInteger(handle)!=actions||
      FileReadInteger(handle)!=embedding||FileReadInteger(handle)!=hidden||
      FileReadInteger(handle)!=frequencies||FileReadLong(handle)==0)
     {
      ReturnFalse
     }
   const int owners=(iqn?6:5);
   for(int i=0;i<owners;i++)
     {
      if(i==1||i==2)
        {
         int inputs=i==1?embedding:hidden;
         if(!AQCReadNetworkBase(handle,defNeuronLSTMOCL,hidden,0)||
            FileReadInteger(handle)!=inputs||FileReadInteger(handle)!=1)
           {
            ReturnFalse
           }
         int count=(inputs+hidden+1)*4*hidden;
         for(int field=0;field<3;field++)
            if(!AQCReadNetworkSpan(handle,count,field==2))
              {
               ReturnFalse
              }
         if(FileReadInteger(handle)==0||FileReadInteger(handle)!=length)
           {
            ReturnFalse
           }
        }
      else
        {
         int inputs=i==0?embedding-1:(iqn&&i==3?frequencies:hidden);
         int outputs=i==0||i==owners-1?actions:hidden;
         if(!AQCReadNetworkBase(handle,defNeuronBaseOCL,inputs,(inputs+1)*outputs)||
            FileReadInteger(handle)==0)
           {
            ReturnFalse
           }
        }
     }
   return(true);
  }

#endif
