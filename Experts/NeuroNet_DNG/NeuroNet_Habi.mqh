//+------------------------------------------------------------------+
//|                                                NeuroNet_Habi.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#ifndef NEURONET_HABI_MQH
#define NEURONET_HABI_MQH

//+------------------------------------------------------------------+
//| EHabiMode - frame modes accepted by the Habi layers              |
//+------------------------------------------------------------------+
enum EHabiMode
  {
   F_GAUSSIAN_BC=1,F_GAUSSIAN_EVAL=2,F_GAUSSIAN_POLICY=3,
   B_PRIOR_KL=11,B_POSTERIOR_BC=12,B_PRIOR_POLICY=13,
   F_DECODER_SAMPLE=21,B_DECODER_BC=23,B_DECODER_POLICY=24,
   F_CRITIC_TRAIN=31,B_CRITIC_MC=32,B_CRITIC_TD=33,
   B_CRITIC_POLICY=34,F_CRITIC_EVAL=35,F_CRITIC_POLICY=36
  };
//+------------------------------------------------------------------+
//| SHabiFrame - forward and backward frame state of a leaf          |
//+------------------------------------------------------------------+
struct SHabiFrame
  {
   uint              id;
   int               mode;
   bool              live;
   void              Reset()
     {
      id=0;
      mode=0;
      live=false;
     }
   bool              Header(CBufferFloat *ctx,int &m,uint &token);
   bool              Forward(CBufferFloat *ctx,int payload,bool training);
   bool              Backward(CBufferFloat *ctx,int payload);
  };
//+------------------------------------------------------------------+
//| SHabiFrame::Header - parse the frame header                      |
//+------------------------------------------------------------------+
bool SHabiFrame::Header(CBufferFloat *ctx,int &m,uint &token)
  {
   if(!ctx || !ctx.BufferRead() || ctx.Total()<4 || ctx.At(0)!=3)
      return(false);

   for(int i=0;i<ctx.Total();i++)
      if(!MathIsValidNumber(ctx.At(i)))
         return(false);

   double lo=ctx.At(2);
   double hi=ctx.At(3);
   double md=ctx.At(1);

   if(lo<0 || lo>65535 || hi<0 || hi>65535 || lo!=MathFloor(lo) ||
      hi!=MathFloor(hi) || md!=MathFloor(md))
      return(false);

   m=(int)md;
   token=(uint)lo|((uint)hi<<16);

   return(true);
  }
//+------------------------------------------------------------------+
//| SHabiFrame::Forward - open a frame from the context              |
//+------------------------------------------------------------------+
bool SHabiFrame::Forward(CBufferFloat *ctx,int payload,bool training)
  {
   int m;
   uint token;

   if(live || !Header(ctx,m,token) || ctx.Total()!=4+payload)
      return(false);

   if(m!=1 && m!=2 && m!=3 && m!=21 && m!=31 && m!=35 && m!=36)
      return(false);

   id=token;
   mode=m;
   live=training && m!=2 && m!=35;

   return(true);
  }
//+------------------------------------------------------------------+
//| SHabiFrame::Backward - match a backward frame                    |
//+------------------------------------------------------------------+
bool SHabiFrame::Backward(CBufferFloat *ctx,int payload)
  {
   int m;
   uint token;

   if(!live || !Header(ctx,m,token) || ctx.Total()!=4+payload || token!=id)
      return(false);

   bool pair=(mode==1 && (m==11 || m==12)) || (mode==3 && m==13) ||
             (mode==21 && (m==23 || m==24)) || (mode==31 && (m==32 || m==33)) ||
             (mode==36 && m==34);

   if(!pair)
      return(false);

   live=false;

   return(true);
  }
//+------------------------------------------------------------------+
//| HabiScratch - allocate or rebind a scratch buffer                |
//+------------------------------------------------------------------+
bool HabiScratch(CBufferFloat *buf,COpenCLMy *cl,int size)
  {
   if(!buf || !cl || size<1)
      return(false);

   if(buf.Total()!=size)
     {
      if(buf.GetIndex()>=0 && !buf.BufferFree())
         return(false);

      if(!buf.BufferInit(size,0))
         return(false);
     }

   if(buf.GetOpenCL()!=cl || buf.GetIndex()<0)
      return(buf.BufferCreate(cl));

   return(true);
  }
//+------------------------------------------------------------------+
//| HabiPayload - copy a payload slice of a buffer                   |
//+------------------------------------------------------------------+
bool HabiPayload(CBufferFloat *ctx,int offset,int size,CBufferFloat *dst,COpenCLMy *cl)
  {
   if(!ctx || offset<0 || size<1 || offset+size>ctx.Total() || !HabiScratch(dst,cl,size))
      return(false);

   for(int i=0;i<size;i++)
      if(!dst.Update(i,ctx.At(offset+i)))
         return(false);

   return(dst.BufferWrite());
  }
//+------------------------------------------------------------------+
//| HabiTransferBuffer - move a buffer to a context                  |
//+------------------------------------------------------------------+
bool HabiTransferBuffer(CBufferFloat *buf,COpenCLMy *cl)
  {
   if(!buf || buf.Total()==0 || buf.GetOpenCL()==cl)
      return(true);

   if(buf.GetIndex()>=0 && !buf.BufferRead())
      return(false);

   return(buf.BufferCreate(cl));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL - base leaf of the Habi protocol              |
//+------------------------------------------------------------------+
class CNeuronHabiLeafOCL : public CNeuronBaseOCL
  {
public:
                     CNeuronHabiLeafOCL();
   virtual bool      Configure(CLayerDescription *d);
   virtual bool      SupportsOpenCLChecked(void) const override
     {
      return(true);
     }
   virtual bool      SetOpenCLChecked(COpenCLMy *cl);
   virtual void      SetOpenCL(COpenCLMy *cl) override
     {
      SetOpenCLChecked(cl);
     }
   void              ReleaseFrame()
     {
      frame.Reset();
     }
   uint              Rows() const
     {
      return(N);
     }
   uint              Latent() const
     {
      return(L);
     }
   bool              Geometry(uint n,uint l,uint a,uint m,bool post) const
     {
      return((n==0 || N==n) && L==l && A==a && M==m && posterior==post);
     }
   virtual bool      Clear() override
     {
      frame.Reset();

      return(CNeuronBaseOCL::Clear());
     }
   virtual bool      WeightsUpdate(CNeuronBaseOCL *s,float tau) override
     {
      return(s && s.Type()==Type() && s.Neurons()==Neurons() && tau>=0 && tau<=1);
     }
   virtual bool      Save(const int h);
   virtual bool      Load(const int h);
protected:
   uint              N;
   uint              L;
   uint              A;
   uint              M;
   uint              Norm;
   bool              posterior;
   SHabiFrame        frame;
   CBufferFloat      scratch[9];
   virtual bool      updateInputWeights(CNeuronBaseOCL *p) override
     {
      return(true);
     }
   virtual bool      updateInputWeights(CNeuronBaseOCL *p,CBufferFloat *ctx) override
     {
      return(updateInputWeights(p));
     }
   bool              Ready(int slot,int count)
     {
      return(HabiScratch(GetPointer(scratch[slot]),OpenCL,count));
     }
   bool              Finite(int count);
   bool              Pass(CNeuronBaseOCL *p)
     {
      return(p && p.Neurons()==Neurons() && CopyBufferRaw(p.getOutput(),Output,Neurons()));
     }
  };
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::CNeuronHabiLeafOCL - set defaults            |
//+------------------------------------------------------------------+
CNeuronHabiLeafOCL::CNeuronHabiLeafOCL()
  {
   N=1;
   L=1;
   A=6;
   M=45;
   Norm=100000;
   posterior=false;

   frame.Reset();
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::Configure - apply a layer description        |
//+------------------------------------------------------------------+
bool CNeuronHabiLeafOCL::Configure(CLayerDescription *d)
  {
   N=d.window;
   L=d.window_out;
   A=d.step;
   M=d.layers;
   Norm=d.variables;
   posterior=d.probability>0;

   return(N>0 && L>0 && A==6 && M>0 && Norm>1);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::SetOpenCLChecked - attach the context        |
//+------------------------------------------------------------------+
bool CNeuronHabiLeafOCL::SetOpenCLChecked(COpenCLMy *cl)
  {
   for(int i=0;i<9;i++)
      if(!HabiTransferBuffer(GetPointer(scratch[i]),cl))
         return(false);

   return(CNeuronBaseOCL::SetOpenCLChecked(cl));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::Finite - check that buffer values are finite |
//+------------------------------------------------------------------+
bool CNeuronHabiLeafOCL::Finite(int count)
  {
   if(!Ready(7,1))
      return(false);

   setBuffer(def_k_DPCCOperatorReduce,0,scratch[6].GetIndex())
   setBuffer(def_k_DPCCOperatorReduce,1,scratch[7].GetIndex())
   setArgument(def_k_DPCCOperatorReduce,2,count)

   uint off[]= {0};
   uint sz[]= {1};

   kernelExecute(def_k_DPCCOperatorReduce,off,sz)

   if(!scratch[7].BufferRead())
      return(false);

   return(scratch[7].At(0)==0);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::Save - write the leaf state                  |
//+------------------------------------------------------------------+
bool CNeuronHabiLeafOCL::Save(const int h)
  {
   if(!CNeuronBaseOCL::Save(h))
      return(false);

   dFileWriteUInt(h,N)
   dFileWriteUInt(h,L)
   dFileWriteUInt(h,A)
   dFileWriteUInt(h,M)
   dFileWriteUInt(h,Norm)
   dFileWriteUInt(h,posterior)

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiLeafOCL::Load - read the leaf state                   |
//+------------------------------------------------------------------+
bool CNeuronHabiLeafOCL::Load(const int h)
  {
   frame.Reset();

   if(!CNeuronBaseOCL::Load(h) || FileSize(h)-FileTell(h)<24)
      return(false);

   uint stored=0;

   dFileReadUInt(h,N)
   dFileReadUInt(h,L)
   dFileReadUInt(h,A)
   dFileReadUInt(h,M)
   dFileReadUInt(h,Norm)

   dFileReadUInt(h,stored)
   posterior=(stored!=0);

   return(N>0 && L>0 && A==6 && M>0 && Norm>1 && Gradient.Total()==Output.Total());
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL - observation encoder of the market    |
//+------------------------------------------------------------------+
class CNeuronHabiObservationOCL : public CNeuronHabiLeafOCL
  {
public:
                     CNeuronHabiObservationOCL()
     {
      initialized=false;
     }
   virtual int       Type() const override
     {
      return(defNeuronHabiObservationOCL);
     }
   bool              BNReady();
   bool              BNStateValid()
     {
      return(!initialized || BNReady());
     }
   bool              BNGeometry();
   bool              CopyBN(CNeuronHabiObservationOCL *src);
   virtual bool      Configure(CLayerDescription *d) override;
   virtual bool      SetOpenCLChecked(COpenCLMy *cl) override;
   virtual bool      Save(const int h) override;
   virtual bool      Load(const int h) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *src,float tau) override;
protected:
   CNeuronBaseOCL      market;
   CNeuronBatchNormOCL bn;
   bool                initialized;
   virtual bool        feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override;
   virtual bool        calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                          ENUM_ACTIVATION ac=None) override;
   virtual bool        updateInputWeights(CNeuronBaseOCL *p) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::feedForward - encode observation      |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx)
  {
   if(!p || p.Neurons()!=Neurons() || !frame.Forward(ctx,0,bTrain) ||
      !Ready(0,Neurons()-M))
      return(false);

   if(!DeConcat(market.getOutput(),GetPointer(scratch[0]),p.getOutput(),M,Neurons()-M,1))
      return(false);

   bool startup=!initialized;

   if(startup)
     {
      if(!bTrain || frame.mode==F_GAUSSIAN_EVAL || !market.getOutput().BufferRead())
         return(false);

      CBufferFloat *o=bn.getBatchOptions();

      for(uint j=0;j<M;j++)
         for(int k=0;k<9;k++)
            if(!o.Update((int)j*9+k,k==0?market.getOutput().At(j):(k==3?1.0f:0.0f)))
               return(false);

      if(!o.BufferWrite() || !bn.SetBatchCount(2))
         return(false);
     }

   bool accumulate=!startup && bTrain && frame.mode==F_GAUSSIAN_BC;
   int count=bn.GetBatchCount();
   bn.TrainMode(accumulate);
   bool ok=bn.FeedForward(GetPointer(market));

   if(!accumulate && !bn.SetBatchCount(count))
      return(false);

   if(!ok || !Concat(bn.getOutput(),GetPointer(scratch[0]),Output,M,Neurons()-M,1))
      return(false);

   initialized=true;

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::calcInputGradients - backward pass    |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token))
      return(false);

   int payload=(int)(mode==11?2*L+1:(mode==12?4*L+1:2*L));

   if(!frame.Backward(ctx,payload) || !Ready(1,Neurons()-M))
      return(false);

   if(!DeConcat(bn.getGradient(),GetPointer(scratch[1]),Gradient,M,Neurons()-M,1) ||
      !market.CalcHiddenGradients(GetPointer(bn)))
      return(false);

   return(Concat(market.getGradient(),GetPointer(scratch[1]),p.getGradient(),M,Neurons()-M,1));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::updateInputWeights - update BN        |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::updateInputWeights(CNeuronBaseOCL *p)
  {
   return(bn.UpdateInputWeights(GetPointer(market)));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::BNReady - check batch statistics      |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::BNReady()
  {
   CBufferFloat *options=bn.getBatchOptions();

   if(!initialized || bn.GetBatchCount()<2 || !options || options.Total()!=9*M ||
      !options.BufferRead())
      return(false);

   for(int j=0;j<options.Total();j++)
      if(!MathIsValidNumber(options.At(j)) || (j%9==1 && options.At(j)<0))
         return(false);

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::BNGeometry - check the BN layout      |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::BNGeometry()
  {
   CLayerDescription *info=bn.GetLayerInfo();

   if(!info)
      return(false);

   bool ok=(bn.Neurons()==M && info.batch==Norm && bn.getBatchOptions()!=NULL &&
            bn.getBatchOptions().Total()==9*M);

   delete info;

   return(ok);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::CopyBN - copy BN from a source        |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::CopyBN(CNeuronHabiObservationOCL *src)
  {
   if(!src || src.M!=M || !src.BNReady() || !bn.WeightsUpdate(GetPointer(src.bn),1))
      return(false);

   initialized=true;

   return(bn.SetBatchCount(src.bn.GetBatchCount()));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::Configure - apply a description       |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::Configure(CLayerDescription *d)
  {
   if(!CNeuronHabiLeafOCL::Configure(d) || Neurons()!=M+13+(posterior?A:0))
      return(false);

   return(market.Init(0,0,OpenCL,M,ADAM,iBatch) && bn.Init(0,0,OpenCL,M,Norm,ADAM));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::SetOpenCLChecked - attach context     |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::SetOpenCLChecked(COpenCLMy *cl)
  {
   return(market.SetOpenCLChecked(cl) && bn.SetOpenCLChecked(cl) &&
          CNeuronHabiLeafOCL::SetOpenCLChecked(cl));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::Save - write encoder state            |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::Save(const int h)
  {
   if(!CNeuronHabiLeafOCL::Save(h))
      return(false);

   dFileWriteUInt(h,initialized)
   dFileWriteUInt(h,bn.GetBatchCount())

   return(bn.Save(h));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::Load - read encoder state             |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::Load(const int h)
  {
   if(!CNeuronHabiLeafOCL::Load(h))
      return(false);

   uint stored=0;

   dFileReadUInt(h,stored)
   initialized=(stored!=0);

   dFileReadUInt(h,stored)
   int count=(int)stored;

   dFileReadUInt(h,stored)
   if(stored!=defNeuronBatchNormOCL)
      return(false);

   bn.AttachOpenCL(OpenCL);

   return(bn.Load(h) && bn.SetBatchCount(count) && BNGeometry() && BNStateValid() &&
          market.Init(0,0,OpenCL,M,ADAM,iBatch));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiObservationOCL::WeightsUpdate - blend encoder weights |
//+------------------------------------------------------------------+
bool CNeuronHabiObservationOCL::WeightsUpdate(CNeuronBaseOCL *src,float tau)
  {
   if(!CNeuronHabiLeafOCL::WeightsUpdate(src,tau))
      return(false);

   CNeuronHabiObservationOCL *s=src;

   if(s.M!=M || !CNeuronBaseOCL::WeightsUpdate(src,tau) ||
      !bn.WeightsUpdate(GetPointer(s.bn),tau))
      return(false);

   initialized=s.initialized;

   return(bn.SetBatchCount(s.bn.GetBatchCount()));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL - gaussian posterior of a leaf            |
//+------------------------------------------------------------------+
class CNeuronHabiGaussianOCL : public CNeuronHabiLeafOCL
  {
public:
                     CNeuronHabiGaussianOCL();
   virtual int       Type() const override
     {
      return(defNeuronHabiGaussianOCL);
     }
   bool              Beta(bool use_adaptive,double value,double lr_beta,double k_target,
                          double r1,double r2,double epsilon);
   float             EffectiveBeta()
     {
      return((float)(adaptive?MathExp(b):beta_fixed));
     }
   virtual bool      Save(const int h) override;
   virtual bool      Load(const int h) override;
protected:
   bool              adaptive;
   double            beta_fixed;
   double            b;
   double            mb;
   double            vb;
   double            eta;
   double            rho1;
   double            rho2;
   double            eps;
   double            target;
   uint              bt;
   double            cached_K;
   virtual bool      feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                        ENUM_ACTIVATION ac=None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *p) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::CNeuronHabiGaussianOCL - set defaults    |
//+------------------------------------------------------------------+
CNeuronHabiGaussianOCL::CNeuronHabiGaussianOCL()
  {
   adaptive=false;
   beta_fixed=0;
   b=0;
   mb=0;
   vb=0;
   bt=0;
   eta=0;
   rho1=.9;
   rho2=.999;
   eps=1e-8;
   target=1;
   cached_K=0;
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::feedForward - evaluate the posterior     |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx)
  {
   if(!p || p.Neurons()!=2*L || Neurons()!=2*L || !frame.Forward(ctx,0,bTrain) ||
      !Ready(0,L) || !Ready(6,L))
      return(false);

   int k=def_k_HabiGaussianForward;

   setBuffer(k,0,p.getOutputIndex())
   setBuffer(k,1,Output.GetIndex())
   setBuffer(k,2,scratch[0].GetIndex())
   setBuffer(k,3,scratch[6].GetIndex())
   setArgument(k,4,(int)L)

   uint off[]= {0};
   uint sz[]= {L};

   kernelExecute(k,off,sz)

   return(Finite(L));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::calcInputGradients - backward pass       |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token) || (posterior?mode!=12:(mode!=11 && mode!=13)))
      return(false);

   int payload=(int)(mode==12?4*L+1:(mode==11?2*L+1:2*L));

   if(!frame.Backward(ctx,payload) || !Ready(1,2*L) || !Ready(2,2*L) || !Ready(3,L) ||
      !Ready(4,1) || !Ready(5,L) || !Ready(8,L))
      return(false);

   float beta=0;

   if(mode==13)
     {
      if(!HabiPayload(ctx,4,2*L,GetPointer(scratch[2]),OpenCL))
         return(false);
     }
   else
     {
      if(!HabiPayload(ctx,4,2*L,GetPointer(scratch[1]),OpenCL))
         return(false);

      beta=ctx.At(4+(mode==12?4:2)*L);

      if(beta<0)
         return(false);

      if(mode==12 && !HabiPayload(ctx,4+2*L,2*L,GetPointer(scratch[2]),OpenCL))
         return(false);
     }

   int k=def_k_HabiGaussianBackward;

   setBuffer(k,0,Output.GetIndex())
   setBuffer(k,1,scratch[1].GetIndex())
   setBuffer(k,2,scratch[2].GetIndex())
   setBuffer(k,3,scratch[0].GetIndex())
   setBuffer(k,4,scratch[5].GetIndex())
   setBuffer(k,5,scratch[8].GetIndex())
   setBuffer(k,6,scratch[3].GetIndex())
   setBuffer(k,7,scratch[6].GetIndex())
   setArgument(k,8,(int)L)
   setArgument(k,9,mode)
   setArgument(k,10,beta)

   uint off[]= {0};
   uint sz[]= {L};

   kernelExecute(k,off,sz)

   if(!Finite(L) || !Concat(GetPointer(scratch[5]),GetPointer(scratch[8]),Gradient,L,L,1) ||
      !DeActivation(p.getOutput(),p.getGradient(),Gradient,None))
      return(false);

   if(mode==12)
     {
      k=def_k_DPCCDenoiserBroadcastGradient;

      setBuffer(k,0,scratch[3].GetIndex())
      setBuffer(k,1,scratch[4].GetIndex())
      setBuffer(k,2,scratch[6].GetIndex())
      setArgument(k,3,(int)L)
      setArgument(k,4,1)

      sz[0]=1;

      kernelExecute(k,off,sz)

      if(!Finite(1) || !scratch[4].BufferRead())
         return(false);

      cached_K=scratch[4].At(0);
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::updateInputWeights - adapt beta          |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::updateInputWeights(CNeuronBaseOCL *p)
  {
   if(!posterior || !adaptive)
      return(true);

   double g=MathLog10(MathMax(cached_K,1e-9))-MathLog10(target);

   mb=rho1*mb+(1-rho1)*g;
   vb=rho2*vb+(1-rho2)*g*g;
   bt++;
   b+=eta*(mb/(1-MathPow(rho1,bt)))/(MathSqrt(vb/(1-MathPow(rho2,bt)))+eps);

   return(MathIsValidNumber(MathExp(b)));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::Beta - set the beta schedule             |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::Beta(bool use_adaptive,double value,double lr_beta,
                                  double k_target,double r1,double r2,double epsilon)
  {
   if(!MathIsValidNumber(value) || value<0 ||
      (use_adaptive && (value<=0 || lr_beta<=0 || k_target<=0 || r1<=0 || r1>=1 ||
                        r2<=0 || r2>=1 || epsilon<=0 ||
                        !MathIsValidNumber(lr_beta+k_target+r1+r2+epsilon))))
      return(false);

   adaptive=use_adaptive;
   beta_fixed=value;

   if(adaptive)
     {
      b=MathLog(value);
      rho1=r1;
      rho2=r2;
      eps=epsilon;
      eta=lr_beta;
      target=k_target;
     }

   mb=0;
   vb=0;
   bt=0;

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::Save - write the gaussian state          |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::Save(const int h)
  {
   if(!CNeuronHabiLeafOCL::Save(h))
      return(false);

   dFileWriteUInt(h,adaptive)
   dFileWriteUInt(h,bt)

   double v[]=
     {
      beta_fixed,b,mb,vb,eta,rho1,rho2,eps,target
     };

   return(FileWriteArray(h,v)==ArraySize(v));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiGaussianOCL::Load - read the gaussian state           |
//+------------------------------------------------------------------+
bool CNeuronHabiGaussianOCL::Load(const int h)
  {
   if(!CNeuronHabiLeafOCL::Load(h) || Neurons()!=2*L)
      return(false);

   uint stored=0;

   dFileReadUInt(h,stored)
   adaptive=(stored!=0);

   dFileReadUInt(h,bt)

   double v[9];

   if(FileReadArray(h,v)!=9)
      return(false);

   for(int i=0;i<9;i++)
      if(!MathIsValidNumber(v[i]))
         return(false);

   beta_fixed=v[0];
   b=v[1];
   mb=v[2];
   vb=v[3];
   eta=v[4];
   rho1=v[5];
   rho2=v[6];
   eps=v[7];
   target=v[8];

   return(beta_fixed>=0 && vb>=0 && rho1>=0 && rho1<1 && rho2>=0 && rho2<1 &&
          eps>0 && (!adaptive || (eta>0 && target>0 && MathExp(b)>0 &&
                                  MathIsValidNumber(MathExp(b)))));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL - VAE sampler of the decoder path              |
//+------------------------------------------------------------------+
class CNeuronHabiVAEOCL : public CVAE
  {
public:
                     CNeuronHabiVAEOCL();
   virtual int       Type() const override
     {
      return(defNeuronHabiVAEOCL);
     }
   virtual bool      Configure(CLayerDescription *d);
   void              ReleaseFrame()
     {
      frame.Reset();
     }
   uint              Rows() const
     {
      return(N);
     }
   uint              Latent() const
     {
      return(L);
     }
   virtual bool      Clear() override
     {
      frame.Reset();

      return(CVAE::Clear());
     }
   virtual bool      SupportsOpenCLChecked() const override
     {
      return(true);
     }
   virtual bool      SetOpenCLChecked(COpenCLMy *cl) override;
   virtual void      SetOpenCL(COpenCLMy *cl) override
     {
      SetOpenCLChecked(cl);
     }
   virtual bool      WeightsUpdate(CNeuronBaseOCL *src,float tau) override
     {
      return(src && src.Type()==Type() && src.Neurons()==Neurons());
     }
   virtual bool      Save(const int h) override;
   virtual bool      Load(const int h) override;
protected:
   uint              N;
   uint              L;
   SHabiFrame        frame;
   CBufferFloat      direct;
   CBufferFloat      flags;
   virtual bool      feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                        ENUM_ACTIVATION ac=None) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::CNeuronHabiVAEOCL - set defaults              |
//+------------------------------------------------------------------+
CNeuronHabiVAEOCL::CNeuronHabiVAEOCL()
  {
   N=1;
   L=1;

   frame.Reset();
   SetKLDMult(0);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::feedForward - sample the latent vector        |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx)
  {
   if(!p || p.Neurons()!=2*L || !frame.Forward(ctx,0,bTrain) || frame.mode!=21 ||
      !HabiScratch(GetPointer(flags),OpenCL,N*L))
      return(false);

   double draws[];

   if(!MathRandomNormal(0,1,N*L,draws) || !m_cRandom.AssignArray(draws) ||
      !m_cRandom.BufferWrite())
      return(false);

   int k=def_k_VAEFeedForward;

   setBuffer(k,0,p.getOutputIndex())
   setBuffer(k,1,m_cRandom.GetIndex())
   setBuffer(k,2,Output.GetIndex())
   setBuffer(k,3,flags.GetIndex())
   setArgument(k,4,1)

   uint off[]= {0,0};
   uint sz[]= {L,N};

   kernelExecute(k,off,sz)

   if(!flags.BufferRead())
      return(false);

   for(int i=0;i<flags.Total();i++)
      if(flags.At(i)!=0)
         return(false);

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::calcInputGradients - backward pass            |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token) || !frame.Backward(ctx,mode==23?6:N*6+N*L))
      return(false);

   if(mode==24)
     {
      if(!HabiPayload(ctx,4+N*6,N*L,GetPointer(direct),OpenCL) ||
         !SumAndNormalize(Gradient,GetPointer(direct),Gradient,N*L,false,0,0,0,1))
         return(false);
     }

   long memory=0;

   if(N>(uint)OpenCL.GetMaxWorkGroupSize() || N>(uint)OpenCL.GetMaxLocalSize(1) ||
      OpenCL.GetKernelInfoInteger(def_k_VAECalcHiddenGradient,CL_KERNEL_WORK_GROUP_SIZE)<N ||
      !OpenCL.GetLocalMemorySize(memory) ||
      OpenCL.GetKernelInfoInteger(def_k_VAECalcHiddenGradient,CL_KERNEL_LOCAL_MEM_SIZE)>memory)
      return(false);

   int k=def_k_VAECalcHiddenGradient;

   setBuffer(k,0,p.getOutputIndex())
   setBuffer(k,1,p.getGradientIndex())
   setBuffer(k,2,m_cRandom.GetIndex())
   setBuffer(k,3,Gradient.GetIndex())
   setArgument(k,4,0.0f)

   uint off[]= {0,0};
   uint sz[]= {L,N};
   uint local[]= {1,N};

   kernelExecuteLoc(k,off,sz,local)

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::Configure - apply a description               |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::Configure(CLayerDescription *d)
  {
   N=d.window;
   L=d.window_out;

   return(N>0 && L>0 && Neurons()==N*L);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::SetOpenCLChecked - attach the context         |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::SetOpenCLChecked(COpenCLMy *cl)
  {
   return(HabiTransferBuffer(GetPointer(direct),cl) &&
          HabiTransferBuffer(GetPointer(flags),cl) && CVAE::SetOpenCLChecked(cl));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::Save - write the sampler state                |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::Save(const int h)
  {
   if(!CVAE::Save(h))
      return(false);

   dFileWriteUInt(h,N)
   dFileWriteUInt(h,L)

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiVAEOCL::Load - read the sampler state                 |
//+------------------------------------------------------------------+
bool CNeuronHabiVAEOCL::Load(const int h)
  {
   frame.Reset();

   if(!CVAE::Load(h))
      return(false);

   dFileReadUInt(h,N)
   dFileReadUInt(h,L)

   SetKLDMult(0);
   SetSampleMode(VAE_SAMPLE_RANDOM);

   return(N>0 && L>0 && Neurons()==N*L);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiDecoderHeadOCL - decoder head of a latent sample      |
//+------------------------------------------------------------------+
class CNeuronHabiDecoderHeadOCL : public CNeuronHabiLeafOCL
  {
public:
   virtual int       Type() const override
     {
      return(defNeuronHabiDecoderHeadOCL);
     }
protected:
   virtual bool      feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override
     {
      return(frame.Forward(ctx,0,bTrain) && frame.mode==21 && Pass(p));
     }
   virtual bool      calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                        ENUM_ACTIVATION ac=None) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiDecoderHeadOCL::calcInputGradients - backward pass    |
//+------------------------------------------------------------------+
bool CNeuronHabiDecoderHeadOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token) || (mode!=23 && mode!=24) ||
      !frame.Backward(ctx,mode==23?A:N*A+N*L))
      return(false);

   if(mode==23)
     {
      if(!Ready(0,N) || !scratch[0].Fill(1) ||
         !HabiPayload(ctx,4,A,GetPointer(scratch[1]),OpenCL) || !Ready(2,N*A) ||
         !MatMul(GetPointer(scratch[0]),GetPointer(scratch[1]),GetPointer(scratch[2]),N,1,A) ||
         !Different(GetPointer(scratch[2]),Output,Gradient,N*A,0,0,0,2.0f/(N*A)))
         return(false);
     }
   else
      if(!HabiPayload(ctx,4,N*A,Gradient,OpenCL))
         return(false);

   return(DeActivation(p.getOutput(),p.getGradient(),Gradient,SIGMOID));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticInputOCL - Critic input of latent and act       |
//+------------------------------------------------------------------+
class CNeuronHabiCriticInputOCL : public CNeuronHabiLeafOCL
  {
public:
   virtual int       Type() const override
     {
      return(defNeuronHabiCriticInputOCL);
     }
   bool              DirectGradient(float &v[])
     {
      return(scratch[1].GetData(v)==(int)(N*L));
     }
protected:
   virtual bool      feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                        ENUM_ACTIVATION ac=None) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiCriticInputOCL::feedForward - build the input         |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticInputOCL::feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx)
  {
   if(!p || p.Neurons()!=N*A || !frame.Forward(ctx,N*L,bTrain) ||
      (frame.mode!=31 && frame.mode!=35 && frame.mode!=36) ||
      !HabiPayload(ctx,4,N*L,GetPointer(scratch[0]),OpenCL))
      return(false);

   return(Concat(GetPointer(scratch[0]),p.getOutput(),Output,L,A,N));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticInputOCL::calcInputGradients - backward pass    |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticInputOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token) || !frame.Backward(ctx,mode==32?N:(mode==33?3*N:0)) ||
      !Ready(1,N*L))
      return(false);

   return(DeConcat(GetPointer(scratch[1]),p.getGradient(),Gradient,L,A,N));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticHeadOCL - Critic head of the risk value         |
//+------------------------------------------------------------------+
class CNeuronHabiCriticHeadOCL : public CNeuronHabiLeafOCL
  {
public:
                     CNeuronHabiCriticHeadOCL()
     {
      gamma=.5f;
     }
   virtual int       Type() const override
     {
      return(defNeuronHabiCriticHeadOCL);
     }
   bool              Discount(float v);
   virtual bool      Save(const int h) override;
   virtual bool      Load(const int h) override;
protected:
   float             gamma;
   virtual bool      feedForward(CNeuronBaseOCL *p,CBufferFloat *ctx) override
     {
      return(frame.Forward(ctx,N*L,bTrain) && (frame.mode==31 || frame.mode==35 ||
             frame.mode==36) && Pass(p));
     }
   virtual bool      calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,CBufferFloat *sg,
                                        ENUM_ACTIVATION ac=None) override;
  };
//+------------------------------------------------------------------+
//| CNeuronHabiCriticHeadOCL::calcInputGradients - backward pass     |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticHeadOCL::calcInputGradients(CNeuronBaseOCL *p,CBufferFloat *ctx,
      CBufferFloat *sg,ENUM_ACTIVATION ac=None)
  {
   int mode;
   uint token;

   if(!frame.Header(ctx,mode,token) || (mode!=32 && mode!=33 && mode!=34) ||
      !frame.Backward(ctx,mode==32?N:(mode==33?3*N:0)) || !Ready(6,N))
      return(false);

   if(mode==33)
      for(uint i=0;i<N;i++)
         if(ctx.At(4+N+i)!=0 && ctx.At(4+N+i)!=1)
            return(false);

   int k=def_k_HabiCriticSeed;

   setBuffer(k,0,Output.GetIndex())
   setBuffer(k,1,ctx.GetIndex())
   setBuffer(k,2,Gradient.GetIndex())
   setBuffer(k,3,scratch[6].GetIndex())
   setArgument(k,4,(int)N)
   setArgument(k,5,mode)
   setArgument(k,6,gamma)

   uint off[]= {0};
   uint sz[]= {N};

   kernelExecute(k,off,sz)

   return(Finite(N) && DeActivation(p.getOutput(),p.getGradient(),Gradient,None));
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticHeadOCL::Discount - set the discount factor     |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticHeadOCL::Discount(float v)
  {
   if(!MathIsValidNumber(v) || v<0 || v>1)
      return(false);

   gamma=v;

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticHeadOCL::Save - write the discount state        |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticHeadOCL::Save(const int h)
  {
   if(!CNeuronHabiLeafOCL::Save(h))
      return(false);

   dFileWriteFloat(h,gamma)

   return(true);
  }
//+------------------------------------------------------------------+
//| CNeuronHabiCriticHeadOCL::Load - read the discount state         |
//+------------------------------------------------------------------+
bool CNeuronHabiCriticHeadOCL::Load(const int h)
  {
   if(!CNeuronHabiLeafOCL::Load(h))
      return(false);

   float stored=0;

   dFileReadUFloat(h,stored)

   return(Discount(stored));
  }
//+------------------------------------------------------------------+
//| HabiNew - create a Habi layer instance by its type id            |
//+------------------------------------------------------------------+
CNeuronBaseOCL *HabiNew(uint type)
  {
   switch(type)
     {
      case defNeuronHabiObservationOCL:
         return(new CNeuronHabiObservationOCL());
      case defNeuronHabiGaussianOCL:
         return(new CNeuronHabiGaussianOCL());
      case defNeuronHabiVAEOCL:
         return(new CNeuronHabiVAEOCL());
      case defNeuronHabiDecoderHeadOCL:
         return(new CNeuronHabiDecoderHeadOCL());
      case defNeuronHabiCriticInputOCL:
         return(new CNeuronHabiCriticInputOCL());
      case defNeuronHabiCriticHeadOCL:
         return(new CNeuronHabiCriticHeadOCL());
     }

   return(NULL);
  }
//+------------------------------------------------------------------+
//| HabiConfigure - configure any Habi layer from a description      |
//+------------------------------------------------------------------+
bool HabiConfigure(CNeuronBaseOCL *obj,CLayerDescription *d)
  {
   if(!obj || !d)
      return(false);

   if(obj.Type()==defNeuronHabiVAEOCL)
      return(((CNeuronHabiVAEOCL*)obj).Configure(d));

   return(((CNeuronHabiLeafOCL*)obj).Configure(d));
  }
#endif

//+------------------------------------------------------------------+
