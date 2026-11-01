#ifndef HABI_MODELS_MQH
#define HABI_MODELS_MQH
#include "..\NeuroNet_DNG\NeuroNet.mqh"
#include "Habi_Settings.mqh"

//+------------------------------------------------------------------+
//| HabiFinite.
//+------------------------------------------------------------------+
bool HabiFinite(const float &v[],int count)
  {
   if(ArraySize(v)!=count)
      return false;
   for(int i=0;i<count;i++)
      if(!MathIsValidNumber(v[i]))
         return false;
   return true;
  }

//+------------------------------------------------------------------+
//| HabiOutput.
//+------------------------------------------------------------------+
bool HabiOutput(CNet *net,uint layer,int count,float &out[])
  {
   vector<float> v;
   if(!net || !net.GetLayerOutput(layer,v) || v.Size()!=count)
      return false;
   ArrayResize(out,count);
   for(int i=0;i<count;i++)
      out[i]=v[i];
   return HabiFinite(out,count);
  }

//+------------------------------------------------------------------+
//| HabiGradient.
//+------------------------------------------------------------------+
bool HabiGradient(CNet *net,float &out[])
  {
   return net && net.Layer(0) && net.Layer(0).getGradient().GetData(out)>0 && HabiFinite(out,ArraySize(out));
  }

//+------------------------------------------------------------------+
//| HabiContext.
//+------------------------------------------------------------------+
bool HabiContext(CBufferFloat &ctx,COpenCLMy *cl,int mode,uint frame,const float &payload[])
  {
   float v[];
   int n=ArraySize(payload);
   ArrayResize(v,n+4);
   v[0]=3;
   v[1]=(float)mode;
   v[2]=(float)(frame&65535);
   v[3]=(float)(frame>>16);
   for(int i=0;i<n;i++)
      v[4+i]=payload[i];
   if(!HabiFinite(v,n+4))
      return false;
   return ctx.AssignArray(v) && (ctx.GetIndex()<0?ctx.BufferCreate(cl):ctx.BufferWrite());
  }

//+------------------------------------------------------------------+
//| HabiForward.
//+------------------------------------------------------------------+
bool HabiForward(CNet *net,const float &values_in[],int mode,uint frame,const float &payload[],bool training)
  {
   CBufferFloat ctx,values;
   return net && HabiFinite(values_in,net.Layer(0).Neurons()) && net.TrainMode(training) && HabiContext(ctx,net.GetOpenCL(),mode,frame,payload) && values.AssignArray(values_in) && net.feedForward(GetPointer(values),1,false,GetPointer(ctx));
  }

//+------------------------------------------------------------------+
//| HabiBackward.
//+------------------------------------------------------------------+
bool HabiBackward(CNet *net,int mode,uint frame,const float &payload[])
  {
   CBufferFloat ctx;
   return HabiContext(ctx,net.GetOpenCL(),mode,frame,payload) && net.backPropGradient(GetPointer(ctx),NULL,-1,false);
  }

//+------------------------------------------------------------------+
//| HabiRelease.
//+------------------------------------------------------------------+
void HabiRelease(CNet *net)
  {
   if(!net)
      return;
   for(int i=0;net.Layer(i);i++)
     {
      CNeuronBaseOCL *p=net.Layer(i);
      int t=p.Type();
      if(t==defNeuronHabiVAEOCL)
         ((CNeuronHabiVAEOCL*)p).ReleaseFrame();
      else
         if(t>=defNeuronHabiObservationOCL && t<=defNeuronHabiCriticHeadOCL) ((CNeuronHabiLeafOCL*)p).ReleaseFrame();
     }
  }

//+------------------------------------------------------------------+
//| HabiLayer.
//+------------------------------------------------------------------+
bool HabiLayer(CArrayObj &list,uint type,uint count,SHabiGeometry &g,int activation=None,bool posterior=false,int fin=0,int fout=0)
  {
   CLayerDescription *d=new CLayerDescription();
   if(!d)
      return false;
   d.type=type;
   d.count=count;
   d.activation=(ENUM_ACTIVATION)activation;
   d.optimization=ADAM;
   d.batch=TrainingBatch;
   if(type==defNeuronConvOCL)
     {
      d.window=fin;
      d.step=fin;
      d.window_out=fout;
      d.layers=1;
      d.variables=1;
     }
   else
     {
      d.window=g.N;
      d.window_out=g.L;
      d.step=g.A;
      d.layers=g.M;
      d.variables=normalization_batch;
      d.probability=posterior?1:0;
     }
   if(!list.Add(d))
     {
      delete d;
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| HabiCreate.
//+------------------------------------------------------------------+
bool HabiCreate(CNet *net,int role,SHabiGeometry &g)
  {
   CArrayObj d;
   d.FreeMode(true);
   uint in=role<=1?g.O+(role==1?g.A:0):(role==2?2*g.L:g.N*g.A);
   if(!HabiLayer(d,defNeuronBaseOCL,in,g))
      return false;
   if(role<=1)
     {
      if(!HabiLayer(d,defNeuronHabiObservationOCL,in,g,None,role==1) || !HabiLayer(d,defNeuronBaseOCL,g.L,g,LReLU) || !HabiLayer(d,defNeuronBaseOCL,g.L,g,LReLU) || !HabiLayer(d,defNeuronBaseOCL,2*g.L,g) || !HabiLayer(d,defNeuronHabiGaussianOCL,2*g.L,g,None,role==1))
         return false;
     }
   else
      if(role==2)
     {
      if(!HabiLayer(d,defNeuronHabiVAEOCL,g.N*g.L,g) || !HabiLayer(d,defNeuronConvOCL,g.N,g,LReLU,false,g.L,g.L) || !HabiLayer(d,defNeuronConvOCL,g.N,g,LReLU,false,g.L,g.L) || !HabiLayer(d,defNeuronConvOCL,g.N,g,SIGMOID,false,g.L,g.A) || !HabiLayer(d,defNeuronHabiDecoderHeadOCL,g.N*g.A,g))
         return false;
     }
   else
     {
      if(!HabiLayer(d,defNeuronHabiCriticInputOCL,g.N*(g.L+g.A),g) || !HabiLayer(d,defNeuronConvOCL,g.N,g,LReLU,false,g.L+g.A,g.L) || !HabiLayer(d,defNeuronConvOCL,g.N,g,LReLU,false,g.L,g.L) || !HabiLayer(d,defNeuronConvOCL,g.N,g,LReLU,false,g.L,g.L) || !HabiLayer(d,defNeuronConvOCL,g.N,g,None,false,g.L,1) || !HabiLayer(d,defNeuronHabiCriticHeadOCL,g.N,g))
         return false;
     }
   return net.Create(GetPointer(d));
  }

//+------------------------------------------------------------------+
//| HabiLoad.
//+------------------------------------------------------------------+
bool HabiLoad(CNet *net,string file)
  {
   if(file=="")
      return false;
   float error,undefined,forecast;
   datetime stamp;
   return net.Load(file,error,undefined,forecast,stamp,CommonFiles);
  }

//+------------------------------------------------------------------+
//| HabiShape.
//+------------------------------------------------------------------+
bool HabiShape(CNet *net,int role,SHabiGeometry &g)
  {
   int depth=role==3?7:6;
   if(!net || !net.Layer(depth-1) || net.Layer(depth))
      return false;
   uint expected[];
   ArrayResize(expected,depth);
   if(role<=1)
     {
      expected[0]=g.O+(role==1?g.A:0);
      expected[1]=expected[0];
      expected[2]=g.L;
      expected[3]=g.L;
      expected[4]=2*g.L;
      expected[5]=2*g.L;
     }
   else
      if(role==2)
     {
      expected[0]=2*g.L;
      expected[1]=g.N*g.L;
      expected[2]=g.N*g.L;
      expected[3]=g.N*g.L;
      expected[4]=g.N*g.A;
      expected[5]=g.N*g.A;
     }
   else
     {
      expected[0]=g.N*g.A;
      expected[1]=g.N*(g.L+g.A);
      expected[2]=g.N*g.L;
      expected[3]=g.N*g.L;
      expected[4]=g.N*g.L;
      expected[5]=g.N;
      expected[6]=g.N;
     }
   for(int i=0;i<depth;i++)
     {
      CNeuronBaseOCL *p=net.Layer(i);
      if(p.Neurons()!=expected[i] || p.getOutput().Total()!=expected[i] || p.getGradient().Total()!=expected[i])
         return false;
      CBufferFloat *weights=p.Type()==defNeuronConvOCL?((CNeuronConvOCL*)p).GetWeightsConv():p.getWeights();
      if(weights && weights.Total()>0)
        {
         if(!weights.BufferRead())
            return false;
         for(int j=0;j<weights.Total();j++)
            if(!MathIsValidNumber(weights.At(j)))
               return false;
        }
      int t=i==0?defNeuronBaseOCL:(role<=1?(i==1?defNeuronHabiObservationOCL:(i==5?defNeuronHabiGaussianOCL:defNeuronBaseOCL)): (role==2?(i==1?defNeuronHabiVAEOCL:(i==5?defNeuronHabiDecoderHeadOCL:defNeuronConvOCL)): (i==1?defNeuronHabiCriticInputOCL:(i==6?defNeuronHabiCriticHeadOCL:defNeuronConvOCL))));
      int act=role<=1?((i==2 || i==3)?LReLU:None): (role==2?((i==2 || i==3)?LReLU:(i==4?SIGMOID:None)):((i>=2 && i<=4)?LReLU:None));
      if(p.Type()!=t || p.Activation()!=act)
         return false;
      if(t==defNeuronHabiObservationOCL && (!((CNeuronHabiObservationOCL*)p).BNGeometry() || !((CNeuronHabiObservationOCL*)p).BNStateValid()))
         return false;
      if(t==defNeuronHabiVAEOCL)
        {
         CNeuronHabiVAEOCL *v=p;
         if(v.Rows()!=g.N || v.Latent()!=g.L)
            return false;
        }
      else
         if(t>=defNeuronHabiObservationOCL && t<=defNeuronHabiCriticHeadOCL)
        {
         if(!((CNeuronHabiLeafOCL*)p).Geometry(role<=1?0:g.N,g.L,g.A,g.M,role==1))
            return false;
        }
      CLayerDescription *info=p.GetLayerInfo();
      if(!info)
         return false;
      if(t==defNeuronConvOCL)
        {
         int fin=expected[i-1]/g.N,fout=expected[i]/g.N;
         CNeuronConvOCL *conv=p;
         bool ok=info.window==fin && info.step==fin && info.window_out==fout && info.count==g.N && conv.GetWeightsConv().Total()==(fin+1)*fout;
         delete info;
         if(!ok)
            return false;
        }
      else
        {
         delete info;
        }
      if(role<=1 && i>=1 && i<=3 && p.getWeights().Total()!=(expected[i]+1)*expected[i+1])
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| HabiImport.
//+------------------------------------------------------------------+
bool HabiImport(CNet *destination,string file,string indices,int role,SHabiGeometry &g)
  {
   if(file=="")
      return indices=="";
   CNet source;
   if(!HabiLoad(GetPointer(source),file))
      return false;
   string parts[];
   int count=StringSplit(indices,',',parts),expected=role==3?4:3;
   if(count!=expected)
      return false;
   int owners[];
   ArrayResize(owners,expected);
   for(int i=0;i<expected;i++)
      owners[i]=role<=1?i+1:i+2;
// First validate every tensor; apply none if any source/target geometry is incompatible.
   for(int i=0;i<expected;i++)
     {
      for(int k=0;k<StringLen(parts[i]);k++)
         if(StringGetCharacter(parts[i],k)<'0' || StringGetCharacter(parts[i],k)>'9')
            return false;
      if(StringLen(parts[i])==0)
         return false;
      int at=(int)StringToInteger(parts[i]);
      CNeuronBaseOCL *src=source.Layer(at),*dst=destination.Layer(owners[i]);
      if(!src || !dst)
         return false;
      CBufferFloat *sb=NULL,*db=NULL;
      if(role<=1)
        {
         if(src.Type()!=defNeuronBaseOCL && src.Type()!=defNeuronHabiObservationOCL)
            return false;
         if(src.Neurons()!=dst.Neurons())
            return false;
         sb=src.getWeights();
         db=dst.getWeights();
        }
      else
        {
         if(src.Type()!=defNeuronConvOCL)
            return false;
         CLayerDescription *si=src.GetLayerInfo(),*di=dst.GetLayerInfo();
         if(!si || !di)
           {
            if(si)
               delete si;
            if(di)
               delete di;
            return false;
           }
         bool match=si.window==di.window && si.window_out==di.window_out && si.step==di.step;
         delete si;
         delete di;
         if(!match)
            return false;
         sb=((CNeuronConvOCL*)src).GetWeightsConv();
         db=((CNeuronConvOCL*)dst).GetWeightsConv();
        }
      if(!sb || !db || sb.Total()!=db.Total() || !sb.BufferRead())
         return false;
      for(int j=0;j<sb.Total();j++)
         if(!MathIsValidNumber(sb.At(j)))
            return false;
     }
   for(int i=0;i<expected;i++)
     {
      CNeuronBaseOCL *src=source.Layer((int)StringToInteger(parts[i])),*dst=destination.Layer(owners[i]);
      CBufferFloat *sb=role<=1?src.getWeights():((CNeuronConvOCL*)src).GetWeightsConv();
      CBufferFloat *db=role<=1?dst.getWeights():((CNeuronConvOCL*)dst).GetWeightsConv();
      float values[];
      if(sb.GetData(values)!=sb.Total() || !db.AssignArray(values) || !db.BufferWrite())
         return false;
     }
   return HabiShape(destination,role,g);
  }
class CHabiModels
  {
   public: CNet *prior,*posterior,*decoder,*critic,*target_prior,*target_decoder,*target_critic;
   SHabiGeometry g;
   uint serial,policy_frame;
   bool policy_live,target_bn_pending;
   float last_z[],last_actions[],last_observation[];
   CHabiModels()
     {
      prior=NULL;
      posterior=NULL;
      decoder=NULL;
      critic=NULL;
      target_prior=NULL;
      target_decoder=NULL;
      target_critic=NULL;
      serial=0;
      policy_frame=0;
      policy_live=false;
      target_bn_pending=false;
     }
   ~CHabiModels()
     {
      if(target_critic)
         delete target_critic;
      if(target_decoder)
         delete target_decoder;
      if(target_prior)
         delete target_prior;
      if(critic)
         delete critic;
      if(decoder)
         delete decoder;
      if(posterior)
         delete posterior;
      if(prior)
         delete prior;
     }
   bool Role(CNet *net,string path,int role,bool training)
     {
      if(path!="")
        {
         if(!HabiLoad(net,path))
            return false;
        }
      else
         if(!training || !HabiCreate(net,role,g)) return false;
      if(!HabiShape(net,role,g))
         return false;
      if(net!=prior && (!net.SetOpenCLChecked(prior.GetOpenCL()) || !net.IsOpenCLTransferReady()))
         return false;
      HabiRelease(net);
      return true;
     }
   bool Init(int stage)
     {
      bool training=stage!=2;
      g.M=HistoryBars*9;
      g.O=g.M+13;
      g.A=6;
      g.L=LatentSize;
      g.N=Candidates;
      prior=new CNet();
      if(!prior)
         return false;
      if(PriorFile!="")
        {
         if(!HabiLoad(prior,PriorFile) || !prior.Layer(5) || prior.Layer(5).Type()!=defNeuronHabiGaussianOCL)
            return false;
         g.L=((CNeuronHabiGaussianOCL*)prior.Layer(5)).Latent();
        }
// Decoder defines N for imported models, before creating any other graph.
      decoder=new CNet();
      if(!decoder)
         return false;
      if(DecoderFile!="")
        {
         if(!HabiLoad(decoder,DecoderFile) || !decoder.Layer(1) || decoder.Layer(1).Type()!=defNeuronHabiVAEOCL)
            return false;
         CNeuronHabiVAEOCL *s=decoder.Layer(1);
         g.N=s.Rows();
         if(PriorFile=="")
            g.L=s.Latent();
         else
            if(g.L!=s.Latent())return false;
        }
      if(g.N<1 || g.L<1 || (ulong)g.N*(g.L+g.A)>INT_MAX || (ulong)(g.L+1)*2*g.L>INT_MAX || g.M>INT_MAX-19)
         return false;
      if(PriorFile=="" && (!training || !HabiCreate(prior,0,g)))
         return false;
      if(!HabiShape(prior,0,g))
         return false;
      if(DecoderFile=="" && (!training || !HabiCreate(decoder,2,g)))
         return false;
      if(!HabiShape(decoder,2,g) || !decoder.SetOpenCLChecked(prior.GetOpenCL()))
         return false;
      critic=new CNet();
      if(!critic || !Role(critic,CriticFile,3,training))
         return false;
      if(training && !((CNeuronHabiCriticHeadOCL*)critic.Layer(6)).Discount((float)DiscountGamma))
         return false;
      if(stage==0)
        {
         posterior=new CNet();
         if(!posterior || !Role(posterior,PosteriorFile,1,true))
            return false;
         if((PosteriorFile=="" || ResetLoadedBeta) && !((CNeuronHabiGaussianOCL*)posterior.Layer(5)).Beta(AdaptiveBeta,BetaValue,BetaLearningRate,KTarget,BetaRho1,BetaRho2,BetaEpsilon))
            return false;
        }
      if(!HabiImport(prior,PriorImportFile,PriorImportLayers,0,g) || !HabiImport(decoder,DecoderImportFile,DecoderImportLayers,2,g) || !HabiImport(critic,CriticImportFile,CriticImportLayers,3,g) || (posterior && !HabiImport(posterior,PosteriorImportFile,PosteriorImportLayers,1,g)))
         return false;
      if(stage==1 && enable_critic_td)
        {
         target_prior=new CNet();
         target_decoder=new CNet();
         target_critic=new CNet();
         if(!target_prior || !target_decoder || !target_critic || !Role(target_prior,TargetPriorFile,0,true) || !Role(target_decoder,TargetDecoderFile,2,true) || !Role(target_critic,TargetCriticFile,3,true))
            return false;
         if((TargetPriorFile=="" && !target_prior.WeightsUpdate(prior,1)) || (TargetDecoderFile=="" && !target_decoder.WeightsUpdate(decoder,1)) || (TargetCriticFile=="" && !target_critic.WeightsUpdate(critic,1)))
            return false;
        }
      if(target_prior)
        {
         bool ready=((CNeuronHabiObservationOCL*)target_prior.Layer(1)).BNReady();
         if(TargetPriorFile!="" && !ready)
            return false;
         target_bn_pending=!ready;
        }
      if(!training && !((CNeuronHabiObservationOCL*)prior.Layer(1)).BNReady())
         return false;
      HabiRelease(prior);
      HabiRelease(decoder);
      return true;
     }
   bool Generate(CNet *p,CNet *d,const float &o[],uint id,bool policy,float &z[],float &actions[])
     {
      float empty[],gaussian[];
      if(!HabiForward(p,o,policy?3:2,id,empty,policy) || !HabiOutput(p,5,2*g.L,gaussian) || !HabiForward(d,gaussian,21,id,empty,policy) || !HabiOutput(d,1,g.N*g.L,z) || !HabiOutput(d,5,g.N*g.A,actions))
         return false;
      return true;
     }
   bool Predict(const float &o[],bool training_policy)
     {
      HabiRelease(prior);
      HabiRelease(decoder);
      policy_live=false;
      policy_frame=++serial;
// A new StudyOnline BN may start on its first real observation.
      if(training_policy || !((CNeuronHabiObservationOCL*)prior.Layer(1)).BNReady())
        {
         float empty[],gaussian[];
         if(!HabiForward(prior,o,3,policy_frame,empty,true) || !HabiOutput(prior,5,2*g.L,gaussian) || !HabiForward(decoder,gaussian,21,policy_frame,empty,training_policy) || !HabiOutput(decoder,1,g.N*g.L,last_z) || !HabiOutput(decoder,5,g.N*g.A,last_actions))
            return false;
        }
      else
         if(!Generate(prior,decoder,o,policy_frame,false,last_z,last_actions))return false;
      if(target_bn_pending)
        {
         if(!((CNeuronHabiObservationOCL*)target_prior.Layer(1)).CopyBN((CNeuronHabiObservationOCL*)prior.Layer(1)))
            return false;
         target_bn_pending=false;
        }
      ArrayCopy(last_observation,o);
      policy_live=training_policy;
      return true;
     }
   bool Scores(CNet *net,const float &z[],const float &a[],uint id,int mode,bool train,float &q[])
     {
      HabiRelease(net);
      return HabiForward(net,a,mode,id,z,train) && HabiOutput(net,6,g.N,q);
     }
   bool BC(const float &o[],const float &teacher[])
     {
      uint id=++serial;
      float empty[],qinput[],pv[],qv[],reconstruction[],payload[];
      ArrayCopy(qinput,o);
      ArrayCopy(qinput,teacher,ArraySize(o));
      HabiRelease(prior);
      HabiRelease(posterior);
      HabiRelease(decoder);
      if(!HabiForward(prior,o,1,id,empty,true) || !HabiForward(posterior,qinput,1,id,empty,true) || !HabiOutput(prior,5,2*g.L,pv) || !HabiOutput(posterior,5,2*g.L,qv))
         return false;
      float beta=((CNeuronHabiGaussianOCL*)posterior.Layer(5)).EffectiveBeta();
      if(!HabiForward(decoder,qv,21,id,empty,true) || !HabiBackward(decoder,23,id,teacher) || !HabiGradient(decoder,reconstruction))
         return false;
      ArrayCopy(payload,pv);
      ArrayCopy(payload,reconstruction,2*g.L);
      ArrayResize(payload,4*g.L+1);
      payload[4*g.L]=beta;
      if(!HabiBackward(posterior,12,id,payload))
         return false;
      ArrayResize(payload,0);
      ArrayCopy(payload,qv);
      ArrayResize(payload,2*g.L+1);
      payload[2*g.L]=beta;
      return HabiBackward(prior,11,id,payload);
     }
   bool MC(const float &o[],const float &teacher[],float &z[],float &a[],uint &id)
     {
      id=++serial;
      float in[],empty[],qv[];
      ArrayCopy(in,o);
      ArrayCopy(in,teacher,ArraySize(o));
      HabiRelease(posterior);
      HabiRelease(decoder);
      return HabiForward(posterior,in,2,id,empty,false) && HabiOutput(posterior,5,2*g.L,qv) && HabiForward(decoder,qv,21,id,empty,false) && HabiOutput(decoder,1,g.N*g.L,z) && HabiOutput(decoder,5,g.N*g.A,a);
     }
   bool LearnMC(const float &z[],const float &a[],const float &targets[],uint id)
     {
      float q[];
      return Scores(critic,z,a,id,31,true,q) && HabiBackward(critic,32,id,targets);
     }
   bool TD(const float &z[],const float &a[],const float &reward[],const float &done[],const float &next[])
     {
      uint id=++serial;
      float boot[],payload[],q[];
      ArrayResize(boot,g.N);
      ArrayInitialize(boot,0);
      bool terminal=true;
      for(uint n=0;n<g.N;n++)
         if(done[n]!=1)
            terminal=false;
      if(!terminal)
        {
         float tz[],ta[],tq[];
         HabiRelease(target_prior);
         HabiRelease(target_decoder);
         if(!Generate(target_prior,target_decoder,next,id,false,tz,ta) || !Scores(target_critic,tz,ta,id,35,false,tq))
            return false;
         float mx=tq[0];
         for(uint n=1;n<g.N;n++)
            mx=MathMax(mx,tq[n]);
         ArrayInitialize(boot,mx);
        }
      ArrayCopy(payload,reward);
      ArrayCopy(payload,done,g.N);
      ArrayCopy(payload,boot,2*g.N);
      return Scores(critic,z,a,id,31,true,q) && HabiBackward(critic,33,id,payload);
     }
   bool Policy()
     {
      if(!policy_live || (!train_prior && !train_decoder))
        {
         HabiRelease(prior);
         HabiRelease(decoder);
         policy_live=false;
         return true;
        }
      float empty[],q[],ga[],gz[],payload[],gp[];
      bool frozen=critic.WeightsUpdateEnabled();
      if(!critic.SetWeightsUpdate(false))
         return false;
      bool ok=Scores(critic,last_z,last_actions,policy_frame,36,true,q) && HabiBackward(critic,34,policy_frame,empty) && HabiGradient(critic,ga) && ((CNeuronHabiCriticInputOCL*)critic.Layer(1)).DirectGradient(gz);
      critic.SetWeightsUpdate(frozen);
      if(!ok)
         return false;
      ArrayCopy(payload,ga);
      ArrayCopy(payload,gz,g.N*g.A);
      bool old=decoder.WeightsUpdateEnabled();
      decoder.SetWeightsUpdate(train_decoder);
      ok=HabiBackward(decoder,24,policy_frame,payload);
      decoder.SetWeightsUpdate(old);
      if(!ok)
         return false;
      if(train_prior && (!HabiGradient(decoder,gp) || !HabiBackward(prior,13,policy_frame,gp)))
         return false;
      HabiRelease(prior);
      policy_live=false;
      return true;
     }
   bool Sync()
     {
      return target_prior.WeightsUpdate(prior,1) && target_decoder.WeightsUpdate(decoder,1) && target_critic.WeightsUpdate(critic,1);
     }
   bool SaveRole(CNet *net,string suffix)
     {
      return !net || net.Save(OutputPrefix+suffix+".nnw",net.getRecentAverageError(),0,0,TimeCurrent(),CommonFiles);
     }
   bool Save()
     {
      return SaveRole(prior,"_Prior") && SaveRole(posterior,"_Posterior") && SaveRole(decoder,"_Decoder") && SaveRole(critic,"_Critic") && SaveRole(target_prior,"_TargetPrior") && SaveRole(target_decoder,"_TargetDecoder") && SaveRole(target_critic,"_TargetCritic");
     }
  };
#endif
