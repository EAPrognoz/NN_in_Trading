//+------------------------------------------------------------------+
//|                                      NeuroNet_SelectiveOpenCL.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_SELECTIVE_OPENCL_MQH
#define NEURONET_SELECTIVE_OPENCL_MQH

#include "NeuroNet.mqh"

//+------------------------------------------------------------------+
//| CSelectiveOpenCL - Build the kernels requested by shared models  |
//+------------------------------------------------------------------+
class CSelectiveOpenCL:public COpenCLMy
  {
private:
   bool   m_collecting;
   string m_required[];

protected:
   bool Require(const int id,const string name)
     {
      if(id<0||id>=ArraySize(m_required))
         ReturnFalse;
      if(StringLen(m_required[id])>0&&m_required[id]!=name)
         ReturnFalseEx(StringFormat("Kernel ID %d already belongs to %s",id,m_required[id]));
      m_required[id]=name;

      return(true);
     }

   virtual string ProgramSource(void) const { return(cl_program); }

private:
   int KernelEnd(const string source,const int start)
     {
      int depth=0;
      for(int i=start;i<StringLen(source);i++)
        {
         const ushort ch=StringGetCharacter(source,i);
         if(ch=='/'&&StringGetCharacter(source,i+1)=='/')
           {
            i=StringFind(source,"\n",i+2);
            if(i<0)
               return(-1);
           }
         else if(ch=='/'&&StringGetCharacter(source,i+1)=='*')
           {
            i=StringFind(source,"*/",i+2);
            if(i<0)
               return(-1);
            i++;
           }
         else if(ch=='{')
            depth++;
         else if(ch=='}'&&--depth==0)
            return(i+1);
        }

      return(-1);
     }

public:
                     CSelectiveOpenCL(void):m_collecting(false) {}

   bool Begin(const int device=CL_USE_ANY)
     {
      if(m_collecting||GetContext()!=INVALID_HANDLE||
         ArrayResize(m_required,AG_KERNEL_COUNT)!=AG_KERNEL_COUNT||!ContextCreate(device))
         ReturnFalse;

      uint size=0;
      if(!CLGetDeviceInfo(GetContext(),CL_DEVICE_MAX_WORK_ITEM_SIZES,uLocalDimension.cdata,size))
         ReturnFalse;
      size=0;
      if(!CLGetDeviceInfo(GetContext(),CL_DEVICE_MAX_WORK_GROUP_SIZE,uMaxWorkGroup.cdata,size))
         ReturnFalse;

      m_collecting=true;

      return(true);
     }

   virtual bool IsCollectingSelectiveKernels(void) const override
     {
      return(m_collecting);
     }

   virtual bool RegisterSelectiveLayer(const int type,const ENUM_OPTIMIZATION optimizer) override
     {
      if(!m_collecting||optimizer!=ADAM)
         ReturnFalseEx("Selective runtime currently supports Adam layers");

      if(!Require(def_k_FeedForward,"FeedForward")||
         !Require(def_k_CalcOutputGradient,"CalcOutputGradient")||
         !Require(def_k_CalcHiddenGradient,"CalcHiddenGradient")||
         !Require(def_k_UpdateWeightsAdam,"UpdateWeightsAdam")||
         !Require(def_k_NormalizeWeights,"NormalizeWeights")||
         !Require(def_k_SoftUpdate,"SoftUpdate")||
         !Require(def_k_SoftUpdateAdam,"SoftUpdateAdam"))
         ReturnFalse;

      switch(type)
        {
         case defNeuronBaseOCL:
            return(true);
         case defNeuronConvOCL:
            return(Require(def_k_FeedForwardConv,"FeedForwardConv")&&
                   Require(def_k_CalcHiddenGradientConv,"CalcHiddenGradientConv")&&
                   Require(def_k_UpdateWeightsConvAdam,"UpdateWeightsConvAdam"));
         case defNeuronBatchNormOCL:
            return(Require(def_k_BatchFeedForward,"BatchFeedForward")&&
                   Require(def_k_CalcHiddenGradientBatch,"CalcHiddenGradientBatch")&&
                   Require(def_k_UpdateBatchOptionsAdam,"UpdateBatchOptionsAdam"));
         case defNeuronConcatenate:
            return(Require(def_k_ConcatFeedForward,"Concat_FeedForward")&&
                   Require(def_k_ConcatCalcHiddenGradient,"Concat_HiddenGradient")&&
                   Require(def_k_ConcatUpdWeightsAdam,"Concat_UpdateWeightsAdam"));
        }

      ReturnFalseEx(StringFormat("Unsupported selective layer type %d",type));
     }

   bool Build(void)
     {
      if(!m_collecting||GetProgram()!=INVALID_HANDLE)
         ReturnFalse;

      const ulong started=GetTickCount64();
      const string full_source=ProgramSource();
      string source="";
      int cursor=0;
      int count=0;
      int found=0;
      for(int id=0;id<ArraySize(m_required);id++)
         if(StringLen(m_required[id])>0)
            count++;

      for(int begin=StringFind(full_source,"__kernel");begin>=0;
          begin=StringFind(full_source,"__kernel",begin+8))
        {
         const int paren=StringFind(full_source,"(",begin);
         if(paren<0)
            ReturnFalse;
         string name=StringSubstr(full_source,begin+13,paren-begin-13);
         StringTrimLeft(name);
         StringTrimRight(name);
         bool keep=false;
         for(int id=0;id<ArraySize(m_required);id++)
            if(StringLen(m_required[id])>0&&m_required[id]==name)
              {
               keep=true;
               found++;
               break;
              }
         if(keep)
            continue;

         const int end=KernelEnd(full_source,paren);
         if(end<=begin)
            ReturnFalseEx(name);
         source+=StringSubstr(full_source,cursor,begin-cursor);
         cursor=end;
        }
      source+=StringSubstr(full_source,cursor);
      if(count==0||found!=count||!SetKernelsCount(ArraySize(m_required)))
         ReturnFalseEx("Selective OpenCL kernel declarations do not match the registry");

      string build_log;
      const int program=CLProgramCreate(GetContext(),source,build_log);
      if(program==INVALID_HANDLE)
         ReturnFalseEx(build_log);

      for(int id=0;id<ArraySize(m_required);id++)
        {
         if(StringLen(m_required[id])==0)
            continue;
         const int kernel=CLKernelCreate(program,m_required[id]);
         if(kernel==INVALID_HANDLE)
           {
            for(int i=0;i<id;i++)
               if(m_kernels[i]!=INVALID_HANDLE)
                 {
                  CLKernelFree(m_kernels[i]);
                  m_kernels[i]=INVALID_HANDLE;
                  m_kernel_names[i]="";
                 }
            CLProgramFree(program);
            ReturnFalseEx(m_required[id]);
           }
         m_kernels[id]=kernel;
         m_kernel_names[id]=m_required[id];
        }
      m_program=program;
      m_collecting=false;
      PrintFormat("Selective OpenCL ready: kernels=%d source_chars=%d elapsed_ms=%I64u",
                  count,StringLen(source),GetTickCount64()-started);

      return(true);
     }
  };

#endif // NEURONET_SELECTIVE_OPENCL_MQH
