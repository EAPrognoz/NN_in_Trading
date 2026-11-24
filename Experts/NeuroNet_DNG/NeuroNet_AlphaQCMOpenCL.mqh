//+------------------------------------------------------------------+
//|                                       NeuroNet_AlphaQCMOpenCL.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_OPENCL_MQH
#define NEURONET_ALPHAQCM_OPENCL_MQH

#include "AlphaQCM_NetworkContract.mqh"

#resource "AlphaQCM_Replay.cl" as string aqcm_replay_program
#resource "AlphaQCM_Network.cl" as string aqcm_network_program
#resource "AlphaQCM_Pool.cl" as string aqcm_pool_program
#resource "AlphaQCM.cl" as string aqcm_program

#define def_k_AQCMGenerate     620
#define def_k_AQCMGridFeatures 621
#define def_k_AQCMGridCosines  622
#define AQCM_KERNEL_COUNT     640

//+------------------------------------------------------------------+
//| CAlphaQCMOpenCL - Collect and build the model's kernel graph      |
//+------------------------------------------------------------------+
class CAlphaQCMOpenCL:public COpenCLMy
  {
private:
   bool              m_collecting;
   int               m_mode;
   int               m_required_count;
   string            m_required[];
   long              m_kernel_group[];
   long              m_kernel_local[];
   long              m_local_memory;
   long              m_global_memory;
   long              m_max_allocation;

public:
   CAlphaQCMOpenCL(void):m_collecting(false),m_mode(AQCM_TRAIN),m_required_count(0),
                                           m_local_memory(0),m_global_memory(0),m_max_allocation(0) {}
   bool Begin(const int device=CL_USE_GPU_ONLY);
   virtual bool IsCollectingSelectiveKernels(void) const override
     {
      return(m_collecting&&GetContext()!=INVALID_HANDLE);
     }
   bool Require(const int id,const string name);
   virtual bool RegisterSelectiveLayer(const int type,const ENUM_OPTIMIZATION optimizer) override;
   bool Build(void);
   uint GroupSize(const int kernel,const uint maximum,const uint minimum,const ulong dynamic_bytes=0) const;
   bool ValidateLocalMemory(const int kernel);
   bool ValidateTradingLocalMemory(const int hidden);
   bool CanAllocate(const ulong elements,const uint bytes_per_element) const;

private:
   bool Supports12(const string version,const string prefix) const;
   bool DeviceLimits(void);
   int KernelEnd(const string &source,const int start) const;
   bool SelectSource(string &source) const;
   bool CreateKernels(const int program);
  };

//+------------------------------------------------------------------+
//| Supports12 - Check the reported OpenCL version                   |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::Supports12(const string version,const string prefix) const
  {
   if(StringFind(version,prefix)!=0)
      return(false);

   const int start=StringLen(prefix);
   const int dot=StringFind(version,".",start);
   if(dot<=start)
      return(false);

   const int major=(int)StringToInteger(StringSubstr(version,start,dot-start));
   string minor_text=StringSubstr(version,dot+1);
   const int end=StringFind(minor_text," ");
   if(end>=0)
      minor_text=StringSubstr(minor_text,0,end);
   const int minor=(int)StringToInteger(minor_text);

   return(major>1||(major==1&&minor>=2));
  }

//+------------------------------------------------------------------+
//| DeviceLimits - Check capabilities before model allocation        |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::DeviceLimits(void)
  {
//--- Native selectors for properties without MQL5 enum entries.
   const int device_type=0x1000;
   const int address_bits=0x100D;
   const int max_alloc_size=0x1010;
   const int single_fp_config=0x101B;
   const int compiler_available=0x1028;
   const long gpu_bit=1<<2;
   const long fp32_required=(1<<1)|(1<<2); // Inf/NaN and round-to-nearest.
   const long type=GetDeviceInfo(device_type);
   if(type<0||(type&gpu_bit)==0)
      ReturnFalseEx("RESOURCE_UNSUPPORTED: the OpenCL context does not expose a GPU device");

   string version;
   string version_c;
   string profile;
   const int context=GetContext();
   if(!CLGetInfoString(context,CL_DEVICE_VERSION,version)||!Supports12(version,"OpenCL ")||
      !CLGetInfoString(context,CL_DEVICE_OPENCL_C_VERSION,version_c)||!Supports12(version_c,"OpenCL C ")||
      !CLGetInfoString(context,CL_DEVICE_PROFILE,profile)||profile!="FULL_PROFILE"||
      GetDeviceInfo(compiler_available)!=1)
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: OpenCL 1.2 full-profile compiler required (device=%s C=%s profile=%s)",
                                version,version_c,profile));

   const long fp32=GetDeviceInfo(single_fp_config);
   const long bits=GetDeviceInfo(address_bits);
   if(fp32<0||(fp32&fp32_required)!=fp32_required||(bits!=32&&bits!=64))
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: FP32 configuration=%I64d address bits=%I64d",fp32,bits));

   const long dimensions=GetDeviceInfoInteger(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
   const long group=GetDeviceInfoInteger(CL_DEVICE_MAX_WORK_GROUP_SIZE);
   uchar work_item_bytes[];
   uint size=0;
   if(dimensions<3||group<2||
      !CLGetDeviceInfo(context,CL_DEVICE_MAX_WORK_ITEM_SIZES,work_item_bytes,size)||
      (long)size>ArraySize(work_item_bytes)||(long)size%dimensions!=0)
      ReturnFalseEx("RESOURCE_UNSUPPORTED: invalid OpenCL work-group geometry");

   const int width=(int)(size/dimensions);
   if(width!=4&&width!=8)
      ReturnFalseEx("RESOURCE_UNSUPPORTED: unrecognized OpenCL work-item limit size");

   for(int dimension=0;dimension<3;dimension++)
     {
      ulong value=0;
      for(int byte_index=0;byte_index<width;byte_index++)
         value|=(ulong)work_item_bytes[dimension*width+byte_index]<<(8*byte_index);

      if(value<1||value>LONG_MAX)
         ReturnFalseEx("RESOURCE_UNSUPPORTED: invalid OpenCL per-dimension work-item limit");

      uLocalDimension.data[dimension]=(long)value;
     }
   uMaxWorkGroup.data[0]=group;

   m_local_memory=GetDeviceInfoInteger(CL_DEVICE_LOCAL_MEM_SIZE);
   m_global_memory=GetDeviceInfoInteger(CL_DEVICE_GLOBAL_MEM_SIZE);
   m_max_allocation=GetDeviceInfo(max_alloc_size);
   if(m_local_memory<1||m_global_memory<1||m_max_allocation<1||GetMaxLocalSize(0)<2)
      ReturnFalseEx("RESOURCE_UNSUPPORTED: insufficient OpenCL memory or parallel work-item limits");

   string name;
   string driver;
   if(!CLGetInfoString(context,CL_DEVICE_NAME,name))
      name="unknown";
   if(!CLGetInfoString(context,CL_DRIVER_VERSION,driver))
      driver="unknown";
   PrintFormat("AlphaQCM OpenCL GPU %s, driver %s, %s, address_bits=%I64d work_group=%I64d local_bytes=%I64d global_bytes=%I64d max_buffer_bytes=%I64d",
               name,driver,version_c,bits,group,m_local_memory,m_global_memory,m_max_allocation);

   return(true);
  }

//+------------------------------------------------------------------+
//| KernelEnd - Find the closing brace outside comments and quotes   |
//+------------------------------------------------------------------+
int CAlphaQCMOpenCL::KernelEnd(const string &source,const int start) const
  {
   int depth=0;
   const int length=StringLen(source);
   for(int i=start;i<length;i++)
     {
      const ushort ch=StringGetCharacter(source,i);
      if(ch=='/'&&i+1<length)
        {
         const ushort next=StringGetCharacter(source,i+1);
         if(next=='/')
           {
            i=StringFind(source,"\n",i+2);
            if(i<0)
               return(-1);
           }
         else if(next=='*')
           {
            i=StringFind(source,"*/",i+2);
            if(i<0)
               return(-1);
            i++;
           }
        }
      else if(ch==34||ch==39)
        {
         while(++i<length&&StringGetCharacter(source,i)!=ch)
            if(StringGetCharacter(source,i)=='\\')
               i++;
        }
      else if(ch=='{')
         depth++;
      else if(ch=='}'&&--depth==0)
         return(i+1);
     }

   return(-1);
  }

//+------------------------------------------------------------------+
//| SelectSource - Keep the registered kernels and donor helpers     |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::SelectSource(string &source) const
  {
//--- Helpers remain in their original order. Replay supplies Philox to both search modules.
   const string full="\n"+cl_program+"\n"+ag_contract+"\n"+ag_program+"\n"+ag_pool_program+"\n"+
                     aqcm_replay_program+"\n"+aqcm_network_program+"\n"+aqcm_pool_program+"\n"+aqcm_program;
   const string marker="\n__kernel void ";
   int seen[];
   if(ArrayResize(seen,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT)
      ReturnFalseEx("Cannot allocate the selected kernel table");
   ArrayInitialize(seen,0);

   int cursor=0;
   int found=0;
   source="";
   for(int begin=StringFind(full,marker);begin>=0;)
     {
      const int name_start=begin+StringLen(marker);
      const int paren=StringFind(full,"(",name_start);
      if(paren<0)
         ReturnFalseEx("Invalid OpenCL kernel declaration");

      string name=StringSubstr(full,name_start,paren-name_start);
      StringTrimRight(name);
      int selected=-1;
      for(int id=0;id<AQCM_KERNEL_COUNT;id++)
         if(m_required[id]==name)
           {
            selected=id;
            break;
           }

      const int end=KernelEnd(full,paren);
      if(end<=paren)
         ReturnFalseEx(StringFormat("Incomplete OpenCL kernel %s",name));

      if(selected<0)
        {
         source+=StringSubstr(full,cursor,begin+1-cursor);
         cursor=end;
        }
      else
        {
         if(seen[selected]!=0)
            ReturnFalseEx(StringFormat("Duplicate OpenCL source kernel %s",name));
         seen[selected]=1;
         found++;
        }
      begin=StringFind(full,marker,end);
     }
   source+=StringSubstr(full,cursor);

   if(found!=m_required_count)
      for(int id=0;id<AQCM_KERNEL_COUNT;id++)
         if(m_required[id]!=""&&seen[id]==0)
            ReturnFalseEx(StringFormat("OpenCL source is missing kernel %d %s",id,m_required[id]));

   return(true);
  }

//+------------------------------------------------------------------+
//| CreateKernels - Populate runtime handles and verify limits       |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::CreateKernels(const int program)
  {
   for(int id=0;id<AQCM_KERNEL_COUNT;id++)
     {
      const string name=m_required[id];
      if(name=="")
         continue;

      //--- Free a failed build without invalidating the context of the live models.
      const int kernel=CLKernelCreate(program,name);
      if(kernel==INVALID_HANDLE)
         ReturnFalseEx(StringFormat("Cannot create OpenCL kernel %s (error=%d)",name,GetLastError()));

      m_kernels[id]=kernel;
      m_kernel_names[id]=name;
      m_kernel_group[id]=CLGetInfoInteger(kernel,CL_KERNEL_WORK_GROUP_SIZE);
      m_kernel_local[id]=CLGetInfoInteger(kernel,CL_KERNEL_LOCAL_MEM_SIZE);
      const long private_bytes=CLGetInfoInteger(kernel,CL_KERNEL_PRIVATE_MEM_SIZE);
      if(m_kernel_group[id]<1||m_kernel_local[id]<0||m_kernel_local[id]>m_local_memory||private_bytes<0)
         ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: kernel=%s group=%I64d local_bytes=%I64d private_bytes=%I64d",
                                   name,m_kernel_group[id],m_kernel_local[id],private_bytes));
      long minimum=1;
      if(id>=AQCMPoolOrderedKernel&&id<=AQCMPoolHistoryGlobalKernel)
         minimum=id==AQCMPoolOrderedKernel?4:2;
      switch(id)
        {
         case def_k_AQCRecur:
         case def_k_AQCBPTT:
         case def_k_AQCRecurOuter:
         case def_k_AQCEmbedOuter:
         case def_k_AQCDenseRows:
         case def_k_AQCDenseInput:
         case def_k_AQCDenseOuter:
         case def_k_AQCProductGrad:
         case def_k_AQCParameterCheck:
         case def_k_AQCSpanCheck:
         case def_k_AQCTargetSelect:
         case def_k_AQCPairLoss:
         case def_k_AQCPriorityLoss:
            minimum=AQC_NET_MIN_GROUP;
            break;
        }
      if(MathMin(m_kernel_group[id],MathMin(GetMaxWorkGroupSize(),GetMaxLocalSize(0)))<minimum)
         ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: kernel %s needs %I64d parallel lanes",name,minimum));
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Begin - Open the GPU context and start dependency collection     |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::Begin(const int device)
  {
   if(GetContext()!=INVALID_HANDLE)
      ReturnFalseEx("AlphaQCM OpenCL already owns a context");
   if(AlphaQCMLoadMode!=AQCM_TRAIN&&AlphaQCMLoadMode!=AQCM_INFER&&AlphaQCMLoadMode!=AQCM_VERIFY)
      ReturnFalseEx("Invalid AlphaQCM load mode");
   if(ArrayResize(m_required,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT)
      ReturnFalseEx("Cannot allocate the AlphaQCM OpenCL registry");

   m_collecting=false;
   m_mode=AlphaQCMLoadMode;
   m_required_count=0;
   for(int id=0;id<AQCM_KERNEL_COUNT;id++)
      m_required[id]="";

   if(!ContextCreate(device))
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: OpenCL device selection %d failed (error=%d)",device,GetLastError()));
   if(!DeviceLimits())
     {
      Shutdown();
      ReturnFalse;
     }

   m_collecting=true;

   return(true);
  }

//+------------------------------------------------------------------+
//| Require - Bind a kernel name to its stable ID during collection  |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::Require(const int id,const string name)
  {
   if(!IsCollectingSelectiveKernels()||id<0||id>=AQCM_KERNEL_COUNT||name=="")
      ReturnFalseEx(StringFormat("Cannot collect kernel %d %s",id,name));
   if(m_required[id]==name)
      return(true);
   if(m_required[id]!="")
      ReturnFalseEx(StringFormat("Kernel ID %d already belongs to %s",id,m_required[id]));
   for(int other_id=0;other_id<AQCM_KERNEL_COUNT;other_id++)
      if(m_required[other_id]==name)
         ReturnFalseEx(StringFormat("Kernel %s already has ID %d",name,other_id));

   m_required[id]=name;
   m_required_count++;

   return(true);
  }

//+------------------------------------------------------------------+
//| RegisterSelectiveLayer - Resolve the layer's mode-specific graph |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::RegisterSelectiveLayer(const int type,const ENUM_OPTIMIZATION optimizer)
  {
   if(!IsCollectingSelectiveKernels()||optimizer!=ADAM)
      ReturnFalseEx("AlphaQCM selective runtime requires Adam layers and an active collection");

   if(type==defNeuronBaseOCL||type==defNeuronAlphaGenEvaluatorOCL||type==defNeuronLSTMOCL)
      return(true);

   if(type==defNeuronAlphaQCMEncoderOCL)
     {
      if(m_mode!=AQCM_TRAIN)
         ReturnFalseEx("AlphaQCM encoder requires TRAIN mode");
      for(int id=570;id<620;id++)
        {
         const string name=AlphaQCMNetworkKernelName(id);
         if(name!=""&&!Require(id,name))
            ReturnFalse;
        }
      return(true);
     }

   if(type!=defNeuronAlphaQCMOCL)
      ReturnFalseEx(StringFormat("Unsupported AlphaQCM layer type %d",type));

   if(!Require(def_k_AGContextMetadata,"AGContextMetadata")||
      !Require(def_k_AGCopySlice,"AGCopySlice")||
      !Require(def_k_AGDistribution,"AGDistribution")||
      !Require(def_k_AGLSTMForward,"AGLSTMForward")||
      !Require(def_k_AGNumeric,"AGNumeric")||
      !Require(def_k_AGNumericFinish,"AGNumericFinish")||
      !Require(def_k_AGPoolEvaluateHistory,"AGPoolEvaluateHistory")||
      !Require(def_k_AGPoolTransaction,"AGPoolTransaction")||
      !Require(def_k_AGReLU,"AGReLU")||
      !Require(def_k_AGSequence,"AGSequence")||
      !Require(def_k_CopyBufferRaw,"CopyBufferRaw")||
      !Require(def_k_FeedForward,"FeedForward")||
      !Require(def_k_MatMult,"MatMult")||
      !Require(501,"AQCMPoolScore")||
      !Require(502,"AQCMPoolAdvance")||
      !Require(504,"AQCMPoolHistoryGlobal"))
      ReturnFalse;

   if(m_mode!=AQCM_TRAIN)
      return(true);

   if(!Require(def_k_AGDenseGradient,"AGDenseGradient")||
      !Require(def_k_AGGAE,"AGGAE")||
      !Require(def_k_AGLSTMBackward,"AGLSTMBackward")||
      !Require(def_k_AGLSTMWeightGradient,"AGLSTMWeightGradient")||
      !Require(def_k_AGPoolEvaluateBatch,"AGPoolEvaluateBatch")||
      !Require(def_k_AGReplayPrepare,"AGReplayPrepare")||
      !Require(def_k_CalcHiddenGradient,"CalcHiddenGradient")||
      !Require(def_k_LSTM_UpdateWeightsAdam,"LSTM_UpdateWeightsAdam")||
      !Require(500,"AQCMPoolOrdered")||
      !Require(503,"AQCMPoolEvaluatePrepared")||
      !Require(530,"AQReplayTreeLeaves")||
      !Require(531,"AQReplayTreeLevel")||
      !Require(532,"AQReplayStageEpisode")||
      !Require(533,"AQReplayPublishEpisode")||
      !Require(534,"AQReplaySample")||
      !Require(535,"AQReplayGather")||
      !Require(536,"AQReplayTau")||
      !Require(537,"AQReplayStagePriorities")||
      !Require(538,"AQReplayCommitPriorities")||
      !Require(539,"AQReplayMirrorTree")||
      !Require(540,"AQReplayCheckIntervals")||
      !Require(541,"AQReplayStatus")||
      !Require(def_k_AQCMGenerate,"AQCMGenerate")||
      !Require(def_k_AQCMGridFeatures,"AQCMGridFeatures")||
      !Require(def_k_AQCMGridCosines,"AQCMGridCosines"))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Build - Compile once after all models are created or loaded      |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::Build(void)
  {
   if(!IsCollectingSelectiveKernels()||GetProgram()!=INVALID_HANDLE||m_required_count<1)
      ReturnFalseEx("AlphaQCM OpenCL needs an active, populated kernel registry");

   m_collecting=false;
   const ulong started=GetTickCount64();
   string source;
   if(!SelectSource(source)||
      ArrayResize(m_kernel_group,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT||
      ArrayResize(m_kernel_local,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT||
      ArrayResize(m_kernels,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT||
      ArrayResize(m_kernel_names,AQCM_KERNEL_COUNT)!=AQCM_KERNEL_COUNT||
      !SetKernelsCount(AQCM_KERNEL_COUNT))
      ReturnFalse;

   string build_log;
   const int program=CLProgramCreate(GetContext(),source,build_log);
   if(program==INVALID_HANDLE)
      ReturnFalseEx(StringFormat("AlphaQCM OpenCL build failed (error=%d) %s",GetLastError(),build_log));

   if(!CreateKernels(program))
     {
      for(int id=0;id<AQCM_KERNEL_COUNT;id++)
         if(m_kernels[id]!=INVALID_HANDLE)
            KernelFree(id);
      CLProgramFree(program);
      ReturnFalse;
     }

//--- Both donor dense launches use local size {1, lanes} and read the
//--- cached dimension-1 limit. Their reductions accept arbitrary lanes.
//--- Keep their dispatch within the device and compiled kernel caps.
   if(m_required[def_k_FeedForward]!=""||m_required[def_k_CalcHiddenGradient]!="")
     {
      long dense_limit=MathMin(GetMaxLocalSize(1),GetMaxWorkGroupSize());
      if(m_required[def_k_FeedForward]!="")
         dense_limit=MathMin(dense_limit,m_kernel_group[def_k_FeedForward]);
      if(m_required[def_k_CalcHiddenGradient]!="")
         dense_limit=MathMin(dense_limit,m_kernel_group[def_k_CalcHiddenGradient]);
      uLocalDimension.data[1]=dense_limit;
     }

   m_program=program;
   PrintFormat("AlphaQCM OpenCL ready: mode=%d kernels=%d source_chars=%d build_ms=%I64u",
               m_mode,m_required_count,StringLen(source),GetTickCount64()-started);

   return(true);
  }

//+------------------------------------------------------------------+
//| GroupSize - Select a portable power-of-two one-dimensional WG    |
//+------------------------------------------------------------------+
uint CAlphaQCMOpenCL::GroupSize(const int kernel,const uint maximum,const uint minimum,const ulong dynamic_bytes) const
  {
   if(GetProgram()==INVALID_HANDLE||kernel<0||kernel>=AQCM_KERNEL_COUNT||
      m_required[kernel]==""||minimum<1||maximum<minimum||
      dynamic_bytes>(ulong)(m_local_memory-m_kernel_local[kernel]))
      return(0);

   const long device_limit=MathMin(GetMaxWorkGroupSize(),GetMaxLocalSize(0));
   const long limit=MathMin((long)maximum,MathMin(m_kernel_group[kernel],device_limit));
   if(limit<minimum)
      return(0);

   uint lanes=1;
   while(lanes<=limit/2)
      lanes*=2;

   return(lanes>=minimum?lanes:0);
  }

//+------------------------------------------------------------------+
//| ValidateLocalMemory - Check the kernel after local bindings      |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::ValidateLocalMemory(const int kernel)
  {
   const int handle=GetKernel(kernel);
   if(handle==INVALID_HANDLE)
      ReturnFalseEx("OpenCL kernel is unavailable");

   const long bytes=CLGetInfoInteger(handle,CL_KERNEL_LOCAL_MEM_SIZE);
   if(bytes<0||bytes>m_local_memory)
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: kernel %s uses %I64d local bytes of %I64d",
                                GetKernelName(kernel),bytes,m_local_memory));

   return(true);
  }

//+------------------------------------------------------------------+
//| ValidateTradingLocalMemory - Check donor bindings before READY   |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::ValidateTradingLocalMemory(const int hidden)
  {
   if(GetProgram()==INVALID_HANDLE||hidden<1||hidden>INT_MAX/(22*(int)sizeof(float)))
      ReturnFalse;

   const int forward=def_k_AGLSTMForward;
   const int forward_bytes=hidden*22*sizeof(float);
   if(forward_bytes>m_local_memory||
      !SetArgumentLocalMemory(forward,def_ag_AGLSTMForward_state,forward_bytes)||
      !ValidateLocalMemory(forward)||AlphaGenGroupSize(forward,128,4)==0)
      ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: trading LSTM forward needs four lanes and %d dynamic local bytes",
                                 forward_bytes));

   long backward_bytes=0;
   const int backward=def_k_AGLSTMBackward;
   if(m_required[backward]!="")
     {
      // The trading policy has one input feature. Match the donor's dynamic tile.
      const long state=(long)hidden*6*sizeof(float);
      const long columns=MathMin((long)hidden+1,(m_local_memory-state)/(4*sizeof(float)));
      backward_bytes=state+columns*4*sizeof(float);
      if(columns<1||backward_bytes>INT_MAX||
         !SetArgumentLocalMemory(backward,def_ag_AGLSTMBackward_scratch,(int)backward_bytes)||
         !ValidateLocalMemory(backward)||AlphaGenGroupSize(backward,256,4)==0)
         ReturnFalseEx(StringFormat("RESOURCE_UNSUPPORTED: trading LSTM backward needs four lanes and %I64d dynamic local bytes",
                                    backward_bytes));
     }
   PrintFormat("AlphaQCM trading LSTM floor: four parallel lanes, local bytes forward=%I64d backward=%I64d (including kernel storage)",
               GetKernelInfoInteger(forward,CL_KERNEL_LOCAL_MEM_SIZE),
               backward_bytes>0?GetKernelInfoInteger(backward,CL_KERNEL_LOCAL_MEM_SIZE):0);
   return(true);
  }

//+------------------------------------------------------------------+
//| CanAllocate - Check a buffer size before narrowing to uint       |
//+------------------------------------------------------------------+
bool CAlphaQCMOpenCL::CanAllocate(const ulong elements,const uint bytes_per_element) const
  {
   if(GetContext()==INVALID_HANDLE||elements==0||bytes_per_element==0)
      return(false);

   const ulong limit=(ulong)MathMin((long)UINT_MAX,MathMin(m_global_memory,m_max_allocation));

   return(elements<=limit/bytes_per_element);
  }

#endif // NEURONET_ALPHAQCM_OPENCL_MQH
