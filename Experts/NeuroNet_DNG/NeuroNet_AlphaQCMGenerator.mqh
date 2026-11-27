//+------------------------------------------------------------------+
//|                                   NeuroNet_AlphaQCMGenerator.mqh |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_GENERATOR_MQH
#define NEURONET_ALPHAQCM_GENERATOR_MQH

//+------------------------------------------------------------------+
//| CAlphaQCMGenerator - Own transient prefix and QCM workspace.     |
//+------------------------------------------------------------------+
class CAlphaQCMGenerator
  {
private:
   CAlphaQCMOpenCL *OpenCL;
   CBufferFloat m_packed;
   CBufferFloat m_work;
   CBufferFloat m_flags;
   CBufferFloat m_projection;
   CBufferFloat m_basis;
   CBufferFloat m_phi;
   int m_capacity;
   int m_length;
   int m_embed;
   int m_hidden;
   int m_frequencies;
   int m_points;
   int m_chunk;
   int m_steps;
   int m_status;
   uint m_lanes;
   uint m_gridLanes;
   float m_epsilon;
   float m_bonus;
   bool m_unknown;

public:
//+------------------------------------------------------------------+
//| CAlphaQCMGenerator - Initialize the object state                 |
//+------------------------------------------------------------------+
   CAlphaQCMGenerator():OpenCL(NULL),m_capacity(0),m_status(AG_TECHNICAL),m_unknown(false) {}
//+------------------------------------------------------------------+
//| Results - Return the results                                     |
//+------------------------------------------------------------------+
   CBufferFloat      *Results(void) { return(GetPointer(m_packed)); }
//+------------------------------------------------------------------+
//| Status - Return the status                                       |
//+------------------------------------------------------------------+
   int               Status(void) const { return(m_status); }
   bool Configure(COpenCLMy *context,const int candidates,const int length,const int embed,
                  const int hidden,const int points,const float epsilon,const float bonus,const int frequencies=0);
   bool Generate(const int config_buffer,const SAQCInferenceView &q,const SAQCInferenceView &iqn,
                 const ulong seed,const ulong first_candidate,const int count,const bool warmup);

private:
   bool Allocate(CBufferFloat &buffer,const long elements);
   bool Complete(const int kernel);
   bool Grid(const int weight);
  };

//+------------------------------------------------------------------+
//| Allocate - Allocate or reuse an OpenCL buffer                    |
//+------------------------------------------------------------------+
bool CAlphaQCMGenerator::Allocate(CBufferFloat &buffer,const long elements)
  {
   if(elements<=0||elements>INT_MAX||!OpenCL.CanAllocate((ulong)elements,sizeof(float))||
      !buffer.BufferInit((uint)elements,0.0f)||!buffer.BufferCreate(OpenCL))
     {
      ReturnFalse
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Complete - Read the generator status                             |
//+------------------------------------------------------------------+
bool CAlphaQCMGenerator::Complete(const int kernel)
  {
   if(OpenCL.WaitAlphaGenKernel(kernel))
      return(true);
   if(OpenCL.ExecutionStatus(kernel)==CL_COMPLETE)
     {
      if(!IsStopped())
         return(true);
      ReturnFalse
     }
   m_unknown=true;
   m_status=AQCM_UPDATE_INCONSISTENT;
   ReturnFalseEx("OpenCL dispatch completion unknown; generator is closed")
  }

//+------------------------------------------------------------------+
//| Grid - Run grid features and cosine kernels                      |
//+------------------------------------------------------------------+
bool CAlphaQCMGenerator::Grid(const int weight)
  {
   const int kernel=def_k_AQCMGridFeatures;
   setBuffer(kernel,0,m_basis.GetIndex())
   setBuffer(kernel,1,weight)
   setBuffer(kernel,2,m_phi.GetIndex())
   setArgument(kernel,3,m_frequencies)
   uint offset[]={0,0};
   uint global[]={(uint)m_hidden*m_gridLanes,(uint)m_points};
   uint local[]={m_gridLanes,1};
   kernelExecuteLoc(kernel,offset,global,local)
   return(Complete(kernel));
  }

//+------------------------------------------------------------------+
//| Configure - Configure the generator and its buffers              |
//+------------------------------------------------------------------+
bool CAlphaQCMGenerator::Configure(COpenCLMy *context,const int candidates,const int length,const int embed,
                                   const int hidden,const int points,const float epsilon,const float bonus,const int frequencies)
  {
   m_status=AG_TECHNICAL;
   if(!context||m_capacity!=0||m_unknown||candidates<1||length<2||embed<1||hidden<1||
      points<4||frequencies<0||epsilon<0||epsilon>1||bonus<0||!MathIsValidNumber(epsilon)||!MathIsValidNumber(bonus))
     {
      ReturnFalse
     }

   OpenCL=(CAlphaQCMOpenCL *)context;
   m_lanes=OpenCL.GroupSize(def_k_AQCMGenerate,64,2);
   m_gridLanes=OpenCL.GroupSize(def_k_AQCMGridFeatures,64,2);
   m_frequencies=(frequencies==0?hidden:frequencies);
   const long basis=(long)points*m_frequencies;
   const long grid=(long)points*hidden;
   const long packed_stride=1+(long)length*(4+AG_VOCABULARY);
   m_status=AQCM_RESOURCE_UNSUPPORTED;
   if(m_lanes<2||m_gridLanes<2||grid>INT_MAX/2||basis>INT_MAX||packed_stride>INT_MAX||
      (long)hidden*m_gridLanes>UINT_MAX||packed_stride*candidates>INT_MAX)
     {
      ReturnFalse
     }
   const long work_stride=(long)15*hidden+embed+1+2*AG_VOCABULARY+
                          2*grid+3*((long)length+1)*AG_VOCABULARY;
   if(work_stride>INT_MAX)
     {
      ReturnFalse
     }
   // In addition to the two shared grids and coefficient row, keep every
   // packed episode until commit, and reuse one candidate chunk of scratch.
   PrintFormat("AlphaQCM generation workspace: FP32 packed=%I64d flags=%d basis=%I64d phi=%I64d projection=%d "
               "scratch_per_candidate=%I64d candidates=%d min_device_bytes=%I64d",
               packed_stride*candidates,candidates,basis,grid,points,work_stride,candidates,
               4*(packed_stride*candidates+candidates+basis+grid+points+work_stride));

   if(!Allocate(m_packed,packed_stride*candidates)||!Allocate(m_flags,candidates)||
      !Allocate(m_basis,basis)||!Allocate(m_phi,grid))
     {
      ReturnFalse
     }
   m_chunk=candidates;
   while(m_chunk>0)
     {
      if((long)m_chunk*m_lanes<=UINT_MAX&&Allocate(m_work,work_stride*m_chunk))
         break;
      m_chunk/=2;
     }
   if(m_chunk==0)
     {
      ReturnFalseEx("Minimum generation candidate scratch could not be allocated")
     }

   m_status=AQCM_NUMERIC_ERROR;
   if(!AlphaQCMProjection(m_projection,OpenCL,points,m_status))
     {
      ReturnFalse
     }

   m_status=AG_TECHNICAL;
   m_embed=embed;
   m_hidden=hidden;
   m_length=length;
   m_points=points;
   m_epsilon=epsilon;
   m_bonus=bonus;
   m_steps=(m_lanes>=32?4:2);

   const int kernel=def_k_AQCMGridCosines;
   setBuffer(kernel,0,m_basis.GetIndex())
   setArgument(kernel,1,points)
   setArgument(kernel,2,m_frequencies)
   uint offset[]={0,0};
   uint global[]={(uint)m_frequencies,(uint)points};
   kernelExecute(kernel,offset,global)
   if(!Complete(kernel))
     {
      ReturnFalse
     }
   m_capacity=candidates;
   m_status=AG_OK;
   PrintFormat("AlphaQCM generation launch: candidate_chunk=%d token_chunk=%d group=%u grid_group=%u "
               "device_bytes=%I64d",m_chunk,m_steps,m_lanes,m_gridLanes,
               4*(packed_stride*candidates+candidates+basis+grid+points+work_stride*m_chunk));
   return(true);
  }

//+------------------------------------------------------------------+
//| Generate - Generate candidate formulas on the device             |
//+------------------------------------------------------------------+
bool CAlphaQCMGenerator::Generate(const int config_buffer,const SAQCInferenceView &q,const SAQCInferenceView &iqn,
                                  const ulong seed,const ulong first_candidate,const int count,const bool warmup)
  {
   if(m_unknown)
     {
      ReturnFalse
     }
   m_status=AG_TECHNICAL;
   if(!OpenCL||m_capacity<1||count<1||count>m_capacity||config_buffer<0||
      (ulong)(count-1)>ULONG_MAX-first_candidate)
     {
      ReturnFalse
     }

   if(q.context!=OpenCL||iqn.context!=OpenCL||q.iqn||!iqn.iqn||
      CheckPointer(q.owner)==POINTER_INVALID||CheckPointer(iqn.owner)==POINTER_INVALID||
      q.length!=m_length||iqn.length!=m_length||
      q.matrices[0].width!=m_embed||iqn.matrices[0].width!=m_embed||
      q.matrices[0].rows!=AG_VOCABULARY||iqn.matrices[0].rows!=AG_VOCABULARY||
      q.matrices[2].width!=2*m_hidden+1||iqn.matrices[2].width!=2*m_hidden+1||
      iqn.matrices[3].width!=m_frequencies+1||
      !q.owner.ValidateInferenceView(q)||!iqn.owner.ValidateInferenceView(iqn))
     {
      ReturnFalseEx("AlphaQCM generator received an incompatible or stale inference view")
     }

   if(!warmup&&!Grid(iqn.matrices[3].buffer))
     {
      ReturnFalse
     }

   const int kernel=def_k_AQCMGenerate,uniform=(warmup?1:0);
   setBuffer(kernel,0,config_buffer)
   for(int i=0;i<5;i++)
     {
      setBuffer(kernel,i+1,q.matrices[i].buffer)
     }
   for(int i=0;i<3;i++)
     {
      setBuffer(kernel,i+6,iqn.matrices[i].buffer)
     }
   setBuffer(kernel,9,iqn.matrices[4].buffer)
   setBuffer(kernel,10,iqn.matrices[5].buffer)
   setBuffer(kernel,11,m_phi.GetIndex())
   setBuffer(kernel,12,m_projection.GetIndex())
   setBuffer(kernel,13,m_packed.GetIndex())
   setBuffer(kernel,14,m_work.GetIndex())
   setBuffer(kernel,15,m_flags.GetIndex())
   setArgument(kernel,16,seed)
   setArgument(kernel,17,first_candidate)
   setArgument(kernel,19,m_length)
   setArgument(kernel,20,m_embed)
   setArgument(kernel,21,m_hidden)
   setArgument(kernel,22,m_points)
   setArgument(kernel,23,uniform)
   setArgument(kernel,24,m_epsilon)
   setArgument(kernel,25,m_bonus)
   uint offset[]={0},local[]={m_lanes},global[]={0};
   for(int base=0;base<count;base+=m_chunk)
     {
      global[0]=(uint)MathMin(m_chunk,count-base)*m_lanes;
      setArgument(kernel,18,base)
      for(int begin=0;begin<m_length;begin+=m_steps)
        {
         const int end=(int)MathMin(m_length,begin+m_steps);
         setArgument(kernel,26,begin)
         setArgument(kernel,27,end)
         kernelExecuteLoc(kernel,offset,global,local)
         if(!Complete(kernel))
           {
            ReturnFalse
           }
        }
     }

   if(!m_flags.BufferRead())
     {
      ReturnFalse
     }
   for(int c=0;c<count;c++)
     {
      int report=(int)m_flags.At(c);
      if(report==0)
         continue;
      m_status=(report%4==2?AQCM_NUMERIC_ERROR:AG_TECHNICAL);
      ReturnFalseEx(StringFormat("candidate=%I64u token_position=%d reason=%s",first_candidate+(ulong)c,
                                 report/4,report%4==2?"nonfinite legal score":"invalid grammar, token or draw"))
     }
   m_status=AG_OK;
   return(true);
  }

#endif // NEURONET_ALPHAQCM_GENERATOR_MQH
