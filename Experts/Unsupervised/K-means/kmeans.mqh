//+------------------------------------------------------------------+
//|                                                       kmeans.mqh |
//|                                                          K-means |
//|                                              Copyright 2022, DNG |
//|                           https://www.mql5.com/ru/articles/10947 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| includes                                                         |
//+------------------------------------------------------------------+
#include "..\..\NeuroNet_DNG\NeuroNet.mqh"
#resource "unsupervised.cl" as string cl_unsupervised
//+------------------------------------------------------------------+
//| Defines                                                          |
//+------------------------------------------------------------------+
#define defUnsupervisedKmeans    0x7901
//---
#define def_k_kmeans_distance    0
#define def_k_kmd_data           0
#define def_k_kmd_means          1
#define def_k_kmd_distance       2
#define def_k_kmd_vector_size    3
//---
#define def_k_kmeans_clustering  1
#define def_k_kmc_distance       0
#define def_k_kmc_clusters       1
#define def_k_kmc_flags          2
#define def_k_kmc_total_k        3
//---
#define def_k_kmeans_updates     2
#define def_k_kmu_data           0
#define def_k_kmu_clusters       1
#define def_k_kmu_means          2
#define def_k_kmu_total_m        3
//---
#define def_k_kmeans_loss        3
#define def_k_kml_data           0
#define def_k_kml_clusters       1
#define def_k_kml_means          2
#define def_k_kml_loss           3
#define def_k_kml_vector_size    4
//---
#define def_k_kmeans_statistic   4
#define def_k_kms_clusters       0
#define def_k_kms_targers        1
#define def_k_kms_probability    2
#define def_k_kms_total_m        3
//---
#define def_k_kmeans_softmax     5
#define def_k_kmsm_distance      0
#define def_k_kmsm_softmax       1
#define def_k_kmsm_total_k       2
//+------------------------------------------------------------------+
//| OpenCLCreate                                                     |
//+------------------------------------------------------------------+
COpenCLMy *OpenCLCreate(string programm)
  {
   COpenCL *result=new COpenCLMy();
   if(CheckPointer(result)==POINTER_INVALID)
      return(NULL);
//---
   if(!result.Initialize(programm,true))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.SetKernelsCount(6))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_distance,"KmeansCulcDistance"))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_clustering,"KmeansClustering"))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_updates,"KmeansUpdating"))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_loss,"KmeansLoss"))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_statistic,"KmeansStatistic"))
     {
      delete result;

      return(NULL);
     }
//---
   if(!result.KernelCreate(def_k_kmeans_softmax,"KmeansSoftMax"))
     {
      delete result;

      return(NULL);
     }
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| CKmeans - Group observations with k-means                        |
//+------------------------------------------------------------------+
class CKmeans  : public CObject
  {
protected:
   int    m_iClusters;
   int    m_iVectorSize;
   double m_dLoss;
   bool   m_bTrained;

   CBufferFloat *c_aMeans;
   COpenCLMy    *c_OpenCL;   ///< Object for working with OpenCL
   //---
   CBufferFloat *c_aDistance;
   CBufferFloat *c_aClasters;
   CBufferFloat *c_aFlags;
   CBufferFloat *c_aSoftMax;
   CBufferFloat *c_aLoss;
   CBufferFloat *c_aProbability;

public:
                     CKmeans(void);
                    ~CKmeans(void);
   //---
   bool              SetOpenCL(COpenCLMy *context);
   bool              Init(COpenCLMy *context,int clusters,int vector_size);
   bool              Study(CBufferFloat *data,bool init_means=true);
   bool              Clustering(CBufferFloat *data);
   CBufferFloat      *SoftMax(CBufferFloat *data);
   bool              Statistic(CBufferFloat *data,CBufferFloat *targets);
   double            GetLoss(CBufferFloat *data);
   CBufferFloat      *GetProbability(CBufferFloat *data);
   //---
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   //---
//+------------------------------------------------------------------+
//| Type - Return the object type identifier                         |
//+------------------------------------------------------------------+
   virtual int       Type(void)
     {
      return(defUnsupervisedKmeans);
     }
  };
//+------------------------------------------------------------------+
//| CKmeans - Group observations with k-means                        |
//+------------------------------------------------------------------+
void CKmeans::CKmeans(void)   :  m_iClusters(2),
   m_iVectorSize(1),
   m_dLoss(-1),
//| Implements m_bTrained                                            |
   m_bTrained(false)
  {
   c_aMeans=new CBufferFloat();
   if(CheckPointer(c_aMeans)!=POINTER_INVALID)
      c_aMeans.BufferInit(m_iClusters * m_iVectorSize,0);
   c_OpenCL=NULL;
  }
//+------------------------------------------------------------------+
//| ~CKmeans - Release owned resources                               |
//+------------------------------------------------------------------+
void CKmeans::~CKmeans(void)
  {
   if(CheckPointer(c_aMeans)==POINTER_DYNAMIC)
      delete c_aMeans;
   if(CheckPointer(c_aDistance)==POINTER_DYNAMIC)
      delete c_aDistance;
   if(CheckPointer(c_aClasters)==POINTER_DYNAMIC)
      delete c_aClasters;
   if(CheckPointer(c_aFlags)==POINTER_DYNAMIC)
      delete c_aFlags;
   if(CheckPointer(c_aSoftMax)==POINTER_DYNAMIC)
      delete c_aSoftMax;
   if(CheckPointer(c_aLoss)==POINTER_DYNAMIC)
      delete c_aLoss;
   if(CheckPointer(c_aProbability)==POINTER_DYNAMIC)
      delete c_aProbability;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| SetOpenCL - | S et Op en CL implementation.                      |
//+------------------------------------------------------------------+
bool CKmeans::SetOpenCL(COpenCLMy *context)
  {
   c_OpenCL=context;
//---
   return(c_OpenCL==context);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Init - | I ni t implementation.                                  |
//+------------------------------------------------------------------+
bool CKmeans::Init(COpenCLMy *context,int clusters,int vector_size)
  {
   if(CheckPointer(context)==POINTER_INVALID||clusters<2||vector_size<1)
      return(false);
//---
   c_OpenCL=context;
   m_iClusters=clusters;
   m_iVectorSize=vector_size;
   if(CheckPointer(c_aMeans)==POINTER_INVALID)
     {
      c_aMeans=new CBufferFloat();
      if(CheckPointer(c_aMeans)==POINTER_INVALID)
         return(false);
     }
   c_aMeans.BufferFree();
   if(!c_aMeans.BufferInit(m_iClusters * m_iVectorSize,0))
     {
      ReturnFalse
     }
   m_bTrained=false;
   m_dLoss=-1;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Study - | S tu dy implementation.                                |
//+------------------------------------------------------------------+
bool CKmeans::Study(CBufferFloat *data,bool init_means=true)
  {
   COpenCLMy *OpenCL=c_OpenCL;

   if(CheckPointer(data)==POINTER_INVALID||CheckPointer(c_OpenCL)==POINTER_INVALID)
      return(false);
//---
   int total=data.Total();
   if(total<=0||m_iClusters<2||(total % m_iVectorSize)!=0)
      return(false);
//---
   int rows=total / m_iVectorSize;
   if(rows<=(10 * m_iClusters))
      return(false);
//---
   bool flags[];
   if(ArrayResize(flags,rows)<=0||!ArrayInitialize(flags,false))
      return(false);
//---
   for(int i=0; (i<m_iClusters&&init_means); i++)
     {
      Comment(StringFormat("Cluster initialization %d of %d",i,m_iClusters));
      int row=(int)((double)MathRand() * MathRand() / MathPow(32767,2) * (rows-1));
      if(flags[row])
        {
         i--;
         continue;
        }
      int start=row * m_iVectorSize;
      int start_c=i * m_iVectorSize;
      for(int c=0; c<m_iVectorSize; c++)
        {
         if(!c_aMeans.Update(start_c+c,data.At(start+c)))
            return(false);
        }
      flags[row]=true;
     }
//---
   if(CheckPointer(c_aDistance)==POINTER_INVALID)
     {
      c_aDistance=new CBufferFloat();
      if(CheckPointer(c_aDistance)==POINTER_INVALID)
         return(false);
     }
   c_aDistance.BufferFree();
   if(!c_aDistance.BufferInit(rows * m_iClusters,0))
     {
      ReturnFalse
     }
//---
   if(CheckPointer(c_aClasters)==POINTER_INVALID)
     {
      c_aClasters=new CBufferFloat();
      if(CheckPointer(c_aClasters)==POINTER_INVALID)
         return(false);
     }
   c_aClasters.BufferFree();
   if(!c_aClasters.BufferInit(rows,0))
     {
      ReturnFalse
     }
//---
   if(CheckPointer(c_aFlags)==POINTER_INVALID)
     {
      c_aFlags=new CBufferFloat();
      if(CheckPointer(c_aFlags)==POINTER_INVALID)
         return(false);
     }
   c_aFlags.BufferFree();
   if(!c_aFlags.BufferInit(rows,0))
     {
      ReturnFalse
     }
//---
   if(!data.BufferCreate(c_OpenCL)||
      !c_aMeans.BufferCreate(c_OpenCL)||
      !c_aDistance.BufferCreate(c_OpenCL)||
      !c_aClasters.BufferCreate(c_OpenCL)||
      !c_aFlags.BufferCreate(c_OpenCL))
      return(false);
//---
   int count=0;
   do
     {
      setBuffer(def_k_kmeans_distance,def_k_kmd_data,data.GetIndex())

      setBuffer(def_k_kmeans_distance,def_k_kmd_means,c_aMeans.GetIndex())

      setBuffer(def_k_kmeans_distance,def_k_kmd_distance,c_aDistance.GetIndex())

      setArgument(def_k_kmeans_distance,def_k_kmd_vector_size,m_iVectorSize)

      uint global_work_offset[2]={0,0};
      uint global_work_size[2];
      global_work_size[0]=rows;
      global_work_size[1]=m_iClusters;
      kernelExecute(def_k_kmeans_distance,global_work_offset,global_work_size)
      //---
      setBuffer(def_k_kmeans_clustering,def_k_kmc_flags,c_aFlags.GetIndex())

      setBuffer(def_k_kmeans_clustering,def_k_kmc_clusters,c_aClasters.GetIndex())

      setBuffer(def_k_kmeans_clustering,def_k_kmc_distance,c_aDistance.GetIndex())

      setArgument(def_k_kmeans_clustering,def_k_kmc_total_k,m_iClusters)

      uint global_work_offset1[1]={0};
      uint global_work_size1[1];
      global_work_size1[0]=rows;
      kernelExecute(def_k_kmeans_clustering,global_work_offset1,global_work_size1)

      if(!c_aFlags.BufferRead())
        {
         ReturnFalse
        }

      m_bTrained=(c_aFlags.Maximum()==0);
      if(m_bTrained)
        {
         if(!c_aClasters.BufferRead())
           {
            ReturnFalse
           }
         break;
        }
      //---
      setBuffer(def_k_kmeans_updates,def_k_kmu_data,data.GetIndex())

      setBuffer(def_k_kmeans_updates,def_k_kmu_means,c_aMeans.GetIndex())

      setBuffer(def_k_kmeans_updates,def_k_kmu_clusters,c_aClasters.GetIndex())

      setArgument(def_k_kmeans_updates,def_k_kmu_total_m,rows)

      global_work_size[0]=m_iVectorSize;
      kernelExecute(def_k_kmeans_updates,global_work_offset,global_work_size)

      count++;
      Comment(StringFormat("Study iterations %d",count));
     }
   while(!m_bTrained&&!IsStopped());
//---
   if(!c_aMeans.BufferRead())
     {
      ReturnFalse
     }

   data.BufferFree();
   c_aDistance.BufferFree();
   c_aFlags.BufferFree();
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Clustering - | C lu st er in g implementation.                   |
//+------------------------------------------------------------------+
bool CKmeans::Clustering(CBufferFloat *data)
  {
   COpenCLMy *OpenCL=c_OpenCL;

   if(!m_bTrained&&!Study(data,(c_aMeans.Maximum()==0)))
      return(false);
//---
   if(CheckPointer(data)==POINTER_INVALID||CheckPointer(c_OpenCL)==POINTER_INVALID)
      return(false);
//---
   int total=data.Total();
   if(total<=0||m_iClusters<2||(total % m_iVectorSize)!=0)
      return(false);
//---
   int rows=total / m_iVectorSize;
   if(rows<1)
      return(false);
//---
   if(CheckPointer(c_aDistance)==POINTER_INVALID)
     {
      c_aDistance=new CBufferFloat();
      if(CheckPointer(c_aDistance)==POINTER_INVALID)
         return(false);
     }
   c_aDistance.BufferFree();
   if(!c_aDistance.BufferInit(rows * m_iClusters,0))
     {
      ReturnFalse
     }
//---
   if(CheckPointer(c_aClasters)==POINTER_INVALID)
     {
      c_aClasters=new CBufferFloat();
      if(CheckPointer(c_aClasters)==POINTER_INVALID)
         return(false);
     }
   c_aClasters.BufferFree();
   if(!c_aClasters.BufferInit(rows,0))
     {
      ReturnFalse
     }
//---
   if(CheckPointer(c_aFlags)==POINTER_INVALID)
     {
      c_aFlags=new CBufferFloat();
      if(CheckPointer(c_aFlags)==POINTER_INVALID)
         return(false);
     }
   c_aFlags.BufferFree();
   if(!c_aFlags.BufferInit(rows,0))
     {
      ReturnFalse
     }
//---
   if(!data.BufferCreate(c_OpenCL)||
      !c_aMeans.BufferCreate(c_OpenCL)||
      !c_aDistance.BufferCreate(c_OpenCL)||
      !c_aClasters.BufferCreate(c_OpenCL)||
      !c_aFlags.BufferCreate(c_OpenCL))
      return(false);
//---
   setBuffer(def_k_kmeans_distance,def_k_kmd_data,data.GetIndex())

   setBuffer(def_k_kmeans_distance,def_k_kmd_means,c_aMeans.GetIndex())

   setBuffer(def_k_kmeans_distance,def_k_kmd_distance,c_aDistance.GetIndex())

   setArgument(def_k_kmeans_distance,def_k_kmd_vector_size,m_iVectorSize)

   uint global_work_offset[2]={0,0};
   uint global_work_size[2];
   global_work_size[0]=rows;
   global_work_size[1]=m_iClusters;
   kernelExecute(def_k_kmeans_distance,global_work_offset,global_work_size)
   if(!c_aDistance.BufferRead())
     {
      ReturnFalse
     }
//---
   setBuffer(def_k_kmeans_clustering,def_k_kmc_flags,c_aFlags.GetIndex())

   setBuffer(def_k_kmeans_clustering,def_k_kmc_clusters,c_aClasters.GetIndex())

   setBuffer(def_k_kmeans_clustering,def_k_kmc_distance,c_aDistance.GetIndex())

   setArgument(def_k_kmeans_clustering,def_k_kmc_total_k,m_iClusters)

   uint global_work_offset1[1]={0};
   uint global_work_size1[1];
   global_work_size1[0]=rows;
   kernelExecute(def_k_kmeans_clustering,global_work_offset1,global_work_size1)
   if(!c_aClasters.BufferRead())
     {
      ReturnFalse
     }
//---
   data.BufferFree();
   c_aDistance.BufferFree();
   c_aFlags.BufferFree();
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| GetLoss - | G et Lo ss implementation.                           |
//+------------------------------------------------------------------+
double CKmeans::GetLoss(CBufferFloat *data)
  {
   if(!Clustering(data))
      return(-1);
//---
   int total=data.Total();
   int rows=total / m_iVectorSize;
//---
   if(CheckPointer(c_aLoss)==POINTER_INVALID)
     {
      c_aLoss=new CBufferFloat();
      if(CheckPointer(c_aLoss)==POINTER_INVALID)
         return(-1);
     }
   if(!c_aLoss.BufferInit(rows,0))
      return(-1);
//---
   if(!data.BufferCreate(c_OpenCL)||
      !c_aLoss.BufferCreate(c_OpenCL))
      return(-1);
//---
   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_loss,def_k_kml_data,data.GetIndex()))
      return(-1);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_loss,def_k_kml_means,c_aMeans.GetIndex()))
      return(-1);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_loss,def_k_kml_clusters,c_aClasters.GetIndex()))
      return(-1);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_loss,def_k_kml_loss,c_aLoss.GetIndex()))
      return(-1);

   if(!c_OpenCL.SetArgument(def_k_kmeans_loss,def_k_kml_vector_size,m_iVectorSize))
      return(-1);

   uint global_work_offset[1]={0};
   uint global_work_size[1];
   global_work_size[0]=rows;
   if(!c_OpenCL.Execute(def_k_kmeans_loss,1,global_work_offset,global_work_size))
      return(-1);
   if(!c_aLoss.BufferRead())
      return(-1);
//---
   m_dLoss=0;
   for(int i=0; i<rows; i++)
      m_dLoss += c_aLoss.At(i);
   m_dLoss /= rows;
//---
   data.BufferFree();
   c_aLoss.BufferFree();

   return(m_dLoss);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Statistic - | S ta ti st ic implementation.                      |
//+------------------------------------------------------------------+
bool CKmeans::Statistic(CBufferFloat *data,CBufferFloat *targets)
  {
   COpenCLMy *OpenCL=c_OpenCL;

   if(CheckPointer(targets)==POINTER_INVALID||
      !Clustering(data))
      return(false);
//---
   if(CheckPointer(c_aProbability)==POINTER_INVALID)
     {
      c_aProbability=new CBufferFloat();
      if(CheckPointer(c_aProbability)==POINTER_INVALID)
         return(false);
     }
   if(!c_aProbability.BufferInit(3 * m_iClusters,0))
     {
      ReturnFalse
     }
//---
   int total=c_aClasters.Total();
   if(!targets.BufferCreate(c_OpenCL)||
      !c_aProbability.BufferCreate(c_OpenCL))
      return(false);
//---
   setBuffer(def_k_kmeans_statistic,def_k_kms_probability,c_aProbability.GetIndex())

   setBuffer(def_k_kmeans_statistic,def_k_kms_targers,targets.GetIndex())

   setBuffer(def_k_kmeans_statistic,def_k_kms_clusters,c_aClasters.GetIndex())

   setArgument(def_k_kmeans_statistic,def_k_kms_total_m,total)

   uint global_work_offset[1]={0};
   uint global_work_size[1];
   global_work_size[0]=m_iClusters;
   kernelExecute(def_k_kmeans_statistic,global_work_offset,global_work_size)
   if(!c_aProbability.BufferRead())
     {
      ReturnFalse
     }
//---
   data.BufferFree();
   targets.BufferFree();
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| GetProbability                                                   |
//+------------------------------------------------------------------+
CBufferFloat *CKmeans::GetProbability(CBufferFloat *data)
  {
   if(CheckPointer(c_aProbability)==POINTER_INVALID||
      !Clustering(data))
      return(NULL);
//---
   CBufferFloat *result=new CBufferFloat();
   if(CheckPointer(result)==POINTER_INVALID)
      return(result);
//---
   int total=c_aClasters.Total();
   if(!result.Reserve(total * 3))
     {
      delete result;

      return(result);
     }
   for(int i=0; i<total; i++)
     {
      int k=(int)c_aClasters.At(i) * 3;
      if(!result.Add(c_aProbability.At(k))||
         !result.Add(c_aProbability.At(k+1))||
         !result.Add(c_aProbability.At(k+2))
        )
        {
         delete result;

         return(result);
        }
     }
//---
   return(result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Save - | S av e implementation.                                  |
//+------------------------------------------------------------------+
bool CKmeans::Save(const int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalseEx("Invalid file handle")
     }

   if(FileWriteInteger(file_handle,Type())<INT_VALUE)
      return(false);

   if(FileWriteInteger(file_handle,m_iClusters)<INT_VALUE)
      return(false);

   if(FileWriteInteger(file_handle,m_iVectorSize)<INT_VALUE)
      return(false);

   dFileWriteUInt(file_handle,m_bTrained)

   if(!c_aMeans.Save(file_handle))
      return(false);

   if(CheckPointer(c_aProbability)==POINTER_INVALID)
     {
      if(FileWriteInteger(file_handle,0)<INT_VALUE)
         return(false);

      return(true);
     }
   if(!c_aProbability.Save(file_handle))
      return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Load - | L oa d implementation.                                  |
//+------------------------------------------------------------------+
bool CKmeans::Load(const int file_handle)
  {
   if(file_handle==INVALID_HANDLE)
     {
      ReturnFalseEx("Invalid file handle")
     }
//---
   if(FileIsEnding(file_handle))
      return(false);
   m_iClusters=FileReadInteger(file_handle);
//---
   if(FileIsEnding(file_handle))
      return(false);
   m_iVectorSize=FileReadInteger(file_handle);
//---
   if(FileIsEnding(file_handle))
      return(false);
   m_bTrained=(bool)FileReadInteger(file_handle);
//---
   if(CheckPointer(c_aMeans)==POINTER_INVALID)
     {
      c_aMeans=new CBufferFloat();
      if(CheckPointer(c_aMeans)==POINTER_INVALID)
         return(false);
     }
   if(FileIsEnding(file_handle)||
      !c_aMeans.Load(file_handle))
      return(false);
//---
   if(FileIsEnding(file_handle))
      return(true);
//---
   if(CheckPointer(c_aProbability)==POINTER_INVALID)
     {
      c_aProbability=new CBufferFloat();
      if(CheckPointer(c_aProbability)==POINTER_INVALID)
         return(false);
     }
   if(!c_aProbability.Load(file_handle))
      return(FileIsEnding(file_handle));
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| SoftMax                                                          |
//+------------------------------------------------------------------+
CBufferFloat *CKmeans::SoftMax(CBufferFloat *data)
  {
   if(!m_bTrained&&!Study(data,(c_aMeans.Maximum()==0)))
      return(NULL);
//---
   if(CheckPointer(data)==POINTER_INVALID||CheckPointer(c_OpenCL)==POINTER_INVALID)
      return(NULL);
//---
   int total=data.Total();
   if(total<=0||m_iClusters<2||(total % m_iVectorSize)!=0)
      return(NULL);
//---
   int rows=total / m_iVectorSize;
   if(rows<1)
      return(NULL);
//---
   if(CheckPointer(c_aDistance)==POINTER_INVALID)
     {
      c_aDistance=new CBufferFloat();
      if(CheckPointer(c_aDistance)==POINTER_INVALID)
         return(NULL);
     }
   c_aDistance.BufferFree();
   if(!c_aDistance.BufferInit(rows * m_iClusters,0))
      return(NULL);
//---
   if(CheckPointer(c_aSoftMax)==POINTER_INVALID)
     {
      c_aSoftMax=new CBufferFloat();
      if(CheckPointer(c_aSoftMax)==POINTER_INVALID)
         return(NULL);
     }
   c_aSoftMax.BufferFree();
   if(!c_aSoftMax.BufferInit(rows * m_iClusters,0))
      return(NULL);
//---
   if(!data.BufferCreate(c_OpenCL)||
      !c_aMeans.BufferCreate(c_OpenCL)||
      !c_aDistance.BufferCreate(c_OpenCL)||
      !c_aSoftMax.BufferCreate(c_OpenCL))
      return(NULL);
//---
   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_distance,def_k_kmd_data,data.GetIndex()))
      return(NULL);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_distance,def_k_kmd_means,c_aMeans.GetIndex()))
      return(NULL);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_distance,def_k_kmd_distance,c_aDistance.GetIndex()))
      return(NULL);

   if(!c_OpenCL.SetArgument(def_k_kmeans_distance,def_k_kmd_vector_size,m_iVectorSize))
      return(NULL);

   uint global_work_offset[2]={0,0};
   uint global_work_size[2];
   global_work_size[0]=rows;
   global_work_size[1]=m_iClusters;
   if(!c_OpenCL.Execute(def_k_kmeans_distance,2,global_work_offset,global_work_size))
      return(NULL);
   if(!c_aDistance.BufferRead())
      return(NULL);
//---
   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_softmax,def_k_kmsm_distance,c_aDistance.GetIndex()))
      return(NULL);

   if(!c_OpenCL.SetArgumentBuffer(def_k_kmeans_softmax,def_k_kmsm_softmax,c_aSoftMax.GetIndex()))
      return(NULL);

   if(!c_OpenCL.SetArgument(def_k_kmeans_softmax,def_k_kmsm_total_k,m_iClusters))
      return(NULL);

   uint global_work_offset1[1]={0};
   uint global_work_size1[1];
   global_work_size1[0]=rows;
   if(!c_OpenCL.Execute(def_k_kmeans_softmax,1,global_work_offset1,global_work_size1))
      return(NULL);
   if(!c_aSoftMax.BufferRead())
      return(NULL);
//---
   data.BufferFree();
   c_aDistance.BufferFree();
//---
   return(c_aSoftMax);
  }
//+------------------------------------------------------------------+

