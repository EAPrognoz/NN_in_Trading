//+------------------------------------------------------------------+
//|                                                        QRDQN.mqh |
//|                                                               RL |
//|                                                   Copyright DNG® |
//|                           https://www.mql5.com/ru/articles/11752 |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define defQRDQN             0x7791   ///<Neuron Net \details Identified class #CQRDQN
//+------------------------------------------------------------------+
//| CQRDQN - Estimate action values with quantile regression         |
//+------------------------------------------------------------------+
class CQRDQN : protected CNet
  {
private:
   uint iCountBackProp;
protected:
   uint iNumbers;
   uint iActions;
   uint iUpdateTarget;
   matrix<float>mTaus;
   //---
   CNet cTargetNet;
public:
   /** Constructor */
                     CQRDQN(void);
//+------------------------------------------------------------------+
//| CQRDQN - Initialize the object state                             |
//+------------------------------------------------------------------+
                     CQRDQN(CArrayObj *Description)  { Create(Description,iActions); }
   bool              Create(CArrayObj *Description,uint actions);
   /** Destructor */~CQRDQN(void);
//+------------------------------------------------------------------+
//| feedForward - Compute the layer output                           |
//+------------------------------------------------------------------+
   bool              feedForward(CArrayFloat *inputVals,int window=1,bool tem=true) ///< Feed Forward method.@param[in] prevLayer Pointer to previos layer. @param[in] window Window of input data. @param[in] tem Use Time Embedding.
     {
      return(CNet::feedForward(inputVals,window,tem,(CBufferFloat*)NULL));
     }
   bool              backProp(CBufferFloat *targetVals,float discount,CArrayFloat *nextState=NULL,int window=1,bool tem=true);                   ///< Back propagation method. @param[in] targetVals Target values
   void              getResults(CBufferFloat *&resultVals);                ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   int               getAction(void);                ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   int               getSample(void);
//+------------------------------------------------------------------+
//| getRecentAverageError - Read the recent average error            |
//+------------------------------------------------------------------+
   float             getRecentAverageError()
     {
      return(recentAverageError);
     } ///< Method to check quality of study. @return Average error
//+------------------------------------------------------------------+
//| Save - Check save                                                |
//+------------------------------------------------------------------+
   bool              Save(string file_name,datetime time,bool common=true)
     {
      return(CNet::Save(file_name,getRecentAverageError(),(float)iActions,0,time,common));
     }
   ///< Save method. @param[in] file_name File name to save @param[in] error Average error @param[in] undefine Undefined percent @param[in] Foecast percent @param[in] time Last study time @param[in] common Common flag
   virtual bool      Save(const int file_handle);
   virtual bool      Load(string file_name,datetime &time,bool common=true);
   ///< Load method. @param[in] file_name File name to save @param[out] error Average error @param[out] undefine Undefined percent @param[out] Foecast percent @param[out] time Last study time @param[in] common Common flag
   virtual bool      Load(const int file_handle);
   //---
//+------------------------------------------------------------------+
//| Type - Return the object type identifier                         |
//+------------------------------------------------------------------+
   virtual int       Type(void)   const
     {
      return(defQRDQN);
     }///< Identificator of class.@return Type of class
//+------------------------------------------------------------------+
//| TrainMode - Set the training mode                                |
//+------------------------------------------------------------------+
   virtual bool      TrainMode(bool flag)
     {
      return(CNet::TrainMode(flag));
     } ///< Set Training Mode Flag
//+------------------------------------------------------------------+
//| GetLayerOutput - Read the layer output                           |
//+------------------------------------------------------------------+
   virtual bool      GetLayerOutput(uint layer,CBufferFloat *&result) ///< Retutn Output data of layer. @param[in] layer Number of layer @param[out] return Buffer with data
     {
      return(CNet::GetLayerOutput(layer,result));
     }
   //---
//+------------------------------------------------------------------+
//| SetUpdateTarget - Set update target                              |
//+------------------------------------------------------------------+
   virtual void      SetUpdateTarget(uint batch)
     {
      iUpdateTarget=batch;
     }
   virtual bool      UpdateTarget(string file_name);
   //---
   virtual bool      SetActions(uint actions);
  };
//+------------------------------------------------------------------+
//| CQRDQN - Estimate action values with quantile regression         |
//+------------------------------------------------------------------+
CQRDQN::CQRDQN()  :  iNumbers(31),
                     iActions(2),
                     iUpdateTarget(1000)
  {
   mTaus=matrix<float>::Ones(1,iNumbers) / iNumbers;
   mTaus[0,0] /= 2;
   mTaus=mTaus.CumSum(0);
   cTargetNet.Create(NULL);
   Create(NULL,iActions);
  }
//+------------------------------------------------------------------+
//| ~CQRDQN - Release owned resources                                |
//+------------------------------------------------------------------+
CQRDQN::~CQRDQN()
  {
  }
//+------------------------------------------------------------------+
//| Create - Check create                                            |
//+------------------------------------------------------------------+
bool CQRDQN::Create(CArrayObj *Description,uint actions)
  {
   if(actions<=0||!CNet::Create(Description))
      return(false);

   int    last_layer=Description.Total()-1;
   CLayer *layer=layers.At(last_layer);
   if(!layer)
      return(false);

   CNeuronBaseOCL *neuron=layer.At(0);
   if(!neuron)
      return(false);

   iActions=actions;
   iNumbers=neuron.Neurons() / actions;
   mTaus=matrix<float>::Ones(1,iNumbers) / iNumbers;
   mTaus[0,0] /= 2;
   mTaus=mTaus.CumSum(0);
   cTargetNet.Create(NULL);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| backProp - Check back prop                                       |
//+------------------------------------------------------------------+
bool CQRDQN::backProp(CBufferFloat *targetVals,float discount,CArrayFloat *nextState=NULL,int window=1,bool tem=true)
  {
//---
   if(!targetVals)
      return(false);

   vectorf target;
   if(!targetVals.GetData(target)||target.Size()!=iActions)
      return(false);
//---
   if(!!nextState)
     {
      if(!cTargetNet.feedForward(nextState,window,tem,(CBufferFloat*)NULL))
         return(false);

      vectorf temp;
      cTargetNet.getResults(targetVals);
      if(!targetVals.GetData(temp))
         return(false);

      matrixf q=matrixf::Zeros(1,temp.Size());
      if(!q.Row(temp,0)||!q.Reshape(iActions,iNumbers))
         return(false);

      temp=q.Mean(0);
      target=target+discount * temp.Max();
     }
//---
   vectorf quantils;
   getResults(targetVals);
   if(!targetVals.GetData(quantils))
      return(false);

   matrixf Q=matrixf::Zeros(1,quantils.Size());
   if(!Q.Row(quantils,0)||!Q.Reshape(iActions,iNumbers))
      return(false);
//---
   for(uint a=0; a<iActions; a++)
     {
      vectorf q=Q.Row(a);
      vectorf dp=q-target[a],dn=dp;
      if(!dp.Clip(0,FLT_MAX)||!dn.Clip(-FLT_MAX,0))
         return(false);

      dp=(mTaus.Row(0)-1) * dp;
      dn=mTaus.Row(0) * dn * (-1);
      if(!Q.Row(dp+dn+q,a))
         return(false);
     }
   if(!targetVals.AssignArray(Q))
      return(false);
//---
   if(iCountBackProp>=iUpdateTarget)
     {
#ifdef FileName
      if(UpdateTarget(FileName+".nnw"))
#else
      if(UpdateTarget("QRDQN.upd"))
#endif
         iCountBackProp=0;
     }
   else
      iCountBackProp++;
//---
   return(CNet::backProp(targetVals,(CBufferFloat*)NULL));
  }
//+------------------------------------------------------------------+
//| getResults - Get results                                         |
//+------------------------------------------------------------------+
void CQRDQN::getResults(CBufferFloat *&resultVals)
  {
   CNet::getResults(resultVals);
   if(!resultVals)
      return;

   vectorf temp;
   if(!resultVals.GetData(temp))
     {
      delete resultVals;
      return;
     }
   matrixf q;
   if(!q.Init(1,temp.Size())||!q.Row(temp,0)||!q.Reshape(iActions,iNumbers))
     {
      delete resultVals;
      return;
     }
//---
   if(!resultVals.AssignArray(q.Mean(1)))
     {
      delete resultVals;
      return;
     }
//---
  }
//+------------------------------------------------------------------+
//| getAction - Get action                                           |
//+------------------------------------------------------------------+
int CQRDQN::getAction(void)
  {
   CBufferFloat *temp;
   getResults(temp);
   if(!temp)
      return(-1);
//---
   return(((CArrayFloat*)temp).Maximum(0,temp.Total()));
  }
//+------------------------------------------------------------------+
//| getSample - Get sample                                           |
//+------------------------------------------------------------------+
int CQRDQN::getSample(void)
  {
   CBufferFloat * resultVals;
   CNet::getResults(resultVals);
   if(!resultVals)
      return(-1);

   vectorf temp;
   if(!resultVals.GetData(temp))
     {
      delete resultVals;

      return(-1);
     }
   delete resultVals;
   matrixf q;
   if(!q.Init(1,temp.Size())||!q.Row(temp,0)||!q.Reshape(iActions,iNumbers))
     {
      delete resultVals;

      return(-1);
     }
//---
   if(!q.Mean(1).Activation(temp,AF_SOFTMAX))
      return(-1);

   temp=temp.CumSum();
   int   err_code;
   float random=(float)MathRandomNormal(0.5,0.5,err_code);
   if(random>=1)
      return((int)temp.Size()-1);

   for(int i=0; i<(int)temp.Size(); i++)
      if(random<=temp[i]&&temp[i]>0)
         return(i);
//---
   return(-1);
  }
//+------------------------------------------------------------------+
//| UpdateTarget - Update target                                     |
//+------------------------------------------------------------------+
bool CQRDQN::UpdateTarget(string file_name)
  {
   if(!Save(file_name,0,false))
      return(false);

   float error;
   float undefine;
   float forecast;
   datetime time;
   if(!cTargetNet.Load(file_name,error,undefine,forecast,time,false))
      return(false);
   iCountBackProp=0;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Save - Check save                                                |
//+------------------------------------------------------------------+
bool CQRDQN::Save(const int file_handle)
  {
   if(!CNet::Save(file_handle))
     {
      ReturnFalseEx("CNet::Save failed")
     }

   dFileWriteUInt(file_handle,iUpdateTarget)
   dFileWriteUInt(file_handle,iActions)
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Load - Check load                                                |
//+------------------------------------------------------------------+
bool CQRDQN::Load(const int file_handle)
  {
   if(!CNet::Load(file_handle))
      return(false);

   if(FileIsEnding(file_handle))
      return(true);

   iUpdateTarget=(uint)FileReadInteger(file_handle);
   if(!SetActions((uint)FileReadInteger(file_handle)))
      return(false);

   if(!UpdateTarget("DQDQN.upd"))
      return(false);
   iCountBackProp=0;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| SetActions - Set actions                                         |
//+------------------------------------------------------------------+
bool CQRDQN::SetActions(uint actions)
  {
   if(actions<=0)
      return(false);

   iActions=actions;
   CLayer *layer=layers.At(layers.Total()-1);
   if(!layer)
      return(false);

   CNeuronBaseOCL *neuron=layer.At(0);
   if(!neuron)
      return(false);

   iNumbers=neuron.Neurons() / iActions;
   mTaus=matrix<float>::Ones(1,iNumbers) / iNumbers;
   mTaus[0,0] /= 2;
   mTaus=mTaus.CumSum(1);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Load - Check load                                                |
//+------------------------------------------------------------------+
bool CQRDQN::Load(string file_name,datetime &time,bool common=true)
  {
   float undefine;
   float forecast;
   if(!CNet::Load(file_name,recentAverageError,undefine,forecast,time,common))
      return(false);
//---
   return(undefine>0 ? SetActions((uint)undefine) : true);
  }
//+------------------------------------------------------------------+

