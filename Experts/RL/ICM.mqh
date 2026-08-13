//+------------------------------------------------------------------+
//|                                                          ICM.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\NeuroNet_DNG\NeuroNet.mqh"
//---
#define defICM              0x7793   ///<Neuron Net \details Identified class #CICM
#define TargetNetFile       "ICM.upd"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CReplayState : public CObject
  {
protected:
   CBufferFloat      cState;
   int               iAction;
   double            dReaward;

public:
                     CReplayState(CBufferFloat *state, int action, double reward);
                    ~CReplayState(void) {};
   bool              GetCurrent(CBufferFloat *&state, int &action);
   bool              GetNext(CBufferFloat *&state, double &reward);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CReplayState::CReplayState(CBufferFloat *state, int action, double reward)
  {
   cState.AssignArray(state);
   iAction = action;
   dReaward = reward;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CReplayState::GetCurrent(CBufferFloat *&state, int &action)
  {
   action = iAction;
   double reward;
   return GetNext(state, reward);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CReplayState::GetNext(CBufferFloat *&state, double &reward)
  {
   reward = dReaward;
   if(!state)
     {
      state = new CBufferFloat();
      if(!state)
         return false;
     }
   return state.AssignArray(GetPointer(cState));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CReplayBuffer : protected CArrayObj
  {
protected:
   uint              iMaxSize;
public:
                     CReplayBuffer(void) : iMaxSize(500) {};
                    ~CReplayBuffer(void) {};
   //---
   void              SetMaxSize(uint size)   {  iMaxSize = size; }
   bool              AddState(CBufferFloat *state, int action, double reward);
   bool              GetRendomState(CBufferFloat *&state1, int &action, double &reward, CBufferFloat*& state2);
   bool              GetState(int position, CBufferFloat *&state1, int &action, double &reward, CBufferFloat*& state2);
   int               Total(void) { return CArrayObj::Total(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CReplayBuffer::AddState(CBufferFloat *state, int action, double reward)
  {
   if(!state)
      return false;
//---
   if(!Add(new CReplayState(state, action, reward)))
      return false;
   while(Total() > (int)iMaxSize)
      Delete(0);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CReplayBuffer::GetRendomState(CBufferFloat *&state1, int &action, double &reward, CBufferFloat *&state2)
  {
   int position = (int)(MathRand() * MathRand() / pow(32767.0, 2.0) * (Total() - 1));
   return GetState(position, state1, action, reward, state2);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CReplayBuffer::GetState(int position, CBufferFloat *&state1, int &action, double &reward, CBufferFloat *&state2)
  {
   if(position < 0 || position >= (Total() - 1))
      return false;
   CReplayState* element = m_data[position];
   if(!element || !element.GetCurrent(state1, action))
      return false;
   element = m_data[position + 1];
   if(!element.GetNext(state2, reward))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CICM : protected CNet
  {
protected:
   uint              iMinBufferSize;
   uint              iStateEmbedingLayer;
   double            dPrevBalance;
   //---
   CNet              cTargetNet;
   CReplayBuffer     cReplay;
   CNet              cInverseNet;
   CNet              cForwardNet;

   virtual bool      AddInputData(CArrayFloat *inputVals);

public:
   /** Constructor */
                     CICM(void);
                     CICM(CArrayObj *Description, CArrayObj *Forward, CArrayObj *Inverse);
   bool              Create(CArrayObj *Description, CArrayObj *Forward, CArrayObj *Inverse);
   /** Destructor */~CICM(void);
   int               feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true, bool sample = true); ///< Feed Forward method.@param[in] prevLayer Pointer to previos layer. @param[in] window Window of input data. @param[in] tem Use Time Embedding.
   bool              backProp(int batch, float discount = 0.999f);
   int               getAction(void);                ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   int               getSample(void);
   float             getRecentAverageError() { return recentAverageError; } ///< Method to check quality of study. @return Average error
   bool              Save(string file_name, bool common = true);
   bool              Save(string dqn, string forward, string invers, bool common = true);
   ///< Save method. @param[in] file_name File name to save @param[in] error Average error @param[in] undefine Undefined percent @param[in] Foecast percent @param[in] time Last study time @param[in] common Common flag
   virtual bool      Load(string file_name, bool common = true);
   bool              Load(string dqn, string forward, string invers, uint state_layer, bool common = true);
   ///< Load method. @param[in] file_name File name to save @param[out] error Average error @param[out] undefine Undefined percent @param[out] Foecast percent @param[out] time Last study time @param[in] common Common flag
   //---
   virtual int       Type(void)   const   {  return defICM;   }///< Identificator of class.@return Type of class
   virtual bool      TrainMode(bool flag) { return (CNet::TrainMode(flag) && cForwardNet.TrainMode(flag) && cInverseNet.TrainMode(flag)); } ///< Set Training Mode Flag
   virtual bool      GetLayerOutput(uint layer, CBufferFloat *&result) ///< Retutn Output data of layer. @param[in] layer Number of layer @param[out] return Buffer with data
     { return        CNet::GetLayerOutput(layer, result); }
   //---
   virtual bool      UpdateTarget(string file_name);
   virtual void      SetStateEmbedingLayer(uint layer) { iStateEmbedingLayer = layer; }
   virtual void      SetBufferSize(uint min, uint max);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CICM::CICM(void)   :  iMinBufferSize(100)
  {
   cTargetNet.Create(NULL);
   Create(NULL, NULL, NULL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CICM::~CICM()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::Create(CArrayObj *Description, CArrayObj *Forward, CArrayObj *Inverse)
  {
   if(!CNet::Create(Description))
      return false;
   if(!cForwardNet.Create(Forward))
      return false;
   if(!cInverseNet.Create(Inverse))
      return false;
   cTargetNet.Create(NULL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::backProp(int batch, float discount = 0.999000f)
  {
//---
   if(cReplay.Total() < (int)iMinBufferSize)
      return true;
   //if(!UpdateTarget(TargetNetFile))
   //   return false;
//---
   CLayer *currentLayer, *nextLayer, *prevLayer;
   CNeuronBaseOCL *neuron;
   CBufferFloat *state1, *state2, *targetVals = new CBufferFloat();
   vector<float> target, actions, st1, st2, result;
   double reward;
   int action;
//--- цикл обучения в размере batch
   for(int i = 0; i < batch; i++)
     {
      //--- получаем случайное состояние и реплай буфера
      if(!cReplay.GetRendomState(state1, action, reward, state2))
         return false;
      //--- прямой проход обучаемой моделм ("текущее" состояие)
      if(!CNet::feedForward(state1, 1, false, (CBufferFloat*)NULL))
         return false;
      //--- выгружаем эмбединг состояния
      if(!GetLayerOutput(iStateEmbedingLayer, state1))
         return false;
      //--- подготавливаем one-hote вектор действия и конкатенируем с вектором текущего состояния
      getResults(target);
      actions = vector<float>::Zeros(target.Size());
      actions[action] = 1;
      if(!targetVals.AssignArray(actions) || !targetVals.AddArray(state1))
         return false;
      //--- прямой проход forward net - прогноз следующего состояния
      if(!cForwardNet.feedForward(targetVals, 1, false, (CBufferFloat*)NULL))
         return false;
      //--- прямой проход
      if(!cTargetNet.feedForward(state2, 1, false, (CBufferFloat*)NULL))
         return false;
      //--- выгружаем эмбединг состояния и соединяем с эмбедингом "текущего" состояния
      if(!cTargetNet.GetLayerOutput(iStateEmbedingLayer, state2))
         return false;
      //--- прямой проход inverse net - определение совершенного действия.
      if(!state1.AddArray(state2) || !cInverseNet.feedForward(state1, 1, false, (CBufferFloat*)NULL))
         return false;
      //--- обратный проход inverse net
      if(!targetVals.AssignArray(actions) || !cInverseNet.backProp(targetVals, (CBufferFloat*)NULL))
         return false;
      //--- обратный проход forward net
      if(!cForwardNet.backProp(state2, (CBufferFloat*)NULL))
         return false;
      //--- корректировка вознаграждения
      cForwardNet.getResults(st1);
      state2.GetData(st2);
      reward += (MathPow(st2 - st1, 2)).Sum();
      cTargetNet.getResults(targetVals);
      target[action] = (float)(reward + discount * targetVals.Maximum());
      if(!targetVals.AssignArray(target))
         return false;
      //--- обратный проход обучаемой модели
        {
         getResults(result);
         float error = result.Loss(target, LOSS_MSE);
         //---
         currentLayer = layers.At(layers.Total() - 1);
         if(CheckPointer(currentLayer) == POINTER_INVALID)
            return false;
         neuron = currentLayer.At(0);
         if(!neuron.calcOutputGradients(targetVals, error))
            return false;
         //---
         backPropCount++;
         recentAverageError += (error - recentAverageError) / fmin(recentAverageSmoothingFactor, (float)backPropCount);
         //--- Calc Hidden Gradients
         int total = layers.Total();
         for(int layerNum = total - 2; layerNum > 0; layerNum--)
           {
            nextLayer = currentLayer;
            currentLayer = layers.At(layerNum);
            neuron = currentLayer.At(0);
            if(!neuron.CalcHiddenGradients(nextLayer.At(0)))
               return false;
            if(layerNum == iStateEmbedingLayer)
              {
               CLayer* temp = cInverseNet.layers.At(0);
               CNeuronBaseOCL* inv = temp.At(0);
               uint global_work_offset[1] = {0};
               uint global_work_size[1];
               global_work_size[0] = neuron.Neurons();
               opencl.SetArgumentBuffer(def_k_MatrixSum, def_k_sum_matrix1, neuron.getGradientIndex());
               opencl.SetArgumentBuffer(def_k_MatrixSum, def_k_sum_matrix2, inv.getGradientIndex());
               opencl.SetArgumentBuffer(def_k_MatrixSum, def_k_sum_matrix_out, neuron.getGradientIndex());
               opencl.SetArgument(def_k_MatrixSum, def_k_sum_dimension, 1);
               opencl.SetArgument(def_k_MatrixSum, def_k_sum_multiplyer, 1.0f);
               if(!opencl.Execute(def_k_MatrixSum, 1, global_work_offset, global_work_size))
                 {
                  printf("Error of execution kernel MatrixSum: %d", GetLastError());
                  return false;
                 }
              }
           }
         //---
         prevLayer = layers.At(total - 1);
         for(int layerNum = total - 1; layerNum > 0; layerNum--)
           {
            currentLayer = prevLayer;
            prevLayer = layers.At(layerNum - 1);
            neuron = currentLayer.At(0);
            if(!neuron.UpdateInputWeights(prevLayer.At(0)))
               return false;
           }
         //---
         for(int layerNum = 0; layerNum < total; layerNum++)
           {
            currentLayer = layers.At(layerNum);
            CNeuronBaseOCL *temp = currentLayer.At(0);
            if(!temp.TrainMode())
               continue;
            if((layerNum + 1) == total && !temp.getGradient().BufferRead())
               return false;
            break;
           }
        }
     }
//---
   delete state1;
   delete state2;
   delete targetVals;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CICM::getAction(void)
  {
   CBufferFloat *temp;
   CNet::getResults(temp);
   if(!temp)
      return -1;
//---
   return temp.Argmax();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CICM::getSample(void)
  {
   CBufferFloat* resultVals;
   CNet::getResults(resultVals);
   if(!resultVals)
      return -1;
   vectorf temp;
   if(!resultVals.GetData(temp))
     {
      delete resultVals;
      return -1;
     }
   delete resultVals;
//---
   if(!temp.Activation(temp, AF_SOFTMAX))
      return -1;
   temp = temp.CumSum();
   int err_code;
   float random = (float)MathRandomNormal(0.5, 0.5, err_code);
   if(random >= 1)
      return (int)temp.Size() - 1;
   for(int i = 0; i < (int)temp.Size(); i++)
      if(random <= temp[i] && temp[i] > 0)
         return i;
//---
   return -1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::UpdateTarget(string file_name)
  {
   if(!CNet::Save(file_name, 0, 0, 0, true))
      return false;
   float error, undefine, forecast;
   datetime time;
   if(!cTargetNet.Load(file_name, error, undefine, forecast, time, true))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::AddInputData(CArrayFloat *inputVals)
  {
   if(!inputVals)
      return false;
   if(!inputVals.Add((float)AccountInfoDouble(ACCOUNT_BALANCE)))
      return false;
   if(!inputVals.Add((float)AccountInfoDouble(ACCOUNT_EQUITY)))
      return false;
   if(!inputVals.Add((float)AccountInfoDouble(ACCOUNT_MARGIN_FREE)))
      return false;
   if(!inputVals.Add((float)AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)))
      return false;
   if(!inputVals.Add((float)AccountInfoDouble(ACCOUNT_PROFIT)))
      return false;
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += PositionGetDouble(POSITION_PROFIT);
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += PositionGetDouble(POSITION_PROFIT);
            break;
        }
     }
   if(!inputVals.Add((float)buy_value))
      return false;
   if(!inputVals.Add((float)sell_value))
      return false;
   if(!inputVals.Add((float)buy_profit))
      return false;
   if(!inputVals.Add((float)sell_profit))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CICM::feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true, bool sample = true)
  {
   if(!AddInputData(inputVals))
      return -1;
//---
   if(!CNet::feedForward(inputVals, window, tem, (CBufferFloat*)NULL))
      return -1;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double reward = (dPrevBalance == 0 ? 0 : balance - dPrevBalance);
   dPrevBalance = balance;
   int action = (sample ? getSample() : getAction());
   if(!cReplay.AddState(inputVals, action, reward))
      return -1;
//---
   return action;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::Save(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name, (common ? FILE_COMMON : 0) | FILE_BIN | FILE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;
//---
   if(FileWriteInteger(handle, iMinBufferSize) <= 0 || FileWriteInteger(handle, iStateEmbedingLayer) <= 0)
     {
      FileClose(handle);
      return false;
     }
   bool result = true;
   if(!CNet::Save(handle) || !cForwardNet.Save(handle) || !cInverseNet.Save(handle))
      result = false;
   FileFlush(handle);
   FileClose(handle);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::Load(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name, (common ? FILE_COMMON : 0) | FILE_BIN | FILE_READ);
   if(handle == INVALID_HANDLE)
      return false;
//---
   iMinBufferSize = (uint)FileReadInteger(handle);
   iStateEmbedingLayer = (uint)FileReadInteger(handle);
   bool result = true;
   if(!CNet::Load(handle))
      result = false;
   if(!cForwardNet.Load(handle))
      result = false;
   if(!cInverseNet.Load(handle))
      result = false;
   FileFlush(handle);
   FileClose(handle);
   UpdateTarget(TargetNetFile);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::Save(string dqn, string forward, string invers, bool common = true)
  {
   if(dqn == NULL || forward == NULL || invers == NULL)
      return false;
   bool result = true;
   if(!CNet::Save(dqn, getRecentAverageError(), 0, 0, 0, common) ||
      !cForwardNet.Save(forward, cForwardNet.getRecentAverageError(), 0, 0, 0, common) ||
      !cInverseNet.Save(invers, cInverseNet.getRecentAverageError(), 0, 0, 0, common))
      result = false;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CICM::Load(string dqn, string forward, string invers, uint state_layer, bool common = true)
  {
   if(dqn == NULL || forward == NULL || invers == NULL)
      return false;
   bool result = true;
   float err, undef, forecast;
   datetime date;
   if(!CNet::Load(dqn, err, undef, forecast, date, common) ||
      !cForwardNet.Load(forward, err, undef, forecast, date, common) ||
      !cInverseNet.Load(invers, err, undef, forecast, date, common))
      result = false;
   iStateEmbedingLayer = state_layer;
   cTargetNet.Load(dqn, err, undef, forecast, date, common);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CICM::SetBufferSize(uint min, uint max)
  {
   iMinBufferSize = MathMin(min, max);
   cReplay.SetMaxSize(max);
  }
//+------------------------------------------------------------------+

