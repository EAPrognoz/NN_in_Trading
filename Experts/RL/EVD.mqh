//+------------------------------------------------------------------+
//|                                                          EVD.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "ICM.mqh"
//---
#define defEVD              0x7794   ///<Neuron Net \details Identified class #CICM
//+------------------------------------------------------------------+
//| Exploration via Disagreement                                     |
//+------------------------------------------------------------------+
class CEVD : protected CNet
  {
protected:
   uint              iMinBufferSize;
   uint              iStateEmbedingLayer;
   double            dPrevBalance;
   bool              bUseTargetNet;
   bool              bTrainMode;
   //---
   CNet              cTargetNet;
   CReplayBuffer     cReplay;
   CNet              cForwardNet;

   virtual bool      AddInputData(CArrayFloat *inputVals);

public:
                     CEVD();
                     CEVD(CArrayObj *Description, CArrayObj *Forward);
   bool              Create(CArrayObj *Description, CArrayObj *Forward);
                    ~CEVD();
   int               feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true); ///< Feed Forward method.@param[in] prevLayer Pointer to previos layer. @param[in] window Window of input data. @param[in] tem Use Time Embedding.
   bool              backProp(int batch, float discount = 0.999f);
   int               getAction(int state_size = 0);              ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   float             getRecentAverageError() { return recentAverageError; } ///< Method to check quality of study. @return Average error
   bool              Save(string file_name, bool common = true);
   bool              Save(string dqn, string forward, bool common = true);
   ///< Save method. @param[in] file_name File name to save @param[in] error Average error @param[in] undefine Undefined percent @param[in] Foecast percent @param[in] time Last study time @param[in] common Common flag
   virtual bool      Load(string file_name, bool common = true);
   bool              Load(string dqn, string forward, uint state_layer, bool common = true);
   ///< Load method. @param[in] file_name File name to save @param[out] error Average error @param[out] undefine Undefined percent @param[out] Foecast percent @param[out] time Last study time @param[in] common Common flag
   //---
   virtual int       Type(void)   const   {  return defEVD;   }///< Identificator of class.@return Type of class
   virtual bool      TrainMode(bool flag) { bTrainMode = flag; return (CNet::TrainMode(flag) && cForwardNet.TrainMode(flag)); } ///< Set Training Mode Flag
   virtual bool      GetLayerOutput(uint layer, CBufferFloat *&result) ///< Retutn Output data of layer. @param[in] layer Number of layer @param[out] return Buffer with data
     { return        CNet::GetLayerOutput(layer, result); }
   //---
   virtual void      SetStateEmbedingLayer(uint layer) { iStateEmbedingLayer = layer; }
   virtual void      SetBufferSize(uint min, uint max);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CEVD::CEVD()   :  iMinBufferSize(100),
   bUseTargetNet(false),
   bTrainMode(true)
  {
   Create(NULL, NULL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CEVD::~CEVD()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::Create(CArrayObj *Description, CArrayObj *Forward)
  {
   if(!CNet::Create(Description))
      return false;
   if(!cForwardNet.Create(Forward))
      return false;
   cTargetNet.Create(Description);
   bUseTargetNet = false;
   bTrainMode = true;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::backProp(int batch, float discount = 0.999000f)
  {
//---
   if(cReplay.Total() < (int)iMinBufferSize || !bTrainMode)
      return true;
//---
   CBufferFloat *state1, *state2, *targetVals = new CBufferFloat();
   vector<float> target, actions, st1, st2, result;
   matrix<float> forward;
   double reward;
   int action;
//--- цикл обучения в размере batch
   for(int i = 0; i < batch; i++)
     {
      //--- получаем случайное состояние и реплай буфера
      if(!cReplay.GetRendomState(state1, action, reward, state2))
         return false;
      //--- прямой проход обучаемой моделм ("текущее" состояие)
      if(!CNet::feedForward(state1, 1, false))
         return false;
      getResults(target);
      //--- выгружаем эмбединг состояния
      if(!GetLayerOutput(iStateEmbedingLayer, state1))
         return false;
      //--- прямой проход target net
      if(!cTargetNet.feedForward(state2, 1, false))
         return false;
      //--- корректировка вознаграждения
      if(bUseTargetNet)
        {
         cTargetNet.getResults(targetVals);
         reward += discount * targetVals.Maximum();
        }
      target[action] = (float)reward;
      if(!targetVals.AssignArray(target))
         return false;
      //--- обратный проход обучаемой модели
      CNet::backProp(targetVals);
      //--- прямой проход forward net - прогноз следующего состояния
      if(!cForwardNet.feedForward(state1, 1, false))
         return false;
      //--- выгружаем эмбединг "будущего" состояния
      if(!cTargetNet.GetLayerOutput(iStateEmbedingLayer, state2))
         return false;
      //--- подготовка целей forward net
      cForwardNet.getResults(result);
      forward.Init(1, result.Size());
      forward.Row(result, 0);
      forward.Reshape(result.Size() / state2.Total(), state2.Total());
      int ensemble = (int)(forward.Rows() / target.Size());
      //--- копируем целевое состояние в матрицу целей ансамбля
      state2.GetData(st2);
      for(int r = 0; r < ensemble; r++)
         forward.Row(st2, r * target.Size() + action);
      //--- обратный проход foward net
      targetVals.AssignArray(forward);
      if(!cForwardNet.backProp(targetVals))
         return false;
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
int CEVD::getAction(int state_size = 0)
  {
   CBufferFloat *temp;
//--- получаем результат обучаемой модели.
   CNet::getResults(temp);
   if(!temp)
      return -1;
//--- в режиме обучения делаем поправку на "любопытство"
   if(bTrainMode && state_size > 0)
     {
      vector<float> model;
      matrix<float> forward;
      cForwardNet.getResults(model);
      forward.Init(1, model.Size());
      forward.Row(model, 0);
      temp.GetData(model);
//      
cForwardNet.GetLayerOutput(1,temp);
      //---
      int actions = (int)model.Size();
      forward.Reshape(forward.Cols() / state_size, state_size);
      matrix<float> ensemble[];
      if(!forward.Hsplit(forward.Rows() / actions, ensemble))
         return -1;
      matrix<float> means = ensemble[0];
      int total = ArraySize(ensemble);
      for(int i = 1; i < total; i++)
         means += ensemble[i];
      means = means / total;
      for(int i = 0; i < total; i++)
         ensemble[i] -= means;
      means = MathPow(ensemble[0], 2.0);
      for(int i = 1; i < total; i++)
         means += MathPow(ensemble[i], 2.0);
      model += means.Sum(1) / total;
      temp.AssignArray(model);
     }
//---
   return temp.Argmax();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::AddInputData(CArrayFloat *inputVals)
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
int CEVD::feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true)
  {
   if(!AddInputData(inputVals))
      return -1;
//---
   if(!CNet::feedForward(inputVals, window, tem))
      return -1;
//---
   int action = -1;
   if(bTrainMode)
     {
      CBufferFloat *state;
      //if(!GetLayerOutput(1, state))
      //   return -1;
      if(!GetLayerOutput(iStateEmbedingLayer, state))
         return -1;
      if(!cForwardNet.feedForward(state, 1, false))
        {
         delete state;
         return -1;
        }
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double reward = (dPrevBalance == 0 ? 0 : balance - dPrevBalance);
      dPrevBalance = balance;
      action = getAction(state.Total());
      delete state;
      if(action < 0 || action > 3)
         return -1;
      if(!cReplay.AddState(inputVals, action, reward))
         return -1;
     }
   else
      action = getAction();
//---
   return action;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::Save(string file_name, bool common = true)
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
   if(!CNet::Save(handle) || !cForwardNet.Save(handle))
      result = false;
   FileFlush(handle);
   FileClose(handle);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::Load(string file_name, bool common = true)
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
   FileFlush(handle);
   FileClose(handle);
   float temp = 0;
   datetime dt = 0;
   if(CNet::Save(TargetNetFile, temp, temp, temp, dt, false))
      bUseTargetNet = cTargetNet.Load(TargetNetFile, temp, temp, temp, dt, false);
   TrainMode(true);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::Save(string dqn, string forward, bool common = true)
  {
   if(dqn == NULL || forward == NULL)
      return false;
   bool result = true;
   if(!CNet::Save(dqn, getRecentAverageError(), 0, 0, 0, common) ||
      !cForwardNet.Save(forward, cForwardNet.getRecentAverageError(), 0, 0, 0, common))
      result = false;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CEVD::Load(string dqn, string forward, uint state_layer, bool common = true)
  {
   if(dqn == NULL || forward == NULL)
      return false;
   bool result = true;
   float err, undef, forecast;
   datetime date;
   if(!CNet::Load(dqn, err, undef, forecast, date, common) ||
      !cForwardNet.Load(forward, err, undef, forecast, date, common) ||
      !cTargetNet.Load(dqn, err, undef, forecast, date, common))
      result = false;
   iStateEmbedingLayer = state_layer;
   TrainMode(true);
   bUseTargetNet = true;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CEVD::SetBufferSize(uint min, uint max)
  {
   iMinBufferSize = MathMin(min, max);
   cReplay.SetMaxSize(max);
  }
//+------------------------------------------------------------------+

