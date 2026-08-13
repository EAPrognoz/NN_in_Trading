//+------------------------------------------------------------------+
//| Net_SAC_DICE.mqh |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\RL\FQF.mqh"
//---
#define defSACDICE             0x7795   ///<Neuron Net
#define LogProbMultiplier      1.0e-5f

//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
class CNet_SAC_DICE  : protected CNet
  {
protected:
   CNet              cActorExploer;
   CNet              cCritic1;
   CNet              cCritic2;
   CNet              cTargetCritic1;
   CNet              cTargetCritic2;
   CNet              cZeta;
   CNet              cNu;
   CNet              cTargetNu;
   float             fLambda;
   float             fLambda_m;
   float             fLambda_v;
   int               iLatentLayer;
   //---
   float             fLoss1;
   float             fLoss2;
   float             fZeta;
   //---
   vector<float>     GetLogProbability(CBufferFloat *Actions);
public:
   //---
                     CNet_SAC_DICE(void);
                    ~CNet_SAC_DICE(void) {}
   //---
   bool              Create(CArrayObj *actor, CArrayObj *critic, CArrayObj *zeta, CArrayObj *nu, int latent_layer = -1);
   //---
   virtual bool      Study(CArrayFloat *State, CArrayFloat *SecondInput, CBufferFloat *Actions, vector<float> &ActionsLogProbab, CBufferFloat *NextState, CBufferFloat *NextSecondInput, float reward, float discount, float tau);
   virtual void      GetLoss(float &loss1, float &loss2)    {  loss1 = fLoss1; loss2 = fLoss2;  }
   //---
   virtual bool      Save(string file_name, bool common = true);
   bool              Load(string file_name, bool common = true);
  };
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet_SAC_DICE::CNet_SAC_DICE(void)  :  fLambda(1.0e-5f),
                                       fLambda_m(0),
                                       fLambda_v(0),
                                       fLoss1(0),
                                       fLoss2(0),
                                       // Implements fZeta.
                                       fZeta(0)
  {
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes .                                       |
//+------------------------------------------------------------------+
bool CNet_SAC_DICE::Create(CArrayObj *actor, CArrayObj *critic, CArrayObj *zeta, CArrayObj *nu, int latent_layer = -1)
  {
   ResetLastError();
//---
   if(!cActorExploer.Create(actor) || !CNet::Create(actor))
     {
      PrintFormat("Error of create Actor: %d", GetLastError());
      return false;
     }
//---
   if(!opencl)
     {
      Print("Don't opened OpenCL context");
      return false;
     }
//---
   if(!cCritic1.Create(critic) || !cCritic2.Create(critic))
     {
      PrintFormat("Error of create Critic: %d", GetLastError());
      return false;
     }
//---
   if(!cZeta.Create(zeta) || !cNu.Create(nu))
     {
      PrintFormat("Error of create function nets: %d", GetLastError());
      return false;
     }
//---
   if(!cTargetCritic1.Create(critic) || !cTargetCritic2.Create(critic) ||
      !cTargetNu.Create(nu))
     {
      PrintFormat("Error of create target models: %d", GetLastError());
      return false;
     }
//---
   cActorExploer.SetOpenCL(opencl);
   cCritic1.SetOpenCL(opencl);
   cCritic2.SetOpenCL(opencl);
   cZeta.SetOpenCL(opencl);
   cNu.SetOpenCL(opencl);
   cTargetCritic1.SetOpenCL(opencl);
   cTargetCritic2.SetOpenCL(opencl);
   cTargetNu.SetOpenCL(opencl);
//---
   if(!cTargetCritic1.WeightsUpdate(GetPointer(cCritic1), 1.0) ||
      !cTargetCritic2.WeightsUpdate(GetPointer(cCritic2), 1.0) ||
      !cTargetNu.WeightsUpdate(GetPointer(cNu), 1.0))
     {
      PrintFormat("Error of update target models: %d", GetLastError());
      return false;
     }
//---
   fLambda = 1.0e-5f;
   fLambda_m = 0;
   fLambda_v = 0;
   fZeta = 0;
   iLatentLayer = latent_layer;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
bool CNet_SAC_DICE::Study(CArrayFloat *State,
                          CArrayFloat *SecondInput,
                          CBufferFloat *Actions,
                          vector<float> &ActionsLogProbab,
                          CBufferFloat *NextState,
                          CBufferFloat *NextSecondInput,
                          float reward,
                          float discount,
                          float tau)
  {
//---
   if(!Actions || Actions.Total()!=ActionsLogProbab.Size())
      return false;
//---
   if(!CNet::feedForward(NextState, 1, false, NextSecondInput))
      return false;
   if(!cTargetCritic1.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1) ||
      !cTargetCritic2.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1))
      return false;
//---
   if(!cTargetNu.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1))
      return false;
   if(!CNet::feedForward(State, 1, false, SecondInput))
      return false;
   CBufferFloat *output = ((CNeuronBaseOCL*)((CLayer*)layers.At(layers.Total() - 1)).At(0)).getOutput();
   output.AssignArray(Actions);
   output.BufferWrite();
   if(!cNu.feedForward(GetPointer(this), iLatentLayer, GetPointer(this)))
      return false;
   if(!cZeta.feedForward(GetPointer(this), iLatentLayer, GetPointer(this)))
      return false;
//---
   vector<float> nu, next_nu, zeta, ones;
   cNu.getResults(nu);
   cTargetNu.getResults(next_nu);
   cZeta.getResults(zeta);
   ones = vector<float>::Ones(zeta.Size());
   vector<float> log_prob = GetLogProbability(output);
   float policy_ratio = MathExp((log_prob - ActionsLogProbab).Sum());
   vector<float> bellman_residuals = next_nu * discount * policy_ratio - nu + policy_ratio * reward;
   vector<float> zeta_loss = zeta * (MathAbs(bellman_residuals) - fLambda) * (-1) + MathPow(zeta, 2.0f) / 2;
   vector<float> nu_loss = zeta * MathAbs(bellman_residuals) + MathPow(nu, 2.0f) / 2.0f;
   float lambda_los = fLambda * (ones - zeta).Sum();
//--- update lambda
   float grad_lambda = (ones - zeta).Sum() * (-lambda_los);
   fLambda_m = b1 * fLambda_m + (1 - b1) * grad_lambda;
   fLambda_v = b2 * fLambda_v + (1 - b2) * MathPow(grad_lambda, 2);
   fLambda += lr * fLambda_m / (fLambda_v != 0.0f ? MathSqrt(fLambda_v) : 1.0f);
//---
   CBufferFloat temp;
   temp.BufferInit(MathMax(Actions.Total(), SecondInput.Total()), 0);
   temp.BufferCreate(opencl);
//--- update nu
   int last_layer = cNu.layers.Total() - 1;
   CLayer *layer = cNu.layers.At(last_layer);
   if(!layer)
      return false;
   CNeuronBaseOCL *neuron = layer.At(0);
   if(!neuron)
      return false;
   CBufferFloat *buffer = neuron.getGradient();
   if(!buffer)
      return false;
   vector<float> nu_grad = nu_loss * (zeta * bellman_residuals / MathAbs(bellman_residuals) + nu);
   if(!buffer.AssignArray(nu_grad) || !buffer.BufferWrite())
      return false;
   if(!cNu.backPropGradient(output, GetPointer(temp)))
      return false;
//--- update zeta
   last_layer = cZeta.layers.Total() - 1;
   layer = cZeta.layers.At(last_layer);
   if(!layer)
      return false;
   neuron = layer.At(0);
   if(!neuron)
      return false;
   buffer = neuron.getGradient();
   if(!buffer)
      return false;
   vector<float> zeta_grad = zeta_loss * (zeta - MathAbs(bellman_residuals) + fLambda) * (-1);
   if(!buffer.AssignArray(zeta_grad) || !buffer.BufferWrite())
      return false;
   if(!cZeta.backPropGradient(output, GetPointer(temp)))
      return false;
//--- feed forward critics
   if(!cCritic1.feedForward(GetPointer(this), iLatentLayer, output) ||
      !cCritic2.feedForward(GetPointer(this), iLatentLayer, output))
      return false;
//--- target
   vector<float> result;
   if(fZeta == 0)
      fZeta = MathAbs(zeta[0]);
   else
      fZeta = 0.9f * fZeta + 0.1f * MathAbs(zeta[0]);
   zeta[0] = MathPow(MathAbs(zeta[0]), 1.0f / 3.0f) / (10.0f * MathPow(fZeta, 1.0f / 3.0f));
   cTargetCritic1.getResults(result);
   float target = result[0];
   cTargetCritic2.getResults(result);
   target = reward + discount * (MathMin(result[0], target) - LogProbMultiplier * log_prob.Sum());
//--- update critic1
   cCritic1.getResults(result);
   float loss = zeta[0] * MathPow(result[0] - target, 2.0f);
   if(fLoss1 == 0)
      fLoss1 = MathSqrt(loss);
   else
      fLoss1 = MathSqrt(0.999f * MathPow(fLoss1, 2.0f) + 0.001f * loss);
   float grad = loss * 2 * zeta[0] * (target - result[0]);
   last_layer = cCritic1.layers.Total() - 1;
   layer = cCritic1.layers.At(last_layer);
   if(!layer)
      return false;
   neuron = layer.At(0);
   if(!neuron)
      return false;
   buffer = neuron.getGradient();
   if(!buffer)
      return false;
   if(!buffer.Update(0, grad) || !buffer.BufferWrite())
      return false;
   if(!cCritic1.backPropGradient(output, GetPointer(temp)) || !backPropGradient(SecondInput, GetPointer(temp), iLatentLayer))
      return false;
//--- update critic2
   cCritic2.getResults(result);
   loss = zeta[0] * MathPow(result[0] - target, 2.0f);
   if(fLoss2 == 0)
      fLoss2 = MathSqrt(loss);
   else
      fLoss2 = MathSqrt(0.999f * MathPow(fLoss1, 2.0f) + 0.001f * loss);
   grad = loss * 2 * zeta[0] * (target - result[0]);
   last_layer = cCritic2.layers.Total() - 1;
   layer = cCritic2.layers.At(last_layer);
   if(!layer)
      return false;
   neuron = layer.At(0);
   if(!neuron)
      return false;
   buffer = neuron.getGradient();
   if(!buffer)
      return false;
   if(!buffer.Update(0, grad) || !buffer.BufferWrite())
      return false;
   if(!cCritic2.backPropGradient(output, GetPointer(temp)) || !backPropGradient(SecondInput, GetPointer(temp), iLatentLayer))
      return false;
//--- update policy
   cCritic1.getResults(result);
   float mean = result[0];
   float var = result[0];
   cCritic2.getResults(result);
   mean += result[0];
   var -= result[0];
   mean /= 2.0f;
   var = MathAbs(var) / 2.0f;
   target = zeta[0] * (mean - 2.5f * var + discount * log_prob.Sum() * LogProbMultiplier) + result[0];
   CBufferFloat bTarget;
   bTarget.Add(target);
   cCritic2.TrainMode(false);
   if(!cCritic2.backProp(GetPointer(bTarget), GetPointer(this)) ||
      !backPropGradient(SecondInput, GetPointer(temp)))
     {
      cCritic2.TrainMode(true);
      return false;
     }
//--- update exploration policy
   if(!cActorExploer.feedForward(State, 1, false, SecondInput))
     {
      cCritic2.TrainMode(true);
      return false;
     }
   output = ((CNeuronBaseOCL*)((CLayer*)cActorExploer.layers.At(layers.Total() - 1)).At(0)).getOutput();
   output.AssignArray(Actions);
   output.BufferWrite();
   cActorExploer.GetLogProbs(log_prob);
   target = zeta[0] * (mean + 2.0f * var + discount * log_prob.Sum() * LogProbMultiplier) + result[0];
   bTarget.Update(0, target);
   if(!cCritic2.backProp(GetPointer(bTarget), GetPointer(cActorExploer)) ||
      !cActorExploer.backPropGradient(SecondInput, GetPointer(temp)))
     {
      cCritic2.TrainMode(true);
      return false;
     }
   cCritic2.TrainMode(true);
//---
   if(!cTargetCritic1.WeightsUpdate(GetPointer(cCritic1), tau) ||
      !cTargetCritic2.WeightsUpdate(GetPointer(cCritic2), tau) ||
      !cTargetNu.WeightsUpdate(GetPointer(cNu), tau))
     {
      PrintFormat("Error of update target models: %d", GetLastError());
      return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets LogProbability.                                           |
//+------------------------------------------------------------------+
vector<float> CNet_SAC_DICE::GetLogProbability(CBufferFloat *Actions)
  {
   CBufferFloat temp;
   vector<float> result = vector<float>::Zeros(0);
   temp.AssignArray(Actions);
   if(!CalcLogProbs(GetPointer(temp)))
      return result;
   temp.GetData(result);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves .                                                        |
//+------------------------------------------------------------------+
bool CNet_SAC_DICE::Save(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name + ".set", (common ? FILE_COMMON : 0) | FILE_BIN | FILE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;
   if(FileWriteFloat(handle, fLambda) < sizeof(fLambda) ||
      FileWriteFloat(handle, fLambda_m) < sizeof(fLambda_m) ||
      FileWriteFloat(handle, fLambda_v) < sizeof(fLambda_v) ||
      FileWriteInteger(handle, iLatentLayer) < sizeof(iLatentLayer))
      return false;
   FileFlush(handle);
   FileClose(handle);
//---
   if(!CNet::Save(file_name + "Act.nnw", 0, 0, 0, TimeCurrent(), common))
      return false;
//---
   if(!cActorExploer.Save(file_name + "ActExp.nnw", 0, 0, 0, TimeCurrent(), common))
      return false;
//---
   if(!cTargetCritic1.Save(file_name + "Crt1.nnw", fLoss1, 0, 0, TimeCurrent(), common))
      return false;
//---
   if(!cTargetCritic2.Save(file_name + "Crt2.nnw", fLoss2, 0, 0, TimeCurrent(), common))
      return false;
//---
   if(!cZeta.Save(file_name + "Zeta.nnw", 0, 0, 0, TimeCurrent(), common))
      return false;
//---
   if(!cTargetNu.Save(file_name + "Nu.nnw", 0, 0, 0, TimeCurrent(), common))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Loads .                                                        |
//+------------------------------------------------------------------+
bool CNet_SAC_DICE::Load(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name + ".set", (common ? FILE_COMMON : 0) | FILE_BIN | FILE_READ);
   if(handle == INVALID_HANDLE)
      return false;
   if(FileIsEnding(handle))
      return false;
   fLambda = FileReadFloat(handle);
   if(FileIsEnding(handle))
      return false;
   fLambda_m = FileReadFloat(handle);
   if(FileIsEnding(handle))
      return false;
   fLambda_v =   FileReadFloat(handle);
   if(FileIsEnding(handle))
      return false;
   iLatentLayer =  FileReadInteger(handle);;
   FileClose(handle);
//---
   float temp;
   datetime dt;
   if(!CNet::Load(file_name + "Act.nnw", temp, temp, temp, dt, common))
      return false;
//---
   if(!cActorExploer.Load(file_name + "ActExp.nnw", temp, temp, temp, dt, common))
      return false;
//---
   if(!cCritic1.Load(file_name + "Crt1.nnw", fLoss1, temp, temp, dt, common) ||
      !cTargetCritic1.Load(file_name + "Crt1.nnw", temp, temp, temp, dt, common))
      return false;
//---
   if(!cCritic2.Load(file_name + "Crt2.nnw", fLoss2, temp, temp, dt, common) ||
      !cTargetCritic2.Load(file_name + "Crt2.nnw", temp, temp, temp, dt, common))
      return false;
//---
   if(!cZeta.Load(file_name + "Zeta.nnw", temp, temp, temp, dt, common))
      return false;
//---
   if(!cNu.Load(file_name + "Nu.nnw", temp, temp, temp, dt, common) ||
      !cTargetNu.Load(file_name + "Nu.nnw", temp, temp, temp, dt, common))
      return false;
//---
   cActorExploer.SetOpenCL(opencl);
   cCritic1.SetOpenCL(opencl);
   cCritic2.SetOpenCL(opencl);
   cZeta.SetOpenCL(opencl);
   cNu.SetOpenCL(opencl);
   cTargetCritic1.SetOpenCL(opencl);
   cTargetCritic2.SetOpenCL(opencl);
   cTargetNu.SetOpenCL(opencl);
//---
   return true;
  }
//+------------------------------------------------------------------+

