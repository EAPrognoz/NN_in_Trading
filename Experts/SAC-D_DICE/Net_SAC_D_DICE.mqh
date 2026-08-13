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
class CNet_SAC_D_DICE  : protected CNet
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
   vector<float>     fLambda;
   vector<float>     fLambda_m;
   vector<float>     fLambda_v;
   int               iLatentLayer;
   float             fCAGrad_C;
   int               iCAGrad_Iters;
   int               iUpdateDelay;
   int               iUpdateDelayCount;
   //---
   float             fLoss1;
   float             fLoss2;
   vector<float>     fZeta;
   vector<float>     fQWeights;
   //---
   vector<float>     GetLogProbability(CBufferFloat *Actions);
   vector<float>     CAGrad(vector<float> &grad);

public:
   //---
                     CNet_SAC_D_DICE(void);
                    ~CNet_SAC_D_DICE(void) {}
   //---
   bool              Create(CArrayObj *actor, CArrayObj *critic, CArrayObj *zeta, CArrayObj *nu, int latent_layer = -1);
   //---
   virtual bool      Study(CArrayFloat *State, CArrayFloat *SecondInput, CBufferFloat *Actions, vector<float> &Rewards, CBufferFloat *NextState, CBufferFloat *NextSecondInput, float discount, float tau);
   virtual void      GetLoss(float &loss1, float &loss2)    {  loss1 = fLoss1; loss2 = fLoss2;              }
   virtual bool      TargetsUpdate(float tau);
//---
   virtual void      SetQWeights(vector<float> &weights)    {  fQWeights=weights;                           }
   virtual void      SetCAGradC(float c)                    {  fCAGrad_C=c;                                 }
   virtual void      SetLambda(vector<float> &lambda)       {  fLambda=lambda;
                                                               fLambda_m=vector<float>::Zeros(lambda.Size());
                                                               fLambda_v=fLambda_m;                         }
   virtual void      TargetsUpdateDelay(int delay)          {  iUpdateDelay=delay; iUpdateDelayCount=delay; }
   //---
   virtual bool      Save(string file_name, bool common = true);
   bool              Load(string file_name, bool common = true);
  };
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet_SAC_D_DICE::CNet_SAC_D_DICE(void) :  fLoss1(0),
                                          fLoss2(0),
                                          fCAGrad_C(0.5f),
                                          iCAGrad_Iters(15),
                                          iUpdateDelay(100),
                                          // Implements iUpdateDelayCount.
                                          iUpdateDelayCount(100)
  {
   fLambda = vector<float>::Full(1, 1.0e-5f);
   fLambda_m = vector<float>::Zeros(1);
   fLambda_v = vector<float>::Zeros(1);
   fZeta = vector<float>::Zeros(1);
   fQWeights = vector<float>::Ones(1);
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates or initializes .                                       |
//+------------------------------------------------------------------+
bool CNet_SAC_D_DICE::Create(CArrayObj *actor, CArrayObj *critic, CArrayObj *zeta, CArrayObj *nu, int latent_layer = -1)
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
   cZeta.getResults(fZeta);
   ulong size = fZeta.Size();
   fLambda = vector<float>::Full(size,1.0e-5f);
   fLambda_m = vector<float>::Zeros(size);
   fLambda_v = vector<float>::Zeros(size);
   fQWeights = vector<float>::Ones(size);
   iLatentLayer = latent_layer;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
bool CNet_SAC_D_DICE::Study(CArrayFloat *State,
                            CArrayFloat *SecondInput,
                            CBufferFloat *Actions,
                            vector<float> &Rewards,
                            CBufferFloat *NextState,
                            CBufferFloat *NextSecondInput,
                            float discount,
                            float tau)
  {
//---
   if(!Actions)
      return false;
//---
   if(!!NextState)
      if(!CNet::feedForward(NextState, 1, false, NextSecondInput))
         return false;
   if(!cTargetCritic1.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1) ||
      !cTargetCritic2.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1))
      return false;
//---
   if(!cTargetNu.feedForward(GetPointer(this), iLatentLayer, GetPointer(this), layers.Total() - 1))
      return false;
//---
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
   cZeta.getResults(zeta);
   if(!!NextState)
      cTargetNu.getResults(next_nu);
   else
      next_nu = vector<float>::Zeros(nu.Size());
   ones = vector<float>::Ones(zeta.Size());
   vector<float> log_prob = GetLogProbability(output);
   int shift = (int)(Rewards.Size() - log_prob.Size());
   if(shift < 0)
      return false;
   float policy_ratio = 0;
   for(ulong i = 0; i < log_prob.Size(); i++)
      policy_ratio += log_prob[i] - Rewards[shift + i] / LogProbMultiplier;
   policy_ratio = MathExp(policy_ratio / log_prob.Size());
   vector<float> bellman_residuals = (next_nu * discount + Rewards) * policy_ratio - nu;
   vector<float> zeta_loss = MathPow(zeta, 2.0f) / 2.0f - zeta * (MathAbs(bellman_residuals) - fLambda) ;
   vector<float> nu_loss = zeta * MathAbs(bellman_residuals) + MathPow(nu, 2.0f) / 2.0f;
   vector<float> lambda_los = fLambda * (ones - zeta);
//--- update lambda
   vector<float> grad_lambda = CAGrad((ones - zeta) * (lambda_los * (-1.0f)));
   fLambda_m = fLambda_m * b1 + grad_lambda * (1 - b1);
   fLambda_v = fLambda_v * b2 + MathPow(grad_lambda, 2) * (1.0f - b2);
   fLambda += fLambda_m * lr / MathSqrt(fLambda_v + lr / 100.0f);
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
   vector<float> nu_grad = CAGrad(nu_loss * (zeta * bellman_residuals / MathAbs(bellman_residuals) - nu));
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
   vector<float> zeta_grad = CAGrad(zeta_loss * (zeta - MathAbs(bellman_residuals) + fLambda) * (-1.0f));
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
   if(fZeta.CompareByDigits(vector<float>::Zeros(fZeta.Size()),8) == 0)
      fZeta = MathAbs(zeta);
   else
      fZeta = fZeta * 0.9f + MathAbs(zeta) * 0.1f;
   zeta = MathPow(MathAbs(zeta), 1.0f / 3.0f) / (MathPow(fZeta, 1.0f / 3.0f) * 10.0f);
   vector<float> target = vector<float>::Zeros(Rewards.Size());
   if(!!NextState)
     {
      cTargetCritic1.getResults(target);
      cTargetCritic2.getResults(result);
      if(fQWeights.Dot(result) < fQWeights.Dot(target))
         target = result;
     }
   target = (target * discount + Rewards);
   ulong total = log_prob.Size();
   for(ulong i = 0; i < total; i++)
      target[shift + i] = log_prob[i] * LogProbMultiplier;
//--- update critic1
   cCritic1.getResults(result);
   vector<float> loss = zeta * MathPow(result - target, 2.0f);
   if(fLoss1 == 0)
      fLoss1 = MathSqrt(fQWeights.Dot(loss) / fQWeights.Sum());
   else
      fLoss1 = MathSqrt(0.999f * MathPow(fLoss1, 2.0f) + 0.001f * fQWeights.Dot(loss) / fQWeights.Sum());
   vector<float> grad = CAGrad(loss * zeta * (target - result) * 2.0f);
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
   if(!buffer.AssignArray(grad) || !buffer.BufferWrite())
      return false;
   if(!cCritic1.backPropGradient(output, GetPointer(temp)) || !backPropGradient(SecondInput, GetPointer(temp), iLatentLayer))
      return false;
//--- update critic2
   cCritic2.getResults(result);
   loss = zeta * MathPow(result - target, 2.0f);
   if(fLoss2 == 0)
      fLoss2 = MathSqrt(fQWeights.Dot(loss) / fQWeights.Sum());
   else
      fLoss2 = MathSqrt(0.999f * MathPow(fLoss2, 2.0f) + 0.001f * fQWeights.Dot(loss) / fQWeights.Sum());
   grad = CAGrad(loss * zeta * (target - result) * 2.0f);
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
   if(!buffer.AssignArray(grad) || !buffer.BufferWrite())
      return false;
   if(!cCritic2.backPropGradient(output, GetPointer(temp)) || !backPropGradient(SecondInput, GetPointer(temp), iLatentLayer))
      return false;
//--- update policy
   vector<float> mean;
   CNet *critic = NULL;
   if(fLoss1 <= fLoss2)
     {
      cCritic1.getResults(result);
      cCritic2.getResults(mean);
      critic = GetPointer(cCritic1);
     }
   else
     {
      cCritic1.getResults(mean);
      cCritic2.getResults(result);
      critic = GetPointer(cCritic2);
     }
   vector<float> var = MathAbs(mean - result) / 2.0f;
   mean += result;
   mean /= 2.0f;
   target = mean;
   for(ulong i = 0; i < log_prob.Size(); i++)
      target[shift + i] = discount * log_prob[i] * LogProbMultiplier;
   target = CAGrad(zeta * (target - var * 2.5f) - result) + result;
   CBufferFloat bTarget;
   bTarget.AssignArray(target);
   critic.TrainMode(false);
   if(!critic.backProp(GetPointer(bTarget), GetPointer(this)) ||
      !backPropGradient(SecondInput, GetPointer(temp)))
     {
      critic.TrainMode(true);
      return false;
     }
//--- update exploration policy
   if(!cActorExploer.feedForward(State, 1, false, SecondInput))
     {
      critic.TrainMode(true);
      return false;
     }
   output = ((CNeuronBaseOCL*)((CLayer*)cActorExploer.layers.At(layers.Total() - 1)).At(0)).getOutput();
   output.AssignArray(Actions);
   output.BufferWrite();
   cActorExploer.GetLogProbs(log_prob);
   target = mean;
   for(ulong i = 0; i < log_prob.Size(); i++)
      target[shift + i] = discount * log_prob[i] * LogProbMultiplier;
   target = CAGrad(zeta * (target + var * 2.0f) - result) + result;
   bTarget.AssignArray(target);
   if(!critic.backProp(GetPointer(bTarget), GetPointer(cActorExploer)) ||
      !cActorExploer.backPropGradient(SecondInput, GetPointer(temp)))
     {
      critic.TrainMode(true);
      return false;
     }
   critic.TrainMode(true);
//---
   if(!!NextState)
     {
      if(iUpdateDelayCount > 0)
        {
         iUpdateDelayCount--;
         return true;
        }
      iUpdateDelayCount = iUpdateDelay;
     }
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
vector<float> CNet_SAC_D_DICE::GetLogProbability(CBufferFloat *Actions)
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
bool CNet_SAC_D_DICE::Save(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name + ".set", (common ? FILE_COMMON : 0) | FILE_BIN | FILE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;
   ulong size = fLambda.Size();
   if(FileWriteInteger(handle, iLatentLayer) < sizeof(iLatentLayer) ||
      FileWriteLong(handle, (long)size) < sizeof(long))
      return false;
   for(ulong i = 0; i < size; i++)
      if(FileWriteFloat(handle, fLambda[i]) < sizeof(fLambda[i]) ||
         FileWriteFloat(handle, fLambda_m[i]) < sizeof(fLambda_m[i]) ||
         FileWriteFloat(handle, fLambda_v[i]) < sizeof(fLambda_v[i]) ||
         FileWriteFloat(handle, fQWeights[i]) < sizeof(fQWeights[i]))
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
bool CNet_SAC_D_DICE::Load(string file_name, bool common = true)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name + ".set", (common ? FILE_COMMON : 0) | FILE_BIN | FILE_READ);
   if(handle == INVALID_HANDLE)
      return false;
   if(FileIsEnding(handle))
      return false;
   iLatentLayer = FileReadInteger(handle);
   if(FileIsEnding(handle))
      return false;
   ulong size = (ulong)FileReadLong(handle);
   fLambda = vector<float>::Zeros(size);
   fLambda_m = fLambda;
   fLambda_v = fLambda;
   fQWeights = vector<float>::Ones(size);
   for(ulong i = 0; i < size; i++)
     {
      if(FileIsEnding(handle))
         return false;
      fLambda[i] = FileReadFloat(handle);
      if(FileIsEnding(handle))
         return false;
      fLambda_m[i] = FileReadFloat(handle);
      if(FileIsEnding(handle))
         return false;
      fLambda_v[i] = FileReadFloat(handle);
      if(FileIsEnding(handle))
         return false;
      fQWeights[i] = FileReadFloat(handle);
     }
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for CAGrad.               |
//+------------------------------------------------------------------+
vector<float> CNet_SAC_D_DICE::CAGrad(vector<float> &grad)
  {
   matrix<float> GG = grad.Outer(grad);
   GG.ReplaceNan(0);
   if(MathAbs(GG).Sum() == 0)
      return grad;
   float scale = MathSqrt(GG.Diag() + 1.0e-4f).Mean();
   GG = GG / MathPow(scale,2);
   vector<float> Gg = GG.Mean(1);
   float gg = Gg.Mean();
   vector<float> w = vector<float>::Zeros(grad.Size());
   float c = MathSqrt(gg + 1.0e-4f) * fCAGrad_C;
   vector<float> w_best = w;
   float obj_best = FLT_MAX;
   vector<float> moment = vector<float>::Zeros(w.Size());
   for(int i = 0; i < iCAGrad_Iters; i++)
     {
      vector<float> ww;
      w.Activation(ww,AF_SOFTMAX);
      float obj = ww.Dot(Gg) + c * MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f);
      if(MathAbs(obj) < obj_best)
        {
         obj_best = MathAbs(obj);
         w_best = w;
        }
      if(i < (iCAGrad_Iters - 1))
        {
         float loss = -obj;
         vector<float> derev = Gg + GG.MatMul(ww) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2) + ww.MatMul(GG) * c / (MathSqrt(ww.MatMul(GG).Dot(ww) + 1.0e-4f) * 2);
         vector<float> delta = derev * loss;
         ulong size = delta.Size();
         matrix<float> ident = matrix<float>::Identity(size, size);
         vector<float> ones = vector<float>::Ones(size);
         matrix<float> sm_der = ones.Outer(ww);
         sm_der = sm_der.Transpose() * (ident - sm_der);
         delta = sm_der.MatMul(delta);
         if(delta.Ptp() != 0)
            delta = delta / delta.Ptp();
         moment = delta * 0.8f + moment * 0.5f;
         w += moment;
         if(w.Ptp() != 0)
            w = w / w.Ptp();
        }
     }
   w_best.Activation(w,AF_SOFTMAX);
   float gw_norm = MathSqrt(w.MatMul(GG).Dot(w) + 1.0e-4f);
   float lmbda = c / (gw_norm + 1.0e-4f);
   vector<float> result = ((w * lmbda + 1.0f / (float)grad.Size()) * grad) / (1 + MathPow(fCAGrad_C,2));
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements TargetsUpdate.                                      |
//+------------------------------------------------------------------+
bool CNet_SAC_D_DICE::TargetsUpdate(float tau)
  {
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

