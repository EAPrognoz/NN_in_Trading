//+------------------------------------------------------------------+
//| Study.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define Study
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 10000;
input float                Tau            = 0.01f;
input float                MaxErrorActorStudy   = 0.15f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 TargetCritic1;
CNet                 TargetCritic2;
CNet                 Convolution;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         Actions;
CBufferFloat         Gradient;
CBufferFloat         *Result;
vector<float>        check;
int                  StartTargetIter;
//---
COpenCLMy           *OpenCL;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
//--- load models
   float temp;
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetCritic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetCritic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *convolution = new CArrayObj();
      if(!CreateDescriptions(actor, critic, convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !Critic1.Create(critic) || !Critic2.Create(critic) ||
         !Convolution.Create(convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      if(!TargetCritic1.Create(critic) || !TargetCritic2.Create(critic))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete convolution;
      //---
      TargetCritic1.WeightsUpdate(GetPointer(Critic1), 1.0f);
      TargetCritic2.WeightsUpdate(GetPointer(Critic2), 1.0f);
      StartTargetIter = StartTargetIteration;
     }
   else
      StartTargetIter = 0;
//---
   if(!Convolution.Load(FileName + "CNN.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new Encoder model");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *convolution = new CArrayObj();
      if(!CreateDescriptions(actor, critic, convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      if(!Convolution.Create(convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete convolution;
     }
//---
   OpenCL = Actor.GetOpenCL();
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   TargetCritic1.SetOpenCL(OpenCL);
   TargetCritic2.SetOpenCL(OpenCL);
   Convolution.SetOpenCL(OpenCL);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(LatentLayer, Result);
   int latent_state = Result.Total();
   Critic1.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Critic doesn't match latent state Actor (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Gradient.BufferInit(AccountDescr, 0);
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
   TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
   Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
   TargetCritic1.Save(FileName + "Crt1.nnw", Critic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   TargetCritic2.Save(FileName + "Crt2.nnw", Critic2.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   Convolution.Save(FileName + "CNN.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Result;
   delete OpenCL;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//| Train function                                                 |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
//---
   int total_states = Buffer[0].Total;
   for(int i = 1; i < total_tr; i++)
      total_states += Buffer[i].Total;
   vector<float> temp, next;
   Convolution.getResults(temp);
   matrix<float> state_embedding = matrix<float>::Zeros(total_states, temp.Size());
   matrix<float> rewards = matrix<float>::Zeros(total_states, NRewards);
   matrix<float> actions = matrix<float>::Zeros(total_states, NActions);
   int state = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      for(int st = 0; st < Buffer[tr].Total; st++)
        {
         State.AssignArray(Buffer[tr].States[st].state);
         float PrevBalance = Buffer[tr].States[MathMax(st - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(st - 1, 0)].account[1];
         State.Add((Buffer[tr].States[st].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr].States[st].account[1] / PrevBalance);
         State.Add((Buffer[tr].States[st].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr].States[st].account[2]);
         State.Add(Buffer[tr].States[st].account[3]);
         State.Add(Buffer[tr].States[st].account[4] / PrevBalance);
         State.Add(Buffer[tr].States[st].account[5] / PrevBalance);
         State.Add(Buffer[tr].States[st].account[6] / PrevBalance);
         double x = (double)Buffer[tr].States[st].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         State.AddArray(vector<float>::Zeros(NActions));
         if(!Convolution.feedForward(GetPointer(State), 1, false, NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         Convolution.getResults(temp);
         if(!state_embedding.Row(temp, state))
            continue;
         if(!temp.Assign(Buffer[tr].States[st].rewards) ||
            !next.Assign(Buffer[tr].States[st + 1].rewards) ||
            !rewards.Row(temp - next * DiscFactor, state))
            continue;
         if(!temp.Assign(Buffer[tr].States[st].action) ||
            !actions.Row(temp, state))
            continue;
         state++;
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %6.2f%%", "Embedding ", state * 100.0 / (double)(total_states));
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   if(state != total_states)
     {
      rewards.Resize(state, NRewards);
      actions.Resize(state, NActions);
      state_embedding.Reshape(state, state_embedding.Cols());
      total_states = state;
     }
//---
   vector<float> rewards1, rewards2, target_reward;
   STarget target;
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
   int bar = (HistoryBars - 1) * BarDescr;
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      if(i < 0)
        {
         iter--;
         continue;
        }
      target_reward = vector<float>::Zeros(NRewards);
      //--- Target
      if(iter >= StartTargetIter)
        {
         State.AssignArray(Buffer[tr].States[i + 1].state);
         float PrevBalance = Buffer[tr].States[i].account[0];
         float PrevEquity = Buffer[tr].States[i].account[1];
         Account.Clear();
         Account.Add((Buffer[tr].States[i + 1].account[0] - PrevBalance) / PrevBalance);
         Account.Add(Buffer[tr].States[i + 1].account[1] / PrevBalance);
         Account.Add((Buffer[tr].States[i + 1].account[1] - PrevEquity) / PrevEquity);
         Account.Add(Buffer[tr].States[i + 1].account[2]);
         Account.Add(Buffer[tr].States[i + 1].account[3]);
         Account.Add(Buffer[tr].States[i + 1].account[4] / PrevBalance);
         Account.Add(Buffer[tr].States[i + 1].account[5] / PrevBalance);
         Account.Add(Buffer[tr].States[i + 1].account[6] / PrevBalance);
         double x = (double)Buffer[tr].States[i + 1].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_W1);
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_D1);
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         //---
         if(Account.GetIndex() >= 0)
            Account.BufferWrite();
         if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //---
         if(!TargetCritic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)) ||
            !TargetCritic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         TargetCritic1.getResults(rewards1);
         TargetCritic2.getResults(rewards2);
         target_reward.Assign(Buffer[tr].States[i + 1].rewards);
         if(rewards1.Sum() <= rewards2.Sum())
            target_reward = rewards1 - target_reward;
         else
            target_reward = rewards2 - target_reward;
         target_reward *= DiscFactor;
         target_reward[NRewards - 1] = EntropyLatentState(Actor);
        }
      //--- Q-function study
      State.AssignArray(Buffer[tr].States[i].state);
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      Account.Clear();
      Account.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      Account.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      Account.Add(Buffer[tr].States[i].account[2]);
      Account.Add(Buffer[tr].States[i].account[3]);
      Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      double x = (double)Buffer[tr].States[i].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_MN1);
      Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_W1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_D1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      if(Account.GetIndex() >= 0)
         Account.BufferWrite();
      //---
      if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      Actions.AssignArray(Buffer[tr].States[i].action);
      if(Actions.GetIndex() >= 0)
         Actions.BufferWrite();
      //---
      if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actions)) ||
         !Critic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actions)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(!State.AddArray(GetPointer(Account)) || !State.AddArray(vector<float>::Zeros(NActions)) ||
         !Convolution.feedForward(GetPointer(State), 1, false, NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Convolution.getResults(temp);
      target = GetTargets(Quant, temp, state_embedding, rewards, actions);
      //---
      Critic1.getResults(rewards1);
      Result.AssignArray(CAGrad(target.rewards + target_reward - rewards1) + rewards1);
      if(!Critic1.backProp(Result, GetPointer(Actions), GetPointer(Gradient)) ||
         !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Critic2.getResults(rewards2);
      Result.AssignArray(CAGrad(target.rewards + target_reward - rewards2) + rewards2);
      if(!Critic2.backProp(Result, GetPointer(Actions), GetPointer(Gradient)) ||
         !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Policy study
      Actor.getResults(rewards1);
      Result.AssignArray(CAGrad(target.actions - rewards1) + rewards1);
      if(!Actor.backProp(Result, GetPointer(Account), GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      CNet *critic = NULL;
      if(Critic1.getRecentAverageError() <= Critic2.getRecentAverageError())
         critic = GetPointer(Critic1);
      else
         critic = GetPointer(Critic2);
      if(MathAbs(critic.getRecentAverageError()) <= MaxErrorActorStudy)
        {
         if(!critic.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         critic.getResults(rewards1);
         Result.AssignArray(CAGrad(target.rewards + target_reward - rewards1) + rewards1);
         critic.TrainMode(false);
         if(!critic.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            critic.TrainMode(true);
            break;
           }
         critic.TrainMode(true);
        }
      //--- Update Target Nets
      if(iter >= StartTargetIter)
        {
         TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
         TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
        }
      else
        {
         TargetCritic1.WeightsUpdate(GetPointer(Critic1), 1);
         TargetCritic2.WeightsUpdate(GetPointer(Critic2), 1);
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic1", iter * 100.0 / (double)(Iterations), Critic1.getRecentAverageError());
         str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic2", iter * 100.0 / (double)(Iterations), Critic2.getRecentAverageError());
         str += StringFormat("%-14s %5.2f%% -> Error %15.8f\n", "Actor", iter * 100.0 / (double)(Iterations), Actor.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic1", Critic1.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic2", Critic2.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> CAGrad(vector<float> &grad)
  {
   matrix<float> GG = grad.Outer(grad);
   GG.ReplaceNan(0);
   if(MathAbs(GG).Sum() == 0)
      return grad;
   float scale = MathSqrt(GG.Diag() + 1.0e-4f).Mean();
   GG = GG / MathPow(scale, 2);
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
      w.Activation(ww, AF_SOFTMAX);
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
   w_best.Activation(w, AF_SOFTMAX);
   float gw_norm = MathSqrt(w.MatMul(GG).Dot(w) + 1.0e-4f);
   float lmbda = c / (gw_norm + 1.0e-4f);
   vector<float> result = ((w * lmbda + 1.0f / (float)grad.Size()) * grad) / (1 + MathPow(fCAGrad_C, 2));
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | G et Ta rg et s implementation.                                |
//+------------------------------------------------------------------+
STarget GetTargets(double quant, vector<float> &embedding, matrix<float> &state_embedding, matrix<float> &rewards, matrix<float> &actions)
  {
   STarget result;
   if(embedding.Size() != state_embedding.Cols())
     {
      PrintFormat("%s -> %d Inconsistent embedding size", __FUNCTION__, __LINE__);
      return result;
     }
//---
   ulong size = embedding.Size();
   ulong states = state_embedding.Rows();
   ulong k = ulong(states * quant);
   matrix<float> temp = matrix<float>::Zeros(states, size);
   for(ulong i = 0; i < size; i++)
      temp.Col(MathAbs(state_embedding.Col(i) - embedding[i]), i);
   float alpha = temp.Max();
   vector<float> dist = MathLog(MathExp(temp / (-alpha)).Sum(1)) * (-alpha);
   vector<float> min_dist = vector<float>::Zeros(k);
   matrix<float> k_rewards = matrix<float>::Zeros(k, NRewards);
   matrix<float> k_actions = matrix<float>::Zeros(k, NActions);
   matrix<float> k_embedding = matrix<float>::Zeros(k + 1, size);
   matrix<float> U, V;
   vector<float> S;
   float max = dist.Quantile(quant);
   float min = dist.Min();
   for(ulong i = 0, cur = 0; (i < states && cur < k); i++)
     {
      if(max < dist[i])
         continue;
      min_dist[cur] = dist[i];
      k_rewards.Row(rewards.Row(i), cur);
      k_actions.Row(actions.Row(i), cur);
      k_embedding.Row(state_embedding.Row(i), cur);
      cur++;
     }
   k_embedding.Row(embedding, k);
//---
   vector<float> sf;
   (min_dist * (-1)).Activation(sf, AF_SOFTMAX);
   result.rewards = sf.MatMul(k_rewards);
//---
   k_embedding.SVD(U, V, S);
   result.rewards[NRewards - 2] = S.Sum() / (MathSqrt(MathPow(k_embedding, 2.0f).Sum() * MathMax(k + 1, size)));
   result.rewards[NRewards - 1] = EntropyLatentState(Actor);
//---
   vector<float> act_sf;
   alpha = MathAbs(k_rewards).Max();
   dist = MathLog(MathExp(k_rewards / (-alpha)).Sum(1)) * (-alpha);
   dist.Activation(act_sf, AF_SOFTMAX);
   result.actions = act_sf.MatMul(k_actions);
//---
   return result;
  }
//+------------------------------------------------------------------+

