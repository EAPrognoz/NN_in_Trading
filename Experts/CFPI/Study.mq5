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
input int                  BatchSize      = 256;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 StateEncoder;
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
   if(!StateEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Cann't load Critic models");
      return INIT_FAILED;
     }
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      if(!CreateDescriptions(actor, critic, critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
     }
//---
   OpenCL = Actor.GetOpenCL();
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   StateEncoder.SetOpenCL(OpenCL);
//---
   StateEncoder.TrainMode(false);
   Critic1.TrainMode(false);
   Critic2.TrainMode(false);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   StateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of State Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   StateEncoder.getResults(Result);
   int latent_state = Result.Total();
   Critic1.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Critic1 doesn't match output State Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Critic2.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Critic2 doesn't match output State Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Actor doesn't match output State Encoder (%d <> %d)", Result.Total(), latent_state);
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
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
      Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
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
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> rewards, rewards1, rewards2, target_reward;
   vector<float> action, action_beta;
   float Improve = 0;
   int bar = (HistoryBars - 1) * BarDescr;
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      matrix<float> mBatch = matrix<float>::Zeros(BatchSize, 4);
      for(int b = 0; b < BatchSize; b++)
        {
         int tr = SampleTrajectory(probability);
         int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
         if(i < 0)
           {
            b--;
            continue;
           }
         //--- State
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
         double time = (double)Buffer[tr].States[i].account[7];
         double x = time / (double)(D'2024.01.01' - D'2023.01.01');
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_MN1);
         Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_W1);
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_D1);
         Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(Account.GetIndex() >= 0)
            Account.BufferWrite();
         //--- State embedding
         if(!StateEncoder.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //--- Action
         if(!Actor.feedForward(GetPointer(StateEncoder), -1, NULL, 1))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //--- Cost
         if(!Critic1.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actor)) ||
            !Critic2.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         Critic1.getResults(rewards1);
         Critic2.getResults(rewards2);
         Actor.getResults(action);
         action_beta.Assign(Buffer[tr].States[i].action);
         rewards.Assign(Buffer[tr].States[i + 1].rewards);
         target_reward.Assign(Buffer[tr].States[i + 2].rewards);
         //--- Collect
         mBatch[b, 0] = float(tr);
         mBatch[b, 1] = float(i);
         mBatch[b, 2] = MathMin(rewards1.Sum(), rewards2.Sum()) - (rewards - target_reward * DiscFactor).Sum();
         mBatch[b, 3] = MathSqrt(MathPow(action - action_beta, 2).Sum());
        }
      //--- Select
      rewards = mBatch.Col(2);
      action = mBatch.Col(3);
      float quant = action.Quantile(0.68);
      vector<float> weights = action - quant - FLT_EPSILON;
      weights.Clip(weights.Min(), 0);
      weights = weights / weights;
      weights.ReplaceNan(0);
      weights = MathAbs(rewards) * weights / action;
      ulong pos = weights.ArgMax();
      int sign = (rewards[pos] >= 0 ? 1 : -1);
      Improve = (Improve * iter + weights[pos]) / (iter + 1);
      int tr = int(mBatch[pos, 0]);
      int i = int(mBatch[pos, 1]);
      //--- Policy study
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
      double time = (double)Buffer[tr].States[i].account[7];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_W1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_D1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      //--- State
      if(Account.GetIndex() >= 0)
         Account.BufferWrite();
      if(!StateEncoder.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Action
      if(!Actor.feedForward(GetPointer(StateEncoder), -1, NULL, 1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Cost
      if(!Critic1.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actor)) ||
         !Critic2.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actor)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      Critic1.getResults(rewards1);
      Critic2.getResults(rewards2);
      //---
      rewards.Assign(Buffer[tr].States[i + 1].rewards);
      target_reward.Assign(Buffer[tr].States[i + 2].rewards);
      rewards = rewards - target_reward * DiscFactor;
      CNet *critic = NULL;
      if(rewards1.Sum() <= rewards2.Sum())
        {
         Result.AssignArray(CAGrad((rewards1 - rewards)*sign) + rewards1);
         critic = GetPointer(Critic1);
        }
      else
        {
         Result.AssignArray(CAGrad((rewards2 - rewards)*sign) + rewards2);
         critic = GetPointer(Critic2);
        }
      if(!critic.backProp(Result, GetPointer(Actor), -1) ||
         !Actor.backPropGradient((CBufferFloat *)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> %15.8f\n", "Mean Improvement", iter * 100.0 / (double)(Iterations), Improve);
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__,  "Mean Improvement", Improve);
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

