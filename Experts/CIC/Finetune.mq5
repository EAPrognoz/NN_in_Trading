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
input int                  Iterations     = 100000;
input float                Tau            = 0.001f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 TargetEncoder;
CNet                 Actor;
CNet                 TargetActor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 TargetCritic1;
CNet                 TargetCritic2;
CNet                 Convolution;
CNet                 Scheduler;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         TargetState;
CBufferFloat         TargetAccount;
CBufferFloat         Actions;
CBufferFloat         Gradient;
CBufferFloat         Skills;
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
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true) ||
      !Convolution.Load(FileName + "CNN.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetCritic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetCritic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("No pretrained models found");
      return INIT_FAILED;
     }
   if(!Scheduler.Load(FileName + "Sch.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *descr = new CArrayObj();
      if(!SchedulerDescriptions(descr) || !Scheduler.Create(descr))
        {
         delete descr;
         return INIT_FAILED;
        }
      delete descr;
     }
//---
   OpenCL = Actor.GetOpenCL();
   Encoder.SetOpenCL(OpenCL);
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   TargetEncoder.SetOpenCL(OpenCL);
   TargetActor.SetOpenCL(OpenCL);
   TargetCritic1.SetOpenCL(OpenCL);
   TargetCritic2.SetOpenCL(OpenCL);
   Scheduler.SetOpenCL(OpenCL);
   Convolution.SetOpenCL(OpenCL);
//---
   Actor.TrainMode(false);
   Encoder.TrainMode(false);
//---
   vector<float> ActorResult;
   Actor.getResults(ActorResult);
   if(ActorResult.Size() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of State Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   vector<float> EncoderResults;
   Actor.GetLayerOutput(0,Result);
   Encoder.getResults(EncoderResults);
   if(Result.Total() != int(EncoderResults.Size()))
     {
      PrintFormat("Input size of Actor doesn't match Encoder outputs (%d <> %d)", Result.Total(), EncoderResults.Size());
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
   TargetCritic1.Save(FileName + "Crt1.nnw", Critic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   TargetCritic2.Save(FileName + "Crt2.nnw", Critic2.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   Scheduler.Save(FileName + "Sch.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Result;
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
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   float loss = 0;
//---
   int total_states = Buffer[0].Total - 1;
   for(int i = 1; i < total_tr; i++)
      total_states += Buffer[i].Total - 1;
   vector<float> temp;
   Convolution.getResults(temp);
   matrix<float> state_embedding = matrix<float>::Zeros(total_states,temp.Size());
   matrix<float> rewards = matrix<float>::Zeros(total_states,NRewards);
   int state = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      for(int st = 0; st < Buffer[tr].Total - 1; st++)
        {
         State.AssignArray(Buffer[tr].States[st].state);
         float PrevBalance = Buffer[tr].States[MathMax(st,0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(st,0)].account[1];
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
         //---
         State.AddArray(Buffer[tr].States[st + 1].state);
         State.Add((Buffer[tr].States[st + 1].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[1] / PrevBalance);
         State.Add((Buffer[tr].States[st + 1].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr].States[st + 1].account[2]);
         State.Add(Buffer[tr].States[st + 1].account[3]);
         State.Add(Buffer[tr].States[st + 1].account[4] / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[5] / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[6] / PrevBalance);
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(!Convolution.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         Convolution.getResults(temp);
         state_embedding.Row(temp,state);
         temp.Assign(Buffer[tr].States[st].rewards);
         for(ulong r = 0; r < temp.Size(); r++)
            temp[r] -= Buffer[tr].States[st + 1].rewards[r] * DiscFactor;
         rewards.Row(temp,state);
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
      state_embedding.Reshape(state,state_embedding.Cols());
      rewards.Reshape(state,NRewards);
      total_states = state;
     }
//---
   vector<float> reward, rewards1, rewards2, target_reward;
   int bar = (HistoryBars - 1) * BarDescr;
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      if(i < 0)
        {
         iter--;
         continue;
        }
      reward = vector<float>::Zeros(NRewards);
      rewards1 = reward;
      rewards2 = reward;
      target_reward = reward;
      //--- State
      State.AssignArray(Buffer[tr].States[i].state);
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      if(PrevBalance == 0.0f || PrevEquity == 0.0f)
         continue;
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
      //--- Encoder State
      if(!Encoder.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Skills
      if(!Scheduler.feedForward(GetPointer(Encoder), -1, NULL,-1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Actor
      if(!Actor.feedForward(GetPointer(Encoder), -1, GetPointer(Scheduler),-1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Next State
      TargetState.AssignArray(Buffer[tr].States[i + 1].state);
      double cl_op = Buffer[tr].States[i + 1].state[bar];
      // Implements SymbolInfoDouble.
      double prof_1l = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT) * cl_op /
                       SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      Actor.getResults(Result);
      vector<float> forecast = ForecastAccount(Buffer[tr].States[i].account,Result,prof_1l,Buffer[tr].States[i + 1].account[7]);
      TargetAccount.AssignArray(forecast);
      if(TargetAccount.GetIndex() >= 0 && !TargetAccount.BufferWrite())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      if(!TargetEncoder.feedForward(GetPointer(TargetState), 1, false, GetPointer(TargetAccount)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Target
      if(!TargetActor.feedForward(GetPointer(TargetEncoder), -1, GetPointer(Scheduler),-1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(!TargetCritic1.feedForward(GetPointer(TargetActor), LatentLayer, GetPointer(TargetActor)) ||
         !TargetCritic2.feedForward(GetPointer(TargetActor), LatentLayer, GetPointer(TargetActor)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      TargetCritic1.getResults(rewards1);
      TargetCritic2.getResults(rewards2);
      if(rewards1.Sum() <= rewards2.Sum())
         target_reward = rewards1;
      else
         target_reward = rewards2;
      target_reward *= DiscFactor;
      State.AddArray(GetPointer(TargetState));
      State.AddArray(GetPointer(TargetAccount));
      if(!Convolution.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Convolution.getResults(rewards1);
      reward[0] += KNNReward(7,rewards1,state_embedding,rewards);
      reward += target_reward;
      Result.AssignArray(reward);
      //---
      if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor),-1) ||
         !Critic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor),-1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Critic1.getResults(rewards1);
      Critic2.getResults(rewards2);
      if(rewards1.Sum() <= rewards2.Sum())
        {
         loss = (loss * MathMin(iter,999) + (reward - rewards1).Sum()) / MathMin(iter + 1,1000);
         if(!Critic1.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Scheduler),-1,-1) ||
            !Scheduler.backPropGradient() ||
            !Critic2.backProp(Result, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
        }
      else
        {
         loss = (loss * MathMin(iter,999) + (reward - rewards2).Sum()) / MathMin(iter + 1,1000);
         if(!Critic2.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Scheduler),-1,-1) ||
            !Scheduler.backPropGradient() ||
            !Critic1.backProp(Result, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
        }
      //--- Update Target Nets
      TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
      TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Critic1", iter * 100.0 / (double)(Iterations), Critic1.getRecentAverageError());
         str += StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Critic2", iter * 100.0 / (double)(Iterations), Critic2.getRecentAverageError());
         str += StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Scheduler", iter * 100.0 / (double)(Iterations), loss);
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Critic1", Critic1.getRecentAverageError());
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Critic2", Critic2.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Scheduler", loss);
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements KNNReward.                                          |
//+------------------------------------------------------------------+
float KNNReward(ulong k, vector<float> &embedding, matrix<float> &state_embedding, matrix<float> &rewards)
  {
   if(embedding.Size() != state_embedding.Cols())
     {
      PrintFormat("%s -> %d Inconsistent embedding size", __FUNCTION__, __LINE__);
      return (0);
     }
//---
   ulong size = embedding.Size();
   ulong states = state_embedding.Rows();
   matrix<float> temp = matrix<float>::Zeros(states,size);
//---
   for(ulong i = 0; i < size; i++)
      temp.Col(MathPow(state_embedding.Col(i) - embedding[i],2.0f),i);
   vector<float> dist = MathSqrt(temp.Sum(1));
   matrix<float> min_dist = matrix<float>::Zeros(k,NRewards + 1);
   for(ulong i = 0; i < k; i++)
     {
      ulong pos = dist.ArgMin();
      min_dist[i,0] = dist[pos];
      dist[pos] = FLT_MAX;
      for(int c = 0; c < NRewards; c++)
         min_dist[i,c + 1] = rewards[pos,c];
     }
//---
   min_dist.Col(vector<float>::Zeros(k),0);
//---
   float result = min_dist.Sum() / (k * NRewards);
//---
   return (result);
  }
//+------------------------------------------------------------------+

