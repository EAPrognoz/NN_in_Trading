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
input float                Tau            = 0.01f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 TargetCritic1;
CNet                 TargetCritic2;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         Actions;
CBufferFloat         Gradient;
CBufferFloat         *Result;
vector<float>        SchedulerResult;
vector<float>        check;
vector<float>        ActorResult;
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
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      if(!CreateDescriptions(actor, critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !Critic1.Create(critic) || !Critic2.Create(critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!TargetCritic1.Create(critic) || !TargetCritic2.Create(critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      //---
      TargetCritic1.WeightsUpdate(GetPointer(Critic1), 1.0f);
      TargetCritic2.WeightsUpdate(GetPointer(Critic2), 1.0f);
     }
//---
   OpenCL = Actor.GetOpenCL();
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   TargetCritic1.SetOpenCL(OpenCL);
   TargetCritic2.SetOpenCL(OpenCL);
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
   TargetCritic1.Save(FileName + "Crt1.nnw", TargetCritic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   TargetCritic2.Save(FileName + "Crt2.nnw", TargetCritic2.getRecentAverageError(), 0, 0, TimeCurrent(), true);
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
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      //--- Target
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
         ExpertRemove();
         break;
        }
      //---
      if(!TargetCritic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)) ||
         !TargetCritic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      vector<float> log_prob;
      if(!Actor.GetLogProbs(log_prob))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      TargetCritic1.getResults(Result);
      float reward = Result[0];
      TargetCritic2.getResults(Result);
      reward = Buffer[tr].Revards[i] + DiscFactor * (MathMin(reward, Result[0]) - Buffer[tr].Revards[i + 1]);
      float log_prob_forecast = DiscFactor * LogProbMultiplier * log_prob.Sum() / (float)PrevBalance;
      //--- Q-function study
      State.AssignArray(Buffer[tr].States[i].state);
      PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      Account.Update(0, (Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      Account.Update(1, Buffer[tr].States[i].account[1] / PrevBalance);
      Account.Update(2, (Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      Account.Update(3, Buffer[tr].States[i].account[2]);
      Account.Update(4, Buffer[tr].States[i].account[3]);
      Account.Update(5, Buffer[tr].States[i].account[4] / PrevBalance);
      Account.Update(6, Buffer[tr].States[i].account[5] / PrevBalance);
      Account.Update(7, Buffer[tr].States[i].account[6] / PrevBalance);
      x = (double)Buffer[tr].States[i].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_MN1);
      Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_W1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_D1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
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
      if((iter % 2) == 0)
        {
         if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)) ||
            !Critic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actions)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         Critic1.getResults(Result);
         Actor.GetLogProbs(log_prob);
         Result.Update(0, reward + LogProbMultiplier * log_prob.Sum() / (float)PrevBalance + log_prob_forecast);
         Critic1.TrainMode(false);
         if(!Critic1.backProp(Result, GetPointer(Actor)) ||
            !Critic1.AlphasGradient(GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient), LatentLayer) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Critic1.TrainMode(true);
            break;
           }
         Critic1.TrainMode(true);
         Result.Update(0, reward);
         if(!Critic2.backProp(Result, GetPointer(Actions), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //--- Update Target Nets
         TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
        }
      else
        {
         if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)) ||
            !Critic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actions)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         Critic2.getResults(Result);
         Actor.GetLogProbs(log_prob);
         Result.Update(0, reward + LogProbMultiplier * log_prob.Sum() / (float)PrevBalance + log_prob_forecast);
         Critic2.TrainMode(false);
         if(!Critic2.backProp(Result, GetPointer(Actor)) ||
            !Critic2.AlphasGradient(GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient), LatentLayer) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Critic2.TrainMode(true);
            break;
           }
         Critic2.TrainMode(true);
         Result.Update(0, reward);
         if(!Critic1.backProp(Result, GetPointer(Actions), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         //--- Update Target Nets
         TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic1", iter * 100.0 / (double)(Iterations), Critic1.getRecentAverageError());
         str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic2", iter * 100.0 / (double)(Iterations), Critic2.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic1", Critic1.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic2", Critic2.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

