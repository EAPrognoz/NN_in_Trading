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
input int                  Iterations     = 1000000;
input int                  UpdatePolicy   = 3;
input int                  UpdateTargets  = 100;
input float                Tau            = 0.01f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 TargetActor;
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
      !TargetActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
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
      if(!TargetActor.Create(actor) || !TargetCritic1.Create(critic) || !TargetCritic2.Create(critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      //---
      TargetActor.WeightsUpdate(GetPointer(Actor), 1.0f);
      TargetCritic1.WeightsUpdate(GetPointer(Critic1), 1.0f);
      TargetCritic2.WeightsUpdate(GetPointer(Critic2), 1.0f);
     }
//---
   OpenCL = Actor.GetOpenCL();
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   TargetActor.SetOpenCL(OpenCL);
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
   TargetActor.WeightsUpdate(GetPointer(Actor), Tau);
   TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
   TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
   TargetActor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
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
      //---
      if(Account.GetIndex() >= 0)
         Account.BufferWrite();
      if(!TargetActor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //---
      if(!TargetCritic1.feedForward(GetPointer(TargetActor), LatentLayer, GetPointer(TargetActor)) ||
         !TargetCritic2.feedForward(GetPointer(TargetActor), LatentLayer, GetPointer(TargetActor)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      TargetCritic1.getResults(Result);
      float reward = Result[0];
      TargetCritic2.getResults(Result);
      reward = DiscFactor * MathMin(reward, Result[0]) + (Buffer[tr].Revards[i] - Buffer[tr].Revards[i + 1]);
      //--- Q-function study
      State.AssignArray(Buffer[tr].States[i].state);
      PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      Account.Clear();
      Account.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      Account.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      Account.Add(Buffer[tr].States[i].account[2]);
      Account.Add(Buffer[tr].States[i].account[3]);
      Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      //---
      if(Account.GetIndex() >= 0)
         Account.BufferWrite();
      //---
      if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      Actor.GetLayerOutput(LatentLayer,Result);
      Actions.AssignArray(Buffer[tr].States[i].action);
      if(Actions.GetIndex()>=0)
        Actions.BufferWrite();
      //---
      if(!Critic1.feedForward(Result,1,false, GetPointer(Actions)) ||
         !Critic2.feedForward(Result,1,false, GetPointer(Actions)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      Result.Clear();
      Result.Add(reward);
      if(!Critic1.backProp(Result, GetPointer(Actions), GetPointer(Gradient)) ||
         !Critic2.backProp(Result, GetPointer(Actions), GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //--- Policy study
      if(iter > 0 && (iter % UpdatePolicy) == 0)
        {
         tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
         i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
         State.AssignArray(Buffer[tr].States[i].state);
         PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         Account.Clear();
         Account.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
         Account.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
         Account.Add(Buffer[tr].States[i].account[2]);
         Account.Add(Buffer[tr].States[i].account[3]);
         Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
         //---
         if(Account.GetIndex() >= 0)
            Account.BufferWrite();
         //---
         if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         //---
         if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         Critic1.getResults(Result);
         float forecast = Result[0];
         Result.Update(0, (forecast > 0 ? forecast + PoliticAdjust : PoliticAdjust));
         Critic1.TrainMode(false);
         if(!Critic1.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
           {
            Critic1.TrainMode(true);
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         Critic1.TrainMode(true);
        }
      //--- Update Target Nets
      if(iter > 0 && (iter % UpdateTargets) == 0)
        {
         TargetActor.WeightsUpdate(GetPointer(Actor), Tau);
         TargetCritic1.WeightsUpdate(GetPointer(Critic1), Tau);
         TargetCritic2.WeightsUpdate(GetPointer(Critic2), Tau);
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

