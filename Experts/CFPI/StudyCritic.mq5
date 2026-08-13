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
input int                  Iterations     = 1e6;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 StateEncoder;
CNet                 Critic1;
CNet                 Critic2;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         Actions;
CBufferFloat         Gradient;
CBufferFloat         *Result;
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
      Print("Init new models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, critic, encoder))
        {
         delete actor;
         delete critic;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Critic1.Create(critic) || !Critic2.Create(critic) ||
         !StateEncoder.Create(encoder))
        {
         delete actor;
         delete critic;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete encoder;
      //---
     }
//---
   OpenCL = Critic1.GetOpenCL();
   Critic2.SetOpenCL(OpenCL);
   StateEncoder.SetOpenCL(OpenCL);
//---
   StateEncoder.getResults(Result);
   if(Result.Total() != LatentCount)
     {
      PrintFormat("The scope of the State Encoder does not match the latent size count (%d <> %d)", LatentCount, Result.Total());
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
   Critic1.GetLayerOutput(0, Result);
   if(Result.Total() != LatentCount)
     {
      PrintFormat("Input size of Critic1 doesn't match State Encoder output (%d <> %d)", Result.Total(), LatentCount);
      return INIT_FAILED;
     }
//---
   Critic2.GetLayerOutput(0, Result);
   if(Result.Total() != LatentCount)
     {
      PrintFormat("Input size of Critic2 doesn't match State Encoder output (%d <> %d)", Result.Total(), LatentCount);
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
     {
      StateEncoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      Critic1.Save(FileName + "Crt1.nnw", Critic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
      Critic2.Save(FileName + "Crt2.nnw", Critic2.getRecentAverageError(), 0, 0, TimeCurrent(), true);
     }
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
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3));
      if(i < 0)
        {
         iter--;
         continue;
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
      //---
      if(!StateEncoder.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      Actions.AssignArray(Buffer[tr].States[i].action);
      if(Actions.GetIndex() >= 0)
         Actions.BufferWrite();
      //---
      if(!Critic1.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actions)) ||
         !Critic2.feedForward(GetPointer(StateEncoder), -1, GetPointer(Actions)))
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
      Result.AssignArray(CAGrad(rewards - rewards1) + rewards1);
      if(!Critic1.backProp(Result, GetPointer(Actions), GetPointer(Gradient)) ||
         !StateEncoder.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Result.AssignArray(CAGrad(rewards - rewards2) + rewards2);
      if(!Critic2.backProp(Result, GetPointer(Actions), GetPointer(Gradient)) ||
         !StateEncoder.backPropGradient(GetPointer(Account), GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
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

