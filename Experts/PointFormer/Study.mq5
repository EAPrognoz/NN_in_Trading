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
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 Actor;
CNet                 Critic;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         bAccount;
CBufferFloat         bGradient;
CBufferFloat         bActions;
CBufferFloat         *Result;
vector<float>        check;
vector<float>        STD_Actor;
vector<float>        STD_Goal;
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
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Encoder");
      CArrayObj *encoder = new CArrayObj();
      if(!CreateEncoderDescriptions(encoder))
        {
         delete encoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder))
        {
         delete encoder;
         return INIT_FAILED;
        }
      delete encoder;
     }
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Create new models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      if(!CreateDescriptions(actor, critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) ||
         !Critic.Create(critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
     }
//---
   Encoder.TrainMode(true);
   Actor.TrainMode(true);
   Critic.TrainMode(true);
   OpenCL = Actor.GetOpenCL();
   Encoder.SetOpenCL(OpenCL);
   Critic.SetOpenCL(OpenCL);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   if(!bGradient.BufferInit(MathMax(AccountDescr, NActions), 0) ||
      !bGradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
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
      Encoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      Critic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
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
//| Train function                                                 |
//+------------------------------------------------------------------+
//| | T ra in implementation.                                        |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target, state;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - NForecast));
      if(i <= 0)
        {
         iter --;
         continue;
        }
      state.Assign(Buffer[tr].States[i].state);
      if(MathAbs(state).Sum() == 0)
        {
         iter --;
         continue;
        }
      bState.AssignArray(state);
      //---
      bTime.Clear();
      double time = (double)Buffer[tr].States[i].account[7];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      bTime.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      bTime.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_W1);
      bTime.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_D1);
      bTime.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      if(bTime.GetIndex() >= 0)
         bTime.BufferWrite();
      //--- State Encoder
      if(!Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Critic
      bActions.AssignArray(Buffer[tr].States[i].action);
      if(bActions.GetIndex() >= 0)
         bActions.BufferWrite();
      if(!Critic.feedForward((CBufferFloat*)GetPointer(bActions), 1, false, GetPointer(Encoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      result.Assign(Buffer[tr].States[i + 1].rewards);
      target.Assign(Buffer[tr].States[i + 2].rewards);
      result = result - target * DiscFactor;
      Result.AssignArray(result);
      Critic.TrainMode(true);
      if(!Critic.backProp(Result, (CNet *)GetPointer(Encoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Policy
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      bAccount.Clear();
      bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      bAccount.Add(Buffer[tr].States[i].account[2]);
      bAccount.Add(Buffer[tr].States[i].account[3]);
      bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      bAccount.AddArray(GetPointer(bTime));
      if(bAccount.GetIndex() >= 0)
         bAccount.BufferWrite();
      //--- Actor
      if(!Actor.feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, GetPointer(Encoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      Critic.TrainMode(false);
      if(!Critic.feedForward((CNet *)GetPointer(Actor), -1, (CNet*)GetPointer(Encoder), LatentLayer))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(Buffer[tr].States[0].rewards[0] > 0)
         if(!Actor.backProp(GetPointer(bActions), GetPointer(Encoder), LatentLayer) ||
            !Encoder.backPropGradient((CBufferFloat*)GetPointer(bState)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
      Critic.getResults(Result);
      for(int c = 0; c < Result.Total(); c++)
        {
         float value = Result.At(c);
         if(value >= 0)
            Result.Update(c, value * 1.01f);
         else
            Result.Update(c, value * 0.99f);
        }
      if(!Critic.backProp(Result, (CNet *)GetPointer(Encoder), LatentLayer) ||
         !Actor.backPropGradient((CNet *)GetPointer(Encoder), LatentLayer, -1,false) ||
         !Encoder.backPropGradient((CBufferFloat*)GetPointer(bState)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Critic", percent, Critic.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", Critic.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

