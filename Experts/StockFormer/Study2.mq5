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
CNet                 Actor;
CNet                 Critic;
CNet                 RelateEncoder;
CNet                 ShortEncoder;
CNet                 ShortDecoder;
CNet                 LongEncoder;
CNet                 LongDecoder;
CNet                 LongShort;
CNet                 PredictRelate;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bActions;
CBufferFloat         bAccount;
CBufferFloat         bGradient;
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
   float temp = 0;
   if(!RelateEncoder.Load(FileName + "RelE.nnw", temp, temp, temp, dtStudied, true) ||
      !ShortEncoder.Load(FileName + "ShE.nnw", temp, temp, temp, dtStudied, true) ||
      !ShortDecoder.Load(FileName + "ShD.nnw", temp, temp, temp, dtStudied, true) ||
      !LongEncoder.Load(FileName + "LnE.nnw", temp, temp, temp, dtStudied, true) ||
      !LongDecoder.Load(FileName + "LnD.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Can't load pretrained Preict&Relate models");
      return INIT_FAILED;
     }
//---
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true) ||
      !LongShort.Load(FileName + "LS.nnw", temp, temp, temp, dtStudied, true) ||
      !PredictRelate.Load(FileName + "PR.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      Print("Create new models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *long_short = new CArrayObj();
      CArrayObj *predict_relate = new CArrayObj();
      if(!CreateDescriptions(actor, critic, long_short, predict_relate))
        {
         delete actor;
         delete critic;
         delete long_short;
         delete predict_relate;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) ||
         !Critic.Create(critic) ||
         !LongShort.Create(long_short) ||
         !PredictRelate.Create(predict_relate))
        {
         delete actor;
         delete critic;
         delete long_short;
         delete predict_relate;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete long_short;
      delete predict_relate;
     }
//---
   Actor.TrainMode(true);
   Critic.TrainMode(true);
   RelateEncoder.TrainMode(false);
   LongEncoder.TrainMode(true);
   LongDecoder.TrainMode(true);
   ShortEncoder.TrainMode(true);
   ShortDecoder.TrainMode(true);
   LongShort.TrainMode(true);
   PredictRelate.TrainMode(true);
//---
   OpenCL = Actor.GetOpenCL();
   Critic.SetOpenCL(OpenCL);
   RelateEncoder.SetOpenCL(OpenCL);
   LongEncoder.SetOpenCL(OpenCL);
   LongDecoder.SetOpenCL(OpenCL);
   ShortEncoder.SetOpenCL(OpenCL);
   ShortDecoder.SetOpenCL(OpenCL);
   LongShort.SetOpenCL(OpenCL);
   PredictRelate.SetOpenCL(OpenCL);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   RelateEncoder.GetLayerOutput(0, Result);
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
      Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      Critic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
      LongShort.Save(FileName + "LS.nnw", 0, 0, 0, TimeCurrent(), true);
      PredictRelate.Save(FileName + "PR.nnw", 0, 0, 0, TimeCurrent(), true);
      ShortEncoder.Save(FileName + "ShE.nnw", 0, 0, 0, TimeCurrent(), true);
      ShortDecoder.Save(FileName + "ShD.nnw", 0, 0, 0, TimeCurrent(), true);
      LongEncoder.Save(FileName + "LnE.nnw", 0, 0, 0, TimeCurrent(), true);
      LongDecoder.Save(FileName + "LnD.nnw", 0, 0, 0, TimeCurrent(), true);
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
      if(!state.Assign(Buffer[tr].States[i].state) ||
         MathAbs(state).Sum() == 0 ||
         !bState.AssignArray(state))
        {
         iter --;
         continue;
        }
      //--- Account
      bAccount.Clear();
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      bAccount.Add(Buffer[tr].States[i].account[2]);
      bAccount.Add(Buffer[tr].States[i].account[3]);
      bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      //---
      double time = (double)Buffer[tr].States[i].account[7];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_W1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_D1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      if(!!bAccount.GetOpenCL())
        {
         if(!bAccount.BufferWrite())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
        }
      //--- Generate Latent state
      if(!RelateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(ShortEncoder)) ||
         !LongEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !LongDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(LongEncoder)) ||
         !LongShort.feedForward(GetPointer(LongDecoder), -1, GetPointer(ShortDecoder), -1) ||
         !PredictRelate.feedForward(GetPointer(LongShort), -1, GetPointer(RelateEncoder), -1)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Critic
      target.Assign(Buffer[tr].States[i].action);
      target.Clip(0, 1);
      bActions.AssignArray(target);
      if(!!bActions.GetOpenCL())
         if(!bActions.BufferWrite())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
      Critic.TrainMode(true);
      if(!Critic.feedForward(GetPointer(PredictRelate), -1, (CBufferFloat*)GetPointer(bActions)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      result.Assign(Buffer[tr].States[i + 1].rewards);
      target.Assign(Buffer[tr].States[i + 2].rewards);
      result = result - target * DiscFactor;
      Result.AssignArray(result);
      if(!Critic.backProp(Result, (CBufferFloat *)GetPointer(bActions), (CBufferFloat *)GetPointer(bGradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      //--- Actor Policy
      if(!Actor.feedForward(GetPointer(PredictRelate), -1, (CBufferFloat*)GetPointer(bAccount)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      Critic.TrainMode(false);
      if(!Critic.feedForward(GetPointer(PredictRelate), -1, (CNet*)GetPointer(Actor), -1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      if(result.Sum() >= 0)
         if(!Actor.backProp(GetPointer(bActions), (CBufferFloat*)GetPointer(bAccount), GetPointer(bGradient)) ||
            !PredictRelate.backPropGradient(GetPointer(RelateEncoder), -1, -1, false) ||
            !LongShort.backPropGradient(GetPointer(ShortDecoder), -1, -1, false) ||
            !ShortDecoder.backPropGradient((CNet *)GetPointer(ShortEncoder), -1, -1, false) ||
            !ShortEncoder.backPropGradient((CBufferFloat*)NULL) ||
            !LongDecoder.backPropGradient((CNet *)GetPointer(LongEncoder), -1, -1, false) ||
            !LongEncoder.backPropGradient((CBufferFloat*)NULL)
           )
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
      if(!Critic.backProp(Result, (CNet *)GetPointer(Actor), LatentLayer) ||
         !Actor.backPropGradient((CBufferFloat*)GetPointer(bAccount), GetPointer(bGradient)) ||
         !PredictRelate.backPropGradient(GetPointer(RelateEncoder), -1, -1, false) ||
         !LongShort.backPropGradient(GetPointer(ShortDecoder), -1, -1, false) ||
         !ShortDecoder.backPropGradient((CNet *)GetPointer(ShortEncoder), -1, -1, false) ||
         !ShortEncoder.backPropGradient((CBufferFloat*)NULL) ||
         !LongDecoder.backPropGradient((CNet *)GetPointer(LongEncoder), -1, -1, false) ||
         !LongEncoder.backPropGradient((CBufferFloat*)NULL)
        )
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

