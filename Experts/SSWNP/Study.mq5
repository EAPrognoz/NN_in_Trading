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
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 StateEncoder;
CNet                 Encoder;
CNet                 Goal;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bGoal;
CBufferFloat         bGradient;
CBufferFloat         bLastEncoder;
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
      Print("Cann't load Encoder model");
      return INIT_FAILED;
     }
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !Goal.Load(FileName + "Goal.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *goal = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, goal, encoder))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !StateEncoder.Create(encoder) || !Goal.Create(goal))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete goal;
      delete encoder;
      //---
     }
//---
   OpenCL = Actor.GetOpenCL();
   StateEncoder.SetOpenCL(OpenCL);
   Encoder.SetOpenCL(OpenCL);
   Goal.SetOpenCL(OpenCL);
   Encoder.TrainMode(false);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   Encoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Encoder does not match the embedding size (%d <> %d)", EmbeddingSize, Result.Total());
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
   StateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("Input size of State Encoder doesn't match Bottleneck (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   StateEncoder.getResults(Result);
   int latent_state = Result.Total();
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Actor doesn't match output State Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Goal.GetLayerOutput(0, Result);
   latent_state = Result.Total();
   Encoder.getResults(Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Goal doesn't match output Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Goal.getResults(Result);
   if(Result.Total() != NRewards)
     {
      PrintFormat("The scope of Goal doesn't match rewards count (%d <> %d)", Result.Total(), NRewards);
      return INIT_FAILED;
     }
//---
   if(!bLastEncoder.BufferInit(EmbeddingSize, 0) ||
      !bGradient.BufferInit(MathMax(EmbeddingSize, AccountDescr), 0) ||
      !bLastEncoder.BufferCreate(OpenCL) ||
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
      StateEncoder.Save(FileName + "StEnc.nnw", 0, 0, 0, TimeCurrent(), true);
      Goal.Save(FileName + "Goal.nnw", 0, 0, 0, TimeCurrent(), true);
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
   vector<float> result, target;
   matrix<float> targets;
   STD_Actor = vector<float>::Zeros(NActions);
   STD_Goal = vector<float>::Zeros(NRewards);
   int std_count = 0;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 50;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
      Encoder.Clear();
      bLastEncoder.BufferInit(EmbeddingSize, 0);
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      for(int i = state; i < end; i++)
        {
         bState.AssignArray(Buffer[tr].States[i].state);
         //---
         if(!bLastEncoder.BufferWrite() ||
            !Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bLastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
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
         double time = (double)Buffer[tr].States[i].account[7];
         double x = time / (double)(D'2024.01.01' - D'2023.01.01');
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_MN1);
         bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_W1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_D1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(bAccount.GetIndex() >= 0)
            bAccount.BufferWrite();
         //--- State embedding
         if(!StateEncoder.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat*)GetPointer(bAccount)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         targets = matrix<float>::Zeros(PrecoderBars, NRewards);
         result.Assign(Buffer[tr].States[i + 1].rewards);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target = result;
            result.Assign(Buffer[tr].States[i + t + 2].rewards);
            target = target - result * DiscFactor;
            targets.Row(target, t);
           }
         for(int t = 1; t < PrecoderBars; t++)
           {
            target = targets.Row(t - 1) + targets.Row(t) * MathPow(DiscFactor, t);
            targets.Row(target, t);
           }
         result = targets.Sum(1);
         ulong row = result.ArgMax();
         target = targets.Row(row);
         bGoal.AssignArray(target);
         //--- Actor
         if(!Actor.feedForward((CNet *)GetPointer(StateEncoder), -1, (CBufferFloat*)GetPointer(bGoal)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         target.Assign(Buffer[tr].States[i].action);
         target.Clip(0, 1);
         Actor.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STD_Actor = MathSqrt((MathPow(STD_Actor, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         check = MathAbs(error) - STD_Actor * STE_Multiplier;
         if(check.Max() > 0)
           {
            Result.AssignArray(CAGrad(error) + result);
            if(!Actor.backProp(Result, (CBufferFloat *)GetPointer(bGoal), (CBufferFloat *)GetPointer(bGradient)) ||
               !StateEncoder.backPropGradient(GetPointer(bAccount), (CBufferFloat *)GetPointer(bGradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //--- Goal
         if(!Goal.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         target=targets.Row(row);
         result = target / (MathAbs(target) + FLT_EPSILON);
         result = MathPow(vector<float>::Full(NRewards, 2), result);
         target = target * result;
         Goal.getResults(result);
         error = target - result;
         std_count = MathMin(std_count, 999);
         STD_Goal = MathSqrt((MathPow(STD_Goal, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         check = MathAbs(error) - STD_Goal * STE_Multiplier;
         if(check.Max() > 0)
           {
            Result.AssignArray(CAGrad(error) + result);
            if(!Goal.backProp(Result, (CBufferFloat *)NULL, (CBufferFloat *)NULL))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         //---
         Encoder.getResults(result);
         bLastEncoder.AssignArray(result);
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Goal", percent, Goal.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Goal", Goal.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

