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
input int                  Batch          = 3 * 60;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
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
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new model");
      CArrayObj *actor = new CArrayObj();
      if(!CreateDescriptions(actor))
        {
         delete actor;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor))
        {
         delete actor;
         return INIT_FAILED;
        }
      delete actor;
     }
//---
   Actor.TrainMode(true);
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
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   if(!bGradient.BufferInit(MathMax(AccountDescr, NActions), 0) ||
      !bGradient.BufferCreate(Actor.GetOpenCL()))
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
     }
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
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target, state;
   matrix<float> fstate = matrix<float>::Zeros(1, NForecast * BarDescr);
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter += Batch)
     {
      int tr = SampleTrajectory(probability);
      int start = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - NForecast - Batch));
      if(start <= 0)
        {
         iter -= Batch;
         continue;
        }
      if(!Actor.Clear())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      for(int i = start; i < MathMin(Buffer[tr].Total, start + Batch); i++)
        {
         if(!state.Assign(Buffer[tr].States[i].state) ||
            // Implements MathAbs.
            MathAbs(state).Sum() == 0 ||
            !bState.AssignArray(state))
           {
            iter -= Batch + start - i;
            break;
           }
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
         //--- Previous Action
         Actor.getResults(result);
         //--- Account
         float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         float profit = float(bState[0] / _Point * (result[0] - result[3]));
         bAccount.Clear();
         bAccount.Add((PrevEquity - PrevBalance) / PrevBalance);
         bAccount.Add((PrevEquity + profit) / PrevEquity);
         bAccount.Add(profit / PrevEquity);
         bAccount.Add(MathMax(result[0] - result[3], 0));
         bAccount.Add(MathMax(result[3] - result[0], 0));
         bAccount.Add((bAccount[3] > 0 ? profit / PrevBalance : 0));
         bAccount.Add((bAccount[4] > 0 ? profit / PrevBalance : 0));
         bAccount.Add(0);
         bAccount.AddArray(GetPointer(bTime));
         if(bAccount.GetIndex() >= 0)
            bAccount.BufferWrite();
         //--- Feed Forward
         if(!Actor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, GetPointer(bAccount)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Look for target
         target = vector<float>::Zeros(NActions);
         bActions.AssignArray(target);
         if(!state.Assign(Buffer[tr].States[i + NForecast].state) ||
            !state.Resize(NForecast * BarDescr) ||
            // Implements MathAbs.
            MathAbs(state).Sum() == 0)
           {
            iter -= Batch + start - i;
            break;
           }
         if(!fstate.Resize(1, NForecast * BarDescr) ||
            !fstate.Row(state, 0) ||
            !fstate.Reshape(NForecast, BarDescr))
           {
            iter -= Batch + start - i;
            break;
           }
         for(int j = 0; j < NForecast / 2; j++)
           {
            if(!fstate.SwapRows(j, NForecast - j - 1))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         target = fstate.Col(0).CumSum();
         if(result[0] > result[3])
           {
            float tp = 0;
            float sl = 0;
            float cur_sl = float(-(result[2] > 0 ? result[2] : 1) * MaxSL * Point());
            int pos = 0;
            for(int j = 0; j < NForecast; j++)
              {
               tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
               pos = j;
               if(cur_sl >= target[j] + fstate[j, 2] - fstate[j, 0])
                  break;
               sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
              }
            if(tp > 0)
              {
               sl = float(MathMin(MathAbs(sl) / (MaxSL * Point()), 1));
               tp = float(MathMin(tp / (MaxTP * Point()), 1));
               result[0] = MathMax(result[0] - result[3], 0.011f);
               result[1] = tp;
               result[2] = sl;
               for(int j = 3; j < NActions; j++)
                  result[j] = 0;
               bActions.AssignArray(result);
              }
           }
         else
           {
            if(result[0] < result[3])
              {
               float tp = 0;
               float sl = 0;
               float cur_sl = float((result[5] > 0 ? result[5] : 1) * MaxSL * Point());
               int pos = 0;
               for(int j = 0; j < NForecast; j++)
                 {
                  tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
                  pos = j;
                  if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
                     break;
                  sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
                 }
               if(tp < 0)
                 {
                  sl = float(MathMin(MathAbs(sl) / (MaxSL * Point()), 1));
                  tp = float(MathMin(-tp / (MaxTP * Point()), 1));
                  result[3] = MathMax(result[3] - result[0], 0.011f);
                  result[4] = tp;
                  result[5] = sl;
                  for(int j = 0; j < 3; j++)
                     result[j] = 0;
                  bActions.AssignArray(result);
                 }
              }
            else
              {
               ulong argmin = target.ArgMin();
               ulong argmax = target.ArgMax();
               while(argmax > 0 && argmin > 0)
                 {
                  if(argmax < argmin && target[argmax] > MathAbs(target[argmin]))
                     break;
                  if(argmax > argmin && target[argmax] < MathAbs(target[argmin]))
                     break;
                  target.Resize(MathMin(argmax, argmin));
                  argmin = target.ArgMin();
                  argmax = target.ArgMax();
                 }
               if(argmin == 0 || (argmax < argmin && argmax > 0))
                 {
                  float tp = 0;
                  float sl = 0;
                  float cur_sl = - float(MaxSL * Point());
                  ulong pos = 0;
                  for(ulong j = 0; j < argmax; j++)
                    {
                     tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
                     pos = j;
                     if(cur_sl >= target[j] + fstate[j, 2] - fstate[j, 0])
                        break;
                     sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
                    }
                  if(tp > 0)
                    {
                     sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                     tp = (float)MathMin(tp / (MaxTP * Point()), 1);
                     result[0] = float(MathMax(Buffer[tr].States[i].account[0] / 100 * 0.01, 0.011));
                     result[1] = tp;
                     result[2] = sl;
                     for(int j = 3; j < NActions; j++)
                        result[j] = 0;
                     bActions.AssignArray(result);
                    }
                 }
               else
                 {
                  if(argmax == 0 || argmax > argmin)
                    {
                     float tp = 0;
                     float sl = 0;
                     float cur_sl = float(MaxSL * Point());
                     ulong pos = 0;
                     for(ulong j = 0; j < argmin; j++)
                       {
                        tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
                        pos = j;
                        if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
                           break;
                        sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
                       }
                     if(tp < 0)
                       {
                        sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                        tp = (float)MathMin(-tp / (MaxTP * Point()), 1);
                        result[3] = float(MathMax(Buffer[tr].States[i].account[0] / 100 * 0.01, 0.011));
                        result[4] = tp;
                        result[5] = sl;
                        for(int j = 0; j < 3; j++)
                           result[j] = 0;
                        bActions.AssignArray(result);
                       }
                    }
                 }
              }
           }
         //--- Actor Policy
         if(!Actor.backProp(GetPointer(bActions), (CBufferFloat*)GetPointer(bAccount), GetPointer(bGradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = double(iter + i - start) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

