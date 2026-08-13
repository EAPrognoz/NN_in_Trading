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
#define Critic
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 100000;
input int                  Batch          = 10;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 cEncoder;
CNet                 cActor;
CNet                 cDirector;
CNet                 cCritic;
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
   CArrayObj *encoder = new CArrayObj();
   CArrayObj *actor = new CArrayObj();
   CArrayObj *director = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(encoder, actor, director, critic))
     {
      delete encoder;
      delete actor;
      delete director;
      delete critic;
      return INIT_FAILED;
     }
   if(!cEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Can't load pretrained Encoder");
      delete encoder;
      delete actor;
      delete director;
      delete critic;
      return INIT_FAILED;
      return INIT_FAILED;
     }
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor.Create(actor))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   if(!cDirector.Load(FileName + "Dir.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Director model");
      if(!cDirector.Create(director))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   if(!cCritic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Critic model");
      if(!cCritic.Create(critic))
        {
         delete encoder;
         delete actor;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   delete encoder;
   delete actor;
   delete director;
   delete critic;
//---
   cEncoder.TrainMode(false);
   cActor.TrainMode(true);
   cDirector.TrainMode(true);
   cCritic.TrainMode(true);
   cEncoder.SetOpenCL(cActor.GetOpenCL());
   cDirector.SetOpenCL(cActor.GetOpenCL());
   cCritic.SetOpenCL(cActor.GetOpenCL());
//---
   cActor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   cEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   if(!bGradient.BufferInit(MathMax(AccountDescr, NActions), 0) ||
      !bGradient.BufferCreate(cActor.GetOpenCL()))
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
      if(!cActor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Actor");
      if(!cDirector.Save(FileName + "Dir.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Direcctor");
      if(!cCritic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Critic");
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
   switch(id)
     {
      case 1001:
         Train();
         break;
      case 1007:
         PrintFormat("Wrong event: %d", id);
         ExpertRemove();
         break;
     }
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = vector<float>::Full(Buffer.Size(), 1.0f / Buffer.Size());
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
      if(
         !cEncoder.Clear()
         || !cActor.Clear()
         || !cDirector.Clear()
         || !cCritic.Clear()
      )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
      result = vector<float>::Zeros(NActions);
      for(int i = start; i < MathMin(Buffer[tr].Total, start + Batch); i++)
        {
         if(!state.Assign(Buffer[tr].States[i].state) ||
            MathAbs(state).Sum() == 0 ||
            !bState.AssignArray(state))
           {
            iter -= Batch + start - i;
            break;
           }
         //---
         bTime.Clear();
         bTime.Reserve(HistoryBars);
         double time = (double)Buffer[tr].States[i].account[7];
         for(int t = i; t >= MathMax(0, i - HistoryBars + 1); t--)
            if(!bTime.Add((float)(double)Buffer[tr].States[t].account[7]))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
         if(bTime.Total() < HistoryBars)
           {
            float period = MathMin(Buffer[tr].States[i + 1].account[7] - Buffer[tr].States[i].account[7],
                                   Buffer[tr].States[i + 2].account[7] - Buffer[tr].States[i + 1].account[7]);
            do
              {
               if(!bTime.Add(bTime[-1] - period))
                 {
                  PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                  Stop = true;
                  break;
                 }
              }
            while(bTime.Total() < HistoryBars);
           }
         if(bTime.GetIndex() >= 0)
            if(!bTime.BufferWrite())
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
         //--- Account
         float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         float profit = float(bState[0] / _Point * (result[0] - result[3]));
         bAccount.Clear();
         bAccount.Add(1);
         bAccount.Add((PrevEquity + profit) / PrevEquity);
         bAccount.Add(profit / PrevEquity);
         bAccount.Add(MathMax(result[0] - result[3], 0));
         bAccount.Add(MathMax(result[3] - result[0], 0));
         bAccount.Add((bAccount[3] > 0 ? profit / PrevEquity : 0));
         bAccount.Add((bAccount[4] > 0 ? profit / PrevEquity : 0));
         bAccount.Add(0);
         double x = time / (double)(D'2024.01.01' - D'2023.01.01');
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_MN1);
         bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_W1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_D1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(bAccount.GetIndex() >= 0)
            if(!bAccount.BufferWrite())
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
         //--- Feed Forward
         if(!cEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, GetPointer(cEncoder), LatentLayer))
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
         //---
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
            if(pos > 0 && tp > 0)
              {
               sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
               tp = float(MathMin(tp / (MaxTP * Point()), 1));
               result[0] = MathMax(result[0] - result[3], 0.011f);
               result[5] = result[1] = tp;
               result[4] = result[2] = sl;
               result[3] = 0;
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
               if(pos > 0 && tp < 0)
                 {
                  sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                  tp = float(MathMin(-tp / (MaxTP * Point()), 1));
                  result[3] = MathMax(result[3] - result[0], 0.011f);
                  result[2] = result[4] = tp;
                  result[1] = result[5] = sl;
                  result[0] = 0;
                  bActions.AssignArray(result);
                 }
              }
            else
              {
               ulong argmin = target.ArgMin();
               ulong argmax = target.ArgMax();
               float max_sl = float(MaxSL * Point());
               while(argmax > 0 && argmin > 0)
                 {
                  if(argmax < argmin && target[argmax] / 2 > MathAbs(target[argmin]) && MathAbs(target[argmin]) < max_sl)
                     break;
                  if(argmax > argmin && target[argmax] < MathAbs(target[argmin] / 2) && target[argmax] < max_sl)
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
                  if(pos > 0 && tp > 0)
                    {
                     sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                     tp = (float)MathMin(tp / (MaxTP * Point()), 1);
                     result[0] = float(MathMax(Buffer[tr].States[i].account[0] / 100 * 0.01, 0.011));
                     result[5] = result[1] = tp;
                     result[4] = result[2] = sl;
                     result[3] = 0;
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
                     if(pos > 0 && tp < 0)
                       {
                        sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * Point()), 1), 0.01);
                        tp = (float)MathMin(-tp / (MaxTP * Point()), 1);
                        result[3] = float(MathMax(Buffer[tr].States[i].account[0] / 100 * 0.01, 0.011));
                        result[2] = result[4] = tp;
                        result[1] = result[5] = sl;
                        result[0] = 0;
                        bActions.AssignArray(result);
                       }
                    }
                 }
              }
           }
         //--- Actor Policy
         bActions.GetData(result);
         if(!cActor.backProp(GetPointer(bActions), (CNet*)GetPointer(cEncoder), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Critic
         if(!cCritic.feedForward(GetPointer(bActions), 1, false, (CNet*)GetPointer(cEncoder), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         float reward = float((bActions[0] - bActions[3]) * fstate[0, 0] / Point());
         Result.Clear();
         if(!Result.Add(reward)
            || !cCritic.backProp(Result, (CNet*)GetPointer(cEncoder), LatentLayer)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Director
         Result.Clear();
         if((MathRand() / 32767.0) > 0.5)
            Result.Add(1);
         else
           {
            target = vector<float>::Zeros(NActions);
            for(int i = 0; i < NActions; i++)
               target[i] = float(MathRand() / 32767.0);
            bActions.AssignArray(target);
            Result.Add(0);
           }
         if(!cDirector.feedForward(GetPointer(bActions), 1, false, (CNet*)GetPointer(cEncoder), LatentLayer)
            || !cDirector.backProp(Result, (CNet*)GetPointer(cEncoder), LatentLayer)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = double(iter + i - start) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, cActor.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Director", percent, cDirector.getRecentAverageError());
            str += StringFormat("%-16s %6.2f%% -> Error %15.8f\n", "Critic", percent, cCritic.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", cActor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Director", cDirector.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", cCritic.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

