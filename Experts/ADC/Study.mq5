//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#define Critic
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                            |
//+------------------------------------------------------------------+
input int                  Batch          = 10;
CNet                 cEncoder;
CNet                 cTask;
CNet                 cActor;
CNet                 cProbability;
CNet                 cDirector;
CNet                 cCritic;
float                dError;
datetime             dtStudied;
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         bAccount;
CBufferFloat         bGradient;
CBufferFloat         bActions;
CBufferFloat         *Result;
vector<float>        check;
vector<float>        STD_Actor;
vector<float>        STD_Goal;
COpenCLMy            *OpenCL;
//+------------------------------------------------------------------+
//| Expert initialization function                              |
//+------------------------------------------------------------------+
  {
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
// Load Models.
   float temp;
   CArrayObj *encoder = new CArrayObj();
   CArrayObj *task = new CArrayObj();
   CArrayObj *actor = new CArrayObj();
   CArrayObj *probability = new CArrayObj();
   CArrayObj *director = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(encoder, task, actor, probability, director, critic))
     {
      delete encoder;
      delete task;
      delete actor;
      delete probability;
      delete director;
      delete critic;
      return INIT_FAILED;
     }
   if(!cEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new State Encoder");
      if(!cEncoder.Create(encoder))
        {
         delete encoder;
         delete task;
         delete actor;
         delete probability;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   if(!cTask.Load(FileName + "Tsk.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Task Encoder");
      if(!cTask.Create(task))
        {
         delete encoder;
         delete task;
         delete actor;
         delete probability;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor.Create(actor))
        {
         delete encoder;
         delete task;
         delete actor;
         delete probability;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   if(!cProbability.Load(FileName + "Prb.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Probability model");
      if(!cProbability.Create(probability))
        {
         delete encoder;
         delete task;
         delete actor;
         delete probability;
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
         delete task;
         delete actor;
         delete probability;
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
         delete task;
         delete actor;
         delete probability;
         delete director;
         delete critic;
         return INIT_FAILED;
        }
     }
   delete encoder;
   delete task;
   delete actor;
   delete probability;
   delete director;
   delete critic;
   cEncoder.TrainMode(true);
   cTask.TrainMode(true);
   cActor.TrainMode(true);
   cProbability.TrainMode(true);
   cDirector.TrainMode(true);
   cCritic.TrainMode(true);
   cProbability.SetOpenCL(cActor.GetOpenCL());
   cEncoder.SetOpenCL(cActor.GetOpenCL());
   cTask.SetOpenCL(cActor.GetOpenCL());
   cDirector.SetOpenCL(cActor.GetOpenCL());
   cCritic.SetOpenCL(cActor.GetOpenCL());
   OpenCL=cActor.GetOpenCL();
   cActor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   cEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
   if(!bGradient.BufferInit(MathMax(AccountDescr, NActions), 0) ||
      !bGradient.BufferCreate(cActor.GetOpenCL()))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                            |
//+------------------------------------------------------------------+
  {
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      cEncoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      cTask.Save(FileName + "Tsk.nnw", 0, 0, 0, TimeCurrent(), true);
      cActor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      cProbability.Save(FileName + "Prb.nnw", 0, 0, 0, TimeCurrent(), true);
      cDirector.Save(FileName + "Dir.nnw", 0, 0, 0, TimeCurrent(), true);
      cCritic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
     }
   delete Result;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                         |
//+------------------------------------------------------------------+
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//| Train function                                              |
//+------------------------------------------------------------------+
  {
   vector<float> probability = vector<float>::Full(Buffer.Size(), 1.0f / Buffer.Size());
   vector<float> result, target, state;
   matrix<float> fstate = matrix<float>::Zeros(1, NForecast * BarDescr);
   bool Stop = false;
   uint ticks = GetTickCount();
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
         || !cTask.Clear()
         || !cActor.Clear()
         || !cProbability.Clear()
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
            // Implements MathAbs.
            MathAbs(state).Sum() == 0 ||
            !bState.AssignArray(state))
           {
            iter -= Batch + start - i;
            break;
           }
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
         // Account.
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
         bAccount.AddArray(GetPointer(bTime));
         if(bAccount.GetIndex() >= 0)
            bAccount.BufferWrite();
         // Feed Forward.
         if(!cEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cTask.feedForward((CBufferFloat*)GetPointer(bState), 1, false, GetPointer(cEncoder), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bAccount), 1, false, GetPointer(cTask), -1))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cProbability.feedForward(GetPointer(cEncoder), LatentLayer, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         // Look for target.
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
         // State Encoder.
         Result.AssignArray(fstate);
         if(!cEncoder.backProp(Result, (CBufferFloat*)NULL, NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
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
               if(tp < 0)
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
                  if(tp > 0)
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
                     if(tp < 0)
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
         // Actor Policy.
         if(!cActor.backProp(GetPointer(bActions), (CNet*)GetPointer(cTask), -1))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         // Critic.
         if(!cCritic.feedForward(GetPointer(bActions), 1, false, (CNet*)GetPointer(cEncoder), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         float reward = float((result[0] - result[3]) * fstate[0, 0] / Point());
         Result.Clear();
         if(!Result.Add(reward)
            || !cCritic.backProp(Result, (CNet*)GetPointer(cEncoder), LatentLayer)
            || !cEncoder.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, LatentLayer, true)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         // Director.
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
            || !cEncoder.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, LatentLayer, true)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         // Agents.
         target = result;
         if(target[0] > 0)
            target[0] = 1;
         if(target[3] > 0)
            target[3] = 1;
         Result.Clear();
         for(int i = 0; i < BarDescr; i++)
           {
            if(!Result.AddArray(target))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
         if(!cTask.backProp(Result, (CNet*)GetPointer(cEncoder), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         // Probability.
         target = vector<float>::Zeros(NActions / 3);
         vector<float> trend = fstate.Col(0).CumSum();
         ulong argmax = MathAbs(trend).ArgMax();
         if(trend[argmax] > 0)
            target[0] = 1;
         else
            if(trend[argmax] < 0)
               target[1] = 1;
         if(!Result.AssignArray(target)
            || !cProbability.backProp(Result, (CNet*)GetPointer(cEncoder), LatentLayer)
            || !cEncoder.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, LatentLayer)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(GetTickCount() - ticks > 500)
           {
            double percent = double(iter + i - start) * 100.0 / (Iterations);
            string str = StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Encoder", percent, cEncoder.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Task", percent, cTask.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, cActor.getRecentAverageError());
            str += StringFormat("%-13s %6.2f%% -> Error %15.8f\n", "Probability", percent, cProbability.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Director", percent, cDirector.getRecentAverageError());
            str += StringFormat("%-16s %6.2f%% -> Error %15.8f\n", "Critic", percent, cCritic.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Encoder", cEncoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Task", cTask.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", cActor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Probability", cProbability.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Director", cDirector.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", cCritic.getRecentAverageError());
   ExpertRemove();
  }
//+------------------------------------------------------------------+

