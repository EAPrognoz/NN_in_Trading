//+------------------------------------------------------------------+
//| Research.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 1000;
input int                  DiscriminatorBatch   =  500;
input int                  AgentBatch     = 500;
input int                  SchedulerBatch  = 500;
input float                DiscountFactor  = 0.99f;
bool                 TrainMode = true;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CFQF                 Actor;
CFQF                 Scheduler;
CNet                 Discriminator;
int                  Models = 1;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         StateSkill;
CBufferFloat         *Result;
vector<float>        ActorResult;
vector<float>        SchedulerResult;
vector<float>        DiscriminatorResult;
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
   if(!Actor.Load(FileName + "Act.nnw", dtStudied, true) ||
      !Scheduler.Load(FileName + "Sch.nnw", dtStudied, true) ||
      !Discriminator.Load(FileName + "Dscr.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *scheduler = new CArrayObj();
      CArrayObj *discriminator = new CArrayObj();
      if(!CreateDescriptions(actor, scheduler, discriminator))
        {
         delete actor;
         delete scheduler;
         delete discriminator;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) ||
         !Scheduler.Create(scheduler) ||
         !Discriminator.Create(discriminator))
        {
         delete actor;
         delete scheduler;
         delete discriminator;
         return INIT_FAILED;
        }
      delete actor;
      delete scheduler;
      delete discriminator;
      //---
#ifdef FileName
      if(!Actor.UpdateTarget(FileName + ".nnw") ||
         !Scheduler.UpdateTarget(FileName + ".nnw"))
#else
      if(!Actor.UpdateTarget("FQF.upd") ||
         !Scheduler.UpdateTarget("FQF.nnw"))
#endif
         return INIT_FAILED;
     }
//---
   Discriminator.getResults(DiscriminatorResult);
   if(DiscriminatorResult.Size() != NSkills * AccountDescr)
     {
      PrintFormat("The scope of the discriminator does not match the skills count (%d <> %d)", NSkills, Result.Total());
      return INIT_FAILED;
     }
   Scheduler.getResults(SchedulerResult);
   Scheduler.SetUpdateTarget(MathMax(Iterations*SchedulerBatch / 100, 500000 / SchedulerBatch));
   if(SchedulerResult.Size() != NSkills)
     {
      PrintFormat("The scope of the scheduler does not match the skills count (%d <> %d)", NSkills, Result.Total());
      return INIT_FAILED;
     }
   Actor.getResults(ActorResult);
   Actor.SetUpdateTarget(MathMax(Iterations*AgentBatch / 100, 500000 / AgentBatch * NSkills));
   if(ActorResult.Size() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
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
   Actor.Save(FileName + "Act.nnw", TimeCurrent(), true);
   Scheduler.Save(FileName + "Sch.nnw", TimeCurrent(), true);
   Discriminator.Save(FileName + "Dscr.nnw", 0, 0, 0, TimeCurrent(), true);
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
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves TotalBase.                                               |
//+------------------------------------------------------------------+
bool SaveTotalBase(void)
  {
   int total = ArraySize(Buffer);
   if(total < 0)
      return true;
   int handle = FileOpen(FileName + ".bd", FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle < 0)
      return false;
   if(FileWriteInteger(handle, total) < INT_VALUE)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Save(handle))
        {
         FileClose(handle);
         return false;
        }
   FileFlush(handle);
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Loads TotalBase.                                               |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   vector<float> account, reward;
   int bar, action;
   int skill, shift;
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      for(int phase = 0; phase < 3; phase++)
        {
         int batch = 0;
         switch(phase)
           {
            case 0:
               batch = DiscriminatorBatch;
               break;
            case 1:
               batch = AgentBatch;
               break;
            case 2:
               batch = SchedulerBatch;
               break;
            default:
               PrintFormat("Incorrect phase %d");
               batch = 0;
               break;
           }
         for(int batch_iter = 0; batch_iter < batch; batch_iter++)
           {
            int tr = (int)(((double)MathRand() / 32767.0) * (total_tr - 1));
            int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
            State.AssignArray(Buffer[tr].States[i].state);
            float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
            float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
            State.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
            State.Add(Buffer[tr].States[i].account[1] / PrevBalance);
            State.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
            State.Add(Buffer[tr].States[i].account[2] / PrevBalance);
            State.Add(Buffer[tr].States[i].account[4] / PrevBalance);
            State.Add(Buffer[tr].States[i].account[5]);
            State.Add(Buffer[tr].States[i].account[6]);
            State.Add(Buffer[tr].States[i].account[7] / PrevBalance);
            State.Add(Buffer[tr].States[i].account[8] / PrevBalance);
            //---
            bar = (HistoryBars - 1) * BarDescr;
            double cl_op = Buffer[tr].States[i + 1].state[bar];
            // Implements SymbolInfoDouble.
            double prof_1l = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT) * cl_op /
                             SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            PrevBalance = Buffer[tr].States[i].account[0];
            PrevEquity = Buffer[tr].States[i].account[1];
            if(IsStopped())
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               ExpertRemove();
               break;
              }
            //---
            switch(phase)
              {
               case 0:
                  if(!Discriminator.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  for(skill = 0; skill < NSkills; skill++)
                    {
                     SchedulerResult = vector<float>::Zeros(NSkills);
                     SchedulerResult[skill] = 1;
                     StateSkill.AssignArray(GetPointer(State));
                     StateSkill.AddArray(SchedulerResult);
                     if(IsStopped())
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        break;
                       }
                     if(!Actor.feedForward(GetPointer(State), 1, false))
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        break;
                       }
                     action = Actor.getSample();
                     account = GetNewState(Buffer[tr].States[i].account, action, prof_1l);
                     shift = skill * AccountDescr;
                     DiscriminatorResult[shift] = (account[0] - PrevBalance) / PrevBalance;
                     DiscriminatorResult[shift + 1] = account[1] / PrevBalance;
                     DiscriminatorResult[shift + 2] = (account[1] - PrevEquity) / PrevEquity;
                     DiscriminatorResult[shift + 3] = account[2] / PrevBalance;
                     DiscriminatorResult[shift + 4] = account[4] / PrevBalance;
                     DiscriminatorResult[shift + 5] = account[5];
                     DiscriminatorResult[shift + 6] = account[6];
                     DiscriminatorResult[shift + 7] = account[7] / PrevBalance;
                     DiscriminatorResult[shift + 8] = account[8] / PrevBalance;
                    }
                  if(!Result)
                    {
                     Result = new CBufferFloat();
                     if(!Result)
                       {
                        PrintFormat("Error of create buffer %d", GetLastError());
                        ExpertRemove();
                        break;
                       }
                    }
                  Result.AssignArray(DiscriminatorResult);
                  if(!Discriminator.backProp(Result,(CBufferFloat*)NULL))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  break;
               case 1:
                  if(!Discriminator.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  Discriminator.getResults(DiscriminatorResult);
                  for(skill = 0; skill < NSkills; skill++)
                    {
                     SchedulerResult = vector<float>::Zeros(NSkills);
                     SchedulerResult[skill] = 1;
                     StateSkill.AssignArray(GetPointer(State));
                     StateSkill.AddArray(SchedulerResult);
                     if(IsStopped())
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        ExpertRemove();
                        break;
                       }
                     if(!Actor.feedForward(GetPointer(State), 1, false))
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        ExpertRemove();
                        break;
                       }
                     reward = GetAgentReward(skill, DiscriminatorResult, Buffer[tr].States[i].account);
                     Result.AssignArray(reward);
                     StateSkill.AssignArray(Buffer[tr].States[i + 1].state);
                     account = GetNewState(Buffer[tr].States[i].account, Actor.getAction(), prof_1l);
                     shift = skill * AccountDescr;
                     StateSkill.Add((account[0] - PrevBalance) / PrevBalance);
                     StateSkill.Add(account[1] / PrevBalance);
                     StateSkill.Add((account[1] - PrevEquity) / PrevEquity);
                     StateSkill.Add(account[2] / PrevBalance);
                     StateSkill.Add(account[4] / PrevBalance);
                     StateSkill.Add(account[5]);
                     StateSkill.Add(account[6]);
                     StateSkill.Add(account[7] / PrevBalance);
                     StateSkill.Add(account[8] / PrevBalance);
                     if(!Actor.backProp(Result, DiscountFactor, GetPointer(StateSkill), 1, false))
                       {
                        PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                        ExpertRemove();
                        break;
                       }
                    }
                  break;
               case 2:
                  if(!Scheduler.feedForward(GetPointer(State), 1, false))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  Scheduler.getResults(SchedulerResult);
                  State.AddArray(SchedulerResult);
                  if(IsStopped())
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  if(!Actor.feedForward(GetPointer(State), 1, false))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  action = Actor.getAction();
                  account = GetNewState(Buffer[tr].States[i].account, action, prof_1l);
                  SchedulerResult = SchedulerResult * (account[0] / PrevBalance - 1.0);
                  Result.AssignArray(SchedulerResult);
                  State.AssignArray(Buffer[tr].States[i + 1].state);
                  State.Add((account[0] - PrevBalance) / PrevBalance);
                  State.Add(account[1] / PrevBalance);
                  State.Add((account[1] - PrevEquity) / PrevEquity);
                  State.Add(account[2] / PrevBalance);
                  State.Add(account[4] / PrevBalance);
                  State.Add(account[5]);
                  State.Add(account[6]);
                  State.Add(account[7] / PrevBalance);
                  State.Add(account[8] / PrevBalance);
                  if(!Scheduler.backProp(Result, DiscountFactor, GetPointer(State), 1, false))
                    {
                     PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
                     ExpertRemove();
                     break;
                    }
                  break;
               default:
                  PrintFormat("Wrong phase %d", phase);
                  break;
              }
            if(GetTickCount() - ticks > 500)
              {
               string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Scheduler", iter * 100.0 / (double)(Iterations), Scheduler.getRecentAverageError());
               str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Discriminator",  iter * 100.0 / (double)(Iterations), Discriminator.getRecentAverageError());
               Comment(str);
               ticks = GetTickCount();
              }
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Scheduler", Scheduler.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Discriminator", Discriminator.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets NewState.                                                 |
//+------------------------------------------------------------------+
vector<float> GetNewState(float &prev_account[], int action, double prof_1l)
  {
   vector<float> result;
//---
   result.Assign(prev_account);
   switch(action)
     {
      case 0:
         result[5] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
      case 1:
         result[6] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
      case 2:
         result[0] += result[4];
         result[1] = result[0];
         result[2] = result[0];
         for(int i = 3; i < AccountDescr; i++)
            result[i] = 0;
         break;
      case 3:
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
     }
//--- return result
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Gets AgentReward.                                              |
//+------------------------------------------------------------------+
vector<float> GetAgentReward(int skill, vector<float> &discriminator, float &prev_account[])
  {
//--- prepare
   matrix<float> discriminator_matrix;
   discriminator_matrix.Init(1, discriminator.Size());
   discriminator_matrix.Row(discriminator, 0);
   discriminator_matrix.Reshape(NSkills, AccountDescr);
   vector<float> forecast = discriminator_matrix.Row(skill);
//--- check action
   int action = 3;
   float buy = forecast[5] - prev_account[5];
   float sell = forecast[6] - prev_account[6];
   if(buy < 0 && sell < 0)
      action = 2;
   else
      if(buy > sell)
         action = 0;
      else
         if(buy < sell)
            action = 1;
//--- calculate reward
   vector<float> result = vector<float>::Zeros(NActions);
   float mean = (forecast / discriminator_matrix.Mean(0)).Mean();
   result[action] = MathLog(MathAbs(mean));
//--- return result
   return result;
  }
//+------------------------------------------------------------------+

