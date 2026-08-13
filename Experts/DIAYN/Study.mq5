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
input int                  Iterations     = 100000;
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
CBufferFloat         State1;
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
   Discriminator.getResults(Result);
   if(Result.Total() != NSkills)
     {
      PrintFormat("The scope of the discriminator does not match the skills count (%d <> %d)", NSkills, Result.Total());
      return INIT_FAILED;
     }
   Scheduler.getResults(Result);
   Scheduler.SetUpdateTarget(MathMax(Iterations / 100, 50000));
   if(Result.Total() != NSkills)
     {
      PrintFormat("The scope of the scheduler does not match the skills count (%d <> %d)", NSkills, Result.Total());
      return INIT_FAILED;
     }
   Actor.getResults(Result);
   Actor.SetUpdateTarget(MathMax(Iterations / 100, 50000));
   if(Result.Total() != NActions)
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
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)(((double)MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      State1.AssignArray(Buffer[tr].States[i].state);
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      State1.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      State1.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      State1.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      State1.Add(Buffer[tr].States[i].account[3] / 100.0f);
      State1.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      State1.Add(Buffer[tr].States[i].account[5]);
      State1.Add(Buffer[tr].States[i].account[6]);
      State1.Add(Buffer[tr].States[i].account[7] / PrevBalance);
      State1.Add(Buffer[tr].States[i].account[8] / PrevBalance);
      //---
      if(IsStopped())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      if(!Scheduler.feedForward(GetPointer(State1), 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      int skill = Scheduler.getAction();
      SchedulerResult = vector<float>::Zeros(NSkills);
      SchedulerResult[skill] = 1;
      State1.AddArray(SchedulerResult);
      //---
      if(IsStopped())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      if(!Actor.feedForward(GetPointer(State1), 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      int action = Actor.getAction();
      if(action < 0 || action >= NActions)
        {
         iter--;
         continue;
        }
      Actor.getResults(ActorResult);;
      State1.AssignArray(Buffer[tr].States[i + 1].state);
      vector<float> account;
      account.Assign(Buffer[tr].States[i].account);
      int bar = (HistoryBars - 1) * BarDescr;
      double cl_op = Buffer[tr].States[i + 1].state[bar];
      // Implements SymbolInfoDouble.
      double prof_1l = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT) * cl_op /
                       SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      switch(action)
        {
         case 0:
            account[5] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            account[7] += account[5] * (float)prof_1l;
            account[8] -= account[6] * (float)prof_1l;
            account[4] = account[7] + account[8];
            account[1] = account[0] + account[4];
            break;
         case 1:
            account[6] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            account[7] += account[5] * (float)prof_1l;
            account[8] -= account[6] * (float)prof_1l;
            account[4] = account[7] + account[8];
            account[1] = account[0] + account[4];
            break;
         case 2:
            account[0] += account[4];
            account[1] = account[0];
            account[2] = account[0];
            for(bar = 3; bar < AccountDescr; bar++)
               account[bar] = 0;
            break;
         case 3:
            account[7] += account[5] * (float)prof_1l;
            account[8] -= account[6] * (float)prof_1l;
            account[4] = account[7] + account[8];
            account[1] = account[0] + account[4];
            break;
        }
      PrevBalance = Buffer[tr].States[i].account[0];
      PrevEquity = Buffer[tr].States[i].account[1];
      State1.Add((account[0] - PrevBalance) / PrevBalance);
      State1.Add(account[1] / PrevBalance);
      State1.Add((account[1] - PrevEquity) / PrevEquity);
      State1.Add(account[3] / 100.0f);
      State1.Add(account[4] / PrevBalance);
      State1.Add(account[5]);
      State1.Add(account[6]);
      State1.Add(account[7] / PrevBalance);
      State1.Add(account[8] / PrevBalance);
      //---
      if(!Discriminator.feedForward(GetPointer(State1), 1, false,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Discriminator.getResults(DiscriminatorResult);
      ActorResult[action] = DiscriminatorResult.Loss(SchedulerResult, LOSS_CCE);
      Result.AssignArray(SchedulerResult);
      if(!Discriminator.backProp(Result,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      Result.AssignArray(SchedulerResult * ((account[0] - PrevBalance) / PrevBalance));
      if(!Scheduler.backProp(Result, DiscountFactor, GetPointer(State1), 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Result.AssignArray(ActorResult);
      State1.AddArray(SchedulerResult);
      if(!Actor.backProp(Result, DiscountFactor, GetPointer(State1), 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
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
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Scheduler", Scheduler.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Discriminator", Discriminator.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

