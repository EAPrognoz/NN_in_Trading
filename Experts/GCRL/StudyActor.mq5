//+------------------------------------------------------------------+
//| StudyActor.mq5 |
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
CNet                 Actor;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         Gradient;
CBufferFloat         *Result;
vector<float>        SchedulerResult;
vector<float>        check;
vector<float>        ActorResult;
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
      //---
     }
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
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   Gradient.BufferInit(AccountDescr, 0);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Actor.Save(FileName + "Act.nnw", Actor.getRecentAverageError(), 0, 0, TimeCurrent(), true);
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
   int total_steps = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      if(Buffer[tr].Total > total_steps)
         total_steps = Buffer[tr].Total;
     }
   uint ticks = GetTickCount();
   int total_iter=(Iterations+total_tr-1)/total_tr;
//---
   for(int iter = 0; (iter < total_iter && !IsStopped()); iter ++)
     {
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (total_steps - 2));
      for(int tr = 0; tr < total_tr; tr++)
        {
         if(i >= (Buffer[tr].Total - 1))
            continue;
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
         //---
         if(Account.GetIndex() >= 0)
            Account.BufferWrite();
         if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         //---
         ActorResult = vector<float>::Zeros(NActions);
         ActorResult[Buffer[tr].Actions[i]] = Buffer[tr].Revards[i];
         Result.AssignArray(ActorResult);
         if(!Actor.backProp(Result, GetPointer(Account), GetPointer(Gradient)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Actor", iter * 100.0 / (double)(total_iter), Actor.getRecentAverageError());
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

