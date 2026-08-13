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
#define StudyExpert
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 100000;
input float                Tau            = 0.01f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet_SAC_DICE        Net;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bActions;
CBufferFloat         bNextState;
CBufferFloat         bNextAccount;
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
   if(!Net.Load(FileName, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      if(!CreateDescriptions(actor, critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!Net.Create(actor, critic, critic, critic, LatentLayer))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
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
   Net.Save(FileName, true);
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
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      if(i<0)
        {
         iter--;
         continue;
        }
      //--- Target
      bNextState.AssignArray(Buffer[tr].States[i + 1].state);
      float PrevBalance = Buffer[tr].States[i].account[0];
      float PrevEquity = Buffer[tr].States[i].account[1];
      if(PrevBalance==0)
        {
         iter--;
         continue;
        }
      bNextAccount.Clear();
      bNextAccount.Add((Buffer[tr].States[i + 1].account[0] - PrevBalance) / PrevBalance);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[1] / PrevBalance);
      bNextAccount.Add((Buffer[tr].States[i + 1].account[1] - PrevEquity) / PrevEquity);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[2]);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[3]);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[4] / PrevBalance);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[5] / PrevBalance);
      bNextAccount.Add(Buffer[tr].States[i + 1].account[6] / PrevBalance);
      double x = (double)Buffer[tr].States[i + 1].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
      bNextAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_MN1);
      bNextAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_W1);
      bNextAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i + 1].account[7] / (double)PeriodSeconds(PERIOD_D1);
      bNextAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      //---
      bState.AssignArray(Buffer[tr].States[i].state);
      PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      bAccount.Clear();
      bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      bAccount.Add(Buffer[tr].States[i].account[2]);
      bAccount.Add(Buffer[tr].States[i].account[3]);
      bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      x = (double)Buffer[tr].States[i].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_MN1);
      bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_W1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_D1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      //---
      bActions.AssignArray(Buffer[tr].States[i].action);
      vector<float> log_prob;
      log_prob.Assign(Buffer[tr].States[i].log_prob);
      //---
      if(!Net.Study(GetPointer(bState), GetPointer(bAccount), GetPointer(bActions), log_prob,
                    GetPointer(bNextState), GetPointer(bNextAccount),
                    Buffer[tr].Revards[i] - DiscFactor * Buffer[tr].Revards[i + 1], DiscFactor, Tau))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         float loss1, loss2;
         Net.GetLoss(loss1, loss2);
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic1", iter * 100.0 / (double)(Iterations), loss1);
         str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Critic2", iter * 100.0 / (double)(Iterations), loss2);
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   float loss1, loss2;
   Net.GetLoss(loss1, loss2);
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic1", loss1);
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic2", loss2);
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

