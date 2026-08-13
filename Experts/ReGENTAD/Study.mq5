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
#define StudyOnline
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input datetime             Start          = D'2024.01.01';
input datetime             End            = D'2026.01.01';
input int                  Epochs         = 100;
input double               MinBalance     = 50;
input double               MaxBalance     = 150;
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet                 cActor;
CNet                 cCritic;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bTime;
CBufferFloat         bGradient;
CBufferFloat         *Result;
CBufferFloat         *Action;
vector<float>        check;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
//--- load models
   float temp;
   CArrayObj *actor = new CArrayObj();
   CArrayObj *critic = new CArrayObj();
   if(!CreateDescriptions(actor, critic))
     {
      DeleteObj(actor)
      DeleteObj(critic)
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
   if(!cActor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Actor");
      if(!cActor.Create(actor))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         DeleteObj(actor)
         DeleteObj(critic)
         return INIT_FAILED;
        }
     }
   if(!cCritic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Create new Critic");
      if(!cCritic.Create(critic))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         DeleteObj(actor)
         DeleteObj(critic)
         return INIT_FAILED;
        }
     }
   DeleteObj(actor)
   DeleteObj(critic)
//---
   cActor.TrainMode(true);
   cCritic.TrainMode(true);
   COpenCL *opencl = cActor.GetOpenCL();
   cCritic.SetOpenCL(opencl);
//---
   if(!Symb.Name(_Symbol) || !Symb.Refresh())
      return INIT_FAILED;
//---
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
//---
   cActor.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr + AccountDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr + AccountDescr));
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
      if(!cCritic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true))
         PrintFormat("Error of save model: %s", "Critic");
     }
   delete Result;
   delete Action;
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
         Print("Event 1007");
         if(!EventChartCustom(ChartID(), 1, 0, 0, "ChartEvent"))
           {
            PrintFormat("Error of create study event: %d", GetLastError());
            ExpertRemove();
           }
         break;
     }
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
//---
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   int count = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars &&
                    CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars &&
                    MACD.BarsCalculated() >= bars
                   );
      Sleep(100);
      count++;
     }
   while(!calculated && count < 100);
   if(!calculated)
     {
      PrintFormat("%s -> %d The training data has not been loaded", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   if(!ArraySetAsSeries(Rates, true))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      ExpertRemove();
      return;
     }
//---
   vector<float> result, target, neg_target;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int epoch = 0; (epoch < Epochs && !IsStopped() && !Stop); epoch ++)
     {
      if(!cActor.Clear() ||
         !cCritic.Clear())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      for(int posit = start - HistoryBars - NForecast - 1; posit >= end; posit--)
        {
         if(!CreateBuffers(posit, GetPointer(bState), GetPointer(bTime), Result))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         const vector<float> account = SampleAccount(GetPointer(bState), datetime(bTime[0]), MaxBalance, MinBalance);
         const vector<float> target_action = OraculAction(account, Result);
         if(!bState.AddArray(account))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         //--- Feed Forward
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
#ifdef _DEBUG
         vector<float> res;
         cActor.getResults(res);
#endif
         if(!cCritic.feedForward(GetPointer(cActor), -1, GetPointer(cActor), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
#ifdef _DEBUG
         cCritic.getResults(res);
#endif
         //--- Study
         cActor.getResults(Action);
         double balance = account[0] * EtalonBalance;
         double reward = CheckAction(Action, balance, posit - NForecast + 1) * balance / EtalonBalance;
         //---
         Result.Clear();
         if(!Result.Add(float(reward)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cCritic.backProp(Result, GetPointer(cActor), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Oracul
         if(!Action.AssignArray(target_action))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         reward = CheckAction(Action, balance, posit - NForecast + 1) * balance / EtalonBalance;
         if(reward > 0)
            if(!cActor.backProp(Action, (CBufferFloat*)NULL))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)) ||
            !cCritic.feedForward(Action, 1, false, GetPointer(cActor), LatentLayer))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!Result.Update(0, float(reward)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!cCritic.backProp(Result, GetPointer(cActor), LatentLayer) ||
            !cActor.backPropGradient((CBufferFloat*)GetPointer(bTime), (CBufferFloat*)NULL, LatentLayer, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         cCritic.TrainMode(false);
         if(!cActor.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bTime)) ||
            !cCritic.feedForward(GetPointer(cActor), -1, GetPointer(cActor), LatentLayer) ||
            !cCritic.backProp(Result, GetPointer(cActor), LatentLayer) ||
            !cActor.backPropGradient((CBufferFloat*)GetPointer(bTime), (CBufferFloat*)NULL, -1, true))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         cCritic.TrainMode(true);
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (epoch + 1.0 - double(posit - end) / (start - end - HistoryBars - NForecast)) / Epochs * 100.0;
            string str = "";
            str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Actor", percent, cActor.getRecentAverageError());
            str += StringFormat("%-12s %6.2f%% -> Error %15.8f\n", "Critic", percent, cCritic.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", cActor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", cCritic.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

