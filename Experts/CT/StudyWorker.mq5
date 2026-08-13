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
input int                  Iterations     = 1e7;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
CNet                 Worker;
CNet                 Descrimitator;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         *Data;
CBufferFloat         *Result;
STrajectory          Buffer[];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- load models
   float temp;
   if(!Worker.Load(FileName + "Work.nnw", temp, temp, temp, dtStudied, true) ||
      !Descrimitator.Load(FileName + "Descr.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *worker = new CArrayObj();
      CArrayObj *descriminator = new CArrayObj();
      if(!CreateWorkerDescriptions(worker, descriminator))
        {
         delete worker;
         delete descriminator;
         return INIT_FAILED;
        }
      if(!Worker.Create(worker) ||
         !Descrimitator.Create(descriminator))
        {
         delete worker;
         delete descriminator;
         return INIT_FAILED;
        }
      delete worker;
      delete descriminator;
      //---
     }
//---
   Descrimitator.SetOpenCL(Worker.GetOpenCL());
//---
   Worker.getResults(Data);
   if(Data.Total() != NActions)
     {
      PrintFormat("The scope of the Worker does not match the actions count (%d <> %d)", NActions, Data.Total());
      return INIT_FAILED;
     }
//---
   Descrimitator.GetLayerOutput(0, Data);
   if(Data.Total() != NActions)
     {
      PrintFormat("Input size of Descriminator doesn't match Worker output (%d <> %d)", Data.Total(), NActions);
      return INIT_FAILED;
     }
   Data.Clear();
   Worker.TrainMode(true);
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
   Worker.Save(FileName + "Work.nnw", 0, 0, 0, TimeCurrent(), true);
   Descrimitator.Save(FileName + "Descr.nnw", 0, 0, 0, TimeCurrent(), true);
   delete Data;
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
   uint ticks = GetTickCount();
//---
   bool StopFlag = false;
   for(int iter = 0; (iter < Iterations && !IsStopped() && !StopFlag); iter ++)
     {
      Data.BufferInit(WorkerInput, 0);
      int pos = int(MathRand() / 32767.0 * (WorkerInput - 1));
      Data.Update(pos, 1.0f);
      //--- Study
      if(!Worker.feedForward(Data,1,false,(CBufferFloat*)NULL) ||
         !Descrimitator.feedForward(GetPointer(Worker),-1,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         StopFlag = true;
         break;
        }
      //vector<float> temp;
      //Worker.getResults(temp);
      //Descrimitator.getResults(Result);
      if(!Descrimitator.backProp(Data,(CBufferFloat *)NULL, (CBufferFloat *)NULL) ||
         !Worker.backPropGradient((CBufferFloat *)NULL, (CBufferFloat *)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         StopFlag = true;
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Desciminator", iter * 100.0 / (double)(Iterations), Descrimitator.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Descriminator", Descrimitator.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

