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
input int                  Iterations     = 5e5;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 Decoder;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Actions;
CBufferFloat         *Result;
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
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Decoder.Load(FileName + "Dec.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new CVAE");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateCVAEDescriptions(encoder,decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) || !Decoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
         delete encoder;
         delete decoder;
     }
//---
   OpenCL = Encoder.GetOpenCL();
   Decoder.SetOpenCL(OpenCL);
//---
   Decoder.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the Decoder does not match the actions count (%d <> %d)", NActions, Result.Total());
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
   Encoder.getResults(Result);
   int latent_state = Result.Total();
   Decoder.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Decoder doesn't match result of Encoder (%d <> %d)", Result.Total(), latent_state);
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
   Encoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
   Decoder.Save(FileName + "Dec.nnw", Decoder.getRecentAverageError(), 0, 0, TimeCurrent(), true);
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
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   int bar = (HistoryBars - 1) * BarDescr;
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = int((MathRand() * MathRand() / MathPow(32767, 2)) * (total_tr));
      int i = int((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      if(i < 0)
         continue;
      State.AssignArray(Buffer[tr].States[i].state);
      Actions.AssignArray(Buffer[tr].States[i].action);
      if(Actions.GetIndex() >= 0)
         Actions.BufferWrite();
      //---
      if(!Encoder.feedForward(GetPointer(State), 1,false, GetPointer(Actions)) ||
         !Decoder.feedForward(GetPointer(Encoder), -1, GetPointer(Encoder),1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(!Decoder.backProp(GetPointer(Actions), GetPointer(Encoder), 1) ||
         !Encoder.backPropGradient(GetPointer(Actions), GetPointer(Actions)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Decoder", iter * 100.0 / (double)(Iterations), Decoder.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Decoder", Decoder.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

