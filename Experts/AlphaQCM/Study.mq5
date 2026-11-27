//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                                         AlphaQCM |
//|                                                     NeuroNet_DNG |
//|                           https://www.mql5.com/ru/articles/26115 |
//+------------------------------------------------------------------+
#define ALPHAGEN_STUDY
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input bool PPOProfile=false;
#define ALPHAGEN_PPO_PROFILE PPOProfile
input int TradeBackwardBars=64;   // Trading LSTM backpropagation bars (0 = full context)
#define ALPHAGEN_TRADE_BACKWARD_BARS TradeBackwardBars
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CAlphaQCMApplication App;
string               StopButton="AlphaQCMStop";
//+------------------------------------------------------------------+
//| OnInit - initialize the EA and its event resources.              |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(TradeBackwardBars<0)
      return(INIT_PARAMETERS_INCORRECT);

   Comment("AlphaQCM Study | initialization");
   ChartRedraw();
   if(!App.Init(true,true))
     {
      return(INIT_FAILED);
     }
   ObjectCreate(0,StopButton,OBJ_BUTTON,0,0,0);
   ObjectSetString(0,StopButton,OBJPROP_TEXT,"Stop and save");
   ObjectSetInteger(0,StopButton,OBJPROP_XSIZE,220);
   ObjectSetInteger(0,StopButton,OBJPROP_YSIZE,30);
   ObjectSetInteger(0,StopButton,OBJPROP_CORNER,CORNER_RIGHT_UPPER);
   ObjectSetInteger(0,StopButton,OBJPROP_XDISTANCE,230);
   ObjectSetInteger(0,StopButton,OBJPROP_YDISTANCE,50);
   Comment("AlphaQCM Study | preparing the first historical bar");
   ChartRedraw();
   if(!EventSetMillisecondTimer(AG_TIMER_MS))
     {
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnTimer - run one bounded training or historical operation.      |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(App.save_retry_pending)
      return;
   ResetLastError();
   if(!App.StudyStep())
     {
      int error=GetLastError();
      App.Fault("Study atomic operation failed");
      Print("AlphaQCM Study stopped; last error for step: ",error);
      if(App.WaitForRecovery())
        {
         ObjectSetInteger(0,StopButton,OBJPROP_STATE,false);
         ObjectSetString(0,StopButton,OBJPROP_TEXT,"Stop and save recovery");
         return;
        }
      EventKillTimer();
      ExpertRemove();
     }
  }
//+------------------------------------------------------------------+
//| OnChartEvent - Record a stop request between atomic operations   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK&&sparam==StopButton)
     {
      if(App.save_retry_pending)
        {
         App.save_retry_pending=false;
         ObjectSetInteger(0,StopButton,OBJPROP_STATE,false);
         ObjectSetString(0,StopButton,OBJPROP_TEXT,"Saving...");
         return;
        }
      if(!App.stop_requested)
        {
         App.stop_requested=true;
         ObjectSetString(0,StopButton,OBJPROP_TEXT,"Stopping and saving...");
         ObjectSetInteger(0,StopButton,OBJPROP_STATE,false);
         Comment("\n\nAlphaQCM Study | stop request accepted\nFinishing the current operation before saving");
         ChartRedraw();
         Print("AlphaQCM Study stop request accepted");
        }
     }
  }
//+------------------------------------------------------------------+
//| OnDeinit - release event resources without trading or saving.    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectDelete(0,StopButton);
   Comment("");
   Print("AlphaQCM Study unloaded, finalized=",App.Finalized);
  }
//+------------------------------------------------------------------+
