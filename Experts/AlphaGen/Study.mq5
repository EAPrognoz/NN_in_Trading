#property strict
#define ALPHAGEN_STUDY
input bool PPOProfile=false;
#define ALPHAGEN_PPO_PROFILE PPOProfile
#include "Trajectory.mqh"
CAlphaGenApplication App;
string StopButton="AlphaGenStop";
//+------------------------------------------------------------------+
//| Initialize the EA and its event resources.                       |
//+------------------------------------------------------------------+
int OnInit()
  {
   Comment("AlphaGen Study | initialization");
   ChartRedraw();
   if(!App.Init(true,true))
     {
      return(INIT_FAILED);
     }
   ObjectCreate(0,StopButton,OBJ_BUTTON,0,0,0);
   ObjectSetString(0,StopButton,OBJPROP_TEXT,"Остановить и сохранить");
   ObjectSetInteger(0,StopButton,OBJPROP_XSIZE,220);
   ObjectSetInteger(0,StopButton,OBJPROP_YSIZE,30);
   Comment("AlphaGen Study | preparing the first historical bar");
   ChartRedraw();
   if(!EventSetMillisecondTimer(AG_TIMER_MS))
     {
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Run one bounded training or historical operation.                |
//+------------------------------------------------------------------+
void OnTimer()
  {
   ResetLastError();
   if(!App.StudyStep())
     {
      int error=GetLastError();
      App.Fault("Study atomic operation failed");
      Print("AlphaGen Study stopped; last error for step: ",error);
      EventKillTimer();
      ExpertRemove();
     }
  }
//+------------------------------------------------------------------+
//| Record the user stop request between atomic operations.          |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(id==CHARTEVENT_OBJECT_CLICK && sparam==StopButton)
     {
      if(!App.stop_requested)
        {
         App.stop_requested=true;
         ObjectSetString(0,StopButton,OBJPROP_TEXT,"Stopping and saving...");
         ObjectSetInteger(0,StopButton,OBJPROP_STATE,false);
         Comment("\n\nAlphaGen Study | stop request accepted\nFinishing the current operation before saving");
         ChartRedraw();
         Print("AlphaGen Study stop request accepted");
        }
     }
  }
//+------------------------------------------------------------------+
//| Release event resources without trading or saving.               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectDelete(0,StopButton);
   Comment("");
   Print("AlphaGen Study unloaded, finalized=",App.Finalized);
  }
