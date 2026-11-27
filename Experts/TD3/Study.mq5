//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                                              TD3 |
//|                                                   Copyright DNG® |
//|                           https://www.mql5.com/ru/articles/12892 |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define Study
#include "Trajectory.mqh"
#include "../NeuroNet_DNG/NeuroNet_SelectiveOpenCL.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int   Iterations=1000000;
input int   UpdatePolicy=3;
input int   UpdateTargets=100;
input float Tau=0.01f;
input bool  SelectiveOpenCL=true;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| CTD3Actor - Expose full OpenCL initialization for the actor      |
//+------------------------------------------------------------------+
class CTD3Actor:public CNet
  {
public:
//+------------------------------------------------------------------+
//| InitializeFull - Initialize full                                 |
//+------------------------------------------------------------------+
   bool InitializeFull(void) { return(OpenCLInit()); }
  };

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CSelectiveOpenCL SelectiveRuntime;
STrajectory Buffer[];
CTD3Actor   Actor;
CNet        Critic1;
CNet        Critic2;
CNet        TargetActor;
CNet        TargetCritic1;
CNet        TargetCritic2;
//---
float    dError;
datetime dtStudied;
//---
CBufferFloat State;
CBufferFloat Account;
CBufferFloat Actions;
CBufferFloat Gradient;
CBufferFloat *Result;
vector<float>SchedulerResult;
vector<float>check;
vector<float>ActorResult;
//---
COpenCLMy *OpenCL;
//+------------------------------------------------------------------+
//| InitializeStudyModels - Prepare the six TD3 networks             |
//+------------------------------------------------------------------+
bool InitializeStudyModels(const bool load_checkpoints,const bool common=true)
  {
   const ulong started=GetTickCount64();
   if(SelectiveOpenCL)
     {
      int device=CL_USE_ANY;
#ifdef NN_OPENCL_DEVICE
      device=NN_OPENCL_DEVICE;
#endif
      if(!SelectiveRuntime.Begin(device)||!Actor.UseOpenCLRuntime(GetPointer(SelectiveRuntime)))
        {
         ReturnFalse
        }
     }
   else if(!Actor.InitializeFull())
     {
      ReturnFalse
     }

   OpenCL=Actor.GetOpenCL();
   if(!Critic1.UseOpenCLRuntime(OpenCL)||!Critic2.UseOpenCLRuntime(OpenCL)||
      !TargetActor.UseOpenCLRuntime(OpenCL)||!TargetCritic1.UseOpenCLRuntime(OpenCL)||
      !TargetCritic2.UseOpenCLRuntime(OpenCL))
     {
      ReturnFalse
     }

//--- load models
   float temp;
   const bool loaded=(load_checkpoints&&Actor.Load(FileName+"Act.nnw",temp,temp,temp,dtStudied,common)&&
                      Critic1.Load(FileName+"Crt1.nnw",temp,temp,temp,dtStudied,common)&&
                      Critic2.Load(FileName+"Crt2.nnw",temp,temp,temp,dtStudied,common)&&
                      TargetActor.Load(FileName+"Act.nnw",temp,temp,temp,dtStudied,common)&&
                      TargetCritic1.Load(FileName+"Crt1.nnw",temp,temp,temp,dtStudied,common)&&
                      TargetCritic2.Load(FileName+"Crt2.nnw",temp,temp,temp,dtStudied,common));
   if(load_checkpoints&&!common&&!loaded)
     {
      ReturnFalseEx("TD3 benchmark checkpoint could not be loaded")
     }
   if(!loaded)
     {
      CArrayObj actor;
      CArrayObj critic;
      CArrayObj *actor_description=GetPointer(actor);
      CArrayObj *critic_description=GetPointer(critic);
      if(!CreateDescriptions(actor_description,critic_description)||
         !Actor.Create(actor_description)||!Critic1.Create(critic_description)||!Critic2.Create(critic_description)||
         !TargetActor.Create(actor_description)||!TargetCritic1.Create(critic_description)||
         !TargetCritic2.Create(critic_description))
        {
         ReturnFalse
        }
     }

   if(SelectiveOpenCL&&!SelectiveRuntime.Build())
     {
      ReturnFalse
     }
   if(!loaded&&(!TargetActor.WeightsUpdate(GetPointer(Actor),1.0f)||
               !TargetCritic1.WeightsUpdate(GetPointer(Critic1),1.0f)||
               !TargetCritic2.WeightsUpdate(GetPointer(Critic2),1.0f)))
     {
      ReturnFalse
     }

   PrintFormat("TD3 OpenCL init: mode=%s elapsed_ms=%I64u context=%d program=%d",
               SelectiveOpenCL?"selective":"full",GetTickCount64()-started,
               OpenCL.GetContext(),OpenCL.GetProgram());
//---
   Actor.getResults(Result);
   if(Result.Total()!=NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)",NActions,Result.Total());

      ReturnFalse
     }
//---
   Actor.GetLayerOutput(0,Result);
   if(Result.Total()!=(HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Actor doesn't match state description (%d <> %d)",Result.Total(),(HistoryBars * BarDescr));

      ReturnFalse
     }
//---
   Actor.GetLayerOutput(LatentLayer,Result);
   int latent_state=Result.Total();
   Critic1.GetLayerOutput(0,Result);
   if(Result.Total()!=latent_state)
     {
      PrintFormat("Input size of Critic doesn't match latent state Actor (%d <> %d)",Result.Total(),latent_state);

      ReturnFalse
     }
//---
   Gradient.BufferInit(AccountDescr,0);
//---
   return(true);
  }

#ifndef TD3_OPENCL_BENCHMARK
//+------------------------------------------------------------------+
//| OnInit - Initialize the training expert                          |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
   if(!LoadTotalBase()||!InitializeStudyModels(true))
      return(INIT_FAILED);

   if(!EventChartCustom(ChartID(),1,0,0,"Init"))
     {
      PrintFormat("Error of create study event: %d",GetLastError());

      return(INIT_FAILED);
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnDeinit - expert deinitialization function                      |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(reason==REASON_INITFAILED)
     {
      DeleteObj(Result)
      return;
     }
//---
   TargetActor.WeightsUpdate(GetPointer(Actor),Tau);
   TargetCritic1.WeightsUpdate(GetPointer(Critic1),Tau);
   TargetCritic2.WeightsUpdate(GetPointer(Critic2),Tau);
   TargetActor.Save(FileName+"Act.nnw",0,0,0,TimeCurrent(),true);
   TargetCritic1.Save(FileName+"Crt1.nnw",TargetCritic1.getRecentAverageError(),0,0,TimeCurrent(),true);
   TargetCritic2.Save(FileName+"Crt2.nnw",TargetCritic2.getRecentAverageError(),0,0,TimeCurrent(),true);
   delete Result;
  }
//+------------------------------------------------------------------+
//| OnChartEvent - chartEvent function                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id==1001)
     {
      Train();
      ExpertRemove();
     }
  }
#endif // TD3_OPENCL_BENCHMARK

//+------------------------------------------------------------------+
//| ForwardCritic - Use actor output and latent gradient on device   |
//+------------------------------------------------------------------+
bool ForwardCritic(CNet &critic,CNet &actor,const bool policy=false)
  {
   CNeuronBaseOCL *latent=actor.Layer(LatentLayer);
   CNeuronBaseOCL *action=actor.Layer(-1);
   CNeuronBaseOCL *previous=critic.Layer(0);
   if(!latent||!action||!previous||latent.Neurons()!=previous.Neurons()||
      actor.GetOpenCL()!=critic.GetOpenCL())
     {
      ReturnFalse
     }

   CNeuronBaseOCL *input_layer=previous;
   CBufferFloat   *output=input_layer.getOutput();
   CBufferFloat   *gradient=input_layer.getGradient();
   CBufferFloat   *latent_output=latent.getOutput();
   CBufferFloat   *latent_gradient=latent.getGradient();
   if(!output||!gradient||!latent_output||!latent_gradient||
      output.Total()!=latent_output.Total()||gradient.Total()!=latent_gradient.Total()||
      !input_layer.SetOutput(latent_output,false))
     {
      ReturnFalse
     }

   const ENUM_ACTIVATION activation=(ENUM_ACTIVATION)input_layer.Activation();
   input_layer.SetActivationFunction((ENUM_ACTIVATION)latent.Activation());
   bool result=(!policy||input_layer.SetGradient(latent_gradient,false));

   for(int index=1;result;index++)
     {
      CNeuronBaseOCL *layer=critic.Layer(index);
      if(!layer)
         break;
      result=layer.FeedForwardSecond(previous,action);
      previous=layer;
     }

   if(result&&policy)
     {
      critic.getResults(Result);
      const float forecast=Result[0];
      if(!Result.Update(0,(forecast>0?forecast+PoliticAdjust:PoliticAdjust)))
         result=false;
      else
        {
         critic.TrainMode(false);
         result=critic.backProp(Result,GetPointer(actor));
         critic.TrainMode(true);
        }
     }

   const bool gradient_restored=input_layer.SetGradient(gradient,false);
   const bool output_restored=input_layer.SetOutput(output,false);
   input_layer.SetActivationFunction(activation);

   return(result&&gradient_restored&&output_restored);
  }

//+------------------------------------------------------------------+
//| Train - Run the TD3 training loop                                |
//+------------------------------------------------------------------+
bool Train(void)
  {
   int  total_tr=ArraySize(Buffer);
   uint ticks=GetTickCount();
//---
   for(int iter=0; (iter<Iterations&&!IsStopped()); iter ++)
     {
      int tr=(int)((MathRand() / 32767.0) * (total_tr-1));
      int i=(int)((MathRand() * MathRand() / MathPow(32767,2)) * (Buffer[tr].Total-2));

   //--- target
      State.AssignArray(Buffer[tr].States[i+1].state);
      float PrevBalance=Buffer[tr].States[i].account[0];
      float PrevEquity=Buffer[tr].States[i].account[1];
      Account.Clear();
      Account.Add((Buffer[tr].States[i+1].account[0]-PrevBalance) / PrevBalance);
      Account.Add(Buffer[tr].States[i+1].account[1] / PrevBalance);
      Account.Add((Buffer[tr].States[i+1].account[1]-PrevEquity) / PrevEquity);
      Account.Add(Buffer[tr].States[i+1].account[2]);
      Account.Add(Buffer[tr].States[i+1].account[3]);
      Account.Add(Buffer[tr].States[i+1].account[4] / PrevBalance);
      Account.Add(Buffer[tr].States[i+1].account[5] / PrevBalance);
      Account.Add(Buffer[tr].States[i+1].account[6] / PrevBalance);
      //---
      if(Account.GetIndex()>=0)
         Account.BufferWrite();
      if(!TargetActor.feedForward(GetPointer(State),1,false,GetPointer(Account)))
        {
         PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
         ReturnFalse
        }
      //---
      if(!ForwardCritic(TargetCritic1,TargetActor)||!ForwardCritic(TargetCritic2,TargetActor))
        {
         PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
         ReturnFalse
        }
      TargetCritic1.getResults(Result);
      float reward=Result[0];
      TargetCritic2.getResults(Result);
      reward=DiscFactor * MathMin(reward,Result[0])+(Buffer[tr].Revards[i]-Buffer[tr].Revards[i+1]);

   //--- Q-function study
      State.AssignArray(Buffer[tr].States[i].state);
      PrevBalance=Buffer[tr].States[MathMax(i-1,0)].account[0];
      PrevEquity=Buffer[tr].States[MathMax(i-1,0)].account[1];
      Account.Clear();
      Account.Add((Buffer[tr].States[i].account[0]-PrevBalance) / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      Account.Add((Buffer[tr].States[i].account[1]-PrevEquity) / PrevEquity);
      Account.Add(Buffer[tr].States[i].account[2]);
      Account.Add(Buffer[tr].States[i].account[3]);
      Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      //---
      if(Account.GetIndex()>=0)
         Account.BufferWrite();
      //---
      if(!Actor.feedForward(GetPointer(State),1,false,GetPointer(Account)))
        {
         PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
         ReturnFalse
        }
      Actor.GetLayerOutput(LatentLayer,Result);
      Actions.AssignArray(Buffer[tr].States[i].action);
      if(Actions.GetIndex()>=0)
        Actions.BufferWrite();
      //---
      if(!Critic1.feedForward(Result,1,false,GetPointer(Actions))||
         !Critic2.feedForward(Result,1,false,GetPointer(Actions)))
        {
         PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
         ReturnFalse
        }
      Result.Clear();
      Result.Add(reward);
      if(!Critic1.backProp(Result,GetPointer(Actions),GetPointer(Gradient))||
         !Critic2.backProp(Result,GetPointer(Actions),GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
         ReturnFalse
        }

   //--- policy study
      if(iter>0&&(iter % UpdatePolicy)==0)
        {
         tr=(int)((MathRand() / 32767.0) * (total_tr-1));
         i=(int)((MathRand() * MathRand() / MathPow(32767,2)) * (Buffer[tr].Total-2));
         State.AssignArray(Buffer[tr].States[i].state);
         PrevBalance=Buffer[tr].States[MathMax(i-1,0)].account[0];
         PrevEquity=Buffer[tr].States[MathMax(i-1,0)].account[1];
         Account.Clear();
         Account.Add((Buffer[tr].States[i].account[0]-PrevBalance) / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
         Account.Add((Buffer[tr].States[i].account[1]-PrevEquity) / PrevEquity);
         Account.Add(Buffer[tr].States[i].account[2]);
         Account.Add(Buffer[tr].States[i].account[3]);
         Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
         //---
         if(Account.GetIndex()>=0)
            Account.BufferWrite();
         //---
         if(!Actor.feedForward(GetPointer(State),1,false,GetPointer(Account)))
           {
            PrintFormat("%s -> %d",__FUNCTION__,__LINE__);
            ReturnFalse
           }
         //---
         if(!ForwardCritic(Critic1,Actor,true)||
            !Actor.backPropGradient(GetPointer(Account),GetPointer(Gradient)))
           {
            ReturnFalse
           }
        }

   //--- update Target Nets
      if(iter>0&&(iter % UpdateTargets)==0)
        {
         if(!TargetActor.WeightsUpdate(GetPointer(Actor),Tau)||
            !TargetCritic1.WeightsUpdate(GetPointer(Critic1),Tau)||
            !TargetCritic2.WeightsUpdate(GetPointer(Critic2),Tau))
           {
            ReturnFalse
           }
        }
      //---
      if(GetTickCount()-ticks>500)
        {
         string str=StringFormat("%-15s %5.2f%% -> Error %15.8f\n","Critic1",iter * 100.0 / (double)(Iterations),Critic1.getRecentAverageError());
         str += StringFormat("%-15s %5.2f%% -> Error %15.8f\n","Critic2",iter * 100.0 / (double)(Iterations),Critic2.getRecentAverageError());
         Comment(str);
         ticks=GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f",__FUNCTION__,__LINE__,"Critic1",Critic1.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f",__FUNCTION__,__LINE__,"Critic2",Critic2.getRecentAverageError());
   return(!IsStopped());
//---
  }
//+------------------------------------------------------------------+

