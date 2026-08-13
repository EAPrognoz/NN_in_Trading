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
input int                  Iterations     = 100000;
input float                Tau            = 0.001f;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 TargetEncoder;
CNet                 Actor;
CNet                 Critic1;
CNet                 Critic2;
CNet                 Convolution;
CNet                 Descriminator;
CNet                 SkillProject;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         Account;
CBufferFloat         TargetState;
CBufferFloat         TargetAccount;
CBufferFloat         Actions;
CBufferFloat         Gradient;
CBufferFloat         Skills;
CBufferFloat         *Result;
vector<float>        check;
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
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic1.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic2.Load(FileName + "Crt2.nnw", temp, temp, temp, dtStudied, true) ||
      !Descriminator.Load(FileName + "Des.nnw", temp, temp, temp, dtStudied, true) ||
      !SkillProject.Load(FileName + "Skp.nnw", temp, temp, temp, dtStudied, true) ||
      !Convolution.Load(FileName + "CNN.nnw", temp, temp, temp, dtStudied, true) ||
      !TargetEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *descrim = new CArrayObj();
      CArrayObj *convolution = new CArrayObj();
      CArrayObj *skill_poject = new CArrayObj();
      if(!CreateDescriptions(encoder,actor, critic, convolution,descrim,skill_poject))
        {
         delete encoder;
         delete actor;
         delete critic;
         delete descrim;
         delete convolution;
         delete skill_poject;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) || !Actor.Create(actor) ||
         !Critic1.Create(critic) || !Critic2.Create(critic) ||
         !Descriminator.Create(descrim) || !SkillProject.Create(skill_poject) ||
         !Convolution.Create(convolution))
        {
         delete encoder;
         delete actor;
         delete critic;
         delete descrim;
         delete convolution;
         delete skill_poject;
         return INIT_FAILED;
        }
      if(!TargetEncoder.Create(encoder))
        {
         delete encoder;
         delete actor;
         delete critic;
         delete descrim;
         delete convolution;
         delete skill_poject;
         return INIT_FAILED;
        }
      delete encoder;
      delete actor;
      delete critic;
      delete descrim;
      delete convolution;
      delete skill_poject;
      //---
      TargetEncoder.WeightsUpdate(GetPointer(Encoder), 1.0f);
     }
//---
   OpenCL = Actor.GetOpenCL();
   Encoder.SetOpenCL(OpenCL);
   Critic1.SetOpenCL(OpenCL);
   Critic2.SetOpenCL(OpenCL);
   TargetEncoder.SetOpenCL(OpenCL);
   Descriminator.SetOpenCL(OpenCL);
   SkillProject.SetOpenCL(OpenCL);
   Convolution.SetOpenCL(OpenCL);
//---
   vector<float> ActorResult;
   Actor.getResults(ActorResult);
   if(ActorResult.Size() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of State Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   vector<float> EncoderResults;
   Actor.GetLayerOutput(0,Result);
   Encoder.getResults(EncoderResults);
   if(Result.Total() != int(EncoderResults.Size()))
     {
      PrintFormat("Input size of Actor doesn't match Encoder outputs (%d <> %d)", Result.Total(), EncoderResults.Size());
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(LatentLayer, Result);
   int latent_state = Result.Total();
   Critic1.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Critic doesn't match latent state Actor (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Gradient.BufferInit(MathMax(AccountDescr,NSkills), 0);
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
   TargetEncoder.WeightsUpdate(GetPointer(Encoder), Tau);
   Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
   TargetEncoder.Save(FileName + "Enc.nnw", Critic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   Critic1.Save(FileName + "Crt1.nnw", Critic1.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   Critic2.Save(FileName + "Crt2.nnw", Critic2.getRecentAverageError(), 0, 0, TimeCurrent(), true);
   Convolution.Save(FileName + "CNN.nnw", 0, 0, 0, TimeCurrent(), true);
   Descriminator.Save(FileName + "Des.nnw", 0, 0, 0, TimeCurrent(), true);
   SkillProject.Save(FileName + "Skp.nnw", 0, 0, 0, TimeCurrent(), true);
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
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
//---
   int total_states = Buffer[0].Total - 1;
   for(int i = 1; i < total_tr; i++)
      total_states += Buffer[i].Total - 1;
   vector<float> temp;
   Convolution.getResults(temp);
   matrix<float> state_embedding = matrix<float>::Zeros(total_states,temp.Size());
   int state = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      for(int st = 0; st < Buffer[tr].Total - 1; st++)
        {
         State.AssignArray(Buffer[tr].States[st].state);
         float PrevBalance = Buffer[tr].States[MathMax(st,0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(st,0)].account[1];
         State.Add((Buffer[tr].States[st].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr].States[st].account[1] / PrevBalance);
         State.Add((Buffer[tr].States[st].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr].States[st].account[2]);
         State.Add(Buffer[tr].States[st].account[3]);
         State.Add(Buffer[tr].States[st].account[4] / PrevBalance);
         State.Add(Buffer[tr].States[st].account[5] / PrevBalance);
         State.Add(Buffer[tr].States[st].account[6] / PrevBalance);
         double x = (double)Buffer[tr].States[st].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         //---
         State.AddArray(Buffer[tr].States[st + 1].state);
         State.Add((Buffer[tr].States[st + 1].account[0] - PrevBalance) / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[1] / PrevBalance);
         State.Add((Buffer[tr].States[st + 1].account[1] - PrevEquity) / PrevEquity);
         State.Add(Buffer[tr].States[st + 1].account[2]);
         State.Add(Buffer[tr].States[st + 1].account[3]);
         State.Add(Buffer[tr].States[st + 1].account[4] / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[5] / PrevBalance);
         State.Add(Buffer[tr].States[st + 1].account[6] / PrevBalance);
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st + 1].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(!Convolution.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            return;
           }
         Convolution.getResults(temp);
         state_embedding.Row(temp,state);
         state++;
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %6.2f%%", "Embedding ", state * 100.0 / (double)(total_states));
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   if(state != total_states)
     {
      state_embedding.Reshape(state,state_embedding.Cols());
      total_states = state;
     }
//---
   vector<float> reward = vector<float>::Zeros(NRewards);
   vector<float> rewards1 = reward, rewards2 = reward;
   int bar = (HistoryBars-1) * BarDescr;
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)((MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      if(i < 0)
        {
         iter--;
         continue;
        }
      //--- State
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
      double x = (double)Buffer[tr].States[i].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_MN1);
      Account.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_W1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = (double)Buffer[tr].States[i].account[7] / (double)PeriodSeconds(PERIOD_D1);
      Account.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      if(Account.GetIndex() >= 0)
         Account.BufferWrite();
      //--- Skills
      vector<float> skills = vector<float>::Zeros(NSkills);
      //for(int sk = 0; sk < NSkills; sk++)
      //   skills[sk] = (float)((double)MathRand() / 32767.0);
      //skills.Activation(skills,AF_SOFTMAX);
      skills[int((double)MathRand() / 32768.0 * NSkills)] = 1;
      Skills.AssignArray(skills);
      if(Skills.GetIndex() >= 0 && !Skills.BufferWrite())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Encoder State
      if(!Encoder.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Actor
      if(!Actor.feedForward(GetPointer(Encoder), -1, GetPointer(Skills)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Next State
      TargetState.AssignArray(Buffer[tr].States[i + 1].state);
      double cl_op = Buffer[tr].States[i + 1].state[bar];
      // Implements SymbolInfoDouble.
      double prof_1l = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT) * cl_op /
                       SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      Actor.getResults(Result);
      vector<float> forecast = ForecastAccount(Buffer[tr].States[i].account,Result,prof_1l,Buffer[tr].States[i + 1].account[7]);
      TargetAccount.AssignArray(forecast);
      if(TargetAccount.GetIndex() >= 0 && !TargetAccount.BufferWrite())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      if(!TargetEncoder.feedForward(GetPointer(TargetState), 1, false, GetPointer(TargetAccount)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //--- Descriminator
      if(!Descriminator.feedForward(GetPointer(Encoder),-1,GetPointer(TargetEncoder),-1) ||
         !SkillProject.feedForward(GetPointer(Skills),1,false,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Descriminator.getResults(rewards1);
      SkillProject.getResults(rewards2);
      float norm1 = rewards1.Norm(VECTOR_NORM_P,2);
      float norm2 = rewards2.Norm(VECTOR_NORM_P,2);
      reward[0] = 0;//(rewards1 / norm1).Dot(rewards2 / norm2);
      Result.AssignArray(rewards2);
      if(!Descriminator.backProp(Result,GetPointer(TargetEncoder)) ||
         !Encoder.backPropGradient(GetPointer(Account),GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Result.AssignArray(rewards1);
      if(!SkillProject.backProp(Result,(CNet *)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      //---
      if(forecast[3] == 0.0f && forecast[4] == 0.f)
         reward[0] -= Buffer[tr].States[i + 1].state[bar + 6] / PrevBalance;
      //---
      State.AddArray(GetPointer(Account));
      State.AddArray(GetPointer(TargetState));
      State.AddArray(GetPointer(TargetAccount));
      if(!Convolution.feedForward(GetPointer(State),1,false,(CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      Convolution.getResults(rewards1);
      reward[0] += KNNReward(7,rewards1,state_embedding);
      Result.AssignArray(reward);
      //---
      if(!Critic1.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor),-1) ||
         !Critic2.feedForward(GetPointer(Actor), LatentLayer, GetPointer(Actor),-1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         break;
        }
      if(Critic1.getRecentAverageError() <= Critic2.getRecentAverageError())
        {
         if(!Critic1.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Skills), GetPointer(Gradient), -1) ||
            !Critic2.backProp(Result, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
        }
      else
        {
         if(!Critic2.backProp(Result, GetPointer(Actor)) ||
            !Actor.backPropGradient(GetPointer(Skills), GetPointer(Gradient), -1) ||
            !Critic1.backProp(Result, GetPointer(Actor)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
        }
      //--- Update Target Nets
      TargetEncoder.WeightsUpdate(GetPointer(Encoder), Tau);
      //---
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Critic1", iter * 100.0 / (double)(Iterations), Critic1.getRecentAverageError());
         str += StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Critic2", iter * 100.0 / (double)(Iterations), Critic2.getRecentAverageError());
         str += StringFormat("%-20s %5.2f%% -> Error %15.8f\n", "Descriminator", iter * 100.0 / (double)(Iterations), Descriminator.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Critic1", Critic1.getRecentAverageError());
   PrintFormat("%s -> %d -> %-20s %10.7f", __FUNCTION__, __LINE__, "Critic2", Critic2.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Implements KNNReward.                                          |
//+------------------------------------------------------------------+
float KNNReward(ulong k, vector<float> &embedding, matrix<float> &state_embedding)
  {
   if(embedding.Size() != state_embedding.Cols())
     {
      PrintFormat("%s -> %d Inconsistent embedding size", __FUNCTION__, __LINE__);
      return (0);
     }
//---
   ulong size = embedding.Size();
   ulong states = state_embedding.Rows();
   matrix<float> temp = matrix<float>::Zeros(states,size);
//---
   for(ulong i = 0; i < size; i++)
      temp.Col(MathPow(state_embedding.Col(i) - embedding[i],2.0f),i);
   vector<float> dist = MathSqrt(temp.Sum(1));
   vector<float> min_dist = vector<float>::Zeros(k);
   for(ulong i = 0; i < k; i++)
     {
      ulong pos = dist.ArgMin();
      min_dist[i] = dist[pos];
      dist[pos] = FLT_MAX;
     }
//---
   vector<float> ri = MathLog(min_dist + 1.0f);
//---
   float result = ri.Mean();
//---
   return (result);
  }
//+------------------------------------------------------------------+

