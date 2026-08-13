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
input int                  Iterations     = 10000;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 MFTEncoder;
CNet                 MFTEndpoints;
CNet                 MFTProbability;
CNet                 StateEncoder;
CNet                 EndpointEncoder;
CNet                 Actor;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bGradient;
CBufferFloat         bProbs;
CBufferFloat         *Result;
vector<float>        check;
vector<float>        STD_Actor;
vector<float>        STD_Goal;
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
   if(!MFTEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !MFTEndpoints.Load(FileName + "Endp.nnw", temp, temp, temp, dtStudied, true) ||
      !MFTProbability.Load(FileName + "Prob.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *endpoint = new CArrayObj();
      CArrayObj *prob = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, endpoint, prob))
        {
         delete endpoint;
         delete prob;
         delete encoder;
         return INIT_FAILED;
        }
      if(!MFTEncoder.Create(encoder) ||
         !MFTEndpoints.Create(endpoint) ||
         !MFTProbability.Create(prob))
        {
         delete endpoint;
         delete prob;
         delete encoder;
         return INIT_FAILED;
        }
      delete endpoint;
      delete prob;
      delete encoder;
     }
//---
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !EndpointEncoder.Load(FileName + "EndEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *endpoint = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, endpoint, encoder))
        {
         delete actor;
         delete endpoint;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !StateEncoder.Create(encoder) || !EndpointEncoder.Create(endpoint))
        {
         delete actor;
         delete endpoint;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete endpoint;
      delete encoder;
      //---
     }
//---
   OpenCL = Actor.GetOpenCL();
   StateEncoder.SetOpenCL(OpenCL);
   EndpointEncoder.SetOpenCL(OpenCL);
   MFTEncoder.SetOpenCL(OpenCL);
   MFTEndpoints.SetOpenCL(OpenCL);
   MFTProbability.SetOpenCL(OpenCL);
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   MFTEndpoints.getResults(Result);
   if(Result.Total() != 3 * NForecast)
     {
      PrintFormat("The scope of the Endpoints does not match forecast endpoints (%d <> %d)", 3 * NForecast, Result.Total());
      return INIT_FAILED;
     }
//---
   MFTEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   if(!bGradient.BufferInit(MathMax(AccountDescr, NForecast), 0) ||
      !bGradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
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
      Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      StateEncoder.Save(FileName + "StEnc.nnw", 0, 0, 0, TimeCurrent(), true);
      EndpointEncoder.Save(FileName + "EndEnc.nnw", 0, 0, 0, TimeCurrent(), true);
      MFTEncoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      MFTEndpoints.Save(FileName + "Endp.nnw", 0, 0, 0, TimeCurrent(), true);
      MFTProbability.Save(FileName + "Prob.nnw", 0, 0, 0, TimeCurrent(), true);
     }
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
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target;
   matrix<float> targets, temp_m;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 48;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
      MFTEncoder.Clear();
      MFTEndpoints.Clear();
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      for(int i = state; i < end; i++)
        {
         bState.AssignArray(Buffer[tr].States[i].state);
         //--- Trajectory
         if(!MFTEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!MFTEndpoints.feedForward((CNet*)GetPointer(MFTEncoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!MFTProbability.feedForward((CNet*)GetPointer(MFTEncoder), -1, (CNet*)GetPointer(MFTEndpoints)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         targets = matrix<float>::Zeros(PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + 1 + t].state);
            if(target.Size() > BarDescr)
              {
               matrix<float> temp(1, target.Size());
               temp.Row(target, 0);
               temp.Reshape(target.Size() / BarDescr, BarDescr);
               temp.Resize(temp.Rows(), 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, t);
           }
         target = targets.Col(0).CumSum();
         targets.Col(target, 0);
         targets.Col(target + targets.Col(1), 1);
         targets.Col(target + targets.Col(2), 2);
         int direct = (Buffer[tr].States[i].state[8] >= Buffer[tr].States[i].state[7] ? 1 : -1);
         ulong extr = (direct > 0 ? target.ArgMax() : target.ArgMin());
         if(extr == 0)
           {
            direct = -direct;
            extr = (direct > 0 ? target.ArgMax() : target.ArgMin());
           }
         targets.Resize(extr + 1, 3);
         if(direct >= 0)
           {
            target = targets.Max(AXIS_HORZ);
            target[2] = targets.Col(2).Min();
           }
         else
           {
            target = targets.Min(AXIS_HORZ);
            target[1] = targets.Col(1).Max();
           }
         //---
         MFTEndpoints.getResults(result);
         targets.Reshape(1, result.Size());
         targets.Row(result, 0);
         targets.Reshape(NForecast, 3);
         temp_m = targets;
         for(int i = 0; i < 3; i++)
            temp_m.Col(temp_m.Col(i) - target[i], i);
         temp_m = MathPow(temp_m, 2.0f);
         ulong pos = temp_m.Sum(AXIS_VERT).ArgMin();
         targets.Row(target, pos);
         Result.AssignArray(targets);
         //---
         if(!MFTEndpoints.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!MFTEncoder.backPropGradient((CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         bProbs.AssignArray(vector<float>::Zeros(NForecast));
         bProbs.Update((int)pos, 1);
         bProbs.BufferWrite();
         if(!MFTProbability.backProp(GetPointer(bProbs), GetPointer(MFTEndpoints)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Policy
         float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         bAccount.Clear();
         bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
         bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
         bAccount.Add(Buffer[tr].States[i].account[2]);
         bAccount.Add(Buffer[tr].States[i].account[3]);
         bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
         double time = (double)Buffer[tr].States[i].account[7];
         double x = time / (double)(D'2024.01.01' - D'2023.01.01');
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_MN1);
         bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_W1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_D1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(bAccount.GetIndex() >= 0)
            bAccount.BufferWrite();
         //--- State embedding
         if(!StateEncoder.feedForward((CNet *)GetPointer(MFTEncoder), -1, (CBufferFloat*)GetPointer(bAccount)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Endpoint embedding
         if(!EndpointEncoder.feedForward((CNet *)GetPointer(MFTEndpoints), -1, (CNet*)GetPointer(MFTProbability)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //--- Actor
         if(!Actor.feedForward((CNet *)GetPointer(StateEncoder), -1, (CNet*)GetPointer(EndpointEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         result = vector<float>::Zeros(NActions);
         double value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
         double risk = AccountInfoDouble(ACCOUNT_EQUITY) * 0.01;
         if(direct > 0)
           {
            if(Buffer[tr].States[i].state[4] > 30 &&
               Buffer[tr].States[i].state[5] > -100
              )
              {
               float tp = float(target[1] / _Point / MaxTP);
               result[1] = tp;
               int sl = int(MathMax(MathMax(target[1] / 3, -target[2]) / _Point, MaxSL / 10));
               result[2] = float(sl) / MaxSL;
               result[0] = float(MathMax(risk / (value * sl), 0.01)) + FLT_EPSILON;
              }
           }
         else
           {
            if(Buffer[tr].States[i].state[4] < 70 &&
               Buffer[tr].States[i].state[5] < 100
              )
              {
               float tp = float((-target[2]) / _Point / MaxTP);
               result[4] = tp;
               int sl = int(MathMax(MathMax((-target[2]) / 3, target[1]) / _Point, MaxSL / 10));
               result[5] = float(sl) / MaxSL;
               result[3] = float(MathMax(risk / (value * sl), 0.01)) + FLT_EPSILON;
              }
           }
         Result.AssignArray(result);
         if(!Actor.backProp(Result, (CNet *)GetPointer(EndpointEncoder)) ||
            !StateEncoder.backPropGradient(GetPointer(bAccount), (CBufferFloat *)GetPointer(bGradient)) ||
            !EndpointEncoder.backPropGradient((CNet*)GetPointer(MFTProbability))
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         if(!MFTEncoder.backPropGradient((CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Endpoints", percent, MFTEndpoints.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Probability", percent, MFTProbability.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Endpoints", MFTEndpoints.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Probability", MFTProbability.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+

