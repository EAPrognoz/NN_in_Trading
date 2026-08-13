//+------------------------------------------------------------------+
//| Research.mq5 |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input double               MinProfit   =  10;
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
input int                  Agent = 1;
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
SState               sState;
STrajectory          Base;
STrajectory          Buffer[];
STrajectory          Frame[1];
CNet                 Actor;
CNet                 Critic;
CNet                 Convolution;
//---
float                dError;
datetime             dtStudied;
//---
CSymbolInfo          Symb;
CTrade               Trade;
//---
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
CBufferFloat         bState;
CBufferFloat         bAccount;
CBufferFloat         bActions;
CBufferFloat         bGradient;
CBufferFloat         *Result;
vector<float>        check;
double               PrevBalance = 0;
double               PrevEquity = 0;
bool                 BaseLoaded;
matrix<float>        state_embeddings;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
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
   if(!RSI.BufferResize(HistoryBars) || !CCI.BufferResize(HistoryBars) ||
      !ATR.BufferResize(HistoryBars) || !MACD.BufferResize(HistoryBars))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//--- load models
   float temp;
   if(!Actor.Load(StringFormat("%sAct%d.nnw", FileName, Agent), temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      if(!CreateDescriptions(actor, critic, critic))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor))
        {
         delete actor;
         delete critic;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      //---
     }
//---
   if(!Critic.Load(FileName + "Crt1.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new Critic and Encoder models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *convolution = new CArrayObj();
      if(!CreateDescriptions(actor, critic, convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      if(!Critic.Create(critic))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete convolution;
      //---
     }
//---
   if(!Convolution.Load(FileName + "CNN.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new Critic and Encoder models");
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *convolution = new CArrayObj();
      if(!CreateDescriptions(actor, critic, convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      if(!Convolution.Create(convolution))
        {
         delete actor;
         delete critic;
         delete convolution;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete convolution;
      //---
     }
//---
   Critic.SetOpenCL(Actor.GetOpenCL());
   Convolution.SetOpenCL(Actor.GetOpenCL());
   Critic.TrainMode(false);
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
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   BaseLoaded = false;
   bGradient.BufferInit(MathMax(AccountDescr, NActions), 0);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   ResetLastError();
   if(!Actor.Save(StringFormat("%sActEx%d.nnw", FileName, Agent), 0, 0, 0, TimeCurrent(), true))
      PrintFormat("Error of saving Agent %d: %d", Agent, GetLastError());
   delete Result;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!IsNewBar())
      return;
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), HistoryBars, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
//---
   float atr = 0;
   for(int b = 0; b < (int)HistoryBars; b++)
     {
      float open = (float)Rates[b].open;
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      int shift = b * BarDescr;
      sState.state[shift] = (float)(Rates[b].close - open);
      sState.state[shift + 1] = (float)(Rates[b].high - open);
      sState.state[shift + 2] = (float)(Rates[b].low - open);
      sState.state[shift + 3] = (float)(Rates[b].tick_volume / 1000.0f);
      sState.state[shift + 4] = rsi;
      sState.state[shift + 5] = cci;
      sState.state[shift + 6] = atr;
      sState.state[shift + 7] = macd;
      sState.state[shift + 8] = sign;
     }
   bState.AssignArray(sState.state);
//---
   sState.account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   sState.account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   double position_discount = 0;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
   int total = PositionsTotal();
   datetime current = TimeCurrent();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      double profit = PositionGetDouble(POSITION_PROFIT);
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += profit;
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += profit;
            break;
        }
      position_discount += profit - (current - PositionGetInteger(POSITION_TIME)) * multiplyer * MathAbs(profit);
     }
   sState.account[2] = (float)buy_value;
   sState.account[3] = (float)sell_value;
   sState.account[4] = (float)buy_profit;
   sState.account[5] = (float)sell_profit;
   sState.account[6] = (float)position_discount;
   sState.account[7] = (float)Rates[0].time;
//---
   bAccount.Clear();
   bAccount.Add((float)((sState.account[0] - PrevBalance) / PrevBalance));
   bAccount.Add((float)(sState.account[1] / PrevBalance));
   bAccount.Add((float)((sState.account[1] - PrevEquity) / PrevEquity));
   bAccount.Add(sState.account[2]);
   bAccount.Add(sState.account[3]);
   bAccount.Add((float)(sState.account[4] / PrevBalance));
   bAccount.Add((float)(sState.account[5] / PrevBalance));
   bAccount.Add((float)(sState.account[6] / PrevBalance));
   double x = (double)Rates[0].time / (double)(D'2024.01.01' - D'2023.01.01');
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_MN1);
   bAccount.Add((float)MathCos(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_W1);
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_D1);
   bAccount.Add((float)MathSin(2.0 * M_PI * x));
//---
   if(bAccount.GetIndex() >= 0)
      if(!bAccount.BufferWrite())
         return;
//---
   if(!Actor.feedForward(GetPointer(bState), 1, false, GetPointer(bAccount)))
      return;
//---
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
//---
   vector<float> temp;
   Actor.getResults(temp);
//---
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = MathMax(Symb.StopsLevel(), 1) * Symb.Point();
   if(temp[0] >= temp[3])
     {
      temp[0] -= temp[3];
      temp[3] = 0;
     }
   else
     {
      temp[3] -= temp[0];
      temp[0] = 0;
     }
//--- buy control
   if(temp[0] < min_lot || (temp[1] * MaxTP * Symb.Point()) <= stops || (temp[2] * MaxSL * Symb.Point()) <= stops)
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(temp[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = NormalizeDouble(Symb.Ask() + temp[1] * MaxTP * Symb.Point(), Symb.Digits());
      double buy_sl = NormalizeDouble(Symb.Ask() - temp[2] * MaxSL * Symb.Point(), Symb.Digits());
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, buy_sl, buy_tp);
      if(buy_value != buy_lot)
        {
         if(buy_value > buy_lot)
            ClosePartial(POSITION_TYPE_BUY, buy_value - buy_lot);
         else
            Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp);
        }
     }
//--- sell control
   if(temp[3] < min_lot || (temp[4] * MaxTP * Symb.Point()) <= stops || (temp[5] * MaxSL * Symb.Point()) <= stops)
     {
      if(sell_value > 0)
         CloseByDirection(POSITION_TYPE_SELL);
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(temp[3] - min_lot) / step_lot) * step_lot;;
      double sell_tp = NormalizeDouble(Symb.Bid() - temp[4] * MaxTP * Symb.Point(), Symb.Digits());
      double sell_sl = NormalizeDouble(Symb.Bid() + temp[5] * MaxSL * Symb.Point(), Symb.Digits());
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sell_sl, sell_tp);
      if(sell_value != sell_lot)
        {
         if(sell_value > sell_lot)
            ClosePartial(POSITION_TYPE_SELL, sell_value - sell_lot);
         else
            Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp);
        }
     }
//---
   sState.rewards[0] = bAccount[0];
   sState.rewards[1] = 1.0f - bAccount[1];
   if((buy_value + sell_value) == 0)
      sState.rewards[2] -= (float)(atr / PrevBalance);
   else
      sState.rewards[2] = 0;
   for(ulong i = 0; i < NActions; i++)
      sState.action[i] = temp[i];
   sState.rewards[3] = 0;
   sState.rewards[4] = 0;
   if(!Base.Add(sState))
      ExpertRemove();
//---
   bState.AddArray(GetPointer(bAccount));
   bState.AddArray(temp);
   bActions.AssignArray(temp);
   if(!Convolution.feedForward(GetPointer(bState), 1, false, (CBufferFloat*)NULL))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return;
     }
   Convolution.getResults(temp);
//---
   if(!BaseLoaded)
      state_embeddings = CreateEmbeddings();
   BaseLoaded = true;
//---
   ulong total_states = state_embeddings.Rows();
   if(total_states <= 0)
     {
      ResetLastError();
      if(!state_embeddings.Resize(total_states + 1, state_embeddings.Cols()) ||
         !state_embeddings.Row(temp, total_states))
         PrintFormat("%s -> %d: Error of adding new embedding %", __FUNCTION__, __LINE__, GetLastError());
      return;
     }
//---
   vector<float> rewards = ResearchReward(Quant, temp, state_embeddings);
   ResetLastError();
   if(!state_embeddings.Resize(total_states + 1, state_embeddings.Cols()) ||
      !state_embeddings.Row(temp, total_states))
      PrintFormat("%s -> %d: Error of adding new embedding %", __FUNCTION__, __LINE__, GetLastError());
//---
   Result.AssignArray(rewards);
   if(!Critic.feedForward(GetPointer(Actor), LatentLayer, GetPointer(bActions)) ||
      !Critic.backProp(Result, GetPointer(bActions), GetPointer(bGradient)) ||
      !Actor.backPropGradient(GetPointer(bAccount), GetPointer(bGradient), LatentLayer))
      PrintFormat("%s -> %d: Error of backpropagation %", __FUNCTION__, __LINE__, GetLastError());
  }
//+------------------------------------------------------------------+
//| Tester function                                                |
//+------------------------------------------------------------------+
//| | Strategy tester callback.                                      |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret = 0.0;
//---
   double profit = TesterStatistics(STAT_PROFIT);
   Frame[0] = Base;
   if(profit >= MinProfit && profit != 0)
      FrameAdd(MQLInfoString(MQL_PROGRAM_NAME), 1, profit, Frame);
//---
   return(ret);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                            |
//+------------------------------------------------------------------+
//| | Tester initialization handler.                                 |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
//---
   BaseLoaded = LoadTotalBase();
  }
//+------------------------------------------------------------------+
//| TesterPass function                                            |
//+------------------------------------------------------------------+
//| | Tester pass callback.                                          |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
//---
   ulong pass;
   string name;
   long id;
   double value;
   STrajectory array[];
   while(FrameNext(pass, name, id, value, array))
     {
      int total = ArraySize(Buffer);
      if(name != MQLInfoString(MQL_PROGRAM_NAME))
         continue;
      if(id <= 0)
         continue;
      if(total >= MaxReplayBuffer)
        {
         for(int a = 0; a < id; a++)
           {
            float min = FLT_MAX;
            int min_tr = 0;
            for(int i = 0; i < total; i++)
              {
               if(Buffer[i].Total<=0)
                {
                 min = FLT_MIN;
                 min_tr = i;
                 break;
                }
               float prof = Buffer[i].States[Buffer[i].Total - 1].account[1];
               if(prof < min)
                 {
                  min = MathMin(prof, min);
                  min_tr = i;
                 }
              }
            float prof = array[a].States[array[a].Total - 1].account[1];
            if(min <= prof)
              {
               Buffer[min_tr] = array[a];
               PrintFormat("Replace %.2f to %.2f -> bars %d", min, prof, array[a].Total);
              }
           }
        }
      else
        {
         if(ArrayResize(Buffer, total + (int)id, 10) < 0)
            return;
         ArrayCopy(Buffer, array, total, 0, (int)id);
        }
     }
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//---
   int total = ArraySize(Buffer);
   printf("total %d", MathMin(total, MaxReplayBuffer));
   Print("Saving...");
   SaveTotalBase();
   Print("Saved");
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
vector<float> ResearchReward(double quant, vector<float> &embedding, matrix<float> &state_embedding)
  {
   vector<float> result = vector<float>::Zeros(NRewards);
   if(embedding.Size() != state_embedding.Cols())
     {
      PrintFormat("%s -> %d Inconsistent embedding size", __FUNCTION__, __LINE__);
      return result;
     }
//---
   ulong size = embedding.Size();
   ulong states = state_embedding.Rows();
   ulong k = ulong(states * quant);
   matrix<float> temp = matrix<float>::Zeros(states, size);
   vector<float> min_dist = vector<float>::Zeros(k);
   matrix<float> k_embedding = matrix<float>::Zeros(k + 1, size);
   matrix<float> U, V;
   vector<float> S;
//---
   for(ulong i = 0; i < size; i++)
      temp.Col(MathAbs(state_embedding.Col(i) - embedding[i]), i);
   float alpha = temp.Max();
   if(alpha == 0)
      alpha = 1;
   vector<float> dist = MathLog(MathExp(temp / (-alpha)).Sum(1)) * (-alpha);
//---
   float max = dist.Quantile(quant);
   for(ulong i = 0, cur = 0; (i < states && cur < k); i++)
     {
      if(max < dist[i])
         continue;
      min_dist[cur] = dist[i];
      k_embedding.Row(state_embedding.Row(i), cur);
      cur++;
     }
   k_embedding.Row(embedding, k);
//---
   k_embedding.SVD(U, V, S);
   result[NRewards - 2] = S.Sum() / (MathSqrt(MathPow(k_embedding, 2.0f).Sum() * MathMax(k + 1, size)));
   result[NRewards - 1] = EntropyLatentState(Actor);
//---
   return result;
  }
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
matrix<float> CreateEmbeddings(void)
  {
   vector<float> temp;
   CBufferFloat  State;
   Convolution.getResults(temp);
   matrix<float> result = matrix<float>::Zeros(0, temp.Size());
//---
   BaseLoaded = LoadTotalBase();
   if(!BaseLoaded)
     {
      PrintFormat("%s - %d => Error of load base", __FUNCTION__, __LINE__);
      return result;
     }
//---
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
//---
   int total_states = Buffer[0].Total;
   for(int i = 1; i < total_tr; i++)
      total_states += Buffer[i].Total;
   result.Resize(total_states, temp.Size());
//---
   int state = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      for(int st = 0; st < Buffer[tr].Total; st++)
        {
         State.AssignArray(Buffer[tr].States[st].state);
         float prevBalance = Buffer[tr].States[MathMax(st - 1, 0)].account[0];
         float prevEquity = Buffer[tr].States[MathMax(st - 1, 0)].account[1];
         State.Add((Buffer[tr].States[st].account[0] - prevBalance) / prevBalance);
         State.Add(Buffer[tr].States[st].account[1] / prevBalance);
         State.Add((Buffer[tr].States[st].account[1] - prevEquity) / prevEquity);
         State.Add(Buffer[tr].States[st].account[2]);
         State.Add(Buffer[tr].States[st].account[3]);
         State.Add(Buffer[tr].States[st].account[4] / prevBalance);
         State.Add(Buffer[tr].States[st].account[5] / prevBalance);
         State.Add(Buffer[tr].States[st].account[6] / prevBalance);
         double x = (double)Buffer[tr].States[st].account[7] / (double)(D'2024.01.01' - D'2023.01.01');
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_MN1);
         State.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_W1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = (double)Buffer[tr].States[st].account[7] / (double)PeriodSeconds(PERIOD_D1);
         State.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         State.AddArray(Buffer[tr].States[st].action);
         if(!Convolution.feedForward(GetPointer(State), 1, false, NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
         Convolution.getResults(temp);
         if(!result.Row(temp, state))
            continue;
         state++;
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %6.2f%%", "Embedding ", state * 100.0 / (double)(total_states));
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
//---
   if(state != total_states)
      result.Reshape(state, result.Cols());
   ArrayFree(Buffer);
//---
   return result;
  }
//+------------------------------------------------------------------+

