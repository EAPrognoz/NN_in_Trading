#ifndef HABI_HISTORY_MQH
#define HABI_HISTORY_MQH
#include "Habi_Settings.mqh"
enum EHabiReason
  {
   HABI_OK=0,HABI_FLAT,HABI_ZERO_PROTECTION,HABI_ECONOMIC_INVALID,HABI_PROFILE_ERROR,HABI_INPUT_ERROR,HABI_SL,HABI_TP,HABI_HORIZON,HABI_ADAPTER_ERROR
  };
struct SHabiSymbol
  {
   double point,tick,lot_min,lot_step,lot_max,lot_limit;
   long stops,freeze,trade_mode;
   bool Read()
     {
      point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
      tick=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
      lot_min=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
      lot_step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
      lot_max=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
      lot_limit=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_LIMIT);
      stops=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
      freeze=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL);
      trade_mode=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE);
      return point>0 && tick>0 && lot_min>0 && lot_step>0 && lot_max>=lot_min && VolumeCap>=lot_min;
     }
  };
struct SHabiCost
  {
   datetime time;
   double buy_value,sell_value,margin_buy,margin_sell,entry_cash,exit_cash,hold_buy,hold_sell,swap_buy,swap_sell;
   bool Valid()
     {
      return buy_value>0 && sell_value>0 && margin_buy>0 && margin_sell>0 && MathIsValidNumber(buy_value+sell_value+margin_buy+margin_sell+entry_cash+exit_cash+hold_buy+hold_sell+swap_buy+swap_sell);
     }
  };
struct SHabiSnapshot
  {
   double balance,equity,previous_balance,previous_equity;
   double lot,entry,sl,tp,profit_buy,profit_sell,age_weight,swap;
   datetime entered;
  };
struct SHabiAction
  {
   double lot,sl,tp;
   int reason;
// Signed target lot; source raw remains unchanged.
  };

//+------------------------------------------------------------------+
//| HabiTick.
//+------------------------------------------------------------------+
double HabiTick(double p,double tick)
  {
   return tick*MathFloor(p/tick+0.5);
  }

//+------------------------------------------------------------------+
//| HabiLot.
//+------------------------------------------------------------------+
double HabiLot(double v,SHabiSymbol &s)
  {
   if(v<s.lot_min)
      return 0;
   return s.lot_min+MathFloor((MathMin(MathMin(v,s.lot_max),VolumeCap)-s.lot_min)/s.lot_step+1e-10)*s.lot_step;
  }

//+------------------------------------------------------------------+
//| HabiProtection.
//+------------------------------------------------------------------+
bool HabiProtection(int side,double sl,double tp,double bid,double ask,SHabiSymbol &s)
  {
   double spread=ask-bid,base=MathMax(s.stops,1)*s.point+spread;
   double mark=side>0?bid:ask,dtp=side*(tp-mark),dsl=side*(mark-sl);
   return sl>0 && tp>0 && dtp>2*base && dsl>base && dtp>s.freeze*s.point && dsl>s.freeze*s.point;
  }

//+------------------------------------------------------------------+
//| HabiMap.
//+------------------------------------------------------------------+
bool HabiMap(const float &raw[],double bid,double ask,SHabiSymbol &s,double free_margin,double margin_buy,double margin_sell,SHabiAction &a)
  {
   a.lot=0;
   a.sl=0;
   a.tp=0;
   a.reason=HABI_OK;
   if(ArraySize(raw)!=6 || bid<=0 || ask<bid)
     {
      a.reason=HABI_INPUT_ERROR;
      return false;
     }
   for(int k=0;k<6;k++)
      if(!MathIsValidNumber(raw[k]) || raw[k]<0 || raw[k]>1)
     {
      a.reason=HABI_INPUT_ERROR;
      return false;
     }
   if(VolumeCap<s.lot_min || s.tick<=0)
     {
      a.reason=HABI_PROFILE_ERROR;
      return false;
     }
   double v=MathAbs(raw[0]-raw[3])*VolumeCap;
   if(v<s.lot_min)
     {
      a.reason=HABI_FLAT;
      return true;
     }
   int side=raw[0]>raw[3]?1:-1,off=side>0?0:3;
   a.lot=side*HabiLot(v,s);
   if(raw[off+1]==0 || raw[off+2]==0)
     {
      a.reason=HABI_ZERO_PROTECTION;
      return true;
     }
   double price=side>0?ask:bid;
   a.tp=HabiTick(price+side*raw[off+1]*MaxTP*s.point,s.tick);
   a.sl=HabiTick(price-side*raw[off+2]*MaxSL*s.point,s.tick);
   bool direction=s.trade_mode==SYMBOL_TRADE_MODE_FULL || (side>0 && s.trade_mode==SYMBOL_TRADE_MODE_LONGONLY) || (side<0 && s.trade_mode==SYMBOL_TRADE_MODE_SHORTONLY);
   if(!direction || !HabiProtection(side,a.sl,a.tp,bid,ask,s) || (s.lot_limit>0 && MathAbs(a.lot)>s.lot_limit) || MathAbs(a.lot)*(side>0?margin_buy:margin_sell)>free_margin)
      a.reason=HABI_ECONOMIC_INVALID;
   return true;
  }

//+------------------------------------------------------------------+
//| HabiAccount.
//+------------------------------------------------------------------+
bool HabiAccount(SHabiSnapshot &s,datetime time,float &account[])
  {
   if(s.previous_balance<=0 || s.previous_equity<=0 || EtalonBalance<=0)
      return false;
   ArrayResize(account,13);
   account[0]=(float)(s.balance/EtalonBalance);
   account[1]=(float)((s.balance-s.previous_balance)/s.previous_balance);
   account[2]=(float)(s.equity/s.previous_balance);
   account[3]=(float)((s.equity-s.previous_equity)/s.previous_equity);
   account[4]=(float)MathMax(s.lot,0);
   account[5]=(float)MathMax(-s.lot,0);
   account[6]=(float)(s.profit_buy/s.previous_balance);
   account[7]=(float)(s.profit_sell/s.previous_balance);
   account[8]=(float)(s.age_weight/(36000*s.previous_balance));
   double pi=3.14159265358979323846;
   account[9]=(float)MathSin(2*pi*(double)time/31536000);
   account[10]=(float)MathCos(2*pi*(double)time/PeriodSeconds(PERIOD_MN1));
   account[11]=(float)MathSin(2*pi*(double)time/PeriodSeconds(PERIOD_W1));
   account[12]=(float)MathSin(2*pi*(double)time/PeriodSeconds(PERIOD_D1));
   return HabiFinite(account,13);
  }

//+------------------------------------------------------------------+
//| HabiFinancialNumber.
//+------------------------------------------------------------------+
bool HabiFinancialNumber(string text,double &value)
  {
   StringTrimLeft(text);
   StringTrimRight(text);
   int n=StringLen(text),i=0,digits=0;
   if(n==0)
      return false;
   ushort ch=StringGetCharacter(text,i);
   if(ch=='+' || ch=='-')
      i++;
   while(i<n && StringGetCharacter(text,i)>='0' && StringGetCharacter(text,i)<='9')
     {
      digits++;
      i++;
     }
   if(i<n && StringGetCharacter(text,i)=='.')
     {
      i++;
      while(i<n && StringGetCharacter(text,i)>='0' && StringGetCharacter(text,i)<='9')
        {
         digits++;
         i++;
        }
     }
   if(digits==0)
      return false;
   if(i<n && (StringGetCharacter(text,i)=='e' || StringGetCharacter(text,i)=='E'))
     {
      i++;
      if(i<n && (StringGetCharacter(text,i)=='+' || StringGetCharacter(text,i)=='-'))
         i++;
      int exponent=0;
      while(i<n && StringGetCharacter(text,i)>='0' && StringGetCharacter(text,i)<='9')
        {
         exponent++;
         i++;
        }
      if(exponent==0)
         return false;
     }
   if(i!=n)
      return false;
   value=StringToDouble(text);
   return MathIsValidNumber(value);
  }
class CHabiHistory
  {
   public: MqlRates rates[];
   SHabiCost costs[];
   SHabiSymbol symbol;
   int rsi,cci,atr,macd;
   CHabiHistory()
     {
      rsi=INVALID_HANDLE;
      cci=INVALID_HANDLE;
      atr=INVALID_HANDLE;
      macd=INVALID_HANDLE;
     }
   ~CHabiHistory()
     {
      if(rsi!=INVALID_HANDLE)
         IndicatorRelease(rsi);
      if(cci!=INVALID_HANDLE)
         IndicatorRelease(cci);
      if(atr!=INVALID_HANDLE)
         IndicatorRelease(atr);
      if(macd!=INVALID_HANDLE)
         IndicatorRelease(macd);
     }
   bool Init()
     {
      if(!symbol.Read())
         return false;
      rsi=iRSI(_Symbol,DataTimeframe,14,PRICE_CLOSE);
      cci=iCCI(_Symbol,DataTimeframe,14,PRICE_TYPICAL);
      atr=iATR(_Symbol,DataTimeframe,14);
      macd=iMACD(_Symbol,DataTimeframe,12,26,9,PRICE_CLOSE);
      return rsi!=INVALID_HANDLE && cci!=INVALID_HANDLE && atr!=INVALID_HANDLE && macd!=INVALID_HANDLE;
     }
   bool Load()
     {
      if(FitStart<=0 || FitEnd<=FitStart || PlanningHorizonBars<1 || NForecast<1 || FitPasses<1 || HistoricalCostsFile=="")
         return false;
      bool range=OfflineBalanceMin!=0 || OfflineBalanceMax!=0;
      if(range)
        {
         if(!MathIsValidNumber(OfflineBalanceMin+OfflineBalanceMax) || OfflineBalanceMin<=0 || OfflineBalanceMax<OfflineBalanceMin)
            return false;
        }
      else
         if(!MathIsValidNumber(InitialOfflineBalance) || InitialOfflineBalance<=0)return false;
      ArraySetAsSeries(rates,true);
      if(CopyRates(_Symbol,DataTimeframe,FitStart,FitEnd,rates)<HistoryBars+MathMax(PlanningHorizonBars,NForecast)+5)
         return false;
      if(rates[0].time>=iTime(_Symbol,DataTimeframe,0))
        {
         Print("Habi: FIT must end before the current unfinished bar");
         return false;
        }
      int h=FileOpen(HistoricalCostsFile,FILE_READ|FILE_CSV|FILE_ANSI|(CommonFiles?FILE_COMMON:0),',');
      if(h==INVALID_HANDLE)
         return false;
      while(!FileIsEnding(h))
        {
         SHabiCost c;
         c.time=StringToTime(FileReadString(h));
         double fields[10];
         for(int k=0;k<10;k++)
           {
            if(FileIsEnding(h) || FileIsLineEnding(h) || !HabiFinancialNumber(FileReadString(h),fields[k]))
              {
               FileClose(h);
               return false;
              }
           }
         if(!FileIsLineEnding(h) && !FileIsEnding(h))
           {
            FileClose(h);
            return false;
           }
         c.buy_value=fields[0];
         c.sell_value=fields[1];
         c.margin_buy=fields[2];
         c.margin_sell=fields[3];
         c.entry_cash=fields[4];
         c.exit_cash=fields[5];
         c.hold_buy=fields[6];
         c.hold_sell=fields[7];
         c.swap_buy=fields[8];
         c.swap_sell=fields[9];
         if(!c.Valid() || c.time<=0)
           {
            FileClose(h);
            return false;
           }
         for(int k=0;k<ArraySize(costs);k++)
            if(costs[k].time==c.time)
           {
            FileClose(h);
            return false;
           }
         int n=ArraySize(costs);
         ArrayResize(costs,n+1);
         costs[n]=c;
        }
      FileClose(h);
      return ArraySize(costs)>0;
     }
   bool Cost(datetime t,SHabiCost &c)
     {
// Explicit historical conversion/cost records; no current-FX substitution.
      for(int i=0;i<ArraySize(costs);i++)
         if(costs[i].time==t)
        {
         c=costs[i];
         return c.Valid();
        }
      return false;
     }
   bool Market(datetime boundary,float &out[])
     {
      int shift=iBarShift(_Symbol,DataTimeframe,boundary,true);
      if(shift<0)
         return false;
      ArrayResize(out,HistoryBars*9);
      MqlRates past[];
      ArraySetAsSeries(past,true);
      if(CopyRates(_Symbol,DataTimeframe,shift+1,HistoryBars,past)!=HistoryBars)
         return false;
      for(int j=0;j<HistoryBars;j++)
        {
         double v[1];
         datetime t=past[j].time;
         int k=iBarShift(_Symbol,DataTimeframe,t,true);
         if(k<0 || iTime(_Symbol,DataTimeframe,k)!=t)
            return false;
         out[j]=(float)(past[j].close-past[j].open);
         out[HistoryBars+j]=(float)(past[j].high-past[j].open);
         out[2*HistoryBars+j]=(float)(past[j].low-past[j].open);
         out[3*HistoryBars+j]=(float)(past[j].tick_volume/1000.0);
         for(int f=4;f<9;f++)
           {
            int handle=f==4?rsi:(f==5?cci:(f==6?atr:macd)),buffer=f==8?1:0;
            if(CopyBuffer(handle,buffer,k,1,v)!=1 || v[0]==EMPTY_VALUE || !MathIsValidNumber(v[0]))
               return false;
            out[f*HistoryBars+j]=(float)v[0];
           }
        }
      return HabiFinite(out,HistoryBars*9);
     }
   double Uniform()
     {
      return (MathRand()+MathRand()/32768.0)/32768.0;
     }
   int Discrete(int count)
     {
      return count>0?(int)MathMin(count-1,MathFloor(Uniform()*count)):-1;
     }
   bool Level(int side,bool tp,double bid,double ask,double &out)
     {
      double maxdist=(tp?MaxTP:MaxSL)*symbol.point,mark=side>0?bid:ask;
      int count=0;
// Reservoir selection is uniform over distinct admissible rounded levels.
      double last=-1;
      for(int k=1;k<=int(MathFloor(maxdist/symbol.tick));k++)
        {
         double p=HabiTick(mark+(tp?side:-side)*k*symbol.tick,symbol.tick),dist=(tp?side:-side)*(p-mark);
         double base=MathMax(symbol.stops,1)*symbol.point+(ask-bid),min_dist=MathMax((tp?2:1)*base,symbol.freeze*symbol.point);
         if(p<=0 || dist<=min_dist || dist>maxdist+symbol.tick*1e-8 || p==last)
            continue;
         last=p;
         count++;
         if(Discrete(count)==0)
            out=p;
        }
      return count>0;
     }
   bool Sample(int e,SHabiSnapshot &s,SState &record,float &o[])
     {
      ZeroMemory(s);
      ZeroMemory(record);
      if(e<0 || e+MathMax(HistoryBars,5)>=ArraySize(rates))
         return false;
      bool range=OfflineBalanceMin!=0 || OfflineBalanceMax!=0;
      if(range && (OfflineBalanceMin<=0 || OfflineBalanceMax<OfflineBalanceMin))
         return false;
      s.balance=range?OfflineBalanceMin+Uniform()*(OfflineBalanceMax-OfflineBalanceMin):InitialOfflineBalance;
      if(!MathIsValidNumber(s.balance) || s.balance<=0)
         return false;
      s.previous_balance=s.balance;
      s.previous_equity=s.balance;
      s.equity=s.balance;
      int side=Discrete(3)-1;
      if(side!=0)
        {
         int age=1+Discrete(5);
         SHabiCost entry,now,prev;
         if(!Cost(rates[e+age].time,entry) || !Cost(rates[e].time,now) || !Cost(rates[e+1].time,prev))
            return false;
         int selected=0,count=0;
         double maximum=MathMin(VolumeCap,symbol.lot_max);
         if(symbol.lot_limit>0)
            maximum=MathMin(maximum,symbol.lot_limit);
         for(int k=0;k<=int(MathFloor((maximum-symbol.lot_min)/symbol.lot_step+1e-8));k++)
           {
            double lot=symbol.lot_min+k*symbol.lot_step;
            if(lot*(side>0?entry.margin_buy:entry.margin_sell)>.5*s.balance)
               continue;
            count++;
            if(Discrete(count)==0)
               selected=k;
           }
         if(count==0)
            return false;
         s.lot=side*(symbol.lot_min+selected*symbol.lot_step);
         s.entered=rates[e+age].time;
         s.entry=rates[e+age].open+(side>0?rates[e+age].spread*symbol.point:0);
         double mark=rates[e].open+(side<0?rates[e].spread*symbol.point:0);
         double pmark=rates[e+1].open+(side<0?rates[e+1].spread*symbol.point:0);
         double profit=s.lot*(mark-s.entry)*(side>0?now.buy_value:now.sell_value);
         double pp=s.lot*(pmark-s.entry)*(side>0?prev.buy_value:prev.sell_value);
// B already includes settled entry and holding costs. Undo only the last interval for Bprev.
         s.previous_balance=s.balance-MathAbs(s.lot)*(side>0?prev.hold_buy:prev.hold_sell);
         double previous_swap=0;
         for(int j=e+age;j>=e+1;j--)
           {
            SHabiCost held;
            if(!Cost(rates[j].time,held))
               return false;
            double increment=MathAbs(s.lot)*(side>0?held.swap_buy:held.swap_sell);
            s.swap+=increment;
            if(j>e+1)
               previous_swap+=increment;
           }
         s.equity=s.balance+profit+s.swap;
         s.previous_equity=s.previous_balance+pp+previous_swap;
         s.profit_buy=side>0?profit:0;
         s.profit_sell=side<0?profit:0;
         s.age_weight=(double)(rates[e].time-s.entered)*MathAbs(profit);
         if(!Level(side,false,rates[e].open,rates[e].open+rates[e].spread*symbol.point,s.sl) || !Level(side,true,rates[e].open,rates[e].open+rates[e].spread*symbol.point,s.tp))
            return false;
        }
      float market[],account[];
      if(!Market(rates[e].time,market) || !HabiAccount(s,rates[e].time,account))
         return false;
      ArrayCopy(record.state,market);
      for(int i=0;i<9;i++)
         record.account[i]=account[i];
      ArrayCopy(o,market);
      ArrayCopy(o,account,ArraySize(market));
      return true;
     }
  };
// All alternatives share one bar loop. Helpers operate on entire columns.
enum EHabiCol
  {
   HC_LOT,HC_ENTRY,HC_SL,HC_TP,HC_SWAP,HC_CASH,HC_PREVIOUS,HC_RETURN,HC_PENALTY,HC_REASON,HC_EXIT_REASON,HC_ACTIVE,HC_COUNT
  };

//+------------------------------------------------------------------+
//| HabiMask.
//+------------------------------------------------------------------+
vector HabiMask(const vector &v,double scalar,int relation)
  {
   vector mask(v.Size());
   for(ulong i=0;i<v.Size();i++)
      mask[i]=relation==0?(v[i]==scalar):(relation==1?(v[i]>scalar):(v[i]<scalar));
   return mask;
  }

//+------------------------------------------------------------------+
//| HabiChoose.
//+------------------------------------------------------------------+
vector HabiChoose(const vector &mask,const vector &yes,const vector &no)
  {
   return mask*yes+(mask*(-1.0)+1.0)*no;
  }

//+------------------------------------------------------------------+
//| HabiValue.
//+------------------------------------------------------------------+
vector HabiValue(const vector &lot,const vector &entry,double bid,double ask,SHabiCost &cost)
  {
   vector buy=HabiMask(lot,0,1);
   return lot*((entry*(-1.0)+bid)*buy*cost.buy_value+(entry*(-1.0)+ask)*(buy*(-1.0)+1.0)*cost.sell_value);
  }
struct SHabiEvaluation
  {
   float target[];
   double money_return[],penalty[];
   int status[],reason[];
  };

//+------------------------------------------------------------------+
//| HabiHistoricalEvaluate.
//+------------------------------------------------------------------+
bool HabiHistoricalEvaluate(CHabiHistory &history,int e,SHabiSnapshot &source,const float &actions[],int C,SHabiEvaluation &result)
  {
   if(C<1 || ArraySize(actions)!=C*6 || e-PlanningHorizonBars+1<0)
      return false;
   matrix agents=matrix::Zeros(C,HC_COUNT);
   SHabiCost first;
   if(!history.Cost(history.rates[e].time,first))
      return false;
   double bid=history.rates[e].open,ask=bid+history.rates[e].spread*history.symbol.point;
   double source_profit=source.profit_buy+source.profit_sell+source.swap;
// Initial command application, once per row, before the common H-bar traversal.
   for(int n=0;n<C;n++)
     {
      double lot=source.lot,entry=source.entry,sl=source.sl,tp=source.tp,cash=0,swap=source.swap;
      int side=lot>0?1:-1;
      double mark=side>0?bid:ask;
      if(lot!=0 && ((side*(mark-sl)<=0) || (side*(mark-tp)>=0)))
        {
         cash=lot*(mark-entry)*(side>0?first.buy_value:first.sell_value)+swap+MathAbs(lot)*first.exit_cash;
         lot=0;
         swap=0;
        }
      float raw[];
      ArrayResize(raw,6);
      ArrayCopy(raw,actions,0,n*6,6);
      SHabiAction a;
      double available_equity=source.balance+cash+swap+lot*((lot>0?bid:ask)-entry)*(lot>0?first.buy_value:first.sell_value);
      if(!HabiMap(raw,bid,ask,history.symbol,available_equity,first.margin_buy,first.margin_sell,a))
         return false;
      bool reverse=lot*a.lot<0,close=a.reason==HABI_FLAT || a.reason==HABI_ZERO_PROTECTION || reverse;
      if(close && lot!=0)
        {
         cash+=lot*((lot>0?bid:ask)-entry)*(lot>0?first.buy_value:first.sell_value)+swap+MathAbs(lot)*first.exit_cash;
         lot=0;
         swap=0;
        }
// Closing a prior exposure changes available equity through its actual exit costs.
// A margin-only rejection before a reverse close is reconsidered after that close.
      if(close && a.reason!=HABI_FLAT && a.reason!=HABI_ZERO_PROTECTION)
        {
         available_equity=source.balance+cash;
         if(!HabiMap(raw,bid,ask,history.symbol,available_equity,first.margin_buy,first.margin_sell,a))
            return false;
        }
      if(a.reason==HABI_OK)
        {
         double old=MathAbs(lot),wanted=MathAbs(a.lot),price=a.lot>0?ask:bid;
         if(wanted<old)
           {
            double reduction=old-wanted;
            cash+=reduction*(lot>0?1:-1)*((lot>0?bid:ask)-entry)*(lot>0?first.buy_value:first.sell_value)+reduction*first.exit_cash+swap*(reduction/old);
            swap*=wanted/old;
           }
         if(wanted>old)
           {
            double addition=wanted-old;
            entry=(old*entry+addition*price)/wanted;
            cash+=addition*first.entry_cash;
           }
         lot=a.lot;
         sl=a.sl;
         tp=a.tp;
        }
      agents[n,HC_LOT]=lot;
      agents[n,HC_ENTRY]=entry;
      agents[n,HC_SL]=sl;
      agents[n,HC_TP]=tp;
      agents[n,HC_SWAP]=swap;
      agents[n,HC_CASH]=cash;
      agents[n,HC_PREVIOUS]=source_profit;
      agents[n,HC_ACTIVE]=lot!=0;
      agents[n,HC_PENALTY]=(a.reason==HABI_ZERO_PROTECTION || a.reason==HABI_ECONOMIC_INVALID)?PenaltyInvalidCurrency:0;
      agents[n,HC_REASON]=a.reason;
     }
   for(int h=0;h<PlanningHorizonBars;h++)
     {
      int i=e-h;
      MqlRates bar=history.rates[i];
      SHabiCost cost;
      if(!history.Cost(bar.time,cost))
         return false;
      double spread=bar.spread*history.symbol.point;
      vector lot=agents.Col(HC_LOT),entry=agents.Col(HC_ENTRY),active=agents.Col(HC_ACTIVE),sl=agents.Col(HC_SL),tp=agents.Col(HC_TP);
      vector buy=HabiMask(lot,0,1),side=buy*2.0-1.0;
      vector opening=buy*bar.open+(buy*(-1.0)+1.0)*(bar.open+spread);
      vector low=buy*bar.low+(buy*(-1.0)+1.0)*(bar.low+spread),high=buy*bar.high+(buy*(-1.0)+1.0)*(bar.high+spread);
      vector hit_open_sl=active*HabiMask(side*(opening-sl),0,-1)+active*HabiMask(side*(opening-sl),0,0);
      vector hit_open_tp=active*(hit_open_sl*(-1.0)+1.0)*(HabiMask(side*(opening-tp),0,1)+HabiMask(side*(opening-tp),0,0));
      vector remaining=active*(hit_open_sl*(-1.0)+1.0)*(hit_open_tp*(-1.0)+1.0);
      vector adverse=buy*low+(buy*(-1.0)+1.0)*high,favourable=buy*high+(buy*(-1.0)+1.0)*low;
      vector hit_sl=remaining*(HabiMask(side*(adverse-sl),0,-1)+HabiMask(side*(adverse-sl),0,0));
      vector hit_tp=remaining*(hit_sl*(-1.0)+1.0)*(HabiMask(side*(favourable-tp),0,1)+HabiMask(side*(favourable-tp),0,0));
      vector close=hit_open_sl+hit_open_tp+hit_sl+hit_tp;
      vector price=(hit_open_sl+hit_open_tp)*opening+hit_sl*sl+hit_tp*tp;
      vector horizon_close=active*(close*(-1.0)+1.0)*(h==PlanningHorizonBars-1?1:0);
      price=price+horizon_close*(buy*bar.close+(buy*(-1.0)+1.0)*(bar.close+spread));
      close=close+horizon_close;
      vector factor=buy*cost.buy_value+(buy*(-1.0)+1.0)*cost.sell_value;
      vector survivors=active*(close*(-1.0)+1.0);
      vector held=survivors+horizon_close;
      vector accrued_swap=agents.Col(HC_SWAP)+held*MathAbs(lot)*(buy*cost.swap_buy+(buy*(-1.0)+1.0)*cost.swap_sell);
      vector realized=close*(lot*(price-entry)*factor+accrued_swap+MathAbs(lot)*cost.exit_cash);
      vector holding=held*MathAbs(lot)*(buy*cost.hold_buy+(buy*(-1.0)+1.0)*cost.hold_sell);
      vector remaining_swap=survivors*accrued_swap;
      vector cash=agents.Col(HC_CASH)+realized+holding;
      vector value=cash+survivors*HabiValue(lot,entry,bar.close,bar.close+spread,cost)+remaining_swap;
      agents.Col(remaining_swap,HC_SWAP);
      vector returns=agents.Col(HC_RETURN)+(value-agents.Col(HC_PREVIOUS))*MathPow(DiscountGamma,h);
      vector reasons=agents.Col(HC_EXIT_REASON);
      reasons=HabiChoose(hit_open_sl+hit_sl,vector::Ones(C)*HABI_SL,reasons);
      reasons=HabiChoose(hit_open_tp+hit_tp,vector::Ones(C)*HABI_TP,reasons);
      reasons=HabiChoose(horizon_close,vector::Ones(C)*HABI_HORIZON,reasons);
      agents.Col(cash,HC_CASH);
      agents.Col(value,HC_PREVIOUS);
      agents.Col(returns,HC_RETURN);
      agents.Col(reasons,HC_EXIT_REASON);
      agents.Col(survivors,HC_ACTIVE);
     }
   ArrayResize(result.target,C);
   ArrayResize(result.money_return,C);
   ArrayResize(result.penalty,C);
   ArrayResize(result.reason,C);
   ArrayResize(result.status,C);
   for(int n=0;n<C;n++)
     {
      double money=agents[n,HC_RETURN],penalty=agents[n,HC_PENALTY],target=money-penalty;
      if(!MathIsValidNumber(target) || MathAbs(target)>FLT_MAX)
         return false;
      result.target[n]=(float)target;
      result.money_return[n]=money;
      result.penalty[n]=penalty;
      result.reason[n]=(int)(penalty>0 || agents[n,HC_EXIT_REASON]==HABI_OK?agents[n,HC_REASON]:agents[n,HC_EXIT_REASON]);
      result.status[n]=1;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| HabiOracul.
//+------------------------------------------------------------------+
vector<float> HabiOracul(const vector<float> &account, const matrix<float> &future)
  {
//--- Look for target
   vector<float> result = vector<float>::Zeros(NActions);
   matrix<float> fstate = future;
//---
   vector<float> target = fstate.Col(0).CumSum();
   if(account[4] > account[5])
     {
      float tp = 0;
      float sl = 0;
      float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT));
      int pos = 0;
      for(int j = 0; j < NForecast; j++)
        {
         tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
         pos = j;
         if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
            break;
         sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
        }
      if(pos > 0 && tp > 0)
        {
         sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01));
         tp = float(MathMax(MathMin(tp / (MaxTP * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01));
         result[0] = MathMax(account[4] - account[5], 0.011f);
         result[5] = result[1] = tp;
         result[4] = result[2] = sl;
         result[3] = 0;
        }
     }
   else
     {
      if(account[4] < account[5])
        {
         float tp = 0;
         float sl = 0;
         float cur_sl = float(MathMax(MathRand() / 32767.0, 0.01) * MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT));
         int pos = 0;
         for(int j = 0; j < NForecast; j++)
           {
            tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
            pos = j;
            if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
               break;
            sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
           }
         if(pos > 0 && tp < 0)
           {
            sl = float(MathMax(MathMin(MathAbs(sl) / (MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01));
            tp = float(MathMax(MathMin(-tp / (MaxTP * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01));
            result[3] = MathMax(account[5] - account[4], 0.011f);
            result[2] = result[4] = tp;
            result[1] = result[5] = sl;
            result[0] = 0;
           }
        }
      else
        {
         ulong argmin = target.ArgMin();
         ulong argmax = target.ArgMax();
         float max_sl = float(MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT));
         double equity = account[2] * account[0] * EtalonBalance / (1 + account[1]);
         while(argmax > 0 && argmin > 0)
           {
            if(argmax < argmin && target[argmax] / 2 > MathAbs(target[argmin]) && MathAbs(target[argmin]) < max_sl)
               break;
            if(argmax > argmin && target[argmax] < MathAbs(target[argmin] / 2) && target[argmax] < max_sl)
               break;
            target.Resize(MathMin(argmax, argmin));
            argmin = target.ArgMin();
            argmax = target.ArgMax();
           }
         if(argmin == 0 || (argmax < argmin && argmax > 0))
           {
            float tp = 0;
            float sl = 0;
            float cur_sl = - float(MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT));
            ulong pos = 0;
            for(ulong j = 0; j < argmax; j++)
              {
               tp = MathMax(tp, target[j] + fstate[j, 1] - fstate[j, 0]);
               pos = j;
               if(cur_sl >= -(target[j] + fstate[j, 2] - fstate[j, 0]))
                  break;
               sl = MathMin(sl, target[j] + fstate[j, 2] - fstate[j, 0]);
              }
            if(pos > 0 && tp > 0)
              {
               sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01);
               tp = (float)MathMin(tp / (MaxTP * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1);
               result[0] = float(MathMax(equity / 100 * 0.01, 0.011));
               result[5] = result[1] = tp;
               result[4] = result[2] = sl;
               result[3] = 0;
              }
           }
         else
           {
            if(argmax == 0 || argmax > argmin)
              {
               float tp = 0;
               float sl = 0;
               float cur_sl = float(MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT));
               ulong pos = 0;
               for(ulong j = 0; j < argmin; j++)
                 {
                  tp = MathMin(tp, target[j] + fstate[j, 2] - fstate[j, 0]);
                  pos = j;
                  if(cur_sl <= target[j] + fstate[j, 1] - fstate[j, 0])
                     break;
                  sl = MathMax(sl, target[j] + fstate[j, 1] - fstate[j, 0]);
                 }
               if(pos > 0 && tp < 0)
                 {
                  sl = (float)MathMax(MathMin(MathAbs(sl) / (MaxSL * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1), 0.01);
                  tp = (float)MathMin(-tp / (MaxTP * SymbolInfoDouble(_Symbol,SYMBOL_POINT)), 1);
                  result[3] = float(MathMax(equity / 100 * 0.01, 0.011));
                  result[2] = result[4] = tp;
                  result[1] = result[5] = sl;
                  result[0] = 0;
                 }
              }
           }
        }
     }
//---
   result[0]=(float)MathMin(1.0,result[0]/VolumeCap);
   result[3]=(float)MathMin(1.0,result[3]/VolumeCap);
   return(result);
  }

//+------------------------------------------------------------------+
//| HabiTeacher.
//+------------------------------------------------------------------+
bool HabiTeacher(CHabiHistory &h,int e,SHabiSnapshot &snapshot,const float &observation[],float &teacher[],double &teacher_return)
  {
   if(e-NForecast+1<0)
      return false;
   matrix<float> future=matrix<float>::Zeros(NForecast,9);
   for(int j=0;j<NForecast;j++)
     {
      MqlRates b=h.rates[e-j];
      future[j,0]=(float)(b.close-b.open);
      future[j,1]=(float)(b.high-b.open);
      future[j,2]=(float)(b.low-b.open);
     }
   float account[];
   if(!HabiAccount(snapshot,h.rates[e].time,account))
      return false;
   vector<float> av;
   if(!av.Assign(account))
      return false;
   vector<float> candidate=HabiOracul(av,future);
   float pair[];
   ArrayResize(pair,12);
   ArrayInitialize(pair,0);
   for(int k=0;k<6;k++)
      pair[k]=candidate[k];
   SHabiEvaluation evaluation;
   if(!HabiHistoricalEvaluate(h,e,snapshot,pair,2,evaluation))
      return false;
   bool valid=evaluation.penalty[0]==0 && evaluation.target[0]>evaluation.target[1];
   ArrayResize(teacher,6);
   for(int k=0;k<6;k++)
      teacher[k]=valid?pair[k]:0;
   teacher_return=evaluation.target[valid?0:1];
   return true;
  }
#endif
