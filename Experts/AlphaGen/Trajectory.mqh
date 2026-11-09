#ifndef ALPHAGEN_TRAJECTORY_MQH
#define ALPHAGEN_TRAJECTORY_MQH
#include "../NeuroNet_DNG/NeuroNet.mqh"
#include <Trade/Trade.mqh>
//+------------------------------------------------------------------+
//| Defines: the single source of AlphaGen algorithm settings.       |
//+------------------------------------------------------------------+
#define AG_WINDOW 512
#define AG_CHANNELS 5
#define AG_ACCOUNT_SIZE 13
#define AG_OUTPUT_SIZE 6
#define AG_LOOKBACK 120
#define AG_HORIZON 1
#define AG_IC_WINDOW 256
#define AG_NORM_WINDOW 256
#define AG_POOL_SIZE 20
#define AG_MAX_TOKENS 20
#define AG_CONTEXT_BARS 256
#define AG_WARMUP_BARS 631
#define AG_WARMUP_TRIALS 256
#define AG_SEARCH_TRIALS 8
#define AG_EMBEDDING 32
#define AG_SEARCH_HIDDEN 64
#define AG_SEARCH_DEPTH 2
#define AG_TRADE_HIDDEN 32
#define AG_MLP_HIDDEN 32
#define AG_TRADING_ROLLOUT 128
#define AG_TOKEN_ROLLOUT 2048
#define AG_PPO_PASSES 4
#define AG_LEARNING_CONTOURS 2
#define AG_CLIP 0.2
#define AG_TRADE_GAMMA 0.99
#define AG_TOKEN_GAMMA 1.0
#define AG_LAMBDA 0.95
#define AG_VALUE_COEF 0.5
#define AG_ENTROPY_COEF 0.01
#define AG_EPSILON_VAR 0.000001
#define AG_ABSOLUTE_TOL 0.00001
#define AG_RELATIVE_TOL 0.0001
#define AG_RIDGE 0.0001
#define AG_SIGNAL_CLIP 10.0
#define AG_SIGMA_FLOOR 0.01
#define AG_VOLUME_CAP 1.0
#define AG_MAX_TP 1000.0
#define AG_MAX_SL 1000.0
#define AG_INVALID_PENALTY 0.0001
#define AG_RUIN_PENALTY 1.0
#define AG_EQUITY_FLOOR 0.10
#define AG_ETALON_BALANCE 10000.0
#define AG_MIN_TRADING_BALANCE 50.0
#define AG_LOT_EPSILON 0.0000000001
#define AG_AGE_SECONDS 36000.0
#define AG_YEAR_SECONDS 31536000.0
#define AG_PI 3.14159265358979323846
#define AG_POSITION_SCALE 10000.0
#define AG_INVALID_TOKEN_REWARD -1.0
#define AG_TIMER_MS 100
#define AG_CONTEXT_METADATA 5
#define AG_LIMB_BITS 16
#define AG_LIMB_MASK 65535
#define AG_TIME_LIMBS 4
#define AG_PROTECTION_MULTIPLIER 2.0
#define AG_CONST_NEGATIVE_ONE -1.0
#define AG_CONST_NEGATIVE_HALF -0.5
#define AG_CONST_HALF 0.5
#define AG_CONST_ONE 1.0
#define AG_CONST_TWO 2.0
#define AG_INTERVAL_ONE 1
#define AG_INTERVAL_FIVE 5
#define AG_INTERVAL_TEN 10
#define AG_INTERVAL_TWENTY 20
#define AG_INTERVAL_SIXTY 60
input ENUM_TIMEFRAMES TimeFrame=PERIOD_H1;
input string ModelFile="AlphaGen\\AlphaGen.nn";
input ulong Seed=0;
#ifdef ALPHAGEN_STUDY
input datetime Start=0;
input datetime End=0;
input int Epochs=1;
input double MinBalance=100;
input double MaxBalance=1500;
#endif
//+------------------------------------------------------------------+
//| Encode exact integer metadata in FP32 service words.             |
//+------------------------------------------------------------------+
void AlphaGenInteger(float &data[],int offset,ulong value)
  {
   for(int i=0;i<AG_TIME_LIMBS;i++)
     {
      data[offset+i]=(float)((value>>(AG_LIMB_BITS*i))&AG_LIMB_MASK);
     }
  }
//+------------------------------------------------------------------+
//| Financial part: each added volume retains its own entry and age. |
//+------------------------------------------------------------------+
struct SAlphaGenPart
  {
   double volume,entry;
   datetime time;
  };
struct SAlphaGenTarget
  {
   double volume,sl,tp;
   bool invalid;
  };
class CAlphaGenEnvironment
  {
   public:
   bool virtual_mode;
   double balance,equity,previous_balance,previous_equity,reference,peak;
   double gross_profit,gross_loss,max_drawdown,max_drawdown_fraction;
   double volume,buy_profit,sell_profit,age_weight,margin,sl,tp;
   double point,tick,lot_min,lot_step,lot_max,lot_limit,k_buy,k_sell;
   long stops,freeze,trade_mode;
   ulong trades,successful_requests;
   datetime entered;
   SAlphaGenPart parts[];
   CTrade trade;
   bool pending,stopout,unknown_request,reconciling;
   uint request_id;
   ulong request_deals[],request_orders[];
   double intended_volume,intended_sl,intended_tp;
//+------------------------------------------------------------------+
//| Initialize owned runtime state and diagnostic counters.          |
//+------------------------------------------------------------------+
   CAlphaGenEnvironment():virtual_mode(false),
   gross_profit(0),
   gross_loss(0),
   max_drawdown(0),
   max_drawdown_fraction(0),
   trades(0),
   successful_requests(0),
   pending(false),
   stopout(false),
   unknown_request(false),
   reconciling(false),
   request_id(0)
     {
     }
//+------------------------------------------------------------------+
//| Validate the run and initialize the shared state.                |
//+------------------------------------------------------------------+
   bool Init(bool simulated,double capital)
     {
      virtual_mode=simulated;
      point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
      tick=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
      lot_min=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
      lot_step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
      lot_max=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
      lot_limit=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_LIMIT);
      stops=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
      freeze=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL);
      trade_mode=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE);
      if(point<=0 || tick<=0 || lot_min<=0 || lot_step<=0 || lot_max<lot_min || capital<=0 ||
         AG_VOLUME_CAP<lot_min || !MathIsValidNumber(point+tick+lot_min+lot_step+lot_max+capital))
        {
         ReturnFalse;
        }
      k_buy=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE_PROFIT)/tick;
      k_sell=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE_LOSS)/tick;
      if(k_buy<=0 || k_sell<=0 || !MathIsValidNumber(k_buy+k_sell))
        {
         ReturnFalse;
        }
      if(!simulated && (PositionsTotal()!=0 || OrdersTotal()!=0))
        {
         ReturnFalse;
        }
      trade.SetAsyncMode(false);
      trade.SetTypeFillingBySymbol(_Symbol);
      Reset(capital);
      return(true);
     }
//+------------------------------------------------------------------+
//| Start a positive financial epoch without creating trading profit.|
//+------------------------------------------------------------------+
   void Reset(double capital)
     {
      balance=equity=previous_balance=previous_equity=reference=peak=capital;
      volume=buy_profit=sell_profit=age_weight=margin=sl=tp=0;
      entered=0;
      stopout=false;
      ArrayResize(parts,0);
     }
//+------------------------------------------------------------------+
//| Reconcile current exposure and mark confirmed account equity.    |
//+------------------------------------------------------------------+
   bool Snapshot(double bid,double ask,datetime time)
     {
      buy_profit=sell_profit=age_weight=margin=volume=0;
      if(virtual_mode)
        {
         for(int i=0;i<ArraySize(parts);i++)
           {
            double q=parts[i].volume;
            double p=q>0?q*k_buy*(bid-parts[i].entry):(-q)*k_sell*(parts[i].entry-ask);
            if(q>0)
              {
               buy_profit+=p;
              }
            else
              {
               sell_profit+=p;
              }
            age_weight+=MathAbs(p)*(double)(time-parts[i].time);
            volume+=q;
            double m=0;
            if(!OrderCalcMargin(q>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,MathAbs(q),q>0?ask:bid,m))
              {
               ReturnFalse;
              }
            margin+=m;
           }
         equity=balance+buy_profit+sell_profit;
        }
      else
        {
         balance=AccountInfoDouble(ACCOUNT_BALANCE);
         equity=AccountInfoDouble(ACCOUNT_EQUITY);
         margin=AccountInfoDouble(ACCOUNT_MARGIN);
         sl=tp=0;
         bool all_protected=true;
         for(int i=PositionsTotal()-1;i>=0;i--)
           {
            if(PositionGetTicket(i)==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol)
              {
               ReturnFalse;
              }
            double q=PositionGetDouble(POSITION_VOLUME);
            bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
            double p=PositionGetDouble(POSITION_PROFIT);
            volume+=buy?q:-q;
            if(buy)
              {
               buy_profit+=p;
              }
            else
              {
               sell_profit+=p;
              }
            age_weight+=MathAbs(p)*(double)(time-(datetime)PositionGetInteger(POSITION_TIME));
            sl=PositionGetDouble(POSITION_SL);
            tp=PositionGetDouble(POSITION_TP);
            if(sl<=0 || tp<=0)
              {
               all_protected=false;
              }
           }
         if(!all_protected)
           {
            sl=tp=0;
           }
// Synchronous requests are reconciled against orders and account state.
         pending=unknown_request || OrdersTotal()>0;
        }
      peak=MathMax(peak,equity);
      max_drawdown=MathMax(max_drawdown,peak-equity);
      if(peak>0)
        {
         max_drawdown_fraction=MathMax(max_drawdown_fraction,(peak-equity)/peak);
        }
      if(!MathIsValidNumber(balance+equity+volume+margin+age_weight))
        {
         ReturnFalse;
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Build Account13 against the original decision snapshots.         |
//+------------------------------------------------------------------+
   bool Account(datetime time,float &a[])
     {
      if(previous_balance<=0 || previous_equity<=0)
        {
         ReturnFalse;
        }
      ArrayResize(a,AG_ACCOUNT_SIZE);
      a[0]=(float)(balance/AG_ETALON_BALANCE);
      a[1]=(float)((balance-previous_balance)/previous_balance);
      a[2]=(float)(equity/previous_balance);
      a[3]=(float)((equity-previous_equity)/previous_equity);
      a[4]=(float)MathMax(volume,0);
      a[5]=(float)MathMax(-volume,0);
      a[6]=(float)(buy_profit/previous_balance);
      a[7]=(float)(sell_profit/previous_balance);
      a[8]=(float)(age_weight/(AG_AGE_SECONDS*previous_balance));
      a[9]=(float)MathSin(2*AG_PI*(double)time/AG_YEAR_SECONDS);
      a[10]=(float)MathCos(2*AG_PI*(double)time/PeriodSeconds(PERIOD_MN1));
      a[11]=(float)MathSin(2*AG_PI*(double)time/PeriodSeconds(PERIOD_W1));
      a[12]=(float)MathSin(2*AG_PI*(double)time/PeriodSeconds(PERIOD_D1));
      for(int i=0;i<AG_ACCOUNT_SIZE;i++)
        {
         if(!MathIsValidNumber(a[i]))
           {
            ReturnFalse;
           }
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Record the confirmed starting snapshot before execution.         |
//+------------------------------------------------------------------+
   void Commit()
     {
      previous_balance=balance;
      previous_equity=equity;
     }
//+------------------------------------------------------------------+
//| Round the target volume to the symbol lot grid.                  |
//+------------------------------------------------------------------+
   double Lot(double v)
     {
      if(v<lot_min)
        {
         return(0);
        }
      return(lot_min+MathFloor((MathMin(MathMin(v,lot_max),AG_VOLUME_CAP)-lot_min)/lot_step+AG_LOT_EPSILON)*lot_step);
     }
//+------------------------------------------------------------------+
//| Validate Action6 and construct the protected directional target. |
//+------------------------------------------------------------------+
   bool Map(const float &raw[],double bid,double ask,SAlphaGenTarget &a)
     {
      a.volume=a.sl=a.tp=0;
      a.invalid=false;
      if(ArraySize(raw)!=AG_OUTPUT_SIZE || bid<=0 || ask<bid)
        {
         a.invalid=true;
         return(true);
        }
      for(int i=0;i<AG_OUTPUT_SIZE;i++)
        {
         if(!MathIsValidNumber(raw[i]) || raw[i]<0 || raw[i]>1)
           {
            a.invalid=true;
            return(true);
           }
        }
      double q=Lot(MathAbs(raw[0]-raw[3])*AG_VOLUME_CAP);
      if(q==0)
        {
         return(true);
        }
      int side=raw[0]>raw[3]?1:-1,off=side>0?0:3;
      double price=side>0?ask:bid,mark=side>0?bid:ask;
      a.volume=side*q;
      a.tp=tick*MathFloor((price+side*raw[off+1]*AG_MAX_TP*point)/tick+AG_CONST_HALF);
      a.sl=tick*MathFloor((price-side*raw[off+2]*AG_MAX_SL*point)/tick+AG_CONST_HALF);
      double base=MathMax(stops,1)*point+(ask-bid);
      double dtp=side*(a.tp-mark),dsl=side*(mark-a.sl);
      bool direction=trade_mode==SYMBOL_TRADE_MODE_FULL || (side>0 && trade_mode==SYMBOL_TRADE_MODE_LONGONLY)
      || (side<0 && trade_mode==SYMBOL_TRADE_MODE_SHORTONLY);
      a.invalid=(!direction || raw[off+1]==0 || raw[off+2]==0 || a.sl<=0 || a.tp<=0 || dtp<=AG_PROTECTION_MULTIPLIER*base
      || dsl<=base || dtp<=freeze*point || dsl<=freeze*point || (lot_limit>0 && q>lot_limit));
      return(true);
     }
//+------------------------------------------------------------------+
//| Realize each closed virtual part once at the available quote.    |
//+------------------------------------------------------------------+
   bool CloseVirtual(double amount,double bid,double ask,datetime time)
     {
      for(int i=ArraySize(parts)-1;i>=0 && amount>AG_LOT_EPSILON;i--)
        {
         double q=parts[i].volume,take=MathMin(MathAbs(q),amount);
         double realized=q>0?take*k_buy*(bid-parts[i].entry):take*k_sell*(parts[i].entry-ask);
         balance+=realized;
         if(realized>=0)
           {
            gross_profit+=realized;
           }
         else
           {
            gross_loss-=realized;
           }
         parts[i].volume+=(q>0?-take:take);
         amount-=take;
         if(MathAbs(parts[i].volume)<AG_LOT_EPSILON)
           {
            ArrayRemove(parts,i,1);
           }
         trades++;
        }
      return(Snapshot(bid,ask,time));
     }
//+------------------------------------------------------------------+
//| Preserve unresolved request outcomes before any retry.           |
//+------------------------------------------------------------------+
   void RecordRequest(MqlTradeResult &result)
     {
      if(result.deal==0 && result.order==0)
        {
         return;
        }
      int n=ArraySize(request_deals);
      if(ArrayResize(request_deals,n+1)!=n+1 || ArrayResize(request_orders,n+1)!=n+1)
        {
         return;
        }
      request_deals[n]=result.deal;
      request_orders[n]=result.order;
     }
//+------------------------------------------------------------------+
//| Inspect request outcomes without repeating an unknown request.  |
//+------------------------------------------------------------------+
   bool ConfirmRequest(bool sent)
     {
      MqlTradeResult result;
      trade.Result(result);
      RecordRequest(result);
      request_id=result.request_id;
      if(result.retcode==TRADE_RETCODE_TIMEOUT || result.retcode==TRADE_RETCODE_CONNECTION)
        {
         unknown_request=true;
         pending=true;
         reconciling=true;
         return(true);
        }
      if(result.retcode==TRADE_RETCODE_PLACED)
        {
         pending=true;
         reconciling=true;
         return(true);
        }
      if(result.retcode==TRADE_RETCODE_DONE_PARTIAL)
        {
         reconciling=true;
        }
      if(result.retcode==TRADE_RETCODE_NO_CHANGES)
        {
         return(true);
        }
      bool confirmed=sent && (result.retcode==TRADE_RETCODE_DONE || result.retcode==TRADE_RETCODE_DONE_PARTIAL);
      if(confirmed)
        {
         successful_requests++;
        }
      return(confirmed);
     }
//+------------------------------------------------------------------+
//| Reconcile a protected target before increasing exposure.         |
//+------------------------------------------------------------------+
   bool Execute(SAlphaGenTarget &a,double bid,double ask,datetime time)
     {
      if(pending)
        {
         return(true);
        }
      intended_volume=a.invalid?0:a.volume;
      intended_sl=a.sl;
      intended_tp=a.tp;
      if(a.invalid)
        {
         a.volume=0;
        }
      double same_side_delta=(volume*a.volume>0)?MathAbs(a.volume)-MathAbs(volume):0;
      if(MathAbs(same_side_delta)>AG_LOT_EPSILON && MathAbs(same_side_delta)<lot_min-AG_LOT_EPSILON)
        {
         a.invalid=true;
         a.volume=0;
         intended_volume=0;
        }
      if(virtual_mode)
        {
         if(volume*a.volume<0 || a.volume==0)
           {
            if(!CloseVirtual(MathAbs(volume),bid,ask,time))
              {
               ReturnFalse;
              }
           }
         double delta=MathAbs(a.volume)-MathAbs(volume);
         if(delta<-AG_LOT_EPSILON)
           {
            if(!CloseVirtual(-delta,bid,ask,time))
              {
               ReturnFalse;
              }
           }
         if(delta>=lot_min-AG_LOT_EPSILON)
           {
            double m=0;
            if(!OrderCalcMargin(a.volume>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,delta,a.volume>0?ask:bid,m))
              {
               ReturnFalse;
              }
            if(m>equity-margin)
              {
               a.invalid=true;
               return(CloseVirtual(MathAbs(volume),bid,ask,time));
              }
            int n=ArraySize(parts);
            ArrayResize(parts,n+1);
            parts[n].volume=a.volume>0?delta:-delta;
            parts[n].entry=a.volume>0?ask:bid;
            parts[n].time=time;
            trades++;
           }
         sl=a.sl;
         tp=a.tp;
         return(Snapshot(bid,ask,time));
        }
      if(!TerminalInfoInteger(TERMINAL_CONNECTED) && !MQLInfoInteger(MQL_TESTER))
        {
         return(false);
        }
// Close opposite exposure before checking incremental margin.
      double reduction_left=MathMax(MathAbs(volume)-MathAbs(a.volume),0);
      for(int i=PositionsTotal()-1;i>=0;i--)
        {
         ulong ticket=PositionGetTicket(i);
         if(ticket==0)
           {
            ReturnFalse;
           }
         bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
         double q=PositionGetDouble(POSITION_VOLUME);
         if(a.volume==0 || (buy && a.volume<0) || (!buy && a.volume>0))
           {
            bool sent=trade.PositionClose(ticket);
            if(!ConfirmRequest(sent))
              {
               return(false);
              }
            if(pending)
              {
               return(true);
              }
           }
         else
           {
            if(reduction_left>=lot_step-AG_LOT_EPSILON)
              {
// Netting reduction uses an opposite deal; hedging closes by ticket.
               double reduction=MathMin(q,reduction_left);
               reduction_left-=reduction;
               bool hedging=AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;
               bool ok=hedging?(reduction>=q-AG_LOT_EPSILON?trade.PositionClose(ticket):trade.PositionClosePartial(ticket,reduction)):(buy?trade.Sell(reduction,_Symbol):trade.Buy(reduction,_Symbol));
               if(!ConfirmRequest(ok))
                 {
                  if(trade.ResultRetcode()==TRADE_RETCODE_INVALID_VOLUME)
                    {
                     a.invalid=true;
                     a.volume=0;
                     return(Execute(a,bid,ask,time));
                    }
                  return(false);
                 }
               if(pending)
                 {
                  return(true);
                 }
              }
           }
        }
      if(!Snapshot(bid,ask,time))
        {
         ReturnFalse;
        }
      if(volume*a.volume<0 || (a.volume==0 && volume!=0))
        {
         reconciling=true;
         return(true);
        }
      if(a.volume==0)
        {
         if(volume==0 && !pending)
           {
            reconciling=false;
           }
         return(volume==0 && !pending);
        }
      for(int i=PositionsTotal()-1;i>=0;i--)
        {
         ulong ticket=PositionGetTicket(i);
         bool modified=trade.PositionModify(ticket,a.sl,a.tp);
         if(!ConfirmRequest(modified))
           {
            if(PositionGetDouble(POSITION_SL)<=0 || PositionGetDouble(POSITION_TP)<=0)
              {
               bool closed=trade.PositionClose(ticket);
               if(!ConfirmRequest(closed))
                 {
                  return(false);
                 }
               if(pending)
                 {
                  return(true);
                 }
               if(!Snapshot(bid,ask,time))
                 {
                  ReturnFalse;
                 }
              }
            return(false);
           }
         if(pending)
           {
            return(true);
           }
        }
      double delta=MathAbs(a.volume)-MathAbs(volume);
      if(delta>=lot_min-AG_LOT_EPSILON)
        {
         double m=0;
         if(!OrderCalcMargin(a.volume>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,delta,a.volume>0?ask:bid,m))
           {
            ReturnFalse;
           }
         if(m>AccountInfoDouble(ACCOUNT_MARGIN_FREE))
           {
            a.invalid=true;
            SAlphaGenTarget flat;
            ZeroMemory(flat);
            return(Execute(flat,bid,ask,time));
           }
         bool ok=a.volume>0?trade.Buy(delta,_Symbol,0,a.sl,a.tp):trade.Sell(delta,_Symbol,0,a.sl,a.tp);
         MqlTradeResult result;
         trade.Result(result);
         RecordRequest(result);
         request_id=result.request_id;
         uint code=result.retcode;
         pending=(code==TRADE_RETCODE_PLACED);
         if(code==TRADE_RETCODE_TIMEOUT || code==TRADE_RETCODE_CONNECTION)
           {
            unknown_request=true;
            pending=true;
            return(true);
           }
         if(!ok || (code!=TRADE_RETCODE_DONE && code!=TRADE_RETCODE_DONE_PARTIAL && code!=TRADE_RETCODE_PLACED))
           {
            return(false);
           }
         if(code==TRADE_RETCODE_DONE || code==TRADE_RETCODE_DONE_PARTIAL)
           {
            successful_requests++;
           }
         reconciling=(code==TRADE_RETCODE_DONE_PARTIAL || code==TRADE_RETCODE_PLACED);
        }
      intended_volume=a.volume;
      intended_sl=a.sl;
      intended_tp=a.tp;
      return(Snapshot(bid,ask,time));
     }
//+------------------------------------------------------------------+
//| Apply opening gaps and protection at an available price.         |
//+------------------------------------------------------------------+
   bool PathPoint(double bid,double spread,datetime time)
     {
      if(!Snapshot(bid,bid+spread,time))
        {
         ReturnFalse;
        }
      bool hit=volume>0?(bid<=sl || bid>=tp):(volume<0 && (bid+spread>=sl || bid+spread<=tp));
      if(hit || equity<=AG_EQUITY_FLOOR*reference)
        {
         return(CloseVirtual(MathAbs(volume),bid,bid+spread,time));
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Find the first protection or equity-floor crossing.              |
//+------------------------------------------------------------------+
   bool Segment(double from,double to,double spread,datetime time)
     {
      if(volume==0)
        {
         return(true);
        }
      double end=to,nearest=AG_CONST_TWO;
      double levels[3];
      levels[0]=volume>0?sl:sl-spread;
      levels[1]=volume>0?tp:tp-spread;
      double slope=MathAbs(volume)*(volume>0?k_buy:-k_sell);
// Equity is linear between OHLC vertices while exposure is unchanged.
      levels[2]=from+(AG_EQUITY_FLOOR*reference-equity)/slope;
      for(int i=0;i<3;i++)
        {
         if(to==from)
           {
            continue;
           }
         double fraction=(levels[i]-from)/(to-from);
         if(fraction>=0 && fraction<=1 && fraction<nearest)
           {
            nearest=fraction;
            end=levels[i];
           }
        }
      if(!PathPoint(end,spread,time))
        {
         ReturnFalse;
        }
      if(nearest<=1 && volume!=0)
        {
         if(!CloseVirtual(MathAbs(volume),end,end+spread,time))
           {
            ReturnFalse;
           }
        }
      return(PathPoint(to,spread,time));
     }
//+------------------------------------------------------------------+
//| Advance one causal historical OHLC execution path.               |
//+------------------------------------------------------------------+
   bool Bar(MqlRates &bar)
     {
      double spread=bar.spread*point;
      if(!PathPoint(bar.open,spread,bar.time))
        {
         ReturnFalse;
        }
      double first=volume>=0?bar.low:bar.high,second=volume>=0?bar.high:bar.low;
      if(!Segment(bar.open,first,spread,bar.time) || !Segment(first,second,spread,bar.time) || !Segment(second,bar.close,spread,bar.time))
        {
         ReturnFalse;
        }
      return(Snapshot(bar.close,bar.close+spread,bar.time+PeriodSeconds(TimeFrame)));
     }
//+------------------------------------------------------------------+
//| Check minimum-lot margin in every permitted direction.           |
//+------------------------------------------------------------------+
   bool Exhausted(double bid,double ask)
     {
      if(volume!=0 || pending)
        {
         return(false);
        }
      bool any=false;
      for(int side=0;side<2;side++)
        {
         if(trade_mode!=SYMBOL_TRADE_MODE_FULL && trade_mode!=(side==0?SYMBOL_TRADE_MODE_LONGONLY:SYMBOL_TRADE_MODE_SHORTONLY))
           {
            continue;
           }
         double m=0;
         if(!OrderCalcMargin(side==0?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,lot_min,side==0?ask:bid,m))
           {
            return(false);
           }
         any=true;
         if(m<=equity-margin)
           {
            return(false);
           }
        }
      return(any);
     }
  };
//+------------------------------------------------------------------+
//| Shared native model, chronological driver and finalizer.         |
//+------------------------------------------------------------------+
struct SAlphaGenDeal
  {
   ulong ticket;
   double cash;
   long milliseconds;
   ulong owner;
  };
struct SAlphaGenInterval
  {
   ulong decision;
   datetime start_boundary,end_boundary;
   long start_msc,end_msc;
   double reference,gamma,start_equity,end_equity;
   int duration;
   bool trained;
  };
struct SAlphaGenRequestOwner
  {
   ulong deal,order,decision;
  };
class CAlphaGenApplication
  {
   public:
   CNet model;
   SAlphaGenDeal ledger[];
   SAlphaGenInterval sealed[];
   SAlphaGenRequestOwner request_owners[];
   long snapshot_msc,decision_start_msc;
   bool fault_announced,fault_settled;
   long trained_money_msc;
   datetime confirmed_boundary,finance_started;
   double policy_cash,service_cash,excluded_cash,reported_equity,external_cash;
   CAlphaGenEnvironment money;
   CBufferFloat service,raw;
   CNeuronAlphaGenOCL *agent;
   datetime LastBarTime,decision_time;
   ulong decision_id,event_id,updates,excluded;
   bool training,simulated,Finalized,failed,stop_requested,open_decision,invalid,cycle_complete,service_rebase;
   datetime observed_boundary,sample_boundary,pool_warmup_boundary,virtual_open_boundary;
   bool virtual_gap_terminal,controlled_stopping;
   int pool_warmup_end;
   int study_finish_stage,study_finish_reason,study_finish_contour;
   bool study_finish_ruin,study_finish_saved;
   int study_finish_batch_total,study_finish_batch_completed;
   ulong study_finish_updates,study_finish_started;
   double start_equity,last_equity,interval_reward,trade_gamma;
   int delta;
   float input_values[],context[];
   MqlRates history[],pool_warmup_rates[];
   int cursor,epoch,window,context_bars,warmup_bars;
   ulong prep_us,forward_us,ppo_us,total_us,max_us,boundaries;
   ulong prep_calls,forward_calls,ppo_calls,max_prep,max_forward,max_ppo;
//+------------------------------------------------------------------+
//| Initialize owned runtime state and diagnostic counters.          |
//+------------------------------------------------------------------+
   CAlphaGenApplication():trained_money_msc(0),
   confirmed_boundary(0),
   snapshot_msc(0),
   decision_start_msc(0),
   fault_announced(false),
   fault_settled(false),
   policy_cash(0),
   service_cash(0),
   excluded_cash(0),
   reported_equity(0),
   external_cash(0),
   agent(NULL),
   LastBarTime(0),
   decision_time(0),
   decision_id(0),
   event_id(0),
   updates(0),
   excluded(0),
   Finalized(false),
   failed(false),
   stop_requested(false),
   open_decision(false),
   invalid(false),
   cycle_complete(false),
   observed_boundary(0),
   sample_boundary(0),
   pool_warmup_boundary(0),
   virtual_open_boundary(0),
   virtual_gap_terminal(false),
   controlled_stopping(false),
   pool_warmup_end(0),
   study_finish_stage(0),
   study_finish_reason(0),
   study_finish_contour(0),
   study_finish_ruin(false),
   study_finish_saved(false),
   study_finish_batch_total(0),
   study_finish_batch_completed(0),
   study_finish_updates(0),
   study_finish_started(0),
   trade_gamma(AG_TRADE_GAMMA),
   cursor(0),
   epoch(0),
   window(AG_WINDOW),
   context_bars(AG_CONTEXT_BARS),
   warmup_bars(AG_WARMUP_BARS),
   prep_us(0),
   forward_us(0),
   ppo_us(0),
   total_us(0),
   max_us(0),
   boundaries(0),
   prep_calls(0),
   forward_calls(0),
   ppo_calls(0),
   max_prep(0),
   max_forward(0),
   max_ppo(0)
     {
     }
//+------------------------------------------------------------------+
//| Transfer the single defines profile to the native configuration. |
//+------------------------------------------------------------------+
   bool RememberRequests()
     {
      for(int i=0;i<ArraySize(money.request_deals);i++)
        {
         int n=ArraySize(request_owners);
         if(ArrayResize(request_owners,n+1)!=n+1)
           {
            return(false);
           }
         request_owners[n].deal=money.request_deals[i];
         request_owners[n].order=money.request_orders[i];
         request_owners[n].decision=open_decision?decision_id:0;
        }
      ArrayResize(money.request_deals,0);
      ArrayResize(money.request_orders,0);
      return(true);
     }
//+------------------------------------------------------------------+
//| Bind each confirmed request result to its actual policy decision.|
//+------------------------------------------------------------------+
   bool ExecuteTarget(SAlphaGenTarget &target,double bid,double ask,datetime time)
     {
      bool result=money.Execute(target,bid,ask,time);
      if(!RememberRequests())
        {
         return(false);
        }
      return(result);
     }
//+------------------------------------------------------------------+
//| Reconcile protective closing without using the failed model.     |
//+------------------------------------------------------------------+
   void FaultCleanup()
     {
      if(!failed || fault_settled)
        {
         return;
        }
      if(simulated)
        {
         fault_settled=true;
         return;
        }
      if(!MQLInfoInteger(MQL_TESTER) && !TerminalInfoInteger(TERMINAL_CONNECTED))
        {
         return;
        }
      MqlTick quote;
      if(!SymbolInfoTick(_Symbol,quote) || quote.bid<=0 || quote.ask<quote.bid)
        {
         return;
        }
      money.Snapshot(quote.bid,quote.ask,quote.time);
      if(money.pending)
        {
         return;
        }
      SAlphaGenTarget flat;
      ZeroMemory(flat);
      money.intended_volume=0;
      money.intended_sl=0;
      money.intended_tp=0;
      ExecuteTarget(flat,quote.bid,quote.ask,quote.time);
      if(money.Snapshot(quote.bid,quote.ask,quote.time) && money.volume==0 && !money.pending)
        {
         fault_settled=true;
         Print("AlphaGen fault exposure confirmed Flat; checkpoint preserved");
        }
     }
//+------------------------------------------------------------------+
//| Disable the model and saving before attempting protective closing.|
//+------------------------------------------------------------------+
   void Fault(string reason)
     {
      failed=true;
      if(!fault_announced)
        {
         fault_announced=true;
         Print("AlphaGen controlled fault: ",reason,"; new ACT and Save disabled");
        }
      FaultCleanup();
     }
//+------------------------------------------------------------------+
//| Retain financial provenance until the first trade PPO update.    |
//+------------------------------------------------------------------+
   bool SealInterval(datetime boundary)
     {
      int n=ArraySize(sealed);
      if(ArrayResize(sealed,n+1)!=n+1)
        {
         return(false);
        }
      sealed[n].decision=decision_id;
      sealed[n].start_boundary=decision_time;
      sealed[n].end_boundary=boundary;
      sealed[n].start_msc=decision_start_msc;
      sealed[n].end_msc=snapshot_msc>0?snapshot_msc:(long)boundary*1000;
      sealed[n].reference=money.reference;
      sealed[n].gamma=trade_gamma;
      sealed[n].start_equity=start_equity;
      sealed[n].end_equity=money.equity;
      sealed[n].duration=MathMax(delta,1);
      sealed[n].trained=false;
      return(true);
     }
//+------------------------------------------------------------------+
//| Attribute a late cash change to its original sealed transition.  |
//+------------------------------------------------------------------+
   bool CorrectSealedCash(ulong owner,long at,double difference)
     {
      if(!training || difference==0)
        {
         return(true);
        }
      int match=-1;
      for(int i=0;i<ArraySize(sealed);i++)
        {
         bool candidate=owner!=0?sealed[i].decision==owner:
         (at>=sealed[i].start_msc && at<=sealed[i].end_msc);
         if(!candidate)
           {
            continue;
           }
         if(match>=0)
           {
            return(false);
           }
         match=i;
        }
      if(match<0)
        {
// Current unsealed cash is already represented by its equity sample.
         if(owner==decision_id && open_decision)
           {
            return(true);
           }
         if(open_decision && at<decision_start_msc)
           {
            return(false);
           }
         if(ArraySize(sealed)==0 || at>sealed[ArraySize(sealed)-1].end_msc)
           {
            return(true);
           }
         return(false);
        }
      if(sealed[match].trained || sealed[match].reference<=0)
        {
         return(false);
        }
      int start_shift=iBarShift(_Symbol,TimeFrame,sealed[match].start_boundary,true);
      int cash_shift=iBarShift(_Symbol,TimeFrame,(datetime)((at-1)/1000),false);
      if(start_shift<0 || cash_shift<0)
        {
         return(false);
        }
      int cash_step=MathMax(start_shift-cash_shift+1,1);
      if(cash_step>sealed[match].duration)
        {
         return(false);
        }
      double adjustment=MathPow(sealed[match].gamma,cash_step-1)*difference/sealed[match].reference;
      if(!MathIsValidNumber(adjustment) || !agent.AdjustReward(sealed[match].decision,(float)adjustment))
        {
         return(false);
        }
// The current confirmed equity already includes this cash correction.
// Remove it from the new interval rather than rewarding it twice.
      if(open_decision)
        {
         last_equity+=difference;
         start_equity+=difference;
        }
      reported_equity+=difference;
      policy_cash+=difference;
      sealed[match].end_equity+=difference;
      for(int i=match+1;i<ArraySize(sealed);i++)
        {
         sealed[i].start_equity+=difference;
         sealed[i].end_equity+=difference;
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Transfer the single defines profile to the native configuration. |
//+------------------------------------------------------------------+
   bool Config(float &c[])
     {
      ArrayResize(c,AG_CFG_SIZE);
      ArrayInitialize(c,0);
      c[AG_CFG_W]=(float)AG_WINDOW;
      c[AG_CFG_LOOKBACK]=(float)AG_LOOKBACK;
      c[AG_CFG_HORIZON]=(float)AG_HORIZON;
      c[AG_CFG_IC_WINDOW]=(float)AG_IC_WINDOW;
      c[AG_CFG_NORM_WINDOW]=(float)AG_NORM_WINDOW;
      c[AG_CFG_POOL]=(float)AG_POOL_SIZE;
      c[AG_CFG_MAX_TOKENS]=(float)AG_MAX_TOKENS;
      c[AG_CFG_CONTEXT]=(float)AG_CONTEXT_BARS;
      c[AG_CFG_EMBEDDING]=(float)AG_EMBEDDING;
      c[AG_CFG_SEARCH_HIDDEN]=(float)AG_SEARCH_HIDDEN;
      c[AG_CFG_TRADE_HIDDEN]=(float)AG_TRADE_HIDDEN;
      c[AG_CFG_MLP]=(float)AG_MLP_HIDDEN;
      c[AG_CFG_EPS_VAR]=(float)AG_EPSILON_VAR;
      c[AG_CFG_ABS_TOL]=(float)AG_ABSOLUTE_TOL;
      c[AG_CFG_REL_TOL]=(float)AG_RELATIVE_TOL;
      c[AG_CFG_RIDGE]=(float)AG_RIDGE;
      c[AG_CFG_SIGNAL_CLIP]=(float)AG_SIGNAL_CLIP;
      c[AG_CFG_SIGMA_FLOOR]=(float)AG_SIGMA_FLOOR;
      c[AG_CFG_PPO_CLIP]=(float)AG_CLIP;
      c[AG_CFG_VALUE_COEF]=(float)AG_VALUE_COEF;
      c[AG_CFG_ENTROPY_COEF]=(float)AG_ENTROPY_COEF;
      c[AG_CFG_TRADE_GAMMA]=(float)AG_TRADE_GAMMA;
      c[AG_CFG_LAMBDA]=(float)AG_LAMBDA;
      c[AG_CFG_TOKEN_GAMMA]=(float)AG_TOKEN_GAMMA;
      c[AG_CFG_PASSES]=(float)AG_PPO_PASSES;
      c[AG_CFG_TRADE_ROLLOUT]=(float)AG_TRADING_ROLLOUT;
      c[AG_CFG_TOKEN_ROLLOUT]=(float)AG_TOKEN_ROLLOUT;
      c[AG_CFG_WARMUP_TRIALS]=(float)AG_WARMUP_TRIALS;
      c[AG_CFG_SEARCH_TRIALS]=(float)AG_SEARCH_TRIALS;
      c[AG_CFG_CONSTANTS]=(float)AG_CONST_NEGATIVE_ONE;
      c[AG_CFG_CONSTANTS+1]=(float)AG_CONST_NEGATIVE_HALF;
      c[AG_CFG_CONSTANTS+2]=(float)AG_CONST_HALF;
      c[AG_CFG_CONSTANTS+3]=(float)AG_CONST_ONE;
      c[AG_CFG_CONSTANTS+4]=(float)AG_CONST_TWO;
      c[AG_CFG_INTERVALS]=(float)AG_INTERVAL_ONE;
      c[AG_CFG_INTERVALS+1]=(float)AG_INTERVAL_FIVE;
      c[AG_CFG_INTERVALS+2]=(float)AG_INTERVAL_TEN;
      c[AG_CFG_INTERVALS+3]=(float)AG_INTERVAL_TWENTY;
      c[AG_CFG_INTERVALS+4]=(float)AG_INTERVAL_SIXTY;
      c[AG_CFG_POSITION_SCALE]=(float)AG_POSITION_SCALE;
      c[AG_CFG_INVALID_TOKEN_REWARD]=(float)AG_INVALID_TOKEN_REWARD;
      return(true);
     }
//+------------------------------------------------------------------+
//| Validate the run and initialize the shared state.                |
//+------------------------------------------------------------------+
   bool Init(bool train,bool offline)
     {
      training=train;
      service_rebase=true;
      simulated=offline;
      LastBarTime=0;
      finance_started=TimeCurrent();
      if(window<AG_LOOKBACK+MathMax(AG_IC_WINDOW+AG_HORIZON,AG_NORM_WINDOW))
        {
         ReturnFalse;
        }
      if(train && !offline && !MQLInfoInteger(MQL_TESTER))
        {
         Print("StudyOnline requires Strategy Tester");
         ReturnFalse;
        }
      float error=0,undefined=0,forecast=0;
      datetime stamp=0;
      bool exists=FileIsExist(ModelFile,FILE_COMMON);
      if(exists)
        {
         if(!model.Load(ModelFile,error,undefined,forecast,stamp,true))
           {
            ReturnFalse;
           }
        }
      else
        {
         if(!train)
           {
            Print("AlphaGen checkpoint is required");
            ReturnFalse;
           }
         ResetLastError();
         CArrayObj description;
         CLayerDescription *in=new CLayerDescription(),*out=new CLayerDescription();
         if(!in || !out)
           {
            if(in)
              {
               DeleteObj(in)
              }
            if(out)
              {
               DeleteObj(out)
              }
            ReturnFalse;
           }
         in.type=defNeuronBaseOCL;
         in.count=AG_CHANNELS*window+AG_ACCOUNT_SIZE;
         in.activation=None;
         in.optimization=ADAM;
         out.type=defNeuronAlphaGenOCL;
         out.count=AG_OUTPUT_SIZE;
         out.activation=None;
         out.optimization=ADAM;
         out.window=window;
         out.window_out=AG_POOL_SIZE;
         out.step=AG_LOOKBACK;
         out.layers=AG_MAX_TOKENS;
         description.Add(in);
         description.Add(out);
         if(!model.Create(GetPointer(description)))
           {
            ReturnFalse;
           }
        }
      if(!model.Layer(0) || !model.Layer(1) || model.Layer(1).Type()!=defNeuronAlphaGenOCL || model.Layer(1).Neurons()!=AG_OUTPUT_SIZE)
        {
         ReturnFalse;
        }
      agent=(CNeuronAlphaGenOCL*)model.Layer(1);
      if(!exists)
        {
         float config[];
         Config(config);
         if(!agent.Configure(config,Seed))
           {
            ReturnFalse;
           }
        }
      window=(int)agent.ConfigValue(AG_CFG_W);
      context_bars=(int)agent.ConfigValue(AG_CFG_CONTEXT);
      trade_gamma=(double)agent.ConfigValue(AG_CFG_TRADE_GAMMA);
      int lookback=(int)agent.ConfigValue(AG_CFG_LOOKBACK),norm=(int)agent.ConfigValue(AG_CFG_NORM_WINDOW);
      int ic=(int)agent.ConfigValue(AG_CFG_IC_WINDOW),horizon=(int)agent.ConfigValue(AG_CFG_HORIZON);
      long required=(long)lookback+norm+context_bars-1;
      long inputs=(long)AG_CHANNELS*window+AG_ACCOUNT_SIZE;
      long metadata=(long)AG_CTX_SIZE+AG_CONTEXT_METADATA*window;
      if(!MathIsValidNumber(trade_gamma) || trade_gamma<=0 || trade_gamma>1 || window<=0 ||
         context_bars<=0 || lookback<0 || norm<=0 || ic<=0 || horizon<=0 || required>INT_MAX ||
         inputs>INT_MAX || metadata>INT_MAX || (long)window<(long)lookback+MathMax((long)ic+horizon,(long)norm) ||
         model.Layer(0).Neurons()!=AG_CHANNELS*window+AG_ACCOUNT_SIZE)
        {
         ReturnFalse;
        }
      warmup_bars=(int)MathMax(window,required);
      if(!model.TrainMode(train))
        {
         ReturnFalse;
        }
      if(ArrayResize(context,(int)metadata)!=(int)metadata || ArrayResize(input_values,(int)inputs)!=(int)inputs)
        {
         ReturnFalse;
        }
      ArrayInitialize(context,0);
      ArrayInitialize(input_values,0);
      double capital=AccountInfoDouble(ACCOUNT_BALANCE);
#ifdef ALPHAGEN_STUDY
      if(offline)
        {
         if(Start>=End || Epochs<1 || MinBalance<=0 || MaxBalance<MinBalance)
           {
            ReturnFalse;
           }
         capital=Deposit();
// Include enough real prehistory; CopyRates never synthesizes missing bars.
         int start_shift=iBarShift(_Symbol,TimeFrame,Start,false);
         if(start_shift<0)
           {
            ReturnFalse;
           }
         int copied=CopyRates(_Symbol,TimeFrame,1,start_shift+warmup_bars,history);
         if(copied<=0)
           {
            ReturnFalse;
           }
         cursor=0;
         while(cursor<copied && history[cursor].time<Start)
           {
            cursor++;
           }
         if(cursor<window || cursor>=copied || history[cursor].time>=End)
           {
            ReturnFalse;
           }
         epoch=1;
        }
#endif
      external_cash=reported_equity=capital;
      if(!money.Init(offline,capital))
        {
         ReturnFalse;
        }
      if(!training && agent.ActiveCount()<1)
        {
         Print("AlphaGen Test requires a checkpoint with a published Active Pool");
         ReturnFalse;
        }
      if(offline && !agent.EnableParallelSearch())
        {
         ReturnFalse;
        }
      PrintFormat("AlphaGen %s model %s | initial seed %I64u | TimeFrame %s | W %d | context %d",exists?"loaded":"created",ModelFile,agent.InitialSeed(),EnumToString(TimeFrame),window,context_bars);
#ifdef ALPHAGEN_STUDY
      PrintFormat("AlphaGen Study requested history [%s, %s), epochs %d, initial virtual capital %.2f",TimeToString(Start),TimeToString(End),Epochs,capital);
#endif
      return(true);
     }
//+------------------------------------------------------------------+
//| Record external capital replacement independently of reward.     |
//+------------------------------------------------------------------+
   void ResetFinance()
     {
      double capital=Deposit();
      external_cash+=capital-money.equity;
      money.Reset(capital);
      reported_equity=capital;
      service_rebase=true;
      LastBarTime=0;
      observed_boundary=0;
      virtual_open_boundary=0;
      pool_warmup_end=0;
      if(!agent.ResetWorking())
        {
         Fault("recurrent reset failed after financial episode");
        }
     }
//+------------------------------------------------------------------+
//| Sample capital from the saved independent financial RNG.         |
//+------------------------------------------------------------------+
   double Deposit()
     {
#ifdef ALPHAGEN_STUDY
      return(MinBalance+(MaxBalance-MinBalance)*agent.FinancialUniform());
#else
      return(AccountInfoDouble(ACCOUNT_BALANCE));
#endif
     }
//+------------------------------------------------------------------+
//| Encode exact identifiers and publish CPU service metadata.       |
//+------------------------------------------------------------------+
   bool Context(int phase,int flags,datetime boundary)
     {
      context[AG_CTX_PHASE]=(float)phase;
      context[AG_CTX_FLAGS]=(float)flags;
      AlphaGenInteger(context,AG_CTX_DECISION,decision_id);
      AlphaGenInteger(context,AG_CTX_EVENT,event_id);
      AlphaGenInteger(context,AG_CTX_BOUNDARY,(ulong)boundary);
      context[AG_CTX_EQUITY]=(float)money.equity;
      context[AG_CTX_REFERENCE]=(float)money.reference;
      if(!service.AssignArray(context) || !(service.GetIndex()<0?service.BufferCreate(model.GetOpenCL()):service.BufferWrite()))
        {
         ReturnFalse;
        }
      agent.Phase(phase);
      return(true);
     }
//+------------------------------------------------------------------+
//| Run one native phase and propagate numerical failures.           |
//+------------------------------------------------------------------+
   bool Forward(int phase,int flags,datetime boundary)
     {
      if(failed)
        {
         ReturnFalse;
        }
      ulong start=GetMicrosecondCount();
      if(!Context(phase,flags,boundary) || !raw.AssignArray(input_values) || !model.feedForward(GetPointer(raw),1,false,GetPointer(service)))
        {
         Fault("native forward failed");
         ReturnFalse;
        }
      ulong elapsed=GetMicrosecondCount()-start;
      forward_us+=elapsed;
      max_forward=MathMax(max_forward,elapsed);
      forward_calls++;
      if(agent.Status()==AG_CORRUPTED || agent.Status()==AG_TECHNICAL)
        {
         Fault("native numerical state is unusable");
         ReturnFalse;
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Apply one native PPO example or drain a terminal ready tail.     |
//+------------------------------------------------------------------+
   bool Train(bool drain=false)
     {
      if(!training)
        {
         return(true);
        }
      ulong begin=GetMicrosecondCount();
      while(agent.HasTraining())
        {
         if(!agent.PrepareUpdate())
           {
            Fault("native PPO preparation failed");
            ReturnFalse;
           }
         if(!Forward(AG_BACKWARD,0,LastBarTime))
           {
            ReturnFalse;
           }
         CBufferFloat echo;
         float zeros[];
         ArrayResize(zeros,AG_OUTPUT_SIZE);
         ArrayInitialize(zeros,0);
         ulong previous_trade_updates=agent.TradeUpdates();
         if(!echo.AssignArray(zeros) || !model.backProp(GetPointer(echo),GetPointer(service)))
           {
            Fault("native PPO backward failed");
            ReturnFalse;
           }
         updates++;
         if(agent.TradeUpdates()>previous_trade_updates)
           {
            for(int i=0;i<ArraySize(sealed);i++)
              {
               if(sealed[i].trained)
                 {
                  continue;
                 }
               sealed[i].trained=true;
               trained_money_msc=MathMax(trained_money_msc,sealed[i].end_msc);
              }
           }
         if(!drain)
           {
            break;
           }
        }
      ulong elapsed=GetMicrosecondCount()-begin;
      ppo_us+=elapsed;
      max_ppo=MathMax(max_ppo,elapsed);
      ppo_calls++;
      return(true);
     }
//+------------------------------------------------------------------+
//| Pack only closed OHLCV bars and the current Account13.           |
//+------------------------------------------------------------------+
   bool Pack(MqlRates &bars[],int end,datetime boundary,bool account)
     {
      ulong begin=GetMicrosecondCount();
      if(end<window || end>ArraySize(bars))
        {
         return(false);
        }
      for(int i=0;i<window;i++)
        {
         MqlRates bar=bars[end-window+i];
         if(!MathIsValidNumber(bar.open+bar.high+bar.low+bar.close) || bar.open<=0 || bar.high<bar.low || bar.close<=0 || bar.spread<0)
           {
            return(false);
           }
         input_values[i]=(float)bar.open;
         input_values[window+i]=(float)bar.high;
         input_values[2*window+i]=(float)bar.low;
         input_values[3*window+i]=(float)bar.close;
         input_values[4*window+i]=(float)bar.tick_volume;
         AlphaGenInteger(context,AG_CTX_SIZE+i*AG_CONTEXT_METADATA,(ulong)bar.time);
         context[AG_CTX_SIZE+i*AG_CONTEXT_METADATA+AG_TIME_LIMBS]=(float)bar.spread;
        }
      float a[];
      datetime account_time=simulated?boundary:(snapshot_msc>0?(datetime)(snapshot_msc/1000):TimeCurrent());
      if(account && !money.Account(account_time,a))
        {
         return(false);
        }
      for(int i=0;i<AG_ACCOUNT_SIZE;i++)
        {
         input_values[AG_CHANNELS*window+i]=account?a[i]:0.0f;
        }
      ulong elapsed=GetMicrosecondCount()-begin;
      prep_us+=elapsed;
      max_prep=MathMax(max_prep,elapsed);
      prep_calls++;
      return(true);
     }
//+------------------------------------------------------------------+
//| Append confirmed virtual equity at its actual bar duration.      |
//+------------------------------------------------------------------+
   void RealMoneySample(bool terminal=false)
     {
      if(simulated || !training || !open_decision)
        {
         return;
        }
      datetime at=snapshot_msc>0?(datetime)(snapshot_msc/1000):TimeCurrent();
      int start=iBarShift(_Symbol,TimeFrame,decision_time,true);
      int finish=iBarShift(_Symbol,TimeFrame,at,false);
      if(start<0 || finish<0)
        {
         return;
        }
// Intrabar risk marks are not reward observations. A terminal event belongs
// to the right boundary of its bar, with exact boundary events ending there.
      int step=start-finish;
      if(terminal && snapshot_msc>(long)iTime(_Symbol,TimeFrame,finish)*1000)
        {
         step++;
        }
      delta=MathMax(step,1);
      interval_reward+=MathPow(trade_gamma,delta-1)*(money.equity-last_equity)/money.reference;
      last_equity=money.equity;
     }
//+------------------------------------------------------------------+
//| Append confirmed virtual equity at its actual bar duration.      |
//+------------------------------------------------------------------+
   void MoneySample()
     {
      if(!training || !open_decision)
        {
         return;
        }
      delta++;
      interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
      last_equity=money.equity;
     }
//+------------------------------------------------------------------+
//| Confirm one real transition without fabricating bootstrap data.  |
//+------------------------------------------------------------------+
   bool Feedback(bool terminal,bool ready,bool ruin,datetime boundary)
     {
      if(!training || !open_decision)
        {
         return(true);
        }
      event_id++;
      context[AG_CTX_MONEY_CONFIRMED]=(!money.pending && !money.reconciling && !money.unknown_request)?1.0f:0.0f;
      context[AG_CTX_TERMINAL]=terminal?1.0f:0.0f;
      context[AG_CTX_NEXT_READY]=ready?1.0f:0.0f;
      context[AG_CTX_DELTA]=(float)MathMax(delta,1);
      context[AG_CTX_REWARD]=(float)(interval_reward-(invalid?AG_INVALID_PENALTY:0)-MathPow(trade_gamma,MathMax(delta-1,0))*(ruin?AG_RUIN_PENALTY:0));
      if(!Forward(AG_FLUSH,terminal?AG_FLAG_END:0,boundary))
        {
         ReturnFalse;
        }
      if(!money.pending && !money.reconciling && !money.unknown_request && (terminal || ready))
        {
         if(!SealInterval(boundary))
           {
            Fault("financial interval ledger allocation failed");
            ReturnFalse;
           }
         policy_cash+=money.equity-start_equity;
         reported_equity=money.equity;
         confirmed_boundary=boundary;
         open_decision=false;
         invalid=false;
         context[AG_CTX_MONEY_CONFIRMED]=context[AG_CTX_TERMINAL]=context[AG_CTX_NEXT_READY]=context[AG_CTX_REWARD]=0;
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Advance market context and finish the causal decision cycle.     |
//+------------------------------------------------------------------+
   bool AdvancePoolWarmup()
     {
      int end=ArraySize(pool_warmup_rates);
      if(pool_warmup_end<window || pool_warmup_end>end)
        {
         Fault("invalid pinned pool warmup cursor");
         ReturnFalse;
        }
      int flags=AG_FLAG_WARMUP;
      if(pool_warmup_end==end-warmup_bars+window)
        {
         flags|=AG_FLAG_RESET_WORK;
        }
      if(pool_warmup_end==end)
        {
         flags|=AG_FLAG_END;
        }
      if(!Pack(pool_warmup_rates,pool_warmup_end,pool_warmup_boundary,false) ||
         !Forward(AG_PREPARE,flags,pool_warmup_boundary))
        {
         ReturnFalse;
        }
      pool_warmup_end++;
      if(agent.NeedsPoolWarmup())
        {
         return(true);
        }
      service_rebase=true;
      if(agent.Status()==AG_PUBLISHED)
        {
         observed_boundary=pool_warmup_boundary;
        }
      LastBarTime=pool_warmup_boundary;
      pool_warmup_end=0;
      pool_warmup_boundary=0;
      ArrayResize(pool_warmup_rates,0);
      cycle_complete=true;
      return(true);
     }
//+------------------------------------------------------------------+
//| Advance market context and finish the causal decision cycle.     |
//+------------------------------------------------------------------+
   bool Boundary(MqlRates &bars[],int end,datetime boundary,double bid,double ask)
     {
      cycle_complete=false;
      if(pool_warmup_end>0)
        {
         return(AdvancePoolWarmup());
        }
      ulong begin=GetMicrosecondCount();
      if(money.pending)
        {
         return(true);
        }
      if(training && agent.HasTraining() && observed_boundary!=boundary)
        {
         if(!Train())
           {
            ReturnFalse;
           }
         return(true);
        }
      if(!Pack(bars,end,boundary,true))
        {
         return(true);
        }
      int count=1;
      datetime market_cursor=observed_boundary>0?observed_boundary:LastBarTime;
      if(market_cursor!=0)
        {
         count=0;
         for(int i=end-1;i>=0 && bars[i].time>=market_cursor;i--)
           {
            count++;
           }
         if(count==0 && observed_boundary!=boundary)
           {
            return(true);
           }
        }
      bool reset=market_cursor==0 || count>window ||
                 (end>=window && market_cursor<bars[end-window].time);
      if(observed_boundary!=boundary)
        {
         int first=reset?end-1:end-count;
         if(first<window-1 || (reset && end<window))
           {
            return(true);
           }
// Fixed-normalizer warmup consumes all valid rows in one W-window.
         for(int i=first;i<end;i++)
           {
            bool latest=i==end-1;
            datetime at=latest?boundary:bars[i+1].time;
            if(!reset && at<=observed_boundary)
              {
               continue;
              }
            if(!Pack(bars,i+1,at,latest))
              {
               return(true);
              }
            int flags=reset?(AG_FLAG_WARMUP|AG_FLAG_RESET_WORK|AG_FLAG_END):AG_FLAG_ADVANCE;
            if(latest && open_decision)
              {
               flags|=AG_FLAG_VALUE;
              }
            if(!Forward(AG_OBSERVE,flags,at))
              {
               ReturnFalse;
              }
            if(agent.Status()==AG_OK)
              {
               observed_boundary=at;
              }
           }
         if(agent.Status()==AG_WAIT && (open_decision || !training))
           {
            return(true);
           }
         if(agent.Status()==AG_INVALID)
           {
            SAlphaGenTarget flat;
            ZeroMemory(flat);
            if(!ExecuteTarget(flat,bid,ask,boundary) || money.pending || money.reconciling || money.volume!=0)
              {
               return(true);
              }
            if(open_decision)
              {
               interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
               last_equity=money.equity;
               if(!Feedback(true,false,false,boundary))
                 {
                  ReturnFalse;
                 }
              }
            else if(training)
              {
               service_rebase=true;
               event_id++;
               context[AG_CTX_TERMINAL]=1.0f;
               context[AG_CTX_MONEY_CONFIRMED]=1.0f;
               context[AG_CTX_NEXT_READY]=0.0f;
               if(!Forward(AG_FLUSH,0,boundary))
                 {
                  ReturnFalse;
                 }
               context[AG_CTX_TERMINAL]=0.0f;
               context[AG_CTX_MONEY_CONFIRMED]=0.0f;
              }
           }
         else
           {
            if(open_decision && agent.Status()!=AG_OK)
              {
               return(true);
              }
            else
              {
               if(open_decision)
                 {
                  RealMoneySample();
                  if(!Feedback(false,true,false,boundary))
                    {
                     ReturnFalse;
                    }
                 }
              }
           }
         observed_boundary=boundary;
        }
      if(training)
        {
         bool preparation_done=false;
         if(agent.HasTraining())
           {
            if(!Train())
              {
               ReturnFalse;
              }
            if(agent.HasTraining())
              {
               return(true);
              }
           }
         if(agent.NeedsPoolWarmup() && end<warmup_bars)
           {
            if(!agent.AbortPoolWarmup())
              {
               ReturnFalse;
              }
            pool_warmup_end=0;
            preparation_done=true;
           }
         if(agent.NeedsPoolWarmup())
           {
// Pin the closed history and publication boundary for this bounded job.
// New bars are caught up only after commit; no stale ACT is issued.
            if(ArrayCopy(pool_warmup_rates,bars,0,0,end)!=end)
              {
               Fault("pool warmup snapshot allocation failed");
               ReturnFalse;
              }
            pool_warmup_boundary=boundary;
            pool_warmup_end=end-warmup_bars+window;
            return(AdvancePoolWarmup());
           }
         if(!preparation_done)
           {
            if(!Pack(bars,end,boundary,true) || !Forward(AG_PREPARE,0,boundary))
              {
               ReturnFalse;
              }
            if(agent.Status()==AG_WORKING || agent.NeedsPoolWarmup())
              {
               return(true);
              }
            if(agent.Status()==AG_WAIT)
              {
               return(true);
              }
            if(!Train())
              {
               ReturnFalse;
              }
            if(agent.HasTraining())
              {
               return(true);
              }
            if(agent.Status()==AG_PUBLISHED)
              {
               service_rebase=true;
               LastBarTime=boundary;
               cycle_complete=true;
               return(true);
              }
           }
        }
      LastBarTime=boundary;
      cycle_complete=true;
      if(service_rebase)
        {
         money.Commit();
        }
      if(!Pack(bars,end,boundary,true))
        {
         return(true);
        }
      service_cash+=money.equity-reported_equity;
      reported_equity=money.equity;
      money.Commit();
      decision_id++;
      if(!Forward(training?AG_ACT:AG_INFER,0,boundary))
        {
         ReturnFalse;
        }
#ifdef ALPHAGEN_TEST_DIAGNOSTICS
      PrintFormat("AlphaGen Test decision %s status=%d active=%d prepared=%d",
                  TimeToString(boundary),agent.Status(),agent.ActiveCount(),agent.PreparedCount());
#endif
      if(agent.Status()!=AG_ACTION_READY)
        {
         service_rebase=true;
         if(agent.Status()==AG_INVALID)
           {
            SAlphaGenTarget flat;
            ZeroMemory(flat);
            if(!ExecuteTarget(flat,bid,ask,boundary))
              {
               return(true);
              }
           }
         return(true);
        }
      service_rebase=false;
      float action[];
      model.getResults(action);
      SAlphaGenTarget target;
      if(!money.Map(action,bid,ask,target))
        {
         ReturnFalse;
        }
#ifdef ALPHAGEN_TEST_DIAGNOSTICS
      PrintFormat("AlphaGen Test action [%.9f %.9f %.9f %.9f %.9f %.9f] volume=%.8f invalid=%d sl=%.8f tp=%.8f",
                  action[0],action[1],action[2],action[3],action[4],action[5],
                  target.volume,(int)target.invalid,target.sl,target.tp);
#endif
      start_equity=last_equity=money.equity;
      interval_reward=0;
      delta=0;
      decision_time=boundary;
      decision_start_msc=snapshot_msc>0?snapshot_msc:(long)boundary*1000;
      open_decision=training;
      invalid=target.invalid;
      double before_volume=money.volume;
      ulong before_requests=money.successful_requests;
      bool executed=ExecuteTarget(target,bid,ask,boundary);
      invalid=target.invalid;
      if(!executed && !money.pending && before_volume==money.volume && before_requests==money.successful_requests)
        {
// A rejected technical request is not a fictitious policy sample.
         if(!agent.DiscardDecision())
           {
            Fault("native decision cancellation failed");
            ReturnFalse;
           }
         open_decision=false;
         service_rebase=true;
         excluded++;
        }
      if(!executed && !money.pending && (before_volume!=money.volume || before_requests!=money.successful_requests))
        {
         SAlphaGenTarget flat;
         ZeroMemory(flat);
         if(!ExecuteTarget(flat,bid,ask,boundary) || money.pending)
           {
            money.intended_volume=0;
            money.intended_sl=0;
            money.intended_tp=0;
            money.reconciling=true;
            return(true);
           }
         interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
         last_equity=money.equity;
         if(!Feedback(true,false,false,boundary) || !Train())
           {
            ReturnFalse;
           }
        }
      if(target.invalid && !money.pending && !money.reconciling && money.volume==0)
        {
         interval_reward+=(money.equity-last_equity)/money.reference;
         last_equity=money.equity;
         if(!Feedback(true,false,false,boundary) || !Train())
           {
            ReturnFalse;
           }
        }
      ulong elapsed=GetMicrosecondCount()-begin;
      total_us+=elapsed;
      max_us=MathMax(max_us,elapsed);
      boundaries++;
      return(true);
     }
//+------------------------------------------------------------------+
//| Read back a temporary native checkpoint before replacement.      |
//+------------------------------------------------------------------+
   bool SavePhase(int phase)
     {
      if(!training || failed || agent.Status()==AG_CORRUPTED || !agent.SafeToSave())
        {
         return(false);
        }
      string temp=ModelFile+".tmp",backup=ModelFile+".bak";
      if(phase==0)
        {
         return(model.Save(temp,0,0,0,TimeCurrent(),true));
        }
      if(phase==1)
        {
// Native loading is indivisible. No percentage is available here.
         CNet verify;
         float a=0,b=0,c=0;
         datetime t=0;
         return(verify.Load(temp,a,b,c,t,true) && CheckPointer(verify.Layer(1))!=POINTER_INVALID &&
                verify.Layer(1).Type()==defNeuronAlphaGenOCL);
        }
      if(phase!=2)
        {
         return(false);
        }
      bool had=FileIsExist(ModelFile,FILE_COMMON);
      if(had && !FileCopy(ModelFile,FILE_COMMON,backup,FILE_COMMON|FILE_REWRITE))
        {
         ReturnFalse;
        }
      if(!FileMove(temp,FILE_COMMON,ModelFile,FILE_COMMON|FILE_REWRITE))
        {
         if(had && !FileCopy(backup,FILE_COMMON,ModelFile,FILE_COMMON|FILE_REWRITE))
           {
            Print("AlphaGen checkpoint restore failed: ",GetLastError());
           }
         ReturnFalse;
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Save and verify synchronously outside the Study timer driver.     |
//+------------------------------------------------------------------+
   bool Save()
     {
      for(int phase=0;phase<3;phase++)
        {
         if(!SavePhase(phase))
           {
            return(false);
           }
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Close, reconcile and finalize once before optional saving.       |
//+------------------------------------------------------------------+
   bool FinishExcluded(bool save,datetime time)
     {
      if(training && open_decision)
        {
         excluded++;
         excluded_cash+=money.equity-start_equity;
         reported_equity=money.equity;
         if(!agent.DiscardDecision())
           {
            Fault("unfinished decision cancellation failed");
            ReturnFalse;
           }
         open_decision=false;
         invalid=false;
        }
      context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=context[AG_CTX_NEXT_READY]=context[AG_CTX_REWARD]=0;
      if(training)
        {
         if(stop_requested)
           {
            if(!agent.CancelTraining())
              {
               Fault("unfinished finalization cancellation failed");
               ReturnFalse;
              }
           }
         else
           {
// DISCARD preserves ready predecessors' recorded bootstrap and ends GAE
// at their buffer edge. It never invents money or a value for the open row.
            for(int contour=0;contour<=AG_LEARNING_CONTOURS;contour++)
              {
               if(!Forward(AG_FLUSH,AG_FLAG_DISCARD|AG_FLAG_END,time) || !Train(true))
                  ReturnFalse;
              }
           }
        }
      if(save && !Save())
        {
         return(false);
        }
      Finalized=true;
      Print("AlphaGen terminal money unresolved; unfinished sample excluded; coherent weights retained");
      return(true);
     }
//+------------------------------------------------------------------+
//| Close, reconcile and finalize once before optional saving.       |
//+------------------------------------------------------------------+
   bool Finish(bool ruin,bool save,bool final_pass=false,bool prepare_only=false,bool training_done=false)
     {
      if(failed)
        {
         FaultCleanup();
         return(false);
        }
      if(Finalized)
        {
         return(true);
        }
      double bid=0,ask=0;
      datetime time=LastBarTime;
      if(simulated && virtual_gap_terminal && cursor<ArraySize(history))
        {
         MqlRates current=history[cursor];
         bid=current.open;
         ask=bid+current.spread*money.point;
         time=current.time;
         snapshot_msc=(long)time*1000;
        }
      else
        {
         if(simulated && cursor>0 && cursor<=ArraySize(history))
           {
            MqlRates last=history[cursor-1];
            bid=last.close;
            ask=bid+last.spread*money.point;
            time=last.time+PeriodSeconds(TimeFrame);
            snapshot_msc=(long)time*1000;
           }
         else
           {
            MqlTick quote;
            if(!SymbolInfoTick(_Symbol,quote))
              {
               return(final_pass?FinishExcluded(save,time):false);
              }
            bid=quote.bid;
            ask=quote.ask;
            time=quote.time;
            snapshot_msc=quote.time_msc;
           }
        }
// A stop overrides the old desired exposure, not the unresolved request.
// Execute still waits for pending/unknown reconciliation before sending.
      if(controlled_stopping)
        {
         money.intended_volume=0;
         money.intended_sl=0;
         money.intended_tp=0;
        }
      SAlphaGenTarget flat;
      ZeroMemory(flat);
      bool settled=ExecuteTarget(flat,bid,ask,time);
      if(!money.Snapshot(bid,ask,time))
        {
         settled=false;
        }
      if(!simulated && !ReconcileDeals(time))
        {
         Fault("INCOMPLETE_DATA: terminal cash reconciliation failed");
         ReturnFalse;
        }
      settled=settled && !money.pending && !money.reconciling && !money.unknown_request && money.volume==0;
      if(!settled)
        {
         return(final_pass?FinishExcluded(save,time):false);
        }
      ruin=ruin || money.stopout || controlled_stopping;
      if(training && open_decision)
        {
         if(!simulated)
           {
            RealMoneySample(true);
           }
         else
           {
            interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
            last_equity=money.equity;
           }
         if(!Feedback(true,false,ruin,time))
           {
            ReturnFalse;
           }
        }
      if(prepare_only)
        {
         return(true);
        }
      if(training && !training_done && stop_requested)
        {
         if(!agent.CancelTraining())
           {
            ReturnFalse;
           }
        }
      if(training && !training_done && !stop_requested && !Train(true))
        {
         ReturnFalse;
        }
      if(training && !training_done && !stop_requested)
        {
         for(int i=0;i<AG_LEARNING_CONTOURS;i++)
           {
            context[AG_CTX_TERMINAL]=1.0f;
            context[AG_CTX_MONEY_CONFIRMED]=1.0f;
            context[AG_CTX_NEXT_READY]=0.0f;
            if(!Forward(AG_FLUSH,AG_FLAG_END,time) || !Train(true))
              {
               ReturnFalse;
              }
           }
         context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
        }
      if(!open_decision)
        {
         service_cash+=money.equity-reported_equity;
         reported_equity=money.equity;
        }
      if(save && !Save())
        {
         return(false);
        }
      Finalized=true;
      string profit_factor=money.gross_loss>0?DoubleToString(money.gross_profit/money.gross_loss,6):"undefined (no realized losses)";
      PrintFormat("AlphaGen actual realized cash gross profit %.2f gross loss %.2f cash profit factor %s max equity drawdown %.2f fraction %.6f",
      money.gross_profit,money.gross_loss,profit_factor,money.max_drawdown,money.max_drawdown_fraction);
      PrintFormat("AlphaGen mean/max us preparation %.1f/%I64u forward %.1f/%I64u PPO %.1f/%I64u",
      prep_calls>0?(double)prep_us/prep_calls:0,max_prep,forward_calls>0?(double)forward_us/forward_calls:0,
      max_forward,ppo_calls>0?(double)ppo_us/ppo_calls:0,max_ppo);
      PrintFormat("AlphaGen cash attribution policy %.2f service %.2f excluded %.2f external %.2f",
      policy_cash,service_cash,excluded_cash,external_cash);
      PrintFormat("AlphaGen final balance %.2f equity %.2f updates %I64u excluded %I64u boundaries %I64u mean_us %.1f max_us %I64u",
      money.balance,money.equity,updates,excluded,boundaries,boundaries>0?(double)total_us/boundaries:0,
      max_us);
      return(true);
     }
//+------------------------------------------------------------------+
//| Account for each native deal cash result exactly once.           |
//+------------------------------------------------------------------+
   bool ReconcileDeals(datetime now)
     {
      if(!HistorySelect(finance_started,now))
        {
         money.pending=true;
         return(true);
        }
      int total=HistoryDealsTotal();
      for(int i=0;i<total;i++)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket==0)
           {
            money.pending=true;
            return(true);
           }
         if(HistoryDealGetString(ticket,DEAL_SYMBOL)!=_Symbol)
           {
            continue;
           }
         if(HistoryDealGetInteger(ticket,DEAL_REASON)==DEAL_REASON_SO)
           {
            money.stopout=true;
           }
         double cash=HistoryDealGetDouble(ticket,DEAL_PROFIT)+HistoryDealGetDouble(ticket,
         DEAL_COMMISSION)+HistoryDealGetDouble(ticket,DEAL_SWAP)+HistoryDealGetDouble(ticket,DEAL_FEE);
         long at=HistoryDealGetInteger(ticket,DEAL_TIME_MSC);
         int found=-1;
         for(int j=0;j<ArraySize(ledger);j++)
           {
            if(ledger[j].ticket==ticket)
              {
               found=j;
               break;
              }
           }
         if(found>=0)
           {
            if(ledger[found].cash==cash)
              {
               continue;
              }
            if(!CorrectSealedCash(ledger[found].owner,at,cash-ledger[found].cash))
              {
               return(false);
              }
            if(ledger[found].cash>=0)
              {
               money.gross_profit-=ledger[found].cash;
              }
            else
              {
               money.gross_loss+=ledger[found].cash;
              }
            if(cash>=0)
              {
               money.gross_profit+=cash;
              }
            else
              {
               money.gross_loss-=cash;
              }
            ledger[found].cash=cash;
           }
         else
           {
            ulong owner=0;
            ulong order=(ulong)HistoryDealGetInteger(ticket,DEAL_ORDER);
            for(int j=0;j<ArraySize(request_owners);j++)
              {
               if(request_owners[j].deal==ticket || (order!=0 && request_owners[j].order==order))
                 {
                  owner=request_owners[j].decision;
                  break;
                 }
              }
            if(owner==0 && open_decision && at>decision_start_msc)
              {
               owner=decision_id;
              }
            if(!CorrectSealedCash(owner,at,cash))
              {
               return(false);
              }
            int n=ArraySize(ledger);
            ArrayResize(ledger,n+1);
            ledger[n].ticket=ticket;
            ledger[n].cash=cash;
            ledger[n].milliseconds=at;
            ledger[n].owner=owner;
            if(cash>=0)
              {
               money.gross_profit+=cash;
              }
            else
              {
               money.gross_loss-=cash;
              }
           }
        }
      return(true);
     }
//+------------------------------------------------------------------+
//| Use transaction notifications to trigger reconciliation.         |
//+------------------------------------------------------------------+
   void Transaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
     {
      if(trans.type==TRADE_TRANSACTION_REQUEST && result.request_id==money.request_id)
        {
         MqlTradeResult outcome=result;
         money.RecordRequest(outcome);
         if(!RememberRequests())
           {
            Fault("request ownership allocation failed");
            return;
           }
         money.unknown_request=false;
        }
      if(trans.type==TRADE_TRANSACTION_DEAL_ADD && HistoryDealSelect(trans.deal))
        {
         if(HistoryDealGetInteger(trans.deal,DEAL_REASON)==DEAL_REASON_SO)
           {
            money.stopout=true;
           }
        }
// Order and deal notification order is not a money confirmation.
     }
//+------------------------------------------------------------------+
//| Reconcile risk before processing any new bar boundary.           |
//+------------------------------------------------------------------+
   void Tick()
     {
      if(Finalized)
        {
         return;
        }
      if(failed)
        {
         FaultCleanup();
         return;
        }
      MqlTick quote;
      if(!SymbolInfoTick(_Symbol,quote))
        {
         return;
        }
      snapshot_msc=quote.time_msc;
      if(!money.Snapshot(quote.bid,quote.ask,quote.time))
        {
         Fault("financial snapshot failed");
         return;
        }
      if(!ReconcileDeals(quote.time))
        {
         Fault("INCOMPLETE_DATA: sealed cash has no mutable causal owner");
         return;
        }
      bool ruin=money.stopout || money.equity<=AG_EQUITY_FLOOR*money.reference || (training
      && money.balance<=AG_MIN_TRADING_BALANCE) || money.Exhausted(quote.bid,quote.ask);
      if(ruin || controlled_stopping)
        {
         controlled_stopping=true;
         if(Finish(true,training))
           {
            ExpertRemove();
           }
         return;
        }
      if(money.volume!=0 && (money.sl<=0 || money.tp<=0) && !money.pending)
        {
         SAlphaGenTarget flat;
         ZeroMemory(flat);
         if(!ExecuteTarget(flat,quote.bid,quote.ask,quote.time) || money.pending)
           {
            return;
           }
         if(training && open_decision)
           {
            RealMoneySample(true);
            if(!Feedback(true,false,false,quote.time) || !Train())
              {
               Fault("protective terminal training failed");
               return;
              }
           }
        }
      if(money.reconciling && !money.pending)
        {
         SAlphaGenTarget remaining;
         remaining.volume=money.intended_volume;
         remaining.sl=money.intended_sl;
         remaining.tp=money.intended_tp;
         remaining.invalid=false;
         if(MathAbs(remaining.volume-money.volume)<money.lot_step)
           {
            money.reconciling=false;
           }
         else
           {
            if(!ExecuteTarget(remaining,quote.bid,quote.ask,quote.time))
              {
               SAlphaGenTarget flat;
               ZeroMemory(flat);
               ExecuteTarget(flat,quote.bid,quote.ask,quote.time);
               money.reconciling=false;
              }
           }
        }
      if(training && invalid && open_decision && money.volume==0 && !money.pending && !money.reconciling)
        {
         RealMoneySample(true);
         if(!Feedback(true,false,false,quote.time) || !Train())
           {
            Fault("invalid action terminal reconciliation failed");
            return;
           }
        }
      if(!TerminalInfoInteger(TERMINAL_CONNECTED) && !MQLInfoInteger(MQL_TESTER))
        {
         return;
        }
      datetime current=iTime(_Symbol,TimeFrame,0);
      if(current>0 && current!=sample_boundary)
        {
         RealMoneySample();
         sample_boundary=current;
        }
      if(current<=0 || (current==LastBarTime && pool_warmup_end==0) || money.pending || money.reconciling)
        {
         return;
        }
      MqlRates bars[];
      int required=warmup_bars;
      if(LastBarTime!=0)
        {
         int missing=iBarShift(_Symbol,TimeFrame,LastBarTime,true);
         if(missing>0 && missing<=window)
           {
            required=MathMax(required,window+missing-1);
           }
        }
      int copied=CopyRates(_Symbol,TimeFrame,1,required,bars);
      if(copied<window && required>window)
        {
         copied=CopyRates(_Symbol,TimeFrame,1,window,bars);
        }
      if(copied<window)
        {
         return;
        }
      if(!Boundary(bars,copied,current,quote.bid,quote.ask))
        {
         Fault("bar boundary processing failed");
        }
     }
#ifdef ALPHAGEN_STUDY
//+------------------------------------------------------------------+
//| Show the next finalization operation before its timer event.      |
//+------------------------------------------------------------------+
   void StudyFinishProgress()
     {
      string stage="Closing the final virtual trajectory";
      if(study_finish_stage==2 || study_finish_stage==4)
        {
            stage=StringFormat("Final PPO | batch %d/%d | final updates %I64u | flush %d/%d",
                               study_finish_batch_completed,study_finish_batch_total,
                               updates-study_finish_updates,study_finish_contour,AG_LEARNING_CONTOURS);
        }
      if(study_finish_stage==3)
        {
            stage=StringFormat("Preparing final flush %d/%d",study_finish_contour+1,AG_LEARNING_CONTOURS);
        }
      if(study_finish_stage==5)
        {
            stage="Final accounting";
        }
      if(study_finish_stage==6)
        {
            stage="Saving 1/3: writing temporary model (indivisible operation)";
        }
      if(study_finish_stage==7)
        {
            stage="Saving 2/3: native reload and OpenCL verification (indivisible operation)";
        }
      if(study_finish_stage==8)
        {
            stage="Saving 3/3: replacing the verified checkpoint";
        }
      if(study_finish_stage==9)
        {
            stage="Finalization complete";
        }
      Comment("\n\nAlphaGen Study | finalization\n",stage,
              StringFormat("\nElapsed %.1f s | total updates %I64u",
                           (GetMicrosecondCount()-study_finish_started)/1000000.0,updates));
      ChartRedraw();
     }
//+------------------------------------------------------------------+
//| Start a finalizer: end=1, stop=2, opening ruin=3, bar ruin=4.       |
//+------------------------------------------------------------------+
   bool BeginStudyFinish(int reason,bool ruin)
     {
      study_finish_reason=reason;
      study_finish_ruin=ruin;
      study_finish_saved=false;
      study_finish_batch_total=0;
      study_finish_batch_completed=0;
      study_finish_contour=0;
      study_finish_updates=updates;
      study_finish_started=GetMicrosecondCount();
      study_finish_stage=1;
      PrintFormat("AlphaGen Study finalization BEGIN reason=%d",reason);
      StudyFinishProgress();
      return(true);
     }
//+------------------------------------------------------------------+
//| Advance one update, flush, or native checkpoint phase per event. |
//+------------------------------------------------------------------+
   bool StudyFinishStep()
     {
      if(stop_requested && study_finish_stage>1 && study_finish_stage<5)
        {
         if(!agent.CancelTraining())
           {
            return(false);
           }
         context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
         study_finish_stage=5;
         study_finish_reason=2;
         StudyFinishProgress();
         return(true);
        }
      int previous=study_finish_stage;
      ulong begin=GetMicrosecondCount();
      if(study_finish_stage==1)
        {
         if(!Finish(study_finish_ruin,false,false,true))
           {
            return(false);
           }
         study_finish_batch_total=agent.TrainingTotal();
         study_finish_batch_completed=agent.TrainingCompleted();
         study_finish_stage=2;
        }
      else if(study_finish_stage==2 || study_finish_stage==4)
        {
         if(agent.HasTraining())
           {
            study_finish_batch_total=agent.TrainingTotal();
            if(!Train())
              {
               return(false);
              }
            study_finish_batch_completed=agent.HasTraining()?agent.TrainingCompleted():study_finish_batch_total;
           }
         else
           {
            study_finish_stage=study_finish_contour<AG_LEARNING_CONTOURS?3:5;
           }
        }
      else if(study_finish_stage==3)
        {
         context[AG_CTX_TERMINAL]=1.0f;
         context[AG_CTX_MONEY_CONFIRMED]=1.0f;
         context[AG_CTX_NEXT_READY]=0.0f;
         datetime time=(datetime)(snapshot_msc/1000);
         if(!Forward(AG_FLUSH,AG_FLAG_END,time))
           {
               return(false);
           }
         study_finish_batch_total=agent.TrainingTotal();
         study_finish_batch_completed=agent.TrainingCompleted();
         study_finish_contour++;
         study_finish_stage=4;
        }
      else if(study_finish_stage==5)
        {
         context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
         if(!Finish(study_finish_ruin,false,false,false,true))
           {
               return(false);
           }
         Finalized=false;
         bool save=stop_requested || study_finish_reason==2 ||
                   (study_finish_reason==1 && epoch>=Epochs);
         study_finish_stage=save?6:9;
        }
      else if(study_finish_stage>=6 && study_finish_stage<=8)
        {
         int phase=study_finish_stage-6;
         PrintFormat("AlphaGen checkpoint phase %d/3 BEGIN",phase+1);
         if(!SavePhase(phase))
           {
            PrintFormat("AlphaGen checkpoint phase %d/3 FAILED error=%d",phase+1,GetLastError());
            return(false);
           }
         PrintFormat("AlphaGen checkpoint phase %d/3 END elapsed_ms=%I64u",phase+1,
                     (GetMicrosecondCount()-begin)/1000);
         if(phase==2)
           {
            study_finish_saved=true;
           }
         study_finish_stage++;
        }
      else if(study_finish_stage==9)
        {
         if(stop_requested && !study_finish_saved)
           {
            study_finish_reason=2;
            study_finish_stage=6;
            StudyFinishProgress();
            return(true);
           }
         study_finish_stage=0;
         if(stop_requested || study_finish_reason==2 || (study_finish_reason==1 && epoch>=Epochs))
           {
            Finalized=true;
            Print("AlphaGen Study finalization complete; verified checkpoint saved");
            ExpertRemove();
            return(true);
           }
         if(study_finish_reason==1)
           {
            epoch++;
            LastBarTime=0;
            observed_boundary=0;
            cursor=0;
            while(cursor<ArraySize(history) && history[cursor].time<Start)
              {
                  cursor++;
              }
           }
         if(study_finish_reason==3)
           {
            virtual_gap_terminal=false;
            cursor++;
           }
         ResetFinance();
         return(true);
        }
      if(previous!=study_finish_stage)
        {
            PrintFormat("AlphaGen Study finalization stage %d complete; last operation %I64u ms",previous,
                        (GetMicrosecondCount()-begin)/1000);
        }
      StudyFinishProgress();
      return(true);
     }
//+------------------------------------------------------------------+
//| Process a bounded portion of one virtual chronological step.     |
//+------------------------------------------------------------------+
   bool StudyStep()
     {
      if(Finalized || failed)
        {
         return(false);
        }
      if(study_finish_stage>0)
        {
         return(StudyFinishStep());
        }
      if(stop_requested)
        {
         // An opening gap may already have been reconciled before a bounded
         // search/PPO operation returned. Finish this bar under its existing
         // protection without another search step or policy action.
         bool partial_bar=cursor<ArraySize(history) && history[cursor].time<End &&
                          virtual_open_boundary==history[cursor].time && !cycle_complete;
         if(partial_bar)
           {
            MqlRates last=history[cursor];
            if(!money.Bar(last))
              {
               Fault("Study stop could not finish the protected virtual bar");
               ReturnFalse;
              }
            cursor++;
            MoneySample();
            snapshot_msc=(long)(last.time+PeriodSeconds(TimeFrame))*1000;
           }
         bool stop_ruin=money.equity<=AG_EQUITY_FLOOR*money.reference;
         return(BeginStudyFinish(2,stop_ruin));
        }
      if(cursor>=ArraySize(history) || history[cursor].time>=End)
        {
         return(BeginStudyFinish(1,false));
        }
      MqlRates bar=history[cursor];
      if(virtual_open_boundary!=bar.time)
        {
         snapshot_msc=(long)bar.time*1000;
         if(!money.PathPoint(bar.open,bar.spread*money.point,bar.time))
           {
            Fault("virtual opening-gap reconciliation failed");
            ReturnFalse;
           }
         virtual_open_boundary=bar.time;
         bool gap_ruin=money.equity<=AG_EQUITY_FLOOR*money.reference;
         if(gap_ruin)
           {
            if(open_decision)
              {
               interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
               last_equity=money.equity;
              }
            virtual_gap_terminal=true;
            return(BeginStudyFinish(3,true));
           }
        }
      if(!money.Snapshot(bar.open,bar.open+bar.spread*money.point,bar.time))
        {
         ReturnFalse;
        }
      if(open_decision)
        {
         interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
         last_equity=money.equity;
        }
      if(!Boundary(history,cursor,bar.time,bar.open,bar.open+bar.spread*money.point) )
        {
         ReturnFalse;
        }
      if(!cycle_complete)
        {
         string stage=StringFormat("Search | attempts %d/%d | last batch %d/%d processed | transitions %d/%d",
                                   agent.SearchTrials(),agent.SearchBudget(),agent.BatchProcessed(),agent.BatchGenerated(),
                                   agent.SearchTransitions(),agent.SearchRolloutTarget());
         if(pool_warmup_end>0)
           {
            int total=ArraySize(pool_warmup_rates);
            int first=total-warmup_bars+window;
            stage=StringFormat("Pool warmup | windows %d/%d",
                               pool_warmup_end-first,total-first+1);
           }
         else if(agent.HasTraining())
            stage=StringFormat("PPO | buffered search transitions %d",agent.SearchTransitions());
         Comment(StringFormat("AlphaGen Study | epoch %d/%d | %s\n%s\nActive pool %d | prepared pool %d\nSearch PPO updates %I64u | trade PPO updates %I64u",
                              epoch,Epochs,TimeToString(bar.time),stage,agent.ActiveCount(),agent.PreparedCount(),
                              agent.TokenUpdates(),agent.TradeUpdates()));
         return(true);
        }
      if(!money.Bar(bar))
        {
         ReturnFalse;
        }
      cursor++;
      MoneySample();
      bool ruin=money.equity<=AG_EQUITY_FLOOR*money.reference || money.Exhausted(bar.close,bar.close+bar.spread*money.point);
      if(ruin)
        {
         return(BeginStudyFinish(4,true));
        }
      string pools=StringFormat("Pool active %d prepared %d | search PPO %I64u trade PPO %I64u",
      agent.ActiveCount(),agent.PreparedCount(),agent.TokenUpdates(),agent.TradeUpdates());
      Comment(pools,"\n",StringFormat("AlphaGen Study | epoch %d/%d | %s\nbalance %.2f equity %.2f | virtual trades %I64u | updates %I64u\nUse the button to stop and save",epoch,Epochs,TimeToString(bar.time),money.balance,money.equity,money.trades,updates));
      return(true);
     }
#endif
  };
#endif
