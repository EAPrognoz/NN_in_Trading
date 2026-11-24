//+------------------------------------------------------------------+
//|                                  NeuroNet_AlphaQCMExecution.mqh  |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_EXECUTION_MQH
#define NEURONET_ALPHAQCM_EXECUTION_MQH

#include <Trade/Trade.mqh>
#include <Generic/HashMap.mqh>

// Market and financial execution adapted from AlphaGen/Trajectory.mqh.

//+------------------------------------------------------------------+
//| Encode exact integer metadata in FP32 service words.             |
//+------------------------------------------------------------------+
void AlphaQCMInteger(float &data[],int offset,ulong value)
  {
   for(int i=0;i<AG_TIME_LIMBS;i++)
     {
      data[offset+i]=(float)((value>>(AG_LIMB_BITS*i))&AG_LIMB_MASK);
     }
  }
//+------------------------------------------------------------------+
//| Financial part: each added volume retains its own entry and age. |
//+------------------------------------------------------------------+
struct SAlphaQCMPart
  {
   double            volume,entry;
   datetime          time;
  };
//+------------------------------------------------------------------+
//| SAlphaQCMTarget                                                  |
//+------------------------------------------------------------------+
struct SAlphaQCMTarget
  {
   double            volume,sl,tp;
   bool              invalid;
  };
//+------------------------------------------------------------------+
//| CAlphaQCMEnvironment                                             |
//+------------------------------------------------------------------+
class CAlphaQCMEnvironment
  {
public:
   bool              virtual_mode;
   double            balance;
   double            equity;
   double            previous_balance;
   double            previous_equity;
   double            reference;
   double            peak;
   double            gross_profit;
   double            gross_loss;
   double            max_drawdown;
   double            max_drawdown_fraction;
   double            volume;
   double            buy_profit;
   double            sell_profit;
   double            age_weight;
   double            margin;
   double            sl;
   double            tp;
   double            point;
   double            tick;
   double            lot_min;
   double            lot_step;
   double            lot_max;
   double            lot_limit;
   double            k_buy;
   double            k_sell;
   long              stops;
   long              freeze;
   long              trade_mode;
   ulong             trades;
   ulong             successful_requests;
   datetime          entered;
   SAlphaQCMPart     parts[];
   CTrade            trade;
   bool              pending;
   bool              stopout;
   bool              unknown_request;
   bool              reconciling;
   uint              request_id;
   ulong             request_deals[];
   ulong             request_orders[];
   double            intended_volume;
   double            intended_sl;
   double            intended_tp;

public:
   //+------------------------------------------------------------------+
   //| Initialize owned runtime state and diagnostic counters.          |
   //+------------------------------------------------------------------+
   CAlphaQCMEnvironment():virtual_mode(false),
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
   bool              Init(bool simulated,double capital);
   void              Reset(double capital);
   bool              Snapshot(double bid,double ask,datetime time);
   bool              Account(datetime time,float &a[]);
   //+------------------------------------------------------------------+
   //| Record the confirmed starting snapshot before execution.         |
   //+------------------------------------------------------------------+
   void              Commit()
     {
      previous_balance=balance;
      previous_equity=equity;
     }
   double            Lot(double v);
   bool              Map(const float &raw[],double bid,double ask,SAlphaQCMTarget &a);
   bool              CloseVirtual(double amount,double bid,double ask,datetime time);
   void              RecordRequest(MqlTradeResult &result);
   bool              ConfirmRequest(bool sent);
   bool              Execute(SAlphaQCMTarget &a,double bid,double ask,datetime time);
   bool              PathPoint(double bid,double spread,datetime time);
   bool              Segment(double from,double to,double spread,datetime time);
   bool              Bar(MqlRates &bar);
   bool              Exhausted(double bid,double ask);
   bool              TradeSessionOpen(const datetime time);
  };

//+------------------------------------------------------------------+
//| Init - validate the run and initialize the shared state.         |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Init(bool simulated,double capital)
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
   if(point<=0||tick<=0||lot_min<=0||lot_step<=0||lot_max<lot_min||capital<=0||
      AG_VOLUME_CAP<lot_min||!MathIsValidNumber(point+tick+lot_min+lot_step+lot_max+capital))
     {
      ReturnFalse;
     }
   k_buy=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE_PROFIT)/tick;
   k_sell=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE_LOSS)/tick;
   if(k_buy<=0||k_sell<=0||!MathIsValidNumber(k_buy+k_sell))
     {
      ReturnFalse;
     }
   if(!simulated&&(PositionsTotal()!=0||OrdersTotal()!=0))
     {
      ReturnFalse;
     }
   trade.SetAsyncMode(false);
   trade.SetTypeFillingBySymbol(_Symbol);
   Reset(capital);

   return(true);
  }

//+------------------------------------------------------------------+
//| Reset - Start an epoch without creating trading profit           |
//+------------------------------------------------------------------+
void CAlphaQCMEnvironment::Reset(double capital)
  {
   balance=equity=previous_balance=previous_equity=reference=peak=capital;
   volume=buy_profit=sell_profit=age_weight=margin=sl=tp=0;
   entered=0;
   stopout=false;
   ArrayResize(parts,0);
  }

//+------------------------------------------------------------------+
//| Snapshot - Reconcile exposure and mark account equity            |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Snapshot(double bid,double ask,datetime time)
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
         if(PositionGetTicket(i)==0||PositionGetString(POSITION_SYMBOL)!=_Symbol)
           {
            ReturnFalse;
           }
         double q=PositionGetDouble(POSITION_VOLUME);
         bool   buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
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
         if(sl<=0||tp<=0)
           {
            all_protected=false;
           }
        }
      if(!all_protected)
        {
         sl=tp=0;
        }
      // Synchronous requests are reconciled against orders and account state.
      pending=unknown_request||OrdersTotal()>0;
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
//| Account - Build Account13 from the decision snapshots            |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Account(datetime time,float &a[])
  {
   if(previous_balance<=0||previous_equity<=0)
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
//| Lot - round the target volume to the symbol lot grid.            |
//+------------------------------------------------------------------+
double CAlphaQCMEnvironment::Lot(double v)
  {
   if(v<lot_min)
     {
      return(0);
     }

   return(lot_min+MathFloor((MathMin(MathMin(v,lot_max),AG_VOLUME_CAP)-lot_min)/lot_step+AG_LOT_EPSILON)*lot_step);
  }

//+------------------------------------------------------------------+
//| Map - Validate Action6 and build the protected target            |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Map(const float &raw[],double bid,double ask,SAlphaQCMTarget &a)
  {
   a.volume=a.sl=a.tp=0;
   a.invalid=false;
   if(ArraySize(raw)!=AG_OUTPUT_SIZE||bid<=0||ask<bid)
     {
      a.invalid=true;

      return(true);
     }
   for(int i=0;i<AG_OUTPUT_SIZE;i++)
     {
      if(!MathIsValidNumber(raw[i])||raw[i]<0||raw[i]>1)
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
   int    side=raw[0]>raw[3]?1:-1,off=side>0?0:3;
   double price=side>0?ask:bid,mark=side>0?bid:ask;
   a.volume=side*q;
   a.tp=tick*MathFloor((price+side*raw[off+1]*AG_MAX_TP*point)/tick+AG_CONST_HALF);
   a.sl=tick*MathFloor((price-side*raw[off+2]*AG_MAX_SL*point)/tick+AG_CONST_HALF);
   double base=MathMax(stops,1)*point+(ask-bid);
   double dtp=side*(a.tp-mark),dsl=side*(mark-a.sl);
   bool direction=trade_mode==SYMBOL_TRADE_MODE_FULL||(side>0&&trade_mode==SYMBOL_TRADE_MODE_LONGONLY)
                  ||(side<0&&trade_mode==SYMBOL_TRADE_MODE_SHORTONLY);
   a.invalid=(!direction||raw[off+1]==0||raw[off+2]==0||a.sl<=0||a.tp<=0||dtp<=AG_PROTECTION_MULTIPLIER*base
              ||dsl<=base||dtp<=freeze*point||dsl<=freeze*point||(lot_limit>0&&q>lot_limit));

   return(true);
  }

//+------------------------------------------------------------------+
//| CloseVirtual - Realize each closed virtual part once             |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::CloseVirtual(double amount,double bid,double ask,datetime time)
  {
   for(int i=ArraySize(parts)-1;i>=0&&amount>AG_LOT_EPSILON;i--)
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
//| RecordRequest - Preserve unresolved outcomes before retry        |
//+------------------------------------------------------------------+
void CAlphaQCMEnvironment::RecordRequest(MqlTradeResult &result)
  {
   if(result.deal==0&&result.order==0)
     {
      return;
     }
   int n=ArraySize(request_deals);
   if(ArrayResize(request_deals,n+1)!=n+1||ArrayResize(request_orders,n+1)!=n+1)
     {
      return;
     }
   request_deals[n]=result.deal;
   request_orders[n]=result.order;
  }

//+------------------------------------------------------------------+
//| ConfirmRequest - Check outcomes before any request retry         |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::ConfirmRequest(bool sent)
  {
   MqlTradeResult result;
   trade.Result(result);
   RecordRequest(result);
   request_id=result.request_id;
   if(result.retcode==TRADE_RETCODE_TIMEOUT||result.retcode==TRADE_RETCODE_CONNECTION)
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
   bool confirmed=sent&&(result.retcode==TRADE_RETCODE_DONE||result.retcode==TRADE_RETCODE_DONE_PARTIAL);
   if(confirmed)
     {
      successful_requests++;
     }

   return(confirmed);
  }

//+------------------------------------------------------------------+
//| Execute - Reconcile the target before increasing exposure        |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Execute(SAlphaQCMTarget &a,double bid,double ask,datetime time)
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
   if(MathAbs(same_side_delta)>AG_LOT_EPSILON&&MathAbs(same_side_delta)<lot_min-AG_LOT_EPSILON)
     {
      a.invalid=true;
      a.volume=0;
      intended_volume=0;
     }
   if(virtual_mode)
     {
      if(volume*a.volume<0||a.volume==0)
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
   if(!TerminalInfoInteger(TERMINAL_CONNECTED)&&!MQLInfoInteger(MQL_TESTER))
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
      bool   buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
      double q=PositionGetDouble(POSITION_VOLUME);
      if(a.volume==0||(buy&&a.volume<0)||(!buy&&a.volume>0))
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
   if(volume*a.volume<0||(a.volume==0&&volume!=0))
     {
      reconciling=true;

      return(true);
     }
   if(a.volume==0)
     {
      if(volume==0&&!pending)
        {
         reconciling=false;
        }

      return(volume==0&&!pending);
     }
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong ticket=PositionGetTicket(i);
      bool  modified=trade.PositionModify(ticket,a.sl,a.tp);
      if(!ConfirmRequest(modified))
        {
         if(PositionGetDouble(POSITION_SL)<=0||PositionGetDouble(POSITION_TP)<=0)
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
#ifdef ALPHAGEN_TEST_DIAGNOSTICS
         PrintFormat("AlphaGen margin rejected volume=%.8f required=%.2f free=%.2f",
                     delta,m,AccountInfoDouble(ACCOUNT_MARGIN_FREE));
#endif
         a.invalid=true;
         SAlphaQCMTarget flat;
         ZeroMemory(flat);

         return(Execute(flat,bid,ask,time));
        }
      bool           ok=a.volume>0?trade.Buy(delta,_Symbol,0,a.sl,a.tp):trade.Sell(delta,_Symbol,0,a.sl,a.tp);
      MqlTradeResult result;
      trade.Result(result);
      RecordRequest(result);
      request_id=result.request_id;
      uint code=result.retcode;
      pending=(code==TRADE_RETCODE_PLACED);
      if(code==TRADE_RETCODE_TIMEOUT||code==TRADE_RETCODE_CONNECTION)
        {
         unknown_request=true;
         pending=true;

         return(true);
        }
      if(!ok||(code!=TRADE_RETCODE_DONE&&code!=TRADE_RETCODE_DONE_PARTIAL&&code!=TRADE_RETCODE_PLACED))
        {
         return(false);
        }
      if(code==TRADE_RETCODE_DONE||code==TRADE_RETCODE_DONE_PARTIAL)
        {
         successful_requests++;
        }
      reconciling=(code==TRADE_RETCODE_DONE_PARTIAL||code==TRADE_RETCODE_PLACED);
     }
   intended_volume=a.volume;
   intended_sl=a.sl;
   intended_tp=a.tp;

   return(Snapshot(bid,ask,time));
  }

//+------------------------------------------------------------------+
//| PathPoint - Apply opening gaps and protection at the quote       |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::PathPoint(double bid,double spread,datetime time)
  {
   if(!Snapshot(bid,bid+spread,time))
     {
      ReturnFalse;
     }
   bool hit=volume>0?(bid<=sl||bid>=tp):(volume<0&&(bid+spread>=sl||bid+spread<=tp));
   if(hit||equity<=AG_EQUITY_FLOOR*reference)
     {
      return(CloseVirtual(MathAbs(volume),bid,bid+spread,time));
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Segment - find the first protection or equity-floor crossing.    |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Segment(double from,double to,double spread,datetime time)
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
      if(fraction>=0&&fraction<=1&&fraction<nearest)
        {
         nearest=fraction;
         end=levels[i];
        }
     }
   if(!PathPoint(end,spread,time))
     {
      ReturnFalse;
     }
   if(nearest<=1&&volume!=0)
     {
      if(!CloseVirtual(MathAbs(volume),end,end+spread,time))
        {
         ReturnFalse;
        }
     }

   return(PathPoint(to,spread,time));
  }

//+------------------------------------------------------------------+
//| Bar - advance one causal historical OHLC execution path.         |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Bar(MqlRates &bar)
  {
   double spread=bar.spread*point;
   if(!PathPoint(bar.open,spread,bar.time))
     {
      ReturnFalse;
     }
   double first=volume>=0?bar.low:bar.high,second=volume>=0?bar.high:bar.low;
   if(!Segment(bar.open,first,spread,bar.time)||!Segment(first,second,spread,bar.time)||!Segment(second,bar.close,spread,bar.time))
     {
      ReturnFalse;
     }

   return(Snapshot(bar.close,bar.close+spread,bar.time+PeriodSeconds(TimeFrame)));
  }

//+------------------------------------------------------------------+
//| Exhausted - Check minimum-lot margin in permitted directions     |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::Exhausted(double bid,double ask)
  {
   if(volume!=0||pending)
     {
      return(false);
     }
   bool any=false;
   for(int side=0;side<2;side++)
     {
      if(trade_mode!=SYMBOL_TRADE_MODE_FULL&&trade_mode!=(side==0?SYMBOL_TRADE_MODE_LONGONLY:SYMBOL_TRADE_MODE_SHORTONLY))
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

//+------------------------------------------------------------------+
//| TradeSessionOpen - rule out a closed broker trading session.     |
//+------------------------------------------------------------------+
bool CAlphaQCMEnvironment::TradeSessionOpen(const datetime time)
  {
   MqlDateTime clock;
   if(!TimeToStruct(time,clock))
      return(false);

   long seconds=clock.hour*3600+clock.min*60+clock.sec;
   for(int days_ago=0;days_ago<2;days_ago++)
     {
      ENUM_DAY_OF_WEEK day=(ENUM_DAY_OF_WEEK)((clock.day_of_week+7-days_ago)%7);
      datetime from=0,to=0;
      for(uint session=0;SymbolInfoSessionTrade(_Symbol,day,session,from,to);session++)
        {
         long begin=(long)from%86400;
         long end=(long)to%86400;
         if(end<=begin)
           {
            end+=86400;
           }
         long second=seconds+days_ago*86400;
         if(second>=begin&&second<end)
            return(true);
        }
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Shared native model, chronological driver and finalizer.         |
//+------------------------------------------------------------------+
struct SAlphaQCMDeal
  {
   ulong             ticket;
   double            cash;
   double            volume;
   long              milliseconds;
   ulong             owner;
  };
//+------------------------------------------------------------------+
//| SAlphaQCMInterval                                                |
//+------------------------------------------------------------------+
struct SAlphaQCMInterval
  {
   ulong             decision;
   datetime          start_boundary,end_boundary;
   long              start_msc,end_msc;
   double            reference,gamma,start_equity,end_equity;
   int               duration;
   bool              trained;
  };
//+------------------------------------------------------------------+
//| SAlphaQCMRequestOwner                                            |
//+------------------------------------------------------------------+
struct SAlphaQCMRequestOwner
  {
   ulong             deal,order,decision;
  };

//+------------------------------------------------------------------+
//| CAlphaQCMApplicationBase                                         |
//+------------------------------------------------------------------+
class CAlphaQCMApplicationBase
  {
public:
   CNet                     model;
   SAlphaQCMDeal     ledger[];
   CHashMap<ulong,int>ledger_index;
   int                   confirmed_requests;
   SAlphaQCMInterval     sealed[];
   SAlphaQCMRequestOwner request_owners[];
   long              snapshot_msc;
   long              decision_start_msc;
   bool              fault_announced;
   bool              fault_settled;
   long              trained_money_msc;
   datetime          confirmed_boundary;
   datetime          finance_started;
   double            policy_cash;
   double            service_cash;
   double            excluded_cash;
   double            reported_equity;
   double            external_cash;
   CAlphaQCMEnvironment money;
   CBufferFloat      service;
   CBufferFloat      raw;
   CNeuronAlphaGenOCL *agent;
   datetime          LastBarTime;
   datetime          decision_time;
   ulong             decision_id;
   ulong             event_id;
   ulong             updates;
   ulong             excluded;
   ulong             request_decision;
   ulong             last_deal_ticket;
   ulong             finance_deal_ticket;
   long              last_deal_msc;
   long              finance_boundary_msc;
   long              finance_deal_msc;
   double            finance_balance;
   double            finance_equity;
   double            finance_gross_profit;
   double            finance_gross_loss;
   double            ledger_volume;
   bool              training;
   bool              simulated;
   bool              Finalized;
   bool              failed;
   bool              stop_requested;
   bool              open_decision;
   bool              invalid;
   bool              cycle_complete;
   bool              service_rebase;
   datetime          observed_boundary;
   datetime          sample_boundary;
   datetime          pool_warmup_boundary;
   datetime          virtual_open_boundary;
   bool              virtual_gap_terminal;
   bool              controlled_stopping;
   int               pool_warmup_end;
   int               study_finish_stage;
   int               study_finish_reason;
   int               study_finish_contour;
   bool              study_finish_ruin;
   bool              study_finish_saved;
   int               study_finish_batch_total;
   int               study_finish_batch_completed;
   ulong             study_finish_updates;
   ulong             study_finish_started;
   double            start_equity;
   double            last_equity;
   double            interval_reward;
   double            trade_gamma;
   int               delta;
   float             input_values[];
   float             context[];
   MqlRates          history[];
   MqlRates          pool_warmup_rates[];
   int               cursor;
   int               epoch;
   int               window;
   int               context_bars;
   int               warmup_bars;
   ulong             prep_us;
   ulong             forward_us;
   ulong             ppo_us;
   ulong             total_us;
   ulong             max_us;
   ulong             boundaries;
   ulong             prep_calls;
   ulong             forward_calls;
   ulong             ppo_calls;
   ulong             max_prep;
   ulong             max_forward;
   ulong             max_ppo;

public:
   //+------------------------------------------------------------------+
   //| Initialize owned runtime state and diagnostic counters.          |
   //+------------------------------------------------------------------+
   CAlphaQCMApplicationBase():confirmed_requests(0),
                     trained_money_msc(0),
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
                      request_decision(0),
                      last_deal_ticket(0),
                      finance_deal_ticket(0),
                      last_deal_msc(0),
                      finance_boundary_msc(0),
                      finance_deal_msc(0),
                      finance_balance(0),
                      finance_equity(0),
                      finance_gross_profit(0),
                      finance_gross_loss(0),
                       ledger_volume(0),
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
   bool              RememberRequests();
   bool              ExecuteTarget(SAlphaQCMTarget &target,double bid,double ask,datetime time,const ulong owner=0);
   void              FaultCleanup();
   void              Fault(string reason);
   bool              SealInterval(datetime boundary);
   bool              CorrectSealedCash(ulong owner,long at,double difference);
   virtual bool      Config(float &c[]);
   void              ResetFinance();
   double            Deposit();
   bool              Pack(MqlRates &bars[],int end,datetime boundary,bool account);
   void              RealMoneySample(bool terminal=false);
   void              MoneySample();
   virtual bool      ReconcileDeals(datetime now);
   void              Transaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result);
  };

//+------------------------------------------------------------------+
//| RememberRequests - bind new deal and order tickets.              |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::RememberRequests()
  {
   for(int i=0;i<ArraySize(money.request_deals);i++)
     {
      int n=ArraySize(request_owners);
      if(ArrayResize(request_owners,n+1)!=n+1)
        {
         ReturnFalse;
        }
      request_owners[n].deal=money.request_deals[i];
      request_owners[n].order=money.request_orders[i];
       request_owners[n].decision=request_decision;
     }
   ArrayResize(money.request_deals,0);
   ArrayResize(money.request_orders,0);

   return(true);
  }

//+------------------------------------------------------------------+
//| ExecuteTarget - Bind confirmed requests to their decision        |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::ExecuteTarget(SAlphaQCMTarget &target,double bid,double ask,datetime time,const ulong owner)
  {
   if(!money.pending)
     {
      request_decision=owner!=0?owner:(open_decision?decision_id:0);
     }
   bool result=money.Execute(target,bid,ask,time);
   if(!RememberRequests())
     {
      ReturnFalse;
     }

   return(result);
  }

//+------------------------------------------------------------------+
//| FaultCleanup - Reconcile protective closing after a model fault  |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::FaultCleanup()
  {
   if(!failed||fault_settled)
     {
      return;
     }
   if(simulated)
     {
      fault_settled=true;
      return;
     }
   if(!MQLInfoInteger(MQL_TESTER)&&!TerminalInfoInteger(TERMINAL_CONNECTED))
     {
      return;
     }
   MqlTick quote;
   if(!SymbolInfoTick(_Symbol,quote)||quote.bid<=0||quote.ask<quote.bid)
     {
      return;
     }
   money.Snapshot(quote.bid,quote.ask,quote.time);
   if(money.pending)
     {
      return;
     }
   SAlphaQCMTarget flat;
   ZeroMemory(flat);
   money.intended_volume=0;
   money.intended_sl=0;
   money.intended_tp=0;
   ExecuteTarget(flat,quote.bid,quote.ask,quote.time);
   if(money.Snapshot(quote.bid,quote.ask,quote.time)&&money.volume==0&&!money.pending)
     {
      fault_settled=true;
      Print("AlphaQCM fault exposure confirmed Flat; checkpoint preserved");
     }
  }

//+------------------------------------------------------------------+
//| Fault - block new model actions and protect open exposure.       |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::Fault(string reason)
  {
   failed=true;
   if(!fault_announced)
     {
      fault_announced=true;
      Print("AlphaQCM controlled fault: ",reason,"; new ACT and automatic Save disabled");
     }
   FaultCleanup();
  }

//+------------------------------------------------------------------+
//| SealInterval - Retain cash provenance until first PPO update     |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::SealInterval(datetime boundary)
  {
   int n=ArraySize(sealed);
   if(ArrayResize(sealed,n+1)!=n+1)
     {
      ReturnFalse;
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
//| CorrectSealedCash - Attribute late cash to its source transition |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::CorrectSealedCash(ulong owner,long at,double difference)
  {
   if(!training||difference==0)
     {
      return(true);
     }
   int match=-1;
   for(int i=0;i<ArraySize(sealed);i++)
     {
      // A timestamp shared by two decisions belongs to the interval ending
      // there. Explicit order ownership still takes precedence.
      bool candidate=owner!=0?sealed[i].decision==owner:
                     (at>sealed[i].start_msc&&at<=sealed[i].end_msc);
      if(!candidate)
        {
         continue;
        }
      if(match>=0)
        {
         PrintFormat("AlphaQCM cash correction rejected: ambiguous owner; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

         ReturnFalse;
        }
      match=i;
     }
   if(match<0)
     {
      // Current unsealed cash is already represented by its equity sample.
      if(owner==decision_id&&open_decision)
        {
         return(true);
        }
      if(open_decision&&at<decision_start_msc)
        {
         PrintFormat("AlphaQCM cash correction rejected: cash precedes open decision; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

         ReturnFalse;
        }
      if(ArraySize(sealed)==0||at>sealed[ArraySize(sealed)-1].end_msc)
        {
         return(true);
        }
      PrintFormat("AlphaQCM cash correction rejected: no matching interval; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

      ReturnFalse;
     }
   if(sealed[match].trained||sealed[match].reference<=0)
     {
      PrintFormat("AlphaQCM cash correction rejected: interval trained or invalid reference; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

      ReturnFalse;
     }
   int start_shift=iBarShift(_Symbol,TimeFrame,sealed[match].start_boundary,true);
   int cash_shift=iBarShift(_Symbol,TimeFrame,(datetime)((at-1)/1000),false);
   if(start_shift<0||cash_shift<0)
     {
      PrintFormat("AlphaQCM cash correction rejected: history shift unavailable; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

      ReturnFalse;
     }
   int cash_step=MathMax(start_shift-cash_shift+1,1);
   if(cash_step>sealed[match].duration)
     {
      PrintFormat("AlphaQCM cash correction rejected: cash outside interval duration; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

      ReturnFalse;
     }
   double adjustment=MathPow(sealed[match].gamma,cash_step-1)*difference/sealed[match].reference;
   if(!MathIsValidNumber(adjustment)||!agent.AdjustReward(sealed[match].decision,(float)adjustment))
     {
      PrintFormat("AlphaQCM cash correction rejected: reward adjustment rejected; owner=%I64u at=%I64d cash=%.8f match=%d",owner,at,difference,match);

      ReturnFalse;
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
//| Config - Transfer the shared profile to native configuration     |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::Config(float &c[])
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
//| ResetFinance - Record capital replacement outside the reward     |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::ResetFinance()
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
//| Deposit - Sample capital from the saved financial RNG            |
//+------------------------------------------------------------------+
double CAlphaQCMApplicationBase::Deposit()
  {
#ifdef ALPHAGEN_STUDY
   return(MinBalance+(MaxBalance-MinBalance)*agent.FinancialUniform());
#else
   return(AccountInfoDouble(ACCOUNT_BALANCE));
#endif
  }

//+------------------------------------------------------------------+
//| Pack - pack only closed OHLCV bars and the current Account13.    |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::Pack(MqlRates &bars[],int end,datetime boundary,bool account)
  {
   ulong begin=GetMicrosecondCount();
   if(end<window||end>ArraySize(bars))
     {
      ReturnFalse;
     }
   for(int i=0;i<window;i++)
     {
      MqlRates bar=bars[end-window+i];
      if(!MathIsValidNumber(bar.open+bar.high+bar.low+bar.close)||bar.open<=0||bar.high<bar.low||
         bar.close<=0||bar.spread<0||bar.tick_volume<0||
         (i>0&&bar.time<=bars[end-window+i-1].time))
        {
         ReturnFalse;
        }
      input_values[i]=(float)bar.open;
      input_values[window+i]=(float)bar.high;
      input_values[2*window+i]=(float)bar.low;
      input_values[3*window+i]=(float)bar.close;
      input_values[4*window+i]=(float)bar.tick_volume;
      AlphaQCMInteger(context,AG_CTX_SIZE+i*AG_CONTEXT_METADATA,(ulong)bar.time);
      context[AG_CTX_SIZE+i*AG_CONTEXT_METADATA+AG_TIME_LIMBS]=(float)bar.spread;
     }
   float    a[];
   datetime account_time=simulated?boundary:(snapshot_msc>0?(datetime)(snapshot_msc/1000):TimeCurrent());
   if(account&&!money.Account(account_time,a))
     {
      ReturnFalse;
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
//| RealMoneySample - Sample confirmed equity at the actual boundary |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::RealMoneySample(bool terminal)
  {
   if(simulated||!training||!open_decision)
     {
      return;
     }
   datetime at=snapshot_msc>0?(datetime)(snapshot_msc/1000):TimeCurrent();
   int      start=iBarShift(_Symbol,TimeFrame,decision_time,true);
   int      finish=iBarShift(_Symbol,TimeFrame,at,false);
   if(start<0||finish<0)
     {
      return;
     }
   // Intrabar risk marks are not reward observations. A terminal event belongs
   // to the right boundary of its bar, with exact boundary events ending there.
   int step=start-finish;
   if(terminal&&snapshot_msc>(long)iTime(_Symbol,TimeFrame,finish)*1000)
     {
      step++;
     }
   delta=MathMax(step,1);
   interval_reward+=MathPow(trade_gamma,delta-1)*(money.equity-last_equity)/money.reference;
   last_equity=money.equity;
  }

//+------------------------------------------------------------------+
//| MoneySample - Append virtual equity at the actual bar duration   |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::MoneySample()
  {
   if(!training||!open_decision)
     {
      return;
     }
   delta++;
   interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
   last_equity=money.equity;
  }

//+------------------------------------------------------------------+
//| ReconcileDeals - Book each native deal cash result once          |
//+------------------------------------------------------------------+
bool CAlphaQCMApplicationBase::ReconcileDeals(datetime now)
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
      int found=-1;
      bool recorded=ledger_index.TryGetValue(ticket,found);
      if(recorded&&ledger[found].cash==cash)
         continue;
      double volume=0;
      ENUM_DEAL_TYPE type=(ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket,DEAL_TYPE);
      if(type==DEAL_TYPE_BUY||type==DEAL_TYPE_SELL)
        {
         volume=HistoryDealGetDouble(ticket,DEAL_VOLUME)*(type==DEAL_TYPE_BUY?1:-1);
        }
      if(!MathIsValidNumber(cash)||!MathIsValidNumber(volume))
         ReturnFalse;
      long at=HistoryDealGetInteger(ticket,DEAL_TIME_MSC);
      if(recorded)
        {
         if(ledger[found].volume!=volume)
           {
            ledger_volume+=volume-ledger[found].volume;
            ledger[found].volume=volume;
           }
         if(!CorrectSealedCash(ledger[found].owner,at,cash-ledger[found].cash))
           {
            ReturnFalse;
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
            if(request_owners[j].deal==ticket||(order!=0&&request_owners[j].order==order))
              {
               owner=request_owners[j].decision;
               break;
              }
           }
         if(owner==0&&open_decision&&at>decision_start_msc)
           {
            owner=decision_id;
           }
         int n=ArraySize(ledger);
         if(ArrayResize(ledger,n+1,256)!=n+1)
            ReturnFalse;

         if(!ledger_index.Add(ticket,n))
           {
            ArrayResize(ledger,n);

            ReturnFalse;
           }
         if(!CorrectSealedCash(owner,at,cash))
           {
            ledger_index.Remove(ticket);
            ArrayResize(ledger,n);

            ReturnFalse;
           }
         ledger[n].ticket=ticket;
         ledger[n].cash=cash;
         ledger[n].volume=volume;
         ledger_volume+=volume;
         ledger[n].milliseconds=at;
         ledger[n].owner=owner;
         if(at>last_deal_msc||(at==last_deal_msc&&ticket>last_deal_ticket))
           {
            last_deal_msc=at;
            last_deal_ticket=ticket;
           }
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
   // The ledger is append-only. Once confirmed, a request stays confirmed.
   int requests=ArraySize(request_owners);
   for(;confirmed_requests<requests;confirmed_requests++)
     {
      ulong deal=request_owners[confirmed_requests].deal;
      if(deal==0)
         continue;
      int row=-1;
      if(!ledger_index.TryGetValue(deal,row))
        {
         money.pending=true;

         return(true);
        }
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Transaction - Trigger reconciliation from trade notifications    |
//+------------------------------------------------------------------+
void CAlphaQCMApplicationBase::Transaction(const MqlTradeTransaction &trans,const MqlTradeRequest &request,const MqlTradeResult &result)
  {
   if(trans.type==TRADE_TRANSACTION_REQUEST&&result.request_id==money.request_id)
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
   if(trans.type==TRADE_TRANSACTION_DEAL_ADD&&HistoryDealSelect(trans.deal))
     {
      if(HistoryDealGetInteger(trans.deal,DEAL_REASON)==DEAL_REASON_SO)
        {
         money.stopout=true;
        }
     }
   // Order and deal notification order is not a money confirmation.
  }

#endif
