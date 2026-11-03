//+------------------------------------------------------------------+
//|                                               Habi_Execution.mqh |
//|                                 Copyright 2026, MetaQuotes Ltd.  |
//|                                                    www.mql5.com  |
//+------------------------------------------------------------------+
#ifndef HABI_EXECUTION_MQH
#define HABI_EXECUTION_MQH
#include "Habi_History.mqh"

//+------------------------------------------------------------------+
//| SHabiPosition - one open position of a slot                      |
//+------------------------------------------------------------------+
struct SHabiPosition
  {
   ulong             ticket;
   ulong             identifier;
   long              created;
   double            lot;
   double            price;
   double            profit;
   double            swap;
  };
//+------------------------------------------------------------------+
//| SHabiSlot - per-slot command and request state                   |
//+------------------------------------------------------------------+
struct SHabiSlot
  {
   ulong             magic;                   // slot Magic number
   ulong             command;                 // command counter of the slot
   ulong             request_id;              // request identity of the slot
   ulong             request_order;           // order ticket of the request
   ulong             request_deal;            // deal ticket of the request
   uint              generation;              // generation of the command
   bool              pending;                 // request is not resolved yet
   bool              unknown;                 // request state is uncertain
   datetime          issued;                  // request issue time
   datetime          bar;                     // bar of the command
   int               reason;                  // mapping reason code
  };
//+------------------------------------------------------------------+
//| SHabiTransition - one observed transition record                 |
//+------------------------------------------------------------------+
struct SHabiTransition
  {
   ulong             id;                      // record identity
   datetime          start;                   // start boundary time
   datetime          finish;                  // end boundary time
   float             observation[];           // observation at the start
   float             distribution[];                     // snapshot of the policy distribution
   float             actions[];               // actions applied to the slots
   float             reward[];                // reward of every slot
   float             done[];                  // terminal flag of every slot
   float             next[];                  // observation of the next state
   double            start_float[];           // floating result at the start
   double            end_float[];             // floating result at the end
   double            start_cash[];            // cash per slot at the start
   double            end_cash[];              // cash per slot at the end
   double            penalty[];               // policy penalty of every slot
   bool              money_known[];           // cash snapshot is complete
   ulong             command[];               // command counter per slot
   ulong             start_cash_revision;     // cash revision at the start
   ulong             end_cash_revision;       // cash revision at the end
   bool              boundary;                // the record reached a boundary
   bool              used;                    // the record is already learned
   bool              gap;                     // the record is not recoverable
  };
//+------------------------------------------------------------------+
//| SHabiPositionOwner - position to slot map                        |
//+------------------------------------------------------------------+
struct SHabiPositionOwner
  {
   ulong             identifier;
   int               slot;
  };
//+------------------------------------------------------------------+
//| SHabiOrderOwner - order ticket to slot map                       |
//+------------------------------------------------------------------+
struct SHabiOrderOwner
  {
   ulong             ticket;
   int               slot;
  };
//+------------------------------------------------------------------+
//| SHabiCashDeal - cash deal bookkeeping record                     |
//+------------------------------------------------------------------+
struct SHabiCashDeal
  {
   ulong             ticket;                    // deal ticket
   ulong             position;                  // position identity of the deal
   ulong             revision;                  // cash revision of the record
   double            amount;                    // profit, swap and fees
   int               slot;                      // owning slot or minus one
   bool              deleted;                   // the deal was reported as deleted
  };
//+------------------------------------------------------------------+
//| SHabiUnownedCash - cash deal without an owner                    |
//+------------------------------------------------------------------+
struct SHabiUnownedCash
  {
   ulong             position;
   ulong             ticket;
  };
//+------------------------------------------------------------------+
//| CHabiExecution - order execution and ownership                   |
//+------------------------------------------------------------------+
class CHabiExecution
  {
public:
   SHabiSlot          slots[];                    // per-slot state
   SHabiTransition    records[];                  // transition records
   SHabiPositionOwner owners[];                   // position ownership
   SHabiOrderOwner    owned_orders[];             // order ownership
   SHabiCashDeal      cash_deals[];               // cash deal book
   SHabiUnownedCash   cash_unowned[];             // deals without owner
   ulong              cash_retry[];               // deferred deal tickets
   double             cash_totals[];              // cash total per slot
   datetime           cash_poll_from;             // start of the cash poll
   ulong              cash_revision;              // cash book revision
   ulong              boundary_cash_revision;     // revision of the last boundary
   bool               cash_ready;                 // cash book is complete
   datetime           run_started;                // start of this run
   SHabiSymbol        symbol;                     // symbol constraints
   ulong              run_id;                     // identity of this run
   ulong              next_command;               // command counter
   ulong              next_record;                // record counter
   ulong              next_request;               // request counter
   uint               generation;                 // current generation
   double             previous_balance;           // balance of the last boundary
   double             previous_equity;            // equity of the last boundary
   bool               test;                       // single slot test mode
   datetime           current_bar;                // bar of the last boundary
   double             boundary_cash[];            // cash snapshot per slot
   double             boundary_float[];           // floating snapshot per slot
   bool               boundary_known[];           // snapshot validity per slot
   double             observed_balance;           // balance of the observation
   double             observed_equity;            // equity of the observation
//+------------------------------------------------------------------+
//| CHabiExecution - constructor resets execution state              |
//+------------------------------------------------------------------+
                     CHabiExecution()
     {
      run_id=0;
      next_command=0;
      next_record=0;
      next_request=0;
      generation=0;
      previous_balance=0;
      previous_equity=0;
      test=false;
      current_bar=0;
      observed_balance=0;
      observed_equity=0;
      cash_poll_from=0;
      cash_revision=0;
      boundary_cash_revision=0;
      cash_ready=true;
     }
//+------------------------------------------------------------------+
//| ~CHabiExecution - destructor keeps the terminal state            |
//+------------------------------------------------------------------+
                    ~CHabiExecution()
     {
     }
//+------------------------------------------------------------------+
//| Init - prepares slots and blocks a foreign restart               |
//+------------------------------------------------------------------+
   bool              Init(int count,bool testing)
     {
      test=testing;

      if(count<1 || !symbol.Read() || (!test && AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING))
         return(false);

      if((test && MagicTest==0) || (!test && (MagicBase==0 || MagicBase>ULONG_MAX-(ulong)count)))
         return(false);

      if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
        {
         Print("Habi execution requires a hedging account for ticket isolation");

         return(false);
        }

      ArrayResize(slots,count);

      run_id=(GetMicrosecondCount()^(ulong)TimeLocal())&0xFFFFFFFFFFFF;
      run_started=TimeCurrent();

      for(int n=0;n<count;n++)
        {
         ZeroMemory(slots[n]);

         slots[n].magic=test?MagicTest:MagicBase+(ulong)n+1;
        }

      //--- an existing position of this run cannot be adopted automatically
      for(int i=0;i<PositionsTotal();i++)
        {
         if(PositionGetTicket(i)==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol)
            continue;

         for(int n=0;n<count;n++)
            if((ulong)PositionGetInteger(POSITION_MAGIC)==slots[n].magic)
              {
               PrintFormat("Habi restart blocked: symbol=%s Magic=%I64u position=%I64u; resolve existing exposure manually before restart; automatic ownership recovery is unsupported",_Symbol,slots[n].magic,(ulong)PositionGetInteger(POSITION_TICKET));

               return(false);
              }
        }

      //--- the same check is repeated for pending orders of this run
      for(int i=0;i<OrdersTotal();i++)
        {
         if(OrderGetTicket(i)==0 || OrderGetString(ORDER_SYMBOL)!=_Symbol)
            continue;

         for(int n=0;n<count;n++)
            if((ulong)OrderGetInteger(ORDER_MAGIC)==slots[n].magic)
              {
               PrintFormat("Habi restart blocked: symbol=%s Magic=%I64u order=%I64u; resolve the pending order manually before restart",_Symbol,slots[n].magic,(ulong)OrderGetInteger(ORDER_TICKET));

               return(false);
              }
        }

      ArrayResize(cash_totals,count);
      ArrayInitialize(cash_totals,0);

      cash_poll_from=run_started;
      previous_balance=AccountInfoDouble(ACCOUNT_BALANCE);
      previous_equity=AccountInfoDouble(ACCOUNT_EQUITY);

      return(previous_balance>0 && previous_equity>0);
     }
//+------------------------------------------------------------------+
//| Positions - collects slot positions sorted by age                |
//+------------------------------------------------------------------+
   bool              Positions(int slot,SHabiPosition &out[])
     {
      ArrayResize(out,0);

      for(int i=0;i<PositionsTotal();i++)
        {
         ulong t=PositionGetTicket(i);

         if(t==0)
            return(false);

         if(PositionGetString(POSITION_SYMBOL)!=_Symbol || (ulong)PositionGetInteger(POSITION_MAGIC)!=slots[slot].magic)
            continue;

         SHabiPosition p;

         p.ticket=t;
         p.identifier=(ulong)PositionGetInteger(POSITION_IDENTIFIER);
         p.created=PositionGetInteger(POSITION_TIME_MSC);
         p.lot=PositionGetDouble(POSITION_VOLUME)*(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1);

         RememberOwner(p.identifier,slot);

         p.price=PositionGetDouble(POSITION_PRICE_OPEN);
         p.profit=PositionGetDouble(POSITION_PROFIT);
         p.swap=PositionGetDouble(POSITION_SWAP);

         int n=ArraySize(out);

         ArrayResize(out,n+1);

         out[n]=p;
        }

      //--- insertion sort keeps the oldest position first
      for(int i=1;i<ArraySize(out);i++)
         for(int j=i;j>0;j--)
           {
            if(out[j-1].created<out[j].created || (out[j-1].created==out[j].created && out[j-1].ticket<out[j].ticket))
               break;

            SHabiPosition temp=out[j-1];

            out[j-1]=out[j];
            out[j]=temp;
           }

      return(true);
     }
//+------------------------------------------------------------------+
//| Floating - sums the floating result of a slot                    |
//+------------------------------------------------------------------+
   bool              Floating(int slot,double &value)
     {
      SHabiPosition p[];

      value=0;

      if(!Positions(slot,p))
         return(false);

      for(int i=0;i<ArraySize(p);i++)
         value+=p[i].profit+p[i].swap;

      return(MathIsValidNumber(value));
     }
//+------------------------------------------------------------------+
//| RememberOwner - binds a position to a slot                       |
//+------------------------------------------------------------------+
   void              RememberOwner(ulong identifier,int slot)
     {
      if(identifier==0)
         return;

      for(int i=0;i<ArraySize(owners);i++)
         if(owners[i].identifier==identifier)
            return;

      int k=ArraySize(owners);

      ArrayResize(owners,k+1);

      owners[k].identifier=identifier;
      owners[k].slot=slot;

      ReattributeCash(identifier,slot);
     }
//+------------------------------------------------------------------+
//| RememberOrder - binds an order ticket to a slot                  |
//+------------------------------------------------------------------+
   void              RememberOrder(ulong ticket,int slot)
     {
      if(ticket==0)
         return;

      for(int i=0;i<ArraySize(owned_orders);i++)
         if(owned_orders[i].ticket==ticket)
            return;

      int k=ArraySize(owned_orders);

      ArrayResize(owned_orders,k+1);

      owned_orders[k].ticket=ticket;
      owned_orders[k].slot=slot;
     }
//+------------------------------------------------------------------+
//| OwnOrder - checks that an order belongs to a slot                |
//+------------------------------------------------------------------+
   bool              OwnOrder(ulong ticket,int slot)
     {
      if(ticket==0)
         return(false);

      for(int i=0;i<ArraySize(owned_orders);i++)
         if(owned_orders[i].ticket==ticket && owned_orders[i].slot==slot)
            return(true);

      return(false);
     }
//+------------------------------------------------------------------+
//| RunEvidence - proves that a deal belongs to the run              |
//+------------------------------------------------------------------+
   bool              RunEvidence(ulong deal,int slot)
     {
      if(OwnOrder((ulong)HistoryDealGetInteger(deal,DEAL_ORDER),slot))
         return(true);

      if(HistoryDealGetInteger(deal,DEAL_TIME)<run_started)
         return(false);

      string prefix=StringFormat("H%I64X:",run_id);

      return(StringFind(HistoryDealGetString(deal,DEAL_COMMENT),prefix)==0);
     }
//+------------------------------------------------------------------+
//| Owner - returns the slot that owns a position                    |
//+------------------------------------------------------------------+
   int               Owner(ulong identifier)
     {
      if(identifier==0)
         return(-1);

      for(int i=0;i<ArraySize(owners);i++)
         if(owners[i].identifier==identifier)
            return(owners[i].slot);

      return(-1);
     }
//+------------------------------------------------------------------+
//| SlotForMagic - maps a Magic number to a slot                     |
//+------------------------------------------------------------------+
   int               SlotForMagic(ulong magic)
     {
      if(test)
         return(magic==MagicTest?0:-1);

      if(magic<=MagicBase || magic-MagicBase>(ulong)ArraySize(slots))
         return(-1);

      return((int)(magic-MagicBase-1));
     }
//+------------------------------------------------------------------+
//| CashIndex - binary search of a deal ticket                       |
//+------------------------------------------------------------------+
   int               CashIndex(ulong ticket,bool &found)
     {
      int lo=0;
      int hi=ArraySize(cash_deals);

      //--- the cash book is kept sorted by ticket
      while(lo<hi)
        {
         int mid=lo+(hi-lo)/2;

         if(cash_deals[mid].ticket<ticket)
            lo=mid+1;
         else
            hi=mid;
        }

      found=lo<ArraySize(cash_deals) && cash_deals[lo].ticket==ticket;

      return(lo);
     }
//+------------------------------------------------------------------+
//| QueueUnownedCash - queues a deal without a position owner        |
//+------------------------------------------------------------------+
   void              QueueUnownedCash(ulong position,ulong ticket)
     {
      int lo=0;
      int hi=ArraySize(cash_unowned);

      while(lo<hi)
        {
         int mid=lo+(hi-lo)/2;

         if(cash_unowned[mid].position<position || (cash_unowned[mid].position==position && cash_unowned[mid].ticket<ticket))
            lo=mid+1;
         else
            hi=mid;
        }

      if(lo<ArraySize(cash_unowned) && cash_unowned[lo].position==position && cash_unowned[lo].ticket==ticket)
         return;

      int count=ArraySize(cash_unowned);

      ArrayResize(cash_unowned,count+1);

      for(int i=count;i>lo;i--)
         cash_unowned[i]=cash_unowned[i-1];

      cash_unowned[lo].position=position;
      cash_unowned[lo].ticket=ticket;
     }
//+------------------------------------------------------------------+
//| ReattributeCash - binds queued deals to a found slot             |
//+------------------------------------------------------------------+
   void              ReattributeCash(ulong position,int slot)
     {
      int lo=0;
      int hi=ArraySize(cash_unowned);

      while(lo<hi)
        {
         int mid=lo+(hi-lo)/2;

         if(cash_unowned[mid].position<position)
            lo=mid+1;
         else
            hi=mid;
        }

      int finish=lo;

      while(finish<ArraySize(cash_unowned) && cash_unowned[finish].position==position)
        {
         bool found=false;

         int at=CashIndex(cash_unowned[finish].ticket,found);

         if(found && cash_deals[at].slot<0)
           {
            double amount=cash_deals[at].amount;

            cash_deals[at].slot=slot;
            cash_totals[slot]+=amount;

            CorrectCashSnapshots(cash_deals[at].revision,slot,amount);
           }

         finish++;
        }

      int remove=finish-lo;

      if(remove>0)
        {
         for(int i=lo;i<ArraySize(cash_unowned)-remove;i++)
            cash_unowned[i]=cash_unowned[i+remove];

         ArrayResize(cash_unowned,ArraySize(cash_unowned)-remove);
        }
     }
//+------------------------------------------------------------------+
//| CorrectCashSnapshots - adjusts earlier cash snapshots            |
//+------------------------------------------------------------------+
   void              CorrectCashSnapshots(ulong revision,int slot,double delta)
     {
      if(slot<0 || delta==0)
         return;

      //--- a correction belongs to the snapshots that already held the ticket
      for(int k=0;k<ArraySize(records);k++)
        {
         if(records[k].start_cash_revision>=revision && ArraySize(records[k].start_cash)>slot)
            records[k].start_cash[slot]+=delta;

         if(records[k].boundary && records[k].end_cash_revision>=revision)
            records[k].end_cash[slot]+=delta;
        }

      if(boundary_cash_revision>=revision && ArraySize(boundary_cash)>slot)
         boundary_cash[slot]+=delta;
     }
//+------------------------------------------------------------------+
//| DeleteCashDeal - removes a deal from the cash book               |
//+------------------------------------------------------------------+
   bool              DeleteCashDeal(ulong ticket)
     {
      bool found=false;

      int at=CashIndex(ticket,found);

      if(!found)
         return(true);

      int slot=cash_deals[at].slot;
      double amount=cash_deals[at].amount;

      if(slot>=0)
        {
         cash_totals[slot]-=amount;

         CorrectCashSnapshots(cash_deals[at].revision,slot,-amount);
        }

      cash_deals[at].amount=0;
      cash_deals[at].slot=-1;
      cash_deals[at].deleted=true;

      return(true);
     }
//+------------------------------------------------------------------+
//| RegisterDealOwner - binds a deal position to a slot              |
//+------------------------------------------------------------------+
   bool              RegisterDealOwner(ulong ticket)
     {
      if(!HistoryDealSelect(ticket))
         return(false);

      if(HistoryDealGetString(ticket,DEAL_SYMBOL)!=_Symbol)
         return(true);

      int slot=SlotForMagic((ulong)HistoryDealGetInteger(ticket,DEAL_MAGIC));

      if(slot>=0 && RunEvidence(ticket,slot))
         RememberOwner((ulong)HistoryDealGetInteger(ticket,DEAL_POSITION_ID),slot);

      return(true);
     }
//+------------------------------------------------------------------+
//| UpdateCashDeal - inserts or updates one cash deal                |
//+------------------------------------------------------------------+
   bool              UpdateCashDeal(ulong ticket)
     {
      if(!RegisterDealOwner(ticket))
        {
         bool found=false;

         int at=CashIndex(ticket,found);

         return(found && cash_deals[at].deleted);
        }

      if(HistoryDealGetString(ticket,DEAL_SYMBOL)!=_Symbol)
         return(true);

      ulong position=(ulong)HistoryDealGetInteger(ticket,DEAL_POSITION_ID);
      int slot=Owner(position);

      if(slot<0 && position==0)
        {
         int candidate=SlotForMagic((ulong)HistoryDealGetInteger(ticket,DEAL_MAGIC));

         if(candidate>=0 && RunEvidence(ticket,candidate))
            slot=candidate;
        }

      double amount=HistoryDealGetDouble(ticket,DEAL_PROFIT)+HistoryDealGetDouble(ticket,DEAL_SWAP)+ HistoryDealGetDouble(ticket,DEAL_COMMISSION)+HistoryDealGetDouble(ticket,DEAL_FEE);

      if(!MathIsValidNumber(amount))
         return(false);

      bool found=false;

      int at=CashIndex(ticket,found);

      if(!found)
        {
         int count=ArraySize(cash_deals);

         ArrayResize(cash_deals,count+1);

         for(int i=count;i>at;i--)
            cash_deals[i]=cash_deals[i-1];

         cash_deals[at].ticket=ticket;
         cash_deals[at].revision=++cash_revision;
         cash_deals[at].slot=-1;
         cash_deals[at].amount=0;
         cash_deals[at].deleted=false;
        }

      int old_slot=cash_deals[at].slot;
      double old_amount=cash_deals[at].amount;

      cash_deals[at].position=position;
      cash_deals[at].deleted=false;

      if(slot<0 && position>0)
         QueueUnownedCash(position,ticket);

      bool unresolved_ours=slot<0 && SlotForMagic((ulong)HistoryDealGetInteger(ticket,DEAL_MAGIC))>=0 && HistoryDealGetInteger(ticket,DEAL_TIME)>=run_started;

      if(old_slot==slot && old_amount==amount)
         return(!unresolved_ours);

      //--- the same deal may arrive twice through a callback and a poll
      if(old_slot>=0)
        {
         cash_totals[old_slot]-=old_amount;

         CorrectCashSnapshots(cash_deals[at].revision,old_slot,-old_amount);
        }

      if(slot>=0)
        {
         cash_totals[slot]+=amount;

         CorrectCashSnapshots(cash_deals[at].revision,slot,amount);
        }

      cash_deals[at].position=position;
      cash_deals[at].slot=slot;
      cash_deals[at].amount=amount;

      return(!unresolved_ours && (slot<0 || MathIsValidNumber(cash_totals[slot])));
     }
//+------------------------------------------------------------------+
//| QueueCashRetry - defers a deal that is not resolved              |
//+------------------------------------------------------------------+
   void              QueueCashRetry(ulong ticket)
     {
      for(int i=0;i<ArraySize(cash_retry);i++)
         if(cash_retry[i]==ticket)
            return;

      int count=ArraySize(cash_retry);

      ArrayResize(cash_retry,count+1);

      cash_retry[count]=ticket;
      cash_ready=false;
     }
//+------------------------------------------------------------------+
//| DealEvent - handles a deal transaction event                     |
//+------------------------------------------------------------------+
   void              DealEvent(ENUM_TRADE_TRANSACTION_TYPE kind,ulong ticket)
     {
      if(ticket==0)
         return;

      if(kind==TRADE_TRANSACTION_DEAL_DELETE)
        {
         DeleteCashDeal(ticket);

         for(int i=ArraySize(cash_retry)-1;i>=0;i--)
            if(cash_retry[i]==ticket)
              {
               for(int j=i;j<ArraySize(cash_retry)-1;j++)
                  cash_retry[j]=cash_retry[j+1];

               ArrayResize(cash_retry,ArraySize(cash_retry)-1);
              }

         return;
        }

      if(kind!=TRADE_TRANSACTION_DEAL_ADD && kind!=TRADE_TRANSACTION_DEAL_UPDATE)
         return;

      if(!UpdateCashDeal(ticket))
         QueueCashRetry(ticket);
     }
//+------------------------------------------------------------------+
//| RefreshCash - polls the incremental deal history                 |
//+------------------------------------------------------------------+
   bool              RefreshCash()
     {
      datetime now=TimeCurrent();

      if(now<cash_poll_from || !HistorySelect(cash_poll_from,now))
        {
         cash_ready=false;

         return(false);
        }

      //--- one incremental range for all slots, exact-ticket upserts avoid duplicates
      ulong tickets[];
      int count=HistoryDealsTotal();

      ArrayResize(tickets,count);

      for(int i=0;i<count;i++)
        {
         tickets[i]=HistoryDealGetTicket(i);

         if(tickets[i]==0)
           {
            cash_ready=false;

            return(false);
           }
        }

      //--- entry ownership is registered before manual and SL closing deals
      for(int i=0;i<count;i++)
         if(!RegisterDealOwner(tickets[i]))
           {
            QueueCashRetry(tickets[i]);
           }

      for(int i=0;i<count;i++)
         if(!UpdateCashDeal(tickets[i]))
           {
            QueueCashRetry(tickets[i]);
           }

      for(int i=ArraySize(cash_retry)-1;i>=0;i--)
        {
         if(!UpdateCashDeal(cash_retry[i]))
            continue;

         for(int j=i;j<ArraySize(cash_retry)-1;j++)
            cash_retry[j]=cash_retry[j+1];

         ArrayResize(cash_retry,ArraySize(cash_retry)-1);
        }

      cash_poll_from=now;
      cash_ready=ArraySize(cash_retry)==0;

      return(cash_ready);
     }
//+------------------------------------------------------------------+
//| Observation - builds the observation of the current state        |
//+------------------------------------------------------------------+
   bool              Observation(CHabiHistory &history,datetime time,float &out[])
     {
      SHabiSnapshot s;

      ZeroMemory(s);

      s.balance=AccountInfoDouble(ACCOUNT_BALANCE);
      s.equity=AccountInfoDouble(ACCOUNT_EQUITY);
      s.previous_balance=previous_balance;
      s.previous_equity=previous_equity;

      for(int i=0;i<PositionsTotal();i++)
        {
         if(PositionGetTicket(i)==0)
            return(false);

         if(PositionGetString(POSITION_SYMBOL)!=_Symbol)
            continue;

         double lot=PositionGetDouble(POSITION_VOLUME);
         double profit=PositionGetDouble(POSITION_PROFIT);
         bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;

         s.lot+=buy?lot:-lot;

         if(buy)
            s.profit_buy+=profit;
         else
            s.profit_sell+=profit;

         s.age_weight+=(double)(time-(datetime)PositionGetInteger(POSITION_TIME))*MathAbs(profit);
        }

      float market[];
      float account[];

      if(!history.Market(time,market) || !HabiAccount(s,time,account))
         return(false);

      ArrayCopy(out,market);
      ArrayCopy(out,account,ArraySize(market));

      observed_balance=s.balance;
      observed_equity=s.equity;

      return(true);
     }
//+------------------------------------------------------------------+
//| CommitAccountBoundary - stores the last account snapshot         |
//+------------------------------------------------------------------+
   void              CommitAccountBoundary()
     {
      previous_balance=observed_balance;
      previous_equity=observed_equity;
     }
//+------------------------------------------------------------------+
//| FinalOrder - checks that an order reached a final state          |
//+------------------------------------------------------------------+
   bool              FinalOrder(int n,ulong order)
     {
      if(order==0 || OrderSelect(order))
         return(false);

      if(!HistoryOrderSelect(order))
         return(false);

      ENUM_ORDER_STATE state=(ENUM_ORDER_STATE)HistoryOrderGetInteger(order,ORDER_STATE);

      if(state!=ORDER_STATE_FILLED && state!=ORDER_STATE_CANCELED && state!=ORDER_STATE_REJECTED && state!=ORDER_STATE_EXPIRED)
        {
         PrintFormat("Habi order unresolved slot=%d order=%I64u state=%d",n,order,(int)state);

         return(false);
        }

      slots[n].pending=false;
      slots[n].unknown=false;

      return(true);
     }
//+------------------------------------------------------------------+
//| ReconcileSlot - resolves the pending state of a slot             |
//+------------------------------------------------------------------+
   bool              ReconcileSlot(int n)
     {
      if(!slots[n].pending && !slots[n].unknown)
         return(true);

      ulong order=slots[n].request_order;
      ulong deal=slots[n].request_deal;

      if(order==0 && deal>0 && HistoryDealSelect(deal))
        {
         order=(ulong)HistoryDealGetInteger(deal,DEAL_ORDER);

         slots[n].request_order=order;

         RememberOrder(order,n);
        }

      //--- a partial deal never proves that the remainder of an order is final
      if(order>0)
         return(FinalOrder(n,order));

      string comment=Comment(n);

      if(!HistorySelect(slots[n].issued,TimeCurrent()))
         return(false);

      ulong found=0;

      //--- one identity lookup and one state check, never recursive
      for(int i=0;i<HistoryOrdersTotal();i++)
        {
         ulong t=HistoryOrderGetTicket(i);

         if(HistoryOrderGetString(t,ORDER_SYMBOL)==_Symbol && (ulong)HistoryOrderGetInteger(t,ORDER_MAGIC)==slots[n].magic && HistoryOrderGetString(t,ORDER_COMMENT)==comment)
           {
            found=t;

            break;
           }
        }

      if(found==0)
         return(false);

      slots[n].request_order=found;

      RememberOrder(found,n);

      return(FinalOrder(n,found));
     }
//+------------------------------------------------------------------+
//| Comment - builds the request comment of a command                |
//+------------------------------------------------------------------+
   string            Comment(int n)
     {
      return(StringFormat("H%I64X:%I64X",run_id,slots[n].request_id));
     }
//+------------------------------------------------------------------+
//| Filling - selects the supported filling mode                     |
//+------------------------------------------------------------------+
   ENUM_ORDER_TYPE_FILLING Filling()
     {
      long flags=SymbolInfoInteger(_Symbol,SYMBOL_FILLING_MODE);

      if((flags&SYMBOL_FILLING_FOK)!=0)
         return(ORDER_FILLING_FOK);

      if((flags&SYMBOL_FILLING_IOC)!=0)
         return(ORDER_FILLING_IOC);

      return(ORDER_FILLING_RETURN);
     }
//+------------------------------------------------------------------+
//| EconomicEvidence - turns a rejection code into a reason          |
//+------------------------------------------------------------------+
   void              EconomicEvidence(int n,const MqlTradeRequest &req,uint code)
     {
      if(req.action!=TRADE_ACTION_DEAL || req.position!=0 || req.volume<=0)
         return;

      bool buy=req.type==ORDER_TYPE_BUY;
      bool proven=false;

      if(code==TRADE_RETCODE_NO_MONEY)
        {
         double margin=0;

         proven=OrderCalcMargin(req.type,_Symbol,req.volume,req.price,margin) && margin>AccountInfoDouble(ACCOUNT_MARGIN_FREE);
        }

      if(code==TRADE_RETCODE_LIMIT_VOLUME && symbol.lot_limit>0)
        {
         double total=req.volume;

         for(int i=0;i<PositionsTotal();i++)
           {
            if(!PositionGetTicket(i) || PositionGetString(POSITION_SYMBOL)!=_Symbol)
               continue;

            if((PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)==buy)
               total+=PositionGetDouble(POSITION_VOLUME);
           }

         for(int i=0;i<OrdersTotal();i++)
           {
            if(!OrderGetTicket(i) || OrderGetString(ORDER_SYMBOL)!=_Symbol)
               continue;

            ENUM_ORDER_TYPE t=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
            bool same=buy?(t==ORDER_TYPE_BUY || t==ORDER_TYPE_BUY_LIMIT || t==ORDER_TYPE_BUY_STOP || t==ORDER_TYPE_BUY_STOP_LIMIT): (t==ORDER_TYPE_SELL || t==ORDER_TYPE_SELL_LIMIT || t==ORDER_TYPE_SELL_STOP || t==ORDER_TYPE_SELL_STOP_LIMIT);

            if(same)
               total+=OrderGetDouble(ORDER_VOLUME_CURRENT);
           }

         proven=total>symbol.lot_limit+1e-9;
        }

      if(code==TRADE_RETCODE_INVALID_STOPS && req.sl>0 && req.tp>0)
        {
         MqlTick now;

         if(SymbolInfoTick(_Symbol,now) && MathAbs(req.price-(buy?now.ask:now.bid))<symbol.tick*.5)
            proven=!HabiProtection(buy?1:-1,req.sl,req.tp,now.bid,now.ask,symbol);
        }

      if(code==TRADE_RETCODE_LONG_ONLY || code==TRADE_RETCODE_SHORT_ONLY || code==TRADE_RETCODE_CLOSE_ONLY)
        {
         long mode=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE);

         proven=mode==SYMBOL_TRADE_MODE_CLOSEONLY || (buy && mode==SYMBOL_TRADE_MODE_SHORTONLY) || (!buy && mode==SYMBOL_TRADE_MODE_LONGONLY);
        }

      if(proven)
         slots[n].reason=HABI_ECONOMIC_INVALID;
     }
//+------------------------------------------------------------------+
//| CheckRequest - validates a request before sending                |
//+------------------------------------------------------------------+
   bool              CheckRequest(int n,MqlTradeRequest &request,MqlTradeCheckResult &check)
     {
      ZeroMemory(check);

      bool checked=OrderCheck(request,check);
      bool supported=request.action==TRADE_ACTION_DEAL || request.action==TRADE_ACTION_SLTP || request.action==TRADE_ACTION_REMOVE;
      bool finite=MathIsValidNumber(check.balance+check.equity+check.profit+check.margin+check.margin_free+check.margin_level);

      //--- native OrderCheck reports retcode zero for these operations
      if(checked && supported && finite && check.retcode==0)
         return(true);

      EconomicEvidence(n,request,check.retcode);

      if(!supported || !finite)
         slots[n].reason=HABI_ADAPTER_ERROR;

      PrintFormat("Habi OrderCheck command=%I64u bool=%d action=%d retcode=%u comment=%s",slots[n].command,(int)checked,(int)request.action,check.retcode,check.comment);

      return(false);
     }
//+------------------------------------------------------------------+
//| PrepareBar - cancels expired entry orders at a new bar           |
//+------------------------------------------------------------------+
   bool              PrepareBar(datetime bar)
     {
      if(!symbol.Read())
         return(false);

      //--- prior generation entry orders expire even when a slot stays pending
      //--- this cancellation must not overwrite the original request identity
      for(int n=0;n<ArraySize(slots);n++)
        {
         ReconcileSlot(n);

         for(int i=OrdersTotal()-1;i>=0;i--)
           {
            ulong order=OrderGetTicket(i);

            if(!order || OrderGetString(ORDER_SYMBOL)!=_Symbol || (ulong)OrderGetInteger(ORDER_MAGIC)!=slots[n].magic)
               continue;

            MqlTradeRequest req;
            MqlTradeResult result;
            MqlTradeCheckResult check;

            ZeroMemory(req);
            ZeroMemory(result);
            ZeroMemory(check);

            req.action=TRADE_ACTION_REMOVE;
            req.order=order;
            req.symbol=_Symbol;
            req.magic=slots[n].magic;
            req.comment=StringFormat("C%I64X:%I64X",run_id,++next_request);

            if(CheckRequest(n,req,check) && !OrderSend(req,result))
               PrintFormat("Habi cancel failed slot=%d order=%I64u retcode=%u",n,order,result.retcode);

            if(OrderSelect(order))
              {
               slots[n].pending=true;

               if(slots[n].request_order==0)
                  slots[n].request_order=order;
              }
           }

         ReconcileSlot(n);
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Request - checks and sends one trade request                     |
//+------------------------------------------------------------------+
   bool              Request(int n,MqlTradeRequest &request)
     {
      if(slots[n].pending || slots[n].unknown || iTime(_Symbol,TimeFrame,0)!=slots[n].bar)
         return(false);

      request.symbol=_Symbol;
      request.magic=slots[n].magic;
      request.comment=StringFormat("H%I64X:%I64X",run_id,next_request+1);
      request.deviation=AllowedDeviationPoints;

      if(request.action==TRADE_ACTION_DEAL)
         request.type_filling=Filling();

      MqlTradeCheckResult check;

      ZeroMemory(check);

      if(!CheckRequest(n,request,check))
         return(false);

      slots[n].request_id=++next_request;

      MqlTradeResult result;

      ZeroMemory(result);

      slots[n].issued=TimeCurrent();

      bool sent=OrderSend(request,result);

      slots[n].request_order=result.order;
      slots[n].request_deal=result.deal;

      RememberOrder(result.order,n);

      if(result.deal>0 && !UpdateCashDeal(result.deal))
         QueueCashRetry(result.deal);

      PrintFormat("Habi request run=%I64u generation=%u slot=%d command=%I64u position_ticket=%I64u order=%I64u deal=%I64u retcode=%u",run_id,generation,n,slots[n].command,request.position,result.order,result.deal,result.retcode);

      if(result.retcode==TRADE_RETCODE_PLACED || result.retcode==TRADE_RETCODE_DONE_PARTIAL || result.retcode==TRADE_RETCODE_TIMEOUT || result.retcode==0)
        {
         slots[n].pending=true;
         slots[n].unknown=result.retcode==TRADE_RETCODE_TIMEOUT || result.retcode==0;

         ReconcileSlot(n);

         //--- an uncertain command is never continued, even if an event resolves it
         return(false);
        }

      if(result.retcode==TRADE_RETCODE_DONE || result.retcode==TRADE_RETCODE_NO_CHANGES)
         return(sent);

      EconomicEvidence(n,request,result.retcode);

      //--- a retcode is evidence to investigate, not an automatic penalty
      return(false);
     }
//+------------------------------------------------------------------+
//| CloseVolume - closes a volume in whole position parts            |
//+------------------------------------------------------------------+
   bool              CloseVolume(int n,double volume)
     {
      SHabiPosition p[];

      if(!Positions(n,p))
         return(false);

      for(int i=0;i<ArraySize(p) && volume>=symbol.lot_min-1e-9;i++)
        {
         double amount=MathMin(volume,MathAbs(p[i].lot));

         if(amount<MathAbs(p[i].lot)-1e-9)
           {
            amount=HabiLot(amount,symbol);

            double rest=MathAbs(p[i].lot)-amount;

            if(rest>1e-9 && rest<symbol.lot_min)
               return(false);
           }

         if(amount<symbol.lot_min)
            return(false);

         MqlTick tick;

         if(!SymbolInfoTick(_Symbol,tick))
            return(false);

         MqlTradeRequest req;

         ZeroMemory(req);

         req.action=TRADE_ACTION_DEAL;
         req.position=p[i].ticket;
         req.volume=amount;
         req.type=p[i].lot>0?ORDER_TYPE_SELL:ORDER_TYPE_BUY;
         req.price=p[i].lot>0?tick.bid:tick.ask;

         if(!Request(n,req))
            return(false);

         double remaining=0;

         if(PositionSelectByTicket(p[i].ticket))
            remaining=PositionGetDouble(POSITION_VOLUME);

         double actual=MathAbs(p[i].lot)-remaining;

         if(actual<=0)
            return(false);

         volume-=actual;
        }

      return(volume<symbol.lot_min-1e-9);
     }
//+------------------------------------------------------------------+
//| CancelEntries - removes the pending orders of a slot             |
//+------------------------------------------------------------------+
   bool              CancelEntries(int n)
     {
      for(int i=OrdersTotal()-1;i>=0;i--)
        {
         ulong t=OrderGetTicket(i);

         if(!t)
            return(false);

         if(OrderGetString(ORDER_SYMBOL)!=_Symbol || (ulong)OrderGetInteger(ORDER_MAGIC)!=slots[n].magic)
            continue;

         MqlTradeRequest req;

         ZeroMemory(req);

         req.action=TRADE_ACTION_REMOVE;
         req.order=t;

         if(!Request(n,req) || OrderSelect(t))
            return(false);
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Execute - brings the position of a slot to a target              |
//+------------------------------------------------------------------+
   bool              Execute(int n,const float &raw[],datetime bar)
     {
      //--- a generation expires at this bar boundary, never replay a skipped command
      if(!ReconcileSlot(n))
         return(true);

      slots[n].command=++next_command;
      slots[n].generation=generation;
      slots[n].bar=bar;
      slots[n].reason=HABI_OK;

      MqlTick tick;

      if(!SymbolInfoTick(_Symbol,tick))
         return(false);

      SHabiPosition p[];

      if(!Positions(n,p))
         return(false);

      double current=0;
      double margin_buy=0;
      double margin_sell=0;

      for(int i=0;i<ArraySize(p);i++)
         current+=p[i].lot;

      double q=symbol.lot_min;

      if(!OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,q,tick.ask,margin_buy) || !OrderCalcMargin(ORDER_TYPE_SELL,_Symbol,q,tick.bid,margin_sell))
         return(false);

      margin_buy/=q;
      margin_sell/=q;

      SHabiAction a;

      double own_margin=MathAbs(current)*(current>0?margin_buy:margin_sell);

      if(!HabiMap(raw,tick.bid,tick.ask,symbol,AccountInfoDouble(ACCOUNT_MARGIN_FREE)+own_margin,margin_buy,margin_sell,a))
         return(false);

      slots[n].reason=a.reason;

      bool close=a.reason==HABI_FLAT || a.reason==HABI_ZERO_PROTECTION || current*a.lot<0;

      if(close)
        {
         if(!CancelEntries(n))
            return(true);

         //--- close gross positions, including both sides of a corrupted state
         double gross=0;

         for(int i=0;i<ArraySize(p);i++)
            gross+=MathAbs(p[i].lot);

         if(gross>0 && !CloseVolume(n,gross))
            return(true);

         if(!Positions(n,p) || ArraySize(p)!=0)
            return(true);

         current=0;

         if(a.reason!=HABI_FLAT && a.reason!=HABI_ZERO_PROTECTION)
           {
            //--- a confirmed closure may change available money through actual fees
            if(!SymbolInfoTick(_Symbol,tick) || !OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,q,tick.ask,margin_buy) || !OrderCalcMargin(ORDER_TYPE_SELL,_Symbol,q,tick.bid,margin_sell))
               return(false);

            margin_buy/=q;
            margin_sell/=q;

            if(!HabiMap(raw,tick.bid,tick.ask,symbol,AccountInfoDouble(ACCOUNT_MARGIN_FREE),margin_buy,margin_sell,a))
               return(false);

            slots[n].reason=a.reason;
           }
        }

      if(a.reason!=HABI_OK)
         return(true);

      if(MathAbs(current)>MathAbs(a.lot))
        {
         if(!CloseVolume(n,MathAbs(current)-MathAbs(a.lot)))
            return(true);
        }

      if(!Positions(n,p))
         return(false);

      current=0;

      for(int i=0;i<ArraySize(p);i++)
         current+=p[i].lot;

      double addition=MathAbs(a.lot)-MathAbs(current);

      if(addition>=symbol.lot_min-1e-9)
        {
         MqlTradeRequest req;

         ZeroMemory(req);

         req.action=TRADE_ACTION_DEAL;
         req.volume=addition;
         req.type=a.lot>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL;
         req.price=a.lot>0?tick.ask:tick.bid;
         req.sl=a.sl;
         req.tp=a.tp;

         if(!Request(n,req))
            return(true);
        }

      if(!Positions(n,p))
         return(false);

      for(int i=0;i<ArraySize(p);i++)
        {
         if(!PositionSelectByTicket(p[i].ticket))
            return(true);

         if(MathAbs(PositionGetDouble(POSITION_SL)-a.sl)<symbol.tick*.1 && MathAbs(PositionGetDouble(POSITION_TP)-a.tp)<symbol.tick*.1)
            continue;

         MqlTradeRequest req;

         ZeroMemory(req);

         req.action=TRADE_ACTION_SLTP;
         req.position=p[i].ticket;
         req.sl=a.sl;
         req.tp=a.tp;

         if(!Request(n,req))
            return(true);
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Begin - opens a transition record and applies actions            |
//+------------------------------------------------------------------+
   bool              Begin(const float &observation[],const float &distribution[],const float &actions[],datetime bar)
     {
      int k=ArraySize(records);
      int N=ArraySize(slots);

      ArrayResize(records,k+1);

      ZeroMemory(records[k]);

      records[k].id=++next_record;
      records[k].start=bar;

      ArrayCopy(records[k].observation,observation);
      ArrayCopy(records[k].distribution,distribution);
      ArrayCopy(records[k].actions,actions);

      ArrayResize(records[k].start_float,N);
      ArrayResize(records[k].end_float,N);
      ArrayResize(records[k].penalty,N);
      ArrayResize(records[k].command,N);
      ArrayResize(records[k].reward,N);
      ArrayResize(records[k].done,N);

      ArrayInitialize(records[k].done,0);

      if(current_bar!=bar || ArraySize(boundary_cash)!=N)
         return(false);

      records[k].start_cash_revision=boundary_cash_revision;

      ArrayCopy(records[k].start_cash,boundary_cash);
      ArrayCopy(records[k].start_float,boundary_float);
      ArrayCopy(records[k].money_known,boundary_known);

      generation++;

      for(int n=0;n<N;n++)
        {
         float raw[];

         ArrayResize(raw,6);

         ArrayCopy(raw,actions,0,n*6,6);

         ulong old=slots[n].command;

         if(!Execute(n,raw,bar))
            return(false);

         records[k].command[n]=slots[n].command;
         records[k].penalty[n]=slots[n].command!=old && (slots[n].reason==HABI_ZERO_PROTECTION || slots[n].reason==HABI_ECONOMIC_INVALID)?PenaltyInvalidCurrency:0;
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Boundary - closes transitions at a bar boundary                  |
//+------------------------------------------------------------------+
   bool              Boundary(datetime bar,const float &next[])
     {
      int count=ArraySize(slots);

      current_bar=bar;

      ArrayResize(boundary_cash,count);
      ArrayResize(boundary_float,count);
      ArrayResize(boundary_known,count);

      for(int n=0;n<count;n++)
        {
         ReconcileSlot(n);

         if(!Floating(n,boundary_float[n]))
            return(false);
        }

      RefreshCash();

      boundary_cash_revision=cash_revision;

      for(int n=0;n<count;n++)
        {
         boundary_cash[n]=cash_totals[n];
         boundary_known[n]=cash_ready && !slots[n].unknown;

         if(!boundary_known[n])
            PrintFormat("Habi UNRESOLVED monetary boundary slot=%d command=%I64u",n,slots[n].command);
        }

      for(int k=0;k<ArraySize(records);k++)
        {
         if(records[k].boundary || records[k].used || records[k].gap || records[k].start>=bar)
            continue;

         int distance=iBarShift(_Symbol,TimeFrame,records[k].start,true);
         int end_shift=iBarShift(_Symbol,TimeFrame,bar,true);

         if(distance<0 || end_shift<0 || distance-end_shift!=1)
           {
            records[k].gap=true;

            PrintFormat("Habi DATA_GAP interval=%I64u missed bar boundary",records[k].id);

            continue;
           }

         records[k].finish=bar;
         records[k].boundary=true;

         ArrayCopy(records[k].next,next);

         records[k].end_cash_revision=boundary_cash_revision;

         ArrayCopy(records[k].end_cash,boundary_cash);
         ArrayCopy(records[k].end_float,boundary_float);

         for(int n=0;n<count;n++)
            records[k].money_known[n]=records[k].money_known[n] && boundary_known[n];
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Ready - checks and stores the reward of a record                 |
//+------------------------------------------------------------------+
   bool              Ready(int k)
     {
      if(!records[k].boundary || records[k].used || records[k].gap)
         return(false);

      int N=ArraySize(slots);

      for(int n=0;n<N;n++)
        {
         if((slots[n].pending || slots[n].unknown) && slots[n].command==records[k].command[n])
            return(false);

         //--- an unestablished monetary snapshot is not repaired with today's cash
         //--- the whole record stays unresolved instead of receiving a fake reward
         if(!records[k].money_known[n])
            return(false);

         double r=records[k].end_cash[n]-records[k].start_cash[n]+records[k].end_float[n]-records[k].start_float[n]-records[k].penalty[n];

         if(!MathIsValidNumber(r) || MathAbs(r)>FLT_MAX)
           {
            records[k].gap=true;

            return(false);
           }

         records[k].reward[n]=(float)r;
        }

      return(true);
     }
//+------------------------------------------------------------------+
//| Prune - drops used and unrecoverable records                     |
//+------------------------------------------------------------------+
   void              Prune(bool td_enabled)
     {
      int keep=0;

      for(int k=0;k<ArraySize(records);k++)
        {
         if(records[k].used || records[k].gap || (!td_enabled && records[k].boundary))
            continue;

         if(keep!=k)
            records[keep]=records[k];

         keep++;
        }

      ArrayResize(records,keep);
     }
//+------------------------------------------------------------------+
//| OldestReady - returns the first record ready for learning        |
//+------------------------------------------------------------------+
   int               OldestReady()
     {
      for(int k=0;k<ArraySize(records);k++)
         if(Ready(k))
            return(k);

      return(-1);
     }
  };
#endif

//+------------------------------------------------------------------+
