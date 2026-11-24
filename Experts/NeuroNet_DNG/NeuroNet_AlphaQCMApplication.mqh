//+------------------------------------------------------------------+
//| Typed AlphaQCM market, finance and checkpoint driver              |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_APPLICATION_MQH
#define NEURONET_ALPHAQCM_APPLICATION_MQH

#include "NeuroNet_AlphaQCMOpenCL.mqh"
#include "NeuroNet_AlphaQCMExecution.mqh"

// The runtime must outlive the inherited CNet and every borrowed buffer.
CAlphaQCMOpenCL AlphaQCMRuntime;

class CAlphaQCMApplication:public CAlphaQCMApplicationBase
  {
private:
   CNeuronAlphaQCMOCL *qcm;
   int               checkpoint_stage;
   datetime          working_boundary;
   datetime          resume_boundary,blocked_action_boundary;
   datetime          publication_checked_boundary;
   bool              publication_only,pool_warmup_staged;
   bool              signal_seen,rewarm_required;
   bool              recoverable_fault,recovery_saving;
   ulong             init_started,ready_at,save_ms,decisions,processed_bars;
public:
   bool              save_retry_pending;
                     CAlphaQCMApplication():qcm(NULL),checkpoint_stage(0),working_boundary(0),
                     resume_boundary(0),blocked_action_boundary(0),publication_checked_boundary(0),
                     publication_only(false),pool_warmup_staged(false),signal_seen(false),rewarm_required(false),
                     recoverable_fault(false),recovery_saving(false),init_started(0),ready_at(0),save_ms(0),
                     decisions(0),processed_bars(0),save_retry_pending(false) {}

   virtual bool      ReconcileDeals(datetime now) override
     {
      if(training&&(!qcm||agent!=qcm))
         ReturnFalse;
      return(CAlphaQCMApplicationBase::ReconcileDeals(now));
     }

   virtual bool      Config(float &config[]) override
     {
      if(!CAlphaQCMApplicationBase::Config(config)||ArraySize(config)!=AG_CFG_SIZE||
         ArrayResize(config,AQCM_CFG_SIZE)!=AQCM_CFG_SIZE)
         ReturnFalse;
      config[AQCM_CFG_BATCH]=(float)AQCM_PROFILE_BATCH;
      config[AQCM_CFG_CANDIDATES]=(float)AQCM_PROFILE_CANDIDATES;
      config[AQCM_CFG_QUANTILES]=(float)AQCM_PROFILE_QUANTILES;
      config[AQCM_CFG_TARGET_QUANTILES]=(float)AQCM_PROFILE_TARGET_QUANTILES;
      config[AQCM_CFG_POINTS]=(float)AQCM_PROFILE_POINTS;
      config[AQCM_CFG_REPLAY]=(float)AQCM_PROFILE_REPLAY;
      config[AQCM_CFG_WARMUP]=(float)AQCM_PROFILE_WARMUP;
      config[AQCM_CFG_CREDIT]=(float)AQCM_PROFILE_CREDIT;
      config[AQCM_CFG_SYNC]=(float)AQCM_PROFILE_SYNC;
      config[AQCM_CFG_GAMMA]=(float)AQCM_PROFILE_GAMMA;
      config[AQCM_CFG_KAPPA]=(float)AQCM_PROFILE_KAPPA;
      config[AQCM_CFG_EPSILON]=(float)AQCM_PROFILE_EPSILON;
      config[AQCM_CFG_BONUS]=(float)AQCM_PROFILE_BONUS;
      config[AQCM_CFG_FREQUENCIES]=(float)AQCM_PROFILE_FREQUENCIES;
      config[AQCM_CFG_PER_EPSILON]=(float)AQCM_PROFILE_PER_EPSILON;
      config[AQCM_CFG_PER_BETA]=(float)AQCM_PROFILE_PER_BETA;
      config[AQCM_CFG_PER_BETA_UPDATES]=(float)AQCM_PROFILE_PER_BETA_UPDATES;
      config[AQCM_CFG_PER_ALPHA]=(float)AQCM_PROFILE_PER_ALPHA;
      return(true);
     }

   bool              InitializeModel(const string path,const bool train,const bool common,bool &exists)
     {
      const ulong started=GetTickCount64();
      if(!SelectiveOpenCL)
         ReturnFalseEx("AlphaQCM requires its selective GPU runtime");
      exists=FileIsExist(path,common?FILE_COMMON:0);
      if(!train&&!exists)
         ReturnFalseEx("AlphaQCM inference requires a checkpoint");
      AlphaQCMLoadMode=train?AQCM_TRAIN:AQCM_INFER;
      if(!Config(AlphaQCMExpectedConfig))
         ReturnFalse;
      if(!AlphaQCMRuntime.Begin()||!model.UseOpenCLRuntime(GetPointer(AlphaQCMRuntime)))
         ReturnFalse;
      float error=0,undefined=0,forecast=0;
      datetime stamp=0;
      if(exists)
        {
         if(!model.Load(path,error,undefined,forecast,stamp,common))
            ReturnFalseEx("AlphaQCM checkpoint load failed: "+path);
        }
      else
        {
         CArrayObj description;
         CLayerDescription *in=new CLayerDescription();
         if(!in)
            ReturnFalse;
         if(!description.Add(in))
           {
            delete in;
            ReturnFalse;
           }
         CLayerDescription *out=new CLayerDescription();
         if(!out)
            ReturnFalse;
         if(!description.Add(out))
           {
            delete out;
            ReturnFalse;
           }
         in.type=defNeuronBaseOCL;
         in.count=AG_CHANNELS*window+AG_ACCOUNT_SIZE;
         in.activation=None;
         in.optimization=ADAM;
         in.batch=1;
         out.type=defNeuronAlphaQCMOCL;
         out.count=AG_OUTPUT_SIZE;
         out.activation=None;
         out.optimization=ADAM;
         out.batch=1;
         out.window=window;
         out.window_out=AG_POOL_SIZE;
         out.step=AG_LOOKBACK;
         out.layers=AG_MAX_TOKENS;
         if(!model.Create(GetPointer(description)))
            ReturnFalse;
        }
      if(!model.Layer(0)||model.Layer(0).Type()!=defNeuronBaseOCL||!model.Layer(1)||CheckPointer(model.Layer(2))!=POINTER_INVALID||
         model.Layer(1).Type()!=defNeuronAlphaQCMOCL||
         model.Layer(1).Neurons()!=AG_OUTPUT_SIZE)
         ReturnFalseEx("Incompatible AlphaQCM model geometry");
      qcm=(CNeuronAlphaQCMOCL*)model.Layer(1);
      // The donor's ledger calls AdjustReward only on these same trade rows.
      agent=qcm;
      if(!exists&&!qcm.Configure(AlphaQCMExpectedConfig,Seed))
         ReturnFalse;
      ENUM_TIMEFRAMES timeframe=TimeFrame==PERIOD_CURRENT?_Period:TimeFrame;
      if(!AlphaQCMRuntime.Build()||!qcm.AfterBuild(!exists)||!qcm.CausalIdentity(_Symbol,timeframe))
         ReturnFalse;
      resume_boundary=(datetime)qcm.CausalWatermark();
      event_id=qcm.LastServiceEvent();
      decision_id=qcm.LastServiceDecision();
      PrintFormat("AlphaQCM OpenCL initialized in %I64u ms, context %d program %d",
                  GetTickCount64()-started,model.GetOpenCL().GetContext(),model.GetOpenCL().GetProgram());
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Init - validate the run and initialize the shared state.         |
   //+------------------------------------------------------------------+
   bool              Init(bool train,bool offline)
     {
      init_started=GetTickCount64();
      training=train;
      service_rebase=true;
      simulated=offline;
      LastBarTime=0;
      finance_started=TimeCurrent();
      if(window<AG_LOOKBACK+MathMax(AG_IC_WINDOW+AG_HORIZON,AG_NORM_WINDOW))
        {
         ReturnFalse;
        }
      if(train&&!offline&&!MQLInfoInteger(MQL_TESTER))
        {
         Print("StudyOnline requires Strategy Tester");
         ReturnFalse;
        }
      bool exists=false;
      if(!InitializeModel(ModelFile,train,true,exists))
         ReturnFalse;
      qcm.SetBoundedLearner(offline);
      if(!offline)
        {
         // Online runs start a separate external session. Reuse the fit,
         // but rebuild its working normalizer on the current history.
         resume_boundary=0;
        }
      window=(int)qcm.ConfigValue(AG_CFG_W);
      context_bars=(int)qcm.ConfigValue(AG_CFG_CONTEXT);
      trade_gamma=(double)qcm.ConfigValue(AG_CFG_TRADE_GAMMA);
      int  lookback=(int)qcm.ConfigValue(AG_CFG_LOOKBACK),norm=(int)qcm.ConfigValue(AG_CFG_NORM_WINDOW);
      int  ic=(int)qcm.ConfigValue(AG_CFG_IC_WINDOW),horizon=(int)qcm.ConfigValue(AG_CFG_HORIZON);
      long required=(long)window+context_bars-1;
      long inputs=(long)AG_CHANNELS*window+AG_ACCOUNT_SIZE;
      long metadata=(long)AG_CTX_SIZE+AG_CONTEXT_METADATA*window;
      if(!MathIsValidNumber(trade_gamma)||trade_gamma<=0||trade_gamma>1||window<=0||
         context_bars<=0||lookback<0||norm<=0||ic<=0||horizon<=0||required>INT_MAX||
         inputs>INT_MAX||metadata>INT_MAX||(long)window<(long)lookback+MathMax((long)ic+horizon,(long)norm)||
         model.Layer(0).Neurons()!=AG_CHANNELS*window+AG_ACCOUNT_SIZE)
        {
         ReturnFalse;
        }
      warmup_bars=(int)MathMax(window,required);
      if(!model.TrainMode(train))
        {
         ReturnFalse;
        }
      if(ArrayResize(context,(int)metadata)!=(int)metadata||ArrayResize(input_values,(int)inputs)!=(int)inputs)
        {
         ReturnFalse;
        }
      ArrayInitialize(context,0);
      ArrayInitialize(input_values,0);
      double capital=AccountInfoDouble(ACCOUNT_BALANCE);
#ifdef ALPHAGEN_STUDY
      if(offline)
        {
         if(Start>=End||Epochs<1||MinBalance<=0||MaxBalance<MinBalance)
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
         int last_shift=MathMax(1,iBarShift(_Symbol,TimeFrame,End-1,false));
         int earliest=start_shift;
         if(resume_boundary>0&&resume_boundary<Start)
           {
            int saved_shift=iBarShift(_Symbol,TimeFrame,resume_boundary,true);
            if(saved_shift>=start_shift)
               earliest=saved_shift;
           }
         long requested=(long)earliest+warmup_bars-last_shift+1;
         if(requested<1||requested>INT_MAX)
            ReturnFalse;
         int copied=CopyRates(_Symbol,TimeFrame,last_shift,(int)requested,history);
         if(copied<=0)
           {
            ReturnFalse;
           }
         cursor=0;
         while(cursor<copied&&history[cursor].time<Start)
           {
            cursor++;
           }
         if(cursor<window||cursor>=copied||history[cursor].time>=End)
           {
            ReturnFalse;
           }
         epoch=1;
        }
#endif
      external_cash=reported_equity=capital;
      finance_balance=finance_equity=capital;
      finance_boundary_msc=(long)finance_started*1000;
      if(!money.Init(offline,capital))
        {
         ReturnFalse;
        }
      if(!training&&qcm.ActiveCount()<1)
        {
         Print("AlphaQCM Test requires a checkpoint with a published Active Pool");
         ReturnFalse;
        }
      if(training)
         PrintFormat("AlphaQCM initial V=%I64u U=%I64u target_syncs=%I64u prepared_revision=%I64u active_generation=%I64u trade_updates=%I64u",
                     qcm.SearchV(),qcm.SearchU(),qcm.TargetSyncs(),qcm.PreparedRevision(),qcm.ActiveGeneration(),qcm.TradeUpdates());
      PrintFormat("AlphaQCM %s model %s | initial seed %I64u | TimeFrame %s | W %d | context %d",exists?"loaded":"created",ModelFile,qcm.InitialSeed(),EnumToString(TimeFrame),window,context_bars);
#ifdef ALPHAGEN_STUDY
      PrintFormat("AlphaQCM Study requested history [%s, %s), epochs %d, initial virtual capital %.2f",TimeToString(Start),TimeToString(End),Epochs,capital);
#endif
      ready_at=GetTickCount64();
      PrintFormat("AlphaQCM READY profile=IQN-QCM-v1 init_ms=%I64u",ready_at-init_started);
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Encode exact identifiers and publish CPU service metadata.       |
   //+------------------------------------------------------------------+
   bool              Context(int phase,int flags,datetime boundary)
     {
      context[AG_CTX_PHASE]=(float)phase;
      context[AG_CTX_FLAGS]=(float)flags;
      AlphaQCMInteger(context,AG_CTX_DECISION,decision_id);
      AlphaQCMInteger(context,AG_CTX_EVENT,event_id);
      AlphaQCMInteger(context,AG_CTX_BOUNDARY,(ulong)boundary);
      context[AG_CTX_EQUITY]=(float)money.equity;
      context[AG_CTX_REFERENCE]=(float)money.reference;
      if(!service.AssignArray(context)||!(service.GetIndex()<0?service.BufferCreate(model.GetOpenCL()):service.BufferWrite()))
        {
         ReturnFalse;
        }
      qcm.Phase(phase);

      return(true);
     }
   //+------------------------------------------------------------------+
   //| ComputationStatus - name each terminal computation status.       |
   //+------------------------------------------------------------------+
   string            ComputationStatus() const
     {
      int status=qcm.Status();
      switch(status)
        {
         case AQCM_RESOURCE_UNSUPPORTED:
            return("RESOURCE_UNSUPPORTED");
         case AQCM_OPENCL_ERROR:
            return("OPENCL_ERROR");
         case AQCM_NUMERIC_ERROR:
            return("NUMERIC_ERROR");
         case AQCM_UPDATE_INCONSISTENT:
         case AG_CORRUPTED:
            return("UPDATE_INCONSISTENT");
         case AQCM_INCOMPLETE_DATA:
            return("INCOMPLETE_DATA");
         case AG_TECHNICAL:
            return("TECHNICAL_ERROR");
        }
      return(IntegerToString(status));
     }
   //+------------------------------------------------------------------+
   //| Forward - run one native phase and propagate numerical failures. |
   //+------------------------------------------------------------------+
   bool              Forward(int phase,int flags,datetime boundary)
     {
      if(failed)
        {
         ReturnFalse;
        }
      ulong start=GetMicrosecondCount();
      if(!Context(phase,flags,boundary)||!raw.AssignArray(input_values)||!model.feedForward(GetPointer(raw),1,false,GetPointer(service)))
        {
         recoverable_fault=simulated&&qcm.CanRecover();
         Fault("native forward failed; status="+ComputationStatus());
         ReturnFalse;
        }
      ulong elapsed=GetMicrosecondCount()-start;
      forward_us+=elapsed;
      max_forward=MathMax(max_forward,elapsed);
      forward_calls++;
      if(qcm.Status()==AG_CORRUPTED||qcm.Status()==AG_TECHNICAL||qcm.Status()>=AQCM_RESOURCE_UNSUPPORTED)
        {
         recoverable_fault=simulated&&qcm.CanRecover();
         Fault("native computation failed; status="+ComputationStatus());
         ReturnFalse;
        }

      return(true);
     }
   //+------------------------------------------------------------------+
   //| Apply one native PPO example or drain a terminal ready tail.     |
   //+------------------------------------------------------------------+
   bool              Train(bool drain=false)
     {
      if(!training)
        {
         return(true);
        }
      ulong begin=GetMicrosecondCount();
      while(qcm.HasTraining())
        {
         if(!qcm.PrepareUpdate())
           {
            Fault("native PPO preparation failed");
            ReturnFalse;
           }
         if(!Forward(AG_BACKWARD,0,LastBarTime))
           {
            ReturnFalse;
           }
         CBufferFloat echo;
         float        zeros[];
         ArrayResize(zeros,AG_OUTPUT_SIZE);
         ArrayInitialize(zeros,0);
         ulong previous_trade_updates=qcm.TradeUpdates();
         if(!echo.AssignArray(zeros)||!model.backProp(GetPointer(echo),GetPointer(service)))
           {
            Fault("native PPO backward failed");
            ReturnFalse;
           }
         updates++;
         if(qcm.TradeUpdates()>previous_trade_updates)
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
   //| Confirm one real transition without fabricating bootstrap data.  |
   //+------------------------------------------------------------------+
   bool              Feedback(bool terminal,bool ready,bool ruin,datetime boundary)
     {
      if(!training||!open_decision)
        {
         return(true);
        }
      // Reconcile against the same available quote as the equity sample.
      datetime confirmed_at=snapshot_msc>0?(datetime)(snapshot_msc/1000):TimeCurrent();
      if(!simulated&&!ReconcileDeals(confirmed_at))
        {
         Fault("INCOMPLETE_DATA: pre-feedback cash reconciliation failed");

         return(false);
        }
      if(!qcm.IssueEvent(event_id))
         ReturnFalse;
      context[AG_CTX_MONEY_CONFIRMED]=(!money.pending&&!money.reconciling&&!money.unknown_request)?1.0f:0.0f;
      context[AG_CTX_TERMINAL]=terminal?1.0f:0.0f;
      context[AG_CTX_NEXT_READY]=ready?1.0f:0.0f;
      context[AG_CTX_DELTA]=(float)MathMax(delta,1);
      context[AG_CTX_REWARD]=(float)(interval_reward-(invalid?AG_INVALID_PENALTY:0)-MathPow(trade_gamma,MathMax(delta-1,0))*(ruin?AG_RUIN_PENALTY:0));
      if(!Forward(AG_FLUSH,terminal?AG_FLAG_END:0,boundary))
        {
         ReturnFalse;
        }
      if(qcm.Status()!=AG_WAIT&&!money.pending&&!money.reconciling&&!money.unknown_request&&(terminal||ready))
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
   // Stage closed OHLCV only. The pool owns the device copy for both
   // independent Active rewarm and preparation of a candidate publication.
   bool              StageHistory(MqlRates &bars[],int end,bool &valid)
     {
      valid=false;
      if(end<warmup_bars)
         return(true);
      if(end>ArraySize(bars)||warmup_bars<window||warmup_bars>INT_MAX/5)
         ReturnFalse;
      float history_values[];
      if(ArrayResize(history_values,5*warmup_bars)!=5*warmup_bars)
         ReturnFalse;
      for(int i=0;i<warmup_bars;i++)
        {
         MqlRates bar=bars[end-warmup_bars+i];
         if(!MathIsValidNumber(bar.open+bar.high+bar.low+bar.close)||bar.open<=0||
            bar.high<bar.low||bar.close<=0||bar.spread<0||bar.tick_volume<0||
            (i>0&&bar.time<=bars[end-warmup_bars+i-1].time))
            return(true);
         history_values[i]=(float)bar.open;
         history_values[warmup_bars+i]=(float)bar.high;
         history_values[2*warmup_bars+i]=(float)bar.low;
         history_values[3*warmup_bars+i]=(float)bar.close;
         history_values[4*warmup_bars+i]=(float)bar.tick_volume;
        }
      valid=true;
      return(qcm.SetPoolWarmupHistory(history_values,warmup_bars));
     }

   //+------------------------------------------------------------------+
   //| Advance market context and finish the causal decision cycle.     |
   //+------------------------------------------------------------------+
   bool              AdvancePoolWarmup()
     {
      int end=ArraySize(pool_warmup_rates);
      if(pool_warmup_end<window||pool_warmup_end>end)
        {
         Fault("invalid pinned pool warmup cursor");
         ReturnFalse;
        }
      if(!pool_warmup_staged)
        {
         bool valid=false;
         if(!StageHistory(pool_warmup_rates,end,valid))
            ReturnFalse;
         if(!valid)
           {
            if(!qcm.AbortPoolWarmup())
               ReturnFalse;
            ClearPoolWarmup();
            return(true);
           }
         if(!Pack(pool_warmup_rates,end,pool_warmup_boundary,false)||
            !Forward(AG_PREPARE,AG_FLAG_WARMUP|AG_FLAG_RESET_WORK,pool_warmup_boundary))
            ReturnFalse;
         pool_warmup_staged=qcm.NeedsPoolWarmup();
         if(!pool_warmup_staged)
           {
            ClearPoolWarmup();
           }

         return(true);
        }

      if(!simulated&&iTime(_Symbol,TimeFrame,0)!=pool_warmup_boundary)
        {
         if(!qcm.AbortPoolWarmup())
            ReturnFalse;
         ClearPoolWarmup();
         publication_only=true;
         return(true);
        }
      if(!Forward(AG_PREPARE,AG_FLAG_WARMUP|AG_FLAG_END,pool_warmup_boundary))
        {
         ReturnFalse;
        }
      bool published=qcm.Status()==AG_PUBLISHED;
      if(published)
        {
         service_rebase=true;
         observed_boundary=pool_warmup_boundary;
         LastBarTime=pool_warmup_boundary;
         publication_only=false;
         signal_seen=true;
         rewarm_required=false;
         if(!simulated)
            processed_bars++;
        }
      ClearPoolWarmup();
      cycle_complete=published;

      return(true);
     }
   //+------------------------------------------------------------------+
   //| ClearPoolWarmup - release the prepared publication snapshot.      |
   //+------------------------------------------------------------------+
   void              ClearPoolWarmup()
     {
      pool_warmup_end=0;
      pool_warmup_boundary=0;
      pool_warmup_staged=false;
      ArrayFree(pool_warmup_rates);
     }
   //+------------------------------------------------------------------+
   //| Advance market context and finish the causal decision cycle.     |
   //+------------------------------------------------------------------+
   bool              SkipBoundary(const datetime boundary)
     {
      if(!simulated&&LastBarTime!=boundary)
         processed_bars++;
      LastBarTime=boundary;
      cycle_complete=true;
      publication_only=false;
      return(true);
     }

   bool              Boundary(MqlRates &bars[],int end,datetime boundary,double bid,double ask)
     {
      cycle_complete=false;
      ulong begin=GetMicrosecondCount();
      if(money.pending||money.reconciling||money.unknown_request)
        {
         return(true);
        }
      if(pool_warmup_end>0)
        {
         if(pool_warmup_boundary==boundary)
            return(AdvancePoolWarmup());
         if(!qcm.AbortPoolWarmup())
            ReturnFalse;
         ClearPoolWarmup();
         publication_only=true;
        }
      if(training&&working_boundary!=0&&working_boundary!=boundary)
        {
         if(qcm.BootstrapPending())
           {
            if(!qcm.ContinueBootstrap())
              {
               Fault("pinned bootstrap failed; status="+ComputationStatus());
               ReturnFalse;
              }
            if(qcm.BootstrapPending())
               return(true);
           }
         if(!qcm.FinishSearch(true))
            ReturnFalse;
         working_boundary=0;
         publication_only=true;
        }
      if(training&&qcm.HasTraining()&&observed_boundary!=boundary)
        {
         if(!Train())
           {
            ReturnFalse;
           }

         return(true);
        }
      if(!Pack(bars,end,boundary,true))
        {
         return(SkipBoundary(boundary));
        }
      int count=1;
      bool resume_here=!rewarm_required&&resume_boundary==boundary&&qcm.ActiveCount()>0;
      datetime market_cursor=rewarm_required?0:observed_boundary;
      if(resume_boundary>0&&resume_boundary<boundary)
        {
         // Require the exact saved boundary and its lookback in the same stream.
         for(int i=end-1;i>=window-1;i--)
            if(bars[i].time<=resume_boundary)
              {
               if(bars[i].time==resume_boundary)
                  market_cursor=resume_boundary;
               break;
              }
        }
      resume_boundary=0;
      if(market_cursor!=0)
        {
         count=0;
         for(int i=end-1;i>=0&&bars[i].time>=market_cursor;i--)
           {
            count++;
           }
         if(count==0&&observed_boundary!=boundary)
           {
            return(SkipBoundary(boundary));
           }
        }
      bool reset=!resume_here&&(market_cursor==0||end-count<window-1);
      if(reset&&qcm.ActiveCount()>0)
        {
         bool valid=false;
         if(!StageHistory(bars,end,valid))
            ReturnFalse;
         if(!valid)
            return(SkipBoundary(boundary));
        }
      if(observed_boundary==boundary&&open_decision)
        {
         if(!Forward(AG_OBSERVE,AG_FLAG_VALUE,boundary))
            ReturnFalse;
         if(qcm.Status()!=AG_OK)
            return(SkipBoundary(boundary));
         RealMoneySample();
         if(!Feedback(false,true,false,boundary))
            ReturnFalse;
         if(open_decision)
            return(true);
        }
      if(observed_boundary!=boundary)
        {
         int first=reset?end-1:end-count;
         if(first<window-1||(reset&&end<window))
           {
            return(SkipBoundary(boundary));
           }
         // Fixed-normalizer warmup consumes all valid rows in one W-window.
         for(int i=first;i<end;i++)
           {
            bool     latest=i==end-1;
            datetime at=latest?boundary:bars[i+1].time;
            if(!reset&&at<=observed_boundary)
              {
               continue;
              }
            if(!Pack(bars,i+1,at,latest))
              {
               return(SkipBoundary(boundary));
              }
            int flags=reset?(AG_FLAG_WARMUP|AG_FLAG_RESET_WORK|AG_FLAG_END):(resume_here?AG_FLAG_VALUE:AG_FLAG_ADVANCE);
            if(latest&&open_decision)
              {
               flags|=AG_FLAG_VALUE;
              }
            if(!Forward(AG_OBSERVE,flags,at))
              {
               ReturnFalse;
              }
            if(qcm.Status()==AG_OK)
              {
               observed_boundary=at;
               signal_seen=true;
               rewarm_required=false;
              }
           }
         if(qcm.Status()==AG_WAIT&&(open_decision||!training))
           {
            return(SkipBoundary(boundary));
           }
         if(qcm.Status()==AG_INVALID)
           {
            if(reset)
              {
               rewarm_required=true;
               if(!training&&!signal_seen)
                 {
                  Fault("NO_VALID_RESULT: checkpoint Active is unusable after causal rewarm");
                  ReturnFalse;
                 }
              }
            SAlphaQCMTarget flat;
            ZeroMemory(flat);
            if(!ExecuteTarget(flat,bid,ask,boundary)||money.pending||money.reconciling||money.volume!=0)
              {
               return(true);
              }
            blocked_action_boundary=boundary;
            if(open_decision)
              {
               interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
               last_equity=money.equity;
               if(!Feedback(true,false,false,boundary))
                 {
                  ReturnFalse;
                 }
              }
            else
               if(training)
                 {
                  service_rebase=true;
                  if(!qcm.IssueEvent(event_id))
                     ReturnFalse;
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
            if(open_decision&&qcm.Status()!=AG_OK)
              {
               return(SkipBoundary(boundary));
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
                   if(open_decision)
                      return(true);
                 }
              }
           }
         observed_boundary=boundary;
        }
      if(training)
        {
         bool preparation_done=false;
         if(qcm.HasTraining())
           {
            if(!Train())
              {
               ReturnFalse;
              }
            if(simulated||qcm.HasTraining())
              {
               return(true);
              }
           }
         if(simulated&&qcm.HasSearchDebt())
           {
            if(!qcm.DrainSearchStep())
              {
               recoverable_fault=qcm.CanRecover();
                Fault("search update failed; status="+ComputationStatus());
               ReturnFalse;
              }
            return(true);
           }
         if(qcm.NeedsPoolWarmup()&&end<warmup_bars)
           {
            if(!qcm.AbortPoolWarmup())
              {
               ReturnFalse;
              }
            pool_warmup_end=0;
            preparation_done=true;
           }
         if(qcm.NeedsPoolWarmup())
           {
            // Pin the closed history and publication boundary for this bounded job.
            // New bars are caught up only after commit; no stale ACT is issued.
            if(ArrayCopy(pool_warmup_rates,bars,0,end-warmup_bars,warmup_bars)!=warmup_bars)
              {
               Fault("pool warmup snapshot allocation failed");
               ReturnFalse;
              }
            pool_warmup_boundary=boundary;
            pool_warmup_end=window;

            return(AdvancePoolWarmup());
           }
         if(!preparation_done&&(!publication_only||publication_checked_boundary!=boundary))
           {
            if(!Pack(bars,end,boundary,true))
               return(SkipBoundary(boundary));
            int flags=publication_only?AQCM_FLAG_PUBLICATION_ONLY:0;
            if(!Forward(AG_PREPARE,flags,boundary))
              {
               ReturnFalse;
              }
            working_boundary=qcm.Status()==AG_WORKING?boundary:0;
            if(working_boundary!=0)
               return(true);
            if(publication_only)
              {
               publication_checked_boundary=boundary;
              }
            if(!simulated&&iTime(_Symbol,TimeFrame,0)!=boundary)
              {
               if(qcm.NeedsPoolWarmup()&&!qcm.AbortPoolWarmup())
                  ReturnFalse;
               publication_only=true;
               return(true);
              }
            if(qcm.NeedsPoolWarmup()&&end>=warmup_bars)
              {
               if(ArrayCopy(pool_warmup_rates,bars,0,end-warmup_bars,warmup_bars)!=warmup_bars)
                  ReturnFalse;
               pool_warmup_boundary=boundary;
               pool_warmup_end=window;
               return(AdvancePoolWarmup());
              }
            if(qcm.NeedsPoolWarmup()&&!qcm.AbortPoolWarmup())
               ReturnFalse;
            if(qcm.Status()==AG_WAIT)
              {
               return(SkipBoundary(boundary));
              }
            if(!Train())
              {
               ReturnFalse;
              }
            if(qcm.HasTraining())
              {
               return(true);
              }
            if(qcm.Status()==AG_PUBLISHED)
              {
               service_rebase=true;
               LastBarTime=boundary;
               cycle_complete=true;
               publication_only=false;
               if(!simulated)
                  processed_bars++;

               return(true);
              }
           }
        }
      if(!simulated&&iTime(_Symbol,TimeFrame,0)!=boundary)
        {
         publication_only=training;
         return(true);
        }
      if(blocked_action_boundary==boundary)
         return(SkipBoundary(boundary));
      if(!simulated)
        {
         MqlTick current;
         if(!SymbolInfoTick(_Symbol,current)||!money.Snapshot(current.bid,current.ask,current.time))
            return(true);
         // A broker-side closure can arrive while search is running. Reconcile
         // it against the existing intervals before taking a new equity anchor.
         if(!ReconcileDeals(current.time))
           {
            Fault("INCOMPLETE_DATA: pre-action cash reconciliation failed");
            ReturnFalse;
           }
         if(iTime(_Symbol,TimeFrame,0)!=boundary)
           {
            publication_only=training;
            return(true);
           }
         if(money.pending||money.reconciling||money.unknown_request)
            return(true);
         bid=current.bid;
         ask=current.ask;
         snapshot_msc=current.time_msc;
        }
      if(!simulated&&LastBarTime!=boundary)
         processed_bars++;
      LastBarTime=boundary;
      cycle_complete=true;
      publication_only=false;
      if(service_rebase)
        {
         money.Commit();
        }
      if(!Pack(bars,end,boundary,true))
        {
         return(SkipBoundary(boundary));
        }
      service_cash+=money.equity-reported_equity;
      reported_equity=money.equity;
      money.Commit();
      if(!qcm.IssueDecision(decision_id))
         ReturnFalse;
      if(!Forward(training?AG_ACT:AG_INFER,0,boundary))
        {
         ReturnFalse;
        }
      if(!simulated&&iTime(_Symbol,TimeFrame,0)!=boundary)
        {
         if(training&&qcm.Status()==AG_ACTION_READY&&!qcm.DiscardDecision())
           {
            Fault("stale native decision cancellation failed");
            ReturnFalse;
           }
         service_rebase=true;
         publication_only=training;
         return(true);
        }
#ifdef ALPHAGEN_DECISION_TRACE
      PrintFormat("AlphaQCM decision %s status=%d active=%d prepared=%d",
                  TimeToString(boundary),qcm.Status(),qcm.ActiveCount(),qcm.PreparedCount());
#endif
      if(qcm.Status()!=AG_ACTION_READY)
        {
         service_rebase=true;
         if(qcm.Status()==AG_INVALID)
           {
            SAlphaQCMTarget flat;
            ZeroMemory(flat);
            if(!ExecuteTarget(flat,bid,ask,boundary))
              {
               return(true);
              }
           }

         return(true);
        }
      service_rebase=false;
      float action[AG_OUTPUT_SIZE];
      CBufferFloat *output=qcm.getOutput();
      if(!output||output.Total()!=AG_OUTPUT_SIZE||
         !model.GetOpenCL().BufferRead(output.GetIndex(),action,0,0,AG_OUTPUT_SIZE))
        {
         Fault("OPENCL_ERROR: trading action read failed");
         ReturnFalse;
        }
      SAlphaQCMTarget target;
      if(!money.Map(action,bid,ask,target))
        {
         ReturnFalse;
        }
      if(!simulated&&iTime(_Symbol,TimeFrame,0)!=boundary)
        {
         if(training&&!qcm.DiscardDecision())
            ReturnFalse;
         service_rebase=true;
         publication_only=training;
         return(true);
        }
      decisions++;
#ifdef ALPHAGEN_DECISION_TRACE
      if(ArraySize(action)==AG_OUTPUT_SIZE)
         PrintFormat("AlphaQCM action [%.9f %.9f %.9f %.9f %.9f %.9f] volume=%.8f invalid=%d sl=%.8f tp=%.8f",
                     action[0],action[1],action[2],action[3],action[4],action[5],
                     target.volume,(int)target.invalid,target.sl,target.tp);
#endif
      if(training)
        {
         start_equity=last_equity=money.equity;
         interval_reward=0;
         delta=0;
         decision_time=boundary;
         decision_start_msc=snapshot_msc>0?snapshot_msc:(long)boundary*1000;
        }
      open_decision=training;
      invalid=target.invalid;
      double before_volume=money.volume;
      ulong  before_requests=money.successful_requests;
      bool   executed=ExecuteTarget(target,bid,ask,boundary,decision_id);
      invalid=target.invalid;
#ifdef ALPHAGEN_DECISION_TRACE
      PrintFormat("AlphaQCM execution result=%d invalid=%d requested=%.8f actual=%.8f requests=%I64u pending=%d search_updates=%I64u trade_updates=%I64u",
                  (int)executed,(int)invalid,target.volume,money.volume,money.successful_requests,
                  (int)money.pending,qcm.SearchU(),qcm.TradeUpdates());
#endif
      if(!executed&&!money.pending&&before_volume==money.volume&&before_requests==money.successful_requests)
        {
         // A rejected technical request is not a fictitious policy sample.
         if(!qcm.DiscardDecision())
           {
            Fault("native decision cancellation failed");
            ReturnFalse;
           }
         open_decision=false;
         service_rebase=true;
         excluded++;
        }
      if(!executed&&!money.pending&&(before_volume!=money.volume||before_requests!=money.successful_requests))
        {
         SAlphaQCMTarget flat;
         ZeroMemory(flat);
         if(!ExecuteTarget(flat,bid,ask,boundary,decision_id)||money.pending)
           {
            money.intended_volume=0;
            money.intended_sl=0;
            money.intended_tp=0;
            money.reconciling=true;

            return(true);
           }
         if(training)
           {
            interval_reward+=MathPow(trade_gamma,MathMax(delta-1,0))*(money.equity-last_equity)/money.reference;
            last_equity=money.equity;
            if(!Feedback(true,false,false,boundary)||!Train())
               ReturnFalse;
           }
        }
      if(training&&target.invalid&&!money.pending&&!money.reconciling&&money.volume==0)
        {
         interval_reward+=(money.equity-last_equity)/money.reference;
         last_equity=money.equity;
         if(!Feedback(true,false,false,boundary)||!Train())
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

   bool              SavePhase(int phase)
     {
      if(!training||failed||!qcm||qcm.Type()!=defNeuronAlphaQCMOCL||
         qcm.Status()==AG_CORRUPTED||qcm.Status()==AG_TECHNICAL||
         qcm.Status()>=AQCM_RESOURCE_UNSUPPORTED||!qcm.SafeToSave()||phase<0||phase>2)
         return(false);
      ulong begin=GetTickCount64();
      string temp=ModelFile+".tmp",backup=ModelFile+".bak";
      if(phase==0)
        {
         checkpoint_stage=0;
         if(!model.Save(temp,0,0,0,TimeCurrent(),true))
            ReturnFalse;
         checkpoint_stage=1;
         save_ms+=GetTickCount64()-begin;
         return(true);
        }
      if(checkpoint_stage!=phase)
         ReturnFalse;
      if(phase==1)
        {
         CNet verify;
         if(!verify.UseOpenCLRuntime(model.GetOpenCL()))
            ReturnFalse;
         int previous=AlphaQCMLoadMode;
         AlphaQCMLoadMode=AQCM_VERIFY;
         float a=0,b=0,c=0;
         datetime t=0;
         bool valid=verify.Load(temp,a,b,c,t,true);
         AlphaQCMLoadMode=previous;
         if(!valid||!verify.Layer(0)||verify.Layer(0).Type()!=defNeuronBaseOCL||
            verify.Layer(0).Neurons()!=AG_CHANNELS*window+AG_ACCOUNT_SIZE||
            !verify.Layer(1)||CheckPointer(verify.Layer(2))!=POINTER_INVALID||verify.Layer(1).Type()!=defNeuronAlphaQCMOCL||
            verify.Layer(1).Neurons()!=AG_OUTPUT_SIZE)
            ReturnFalse;
         checkpoint_stage=2;
         save_ms+=GetTickCount64()-begin;
         return(true);
        }
      bool had=FileIsExist(ModelFile,FILE_COMMON);
      if(had&&!FileCopy(ModelFile,FILE_COMMON,backup,FILE_COMMON|FILE_REWRITE))
         ReturnFalse;
      if(!FileMove(temp,FILE_COMMON,ModelFile,FILE_COMMON|FILE_REWRITE))
        {
         if(had&&!FileCopy(backup,FILE_COMMON,ModelFile,FILE_COMMON|FILE_REWRITE))
            Print("AlphaQCM checkpoint restore failed: ",GetLastError());
         ReturnFalse;
        }
      checkpoint_stage=0;
      save_ms+=GetTickCount64()-begin;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Save and verify synchronously outside the Study timer driver.    |
   //+------------------------------------------------------------------+
   bool              Save()
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
   //| ConfirmFinanceBoundary - align the ledger with the account.      |
   //+------------------------------------------------------------------+
   bool              ConfirmFinanceBoundary()
     {
      double ledger_balance=money.reference+money.gross_profit-money.gross_loss;
      double tolerance=0.5*MathPow(10.0,-(double)AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS));
      if(money.pending||money.reconciling||money.unknown_request||
         MathAbs(ledger_balance-money.balance)>tolerance)
         return(false);
      finance_boundary_msc=snapshot_msc>last_deal_msc?snapshot_msc:last_deal_msc;
      finance_deal_msc=last_deal_msc;
      finance_deal_ticket=last_deal_ticket;
      finance_balance=money.balance;
      finance_equity=money.equity;
      finance_gross_profit=money.gross_profit;
      finance_gross_loss=money.gross_loss;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| PrintTestFinance - report the last confirmed monetary boundary.   |
   //+------------------------------------------------------------------+
   void              PrintTestFinance(const bool complete)
     {
      string profit_factor=finance_gross_loss>0?
                           DoubleToString(finance_gross_profit/finance_gross_loss,6):
                           "undefined (no realized losses)";
      PrintFormat("AlphaQCM Test finance %s boundary_msc=%I64d last_deal_msc=%I64d last_deal=%I64u "
                  "balance=%.2f equity=%.2f gross_profit=%.2f gross_loss=%.2f cash_profit_factor=%s",
                  complete?"CONFIRMED":"PRELIMINARY",finance_boundary_msc,finance_deal_msc,finance_deal_ticket,
                  finance_balance,finance_equity,finance_gross_profit,finance_gross_loss,profit_factor);
      if(!complete&&MQLInfoInteger(MQL_TESTER))
        {
         Print("AlphaQCM Tester report can include end-of-test events beyond this confirmed boundary");
        }
     }
   //+------------------------------------------------------------------+
   //| PrintRunTiming - record elapsed time and actual work performed.   |
   //+------------------------------------------------------------------+
   void              PrintRunTiming()
     {
      ulong now=GetTickCount64();
      PrintFormat("AlphaQCM workload total_ms=%I64u run_ms=%I64u save_ms=%I64u bars=%I64u "
                  "decisions=%I64u virtual_trades=%I64u confirmed_deals=%d",
                  now-init_started,now-ready_at,save_ms,processed_bars,decisions,money.trades,ArraySize(ledger));
      if(training)
        {
         PrintFormat("AlphaQCM search workload attempts=%I64u verified_episodes=%I64u transitions=%I64u",
                     qcm.SearchAttempts(),qcm.SearchEpisodes(),qcm.SearchV());
        }
     }
   //+------------------------------------------------------------------+
   //| FinishTestReport - reconcile Tester cash without another order.   |
   //+------------------------------------------------------------------+
   bool              FinishTestReport()
     {
      if(Finalized)
         return(true);
      if(training||failed)
         ReturnFalse;
      MqlTick quote;
      bool complete=false;
      if(SymbolInfoTick(_Symbol,quote))
        {
         snapshot_msc=quote.time_msc;
         datetime latest=TimeCurrent();
         if(latest<quote.time)
            latest=quote.time;
         if(money.Snapshot(quote.bid,quote.ask,quote.time)&&ReconcileDeals(latest))
           {
            complete=ConfirmFinanceBoundary()&&money.volume==0;
           }
        }
      PrintTestFinance(complete);
      PrintRunTiming();
      Finalized=true;
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Close, reconcile and finalize once before optional saving.       |
   //+------------------------------------------------------------------+
   bool              FinishExcluded(bool save,datetime time)
     {
      if(training&&!qcm.FinishSearch(!stop_requested))
         ReturnFalse;
      if(training&&open_decision)
        {
         excluded++;
         excluded_cash+=money.equity-start_equity;
         reported_equity=money.equity;
         if(!qcm.DiscardDecision())
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
            if(!qcm.CancelTraining())
              {
               Fault("unfinished finalization cancellation failed");
               ReturnFalse;
              }
           }
         else
           {
            // DISCARD preserves ready predecessors' recorded bootstrap and ends GAE
            // at their buffer edge. It never invents money or a value for the open row.
            if(!Forward(AG_FLUSH,AG_FLAG_DISCARD|AG_FLAG_END,time)||!Train(true))
               ReturnFalse;
           }
        }
      if(save&&!Save())
        {
         return(false);
        }
      Finalized=true;
      Print("AlphaQCM terminal money unresolved; unfinished sample excluded; coherent weights retained");

      return(true);
     }
   //+------------------------------------------------------------------+
   //| Close, reconcile and finalize once before optional saving.       |
   //+------------------------------------------------------------------+
   bool              Finish(bool ruin,bool save,bool final_pass=false,bool prepare_only=false,bool training_done=false)
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
      double   bid=0,ask=0;
      datetime time=LastBarTime;
      if(simulated&&virtual_gap_terminal&&cursor<ArraySize(history))
        {
         MqlRates current=history[cursor];
         bid=current.open;
         ask=bid+current.spread*money.point;
         time=current.time;
         snapshot_msc=(long)time*1000;
        }
      else
        {
         if(simulated&&cursor>0&&cursor<=ArraySize(history))
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
      SAlphaQCMTarget flat;
      ZeroMemory(flat);
      bool settled=ExecuteTarget(flat,bid,ask,time);
      if(!money.Snapshot(bid,ask,time))
        {
         settled=false;
        }
      if(!simulated&&!ReconcileDeals(time))
        {
         Fault("INCOMPLETE_DATA: terminal cash reconciliation failed");
         ReturnFalse;
        }
      settled=settled&&!money.pending&&!money.reconciling&&!money.unknown_request&&money.volume==0;
      if(!settled)
        {
         return(final_pass?FinishExcluded(save,time):false);
        }
      ruin=ruin||money.stopout||controlled_stopping;
      if(training&&open_decision)
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
         if(stop_requested)
           {
            if(!qcm.DiscardDecision())
               ReturnFalse;
            excluded++;
            excluded_cash+=money.equity-start_equity;
            reported_equity=money.equity;
            open_decision=false;
            invalid=false;
            Print("AlphaQCM Stop discarded the final untrained trading interval after accounting for cash");
           }
         else
           if(!Feedback(true,false,ruin,time))
           {
            ReturnFalse;
           }
        }
      if(training&&!training_done&&!qcm.FinishSearch(!stop_requested))
         ReturnFalse;
      if(prepare_only)
        {
         return(true);
        }
      if(training&&!training_done&&stop_requested)
        {
         if(!qcm.CancelTraining())
           {
            ReturnFalse;
           }
        }
      if(training&&!training_done&&!stop_requested&&!Train(true))
        {
         ReturnFalse;
        }
      if(training&&!training_done&&!stop_requested)
        {
         context[AG_CTX_TERMINAL]=1.0f;
         context[AG_CTX_MONEY_CONFIRMED]=1.0f;
         context[AG_CTX_NEXT_READY]=0.0f;
         if(!Forward(AG_FLUSH,AG_FLAG_END,time)||!Train(true))
            ReturnFalse;
         context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
        }
      if(!open_decision)
        {
         service_cash+=money.equity-reported_equity;
         reported_equity=money.equity;
        }
      if(save&&!Save())
        {
         return(false);
        }
      Finalized=true;
      string profit_factor=money.gross_loss>0?DoubleToString(money.gross_profit/money.gross_loss,6):"undefined (no realized losses)";
      if(training)
        {
         PrintFormat("AlphaQCM actual realized cash gross profit %.2f gross loss %.2f cash profit factor %s max equity drawdown %.2f fraction %.6f",
                     money.gross_profit,money.gross_loss,profit_factor,money.max_drawdown,money.max_drawdown_fraction);
        }
      else
        {
         PrintTestFinance(ConfirmFinanceBoundary());
        }
      PrintFormat("AlphaQCM mean/max us preparation %.1f/%I64u forward %.1f/%I64u PPO %.1f/%I64u",
                  prep_calls>0?(double)prep_us/prep_calls:0,max_prep,forward_calls>0?(double)forward_us/forward_calls:0,
                  max_forward,ppo_calls>0?(double)ppo_us/ppo_calls:0,max_ppo);
      PrintFormat("AlphaQCM cash attribution policy %.2f service %.2f excluded %.2f external %.2f",
                  policy_cash,service_cash,excluded_cash,external_cash);
      if(training)
         PrintFormat("AlphaQCM final V=%I64u U=%I64u target_syncs=%I64u prepared_revision=%I64u active_generation=%I64u trade_updates=%I64u",
                     qcm.SearchV(),qcm.SearchU(),qcm.TargetSyncs(),qcm.PreparedRevision(),qcm.ActiveGeneration(),qcm.TradeUpdates());
      PrintFormat("AlphaQCM final balance %.2f equity %.2f updates %I64u excluded %I64u boundaries %I64u mean_us %.1f max_us %I64u",
                  money.balance,money.equity,updates,excluded,boundaries,boundaries>0?(double)total_us/boundaries:0,
                  max_us);
      if(!simulated)
         PrintRunTiming();

      return(true);
     }
   //+------------------------------------------------------------------+
   //+------------------------------------------------------------------+
   //| Tick - reconcile risk before processing any new bar boundary.    |
   //+------------------------------------------------------------------+
   void              Tick()
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
      if(!TerminalInfoInteger(TERMINAL_CONNECTED)&&!MQLInfoInteger(MQL_TESTER))
        {
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
      if(!training)
         ConfirmFinanceBoundary();
      bool ruin=money.stopout||money.equity<=AG_EQUITY_FLOOR*money.reference||(training
                &&money.balance<=AG_MIN_TRADING_BALANCE)||
                (!money.reconciling&&!money.unknown_request&&money.Exhausted(quote.bid,quote.ask)&&
                 money.TradeSessionOpen(quote.time));
      if(ruin||controlled_stopping)
        {
         controlled_stopping=true;
         if(Finish(true,training))
           {
            ExpertRemove();
           }
         return;
        }
      if(money.volume!=0&&(money.sl<=0||money.tp<=0)&&!money.pending)
        {
         SAlphaQCMTarget flat;
         ZeroMemory(flat);
         if(!ExecuteTarget(flat,quote.bid,quote.ask,quote.time,money.reconciling?request_decision:0)||money.pending)
           {
            return;
           }
         blocked_action_boundary=iTime(_Symbol,TimeFrame,0);
         if(training&&open_decision)
           {
            RealMoneySample(true);
            if(!Feedback(true,false,false,quote.time)||!Train())
              {
               Fault("protective terminal training failed");
               return;
              }
           }
        }
      if(money.reconciling&&!money.pending)
        {
         SAlphaQCMTarget remaining;
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
            ulong owner=request_decision;
            if(!ExecuteTarget(remaining,quote.bid,quote.ask,quote.time,owner))
              {
               SAlphaQCMTarget flat;
               ZeroMemory(flat);
               money.intended_volume=0;
               money.intended_sl=0;
               money.intended_tp=0;
               if(!ExecuteTarget(flat,quote.bid,quote.ask,quote.time,owner)||money.pending||money.volume!=0)
                  money.reconciling=true;
              }
           }
        }
      if(training&&invalid&&open_decision&&money.volume==0&&!money.pending&&!money.reconciling)
        {
         blocked_action_boundary=iTime(_Symbol,TimeFrame,0);
         RealMoneySample(true);
         if(!Feedback(true,false,false,quote.time)||!Train())
           {
            Fault("invalid action terminal reconciliation failed");
            return;
           }
        }
      datetime current=iTime(_Symbol,TimeFrame,0);
      if(current>0&&current!=sample_boundary)
        {
         RealMoneySample();
         sample_boundary=current;
        }
      if(current<=0||(current==LastBarTime&&pool_warmup_end==0)||money.pending||money.reconciling)
        {
         return;
        }
      MqlRates bars[];
      int      required=warmup_bars;
      datetime previous=observed_boundary>0?observed_boundary:resume_boundary;
      if(previous>0&&previous<current)
        {
         int missing=iBarShift(_Symbol,TimeFrame,previous,true);
         if(missing>0&&missing<=INT_MAX-window)
           {
            required=MathMax(required,window+missing);
           }
        }
      int copied=CopyRates(_Symbol,TimeFrame,1,required,bars);
      if(copied<window&&required>window)
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
   //| WaitForRecovery - allow an explicit Study recovery save only.    |
   //+------------------------------------------------------------------+
   bool              WaitForRecovery()
     {
      if(!simulated||recovery_saving||!recoverable_fault||!qcm||!qcm.CanRecover())
         return(false);
      stop_requested=false;
      Print("AlphaQCM Study stopped at a recoverable boundary. Use Stop and save to save recovery.");
      Comment("AlphaQCM Study stopped. Use Stop and save to save a recovery checkpoint.");
      ChartRedraw();
      return(true);
     }
   //+------------------------------------------------------------------+
   //| Show the next finalization operation before its timer event.     |
   //+------------------------------------------------------------------+
   void              StudyFinishProgress()
     {
      string stage="Closing the final virtual trajectory";
      if(study_finish_stage==2||study_finish_stage==4)
        {
         stage=StringFormat("Final PPO | batch %d/%d | final updates %I64u | flush %d/%d",
                            study_finish_batch_completed,study_finish_batch_total,
                            updates-study_finish_updates,study_finish_contour,1);
        }
      if(study_finish_stage==3)
        {
         stage=StringFormat("Preparing final flush %d/%d",study_finish_contour+1,1);
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
      if(study_finish_stage==10)
        {
         stage="Applying one confirmed search update";
        }
      Comment("\n\nAlphaQCM Study | finalization\n",stage,
              StringFormat("\nElapsed %.1f s | total updates %I64u",
                           (GetMicrosecondCount()-study_finish_started)/1000000.0,updates));
      ChartRedraw();
     }
   //+------------------------------------------------------------------+
   //| Start a finalizer: end=1, stop=2, opening ruin=3, bar ruin=4.    |
   //+------------------------------------------------------------------+
   bool              BeginStudyFinish(int reason,bool ruin)
     {
      study_finish_reason=reason;
      study_finish_ruin=ruin;
      study_finish_saved=false;
      study_finish_batch_total=0;
      study_finish_batch_completed=0;
      study_finish_contour=0;
      study_finish_updates=updates;
      study_finish_started=GetMicrosecondCount();
      study_finish_stage=!stop_requested&&qcm.HasSearchDebt()?10:1;
      PrintFormat("AlphaQCM Study finalization BEGIN reason=%d",reason);
      StudyFinishProgress();

      return(true);
     }
   //+------------------------------------------------------------------+
   //| Advance one update, flush, or native checkpoint phase per event. |
   //+------------------------------------------------------------------+
   bool              StudyFinishStep()
     {
      if(study_finish_stage==10)
        {
         if(stop_requested||!qcm.HasSearchDebt())
           {
            study_finish_stage=1;
            StudyFinishProgress();
            return(true);
           }
         if(!qcm.DrainSearchStep())
           {
            recoverable_fault=qcm.CanRecover();
            ReturnFalse;
           }
         StudyFinishProgress();
         return(true);
        }
      if(stop_requested&&study_finish_stage>1&&study_finish_stage<5)
        {
         if(!qcm.CancelTraining())
           {
            return(false);
           }
         context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
         study_finish_stage=5;
         study_finish_reason=2;
         StudyFinishProgress();

         return(true);
        }
      int   previous=study_finish_stage;
      ulong begin=GetMicrosecondCount();
      if(study_finish_stage==1)
        {
         if(!Finish(study_finish_ruin,false,false,true))
           {
            return(false);
           }
         study_finish_batch_total=qcm.TrainingTotal();
         study_finish_batch_completed=qcm.TrainingCompleted();
         study_finish_stage=2;
        }
      else
         if(study_finish_stage==2||study_finish_stage==4)
           {
            if(qcm.HasTraining())
              {
               study_finish_batch_total=qcm.TrainingTotal();
               if(!Train())
                 {
                  return(false);
                 }
               study_finish_batch_completed=qcm.HasTraining()?qcm.TrainingCompleted():study_finish_batch_total;
              }
            else
              {
               study_finish_stage=study_finish_contour<1?3:5;
              }
           }
         else
            if(study_finish_stage==3)
              {
               context[AG_CTX_TERMINAL]=1.0f;
               context[AG_CTX_MONEY_CONFIRMED]=1.0f;
               context[AG_CTX_NEXT_READY]=0.0f;
               datetime time=(datetime)(snapshot_msc/1000);
               if(!Forward(AG_FLUSH,AG_FLAG_END,time))
                 {
                  return(false);
                 }
               study_finish_batch_total=qcm.TrainingTotal();
               study_finish_batch_completed=qcm.TrainingCompleted();
               study_finish_contour++;
               study_finish_stage=4;
              }
            else
               if(study_finish_stage==5)
                 {
                  context[AG_CTX_TERMINAL]=context[AG_CTX_MONEY_CONFIRMED]=0.0f;
                  if(!Finish(study_finish_ruin,false,false,false,true))
                    {
                     return(false);
                    }
                  Finalized=false;
                  bool save=stop_requested||study_finish_reason==2||
                            (study_finish_reason==1&&epoch>=Epochs);
                  study_finish_stage=save?6:9;
                 }
               else
                  if(study_finish_stage>=6&&study_finish_stage<=8)
                    {
                     int phase=study_finish_stage-6;
                     PrintFormat("AlphaQCM checkpoint phase %d/3 BEGIN",phase+1);
                     if(!SavePhase(phase))
                       {
                        PrintFormat("AlphaQCM checkpoint phase %d/3 FAILED error=%d",phase+1,GetLastError());

                        save_retry_pending=true;
                        ObjectSetString(0,"AlphaQCMStop",OBJPROP_TEXT,"Retry save");
                        Comment("AlphaQCM checkpoint write failed. Click Retry save to retry.");
                        return(true);
                       }
                     PrintFormat("AlphaQCM checkpoint phase %d/3 END elapsed_ms=%I64u",phase+1,
                                 (GetMicrosecondCount()-begin)/1000);
                     if(phase==2)
                       {
                        study_finish_saved=true;
                       }
                     study_finish_stage++;
                    }
                  else
                     if(study_finish_stage==9)
                       {
                        if(stop_requested&&!study_finish_saved)
                          {
                           study_finish_reason=2;
                           study_finish_stage=6;
                           StudyFinishProgress();

                           return(true);
                          }
                        study_finish_stage=0;
                        if(stop_requested||study_finish_reason==2||(study_finish_reason==1&&epoch>=Epochs))
                          {
                           Finalized=true;
                            PrintRunTiming();
                            if(recovery_saving)
                               Print("AlphaQCM RECOVERY_SAVED; verified recovery checkpoint saved");
                            else
                               Print("AlphaQCM Study finalization complete; verified checkpoint saved");
                           ExpertRemove();

                           return(true);
                          }
                        if(study_finish_reason==1)
                          {
                           epoch++;
                           LastBarTime=0;
                           observed_boundary=0;
                           cursor=0;
                           while(cursor<ArraySize(history)&&history[cursor].time<Start)
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
         PrintFormat("AlphaQCM Study finalization stage %d complete; last operation %I64u ms",previous,
                     (GetMicrosecondCount()-begin)/1000);
        }
      StudyFinishProgress();

      return(true);
     }
   //+------------------------------------------------------------------+
   //| Process a bounded portion of one virtual chronological step.     |
   //+------------------------------------------------------------------+
   bool              StudyStep()
     {
      if(Finalized)
        {
         return(false);
        }
      if(failed)
        {
         if(!recoverable_fault||!qcm.CanRecover())
            ReturnFalse;
         if(!stop_requested)
            return(true);
         if(!qcm.ResumeRecovery())
            ReturnFalse;
         failed=false;
         recoverable_fault=false;
         recovery_saving=true;
         study_finish_stage=0;
        }
      if(study_finish_stage>0)
        {
         return(StudyFinishStep());
        }
      if(stop_requested)
        {
         if(qcm.NeedsPoolWarmup()&&!qcm.AbortPoolWarmup())
            ReturnFalse;
         ClearPoolWarmup();
         // An opening gap may already have been reconciled before a bounded
         // search/PPO operation returned. Finish this bar under its existing
         // protection without another search step or policy action.
         bool partial_bar=cursor<ArraySize(history)&&history[cursor].time+PeriodSeconds(TimeFrame)<=End&&
                           virtual_open_boundary==history[cursor].time&&!cycle_complete&&!virtual_gap_terminal;
         if(partial_bar)
           {
            MqlRates last=history[cursor];
            if(!money.Bar(last))
              {
               Fault("Study stop could not finish the protected virtual bar");
               ReturnFalse;
              }
            cursor++;
             processed_bars++;
            MoneySample();
            snapshot_msc=(long)(last.time+PeriodSeconds(TimeFrame))*1000;
           }
         bool stop_ruin=money.equity<=AG_EQUITY_FLOOR*money.reference;

         return(BeginStudyFinish(2,stop_ruin));
        }
      if(cursor>=ArraySize(history)||history[cursor].time+PeriodSeconds(TimeFrame)>End)
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
      if(!Boundary(history,cursor,bar.time,bar.open,bar.open+bar.spread*money.point))
        {
         ReturnFalse;
        }
      if(!cycle_complete)
        {
         string stage=StringFormat("IQN-QCM | V %I64u | U %I64u",qcm.SearchV(),qcm.SearchU());
         if(pool_warmup_end>0)
           {
            stage=pool_warmup_staged?"Pool staged | awaiting publication":"Preparing pool context";
           }
         else
            if(qcm.HasTraining())
               stage="Applying a training update";
         Comment(StringFormat("AlphaQCM Study | epoch %d/%d | %s\n%s\nActive pool %d | prepared pool %d\nQ/IQN updates %I64u | trade PPO updates %I64u",
                              epoch,Epochs,TimeToString(bar.time),stage,qcm.ActiveCount(),qcm.PreparedCount(),
                              qcm.SearchU(),qcm.TradeUpdates()));

         return(true);
        }
      if(!money.Bar(bar))
        {
         ReturnFalse;
        }
      cursor++;
      processed_bars++;
      MoneySample();
      bool ruin=money.equity<=AG_EQUITY_FLOOR*money.reference||money.Exhausted(bar.close,bar.close+bar.spread*money.point);
      if(ruin)
        {
         return(BeginStudyFinish(4,true));
        }
      string pools=StringFormat("Active %d gen %I64u | prepared %d rev %I64u | V %I64u U %I64u | trade PPO %I64u",
                                qcm.ActiveCount(),qcm.ActiveGeneration(),qcm.PreparedCount(),qcm.PreparedRevision(),
                                qcm.SearchV(),qcm.SearchU(),qcm.TradeUpdates());
      Comment(pools,"\n",StringFormat("AlphaQCM Study | epoch %d/%d | %s\nbalance %.2f equity %.2f | virtual trades %I64u | updates %I64u\nUse the button to stop and save",epoch,Epochs,TimeToString(bar.time),money.balance,money.equity,money.trades,updates));

      return(true);
     }
#endif
  };
#endif
//+------------------------------------------------------------------+
