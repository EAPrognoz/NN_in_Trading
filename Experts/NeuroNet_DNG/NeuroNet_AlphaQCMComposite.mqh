//+------------------------------------------------------------------+
//|                                   NeuroNet_AlphaQCMComposite.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_COMPOSITE_MQH
#define NEURONET_ALPHAQCM_COMPOSITE_MQH

#define AQCM_FILE_VERSION 1
#define AQCM_NETWORK_COUNT 4

//+------------------------------------------------------------------+
//| CNeuronAlphaQCMOCL - Own the search and trading lifecycles.        |
//+------------------------------------------------------------------+
class CNeuronAlphaQCMOCL : public CNeuronAlphaGenOCL
  {
protected:
   CNet                    m_qOnline;
   CNet                    m_qTarget;
   CNet                    m_zOnline;
   CNet                    m_zTarget;
   CAlphaQCMReplay         m_replay;
   CAlphaQCMGenerator      m_generator;
   CAlphaQCMObjective      m_objective;
   AQCReplayBatch          m_batch;
   CBufferFloat            m_reward;
   CBufferFloat            m_preflight;
   float                   m_qcmConfig[];
   int                     m_loadMode;
   ulong                   m_updates;
   ulong                   m_syncs;
   ulong                   m_candidate;
   ulong                   m_lastEvent;
   ulong                   m_lastDecision;
   ulong                   m_preparedRevision;
   ulong                   m_activeGeneration;
   ulong                   m_preparedWindow;
   float                   m_preparedIC;
   bool                    m_preparedUsable;
   ulong                   m_observationWatermark;
   string                  m_historySymbol;
   string                  m_historyServer;
   ENUM_TIMEFRAMES         m_historyTimeframe;
   bool                    m_qReady;
   bool                    m_updateInProgress;
   bool                    m_boundedLearner;
   bool                    m_bootstrap;
   bool                    m_recoverable;
   ulong                   m_searchAttempts;
   ulong                   m_searchEpisodes;

   bool     TradeWorkspace(void);
   bool     SearchWorkspace(const bool create);
   bool     BuildNetwork(CNet *net,const bool iqn,const bool configure=true);
   bool     NetworkSave(CNet *net,const int handle);
   bool     NetworkLoad(CNet *net,const int handle,const bool restore,const bool iqn);
   bool     Search(CBufferFloat *market,const ulong boundary,const bool defer_publication=false);
   bool     PreparePublication(CBufferFloat *market,const ulong boundary);
   bool     TradingAction(CNeuronBaseOCL *source,const ulong boundary);
   bool     CheckTradingAction(void);
   bool     PublishPool(CNeuronBaseOCL *source,const int flags,const ulong boundary);
   bool     ObservePool(CNeuronBaseOCL *source,const int flags,const ulong boundary);
   bool     Learn(void);
   bool     LearnBatch(AQCReplayBatch &batch);
   bool     Broken(const int status,const bool recoverable=false);
   int      ReplayFailure(void) const;

   virtual bool feedForward(CNeuronBaseOCL *source) override;
   virtual bool feedForward(CNeuronBaseOCL *source,CBufferFloat *context) override;

public:
                     CNeuronAlphaQCMOCL(void);
   virtual bool      Init(uint outputs,uint index,COpenCLMy *context,uint count,ENUM_OPTIMIZATION method,uint batch) override;
   virtual int       Type(void) const override { return(defNeuronAlphaQCMOCL); }
   virtual bool      Save(const int handle) override;
   virtual bool      Load(const int handle) override;
   bool              Configure(const float &config[],const ulong seed);
   bool              AfterBuild(const bool created);
   bool              FinishSearch(const bool drain=true);
   bool              BootstrapPending(void) const { return(m_bootstrap&&m_bSearchBudgetPending); }
   bool              ContinueBootstrap(void);
   void              SetBoundedLearner(const bool bounded) { m_boundedLearner=bounded; }
   bool              HasSearchDebt(void);
   bool              DrainSearchStep(void);
   bool              CanRecover(void) const;
   bool              ResumeRecovery(void);
   ulong             SearchAttempts(void) const { return(m_searchAttempts); }
   ulong             SearchEpisodes(void) const { return(m_searchEpisodes); }
   bool              SafeToSave(void) const;
   bool              CausalIdentity(const string symbol,const ENUM_TIMEFRAMES timeframe);
   ulong             LastServiceEvent(void) const { return(m_lastEvent); }
   ulong             LastServiceDecision(void) const { return(m_lastDecision); }
   bool              IssueEvent(ulong &id)
     {
      if(m_lastEvent==ULONG_MAX)
         ReturnFalse;
      id=++m_lastEvent;
      return(true);
     }
   bool              IssueDecision(ulong &id)
     {
      if(m_lastDecision==ULONG_MAX)
         ReturnFalse;
      id=++m_lastDecision;
      return(true);
     }
   ulong             SearchV(void) { return(m_replay.Verified()); }
   ulong             SearchU(void) const { return(m_updates); }
   ulong             TargetSyncs(void) const { return(m_syncs); }
   ulong             ActiveGeneration(void) const { return(m_activeGeneration); }
   ulong             PreparedRevision(void) const { return(m_preparedRevision); }
   ulong             CausalWatermark(void) const { return(m_observationWatermark); }
   bool              SetPoolWarmupHistory(const float &history[],const int count)
     {
      return(m_cPool.SetWarmupHistory(history,count));
     }
  };

//+------------------------------------------------------------------+
//| Initialize without allocating the legacy token PPO.               |
//+------------------------------------------------------------------+
CNeuronAlphaQCMOCL::CNeuronAlphaQCMOCL(void)
   : m_loadMode(AlphaQCMLoadMode),m_updates(0),m_syncs(0),m_candidate(0),m_lastEvent(0),m_lastDecision(0),
     m_preparedRevision(0),m_activeGeneration(0),m_preparedWindow(0),m_preparedIC(0),m_preparedUsable(false),
     m_observationWatermark(0),m_historyTimeframe(PERIOD_CURRENT),
     m_qReady(false),m_updateInProgress(false),m_boundedLearner(false),m_bootstrap(false),m_recoverable(false),
     m_searchAttempts(0),m_searchEpisodes(0)
  {
  }

bool CNeuronAlphaQCMOCL::Init(uint outputs,uint index,COpenCLMy *context,uint count,
                             ENUM_OPTIMIZATION method,uint batch)
  {
   if(!context||method!=ADAM||m_loadMode!=AQCM_TRAIN)
      ReturnFalse;

   return(CNeuronAlphaGenOCL::Init(outputs,index,context,count,method,batch));
  }

//+------------------------------------------------------------------+
//| Retain only the working and training buffers used by trading.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::TradeWorkspace(void)
  {
   int capacity=(int)m_fConfig[AG_CFG_TRADE_ROLLOUT];
   int context=(int)m_fConfig[AG_CFG_CONTEXT];
   int hidden=(int)m_fConfig[AG_CFG_TRADE_HIDDEN];

   if(context<1||hidden<1||
      !Allocate(m_cAccount,AG_ACCOUNT)||!Allocate(m_cWorkMetadata,(long)context*8)||
      !Allocate(m_cWorkAnchor,hidden*2)||!Allocate(m_cStageAnchor,hidden*2)||!Allocate(m_cEvicted,1))
      ReturnFalse;

   if(m_loadMode!=AQCM_TRAIN)
      return(true);

   if(capacity<1||
      !Allocate(m_cTradeContextMetadata,(long)capacity*context*8)||
      !Allocate(m_cTradeIdentifiers,(long)capacity*8)||
      !Allocate(m_cTradeContexts,(long)capacity*context)||
      !Allocate(m_cTradeAnchors,(long)capacity*hidden*2)||
      !Allocate(m_cTradeAccounts,(long)capacity*AG_ACCOUNT)||
      !Allocate(m_cTradeSamples,(long)capacity*AG_ACTIONS)||
      !Allocate(m_cTradeActions,(long)capacity*AG_ACTIONS)||
      !Allocate(m_cTradeMetadata,(long)capacity*7)||
      !Allocate(m_cTradeGAE,(long)capacity*AG_GAE_FIELDS)||
      !Allocate(m_cReplayContext,context)||!Allocate(m_cReplayAnchor,hidden*2)||
      !Allocate(m_cReplayAccount,AG_ACCOUNT)||
      ArrayResize(m_uTradeDecisions,capacity)!=capacity||
      ArrayResize(m_fTradeRewards,capacity)!=capacity)
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Configure the shared pool, trading objects and search networks.    |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::Configure(const float &config[],const ulong seed)
  {
   if(m_bConfigured||m_loadMode!=AQCM_TRAIN||ArraySize(config)!=AQCM_CFG_SIZE||
      ArrayCopy(m_fConfig,config,0,0,AG_CFG_SIZE)!=AG_CFG_SIZE||
      ArrayCopy(m_qcmConfig,config,0,AG_CFG_SIZE,AQCM_CFG_SIZE-AG_CFG_SIZE)!=AQCM_CFG_SIZE-AG_CFG_SIZE)
      ReturnFalse;

   m_uSeed=seed;
   uint mixed=(uint)(seed^(seed>>32));
   m_iFinanceRandom=mixed^0xc2b2ae35;
   if(m_iFinanceRandom==0)
      m_iFinanceRandom=1;

   if(!m_cEvaluator.Configure(OpenCL,m_fConfig)||!m_cPool.Configure(OpenCL,m_fConfig)||
      !m_cTrading.Configure(OpenCL,m_fConfig,mixed+0x85ebca6b)||
      !TradeWorkspace()||!SearchWorkspace(true))
      ReturnFalse;

   m_bConfigured=true;

   return(true);
  }

bool CNeuronAlphaQCMOCL::BuildNetwork(CNet *net,const bool iqn,const bool configure)
  {
   CArrayObj descriptions;
   CLayerDescription *inputDescription=new CLayerDescription();
   CLayerDescription *encoder=new CLayerDescription();

   if(!inputDescription||!encoder)
     {
      DeleteObj(inputDescription)
      DeleteObj(encoder)
      ReturnFalse;
     }

   inputDescription.type=defNeuronBaseOCL;
   inputDescription.count=(int)m_fConfig[AG_CFG_MAX_TOKENS];
   inputDescription.activation=None;
   inputDescription.optimization=ADAM;
   inputDescription.batch=1;
   encoder.type=defNeuronAlphaQCMEncoderOCL;
   encoder.count=AG_VOCABULARY;
   encoder.activation=None;
   encoder.optimization=ADAM;
   encoder.batch=1;
   if(!descriptions.Add(inputDescription))
     {
      DeleteObj(inputDescription)
      DeleteObj(encoder)
      ReturnFalse;
     }
   if(!descriptions.Add(encoder))
     {
      DeleteObj(encoder)
      ReturnFalse;
     }

   if(!net.UseOpenCLRuntime(OpenCL)||!net.Create(GetPointer(descriptions)))
      ReturnFalse;

   CNeuronAlphaQCMEncoderOCL *layer=(CNeuronAlphaQCMEncoderOCL *)net.Layer(1);
   if(!layer||(configure&&!layer.Configure(OpenCL,iqn,(int)m_fConfig[AG_CFG_MAX_TOKENS],AG_VOCABULARY,
                               (int)m_fConfig[AG_CFG_EMBEDDING],(int)m_fConfig[AG_CFG_SEARCH_HIDDEN],
                               (int)m_qcmConfig[AQCM_CFG_FREQUENCIES-AG_CFG_SIZE])))
      ReturnFalse;

   return(true);
  }

bool CNeuronAlphaQCMOCL::SearchWorkspace(const bool create)
  {
   if(m_loadMode!=AQCM_TRAIN)
      return(true);

   if(!m_replay.Configure(OpenCL,(int)m_qcmConfig[AQCM_CFG_REPLAY-AG_CFG_SIZE],
                          (int)m_fConfig[AG_CFG_MAX_TOKENS],AG_VOCABULARY,
                          (int)m_qcmConfig[AQCM_CFG_BATCH-AG_CFG_SIZE],
                          (int)m_qcmConfig[AQCM_CFG_QUANTILES-AG_CFG_SIZE],
                          (int)m_qcmConfig[AQCM_CFG_TARGET_QUANTILES-AG_CFG_SIZE],
                          m_qcmConfig[AQCM_CFG_PER_EPSILON-AG_CFG_SIZE],
                          m_qcmConfig[AQCM_CFG_PER_BETA-AG_CFG_SIZE],
                          (ulong)m_qcmConfig[AQCM_CFG_PER_BETA_UPDATES-AG_CFG_SIZE],
                          m_qcmConfig[AQCM_CFG_PER_ALPHA-AG_CFG_SIZE])||
      !Allocate(m_preflight,3)||!Allocate(m_reward,1)||
      !Allocate(m_cSearchMarket,(long)m_fConfig[AG_CFG_W]*5+AG_ACCOUNT))
      ReturnFalse;

   if(create&&(!BuildNetwork(GetPointer(m_qOnline),false)||!BuildNetwork(GetPointer(m_qTarget),false)||
               !BuildNetwork(GetPointer(m_zOnline),true)||!BuildNetwork(GetPointer(m_zTarget),true)))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Initialize device-only state once after selective Build.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::AfterBuild(const bool created)
  {
   if(!m_bConfigured||m_qReady||m_loadMode==AQCM_VERIFY)
      ReturnFalse;

   CAlphaQCMOpenCL *runtime=(CAlphaQCMOpenCL *)OpenCL;
   if(!runtime.ValidateTradingLocalMemory((int)m_fConfig[AG_CFG_TRADE_HIDDEN]))
      return(Broken(AQCM_RESOURCE_UNSUPPORTED));
   int candidates=m_loadMode==AQCM_TRAIN?(int)m_qcmConfig[AQCM_CFG_CANDIDATES-AG_CFG_SIZE]:0;
   if(!m_cPool.QCMValidateWorkspace(candidates))
      return(Broken(m_cPool.QCMStatus()));
   if(m_loadMode==AQCM_TRAIN)
     {
      if(!m_generator.Configure(OpenCL,(int)m_qcmConfig[AQCM_CFG_CANDIDATES-AG_CFG_SIZE],
                                (int)m_fConfig[AG_CFG_MAX_TOKENS],(int)m_fConfig[AG_CFG_EMBEDDING],
                                (int)m_fConfig[AG_CFG_SEARCH_HIDDEN],(int)m_qcmConfig[AQCM_CFG_POINTS-AG_CFG_SIZE],
                                m_qcmConfig[AQCM_CFG_EPSILON-AG_CFG_SIZE],m_qcmConfig[AQCM_CFG_BONUS-AG_CFG_SIZE],
                                (int)m_qcmConfig[AQCM_CFG_FREQUENCIES-AG_CFG_SIZE]))
         ReturnFalse;

      if(created)
        {
         CNeuronAlphaQCMEncoderOCL *q=(CNeuronAlphaQCMEncoderOCL *)m_qOnline.Layer(1);
         CNeuronAlphaQCMEncoderOCL *z=(CNeuronAlphaQCMEncoderOCL *)m_zOnline.Layer(1);
         if(!q||!z||!m_replay.Initialize()||!q.InitializeWeights(m_uSeed,0)||!z.InitializeWeights(m_uSeed,1)||
            !m_qTarget.WeightsUpdate(GetPointer(m_qOnline),1.0f)||!m_zTarget.WeightsUpdate(GetPointer(m_zOnline),1.0f))
            ReturnFalse;
        }
      else
         if(!m_replay.Restore())
            ReturnFalse;
      if(!m_qTarget.TrainMode(false)||!m_zTarget.TrainMode(false)||
         !m_qOnline.TrainMode(true)||!m_zOnline.TrainMode(true))
         ReturnFalse;
     }

   m_qReady=true;

   return(true);
  }

bool CNeuronAlphaQCMOCL::Broken(const int status,const bool recoverable)
  {
   m_iStatus=status;
   m_recoverable=recoverable&&!m_updateInProgress&&m_bConfigured;
   m_bCorrupted=true;

   ReturnFalseEx(StringFormat("AlphaQCM status=%d, V=%I64u, U=%I64u",status,SearchV(),SearchU()));
  }

int CNeuronAlphaQCMOCL::ReplayFailure(void) const
  {
   switch(m_replay.LastStatus())
     {
      case AQC_REPLAY_RESOURCE:
         return(AQCM_RESOURCE_UNSUPPORTED);
      case AQC_REPLAY_NUMERIC:
         return(AQCM_NUMERIC_ERROR);
      case AQC_REPLAY_INCONSISTENT:
         return(AQCM_UPDATE_INCONSISTENT);
      case AQC_REPLAY_OPENCL_ERROR:
         return(AQCM_OPENCL_ERROR);
     }
   return(AG_TECHNICAL);
  }

bool CNeuronAlphaQCMOCL::SafeToSave(void) const
  {
   return(m_loadMode==AQCM_TRAIN&&m_qReady&&CNeuronAlphaGenOCL::SafeToSave()&&
          !m_updateInProgress&&m_iPendingTrade<0);
  }

bool CNeuronAlphaQCMOCL::CausalIdentity(const string symbol,const ENUM_TIMEFRAMES timeframe)
  {
   if(symbol==""||timeframe==PERIOD_CURRENT)
      ReturnFalse;

   string server=AccountInfoString(ACCOUNT_SERVER);
   if(m_historySymbol!=symbol||m_historyTimeframe!=timeframe||m_historyServer!=server)
      m_observationWatermark=0;

   m_uObserved=m_observationWatermark;
   m_historyServer=server;

   m_historySymbol=symbol;
   m_historyTimeframe=timeframe;
   return(true);
  }

//+------------------------------------------------------------------+
//| Keep inherited PPO calls on the trading-only branch.              |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::feedForward(CNeuronBaseOCL *source,CBufferFloat *context)
  {
   if(context)
      m_cService=context;

   return(feedForward(source));
  }

bool CNeuronAlphaQCMOCL::feedForward(CNeuronBaseOCL *source)
  {
   if(!m_qReady||!source||!m_cService||m_cService.Total()<AG_CTX_SIZE||
      source.Neurons()!=(uint)(5*m_fConfig[AG_CFG_W]+AG_ACCOUNT)||m_bFrame||m_bCorrupted)
      ReturnFalse;

   m_iPhase=(int)m_cService.At(AG_CTX_PHASE);
   int flags=(int)m_cService.At(AG_CTX_FLAGS);
   ulong boundary=Identifier(m_cService,AG_CTX_BOUNDARY);
   if(m_iPhase==AG_BACKWARD)
      return(m_loadMode==AQCM_TRAIN&&m_iTrainingKind==2&&Replay());

   m_iStatus=AG_OK;
   if(!ZeroOutput())
      ReturnFalse;

   switch(m_iPhase)
     {
      case AG_OBSERVE:
         return(ObservePool(source,flags,boundary));
      case AG_ACT:
      case AG_INFER:
         return(TradingAction(source,boundary));
      case AG_EXIT_PENDING:
      case AG_FLUSH:
         return(Flush());
      case AG_PREPARE:
         if(m_loadMode!=AQCM_TRAIN||!bTrain)
           {
            m_iStatus=AG_WAIT;
            return(true);
           }
         if(m_bPoolWarmup&&(flags&AG_FLAG_WARMUP)!=0)
            return(PublishPool(source,flags,boundary));
         if((flags&AQCM_FLAG_PUBLICATION_ONLY)!=0)
            return(PreparePublication(source.getOutput(),boundary));
         return(Search(source.getOutput(),boundary));
      case AG_FINALIZED:
         m_iStatus=AG_FINALIZED;
         return(true);
     }

   ReturnFalse;
  }

//+------------------------------------------------------------------+
//| Observe the same working context used by donor Act and feedback.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::ObservePool(CNeuronBaseOCL *source,const int flags,const ulong boundary)
  {
   m_bNextReady=false;
   CBufferFloat *market=source.getOutput();

   if(!Slice(market,(int)m_fConfig[AG_CFG_W]*5,GetPointer(m_cAccount),0,AG_ACCOUNT)||!PoolControl())
      ReturnFalse;

   bool first=((flags&AG_FLAG_RESET_WORK)!=0&&(flags&AG_FLAG_WARMUP)!=0);
   if(m_cPool.Control().At(AG_POOL_ACTIVE_READY)!=1&&!(first&&ActiveCount()>0))
     {
      m_iStatus=AG_WAIT;
      m_bWorkReady=false;
      return(true);
     }

   bool warmup=((flags&AG_FLAG_WARMUP)!=0);
   if(warmup)
     {
      if(first&&(!m_cPool.ResetContext()||!m_cWorkAnchor.BufferInit(m_cWorkAnchor.Total(),0)||
                 !m_cWorkAnchor.BufferWrite()))
         ReturnFalse;
      if(!(first?m_cPool.QCMRewarmActive(market):m_cPool.QCMWarmup(market,false)))
         return(Broken(m_cPool.QCMStatus()));
     }
   else
      if(boundary!=m_uObserved)
        {
         if(m_bWorkReady&&(!Slice(m_cPool.Context(),0,GetPointer(m_cEvicted),0,1)||
            !m_cTrading.Encode(GetPointer(m_cEvicted),1,GetPointer(m_cWorkAnchor))||
            !m_cTrading.CommitAnchor()||
            !Slice(m_cTrading.Anchor(),0,GetPointer(m_cWorkAnchor),0,m_cWorkAnchor.Total())))
            ReturnFalse;
         if(!m_cPool.QCMObserve(market))
            return(Broken(m_cPool.QCMStatus()));
        }

   if(!PoolControl())
      ReturnFalse;
   if(m_cPool.Control().At(AG_POOL_SIGNAL_READY)!=1)
     {
      m_bWorkReady=false;
      m_iStatus=m_cPool.Control().At(AG_POOL_STATUS)==AG_WAIT?AG_WAIT:AG_INVALID;
      return(true);
     }

   if(!Metadata(warmup)||!m_cTrading.Encode(m_cPool.Context(),(int)m_fConfig[AG_CFG_CONTEXT],
                                           GetPointer(m_cWorkAnchor)))
      ReturnFalse;

   m_bWorkReady=true;
   m_uObserved=boundary;
   m_observationWatermark=boundary;
   if(m_iPendingTrade>=0&&(flags&AG_FLAG_VALUE)!=0)
     {
      if(!m_cTrading.Heads(GetPointer(m_cAccount),1))
         ReturnFalse;
      m_bNextReady=true;
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Warm the pinned pool before making a new revision active.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::PublishPool(CNeuronBaseOCL *source,const int flags,const ulong boundary)
  {
   if(m_iPendingTrade>=0||HasTraining()||!m_bPublicationPermit)
      ReturnFalse;
   bool finish=((flags&AG_FLAG_END)!=0);
   if(!finish)
     {
      if(!m_cPool.QCMPrepareAppend(source.getOutput(),(flags&AG_FLAG_RESET_WORK)!=0))
         return(Broken(m_cPool.QCMStatus()));
      if(!PoolControl())
         return(Broken(AQCM_OPENCL_ERROR));
     }

   if(m_cPool.Control().At(AG_POOL_SIGNAL_READY)!=1)
     {
      if(!m_cPool.AbortPrepare())
         return(Broken(AQCM_OPENCL_ERROR));
      m_bPoolWarmup=false;
      m_iStatus=AG_WAIT;
      return(true);
     }

   if(!finish)
     {
      if(m_activeGeneration==ULONG_MAX||!m_cStageAnchor.BufferInit(m_cStageAnchor.Total(),0)||
         !m_cStageAnchor.BufferWrite()||
         !m_cTrading.Encode(m_cPool.StagedContext(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cStageAnchor)))
         return(Broken(AQCM_OPENCL_ERROR));
      m_iStatus=AG_PREPARING_POOL;
      return(true);
     }

   if(!m_cPool.Commit()||!PoolControl())
      return(Broken(AQCM_UPDATE_INCONSISTENT));

   m_bPoolWarmup=false;
   if(m_cPool.Control().At(AG_POOL_PUBLISHED)!=1)
     {
      m_iStatus=AG_WAIT;
      return(true);
     }
   if(!Slice(GetPointer(m_cStageAnchor),0,GetPointer(m_cWorkAnchor),0,m_cWorkAnchor.Total())||!Metadata(true))
      return(Broken(AQCM_UPDATE_INCONSISTENT));

   m_activeGeneration++;
   m_bPublicationPermit=false;
   m_bWorkReady=true;
   m_uObserved=boundary;
   m_uActionBoundary=boundary;
   m_observationWatermark=boundary;
   m_iStatus=AG_PUBLISHED;

   return(true);
  }

//+------------------------------------------------------------------+
//| CheckTradingAction - Distinguish dispatch and numeric failure.   |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::CheckTradingAction(void)
  {
   CBufferFloat *result=m_cEvaluator.NumericResult();
   if(!result||!m_cEvaluator.NumericBatch(m_cTrading.Action(),m_cTrading.Metadata(),NULL,NULL,true))
      return(Broken(AQCM_OPENCL_ERROR));
   setBuffer(def_k_AGNumericFinish,0,result.GetIndex())
   uint offset[]={0};
   uint global[]={1};
   kernelExecute(def_k_AGNumericFinish,offset,global)
   float status[1];
   if(!OpenCL.BufferRead(result.GetIndex(),status,0,0,1))
      return(Broken(AQCM_OPENCL_ERROR));
   if(status[0]!=AG_OK)
      return(Broken(AQCM_NUMERIC_ERROR,true));

   return(true);
  }

//+------------------------------------------------------------------+
//| TradingAction - Use donor policy and retain a typed result.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::TradingAction(CNeuronBaseOCL *source,const ulong boundary)
  {
   if(!m_bWorkReady||boundary!=m_uObserved||m_bSearchBudgetPending||m_bPoolWarmup||
      HasTraining()||m_iPendingTrade>=0||boundary==m_uActionBoundary)
     {
      m_iStatus=AG_WAIT;

      return(true);
     }

   if(!Slice(source.getOutput(),(int)m_fConfig[AG_CFG_W]*5,GetPointer(m_cAccount),0,AG_ACCOUNT))
      return(Broken(AQCM_OPENCL_ERROR));

   if(!m_cTrading.Forward(m_cPool.Context(),(int)m_fConfig[AG_CFG_CONTEXT],GetPointer(m_cWorkAnchor),
                          GetPointer(m_cAccount),bTrain&&m_iPhase==AG_ACT?0:1))
      return(Broken(AQCM_OPENCL_ERROR));

   if(!CheckTradingAction())
     {
      if(m_iStatus==AG_OK)
         return(Broken(AQCM_OPENCL_ERROR));
      ReturnFalse;
     }

   if(!Slice(m_cTrading.Action(),0,Output,0,AG_ACTIONS))
      return(Broken(AQCM_OPENCL_ERROR));

   m_uActionBoundary=boundary;

   if(bTrain&&m_iPhase==AG_ACT)
     {
      int row=m_iTradeCount;
      int context=(int)m_fConfig[AG_CFG_CONTEXT];
      int hidden=(int)m_fConfig[AG_CFG_TRADE_HIDDEN];

      if(row>=(int)m_fConfig[AG_CFG_TRADE_ROLLOUT])
         ReturnFalse;

      int version_offset=10+((int)m_fConfig[AG_CFG_POOL]+1)*(13+(int)m_fConfig[AG_CFG_MAX_TOKENS])+5;

      if(!Slice(GetPointer(m_cWorkMetadata),0,GetPointer(m_cTradeContextMetadata),row*context*8,context*8)||
         !Slice(m_cService,AG_CTX_DECISION,GetPointer(m_cTradeIdentifiers),row*8,4)||
         !Slice(m_cPool.Banks(),version_offset,GetPointer(m_cTradeIdentifiers),row*8+4,4)||
         !Slice(m_cPool.Context(),0,GetPointer(m_cTradeContexts),row*context,context)||
         !Slice(GetPointer(m_cWorkAnchor),0,GetPointer(m_cTradeAnchors),row*hidden*2,hidden*2)||
         !Slice(GetPointer(m_cAccount),0,GetPointer(m_cTradeAccounts),row*AG_ACCOUNT,AG_ACCOUNT)||
         !Slice(m_cTrading.Sample(),0,GetPointer(m_cTradeSamples),row*AG_ACTIONS,AG_ACTIONS)||
         !Slice(m_cTrading.Action(),0,GetPointer(m_cTradeActions),row*AG_ACTIONS,AG_ACTIONS)||
         !Slice(m_cTrading.Metadata(),0,GetPointer(m_cTradeMetadata),row*7,7)||
         !Slice(m_cTrading.Metadata(),0,GetPointer(m_cTradeMetadata),row*7+3,1)||
         !Slice(m_cTrading.Metadata(),2,GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_VALUE,1)||
         !Scalar(GetPointer(m_cTradeGAE),row*AG_GAE_FIELDS+AG_GAE_READY,0))
         return(Broken(AQCM_UPDATE_INCONSISTENT));

      m_bNextReady=false;
      m_iPendingTrade=row;
      m_uDecision=Identifier(m_cService,AG_CTX_DECISION);
      m_uTradeDecisions[row]=m_uDecision;
      m_bPublicationPermit=false;
     }
   m_iStatus=AG_ACTION_READY;

   return(true);
  }
//+------------------------------------------------------------------+
//| Generate and commit one bounded transaction per driver step.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::Search(CBufferFloat *market,const ulong boundary,const bool defer_publication)
  {
   if(m_iPendingTrade>=0||HasTraining()||m_bPoolWarmup)
     {
      m_iStatus=AG_WAIT;
      return(true);
     }
   if(m_boundedLearner&&HasSearchDebt())
     {
      if(!DrainSearchStep())
         ReturnFalse;
      m_iStatus=AG_WORKING;
      return(true);
     }
   if(!m_boundedLearner&&!Learn())
      ReturnFalse;

   if(!m_bSearchBudgetPending)
     {
      if(boundary==m_uBoundary)
         return(true);
      m_uBoundary=boundary;
      m_iTrials=0;
      m_iBatchGenerated=0;
      m_iBatchProcessed=0;
      bool active=m_cPool.Control().At(AG_POOL_ACTIVE_READY)==1;
      m_bootstrap=!active&&!m_bWarmBudget;
      m_iBudget=(int)m_fConfig[m_bootstrap?AG_CFG_WARMUP_TRIALS:AG_CFG_SEARCH_TRIALS];

      if(!m_cPool.QCMRefresh(market,false,boundary))
         return(Broken(m_cPool.QCMStatus()));
      int status=(int)m_cPool.Control().At(AG_POOL_STATUS);
      if(status==AG_SKIP||status==AG_WAIT)
        {
         m_iStatus=AG_OK;
         return(true);
        }
      if(status!=AG_OK)
         return(Broken(AG_TECHNICAL));
      if(m_cPool.QCMChanged()||m_preparedWindow!=boundary)
        {
         if(m_preparedRevision==ULONG_MAX)
            return(Broken(AG_TECHNICAL));
         m_preparedRevision++;
         m_preparedWindow=boundary;
        }
      m_preparedIC=m_cPool.QCMFitIC();
      m_preparedUsable=m_cPool.QCMAggregateUsable();
      if(m_bootstrap&&!Slice(market,0,GetPointer(m_cSearchMarket),0,m_cSearchMarket.Total()))
         return(Broken(AQCM_OPENCL_ERROR));
      m_bSearchBudgetPending=true;
     }

   if(m_iTrials==m_iBudget)
     {
      m_bSearchBudgetPending=false;
      m_bootstrap=false;
      return(defer_publication?true:PreparePublication(market,boundary));
     }
   if(m_bootstrap)
      market=GetPointer(m_cSearchMarket);

   int count=MathMin(m_iBudget-m_iTrials,(int)m_qcmConfig[AQCM_CFG_CANDIDATES-AG_CFG_SIZE]);
   if(count<1||m_candidate>ULONG_MAX-(ulong)count||m_preparedRevision>ULONG_MAX-(ulong)count)
      return(Broken(AG_TECHNICAL));

   SAQCInferenceView qWeights;
   SAQCInferenceView zWeights;
   CNeuronAlphaQCMEncoderOCL *q=(CNeuronAlphaQCMEncoderOCL *)m_qOnline.Layer(1);
   CNeuronAlphaQCMEncoderOCL *z=(CNeuronAlphaQCMEncoderOCL *)m_zOnline.Layer(1);
   if(!q||!z||!q.ExportInferenceViews(qWeights)||!z.ExportInferenceViews(zWeights))
      return(Broken(AG_TECHNICAL));

   // The first newly issued ID also uniquely identifies this transaction.
   ulong first=m_candidate;
   m_candidate+=(ulong)count;
   m_searchAttempts+=(ulong)count;
   m_iTrials+=count;
   m_bWarmBudget=true;
   m_iBatchGenerated=count;
   m_iBatchProcessed=0;
   bool warmup=SearchV()<(ulong)m_qcmConfig[AQCM_CFG_WARMUP-AG_CFG_SIZE];
   if(!m_generator.Generate(m_cEvaluator.Config().GetIndex(),qWeights,zWeights,m_uSeed,first,count,warmup))
      return(Broken(m_generator.Status()));

   CBufferFloat *packed=m_generator.Results();
   if(!packed||!m_cPool.QCMEvaluateBatch(market,packed,count))
      return(Broken(m_cPool.QCMStatus()));
   int stride=1+(int)m_fConfig[AG_CFG_MAX_TOKENS]*(4+AG_VOCABULARY);
   float size[1];
   for(int row=0;row<count;row++)
     {
      ulong before=m_preparedRevision;
      if(!m_cPool.QCMCommitCandidate(market,packed,count,row))
         return(Broken(m_cPool.QCMStatus()));
      int outcome=(int)m_cPool.Control().At(AG_POOL_STATUS);
      if(outcome==AG_TECHNICAL)
         return(Broken(AG_TECHNICAL));
      if(m_cPool.QCMChanged())
        {
         m_preparedRevision++;
         m_preparedIC=m_cPool.QCMFitIC();
         m_preparedUsable=m_cPool.QCMAggregateUsable();
        }
      m_iBatchProcessed++;

      if(m_cPool.Control().At(AG_POOL_REWARD_READY)==1)
        {
         if(!OpenCL.BufferRead(packed.GetIndex(),size,(uint)(row*stride*sizeof(float)),0,1)||
            !MathIsValidNumber(size[0])||size[0]!=(float)(int)size[0]||
            !Slice(m_cPool.Control(),AG_POOL_REWARD,GetPointer(m_reward),0,1)||
            !m_replay.PublishEpisode(packed.GetIndex(),row,(int)size[0],m_reward.GetIndex(),0,first+(ulong)row,
                                     boundary,before,m_preparedRevision,m_activeGeneration))
            return(Broken(AQCM_UPDATE_INCONSISTENT));
         m_searchEpisodes++;
        }
     }

   if(!m_boundedLearner&&!Learn())
      ReturnFalse;
   if(m_boundedLearner||m_iTrials<m_iBudget||HasSearchDebt())
     {
      m_iStatus=AG_WORKING;
      return(true);
     }
   m_bSearchBudgetPending=false;
   m_bootstrap=false;

   return(defer_publication?true:PreparePublication(market,boundary));
  }

//+------------------------------------------------------------------+
//| ContinueBootstrap - Finish work on the pinned market window.      |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::ContinueBootstrap(void)
  {
   if(!BootstrapPending()||m_bCorrupted)
      ReturnFalse;
   return(Search(GetPointer(m_cSearchMarket),m_uBoundary,true));
  }

//+------------------------------------------------------------------+
//| PreparePublication - Refresh the pool on the current window.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::PreparePublication(CBufferFloat *market,const ulong boundary)
  {
   if(m_bPublicationPermit&&m_iTradeCount==0&&PreparedCount()>0)
     {
      if(!m_cPool.QCMRefresh(market,true,boundary))
         return(Broken(m_cPool.QCMStatus()));
      int status=(int)m_cPool.Control().At(AG_POOL_STATUS);
      if(status==AG_TECHNICAL)
         return(Broken(AG_TECHNICAL));
      if(status==AG_OK)
        {
         if(m_cPool.QCMChanged()||m_preparedWindow!=boundary)
           {
            if(m_preparedRevision==ULONG_MAX)
               return(Broken(AQCM_UPDATE_INCONSISTENT));
            m_preparedRevision++;
            m_preparedWindow=boundary;
           }
         m_preparedIC=m_cPool.QCMFitIC();
         m_preparedUsable=m_cPool.QCMAggregateUsable();
        }
      if(status==AG_OK&&m_preparedUsable)
        {
         m_bPoolWarmup=true;
         m_iStatus=AG_PREPARING_POOL;
         return(true);
        }
     }
   m_iStatus=AG_OK;

   return(true);
  }

bool CNeuronAlphaQCMOCL::FinishSearch(const bool drain)
  {
   if(!m_qReady||m_bCorrupted||m_updateInProgress)
      ReturnFalse;
   m_bSearchBudgetPending=false;
   m_bootstrap=false;

   if(m_loadMode==AQCM_TRAIN&&drain)
      return(Learn());
   return(true);
  }

//+------------------------------------------------------------------+
//| Drain confirmed transition credit without changing logical B.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::Learn(void)
  {
   while(HasSearchDebt())
     {
      if(!DrainSearchStep())
         ReturnFalse;
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| HasSearchDebt - Check confirmed transition credit.               |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::HasSearchDebt(void)
  {
   ulong warmup=(ulong)m_qcmConfig[AQCM_CFG_WARMUP-AG_CFG_SIZE];
   ulong credit=(ulong)m_qcmConfig[AQCM_CFG_CREDIT-AG_CFG_SIZE];
   ulong verified=SearchV();
   return(credit>0&&verified>warmup&&m_updates<(verified-warmup)/credit);
  }

//+------------------------------------------------------------------+
//| DrainSearchStep - Begin at most one complete learner update.     |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::DrainSearchStep(void)
  {
   if(!m_qReady||m_bCorrupted||m_updateInProgress||IsStopped()||
      m_qcmConfig[AQCM_CFG_CREDIT-AG_CFG_SIZE]<1)
      ReturnFalse;
   return(!HasSearchDebt()||LearnBatch(m_batch));
  }

//+------------------------------------------------------------------+
//| CanRecover - Gate recovery on a proven clean failure.            |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::CanRecover(void) const
  {
   return(m_loadMode==AQCM_TRAIN&&m_boundedLearner&&m_qReady&&m_bCorrupted&&m_recoverable&&
          !m_updateInProgress&&!m_bFrame);
  }

//+------------------------------------------------------------------+
//| ResumeRecovery - Permit the operator's finalization path.        |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::ResumeRecovery(void)
  {
   if(!CanRecover())
      ReturnFalse;
   m_bSearchBudgetPending=false;
   m_bootstrap=false;
   m_bCorrupted=false;
   m_recoverable=false;
   m_iStatus=AG_OK;
   return(true);
  }

//+------------------------------------------------------------------+
//| LearnBatch - Commit one update after common preflight.           |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::LearnBatch(AQCReplayBatch &batch)
  {
   int rows=(int)m_qcmConfig[AQCM_CFG_BATCH-AG_CFG_SIZE];
   int current=(int)m_qcmConfig[AQCM_CFG_QUANTILES-AG_CFG_SIZE];
   int target=(int)m_qcmConfig[AQCM_CFG_TARGET_QUANTILES-AG_CFG_SIZE];
   int period=(int)m_qcmConfig[AQCM_CFG_SYNC-AG_CFG_SIZE];
   CNeuronAlphaQCMEncoderOCL *q=(CNeuronAlphaQCMEncoderOCL *)m_qOnline.Layer(1);
   CNeuronAlphaQCMEncoderOCL *z=(CNeuronAlphaQCMEncoderOCL *)m_zOnline.Layer(1);
   CNeuronAlphaQCMEncoderOCL *qt=(CNeuronAlphaQCMEncoderOCL *)m_qTarget.Layer(1);
   CNeuronAlphaQCMEncoderOCL *zt=(CNeuronAlphaQCMEncoderOCL *)m_zTarget.Layer(1);
   if(period<1||!q||!z||!qt||!zt)
      return(Broken(AG_TECHNICAL));

   if(!m_replay.Sample(m_uSeed,m_updates,m_updates,batch))
      return(Broken(ReplayFailure()));
   if(!m_objective.Accumulate(OpenCL,batch,GetPointer(m_qOnline),GetPointer(m_zOnline),
                              GetPointer(m_qTarget),GetPointer(m_zTarget),rows,AG_VOCABULARY,current,target,
                              m_qcmConfig[AQCM_CFG_GAMMA-AG_CFG_SIZE],m_qcmConfig[AQCM_CFG_KAPPA-AG_CFG_SIZE],
                              m_fConfig[AG_CFG_POSITION_SCALE],m_preflight.GetIndex()))
      return(Broken(m_objective.Status()));
   if(!m_preflight.BufferRead())
      return(Broken(AG_TECHNICAL));
   if(m_preflight.At(0)!=0)
      return(Broken(AQCM_NUMERIC_ERROR,true));
   if(!m_replay.PreparePriorities(m_objective.PriorityLoss()))
      return(Broken(ReplayFailure()));

   m_updateInProgress=true;
   if(!q.Apply()||!z.Apply()||!m_replay.CommitPriorities())
      return(Broken(AQCM_UPDATE_INCONSISTENT));
   if((m_updates+1)%(ulong)period==0)
     {
      if(!m_qTarget.WeightsUpdate(GetPointer(m_qOnline),1.0f)||
         !m_zTarget.WeightsUpdate(GetPointer(m_zOnline),1.0f))
         return(Broken(AQCM_UPDATE_INCONSISTENT));
      m_syncs++;
     }
   m_updates++;
   m_updateInProgress=false;

   return(true);
  }

//+------------------------------------------------------------------+
//| Frame one internal graph's persistent encoder/head payload.       |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::NetworkSave(CNet *net,const int handle)
  {
   CNeuronBaseOCL *layer=net.Layer(1);
   ulong marker=FileTell(handle);
   if(!layer||layer.Type()!=defNeuronAlphaQCMEncoderOCL||
      FileWriteLong(handle,0)!=sizeof(long)||!layer.Save(handle))
      ReturnFalse;

   ulong end=FileTell(handle);
   if(end<=marker+sizeof(long)||!FileSeek(handle,(long)marker,SEEK_SET)||
      FileWriteLong(handle,(long)(end-marker-sizeof(long)))!=sizeof(long)||
      !FileSeek(handle,(long)end,SEEK_SET))
      ReturnFalse;

   return(true);
  }

bool CNeuronAlphaQCMOCL::NetworkLoad(CNet *net,const int handle,const bool restore,const bool iqn)
  {
   if(FileSize(handle)-FileTell(handle)<sizeof(long))
      ReturnFalse;

   long size=FileReadLong(handle);
   ulong start=FileTell(handle);
   if(size<INT_VALUE||(ulong)size>FileSize(handle)-start)
      ReturnFalse;
   if(!AlphaQCMSkipNetwork(handle,iqn,(int)m_fConfig[AG_CFG_MAX_TOKENS],AG_VOCABULARY,
                          (int)m_fConfig[AG_CFG_EMBEDDING],(int)m_fConfig[AG_CFG_SEARCH_HIDDEN],
                          (int)m_qcmConfig[AQCM_CFG_FREQUENCIES-AG_CFG_SIZE])||
      FileTell(handle)!=start+(ulong)size)
      ReturnFalse;

   if(restore&&(!FileSeek(handle,(long)(start+INT_VALUE),SEEK_SET)||
      !BuildNetwork(net,iqn,false)||!net.Layer(1).Load(handle)||FileTell(handle)!=start+(ulong)size))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Save the donor components and the framed training state.          |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::Save(const int handle)
  {
   if(!SafeToSave()||!Output.BufferRead()||!PrevOutput.BufferRead()||!Gradient.BufferRead()||
      !CNeuronBaseOCL::Save(handle))
      ReturnFalse;

   dFileWriteUInt(handle,AQCM_FILE_VERSION)
   dFileWriteUInt(handle,AQCM_CFG_SIZE)
   for(int i=0;i<AG_CFG_SIZE;i++)
     {
      dFileWriteFloat(handle,m_fConfig[i])
     }
   for(int i=0;i<AQCM_CFG_SIZE-AG_CFG_SIZE;i++)
     {
      dFileWriteFloat(handle,m_qcmConfig[i])
     }
   uint vocabulary[]=AG_VOCABULARY_ORDER;
   dFileWriteUInt(handle,AG_VOCABULARY)
   for(int i=0;i<AG_VOCABULARY;i++)
     {
      dFileWriteUInt(handle,vocabulary[i])
     }
   dFileWriteUInt(handle,m_iFinanceRandom)
   dFileWriteUInt(handle,m_bWarmBudget)
   dFileWriteUInt(handle,m_historyTimeframe)
   dFileWriteUInt(handle,m_preparedUsable)
   dFileWriteFloat(handle,m_preparedIC)
   ulong state[]={m_uSeed,m_uTradeUpdates,m_updates,m_syncs,m_candidate,m_preparedRevision,
                  m_activeGeneration,m_preparedWindow,m_observationWatermark,m_lastEvent,m_lastDecision};
   for(int i=0;i<ArraySize(state);i++)
      if(FileWriteLong(handle,(long)state[i])!=sizeof(long))
         ReturnFalse;

   string identity[]={m_historySymbol,m_historyServer};
   for(int i=0;i<ArraySize(identity);i++)
     {
      int length=StringLen(identity[i]);
      if(length>256)
         ReturnFalse;
      dFileWriteUInt(handle,length)
      for(int j=0;j<length;j++)
         if(FileWriteInteger(handle,StringGetCharacter(identity[i],j),SHORT_VALUE)!=SHORT_VALUE)
            ReturnFalse;
     }
   if(!m_cPool.Save(handle)||!m_cTrading.Save(handle)||!NetworkSave(GetPointer(m_qOnline),handle)||
      !NetworkSave(GetPointer(m_qTarget),handle)||!NetworkSave(GetPointer(m_zOnline),handle)||
      !NetworkSave(GetPointer(m_zTarget),handle)||!m_replay.Save(handle))
      ReturnFalse;

   return(true);
  }

//+------------------------------------------------------------------+
//| Restore owners in TRAIN and validate bounded sections otherwise.  |
//+------------------------------------------------------------------+
bool CNeuronAlphaQCMOCL::Load(const int handle)
  {
   if(!OpenCL||!CNeuronBaseOCL::Load(handle)||Neurons()!=AG_ACTIONS||Optimization()!=ADAM)
      ReturnFalse;
   // The native weightless branch returns before registering the layer.
   if(OpenCL.IsCollectingSelectiveKernels()&&!OpenCL.RegisterSelectiveLayer(Type(),ADAM))
      ReturnFalse;

   uint version=0,length=0;
   dFileReadUInt(handle,version)
   dFileReadUInt(handle,length)
   if(version!=AQCM_FILE_VERSION||length!=AQCM_CFG_SIZE||ArraySize(AlphaQCMExpectedConfig)!=AQCM_CFG_SIZE||
      ArrayResize(m_fConfig,AG_CFG_SIZE)!=AG_CFG_SIZE||
      ArrayResize(m_qcmConfig,AQCM_CFG_SIZE-AG_CFG_SIZE)!=AQCM_CFG_SIZE-AG_CFG_SIZE)
      ReturnFalse;
   for(int i=0;i<AG_CFG_SIZE;i++)
     {
      dFileReadUFloat(handle,m_fConfig[i])
      if(!MathIsValidNumber(m_fConfig[i])||m_fConfig[i]!=AlphaQCMExpectedConfig[i])
         ReturnFalse;
     }
   for(int i=0;i<AQCM_CFG_SIZE-AG_CFG_SIZE;i++)
     {
      dFileReadUFloat(handle,m_qcmConfig[i])
      if(!MathIsValidNumber(m_qcmConfig[i])||m_qcmConfig[i]!=AlphaQCMExpectedConfig[i+AG_CFG_SIZE])
         ReturnFalse;
     }
   uint vocabulary[]=AG_VOCABULARY_ORDER;
   uint token=0;
   dFileReadUInt(handle,length)
   if(length!=AG_VOCABULARY)
      ReturnFalse;
   for(int i=0;i<AG_VOCABULARY;i++)
     {
      dFileReadUInt(handle,token)
      if(token!=vocabulary[i])
         ReturnFalse;
     }

   uint warm=0,tf=0,aggregate=0;
   dFileReadUInt(handle,m_iFinanceRandom)
   dFileReadUInt(handle,warm)
   dFileReadUInt(handle,tf)
   dFileReadUInt(handle,aggregate)
   dFileReadUFloat(handle,m_preparedIC)
   if(warm>1||aggregate>1||!MathIsValidNumber(m_preparedIC)||m_iFinanceRandom==0)
      ReturnFalse;
   m_bWarmBudget=(warm==1);
   m_preparedUsable=(aggregate==1);
   m_historyTimeframe=(ENUM_TIMEFRAMES)tf;
   ulong state[11];
   for(int i=0;i<ArraySize(state);i++)
     {
      if(FileSize(handle)-FileTell(handle)<sizeof(long))
         ReturnFalse;
      state[i]=(ulong)FileReadLong(handle);
     }
   m_uSeed=state[0];
   m_uTradeUpdates=state[1];
   m_updates=state[2];
   m_syncs=state[3];
   m_candidate=state[4];
   m_preparedRevision=state[5];
   m_activeGeneration=state[6];
   m_preparedWindow=state[7];
   m_observationWatermark=state[8];
   m_lastEvent=state[9];
   m_lastDecision=state[10];
   string identity[2];
   for(int i=0;i<ArraySize(identity);i++)
     {
      dFileReadUInt(handle,length)
      if(length>256||(ulong)length*SHORT_VALUE>FileSize(handle)-FileTell(handle))
         ReturnFalse;
      for(uint j=0;j<length;j++)
         identity[i]+=ShortToString((ushort)FileReadInteger(handle,SHORT_VALUE));
     }
   m_historySymbol=identity[0];
   m_historyServer=identity[1];

   bool restore=m_loadMode==AQCM_TRAIN;
   if(!m_cEvaluator.Configure(OpenCL,m_fConfig)||!m_cPool.Configure(OpenCL,m_fConfig)||
      !m_cPool.Load(handle)||!m_cTrading.Load(handle,OpenCL)||!TradeWorkspace()||!SearchWorkspace(false)||
      !NetworkLoad(GetPointer(m_qOnline),handle,restore,false)||!NetworkLoad(GetPointer(m_qTarget),handle,restore,false)||
      !NetworkLoad(GetPointer(m_zOnline),handle,restore,true)||!NetworkLoad(GetPointer(m_zTarget),handle,restore,true)||
      !m_replay.Load(handle,(int)m_qcmConfig[AQCM_CFG_REPLAY-AG_CFG_SIZE],
                     (int)m_fConfig[AG_CFG_MAX_TOKENS],AG_VOCABULARY,restore))
      ReturnFalse;

   ulong warmup=(ulong)m_qcmConfig[AQCM_CFG_WARMUP-AG_CFG_SIZE];
   ulong credit=(ulong)m_qcmConfig[AQCM_CFG_CREDIT-AG_CFG_SIZE];
   ulong period=(ulong)m_qcmConfig[AQCM_CFG_SYNC-AG_CFG_SIZE];
   ulong verified=m_replay.Verified();
   if(credit==0||period==0||m_syncs!=m_updates/period||
      m_updates>(verified>warmup?(verified-warmup)/credit:0)||!m_cPool.QCMValidLoadedActive())
      ReturnFalse;

   m_bPublicationPermit=m_cPool.Control().At(AG_POOL_ACTIVE_READY)!=1;
   m_bConfigured=true;

   return(true);
  }

#endif // NEURONET_ALPHAQCM_COMPOSITE_MQH
