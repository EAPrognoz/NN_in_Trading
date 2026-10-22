//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property strict
#include "Trajectory.mqh"
//---
input group                       "---- AC-SRM training ----"
input datetime                    Start          = D'2024.01.01'; //Training period start
input datetime                    End            = D'2026.01.01'; //Training period end
input int                         Iterations     = 200000; //Training iterations
input int                         EpisodeBars    = 2 * StackSize; //Bars per episode
input int                         InpDealHorizon = 24; //Deal evaluation horizon (bars)
input double                      MinBalance     = 50.0; //Minimum account balance
input int                         UpdatePolicy   = 2; //Actor update interval (PolicyUpdatePeriod)
input int                         InpSeed        = 20260921; //Replay seed base (per-owner)
input int                         InpUpdateTargets = 20; //Target update interval (manifest metadata)
input float                       InpTauT          = 1.0f; //Target copy tau (1.0 = full copy)
//---
input group                       "---- SRM critic head ----"
input int                         InpRiskType    = defSRM_MEAN_CVAR; //Risk measure (defSRM_*)
input float                       InpRiskParameter = 0.1f; //Risk parameter (alpha/lambda/nu)
input float                       InpMeanWeight  = 0.0f; //Mean-CVaR omega
input float                       InpProbWeight  = 1.0f; //Distribution probability weight
input float                       InpValueWeight = 1.0f; //Distribution value weight
input int                         InpQuantiles   = 32; //Head quantiles (factory-fixed)
input int                         InpSelector    = 0; //Policy SRM gradient source (0=owner 1=Q1 2=Q2)
//---
input group                       "---- AC-SRM production checkpoint ----"
input ENUM_OMPB_STAGE             InpOMPBStage   = OMPB_STAGE_BASE_POLICY; //Chain stage: keep 03 (base policy)
//---
int            Epochs = 0;
CNet           Actor;
CNet           Q1;
CNet           Q2;

CBufferFloat   State;
CBufferFloat   TimeState;
CBufferFloat   Account;
CBufferFloat   NextAccount;
CBufferFloat   CurrentAction;
CBufferFloat   TeacherAction;
CBufferFloat   RandomAction;
CBufferFloat   CriticInput;
CBufferFloat   SRMValue;
CBufferFloat   SRMGrad;

ulong          PolicyTransitions = 0;
ulong          PolicyEvents = 0;
ulong          PolicySkippedInvalid = 0;
ulong          CriticTransitions = 0;
ulong          CriticQ1Transitions = 0;
ulong          CriticQ2Transitions = 0;
ulong          TeacherTransitions = 0;
ulong          TeacherCriticTransitions = 0;
ulong          RandomCriticTransitions = 0;
ulong          CriticQ1Total = 0;
ulong          CriticQ2Total = 0;
ulong          UnresolvedLabels = 0;
ulong          TagTP = 0;
ulong          TagSL = 0;
ulong          TagHorizon = 0;
double         RewardSum = 0;
double         CriticQ1Error = 0;
double         CriticQ2Error = 0;
bool           Stage03StopRequestedLogged = false;
//+-------------------------------------------------------------------+
//| Logs one user-requested Stage 03 stop without misclassifying it.  |
//+-------------------------------------------------------------------+
void LogStage03StopRequested(const string phase)
  {
   if(Stage03StopRequestedLogged)
      return;
   PrintFormat("ACSRM_STAGE03_STOP_REQUESTED phase=%s", phase);
   Stage03StopRequestedLogged = true;
  }
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Creates the AC-SRM Actor and distribution-Critic architectures.  Actor : Base(13)+Cross(3)+Conv(GELU 48->16)+Conv(SIGMOID 16->6).  Critic: Base(19)+Cross(5)+Conv(GELU 80->16)+Base(32,None)+          ACSRM head{count=1,window_out=32,None} (factory-fixed).  |
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool CreateActorCriticDescriptions(CArrayObj *&actor_descr, CArrayObj *&critic_descr)
  {
   actor_descr = new CArrayObj();
   critic_descr = new CArrayObj();
   if(!actor_descr || !critic_descr)
     {
      DeleteObj(actor_descr);
      DeleteObjAndFalse(critic_descr);
     }
   actor_descr.FreeMode(true);
   critic_descr.FreeMode(true);
//--- Actor: plain scenario trunk without the D2Skill bank layer.
   if(!SkillAddBase(actor_descr, AccountDescr) ||
      !SkillAddCross(actor_descr, false) ||
      !SkillAddConv(actor_descr, 1, (3 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !SkillAddConv(actor_descr, 1, EmbeddingSize, NActions / 2, 2, SIGMOID))
     {
      DeleteObj(actor_descr);
      DeleteObjAndFalse(critic_descr);
     }
//--- Critic: Base(19) feeds the Cross(5) trunk; the last plain Base(32,None)
//--- presents exactly 32 inputs to the factory-fixed ACSRM head.
   if(!SkillAddBase(critic_descr, AccountDescr + NActions) ||
      !SkillAddCross(critic_descr, true) ||
      !SkillAddConv(critic_descr, 1, (5 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !SkillAddBase(critic_descr, 32))
     {
      DeleteObj(actor_descr);
      DeleteObjAndFalse(critic_descr);
     }
   CLayerDescription *head = new CLayerDescription();
   if(!head)
     {
      DeleteObj(actor_descr);
      DeleteObjAndFalse(critic_descr);
     }
   head.type = defNeuronACSRM;
   head.count = 1;
   head.window_out = InpQuantiles;
   head.activation = None;
   head.optimization = ADAM;
   head.batch = BatchSize;
   if(!critic_descr.Add(head))
     {
      DeleteObjAndFalse(head);
      DeleteObj(actor_descr);
      DeleteObjAndFalse(critic_descr);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Validates the AC-SRM policy shapes (no D2 banks, ACSRM heads).   |
//+------------------------------------------------------------------+
bool ValidatePolicyShapeACSRM(CNet &actor, CNet &q1, CNet &q2)
  {
   CNeuronBaseOCL *a0 = actor.Layer(0);
   CNeuronBaseOCL *a1 = actor.Layer(1);
   CNeuronBaseOCL *a3 = actor.Layer(3);
   if(!a0 || !a1 || !a3 || a0.getOutput().Total() != (int)AccountDescr ||
      a1.Type() != defNeuronScenarioCrossAttention ||
      a3.getOutput().Total() != (int)NActions)
      ReturnFalse;
   for(int i = 0; i < 2; i++)
     {
      CNet *critic = (i == 0 ? GetPointer(q1) : GetPointer(q2));
      CNeuronBaseOCL *c0 = critic.Layer(0);
      CNeuronBaseOCL *c3 = critic.Layer(3);
      CNeuronBaseOCL *c4 = critic.Layer(4);
      if(!c0 || !c3 || !c4 ||
         c0.getOutput().Total() != (int)(AccountDescr + NActions) ||
         c3.getOutput().Total() != 32 ||
         c4.Type() != defNeuronACSRM ||
         c4.getOutput().Total() != 1)
         ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates fresh AC-SRM Actor + Q1 + Q2 on the Market context.      |
//+------------------------------------------------------------------+
bool CreatePolicySet(void)
  {
   CArrayObj *actor_descr = NULL, *critic_descr = NULL;
   if(!CreateActorCriticDescriptions(actor_descr, critic_descr))
     {
      DeleteObj(actor_descr);
      DeleteObj(critic_descr);
      ReturnFalse;
     }
   bool result = Actor.Create(actor_descr);
   if(result)
      result = Actor.SetOpenCLChecked(SkillMarket.GetOpenCL());
   if(result)
      result = Q1.Create(critic_descr);
   if(result)
      result = Q1.SetOpenCLChecked(SkillMarket.GetOpenCL());
   if(result)
      result = Q2.Create(critic_descr);
   if(result)
      result = Q2.SetOpenCLChecked(SkillMarket.GetOpenCL());
   DeleteObj(actor_descr);
   DeleteObj(critic_descr);
   return(result);
  }
//+------------------------------------------------------------------+
//| Loads or creates ACSRMActor.nnw / ACSRMQ1.nnw / ACSRMQ2.nnw.     |
//+------------------------------------------------------------------+
bool LoadOrCreatePolicies(void)
  {
   if(!SkillACRecoverTransaction(Skill_ACTOR_FILE, Skill_Q1_FILE, Skill_Q2_FILE,
                                   Skill_AC_MANIFEST_FILE, Skill_ACTOR_NEXT_FILE,
                                   Skill_Q1_NEXT_FILE, Skill_Q2_NEXT_FILE,
                                   Skill_AC_MANIFEST_NEXT_FILE, Skill_ACTOR_PREVIOUS_FILE,
                                   Skill_Q1_PREVIOUS_FILE, Skill_Q2_PREVIOUS_FILE,
                                   Skill_AC_MANIFEST_PREVIOUS_FILE,
                                   Skill_AC_TRANSACTION_FILE))
     {
      Print("Skill policy recovery=FAIL; incomplete checkpoint requires repair");
      ReturnFalse;
     }
   const bool manifest_exists = FileIsExist(Skill_AC_MANIFEST_FILE, FILE_COMMON);
   bool loaded = false;
   if(manifest_exists)
     {
      loaded = (SkillValidateACManifestFile(true, Skill_AC_MANIFEST_FILE, false) &&
                SkillLoadPolicyNet(Actor, Skill_ACTOR_FILE) &&
                SkillLoadPolicyNet(Q1, Skill_Q1_FILE) &&
                SkillLoadPolicyNet(Q2, Skill_Q2_FILE));
      if(!loaded)
         Print("Skill policy restore=FAIL; recreating incompatible policy set");
     }
   if(!loaded)
     {
      if(!CreatePolicySet())
         ReturnFalse;
     }
   if(!SkillBindPolicyOpenCLChecked(Actor, Q1, Q2) ||
      !ValidatePolicyShapeACSRM(Actor, Q1, Q2))
      ReturnFalse;
   Actor.TrainMode(true);
   Q1.TrainMode(true);
   Q2.TrainMode(true);
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the Stage 03 AC-SRM policy checkpoint.                     |
//+------------------------------------------------------------------+
bool SavePolicies(void)
  {
   return(SkillSaveStage03StopCheckpoint(Actor, Q1, Q2));
  }
//+------------------------------------------------------------------+
//| Implements PrepareHistory.                                       |
//+------------------------------------------------------------------+
bool PrepareHistory(int &first_position, int &last_position)
  {
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
      ReturnFalse;
   int wait = -1;
   bool calculated = false;
   do
     {
      calculated = (RSI.BarsCalculated() >= bars && CCI.BarsCalculated() >= bars &&
                    ATR.BarsCalculated() >= bars && MACD.BarsCalculated() >= bars);
      Sleep(100);
      wait++;
     }
   while(!calculated && wait < 100);
   if(!calculated)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
      ReturnFalse;
//--- The deal outcome must be resolvable INSIDE the training window: the
//--- full horizon is reserved past the End boundary so every label
//--- (TP/SL/HORIZON) completes within [Start, End].
   first_position = end + MathMax(1, SkillManifestDealHorizon);
//--- CreateBuffers(position,...,forecast) forms its state from
//--- position+NForecast and needs HistoryBars bars behind it.
   last_position = start - HistoryBars - NForecast;
   return(last_position > first_position);
  }
//+-------------------------------------------------------------------+
//| Applies the distribution target (realized return) to one critic.  |
//+-------------------------------------------------------------------+
bool UpdateCriticDistribution(CNet &critic, const double reward, const bool is_q1)
  {
   CNeuronBaseOCL *base = critic.Layer(3);
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!base || !head || head.Type() != defNeuronACSRM || !MathIsValidNumber(reward) ||
      base.getOutput().Total() != 32)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   if(!acsrm.calcDistributionTargetGradients(base, float(reward),
                                             InpProbWeight, InpValueWeight))
      ReturnFalse;
   //--- Real ACSRM critic error: RMSE of the FQF centroids from the realized
//--- outcome.  backPropGradient never updates getRecentAverageError(), so
//--- the old dashboard rRMSE stayed 0 forever; track the loss here.
   double rmse = -1;
   {
    CBufferFloat *quantiles = NULL, *probabilities = NULL;
    if(acsrm.GetDistributionBuffers(quantiles, probabilities) &&
       quantiles != NULL && quantiles.Total() > 0 && quantiles.BufferRead())
      {
       double sum_sq = 0;
       bool valid = true;
       for(int i = 0; i < quantiles.Total(); i++)
         {
          const double z = double(quantiles[i]);
          if(!MathIsValidNumber((float)z))
            { valid = false; break; }
          sum_sq += (z - reward) * (z - reward);
         }
       if(valid)
          rmse = MathSqrt(sum_sq / double(quantiles.Total()));
      }
   }
   if(rmse >= 0)
     {
      if(is_q1)
         CriticQ1Error += (rmse - CriticQ1Error) * 0.01;
      else
         CriticQ2Error += (rmse - CriticQ2Error) * 0.01;
     }
   return(critic.backPropGradient(GetPointer(SkillMarket), -1, -1, true));
  }
//+------------------------------------------------------------------+
//| Actor policy update through the frozen critic's SRM.             |
//+------------------------------------------------------------------+
int SelectPolicyCritic(const int event_index)
  {
//--- Exactly ONE Critic is the gradient source per policy-update event.
//--- The default selector is balanced over Actor-update EVENTS (a hash of
//--- the event counter and the replay seed), NOT over the Owner of the
//--- training example: with a sequential position-- sampler consecutive
//--- policy events inside a stretch would share one Owner parity.
   switch(InpSelector)
     {
      case 1:  return(0);
      case 2:  return(1);
      default: return(int((ulong(MathMax(event_index, 0)) * ulong(2654435761u) +
                           ulong(InpSeed)) & 1));
     }
  }
//+------------------------------------------------------------------+
//| Returns the scalar SRM value of one Critic head distribution.    |
//+------------------------------------------------------------------+
bool CriticSRMValue(CNet &critic, float &value)
  {
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   if(!SRMValue.BufferInit(1, 0) || !SRMValue.BufferCreate(SkillMarket.GetOpenCL()) ||
      !SRMValue.BufferWrite())
      ReturnFalse;
   if(!acsrm.SRMForward(GetPointer(SRMValue), 1, InpRiskType, InpRiskParameter, InpMeanWeight))
      ReturnFalse;
   if(!SRMValue.BufferRead())
      ReturnFalse;
   value = float(SRMValue[0]);
   return(MathIsValidNumber(value));
  }
//+----------------------------------------------------------------------------------------------------------------+
//| Reports whether the last SRM refusal was a distribution-contract  rejection rather than an execution failure.  |
//+----------------------------------------------------------------------------------------------------------------+
bool CriticLastSRMInvalid(CNet &critic)
  {
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      return(false);
   return(((CNeuronACSRM*)head).LastSRMInvalid());
  }
//+-----------------------------------------------------------------------------------+
//| Actor policy update through the frozen critic's SRM (PolicyOutputGradientACSRM).  |
//+-----------------------------------------------------------------------------------+
bool PolicyOutputGradientACSRM(CNet &actor, CNet &critic, const float upstream = 1.0f)
  {
   CNeuronBaseOCL *critic_head = critic.Layer(4);
   CNeuronBaseOCL *critic_base = critic.Layer(0);
   CNeuronBaseOCL *actor_output = actor.Layer(3);
   if(!critic_head || critic_head.Type() != defNeuronACSRM || !critic_base || !actor_output ||
      critic_base.getGradient().Total() != (int)(AccountDescr + NActions) ||
      actor_output.getGradient().Total() != (int)NActions)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)critic_head;
   if(!SRMValue.BufferInit(1, 0) || !SRMValue.BufferCreate(SkillMarket.GetOpenCL()) ||
      !SRMValue.BufferWrite())
      ReturnFalse;
   if(!acsrm.SRMForward(GetPointer(SRMValue), 1, InpRiskType, InpRiskParameter, InpMeanWeight))
      ReturnFalse;
   if(!SRMGrad.BufferInit(1, upstream) || !SRMGrad.BufferCreate(SkillMarket.GetOpenCL()) ||
      !SRMGrad.BufferWrite())
      ReturnFalse;
   if(!acsrm.SRMGradient(GetPointer(SRMGrad), 1, InpRiskType, InpRiskParameter, InpMeanWeight))
      ReturnFalse;
//--- Freeze critic weights while propagating the policy gradient.
   if(!critic.SetWeightsUpdate(false))
      ReturnFalse;
   bool result = critic.backPropGradient(GetPointer(SkillMarket), -1, -1, true);
   if(!critic.SetWeightsUpdate(true))
      result = false;
   CBufferFloat account_gradient;
   if(!(result &&
        account_gradient.BufferInit(AccountDescr, 0) &&
        (account_gradient.GetIndex() >= 0 || account_gradient.BufferCreate(SkillMarket.GetOpenCL())) &&
        SkillDevice.Split2(GetPointer(account_gradient), actor_output.getGradient(),
                           critic_base.getGradient(), AccountDescr, NActions) &&
        ApplyActorSigmoidDerivative(actor_output)))
      result = false;
   return(result);
  }
//+---------------------------------------------------------------------------------------------------------------------+
//| Accumulates one supervised signal (teacher/random) into the  actor output-layer gradient: (t - a) .* a .* (1 - a).  |
//+---------------------------------------------------------------------------------------------------------------------+
bool AddActorSupervisedGradient(CNet &actor, CBufferFloat *target)
  {
   CNeuronBaseOCL *actor_output = actor.Layer(3);
   if(!actor_output || !target || target.Total() != (int)NActions ||
      actor_output.getOutput().Total() != (int)NActions ||
      actor_output.getGradient().Total() != (int)NActions || target.GetIndex() < 0 ||
      actor_output.getOutput().GetIndex() < 0 || actor_output.getGradient().GetIndex() < 0)
      ReturnFalse;
   if(!ActorDerivOneMinus.BufferInit(NActions, 0) ||
      !ActorDerivOneMinus.BufferCreate(SkillMarket.GetOpenCL()) ||
      !ActorDerivOneMinus.BufferWrite() ||
      !ActorGradScratch.BufferInit(NActions, 0) ||
      !ActorGradScratch.BufferCreate(SkillMarket.GetOpenCL()) ||
      !ActorGradScratch.BufferWrite() ||
      !SkillDevice.Bind(SkillMarket.GetOpenCL()))
      ReturnFalse;
//--- diff = t - a; scaled = diff * a * (1 - a); accumulate on device.
//--- This reproduces the library's calcOutputGradients for a SIGMOID
//--- output exactly, but pre-update and summed before ONE weight update.
   if(!SkillDevice.Subtract(target, actor_output.getOutput(), GetPointer(ActorDerivOneMinus), NActions) ||
      !SkillDevice.DeActivationOnDevice(actor_output.getOutput(), GetPointer(ActorGradScratch),
                                        GetPointer(ActorDerivOneMinus), SIGMOID) ||
      !SkillDevice.Add(actor_output.getGradient(), GetPointer(ActorGradScratch),
                       actor_output.getGradient(), NActions))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements TrainTransition.                                      |
//+------------------------------------------------------------------+
bool TrainTransition(const int position, const int iteration, bool &terminal)
  {
   terminal = false;
   const bool trace_td = (iteration < 24 || iteration % 10000 == 0 || iteration >= Iterations - 1);
   if(!SkillForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
     { Print("TrainTransition stage=current_forecast"); ReturnFalse; }
   double reward = 0;
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(CurrentAction)) ||
      !AdvanceAccount(GetPointer(Account), GetPointer(CurrentAction), position, MinBalance,
                      GetPointer(NextAccount), reward, terminal))
     { Print("TrainTransition stage=account_transition"); ReturnFalse; }
   const double buy_lot = MathMax(0.0, double(CurrentAction[0] - CurrentAction[3]));
   const double sell_lot = MathMax(0.0, double(CurrentAction[3] - CurrentAction[0]));
   if(trace_td)
      PrintFormat("ACSRM action iteration=%d balance=%.2f buy_lot=%.8f buy_tp=%.8f buy_sl=%.8f " +
                  "sell_lot=%.8f sell_tp=%.8f sell_sl=%.8f",
                  iteration, MathMax(0.0, double(Account[0]) * EtalonBalance),
                  buy_lot, double(CurrentAction[1]), double(CurrentAction[2]),
                  sell_lot, double(CurrentAction[4]), double(CurrentAction[5]));
   if(trace_td)
      PrintFormat("ACSRM return iteration=%d reward=%.8f balance=%.2f terminal=%s",
                  iteration, reward, MathMax(0.0, double(NextAccount[0]) * EtalonBalance),
                  (terminal ? "true" : "false"));
//--- Critic input is built from the live device activations.
   CNeuronBaseOCL *context_layer = Actor.Layer(0);
   CNeuronBaseOCL *actor_layer = Actor.Layer(3);
   if(!context_layer || !actor_layer ||
      !BuildCriticInput(context_layer.getOutput(), actor_layer.getOutput(), GetPointer(CriticInput)))
     { Print("TrainTransition stage=critic_input"); ReturnFalse; }
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1))
     { Print("TrainTransition stage=critic_forward"); ReturnFalse; }
//--- Offline target: the COMPLETED result of the specific deal fixed at this
//--- bar over the configured horizon (TP first / SL first / HORIZON outcome),
//--- discounted by elapsed bars.  Realized future bars may label the deal but
//--- never enter the decision state (SkillForwardForecast already shifts the
//--- window to the strictly-past bars).  It is a finished result and needs no
//--- target bootstrap.
   int deal_tag = 0;
   const double deal_y = EvaluateAction(GetPointer(CurrentAction),
                                        MathMax(0.0, double(Account[0]) * EtalonBalance),
                                        (uint)position, InpDealHorizon, deal_tag);
   if(!MathIsValidNumber(deal_y))
     { Print("TrainTransition stage=deal_label"); ReturnFalse; }
   if(trace_td)
      PrintFormat("ACSRM offline iteration=%d outcome=%.8f tag=%d advance_reward=%.8f horizon=%d",
                  iteration, deal_y, deal_tag, reward, InpDealHorizon);
//--- Offline Owner separation: the example id (position,slot,seed) fixes one
//--- stable Owner-Critic; re-reading the same record keeps its label there.
   const int owner = SkillObservationOwner(position, 0, InpSeed);
   if(deal_tag == 0)
      UnresolvedLabels++;
   else
     {
      CNet *owner_critic = (owner == 0 ? GetPointer(Q1) : GetPointer(Q2));
      if(!owner_critic.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
         !UpdateCriticDistribution(owner_critic, deal_y, (owner_critic == GetPointer(Q1))))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("critic_distribution_backward");
            return(false);
           }
         Print("TrainTransition stage=critic_distribution_backward");
         ReturnFalse;
        }
      CriticTransitions++;
      if(owner_critic == GetPointer(Q1))
        {
         CriticQ1Transitions++;
         CriticQ1Total++;
        }
      else
        {
         CriticQ2Transitions++;
         CriticQ2Total++;
        }
      RewardSum += deal_y;
      if(deal_tag == 1)
         TagTP++;
      else
         if(deal_tag == 2)
            TagSL++;
         else
            TagHorizon++;
     }
//--- Actor multi-objective update.  Policy, teacher and random signals are
//--- computed on the SAME forward activations and the SAME pre-update
//--- weights and ACCUMULATED into the output-layer gradient; ONE
//--- backPropGradient applies them together.  No weight change happens
//--- between the contributions so credit assignment stays consistent.
   bool actor_update_needed = false;
   bool actor_update_blocked = false;
//--- Gradient hygiene: clear the Actor output gradient once BEFORE the
//--- current observation accumulates its contributions.  A deferred policy
//--- event (UpdatePolicy > 1) would otherwise Add teacher/random on top of
//--- the stale gradient the previous combined update consumed, since Split2
//--- only overwrites the gradient on the policy event.
   if(actor_layer.getGradient().Total() != (int)NActions ||
      !actor_layer.getGradient().Fill(0))
     { Print("TrainTransition stage=actor_gradient_reset"); ReturnFalse; }
//--- Policy gate: a RESOLVABLE deal label (tag != 0).  Requiring broker
//--- executability here would give the cold-start Actor no gradient while
//--- its lots are below LotsMin(); with the min-lot floor inside
//--- CheckAction the gradient direction exists from the first iteration.
   const bool resolvable_action = (deal_tag != 0);
   if(iteration > 0 && UpdatePolicy > 0 && iteration % UpdatePolicy == 0)
     {
      if(resolvable_action)
        {
         //--- Fresh Critic activations on the same observation AFTER the Critic
         //--- weight update: the policy gradient must not mix pre-update
         //--- activations with post-update weights.
         if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
            !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1))
           { Print("TrainTransition stage=policy_critic_forward"); ReturnFalse; }
         const int pick = SelectPolicyCritic(int(PolicyEvents++));
         CNet *policy_critic = (pick == 0 ? GetPointer(Q1) : GetPointer(Q2));
         float policy_srm = 0.0f;
         if(!CriticSRMValue(*policy_critic, policy_srm))
           {
            if(CriticLastSRMInvalid(*policy_critic))
              {
               PolicySkippedInvalid++;
               actor_update_blocked = true;
               if(trace_td)
                  PrintFormat("ACSRM policy skipped iteration=%d reason=invalid_srm_distribution", iteration);
              }
            else
              { Print("TrainTransition stage=policy_critic_srm"); ReturnFalse; }
           }
         else
           {
            const bool policy_ok = PolicyOutputGradientACSRM(Actor, *policy_critic, 1.0f);
            if(!policy_ok)
              {
               if(CriticLastSRMInvalid(*policy_critic))
                 {
                  PolicySkippedInvalid++;
                  actor_update_blocked = true;
                  if(trace_td)
                     PrintFormat("ACSRM policy skipped iteration=%d reason=invalid_srm_gradient", iteration);
                 }
               else
                 { Print("TrainTransition stage=policy_backward"); ReturnFalse; }
              }
            else
              {
               PolicyTransitions++;
               actor_update_needed = true;
              }
           }
        }
      else
         PolicySkippedInvalid++;
     }
//--- Restored original-project Actor signals: a profitable realized-future
//--- teacher and a profitable random executable action re-enter the action
//--- manifold; their deals also expand Owner-Critic coverage.  Only
//--- RESOLVABLE labels (tag != 0) train anything: with the SL sign fixed
//--- the reward > 0 gate now stands for a genuine TP profit.
   double teacher_reward = 0;
   int teacher_tag = 0;
   if(!BuildTeacherAction(position, GetPointer(Account), GetPointer(TeacherAction),
                          teacher_reward, InpDealHorizon, teacher_tag))
     { Print("TrainTransition stage=teacher_action"); ReturnFalse; }
   if(trace_td)
      PrintFormat("ACSRM teacher iteration=%d reward=%.8f tag=%d", iteration, teacher_reward, teacher_tag);
   if(teacher_reward > 0.0 && teacher_tag != 0)
     {
      if(!AddActorSupervisedGradient(Actor, GetPointer(TeacherAction)))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("teacher_actor_backward");
            return(false);
           }
         Print("TrainTransition stage=teacher_actor_backward");
         ReturnFalse;
        }
      TeacherTransitions++;
      actor_update_needed = true;
     }
   if(teacher_tag == 0)
      UnresolvedLabels++;
   else
     {
      const int teacher_owner = SkillObservationOwner(position, 1, InpSeed);
      CNet *teacher_critic = (teacher_owner == 0 ? GetPointer(Q1) : GetPointer(Q2));
      if(!BuildCriticInput(context_layer.getOutput(), GetPointer(TeacherAction), GetPointer(CriticInput)) ||
         !teacher_critic.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
         !UpdateCriticDistribution(teacher_critic, teacher_reward, (teacher_critic == GetPointer(Q1))))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("teacher_critic_backward");
            return(false);
           }
         Print("TrainTransition stage=teacher_critic_backward");
         ReturnFalse;
        }
      TeacherCriticTransitions++;
      if(teacher_critic == GetPointer(Q1))
         CriticQ1Total++;
      else
         CriticQ2Total++;
     }
   double random_reward = 0;
   int random_tag = 0;
   if(!BuildRandomAction(position, GetPointer(Account), GetPointer(RandomAction),
                         random_reward, InpDealHorizon, random_tag))
     { Print("TrainTransition stage=random_action"); ReturnFalse; }
   if(trace_td)
      PrintFormat("ACSRM random iteration=%d reward=%.8f tag=%d", iteration, random_reward, random_tag);
   if(random_reward > 0.0 && random_tag != 0)
     {
      if(!AddActorSupervisedGradient(Actor, GetPointer(RandomAction)))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("random_actor_backward");
            return(false);
           }
         Print("TrainTransition stage=random_actor_backward");
         ReturnFalse;
        }
      actor_update_needed = true;
     }
   if(random_tag == 0)
      UnresolvedLabels++;
   else
     {
      const int random_owner = SkillObservationOwner(position, 2, InpSeed);
      CNet *random_critic = (random_owner == 0 ? GetPointer(Q1) : GetPointer(Q2));
      if(!BuildCriticInput(context_layer.getOutput(), GetPointer(RandomAction), GetPointer(CriticInput)) ||
         !random_critic.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
         !UpdateCriticDistribution(random_critic, random_reward, (random_critic == GetPointer(Q1))))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("random_critic_backward");
            return(false);
           }
         Print("TrainTransition stage=random_critic_backward");
         ReturnFalse;
        }
      RandomCriticTransitions++;
      if(random_critic == GetPointer(Q1))
         CriticQ1Total++;
      else
         CriticQ2Total++;
     }
//--- ONE coherent Actor weight update with the combined gradient (if any).
//--- An invalid SRM distribution blocks the whole Actor update for this
//--- observation, matching srm_invalid_policy=skip_actor_update.
   if(actor_update_needed && !actor_update_blocked &&
      !Actor.backPropGradient(GetPointer(SkillMarket), -1, -1, true))
     {
      if(IsStopped())
        {
         LogStage03StopRequested("actor_combined_backward");
         return(false);
        }
      Print("TrainTransition stage=actor_combined_backward");
      ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements TrainACSRMActorCritic.                                |
//+------------------------------------------------------------------+
void TrainACSRMActorCritic(void)
  {
   Stage03StopRequestedLogged = false;
   int first = 0, last = 0;
   if(!PrepareHistory(first, last))
     { PrintFormat("%s -> %d history unavailable", __FUNCTION__, __LINE__); return; }
   uint shown = GetTickCount();
   int completed_iterations = 0;
   int failed_iteration = -1;
   int failed_position = -1;
   int position = last;
   int episode = 0;
   PolicyTransitions = 0;
   PolicyEvents = 0;
   PolicySkippedInvalid = 0;
   CriticTransitions = 0;
   CriticQ1Transitions = 0;
   CriticQ2Transitions = 0;
   TeacherTransitions = 0;
   TeacherCriticTransitions = 0;
   RandomCriticTransitions = 0;
   CriticQ1Total = 0;
   CriticQ2Total = 0;
   UnresolvedLabels = 0;
   TagTP = 0;
   TagSL = 0;
   TagHorizon = 0;
   RewardSum = 0;
   CriticQ1Error = 0;
   CriticQ2Error = 0;
   for(int iteration = 0; iteration < Iterations && !IsStopped(); iteration++)
     {
      //--- Fresh episode: seed the sampler from the observation Owner.
      if(episode == 0)
        {
         if(!SkillForwardForecast(position, GetPointer(State), GetPointer(TimeState)))
            break;
         MathSrand(InpSeed + position);
         const vector<float> sampled = SampleAccount(GetPointer(State), Rates[position].time,
                                                      EtalonBalance, MinBalance);
         if(sampled.Size() != AccountDescr || !Account.AssignArray(sampled) ||
            (Account.GetIndex() < 0 && !Account.BufferCreate(SkillMarket.GetOpenCL())) ||
            !Account.BufferWrite() || !Actor.Clear())
            break;
        }
      bool terminal = false;
      if(!TrainTransition(position, iteration, terminal))
        {
         if(IsStopped())
           {
            LogStage03StopRequested("train_transition");
            break;
           }
         failed_iteration = iteration;
         failed_position = position;
         break;
        }
      completed_iterations++;
      if((NextAccount.GetIndex() >= 0 && !NextAccount.BufferRead()) ||
         !Account.AssignArray(GetPointer(NextAccount)) ||
         (Account.GetIndex() >= 0 && !Account.BufferWrite()))
        {
         failed_iteration = iteration;
         failed_position = position;
         break;
        }
      position--;
      episode++;
      //--- Close the episode on range exhaustion, its bar limit or a
      //--- terminal account state.
      if(position < first || episode >= MathMax(1, EpisodeBars) || terminal)
        {
         if(iteration % 10000 == 0)
            PrintFormat("ACSRM episode closed iteration=%d episode=%d position=%d terminal=%s",
                        iteration, episode, position, (terminal ? "true" : "false"));
         if(position < first)
            position = last;
         episode = 0;
        }
      //--- Refresh the progress panel without delaying training updates.
      if(GetTickCount() - shown > 500)
        {
         
                                
         
                                
         const double r_mean = (CriticTransitions > 0 ? RewardSum / double(CriticTransitions) : 0.0);
         Comment(StringFormat("ACSRM AC %6.2f%% Q1 err %.8f Q2 err %.8f critic %I64u " +
                              "policy %I64u skip %I64u Rmean %.4f",
                              100.0 * iteration / MathMax(Iterations, 1), CriticQ1Error, CriticQ2Error,
                              CriticTransitions, PolicyTransitions, PolicySkippedInvalid,
                              r_mean));
         shown = GetTickCount();
        }
     }
   if(IsStopped())
     {
//--- A stop can land between parameter updates; never publish a partial step.
      LogStage03StopRequested("training_loop");
      Comment("");
      PrintFormat("ACSRM_STAGE03_STOP_NO_PUBLISH completed=%d/%d critic=%I64u q1=%I64u q2=%I64u policy=%I64u",
                  completed_iterations, Iterations, CriticTransitions,
                  CriticQ1Transitions, CriticQ2Transitions, PolicyTransitions);
      return;
     }
   Comment("");
//--- Reject publication unless every requested iteration completed safely.
   if(completed_iterations != Iterations)
     {
      if(failed_iteration >= 0)
         PrintFormat("%s -> %d publication aborted: TrainTransition failed iteration=%d position=%d completed=%d/%d",
                     __FUNCTION__, __LINE__, failed_iteration, failed_position,
                     completed_iterations, Iterations);
      else
         PrintFormat("%s -> %d publication aborted: stop requested completed=%d/%d",
                     __FUNCTION__, __LINE__, completed_iterations, Iterations);
      return;
     }
   if(!SavePolicies())
      PrintFormat("%s -> %d save failed", __FUNCTION__, __LINE__);
   else
      PrintFormat("ACSRM_STUDY_PASS iterations=%d critic=%I64u q1=%I64u q2=%I64u " +
                  "q1_total=%I64u q2_total=%I64u policy=%I64u skip=%I64u teacher=%I64u " +
                  "teacher_critic=%I64u random_critic=%I64u unresolved=%I64u " +
                  "tags_tp=%I64u tags_sl=%I64u tags_horizon=%I64u " +
                  "q1_err=%.8f q2_err=%.8f published=canonical",
                  completed_iterations, CriticTransitions, CriticQ1Transitions, CriticQ2Transitions,
                  CriticQ1Total, CriticQ2Total,
                  PolicyTransitions, PolicySkippedInvalid, TeacherTransitions,
                  TeacherCriticTransitions, RandomCriticTransitions, UnresolvedLabels,
                  TagTP, TagSL, TagHorizon, CriticQ1Error, CriticQ2Error);
  }
//+------------------------------------------------------------------+
//| Initializes the Stage 03 BasePolicy Expert.                      |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
//--- Pin the deal-evaluation horizon for manifest writer/validator symmetry.
   SkillManifestDealHorizon = MathMax(1, InpDealHorizon);
   SkillManifestTargetTau = double(InpTauT);
   SkillManifestTargetUpdatePeriod = InpUpdateTargets;
   if(InpSelector < 0 || InpSelector > 2 ||
      InpUpdateTargets < 0 || InpTauT < 0.0f || InpTauT > 1.0f)
     {
      PrintFormat("ACSRM input validation failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
//--- Confirm Stage 03 and that the frozen Forecast checkpoint is intact.
   if(InpOMPBStage != OMPB_STAGE_BASE_POLICY ||
      !SkillInitIndicators() || !SkillLoadForecastInference() ||
      !SkillConfigureProductionACSRMCheckpoint() ||
      !SkillValidateProductionACSRMCheckpoint() ||
      !SkillCaptureProductionACSRMFingerprints() || !LoadOrCreatePolicies() ||
      !SkillVerifyFrozenForecastExact() ||
      !SkillVerifyProductionACSRMFingerprints())
     {
      PrintFormat("ACSRM Actor-Critic initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
   if(!EventSetMillisecondTimer(1))
     {
      PrintFormat("ACSRM Actor-Critic timer initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
   //--- Finalize initialization only after the Stage 03 timer is armed.
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
//--- Only a completed explicit training loop persists all three artifacts
//--- and writes the hash manifest last. Deinit must not create a mixed
//--- generation.
   if(SkillForecast != NULL && !SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   if(SkillProductionSignatureReady && !SkillVerifyProductionACSRMFingerprints())
      PrintFormat("%s -> %d ACSRM production signature mutation", __FUNCTION__, __LINE__);
   SkillForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Runs the one-shot training lifecycle.                            |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   TrainACSRMActorCritic();
   ExpertRemove();
  }
//+------------------------------------------------------------------+
