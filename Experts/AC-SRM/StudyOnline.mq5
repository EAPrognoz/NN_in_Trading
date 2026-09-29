//+------------------------------------------------------------------+
//|                                                  StudyOnline.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "AC-SRM Stage 03 online: дообучение политики на LIVE-потоке (кросс-таргет TD)."
#property description "Один переход на закрытый бар: r — прирост баланса/эквити, бутстрап y1/y2 от TargetQ1/Q2."
#property description "Источник — LIVE-котировки и счёт тестерного терминала; результат — состояние энкодера/критиков."
#define Online
#include "Trajectory.mqh"
//---
input group                       "---- AC-SRM online training ----"
input int                         InpDealHorizon = 24;            //Offline pretraining horizon; checkpoint compatibility
input double                      MinBalance     = 50.0;          //Minimum account balance
input int                         UpdatePolicy   = 2;             //Actor update interval at deal-open states
input int                         InpUpdateTargets = 2;           //Target-network update interval
input float                       InpTauT        = 0.05f;         //Cross-Polyak target smoothing tauT
input float                       InpGamma       = 0.5f;          //TD discount per resolved bar (gamma)
input int                         InpSelector    = 0;             //Policy SRM gradient source (0=event-hash 1=Q1 2=Q2)
input int                         InpSeed        = 20260921;      //Replay seed base (per-owner)
//---
input group                       "---- SRM critic head ----"
input int                         InpRiskType    = defSRM_MEAN_CVAR; //Risk measure (defSRM_*)
input float                       InpRiskParameter = 0.1f;           //Risk parameter (alpha/lambda/nu)
input float                       InpMeanWeight  = 0.0f;             //Mean-CVaR omega
input float                       InpProbWeight  = 1.0f;             //Distribution probability weight
input float                       InpValueWeight = 1.0f;             //Distribution value weight
input int                         InpQuantiles   = 32;               //Head quantiles (factory-fixed)
//---
input group                       "---- AC-SRM production checkpoint ----"
input ENUM_OMPB_STAGE             InpOMPBStage   = OMPB_STAGE_BASE_POLICY; //Chain stage: keep 03 (base policy)
//---
int            Epochs = 0;
CNet           Actor;
CNet           TargetActor;
CNet           Q1;
CNet           Q2;
CNet           TargetQ1;
CNet           TargetQ2;

CBufferFloat   State;
CBufferFloat   TimeState;
CBufferFloat   Account;
CBufferFloat   CurrentAction;
CBufferFloat   CriticInput;
CBufferFloat   TargetCriticInput;
CBufferFloat   SRMValue;
CBufferFloat   SRMGrad;
CBufferFloat   TargetSample;
CBufferFloat   TargetRandom;
CBufferFloat   NextState;
CBufferFloat   NextTimeState;
CBufferFloat   NextAccount;

ulong          PolicyTransitions = 0;
ulong          PolicyEvents = 0;
ulong          PolicySkippedInvalid = 0;
ulong          CriticTransitions = 0;
ulong          CriticQ1Total = 0;
ulong          CriticQ2Total = 0;
ulong          TargetTransitions = 0;
ulong          TargetUpdates = 0;
bool           TargetQ1First = false;
double         RewardSum = 0;
double         BootstrapSum = 0;
//--- LIVE stream training state.
int            LiveSteps = 0;          //Actions taken on the live stream
int            LiveIteration = -1;     //Index of the PENDING action awaiting its LIVE reward
double         LivePrevBalance = 0;    //Post-execution LIVE snapshot of the previous bar
double         LivePrevEquity  = 0;
bool           LiveReady = false;
bool           LiveFinalized = false;
bool           LiveStopRequested = false;
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
//| Applies the distribution TD target to one critic.                |
//+------------------------------------------------------------------+
bool UpdateCriticDistribution(CNet &critic, const double target)
  {
   CNeuronBaseOCL *base = critic.Layer(3);
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!base || !head || head.Type() != defNeuronACSRM || !MathIsValidNumber(target) ||
      base.getOutput().Total() != 32)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   if(!acsrm.calcDistributionTargetGradients(base, float(target),
                                             InpProbWeight, InpValueWeight))
      ReturnFalse;
   return(critic.backPropGradient(GetPointer(SkillMarket), -1, -1, true));
  }
//+------------------------------------------------------------------+
//| Samples one scalar realization from an ACSRM head distribution.  |
//+------------------------------------------------------------------+
bool SampleTargetHead(CNet &critic, CBufferFloat &sample, CBufferFloat &random, float &value)
  {
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
//--- BufferInit before BufferCreate: creating an empty buffer leaves the GPU
//--- index invalid and a subsequent BufferWrite would fail.  The buffers are
//--- initialized once and reused on the hot bootstrap path.
   if((sample.Total() != 1 || sample.GetIndex() < 0) &&
      (!sample.BufferInit(1, 0) || !sample.BufferCreate(SkillMarket.GetOpenCL())))
      ReturnFalse;
   if((random.Total() != 1 || random.GetIndex() < 0) &&
      (!random.BufferInit(1, 0) || !random.BufferCreate(SkillMarket.GetOpenCL())))
      ReturnFalse;
   //--- rev5: u is sampled on the OPEN interval (0,1).  MathRand()/32768
//--- would emit the exact 0 for MathRand()==0 and degenerate the CDF
//--- crossing; the +0.5 shift keeps u strictly inside (0,1) for every seed.
   if(!random.Update(0, float(MathRand() + 0.5) / 32768.0f) || !random.BufferWrite())
      ReturnFalse;
   if(!acsrm.Sample(GetPointer(sample), GetPointer(random), 1))
      ReturnFalse;
   if(!sample.BufferRead())
      ReturnFalse;
   value = sample[0];
   return(MathIsValidNumber(value));
  }
//+-------------------------------------------------------------------------------------------------------------------------------------+
//| Applies the TD target to one critic and returns the prepared FQF  value/probability gradient norms (diagnostics, before backward).  |
//+-------------------------------------------------------------------------------------------------------------------------------------+
bool UpdateCriticDistributionDiag(CNet &critic, const double target,
                                  double &norm_value_gradient, double &norm_prob_gradient)
  {
   norm_value_gradient = 0.0;
   norm_prob_gradient = 0.0;
   CNeuronBaseOCL *base = critic.Layer(3);
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!base || !head || head.Type() != defNeuronACSRM || !MathIsValidNumber(target) ||
      base.getOutput().Total() != 32)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   if(!acsrm.calcDistributionTargetGradients(base, float(target),
                                             InpProbWeight, InpValueWeight))
      ReturnFalse;
//--- Norms are captured BEFORE the backward pass consumes the prepared
//--- distribution gradients.
   CBufferFloat *quantiles_gr = NULL;
   CBufferFloat *probabilities_gr = NULL;
   if(acsrm.GetDistributionGradientBuffers(quantiles_gr, probabilities_gr))
     {
      if(quantiles_gr.GetIndex() >= 0 && !quantiles_gr.BufferRead())
         ReturnFalse;
      if(probabilities_gr.GetIndex() >= 0 && !probabilities_gr.BufferRead())
         ReturnFalse;
      double nv = 0.0, np = 0.0;
      for(int i = 0; i < quantiles_gr.Total(); i++)
         nv += double(quantiles_gr[i]) * double(quantiles_gr[i]);
      for(int i = 0; i < probabilities_gr.Total(); i++)
         np += double(probabilities_gr[i]) * double(probabilities_gr[i]);
      norm_value_gradient = MathSqrt(nv);
      norm_prob_gradient = MathSqrt(np);
     }
   return(critic.backPropGradient(GetPointer(SkillMarket), -1, -1, true));
  }
//+------------------------------------------------------------------------------------------------------------------------------+
//| Emits the FQF head health report for one critic: distribution  support, SRM/mean, entropy, gradient norms and weight scale.  |
//+------------------------------------------------------------------------------------------------------------------------------+
bool FQFDiagnosticsOnline(CNet &critic, const double y, const double reward,
                          const float z, const double norm_value_gradient,
                          const double norm_prob_gradient, const string tag)
  {
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   CBufferFloat *quantiles = NULL;
   CBufferFloat *probabilities = NULL;
   if(!acsrm.GetDistributionBuffers(quantiles, probabilities) ||
      quantiles.GetIndex() < 0 || probabilities.GetIndex() < 0 ||
      !quantiles.BufferRead() || !probabilities.BufferRead())
      ReturnFalse;
   const int n = quantiles.Total();
   if(n < 2 || probabilities.Total() != n)
      ReturnFalse;
   double z_min = double(quantiles[0]);
   double z_max = double(quantiles[0]);
   double z_sum = 0.0;
   double ent = 0.0;
   double p_min = 1.0;
   for(int i = 0; i < n; i++)
     {
      const double q = double(quantiles[i]);
      const double p = MathMax(0.0, double(probabilities[i]));
      z_min = MathMin(z_min, q);
      z_max = MathMax(z_max, q);
      z_sum += q;
      if(p > 0.0)
         ent -= p * MathLog(p);
      p_min = MathMin(p_min, p);
     }
   const double z_mean = z_sum / n;
   float srm = 0;
   float mean_value = 0;
   if(!CriticSRMValue(critic, srm) || !TargetMeanValue(critic, mean_value))
      ReturnFalse;
   double w_norm = 0.0;
   if(!acsrm.GetHeadWeightNorm(w_norm))
      ReturnFalse;
   PrintFormat("ACSRM_FQF_DIAG %s y=%.8f r=%.8f z=%.8f z_min=%.8f z_max=%.8f z_mean=%.8f " +
               "srm=%.8f mean=%.8f ent=%.6f p_min=%.8f nv=%.8f np=%.8f wnorm=%.8f n=%d",
               tag, y, reward, z, z_min, z_max, z_mean, srm, mean_value,
               ent, p_min, norm_value_gradient, norm_prob_gradient, w_norm, n);
   return(MathIsValidNumber(z_min) && MathIsValidNumber(z_max) &&
          MathIsValidNumber(srm) && MathIsValidNumber(mean_value));
  }
//+-------------------------------------------------------------------------------------------------------------------------------+
//| Reads the distribution support of an ACSRM head: min/max/mean  of the quantile bins (target nets right after the bootstrap).  |
//+-------------------------------------------------------------------------------------------------------------------------------+
bool HeadDistributionRead(CNet &critic, double &z_min, double &z_max, double &z_mean)
  {
   z_min = 0.0; z_max = 0.0; z_mean = 0.0;
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   CBufferFloat *quantiles = NULL;
   CBufferFloat *probabilities = NULL;
   if(!acsrm.GetDistributionBuffers(quantiles, probabilities) ||
      quantiles.GetIndex() < 0 || probabilities.GetIndex() < 0 ||
      !quantiles.BufferRead() || !probabilities.BufferRead())
      ReturnFalse;
   const int n = quantiles.Total();
   if(n < 2 || probabilities.Total() != n)
      ReturnFalse;
   double lo = double(quantiles[0]);
   double hi = double(quantiles[0]);
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      const double q = double(quantiles[i]);
      lo = MathMin(lo, q);
      hi = MathMax(hi, q);
      sum += q;
     }
   z_min = lo; z_max = hi; z_mean = sum / n;
   return(true);
  }
//+---------------------------------------------------------------------------------------------------------------------+
//| L2 norm of the full network weights: sqrt of the sum of squared  weights over every layer's public weights matrix.  |
//+---------------------------------------------------------------------------------------------------------------------+
bool NetWeightNormL2(CNet &net, const int layer_count, double &norm)
  {
   norm = 0.0;
   double s = 0.0;
   for(int l = 0; l < layer_count; l++)
     {
      CNeuronBaseOCL *neuron = net.Layer(l);
      if(!neuron)
         ReturnFalse;
      CBufferFloat *w = neuron.getWeights();
      if(!w || w.Total() == 0 || w.GetIndex() < 0)
         continue;
      if(!w.BufferRead())
         ReturnFalse;
      for(int i = 0; i < w.Total(); i++)
         s += double(w[i]) * double(w[i]);
     }
   norm = MathSqrt(s);
   return(MathIsValidNumber(norm));
  }
//+------------------------------------------------------------------+
//| L2 gap between two same-shaped networks: ||theta_S - theta_T||.  |
//+------------------------------------------------------------------+
bool NetWeightGapL2(CNet &tgt, CNet &src, const int layer_count, double &gap)
  {
   gap = 0.0;
   if(layer_count < 1)
      ReturnFalse;
   double s = 0.0;
   int measured = 0;
   for(int l = 0; l < layer_count; l++)
     {
      CNeuronBaseOCL *tn = tgt.Layer(l);
      CNeuronBaseOCL *sn = src.Layer(l);
      if(!tn || !sn)
        { PrintFormat("ACSRM_TARGET_DIAG detail=gap_layer_null l=%d", l); return(false); }
      CBufferFloat *tw = tn.getWeights();
      CBufferFloat *sw = sn.getWeights();
      if(!tw || !sw || tw.Total() == 0 || sw.Total() == 0)
         continue;
      if(tw.Total() != sw.Total())
        { PrintFormat("ACSRM_TARGET_DIAG detail=gap_size_mismatch l=%d tgt=%d src=%d", l, tw.Total(), sw.Total()); continue; }
      if(tw.GetIndex() < 0 || sw.GetIndex() < 0)
         continue;
      if(!tw.BufferRead() || !sw.BufferRead())
        { PrintFormat("ACSRM_TARGET_DIAG detail=gap_read_fail l=%d", l); continue; }
      for(int i = 0; i < tw.Total(); i++)
        {
         const double d = double(tw[i]) - double(sw[i]);
         s += d * d;
        }
      measured++;
     }
   gap = MathSqrt(s);
   return(measured > 0 && MathIsValidNumber(gap));
  }
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Emits the target-network health report after the s_{t+1} cross-  target bootstrap pass: shadow critic distribution support, the  target weight scales and the ||theta_S - theta_T|| L2 gaps.  Best-effort diagnostics: a missing part is reported with an  ACSRM_TARGET_DIAG_PARTIAL marker and never aborts the training.  |
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool TargetDiagnosticsOnline()
  {
   double tq1_min = 0.0, tq1_max = 0.0, tq1_mean = 0.0;
   double tq2_min = 0.0, tq2_max = 0.0, tq2_mean = 0.0;
   double tq1_wn = 0.0, tq2_wn = 0.0, ta_wn = 0.0;
   double gap_actor = 0.0, gap_q1 = 0.0, gap_q2 = 0.0;
   if(!HeadDistributionRead(TargetQ1, tq1_min, tq1_max, tq1_mean) ||
      !HeadDistributionRead(TargetQ2, tq2_min, tq2_max, tq2_mean))
     { Print("ACSRM_TARGET_DIAG_PARTIAL reason=head_distribution_read"); return(true); }
   CNeuronBaseOCL *h1 = TargetQ1.Layer(4);
   CNeuronBaseOCL *h2 = TargetQ2.Layer(4);
   if(!h1 || !h2 || h1.Type() != defNeuronACSRM || h2.Type() != defNeuronACSRM)
     { Print("ACSRM_TARGET_DIAG_PARTIAL reason=head_layer"); return(true); }
   if(!((CNeuronACSRM*)h1).GetHeadWeightNorm(tq1_wn) ||
      !((CNeuronACSRM*)h2).GetHeadWeightNorm(tq2_wn))
     { Print("ACSRM_TARGET_DIAG_PARTIAL reason=head_weight_norm"); return(true); }
   if(!NetWeightNormL2(TargetActor, 4, ta_wn))
     { Print("ACSRM_TARGET_DIAG_PARTIAL reason=actor_weight_norm"); return(true); }
   if(!NetWeightGapL2(TargetActor, Actor, 4, gap_actor) ||
      !NetWeightGapL2(TargetQ1, Q2, 5, gap_q1) ||
      !NetWeightGapL2(TargetQ2, Q1, 5, gap_q2))
     { Print("ACSRM_TARGET_DIAG_PARTIAL reason=weight_gap"); return(true); }
   PrintFormat("ACSRM_TARGET_DIAG tq1_z=[%.6f,%.6f] tq1_mean=%.6f tq2_z=[%.6f,%.6f] tq2_mean=%.6f " +
               "tq1_wn=%.6f tq2_wn=%.6f ta_wn=%.6f gap_actor=%.6f gap_q1=%.6f gap_q2=%.6f",
               tq1_min, tq1_max, tq1_mean, tq2_min, tq2_max, tq2_mean,
               tq1_wn, tq2_wn, ta_wn, gap_actor, gap_q1, gap_q2);
   return(true);
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
//+----------------------------------------------------------------------+
//| Distribution MEAN of one Critic head (stable bootstrap  statistic).  |
//+----------------------------------------------------------------------+
bool TargetMeanValue(CNet &critic, float &value)
  {
   CNeuronBaseOCL *head = critic.Layer(4);
   if(!head || head.Type() != defNeuronACSRM)
      ReturnFalse;
   CNeuronACSRM *acsrm = (CNeuronACSRM*)head;
   if(!SRMValue.BufferInit(1, 0) || !SRMValue.BufferCreate(SkillMarket.GetOpenCL()) ||
      !SRMValue.BufferWrite())
      ReturnFalse;
   if(!acsrm.SRMForward(GetPointer(SRMValue), 1, defSRM_MEAN, 0.0f, 1.0f))
      ReturnFalse;
   if(!SRMValue.BufferRead())
      ReturnFalse;
   value = float(SRMValue[0]);
   return(MathIsValidNumber(value));
  }
//+------------------------------------------------------------------+
//| Selects the policy-gradient Critic by the configured selector.   |
//+------------------------------------------------------------------+
int SelectPolicyCritic(const int event_index)
  {
//--- Exactly ONE Critic is the gradient source per policy-update event.
//--- Default: balanced hash over Actor-update EVENTS, independent of the
//--- example Owner (sequential position-- would give one Owner parity).
   switch(InpSelector)
     {
      case 1:  return(0);
      case 2:  return(1);
      default: return(int((ulong(MathMax(event_index, 0)) * ulong(2654435761u) +
                           ulong(InpSeed)) & 1));
     }
  }
//+----------------------------------------------------------------------+
//| Computes ONLY the output-layer actor policy gradient  (pre-update).  |
//+----------------------------------------------------------------------+
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
//+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Refreshes the live market state from the last CLOSED bar and the  scenario (forecast) context consumed by the actor/target cross-  attention.  SkillRefreshLiveMarket fills the Rates/indicator  window; the explicit Market feed refreshes z/u/pi.  |
//+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveRefreshState(CBufferFloat *state, CBufferFloat *time_state)
  {
   if(!SkillRefreshLiveMarket(state, time_state))
      ReturnFalse;
   return(SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL) &&
          SkillForecast != NULL);
  }
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Applies the PENDING live transition (bar t -> t+1): the reward is  the LIVE equity change read on the next bar open.  The next  closed-bar context is built into dedicated Next* buffers while  State/Account still hold the bar-t context; the cross-target  bootstrap is sampled at s_{t+1} = (NextState, NextAccount), the  scenario context is then restored to s_t for the pending-tuple  critic/policy updates, and finally the runtime context is  switched to the new bar before the next action is decided.  |
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool ApplyLiveTransition(const int iteration, const double balance_now,
                         const double equity_now, double &buy_value,
                         double &sell_value)
  {
   const bool trace_td = (iteration < 64 || iteration % 10000 == 0);
   const double reward = equity_now - LivePrevEquity;
   if(!MathIsValidNumber(reward) || !MathIsValidNumber(balance_now) ||
      !MathIsValidNumber(equity_now))
      ReturnFalse;
   RewardSum += reward;
//--- 1) Build the NEXT closed-bar context into dedicated Next* buffers;
//---    State/Account still hold the bar-t context s_t.
   if(!LiveRefreshState(GetPointer(NextState), GetPointer(NextTimeState)))
      ReturnFalse;
   datetime next_time = TimeCurrent();
   if(!SkillBuildLiveAccount(LivePrevBalance, LivePrevEquity, next_time,
                             GetPointer(NextAccount), buy_value, sell_value) ||
      (NextAccount.GetIndex() < 0 && !NextAccount.BufferCreate(SkillMarket.GetOpenCL())) ||
      (NextAccount.GetIndex() >= 0 && !NextAccount.BufferWrite()))
      ReturnFalse;
//--- 2) Cross-target bootstrap at s_{t+1} = (NextState, NextAccount):
//---    SkillMarket already holds the next-bar scenario context.
   double y1 = reward;
   double y2 = reward;
   float z1 = 0.0f, z2 = 0.0f;
   if(!TargetActor.feedForward(GetPointer(NextAccount), 1, false, GetPointer(SkillMarket), -1))
     { Print("LiveTrain stage=target_actor_forward"); ReturnFalse; }
   CNeuronBaseOCL *target_context = TargetActor.Layer(0);
   CNeuronBaseOCL *target_actor_layer = TargetActor.Layer(3);
   if(!target_context || !target_actor_layer ||
      !BuildCriticInput(target_context.getOutput(), target_actor_layer.getOutput(),
                        GetPointer(TargetCriticInput)) ||
      !TargetQ1.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !TargetQ2.feedForward(GetPointer(TargetCriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !SampleTargetHead(TargetQ1, TargetSample, TargetRandom, z1) ||
      !SampleTargetHead(TargetQ2, TargetSample, TargetRandom, z2))
     { Print("LiveTrain stage=target_bootstrap"); ReturnFalse; }
   y1 = reward + double(InpGamma) * double(z1);
   y2 = reward + double(InpGamma) * double(z2);
   if(!MathIsValidNumber(y1) || !MathIsValidNumber(y2))
     { Print("LiveTrain stage=target_scalar"); ReturnFalse; }
   TargetTransitions++;
   BootstrapSum += (y1 + y2);
   if(trace_td)
     {
      PrintFormat("ACSRM online target iteration=%d r=%.8f z1=%.8f z2=%.8f y1=%.8f y2=%.8f " +
                  "terminal=false bootstrap=true",
                  iteration, reward, z1, z2, y1, y2);
//--- Best-effort target health report: diagnostics never abort the step.
      TargetDiagnosticsOnline();
     }
//--- 3) Restore the bar-t scenario context for the pending-tuple updates.
   if(!SkillMarket.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
     { Print("LiveTrain stage=restore_state_context"); ReturnFalse; }
//--- 4) Critic TD updates on the PENDING tuple (CriticInput holds the bar-t
//--- (state, action) activations retained before this bar's decision).
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1))
     { Print("LiveTrain stage=critic_forward"); ReturnFalse; }
   double nv1 = 0, np1 = 0, nv2 = 0, np2 = 0;
   if(!UpdateCriticDistributionDiag(Q1, y1, nv1, np1) ||
      !UpdateCriticDistributionDiag(Q2, y2, nv2, np2))
     { Print("LiveTrain stage=critic_backward"); ReturnFalse; }
   CriticTransitions++;
   CriticQ1Total++;
   CriticQ2Total++;
   if(trace_td)
     {
      if(!FQFDiagnosticsOnline(Q1, y1, reward, z1, nv1, np1, "q1") ||
         !FQFDiagnosticsOnline(Q2, y2, reward, z2, nv2, np2, "q2"))
        { Print("LiveTrain stage=fqf_diagnostics"); ReturnFalse; }
     }
//--- 5) Actor SRM policy update at cadence: fresh critic activations on the
//--- SAME pending tuple; the Actor output gradient was accumulated by the
//--- bar-t forward (LiveActForBar) and is consumed here.
   if(iteration > 0 && UpdatePolicy > 0 && iteration % UpdatePolicy == 0)
     {
      if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
         !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1))
        { Print("LiveTrain stage=policy_critic_forward"); ReturnFalse; }
      const int pick = SelectPolicyCritic(int(PolicyEvents++));
      CNet *policy_critic = (pick == 0 ? GetPointer(Q1) : GetPointer(Q2));
      float policy_srm = 0.0f;
      if(!CriticSRMValue(*policy_critic, policy_srm))
        {
         if(CriticLastSRMInvalid(*policy_critic))
            PolicySkippedInvalid++;
         else
           { Print("LiveTrain stage=policy_critic_srm"); ReturnFalse; }
        }
      else
        {
         const bool policy_ok = PolicyOutputGradientACSRM(Actor, *policy_critic, 1.0f);
         if(!policy_ok)
           {
            if(!CriticLastSRMInvalid(*policy_critic))
              { Print("LiveTrain stage=policy_backward"); ReturnFalse; }
            PolicySkippedInvalid++;
           }
         else
           {
            if(!Actor.backPropGradient(GetPointer(SkillMarket), -1, -1, true))
              { Print("LiveTrain stage=actor_backward"); ReturnFalse; }
            PolicyTransitions++;
           }
        }
     }
//--- 6) Cross-Polyak target update at cadence: one crossed target Critic per
//--- event (alternating) keeps the pair balanced.
   if(iteration > 0 && InpUpdateTargets > 0 && iteration % InpUpdateTargets == 0)
     {
      if(!TargetActor.WeightsUpdate(GetPointer(Actor), InpTauT))
        { Print("LiveTrain stage=target_actor_update"); ReturnFalse; }
      if((TargetUpdates & 1) == 0)
         TargetQ1First = ((MathRand() & 1) == 0);
      const bool update_q1 = (TargetQ1First ? (TargetUpdates & 1) == 0 : (TargetUpdates & 1) != 0);
      if(update_q1)
        {
         if(!TargetQ1.WeightsUpdate(GetPointer(Q2), InpTauT))
           { Print("LiveTrain stage=target_q1_update"); ReturnFalse; }
        }
      else
        {
         if(!TargetQ2.WeightsUpdate(GetPointer(Q1), InpTauT))
           { Print("LiveTrain stage=target_q2_update"); ReturnFalse; }
        }
      TargetUpdates++;
     }
//--- 7) Switch the runtime context to the new bar: State/TimeState/
//---    Account become s_{t+1} and SkillMarket is re-fed so the action
//---    decision below sees the next-bar scenario context.
   if(!State.AssignArray(GetPointer(NextState)) ||
      !TimeState.AssignArray(GetPointer(NextTimeState)) ||
      !Account.AssignArray(GetPointer(NextAccount)) ||
      (Account.GetIndex() >= 0 && !Account.BufferWrite()) ||
      !SkillMarket.feedForward(GetPointer(State), 1, false, (CBufferFloat*)NULL))
     { Print("LiveTrain stage=state_switch"); ReturnFalse; }
   return(true);
  }
//+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Decides the NEW bar and places REAL orders on the live account.  The Actor feed accumulates the output gradient that the next  bar's policy update consumes; the (state, action) tuple is  retained in CriticInput for the next-bar TD apply.  |
//+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveActForBar(const double balance_now, const double equity_now,
                   const double buy_value, const double sell_value)
  {
   const int action_iteration = LiveIteration + 1;
   const bool trace_action = (action_iteration < 64 || action_iteration % 10000 == 0);
   CNeuronBaseOCL *actor_output = Actor.Layer(3);
   if(!actor_output || actor_output.getGradient().Total() != (int)NActions ||
      !actor_output.getGradient().Fill(0) || !Actor.Clear())
      ReturnFalse;
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(CurrentAction)))
      ReturnFalse;
   CNeuronBaseOCL *context_layer = Actor.Layer(0);
   if(!context_layer)
      ReturnFalse;
   const double buy_lot = MathMax(0.0, double(CurrentAction[0] - CurrentAction[3]));
   const double sell_lot = MathMax(0.0, double(CurrentAction[3] - CurrentAction[0]));
   if(trace_action)
      PrintFormat("ACSRM online action iteration=%d balance=%.2f buy=%.8f sell=%.8f " +
                  "btp=%.4f bsl=%.4f stp=%.4f ssl=%.4f",
                  action_iteration, balance_now, buy_lot, sell_lot,
                  double(CurrentAction[1]), double(CurrentAction[2]),
                  double(CurrentAction[4]), double(CurrentAction[5]));
   if(!BuildCriticInput(context_layer.getOutput(), actor_output.getOutput(),
                        GetPointer(CriticInput)))
      ReturnFalse;
   double margin_penalty = 0;
   bool market_closed = false;
   if(!SkillExecuteAction(GetPointer(CurrentAction), buy_value, sell_value,
                          margin_penalty, market_closed))
      ReturnFalse;
//--- Post-execution LIVE snapshot: the reward of THIS bar is evaluated
//--- from it on the next bar open.
   LivePrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   LivePrevEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
   LiveIteration = action_iteration;
   LiveSteps++;
   return(true);
  }
//+-------------------------------------------------------------------------------------------------------------------------+
//| Emits the final online report once (PASS on completed training,  STOP otherwise) and requests the EA to remove itself.  |
//+-------------------------------------------------------------------------------------------------------------------------+
bool LiveFinish(const bool completed)
  {
   if(LiveFinalized)
      return(true);
   LiveFinalized = true;
   if(completed)
     {
      if(!SavePolicies())
        {
         PrintFormat("ACSRM_ONLINE_CHECKPOINT_FAIL error=%d", GetLastError());
         return(false);
        }
      PrintFormat("ACSRM_ONLINE_TRAIN_PASS iterations=%d critic=%I64u policy=%I64u skip=%I64u " +
                  "target_pairs=%I64u target_events=%I64u target_transitions=%I64u " +
                  "q1_total=%I64u q2_total=%I64u published=canonical",
                  LiveSteps, CriticTransitions, PolicyTransitions, PolicySkippedInvalid,
                  TargetUpdates / 2, TargetUpdates % 2,
                  TargetTransitions, CriticQ1Total, CriticQ2Total);
     }
   else
     {
      PrintFormat("ACSRM_ONLINE_STOP iteration=%d completed=%d critic=%I64u policy=%I64u target=%I64u",
                  LiveIteration, LiveSteps, CriticTransitions, PolicyTransitions, TargetUpdates);
     }
   LiveStopRequested = true;
   return(true);
  }
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| One new-bar online step: LIVE reward of the pending transition,  cross-target bootstrap + critic/actor/target updates, then the  new bar decision with REAL orders.  |
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveOnlineStep(void)
  {
   if(!LiveReady)
     {
//--- First stream bar: no previous transition exists.
      LiveReady = true;
      LiveIteration = -1;
      LivePrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      LivePrevEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
      if(!LiveRefreshState(GetPointer(State), GetPointer(TimeState)))
         ReturnFalse;
      datetime now = TimeCurrent();
      double buy_value = 0, sell_value = 0;
      if(!SkillBuildLiveAccount(LivePrevBalance, LivePrevEquity, now,
                                GetPointer(Account), buy_value, sell_value) ||
         (Account.GetIndex() < 0 && !Account.BufferCreate(SkillMarket.GetOpenCL())) ||
         (Account.GetIndex() >= 0 && !Account.BufferWrite()))
         ReturnFalse;
      return(LiveActForBar(LivePrevBalance, LivePrevEquity, buy_value, sell_value));
     }
//--- LIVE result of the just-closed bar and the new-bar market context.
   const double balance_now = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity_now  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(balance_now <= MinBalance)
     {
      PrintFormat("%s online live terminal balance=%.2f equity=%.2f steps=%d",
                  ACSRM_LOG_PREFIX, balance_now, equity_now, LiveSteps);
      CloseByDirection(POSITION_TYPE_BUY);
      CloseByDirection(POSITION_TYPE_SELL);
      return(LiveFinish(true));
     }
//--- Apply the PENDING transition with the LIVE reward (iteration index
//--- of the action whose bar just closed).  The transition builds the
//--- next-bar context into State/Account and returns the margin values
//--- needed by the new-bar decision below.
   double buy_value = 0, sell_value = 0;
   if(LiveIteration >= 0)
     {
      if(!ApplyLiveTransition(LiveIteration, balance_now, equity_now,
                              buy_value, sell_value))
         ReturnFalse;
     }
   return(LiveActForBar(balance_now, equity_now, buy_value, sell_value));
  }
//+------------------------------------------------------------------+
//| Saves the Stage 03 Actor + Q1 + Q2 stop checkpoint.              |
//+------------------------------------------------------------------+
bool SavePolicies(void)
  {
   return(SkillSaveStage03StopCheckpoint(Actor, Q1, Q2));
  }
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Loads the ADAM shadow targets from the same .nnw set as the  trained policies (identical optimizer-dependent structure) and  crosses them via tau=1.0 hard copy: TargetActor<-Actor,  TargetQ1<-Q2, TargetQ2<-Q1.  Runtime synchronization re-applies  the same crossed hard copies every InpUpdateTargets transitions;  tau<1 with ADAM targets is NOT used (would re-enter the ADAM  WeightsUpdateAdam path and re-introduce divergence).  |
//+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LoadOnlineTargets(void)
  {
   if(!SkillLoadPolicyNet(TargetActor, Skill_ACTOR_FILE) ||
      !SkillLoadPolicyNet(TargetQ1, Skill_Q1_FILE) ||
      !SkillLoadPolicyNet(TargetQ2, Skill_Q2_FILE) ||
      !SkillBindPolicyOpenCLChecked(TargetActor, TargetQ1, TargetQ2) ||
      !ValidatePolicyShapeACSRM(TargetActor, TargetQ1, TargetQ2))
      ReturnFalse;
//--- Crossed hard initialization: TargetActor <- Actor, TargetQ1 <- Q2,
//--- TargetQ2 <- Q1 (deliberate divergence from direct mirroring).
   if(!TargetActor.WeightsUpdate(GetPointer(Actor), 1.0f) ||
      !TargetQ1.WeightsUpdate(GetPointer(Q2), 1.0f) ||
      !TargetQ2.WeightsUpdate(GetPointer(Q1), 1.0f))
      ReturnFalse;
//--- Frozen targets: no training writes, explicit hard copies only.
   TargetActor.TrainMode(false);
   TargetQ1.TrainMode(false);
   TargetQ2.TrainMode(false);
   if(!TargetActor.SetWeightsUpdate(false) || !TargetQ1.SetWeightsUpdate(false) ||
      !TargetQ2.SetWeightsUpdate(false))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Initializes the Stage 03 Online cross-target Expert.             |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
//--- Pin the deal horizon and the online TD discount for manifest symmetry.
   SkillManifestDealHorizon = MathMax(1, InpDealHorizon);
   SkillManifestOnlineDiscount = double(InpGamma);
   SkillManifestTargetTau = double(InpTauT);
   SkillManifestTargetUpdatePeriod = InpUpdateTargets;
   if(InpSelector < 0 || InpSelector > 2 ||
      InpUpdateTargets < 0 || InpTauT < 0.0f || InpTauT > 1.0f ||
      InpGamma <= 0.0f || InpGamma > 1.0f)
     {
      PrintFormat("ACSRM online input validation failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
//--- Confirm Stage 03, the frozen Forecast checkpoint and the policy tuple.
   if(InpOMPBStage != OMPB_STAGE_BASE_POLICY ||
      !SkillInitIndicators() || !SkillLoadForecastInference() ||
      !SkillConfigureProductionACSRMCheckpoint() ||
      !SkillValidateProductionACSRMCheckpoint() ||
      !SkillCaptureProductionACSRMFingerprints() || !LoadOrCreatePolicies() ||
      !LoadOnlineTargets() ||
      !SkillVerifyFrozenForecastExact() ||
      !SkillVerifyProductionACSRMFingerprints())
     {
      PrintFormat("ACSRM online initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
   PrintFormat("ACSRM_ONLINE_INIT_PASS tester=%d", MQLInfoInteger(MQL_TESTER));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(MQLInfoInteger(MQL_TESTER) == 1 && !LiveFinalized)
     {
      PrintFormat("%s live finalize reason=%d steps=%d", ACSRM_LOG_PREFIX, reason, LiveSteps);
      LiveFinish(true);
     }
//--- Online does not publish; verify the production tuple was untouched.
   if(SkillForecast != NULL && !SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   if(SkillProductionSignatureReady && !SkillVerifyProductionACSRMFingerprints())
      PrintFormat("%s -> %d ACSRM production signature mutation", __FUNCTION__, __LINE__);
   SkillForecast = NULL;
  }
//+-----------------------------------------------------------------------------------------+
//| LIVE stream training: one online transition per new bar on the  tester account stream.  |
//+-----------------------------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
   if(!LiveOnlineStep())
     {
      PrintFormat("ACSRM_ONLINE_FAIL reason=live_step_failed steps=%d", LiveSteps);
      ExpertRemove();
      return;
     }
   if(LiveStopRequested)
      ExpertRemove();
  }
//+---
//+------------------------------------------------------------------+