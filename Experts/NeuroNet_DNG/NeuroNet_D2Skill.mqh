//+------------------------------------------------------------------+
//|                                              NeuroNet_D2Skill.mqh |
//|                     GPU-resident D2Skill primitives              |
//+------------------------------------------------------------------+
#ifndef NEURONET_D2SKILL_MQH
#define NEURONET_D2SKILL_MQH
//+------------------------------------------------------------------+
//--- Reuse boundary: D2Skill keeps its own flat GPU bank because its
//--- signed corrections, direction agreement, utility/age eviction,
//--- candidate promotion and Task/Step dual routing are not represented by
//--- CScenarioCodebook.  RAG score/Top-K and persistence conventions are
//--- reused where their contracts match; CNeuronRAGMemory is intentionally
//--- not used because its publish path is CPU copy-on-write plus re-upload.
//+------------------------------------------------------------------+
enum ED2SkillRepresentation
  {
   D2SkillFullResidual = 0,
   D2SkillDirectionMagnitude = 1
  };
//+------------------------------------------------------------------+
//| One reusable latent correction.                                  |
//+------------------------------------------------------------------+
class CD2SkillItem : public CNeuronBaseOCL
  {
protected:
   CBufferFloat                *m_correction;
   ED2SkillRepresentation       m_representation;
   float                        m_beta_correction;
   float                        m_beta_direction;
   float                        m_beta_magnitude;
   uint                         m_item_dimension;
   CBufferFloat                 m_item_correction;
   CBufferFloat                 m_item_direction;
   CBufferFloat                 m_item_scale;
   CBufferFloat                 m_item_observations;
   CBufferFloat                 m_item_mass;
   CBufferFloat                 m_item_partials;
   CBufferFloat                 m_item_reduced_norm;

   bool              EnsureItemState(void);
   bool              ItemLaunchLayout(uint &local_size, uint &groups,
                                      uint &partial_count) const;
   bool              EnsureItemScratch(void);
   bool              CalculateItemNorm(CBufferFloat *source, const uint local_size,
                                       const uint groups);
   void              RebindItemBuffer(CBufferFloat &buffer, COpenCLMy *obj)
     {
      if(!obj)
         return;
      if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
         buffer.BufferFree();
      if(buffer.Total() > 0)
         buffer.BufferCreate(obj);
     }

public:
                     CD2SkillItem(void) : m_correction(NULL),
                     m_representation(D2SkillDirectionMagnitude),
                     m_beta_correction(0.05f), m_beta_direction(0.05f),
                     m_beta_magnitude(0.05f), m_item_dimension(0) {}
   virtual            ~CD2SkillItem(void) {}

   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint numNeurons, ENUM_OPTIMIZATION optimization_type,
                          uint batch) override
     {
      m_correction = NULL;
      m_item_dimension = numNeurons;
      if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons,
                               optimization_type, batch))
         return(false);
      return(EnsureItemState());
     }

   bool              SetCorrectionBuffer(CBufferFloat *correction)
     {
      if(!correction || CheckPointer(correction) == POINTER_INVALID)
        {
         m_correction = NULL;
         return(true);
        }
      if(!Output || correction.Total() != Output.Total() ||
         correction.GetOpenCL() != OpenCL)
         return(false);
      m_correction = correction;
      return(true);
     }
   CBufferFloat      *GetCorrectionBuffer(void) const { return(m_correction); }

   void              SetRepresentation(const ED2SkillRepresentation mode)
     {
      m_representation = mode;
     }
   ED2SkillRepresentation Representation(void) const { return(m_representation); }
   void              SetEMA(const float beta_correction, const float beta_direction,
                            const float beta_magnitude)
     {
      m_beta_correction = MathMin(MathMax(beta_correction, 0.0f), 1.0f);
      m_beta_direction = MathMin(MathMax(beta_direction, 0.0f), 1.0f);
      m_beta_magnitude = MathMin(MathMax(beta_magnitude, 0.0f), 1.0f);
     }
   float             BetaCorrection(void) const { return(m_beta_correction); }
   float             BetaDirection(void) const { return(m_beta_direction); }
   float             BetaMagnitude(void) const { return(m_beta_magnitude); }

   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   bool              Load(const int file_handle, COpenCLMy *opencl);
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   bool              UpdateFromGradient(CBufferFloat *gradient);
   CBufferFloat      *Correction(void) { return(m_correction); }
   CBufferFloat      *Direction(void) { return(GetPointer(m_item_direction)); }
   CBufferFloat      *Scale(void) { return(GetPointer(m_item_scale)); }
   CBufferFloat      *Observations(void) { return(GetPointer(m_item_observations)); }
   CBufferFloat      *Mass(void) { return(GetPointer(m_item_mass)); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override
     {
      return(true);
     }
   virtual int       Type(void) const override { return(defNeuronD2SkillItem); }
  };
//+------------------------------------------------------------------+
//| Flat, bounded GPU-resident bank of CD2SkillItem slots.            |
//+------------------------------------------------------------------+
class CD2SkillBank : public CD2SkillItem
  {
protected:
   uint                         m_slots;
   uint                         m_dimension;
   bool                         m_enabled;
   float                        m_similarity_threshold;
   float                        m_direction_threshold;
   float                        m_utility_weight;
   float                        m_alpha;
   float                        m_min_utility;
   bool                         m_utility_aware;
   float                        m_beta_key;
   float                        m_beta_utility;
   float                        m_max_correction;
   uint                         m_min_confirmations;
   uint                         m_protection_age;
   uint                         m_inactivity_age;
   uint                         m_active_slot;
   CBufferFloat                 m_keys;
   CBufferFloat                 m_corrections;
   CBufferFloat                 m_directions;
   CBufferFloat                 m_scales;
   CBufferFloat                 m_utility;
   CBufferFloat                 m_observations;
   CBufferFloat                 m_uses;
   CBufferFloat                 m_mass;
   CBufferFloat                 m_used;
   CBufferFloat                 m_state;
   CBufferFloat                 m_age;
   CBufferFloat                 m_protection;
   CBufferFloat                 m_candidate_streak;
   CBufferFloat                 m_candidate_valid;
   CBufferFloat                 m_candidate_key;
   CBufferFloat                 m_candidate_direction;
   CBufferFloat                 m_candidate_scale;
   CBufferFloat                 m_selected_correction;
   CBufferFloat                 m_selected_slot;
   CBufferFloat                 m_selected_score;
   CBufferFloat                 m_diagnostics;
   CBufferFloat                 m_utility_distribution;
   CBufferFloat                 m_usage_distribution;
   CBufferFloat                 m_retrieval_scores;
   CBufferFloat                 m_retrieval_partial;
   CBufferFloat                 m_retrieval_merge;
   CBufferFloat                 m_retrieval_top;
   CBufferFloat                 m_free_scores;
   CBufferFloat                 m_eviction_scores;
   CBufferFloat                 m_free_top;
   CBufferFloat                 m_eviction_top;
   CBufferFloat                 m_metric_partials;
   CBufferFloat                 m_metrics;
   CBufferFloat                 m_decision;
   CBufferFloat                 m_diagnostics_partials;

   void              RebindBuffer(CBufferFloat &buffer, COpenCLMy *obj)
     {
      if(!obj)
         return;
      if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
         buffer.BufferFree();
      buffer.BufferCreate(obj);
     }

   bool              ValidateBuffers(void)
     {
      const long total = (long)m_slots * (long)m_dimension;
      if(total <= 0 || total > 2147483647 ||
         !Output || !Gradient || Output.Total() != (int)m_dimension ||
         Gradient.Total() != (int)m_dimension)
         return(false);
      return(m_keys.Total() == (int)total &&
             m_corrections.Total() == (int)total &&
             m_directions.Total() == (int)total &&
             m_scales.Total() == (int)m_slots &&
             m_utility.Total() == (int)m_slots &&
             m_observations.Total() == (int)m_slots &&
             m_uses.Total() == (int)m_slots &&
             m_mass.Total() == (int)m_slots &&
             m_used.Total() == (int)m_slots &&
             m_state.Total() == (int)m_slots &&
             m_age.Total() == (int)m_slots &&
             m_protection.Total() == (int)m_slots &&
             m_candidate_streak.Total() == 1 &&
             m_candidate_valid.Total() == 1 &&
             m_candidate_key.Total() == (int)m_dimension &&
             m_candidate_direction.Total() == (int)m_dimension &&
             m_candidate_scale.Total() == 1 &&
             m_selected_correction.Total() == (int)m_dimension &&
             m_selected_slot.Total() == 1 &&
             m_selected_score.Total() == 1 &&
             m_diagnostics.Total() == D2SKILL_DIAGNOSTICS &&
             m_utility_distribution.Total() == D2SKILL_DISTRIBUTION_BINS &&
             m_usage_distribution.Total() == D2SKILL_DISTRIBUTION_BINS);
     }

   bool              InitBuffer(CBufferFloat &buffer, const int total, const float value = 0.0f)
     {
      return(buffer.BufferInit(total, value) && buffer.BufferCreate(OpenCL));
     }
   bool              RetrievalLaunchLayout(uint &local_size, uint &partial_count) const;
   bool              EnsureRetrievalBuffer(CBufferFloat &buffer, const int total);
   bool              EnsureRetrievalScratch(void);
   bool              UpdateLaunchLayout(uint &local_size, uint &groups) const;
   bool              EnsureUpdateScratch(void);
   bool              RunTop1(CBufferFloat &scores, CBufferFloat &top);
   bool              RunMetrics(CBufferFloat *state, CBufferFloat *gradient,
                                const uint local_size, const uint groups);
   bool              CalculateBankVectorNorm(const uint local_size, const uint groups);
   bool              Retrieve(CBufferFloat *state);

public:
                     CD2SkillBank(void) : m_slots(0), m_dimension(0), m_enabled(false),
                     m_similarity_threshold(0.65f), m_direction_threshold(0.0f),
                     m_utility_weight(0.05f), m_beta_key(0.05f), m_beta_utility(0.05f),
                     m_alpha(1.0f), m_min_utility(-1.0f), m_utility_aware(false),
                     m_max_correction(10.0f), m_min_confirmations(3), m_protection_age(32),
                     m_inactivity_age(256),
                     m_active_slot(UINT_MAX) {}
   virtual            ~CD2SkillBank(void) {}

   virtual bool      Init(uint dimension, uint myIndex, COpenCLMy *open_cl,
                          uint slots, ENUM_OPTIMIZATION optimization_type,
                          uint batch) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   bool              Load(const int file_handle, COpenCLMy *opencl);
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   virtual int       Type(void) const override { return(defNeuronD2SkillBank); }

   bool              SetEnabled(const bool enabled) { m_enabled = enabled; return(true); }
   bool              Enabled(void) const { return(m_enabled); }
   bool              SetThresholds(const float similarity, const float direction)
     {
      if(!MathIsValidNumber(similarity) || !MathIsValidNumber(direction))
         return(false);
      m_similarity_threshold = similarity;
      m_direction_threshold = direction;
      return(true);
     }
   bool              SetUtilityWeight(const float value)
     {
      if(!MathIsValidNumber(value))
         return(false);
      m_utility_weight = value;
      return(true);
     }
   bool              SetAlpha(const float value)
     {
      if(!MathIsValidNumber(value) || value < 0.0f || value > 1.0f)
         return(false);
      m_alpha = value;
      return(true);
     }
   float             Alpha(void) const { return(m_alpha); }
   bool              SetUtilityPolicy(const bool utility_aware, const float min_utility = 0.0f)
     {
      if(!MathIsValidNumber(min_utility) || min_utility < -1.0f || min_utility > 1.0f)
         return(false);
      m_utility_aware = utility_aware;
      m_min_utility = min_utility;
      return(true);
     }
   bool              UtilityAware(void) const { return(m_utility_aware); }
   float             MinUtility(void) const { return(m_min_utility); }
   bool              SetLifecycle(const uint confirmations, const uint protection_age,
                                  const uint inactivity_age = 256)
     {
      if(confirmations == 0)
         return(false);
      m_min_confirmations = confirmations;
      m_protection_age = protection_age;
      m_inactivity_age = MathMax(inactivity_age, protection_age + 1);
      return(true);
     }
   bool              SetMaxCorrection(const float value)
     {
      if(!MathIsValidNumber(value) || value < 0.0f)
         return(false);
      m_max_correction = value;
      return(true);
     }
   float             MaxCorrection(void) const { return(m_max_correction); }
   bool              UpdateFromGradient(CBufferFloat *state, CBufferFloat *gradient);
   bool              UpdateUtility(const float delta_j);
   uint              Slots(void) const { return(m_slots); }
   uint              Dimension(void) const { return(m_dimension); }
   uint              ActiveSlot(void) const { return(m_active_slot); }
   CBufferFloat      *SelectedCorrection(void) { return(GetPointer(m_selected_correction)); }
   CBufferFloat      *SelectedSlot(void) { return(GetPointer(m_selected_slot)); }
   CBufferFloat      *SelectedScore(void) { return(GetPointer(m_selected_score)); }
   CBufferFloat      *Keys(void) { return(GetPointer(m_keys)); }
   CBufferFloat      *Corrections(void) { return(GetPointer(m_corrections)); }
   CBufferFloat      *Directions(void) { return(GetPointer(m_directions)); }
   CBufferFloat      *Scales(void) { return(GetPointer(m_scales)); }
   CBufferFloat      *ObservationsBank(void) { return(GetPointer(m_observations)); }
   CBufferFloat      *Uses(void) { return(GetPointer(m_uses)); }
   CBufferFloat      *MassBank(void) { return(GetPointer(m_mass)); }
   CBufferFloat      *Utility(void) { return(GetPointer(m_utility)); }
   CBufferFloat      *Used(void) { return(GetPointer(m_used)); }
   CBufferFloat      *State(void) { return(GetPointer(m_state)); }
   CBufferFloat      *Ages(void) { return(GetPointer(m_age)); }
   CBufferFloat      *Protection(void) { return(GetPointer(m_protection)); }
   CBufferFloat      *Diagnostics(void) { return(GetPointer(m_diagnostics)); }
   CBufferFloat      *UtilityDistribution(void)
     {
      return(GetPointer(m_utility_distribution));
     }
   CBufferFloat      *UsageDistribution(void)
     {
      return(GetPointer(m_usage_distribution));
     }
   CBufferFloat      *CandidateValid(void) { return(GetPointer(m_candidate_valid)); }
   CBufferFloat      *CandidateStreak(void) { return(GetPointer(m_candidate_streak)); }
   CBufferFloat      *CandidateKey(void) { return(GetPointer(m_candidate_key)); }
   CBufferFloat      *CandidateDirection(void) { return(GetPointer(m_candidate_direction)); }
   CBufferFloat      *CandidateScale(void) { return(GetPointer(m_candidate_scale)); }
   bool              RefreshDiagnostics(void);
  };
//+------------------------------------------------------------------+
//| Top-level D2Skill coordinator.  It owns exactly two banks.        |
//+------------------------------------------------------------------+
class CD2Skill : public CNeuronScenarioCrossAttention
  {
private:
   CD2SkillBank                 m_task_bank;
   CD2SkillBank                 m_step_bank;
   bool                         m_ready;

protected:
   virtual bool      ApplyTaskSkill(CNeuronBaseOCL *source) override;
   virtual bool      ApplyStepSkill(CNeuronBaseOCL *source) override;
   virtual CNeuronBaseOCL *TaskSkillOutput(CNeuronBaseOCL *source) override;
   virtual CNeuronBaseOCL *StepSkillOutput(CNeuronBaseOCL *source) override;
   virtual bool      UpdateTaskSkill(CBufferFloat *state, CBufferFloat *gradient) override;
   virtual bool      UpdateStepSkill(CBufferFloat *state, CBufferFloat *gradient) override;
   virtual bool      RouteTaskGradient(void) override;
   virtual bool      RouteStepGradient(void) override;
   bool              InitBanks(ENUM_OPTIMIZATION optimization_type, uint batch);

public:
                     CD2Skill(void) : m_ready(false) {}
   virtual            ~CD2Skill(void) {}
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *opencl,
                          uint scenarios = 21, uint horizon = 12, uint latent = 16,
                          bool critic = false, ENUM_OPTIMIZATION optimization_type = ADAM,
                          uint batch = 1, uint variables = 1, uint stack_size = 1,
                          uint top_k = 1) override;
   virtual int       Type(void) const override { return(defNeuronD2Skill); }
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   bool              Load(const int file_handle, COpenCLMy *opencl);
   virtual void      SetOpenCL(COpenCLMy *opencl) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   bool              Ready(void) const { return(m_ready); }
   CD2SkillBank      *TaskBank(void) { return(m_ready ? GetPointer(m_task_bank) : NULL); }
   CD2SkillBank      *StepBank(void) { return(m_ready ? GetPointer(m_step_bank) : NULL); }
   CD2SkillBank      *Bank(void) { return(TaskBank()); }
   const CD2SkillBank *Bank(void) const { return(TaskBank()); }
   const CD2SkillBank *TaskBank(void) const { return(m_ready ? GetPointer(m_task_bank) : NULL); }
   const CD2SkillBank *StepBank(void) const { return(m_ready ? GetPointer(m_step_bank) : NULL); }
   bool              Enable(const bool task, const bool step)
     {
      if(!m_ready)
         return(false);
      m_task_bank.SetEnabled(task);
      m_step_bank.SetEnabled(step);
      return(true);
     }
   bool              TaskFeedForward(CNeuronBaseOCL *source)
     {
      return(m_ready && m_task_bank.FeedForward(source));
     }
   bool              StepFeedForward(CNeuronBaseOCL *source)
     {
      return(m_ready && m_step_bank.FeedForward(source));
     }
   bool              TaskUpdate(CBufferFloat *state, CBufferFloat *gradient)
     {
      return(m_ready && m_task_bank.UpdateFromGradient(state, gradient));
     }
   bool              StepUpdate(CBufferFloat *state, CBufferFloat *gradient)
     {
      return(m_ready && m_step_bank.UpdateFromGradient(state, gradient));
     }
   bool              TaskUtility(const float delta_j)
     {
      return(m_ready && m_task_bank.UpdateUtility(delta_j));
     }
   bool              StepUtility(const float delta_j)
     {
      return(m_ready && m_step_bank.UpdateUtility(delta_j));
     }
   bool              BackwardSecond(CNeuronBaseOCL *source, CNeuronBaseOCL *forecast)
     {
      return(m_ready && calcInputGradientsSecond(source, forecast));
     }
  };
//+------------------------------------------------------------------+
bool CD2Skill::Init(uint numOutputs, uint myIndex, COpenCLMy *opencl,
                    uint scenarios, uint horizon, uint latent, bool critic,
                    ENUM_OPTIMIZATION optimization_type, uint batch,
                    uint variables, uint stack_size, uint top_k)
  {
   if(critic)
      return(false);
   m_ready = false;
   if(!CNeuronScenarioCrossAttention::Init(numOutputs, myIndex, opencl,
                                           scenarios, horizon, latent, false,
                                           optimization_type, batch, variables,
                                           stack_size, top_k))
      return(false);
   if(!InitBanks(optimization_type, batch))
      return(false);
   m_task_bank.SetEnabled(false);
   m_step_bank.SetEnabled(false);
   m_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2Skill::InitBanks(ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   const uint task_dimension = iQueries * iScenarios * iDimension;
   const uint step_dimension = iScenarios * iVariables * iHorizon * iDimension;
   if(task_dimension == 0 || step_dimension == 0 || !OpenCL)
      return(false);
   if(!m_task_bank.Init(task_dimension, 100000, OpenCL, iScenarios,
                        optimization_type, batch))
      return(false);
   if(!m_step_bank.Init(step_dimension, 100001, OpenCL, iScenarios,
                        optimization_type, batch))
      return(false);
   m_task_bank.SetActivationFunction(None);
   m_step_bank.SetActivationFunction(None);
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2Skill::ApplyTaskSkill(CNeuronBaseOCL *source)
  {
   return(m_ready && m_task_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
bool CD2Skill::ApplyStepSkill(CNeuronBaseOCL *source)
  {
   return(m_ready && m_step_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
CNeuronBaseOCL *CD2Skill::TaskSkillOutput(CNeuronBaseOCL *source)
  {
   return(m_ready ? m_task_bank.AsObject() : source);
  }
//+------------------------------------------------------------------+
CNeuronBaseOCL *CD2Skill::StepSkillOutput(CNeuronBaseOCL *source)
  {
   return(m_ready ? m_step_bank.AsObject() : source);
  }
//+------------------------------------------------------------------+
bool CD2Skill::UpdateTaskSkill(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(!m_ready || !m_task_bank.Enabled() ||
          m_task_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
bool CD2Skill::UpdateStepSkill(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(!m_ready || !m_step_bank.Enabled() ||
          m_step_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
bool CD2Skill::RouteTaskGradient(void)
  {
   if(!m_ready || !m_task_bank.Enabled())
      return(CNeuronScenarioCrossAttention::RouteTaskGradient());
//--- The bank is an identity-with-correction edge between the query
//--- projection and attention.  First calculate dL/d(bank_output) from
//--- attention, then propagate that gradient through the bank to the query
//--- projection.  The bank itself has no trainable weights.
   if(!m_task_bank.CalcHiddenGradients(cAttention.AsObject(), cKV.getOutput(),
                                       cKV.getGradient(), None))
      return(false);
   return(cQueryProjection.CalcHiddenGradients(m_task_bank.AsObject()));
  }
//+------------------------------------------------------------------+
bool CD2Skill::RouteStepGradient(void)
  {
   if(!m_ready || !m_step_bank.Enabled())
      return(CNeuronScenarioCrossAttention::RouteStepGradient());
   if(!m_step_bank.CalcHiddenGradients(cTokensTranspose.AsObject()))
      return(false);
   return(cTokenPE.CalcHiddenGradients(m_step_bank.AsObject()));
  }
//+------------------------------------------------------------------+
bool CD2Skill::Save(const int file_handle)
  {
   if(!m_ready || !CNeuronScenarioCrossAttention::Save(file_handle))
      return(false);
   dFileWriteUInt(file_handle, D2SKILL_LAYER_FORMAT);
   return(m_task_bank.Save(file_handle) && m_step_bank.Save(file_handle));
  }
//+------------------------------------------------------------------+
bool CD2Skill::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronScenarioCrossAttention::Load(file_handle))
      return(false);
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_LAYER_FORMAT || bCritic || !OpenCL)
      return(false);
   m_ready = false;
   m_task_bank.SetOpenCL(OpenCL);
   m_step_bank.SetOpenCL(OpenCL);
   if(FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank ||
      !m_task_bank.Load(file_handle) ||
      FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank ||
      !m_step_bank.Load(file_handle))
      return(false);
   const uint task_dimension = iQueries * iScenarios * iDimension;
   const uint step_dimension = iScenarios * iVariables * iHorizon * iDimension;
   if(m_task_bank.Dimension() != task_dimension || m_task_bank.Slots() != iScenarios ||
      m_step_bank.Dimension() != step_dimension || m_step_bank.Slots() != iScenarios)
      return(false);
   m_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2Skill::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl || FileReadInteger(file_handle) != Type())
      return(false);
   OpenCL = opencl;
   return(Load(file_handle));
  }
//+------------------------------------------------------------------+
void CD2Skill::SetOpenCL(COpenCLMy *opencl)
  {
   CNeuronScenarioCrossAttention::SetOpenCL(opencl);
   m_task_bank.SetOpenCL(opencl);
   m_step_bank.SetOpenCL(opencl);
  }
//+------------------------------------------------------------------+
void CD2Skill::TrainMode(bool flag)
  {
   CNeuronScenarioCrossAttention::TrainMode(flag);
   m_task_bank.TrainMode(flag);
   m_step_bank.TrainMode(flag);
  }
//+------------------------------------------------------------------+
bool CD2Skill::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      return(false);
   CD2Skill *Source = source;
   if(!CNeuronScenarioCrossAttention::WeightsUpdate(source, tau))
      return(false);
   return(m_task_bank.WeightsUpdate(Source.TaskBank(), tau) &&
          m_step_bank.WeightsUpdate(Source.StepBank(), tau));
  }
//+------------------------------------------------------------------+
bool CD2Skill::Clear(void)
  {
   m_ready = false;
   return(CNeuronScenarioCrossAttention::Clear() &&
          m_task_bank.Clear() && m_step_bank.Clear() && (m_ready = true));
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Output) == POINTER_INVALID)
      return(false);
   CBufferFloat *source = NeuronOCL.getOutput();
   if(!source || source.Total() != Output.Total() || source.GetOpenCL() != OpenCL)
      return(false);
   if(CheckPointer(m_correction) == POINTER_INVALID)
      return(SumAndNormalize(source, source, Output, source.Total(), false,
                             0, 0, 0, 0.5f));
   return(SumAndNormalize(source, m_correction, Output, source.Total(), false,
                          0, 0, 0, 1.0f));
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Gradient) == POINTER_INVALID)
      return(false);
   CBufferFloat *source_gradient = NeuronOCL.getGradient();
   if(!source_gradient || source_gradient.Total() != Gradient.Total() ||
      source_gradient.GetOpenCL() != OpenCL)
      return(false);
//--- A skill item has no weights: dL/dx receives dL/dy.  Preserve any
//--- gradient already accumulated by another branch (for example the
//--- Actor history path) and add the bank output gradient on the device.
   return(SumAndNormalize(source_gradient, Gradient, source_gradient,
                          Gradient.Total(), false, 0, 0, 0, 1.0f));
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::EnsureItemState(void)
  {
   if(!OpenCL || m_item_dimension == 0)
      return(false);
   if(m_item_correction.Total() != (int)m_item_dimension &&
      !m_item_correction.BufferInit((int)m_item_dimension, 0.0f))
      return(false);
   if((m_item_correction.GetOpenCL() != OpenCL || m_item_correction.GetIndex() < 0) &&
      !m_item_correction.BufferCreate(OpenCL))
      return(false);
   if(m_correction == NULL)
     {
      m_correction = GetPointer(m_item_correction);
     }
   if(m_item_direction.Total() != (int)m_item_dimension &&
      !m_item_direction.BufferInit((int)m_item_dimension, 0.0f))
      return(false);
   if((m_item_direction.GetOpenCL() != OpenCL || m_item_direction.GetIndex() < 0) &&
      !m_item_direction.BufferCreate(OpenCL))
      return(false);
   if(m_item_scale.Total() != 1 && !m_item_scale.BufferInit(1, 0.0f))
      return(false);
   if((m_item_scale.GetOpenCL() != OpenCL || m_item_scale.GetIndex() < 0) &&
      !m_item_scale.BufferCreate(OpenCL))
      return(false);
   if(m_item_observations.Total() != 1 && !m_item_observations.BufferInit(1, 0.0f))
      return(false);
   if((m_item_observations.GetOpenCL() != OpenCL || m_item_observations.GetIndex() < 0) &&
      !m_item_observations.BufferCreate(OpenCL))
      return(false);
   if(m_item_mass.Total() != 1 && !m_item_mass.BufferInit(1, 0.0f))
      return(false);
   if((m_item_mass.GetOpenCL() != OpenCL || m_item_mass.GetIndex() < 0) &&
      !m_item_mass.BufferCreate(OpenCL))
      return(false);
   return(m_correction && m_correction.GetOpenCL() == OpenCL &&
           m_correction.Total() == (int)m_item_dimension &&
           m_correction.GetIndex() >= 0 && m_item_direction.GetIndex() >= 0 &&
           m_item_scale.GetIndex() >= 0 && m_item_observations.GetIndex() >= 0 &&
           m_item_mass.GetIndex() >= 0 && EnsureItemScratch());
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::ItemLaunchLayout(uint &local_size, uint &groups,
                                    uint &partial_count) const
  {
   if(!OpenCL || m_item_dimension == 0)
      return(false);
   long limit = 64;
   limit = MathMin(limit, OpenCL.GetMaxWorkGroupSize());
   limit = MathMin(limit, OpenCL.GetMaxLocalSize(0));
   if(limit <= 0)
      return(false);
   local_size = 1;
   while(local_size <= (uint)limit / 2)
      local_size *= 2;
   ulong count = ((ulong)m_item_dimension + local_size - 1) / local_size;
   ulong total = count;
   groups = (uint)count;
   while(count > 1)
     {
      count = (count + 2 * local_size - 1) / (2 * local_size);
      if(count > 1)
         total += count;
     }
   if(total == 0 || total > 2147483647)
      return(false);
   partial_count = (uint)total;
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::EnsureItemScratch(void)
  {
   uint local_size = 0;
   uint groups = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count))
      return(false);
   if(m_item_partials.Total() != (int)partial_count &&
      !m_item_partials.BufferInit(partial_count, 0.0f))
      return(false);
   if((m_item_partials.GetOpenCL() != OpenCL || m_item_partials.GetIndex() < 0) &&
      !m_item_partials.BufferCreate(OpenCL))
      return(false);
   if(m_item_reduced_norm.Total() != 1 && !m_item_reduced_norm.BufferInit(1, 0.0f))
      return(false);
   if((m_item_reduced_norm.GetOpenCL() != OpenCL || m_item_reduced_norm.GetIndex() < 0) &&
      !m_item_reduced_norm.BufferCreate(OpenCL))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::CalculateItemNorm(CBufferFloat *source, const uint local_size,
                                     const uint groups)
  {
   if(!source || source.GetOpenCL() != OpenCL || source.GetIndex() < 0 ||
      local_size == 0 || groups == 0)
      return(false);
   const int partial_kernel = def_k_D2SkillItemPartial;
   setBuffer(partial_kernel, def_k_d2ip_source, source.GetIndex())
   setBuffer(partial_kernel, def_k_d2ip_partials, m_item_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2ip_dimension, (int)m_item_dimension)
   setArgument(partial_kernel, def_k_d2ip_output_offset, 0)
   uint offset[] = {0};
   uint global[] = {groups * local_size};
   uint local[] = {local_size};
   kernelExecuteLoc(partial_kernel, offset, global, local)
//--- Recursively reduce into disjoint regions of the scratch buffer.  The
//--- final scalar is written to its dedicated non-persistent buffer.
   uint input_count = groups;
   uint input_offset = 0;
   uint output_offset = groups;
   const int reduce_kernel = def_k_D2SkillItemReduce;
   while(input_count > 0)
     {
      const uint reduce_groups = (input_count + 2 * local_size - 1) / (2 * local_size);
      const int final_output = (reduce_groups == 1 ? 1 : 0);
      setBuffer(reduce_kernel, def_k_d2ir_partials, m_item_partials.GetIndex())
      setBuffer(reduce_kernel, def_k_d2ir_reduced, m_item_reduced_norm.GetIndex())
      setArgument(reduce_kernel, def_k_d2ir_input_count, (int)input_count)
      setArgument(reduce_kernel, def_k_d2ir_input_offset, (int)input_offset)
      setArgument(reduce_kernel, def_k_d2ir_output_offset, (int)output_offset)
      setArgument(reduce_kernel, def_k_d2ir_final_output, final_output)
      uint reduce_global[] = {reduce_groups * local_size};
      uint reduce_local[] = {local_size};
      kernelExecuteLoc(reduce_kernel, offset, reduce_global, reduce_local)
      if(final_output != 0)
         break;
      input_count = reduce_groups;
      input_offset = output_offset;
      output_offset += reduce_groups;
     }
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::UpdateFromGradient(CBufferFloat *gradient)
  {
   if(!gradient || !OpenCL || gradient.GetOpenCL() != OpenCL ||
      gradient.Total() != (int)m_item_dimension || gradient.GetIndex() < 0 ||
      !EnsureItemState())
      return(false);
   uint local_size = 0;
   uint groups = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count) ||
      !CalculateItemNorm(gradient, local_size, groups))
      return(false);
   const int kernel = def_k_D2SkillItemApply;
   setBuffer(kernel, def_k_d2ia_gradient, gradient.GetIndex())
   setBuffer(kernel, def_k_d2ia_correction, m_correction.GetIndex())
   setBuffer(kernel, def_k_d2ia_direction, m_item_direction.GetIndex())
   setBuffer(kernel, def_k_d2ia_scale, m_item_scale.GetIndex())
   setBuffer(kernel, def_k_d2ia_observations, m_item_observations.GetIndex())
   setBuffer(kernel, def_k_d2ia_mass, m_item_mass.GetIndex())
   setBuffer(kernel, def_k_d2ia_reduced_norm, m_item_reduced_norm.GetIndex())
   setArgument(kernel, def_k_d2ia_dimension, (int)m_item_dimension)
   setArgument(kernel, def_k_d2ia_representation, (int)m_representation)
   setArgument(kernel, def_k_d2ia_beta_correction, m_beta_correction)
   setArgument(kernel, def_k_d2ia_beta_direction, m_beta_direction)
   setArgument(kernel, def_k_d2ia_beta_magnitude, m_beta_magnitude)
   setArgument(kernel, def_k_d2ia_phase, 0)
   uint offset[] = {0};
   uint global[] = {groups * local_size};
   uint local[] = {local_size};
   kernelExecuteLoc(kernel, offset, global, local)
   if(m_representation == D2SkillDirectionMagnitude)
     {
      if(!CalculateItemNorm(GetPointer(m_item_direction), local_size, groups))
         return(false);
      setArgument(kernel, def_k_d2ia_phase, 1)
      kernelExecuteLoc(kernel, offset, global, local)
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Save(file_handle) ||
      m_item_dimension == 0 || !EnsureItemState())
      return(false);
   dFileWriteUInt(file_handle, D2SKILL_ITEM_FORMAT);
   dFileWriteUInt(file_handle, m_item_dimension);
   dFileWriteUInt(file_handle, (uint)m_representation);
   dFileWriteFloat(file_handle, m_beta_correction);
   dFileWriteFloat(file_handle, m_beta_direction);
   dFileWriteFloat(file_handle, m_beta_magnitude);
   return(m_item_correction.Save(file_handle) && m_item_direction.Save(file_handle) &&
          m_item_scale.Save(file_handle) && m_item_observations.Save(file_handle) &&
          m_item_mass.Save(file_handle));
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Load(file_handle))
      return(false);
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_ITEM_FORMAT)
      return(false);
   m_item_dimension = (uint)FileReadInteger(file_handle);
   m_representation = (ED2SkillRepresentation)FileReadInteger(file_handle);
   dFileReadUFloat(file_handle, m_beta_correction);
   dFileReadUFloat(file_handle, m_beta_direction);
   dFileReadUFloat(file_handle, m_beta_magnitude);
   if(m_item_dimension == 0 || m_representation < D2SkillFullResidual ||
      m_representation > D2SkillDirectionMagnitude)
      return(false);
   if(!m_item_correction.Load(file_handle) || !m_item_direction.Load(file_handle) ||
      !m_item_scale.Load(file_handle) || !m_item_observations.Load(file_handle) ||
      !m_item_mass.Load(file_handle))
      return(false);
   if(m_item_correction.Total() != (int)m_item_dimension ||
      m_item_direction.Total() != (int)m_item_dimension || m_item_scale.Total() != 1 ||
      m_item_observations.Total() != 1 || m_item_mass.Total() != 1)
      return(false);
   SetOpenCL(OpenCL);
   m_correction = GetPointer(m_item_correction);
   return(m_correction.GetOpenCL() == OpenCL && m_correction.GetIndex() >= 0);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl ||
      FileReadInteger(file_handle, INT_VALUE) != Type())
      return(false);
   AttachOpenCL(opencl);
   return(Load(file_handle));
  }
//+------------------------------------------------------------------+
void CD2SkillItem::SetOpenCL(COpenCLMy *obj)
  {
   if(!obj)
      return;
   CNeuronBaseOCL::SetOpenCL(obj);
   RebindItemBuffer(m_item_correction, obj);
   RebindItemBuffer(m_item_direction, obj);
   RebindItemBuffer(m_item_scale, obj);
   RebindItemBuffer(m_item_observations, obj);
   RebindItemBuffer(m_item_mass, obj);
   RebindItemBuffer(m_item_partials, obj);
   RebindItemBuffer(m_item_reduced_norm, obj);
   EnsureItemScratch();
   if(m_item_correction.Total() > 0 && m_item_correction.GetIndex() >= 0)
      m_correction = GetPointer(m_item_correction);
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::Clear(void)
  {
   m_correction = NULL;
   if(!CNeuronBaseOCL::Clear())
      return(false);
   return((m_item_correction.Total() == 0 || m_item_correction.Fill(0)) &&
          (m_item_direction.Total() == 0 || m_item_direction.Fill(0)) &&
           (m_item_scale.Total() == 0 || m_item_scale.Fill(0)) &&
           (m_item_observations.Total() == 0 || m_item_observations.Fill(0)) &&
           (m_item_mass.Total() == 0 || m_item_mass.Fill(0)) &&
           (m_item_partials.Total() == 0 || m_item_partials.Fill(0)) &&
           (m_item_reduced_norm.Total() == 0 || m_item_reduced_norm.Fill(0)));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::RetrievalLaunchLayout(uint &local_size, uint &partial_count) const
  {
   if(!OpenCL || m_slots == 0 || m_dimension == 0 || m_slots > RAG_FLOAT_INDEX_LIMIT ||
      OpenCL.GetMaxWorkGroupSize() < RAG_TOPK_LOCAL_WIDTH ||
      OpenCL.GetMaxLocalSize(0) < RAG_TOPK_LOCAL_WIDTH)
      return(false);
   long limit = 64;
   limit = MathMin(limit, OpenCL.GetMaxWorkGroupSize());
   limit = MathMin(limit, OpenCL.GetMaxLocalSize(0));
   if(limit <= 0)
      return(false);
   local_size = 1;
   while(local_size <= (uint)limit / 2)
      local_size *= 2;
   partial_count = (m_slots + RAG_TOPK_LOCAL_WIDTH - 1) / RAG_TOPK_LOCAL_WIDTH;
   return(local_size > 0 && partial_count > 0);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureRetrievalBuffer(CBufferFloat &buffer, const int total)
  {
   if(!OpenCL || total <= 0)
      return(false);
   if(buffer.Total() != total && !buffer.BufferInit(total, 0.0f))
      return(false);
   if(buffer.GetOpenCL() == OpenCL && buffer.GetIndex() >= 0)
      return(true);
   if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
      buffer.BufferFree();
   return(buffer.BufferCreate(OpenCL));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureRetrievalScratch(void)
  {
   uint local_size = 0;
   uint partial_count = 0;
   if(!RetrievalLaunchLayout(local_size, partial_count) || partial_count > INT_MAX / 2)
      return(false);
   const int pairs = (int)(2 * partial_count);
   return(EnsureRetrievalBuffer(m_retrieval_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_retrieval_partial, pairs) &&
           EnsureRetrievalBuffer(m_retrieval_merge, pairs) &&
           EnsureRetrievalBuffer(m_retrieval_top, 2));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateLaunchLayout(uint &local_size, uint &groups) const
  {
   if(!OpenCL || m_dimension == 0)
      return(false);
   uint limit = (uint)MathMin(64, MathMin(OpenCL.GetMaxWorkGroupSize(), OpenCL.GetMaxLocalSize(0)));
   if(limit == 0)
      return(false);
   local_size = 1;
   while(local_size <= limit / 2)
      local_size *= 2;
   groups = (m_dimension + local_size - 1) / local_size;
   return(groups > 0);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureUpdateScratch(void)
  {
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups) || groups > INT_MAX / D2SKILL_METRICS ||
      m_slots > INT_MAX / D2SKILL_DIAGNOSTIC_PARTIALS)
      return(false);
   return(EnsureRetrievalBuffer(m_free_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_eviction_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_free_top, 2) && EnsureRetrievalBuffer(m_eviction_top, 2) &&
          EnsureRetrievalBuffer(m_metric_partials, (int)(groups * D2SKILL_METRICS)) &&
          EnsureRetrievalBuffer(m_metrics, D2SKILL_METRICS) &&
          EnsureRetrievalBuffer(m_decision, 4) &&
          EnsureRetrievalBuffer(m_diagnostics_partials,
                                (int)(m_slots * D2SKILL_DIAGNOSTIC_PARTIALS)));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::RunTop1(CBufferFloat &scores, CBufferFloat &top)
  {
   uint local_size = 0;
   uint partial_count = 0;
   if(!RetrievalLaunchLayout(local_size, partial_count) || !EnsureRetrievalScratch())
      return(false);
   uint offset[1] = {0};
   const uint block_size = RAG_TOPK_LOCAL_WIDTH;
   const int partial_kernel = def_k_RAGPartial;
   setBuffer(partial_kernel, def_k_ragp_scores, scores.GetIndex())
   setBuffer(partial_kernel, def_k_ragp_partial, m_retrieval_partial.GetIndex())
   setArgument(partial_kernel, def_k_ragp_scenario_count, (int)m_slots)
   setArgument(partial_kernel, def_k_ragp_block_size, (int)block_size)
   setArgument(partial_kernel, def_k_ragp_top_k, 1)
   uint partial_global[1] = {partial_count * block_size};
   uint partial_local[1] = {block_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)
   uint count = partial_count;
   CBufferFloat *source = GetPointer(m_retrieval_partial);
   CBufferFloat *destination = GetPointer(m_retrieval_merge);
   const int merge_kernel = def_k_RAGMerge;
   while(count > block_size)
     {
      const uint blocks = (count + block_size - 1) / block_size;
      setBuffer(merge_kernel, def_k_ragm_source, source.GetIndex())
      setBuffer(merge_kernel, def_k_ragm_destination, destination.GetIndex())
      setArgument(merge_kernel, def_k_ragm_candidate_count, (int)count)
      setArgument(merge_kernel, def_k_ragm_block_size, (int)block_size)
      setArgument(merge_kernel, def_k_ragm_top_k, 1)
      uint global[1] = {blocks * block_size};
      uint local[1] = {block_size};
      kernelExecuteLoc(merge_kernel, offset, global, local)
      count = blocks;
      CBufferFloat *swap = source;
      source = destination;
      destination = swap;
     }
   setBuffer(merge_kernel, def_k_ragm_source, source.GetIndex())
   setBuffer(merge_kernel, def_k_ragm_destination, top.GetIndex())
   setArgument(merge_kernel, def_k_ragm_candidate_count, (int)count)
   setArgument(merge_kernel, def_k_ragm_block_size, (int)block_size)
   setArgument(merge_kernel, def_k_ragm_top_k, 1)
   uint global[1] = {block_size};
   uint local[1] = {block_size};
   kernelExecuteLoc(merge_kernel, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::RunMetrics(CBufferFloat *state, CBufferFloat *gradient,
                              const uint local_size, const uint groups)
  {
   if(!state || !gradient)
      return(false);
   uint offset[1] = {0};
   const int partial_kernel = def_k_D2SkillMetricPartial;
   setBuffer(partial_kernel, def_k_d2mp_query, state.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_gradient, gradient.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_selected_slot, m_selected_slot.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_keys, m_keys.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_directions, m_directions.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_used, m_used.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_valid, m_candidate_valid.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_key, m_candidate_key.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_direction, m_candidate_direction.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_partials, m_metric_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2mp_slots, (int)m_slots)
   setArgument(partial_kernel, def_k_d2mp_dimension, (int)m_dimension)
   uint partial_global[1] = {groups * local_size};
   uint partial_local[1] = {local_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)
   const int reduce_kernel = def_k_D2SkillMetricReduce;
   setBuffer(reduce_kernel, def_k_d2mr_partials, m_metric_partials.GetIndex())
   setBuffer(reduce_kernel, def_k_d2mr_metrics, m_metrics.GetIndex())
   setArgument(reduce_kernel, def_k_d2mr_input_count, (int)groups)
   uint global[1] = {D2SKILL_METRICS * local_size};
   uint local[1] = {local_size};
   kernelExecuteLoc(reduce_kernel, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::CalculateBankVectorNorm(const uint local_size, const uint groups)
  {
   if(local_size == 0 || groups == 0 || !EnsureItemScratch())
      return(false);
   uint offset[1] = {0};
   const int partial_kernel = def_k_D2SkillBankVectorPartial;
   setBuffer(partial_kernel, def_k_d2bvp_corrections, m_corrections.GetIndex())
   setBuffer(partial_kernel, def_k_d2bvp_directions, m_directions.GetIndex())
   setBuffer(partial_kernel, def_k_d2bvp_decision, m_decision.GetIndex())
   setBuffer(partial_kernel, def_k_d2bvp_partials, m_item_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2bvp_dimension, (int)m_dimension)
   setArgument(partial_kernel, def_k_d2bvp_representation, (int)m_representation)
   setArgument(partial_kernel, def_k_d2bvp_output_offset, 0)
   uint partial_global[1] = {groups * local_size};
   uint partial_local[1] = {local_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)
   uint input_count = groups;
   uint input_offset = 0;
   uint output_offset = groups;
   const int reduce_kernel = def_k_D2SkillItemReduce;
   while(input_count > 0)
     {
      const uint reduce_groups = (input_count + 2 * local_size - 1) / (2 * local_size);
      const int final_output = (reduce_groups == 1 ? 1 : 0);
      setBuffer(reduce_kernel, def_k_d2ir_partials, m_item_partials.GetIndex())
      setBuffer(reduce_kernel, def_k_d2ir_reduced, m_item_reduced_norm.GetIndex())
      setArgument(reduce_kernel, def_k_d2ir_input_count, (int)input_count)
      setArgument(reduce_kernel, def_k_d2ir_input_offset, (int)input_offset)
      setArgument(reduce_kernel, def_k_d2ir_output_offset, (int)output_offset)
      setArgument(reduce_kernel, def_k_d2ir_final_output, final_output)
      uint reduce_global[1] = {reduce_groups * local_size};
      uint reduce_local[1] = {local_size};
      kernelExecuteLoc(reduce_kernel, offset, reduce_global, reduce_local)
      if(final_output != 0)
         break;
      input_count = reduce_groups;
      input_offset = output_offset;
      output_offset += reduce_groups;
     }
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Init(uint dimension, uint myIndex, COpenCLMy *open_cl,
                        uint slots, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!open_cl || dimension == 0 || slots == 0 || slots > RAG_FLOAT_INDEX_LIMIT ||
      !CD2SkillItem::Init(0, myIndex, open_cl, dimension, optimization_type, batch))
      return(false);
   m_dimension = dimension;
   m_slots = slots;
   m_active_slot = UINT_MAX;
   const long total = (long)m_slots * (long)m_dimension;
   if(total <= 0 || total > 2147483647)
      return(false);
   if(!InitBuffer(m_keys, (int)total) || !InitBuffer(m_corrections, (int)total) ||
      !InitBuffer(m_directions, (int)total) || !InitBuffer(m_scales, (int)m_slots) ||
      !InitBuffer(m_utility, (int)m_slots) || !InitBuffer(m_observations, (int)m_slots) ||
      !InitBuffer(m_uses, (int)m_slots) || !InitBuffer(m_mass, (int)m_slots) ||
      !InitBuffer(m_used, (int)m_slots) || !InitBuffer(m_state, (int)m_slots) ||
      !InitBuffer(m_age, (int)m_slots) || !InitBuffer(m_protection, (int)m_slots) ||
      !InitBuffer(m_candidate_streak, 1) || !InitBuffer(m_candidate_valid, 1) ||
      !InitBuffer(m_candidate_key, (int)m_dimension) ||
      !InitBuffer(m_candidate_direction, (int)m_dimension) ||
      !InitBuffer(m_candidate_scale, 1) ||
      !InitBuffer(m_selected_correction, (int)m_dimension) ||
      !InitBuffer(m_selected_slot, 1, -1.0f) ||
      !InitBuffer(m_selected_score, 1, -3.402823e+38f) ||
      !InitBuffer(m_diagnostics, D2SKILL_DIAGNOSTICS) ||
      !InitBuffer(m_utility_distribution, D2SKILL_DISTRIBUTION_BINS) ||
      !InitBuffer(m_usage_distribution, D2SKILL_DISTRIBUTION_BINS) ||
      !EnsureRetrievalScratch() || !EnsureUpdateScratch())
      return(false);
   if(!SetCorrectionBuffer(GetPointer(m_selected_correction)))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
void CD2SkillBank::SetOpenCL(COpenCLMy *obj)
  {
   CD2SkillItem::SetOpenCL(obj);
   RebindBuffer(m_keys, obj);
   RebindBuffer(m_corrections, obj);
   RebindBuffer(m_directions, obj);
   RebindBuffer(m_scales, obj);
   RebindBuffer(m_utility, obj);
   RebindBuffer(m_observations, obj);
   RebindBuffer(m_uses, obj);
   RebindBuffer(m_mass, obj);
   RebindBuffer(m_used, obj);
   RebindBuffer(m_state, obj);
   RebindBuffer(m_age, obj);
   RebindBuffer(m_protection, obj);
   RebindBuffer(m_candidate_streak, obj);
   RebindBuffer(m_candidate_valid, obj);
   RebindBuffer(m_candidate_key, obj);
   RebindBuffer(m_candidate_direction, obj);
   RebindBuffer(m_candidate_scale, obj);
   RebindBuffer(m_selected_correction, obj);
   RebindBuffer(m_selected_slot, obj);
   RebindBuffer(m_selected_score, obj);
   RebindBuffer(m_diagnostics, obj);
   RebindBuffer(m_utility_distribution, obj);
   RebindBuffer(m_usage_distribution, obj);
   RebindBuffer(m_free_scores, obj);
   RebindBuffer(m_eviction_scores, obj);
   RebindBuffer(m_free_top, obj);
   RebindBuffer(m_eviction_top, obj);
   RebindBuffer(m_metric_partials, obj);
   RebindBuffer(m_metrics, obj);
   RebindBuffer(m_decision, obj);
   RebindBuffer(m_diagnostics_partials, obj);
   EnsureRetrievalScratch();
   EnsureUpdateScratch();
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Retrieve(CBufferFloat *state)
  {
   if(!m_enabled || !state || state.Total() != (int)m_dimension ||
      state.GetOpenCL() != OpenCL || state.GetIndex() < 0 ||
      !EnsureRetrievalScratch())
      return(false);
   uint score_local_size = 0;
   uint partial_count = 0;
   if(!RetrievalLaunchLayout(score_local_size, partial_count))
      return(false);
   uint offset[1] = {0};
   const int score_kernel = def_k_D2SkillScoreSlots;
   setBuffer(score_kernel, def_k_d2ss_query, state.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_keys, m_keys.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_utility, m_utility.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_used, m_used.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_state, m_state.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_scores, m_retrieval_scores.GetIndex())
   setArgument(score_kernel, def_k_d2ss_slots, (int)m_slots)
   setArgument(score_kernel, def_k_d2ss_dimension, (int)m_dimension)
   setArgument(score_kernel, def_k_d2ss_similarity_threshold, m_similarity_threshold)
   setArgument(score_kernel, def_k_d2ss_utility_weight, m_utility_weight)
   setArgument(score_kernel, def_k_d2ss_min_utility, m_min_utility)
   setArgument(score_kernel, def_k_d2ss_utility_policy, (int)m_utility_aware)
   uint score_global[1] = {m_slots * score_local_size};
   uint score_local[1] = {score_local_size};
   kernelExecuteLoc(score_kernel, offset, score_global, score_local)

//--- Reuse the generic RAG Top-1 contracts without changing their data
//--- layout: float scores become float2(slot, score) ping-pong candidates.
   const uint block_size = RAG_TOPK_LOCAL_WIDTH;
   const uint top_k = 1;
   const int partial_kernel = def_k_RAGPartial;
   setBuffer(partial_kernel, def_k_ragp_scores, m_retrieval_scores.GetIndex())
   setBuffer(partial_kernel, def_k_ragp_partial, m_retrieval_partial.GetIndex())
   setArgument(partial_kernel, def_k_ragp_scenario_count, (int)m_slots)
   setArgument(partial_kernel, def_k_ragp_block_size, (int)block_size)
   setArgument(partial_kernel, def_k_ragp_top_k, (int)top_k)
   uint partial_global[1] = {partial_count * block_size};
   uint partial_local[1] = {block_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)

   uint merge_count = partial_count;
   CBufferFloat *source = GetPointer(m_retrieval_partial);
   CBufferFloat *destination = GetPointer(m_retrieval_merge);
   const int merge_kernel = def_k_RAGMerge;
   while(merge_count > block_size)
     {
      const uint merge_blocks = (merge_count + block_size - 1) / block_size;
      const uint next_count = merge_blocks * top_k;
      setBuffer(merge_kernel, def_k_ragm_source, source.GetIndex())
      setBuffer(merge_kernel, def_k_ragm_destination, destination.GetIndex())
      setArgument(merge_kernel, def_k_ragm_candidate_count, (int)merge_count)
      setArgument(merge_kernel, def_k_ragm_block_size, (int)block_size)
      setArgument(merge_kernel, def_k_ragm_top_k, (int)top_k)
      uint merge_global[1] = {merge_blocks * block_size};
      uint merge_local[1] = {block_size};
      kernelExecuteLoc(merge_kernel, offset, merge_global, merge_local)
      merge_count = next_count;
      CBufferFloat *swap = source;
      source = destination;
      destination = swap;
     }
   setBuffer(merge_kernel, def_k_ragm_source, source.GetIndex())
   setBuffer(merge_kernel, def_k_ragm_destination, m_retrieval_top.GetIndex())
   setArgument(merge_kernel, def_k_ragm_candidate_count, (int)merge_count)
   setArgument(merge_kernel, def_k_ragm_block_size, (int)block_size)
   setArgument(merge_kernel, def_k_ragm_top_k, (int)top_k)
   uint merge_global[1] = {block_size};
   uint merge_local[1] = {block_size};
   kernelExecuteLoc(merge_kernel, offset, merge_global, merge_local)

   const int gather_kernel = def_k_D2SkillGatherCorrection;
   setBuffer(gather_kernel, def_k_d2gc_top, m_retrieval_top.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_corrections, m_corrections.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_directions, m_directions.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_scales, m_scales.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_selected_correction, m_selected_correction.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_selected_slot, m_selected_slot.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_selected_score, m_selected_score.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_uses, m_uses.GetIndex())
   setBuffer(gather_kernel, def_k_d2gc_diagnostics, m_diagnostics.GetIndex())
   setArgument(gather_kernel, def_k_d2gc_slots, (int)m_slots)
   setArgument(gather_kernel, def_k_d2gc_dimension, (int)m_dimension)
   setArgument(gather_kernel, def_k_d2gc_representation, (int)m_representation)
   setArgument(gather_kernel, def_k_d2gc_alpha, m_alpha)
   uint gather_global[1] = {m_dimension};
   kernelExecute(gather_kernel, offset, gather_global)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Output) == POINTER_INVALID)
      return(false);
   CBufferFloat *source = NeuronOCL.getOutput();
   if(!source || source.Total() != (int)m_dimension || source.GetOpenCL() != OpenCL)
     {
      PrintFormat("D2SkillBank forward contract: source=%p total=%d expected=%u source_cl=%p bank_cl=%p",
                  source, (source ? source.Total() : -1), m_dimension,
                  (source ? source.GetOpenCL() : NULL), OpenCL);
      return(false);
     }
   if(!m_enabled)
      return(SumAndNormalize(source, source, Output, (int)m_dimension, false,
                             0, 0, 0, 0.5f));
   if(!Retrieve(source))
      return(false);
//--- CD2SkillItem owns the residual application.  The bank only selects
//--- m_selected_correction and exposes it through the inherited buffer.
   return(CD2SkillItem::feedForward(NeuronOCL));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateFromGradient(CBufferFloat *state, CBufferFloat *gradient)
  {
//--- The D2 update consumes the pseudo-residual g=-dL_Actor/dh.
//--- CNeuronBaseOCL::calcOutputGradients writes target-output (the
//--- negative MSE derivative), so the gradient buffer is already g and
//--- must not be negated again here.  Direct callers must provide the
//--- same pseudo-residual contract.
   if(!m_enabled || !OpenCL || !state || !gradient ||
       state.Total() != (int)m_dimension || gradient.Total() != (int)m_dimension ||
       state.GetOpenCL() != OpenCL || gradient.GetOpenCL() != OpenCL ||
       !EnsureUpdateScratch())
      return(false);
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      return(false);
   uint offset[1] = {0};
   const int age_kernel = def_k_D2SkillAgeLifecycle;
   setBuffer(age_kernel, def_k_d2al_used, m_used.GetIndex())
   setBuffer(age_kernel, def_k_d2al_state, m_state.GetIndex())
   setBuffer(age_kernel, def_k_d2al_age, m_age.GetIndex())
   setBuffer(age_kernel, def_k_d2al_protection, m_protection.GetIndex())
   setBuffer(age_kernel, def_k_d2al_utility, m_utility.GetIndex())
   setBuffer(age_kernel, def_k_d2al_free_scores, m_free_scores.GetIndex())
   setBuffer(age_kernel, def_k_d2al_eviction_scores, m_eviction_scores.GetIndex())
   setArgument(age_kernel, def_k_d2al_slots, (int)m_slots)
   setArgument(age_kernel, def_k_d2al_inactivity_age, (int)m_inactivity_age)
   uint slot_global[1] = {m_slots};
   kernelExecute(age_kernel, offset, slot_global)
   if(!RunTop1(m_free_scores, m_free_top) || !RunTop1(m_eviction_scores, m_eviction_top) ||
      !RunMetrics(state, gradient, local_size, groups))
      return(false);
   const int leader_kernel = def_k_D2SkillCandidateLeader;
   setBuffer(leader_kernel, def_k_d2cl_selected_slot, m_selected_slot.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_used, m_used.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_scales, m_scales.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_observations, m_observations.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_mass, m_mass.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_age, m_age.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_streak, m_candidate_streak.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_valid, m_candidate_valid.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_scale, m_candidate_scale.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_metrics, m_metrics.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_free_top, m_free_top.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_eviction_top, m_eviction_top.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_decision, m_decision.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_diagnostics, m_diagnostics.GetIndex())
   setArgument(leader_kernel, def_k_d2cl_slots, (int)m_slots)
   setArgument(leader_kernel, def_k_d2cl_representation, (int)m_representation)
   setArgument(leader_kernel, def_k_d2cl_beta_magnitude, m_beta_magnitude)
   setArgument(leader_kernel, def_k_d2cl_max_correction, m_max_correction)
   setArgument(leader_kernel, def_k_d2cl_min_confirmations, (int)m_min_confirmations)
   setArgument(leader_kernel, def_k_d2cl_similarity_threshold, m_similarity_threshold)
   setArgument(leader_kernel, def_k_d2cl_direction_threshold, m_direction_threshold)
   uint scalar_global[1] = {1};
   kernelExecute(leader_kernel, offset, scalar_global)
   const int vector_kernel = def_k_D2SkillVectorUpdate;
   setBuffer(vector_kernel, def_k_d2vu_query, state.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_gradient, gradient.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_keys, m_keys.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_corrections, m_corrections.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_directions, m_directions.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_scales, m_scales.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_key, m_candidate_key.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_direction, m_candidate_direction.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_scale, m_candidate_scale.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_decision, m_decision.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_reduced_norm, m_item_reduced_norm.GetIndex())
   setArgument(vector_kernel, def_k_d2vu_dimension, (int)m_dimension)
   setArgument(vector_kernel, def_k_d2vu_representation, (int)m_representation)
   setArgument(vector_kernel, def_k_d2vu_beta_correction, m_beta_correction)
   setArgument(vector_kernel, def_k_d2vu_beta_direction, m_beta_direction)
   setArgument(vector_kernel, def_k_d2vu_beta_key, m_beta_key)
   setArgument(vector_kernel, def_k_d2vu_max_correction, m_max_correction)
   setArgument(vector_kernel, def_k_d2vu_phase, 0)
   uint vector_global[1] = {groups * local_size};
   uint vector_local[1] = {local_size};
   kernelExecuteLoc(vector_kernel, offset, vector_global, vector_local)
   if(!CalculateBankVectorNorm(local_size, groups))
      return(false);
   setArgument(vector_kernel, def_k_d2vu_phase, 1)
   kernelExecuteLoc(vector_kernel, offset, vector_global, vector_local)
   const int promote_kernel = def_k_D2SkillPromoteSlot;
   setBuffer(promote_kernel, def_k_d2ps_scales, m_scales.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_utility, m_utility.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_observations, m_observations.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_uses, m_uses.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_mass, m_mass.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_used, m_used.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_state, m_state.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_age, m_age.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_protection, m_protection.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_candidate_streak, m_candidate_streak.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_candidate_valid, m_candidate_valid.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_candidate_scale, m_candidate_scale.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_decision, m_decision.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_diagnostics, m_diagnostics.GetIndex())
   setArgument(promote_kernel, def_k_d2ps_slots, (int)m_slots)
   setArgument(promote_kernel, def_k_d2ps_protection_age, (int)m_protection_age)
   kernelExecute(promote_kernel, offset, scalar_global)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateUtility(const float delta_j)
  {
   if(!m_enabled || !OpenCL || !MathIsValidNumber(delta_j))
      return(false);
   const int kernel = def_k_D2SkillUtility;
   setBuffer(kernel, def_k_d2ut_selected_slot, m_selected_slot.GetIndex())
   setBuffer(kernel, def_k_d2ut_utility, m_utility.GetIndex())
   setBuffer(kernel, def_k_d2ut_used, m_used.GetIndex())
   setBuffer(kernel, def_k_d2ut_diagnostics, m_diagnostics.GetIndex())
   setArgument(kernel, def_k_d2ut_delta, delta_j)
   setArgument(kernel, def_k_d2ut_beta, m_beta_utility)
   setArgument(kernel, def_k_d2ut_slots, (int)m_slots)
   uint offset[1] = {0};
   uint global[1] = {1};
   kernelExecute(kernel, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::RefreshDiagnostics(void)
  {
   if(!OpenCL || !ValidateBuffers() || !EnsureUpdateScratch())
      return(false);
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      return(false);
   uint offset[1] = {0};
   const int partial_kernel = def_k_D2SkillDiagnosticsPartial;
   setBuffer(partial_kernel, def_k_d2dp_corrections, m_corrections.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_utility, m_utility.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_uses, m_uses.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_used, m_used.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_state, m_state.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_age, m_age.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_partials, m_diagnostics_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2dp_slots, (int)m_slots)
   setArgument(partial_kernel, def_k_d2dp_dimension, (int)m_dimension)
   uint partial_global[1] = {m_slots * local_size};
   uint partial_local[1] = {local_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)
   const int reduce_kernel = def_k_D2SkillDiagnosticsReduce;
   setBuffer(reduce_kernel, def_k_d2dr_partials, m_diagnostics_partials.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_candidate_valid, m_candidate_valid.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_diagnostics, m_diagnostics.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_utility_distribution, m_utility_distribution.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_usage_distribution, m_usage_distribution.GetIndex())
   setArgument(reduce_kernel, def_k_d2dr_slots, (int)m_slots)
   uint reduce_global[1] = {local_size};
   uint reduce_local[1] = {local_size};
   kernelExecuteLoc(reduce_kernel, offset, reduce_global, reduce_local)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !RefreshDiagnostics() || !CNeuronBaseOCL::Save(file_handle))
      return(false);
   dFileWriteUInt(file_handle, D2SKILL_FORMAT);
   dFileWriteUInt(file_handle, m_slots);
   dFileWriteUInt(file_handle, m_dimension);
   dFileWriteUInt(file_handle, (uint)m_representation);
   dFileWriteUInt(file_handle, (uint)m_enabled);
   dFileWriteUInt(file_handle, m_min_confirmations);
   dFileWriteUInt(file_handle, m_protection_age);
   dFileWriteUInt(file_handle, m_inactivity_age);
   dFileWriteFloat(file_handle, m_similarity_threshold);
   dFileWriteFloat(file_handle, m_direction_threshold);
   dFileWriteFloat(file_handle, m_utility_weight);
   dFileWriteFloat(file_handle, m_alpha);
   dFileWriteFloat(file_handle, m_min_utility);
   dFileWriteUInt(file_handle, (uint)m_utility_aware);
   dFileWriteFloat(file_handle, m_beta_key);
   dFileWriteFloat(file_handle, m_beta_utility);
   dFileWriteFloat(file_handle, m_max_correction);
   dFileWriteFloat(file_handle, m_beta_correction);
   dFileWriteFloat(file_handle, m_beta_direction);
   dFileWriteFloat(file_handle, m_beta_magnitude);
   return(m_keys.Save(file_handle) && m_corrections.Save(file_handle) &&
          m_directions.Save(file_handle) && m_scales.Save(file_handle) &&
          m_utility.Save(file_handle) && m_observations.Save(file_handle) &&
          m_uses.Save(file_handle) && m_mass.Save(file_handle) && m_used.Save(file_handle) &&
          m_state.Save(file_handle) && m_age.Save(file_handle) && m_protection.Save(file_handle) &&
          m_candidate_streak.Save(file_handle) && m_candidate_valid.Save(file_handle) &&
          m_candidate_key.Save(file_handle) && m_candidate_direction.Save(file_handle) &&
          m_candidate_scale.Save(file_handle) && m_selected_correction.Save(file_handle) &&
          m_selected_slot.Save(file_handle) && m_selected_score.Save(file_handle) &&
          m_diagnostics.Save(file_handle) &&
          m_utility_distribution.Save(file_handle) &&
          m_usage_distribution.Save(file_handle));
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Load(file_handle))
      return(false);
   const uint format = (uint)FileReadInteger(file_handle);
   if(format < 3 || format > D2SKILL_FORMAT)
      return(false);
   m_slots = (uint)FileReadInteger(file_handle);
   m_dimension = (uint)FileReadInteger(file_handle);
   m_item_dimension = m_dimension;
   m_representation = (ED2SkillRepresentation)FileReadInteger(file_handle);
   m_enabled = (FileReadInteger(file_handle) != 0);
   m_min_confirmations = (uint)FileReadInteger(file_handle);
   m_protection_age = (uint)FileReadInteger(file_handle);
   m_inactivity_age = (format >= 4 ? (uint)FileReadInteger(file_handle) : 256);
   dFileReadUFloat(file_handle, m_similarity_threshold);
   dFileReadUFloat(file_handle, m_direction_threshold);
   dFileReadUFloat(file_handle, m_utility_weight);
   if(format >= 5)
     {
      dFileReadUFloat(file_handle, m_alpha);
      dFileReadUFloat(file_handle, m_min_utility);
      m_utility_aware = (FileReadInteger(file_handle) != 0);
     }
   else
     {
      m_alpha = 1.0f;
      m_min_utility = -1.0f;
      m_utility_aware = false;
     }
   dFileReadUFloat(file_handle, m_beta_key);
   dFileReadUFloat(file_handle, m_beta_utility);
   dFileReadUFloat(file_handle, m_max_correction);
   dFileReadUFloat(file_handle, m_beta_correction);
   dFileReadUFloat(file_handle, m_beta_direction);
   dFileReadUFloat(file_handle, m_beta_magnitude);
   if(m_slots == 0 || m_dimension == 0 || m_min_confirmations == 0)
      return(false);
   const long total = (long)m_slots * (long)m_dimension;
   if(!m_keys.Load(file_handle) || !m_corrections.Load(file_handle) ||
      !m_directions.Load(file_handle) || !m_scales.Load(file_handle) ||
      !m_utility.Load(file_handle) || !m_observations.Load(file_handle) ||
      !m_uses.Load(file_handle) || !m_mass.Load(file_handle) || !m_used.Load(file_handle) ||
      !m_state.Load(file_handle) || !m_age.Load(file_handle) || !m_protection.Load(file_handle) ||
      !m_candidate_streak.Load(file_handle) || !m_candidate_valid.Load(file_handle) ||
      !m_candidate_key.Load(file_handle) || !m_candidate_direction.Load(file_handle) ||
      !m_candidate_scale.Load(file_handle) || !m_selected_correction.Load(file_handle) ||
      !m_selected_slot.Load(file_handle) || !m_selected_score.Load(file_handle))
      return(false);
   if(format >= 5 && !m_diagnostics.Load(file_handle))
      return(false);
   if(format < 5 || m_diagnostics.Total() != D2SKILL_DIAGNOSTICS)
      if(!InitBuffer(m_diagnostics, D2SKILL_DIAGNOSTICS))
         return(false);
   if(format >= 7)
     {
      if(!m_utility_distribution.Load(file_handle) ||
         !m_usage_distribution.Load(file_handle))
         return(false);
     }
   else
     {
      if(!InitBuffer(m_utility_distribution, D2SKILL_DISTRIBUTION_BINS) ||
         !InitBuffer(m_usage_distribution, D2SKILL_DISTRIBUTION_BINS))
         return(false);
     }
   if(total > 2147483647 || !ValidateBuffers())
      return(false);
   SetOpenCL(OpenCL);
   if(!EnsureRetrievalScratch() || !EnsureUpdateScratch() ||
       !SetCorrectionBuffer(GetPointer(m_selected_correction)))
      return(false);
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl ||
      FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank)
      return(false);
   OpenCL = opencl;
   if(!Load(file_handle))
      return(false);
   SetOpenCL(opencl);
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Clear(void)
  {
   m_correction = NULL;
   if(!CNeuronBaseOCL::Clear())
      return(false);
   m_active_slot = UINT_MAX;
   if(!(m_keys.Fill(0) && m_corrections.Fill(0) && m_directions.Fill(0) &&
        m_scales.Fill(0) && m_utility.Fill(0) && m_observations.Fill(0) &&
        m_uses.Fill(0) && m_mass.Fill(0) && m_used.Fill(0) && m_state.Fill(0) &&
        m_age.Fill(0) && m_protection.Fill(0) && m_candidate_streak.Fill(0) &&
        m_candidate_valid.Fill(0) && m_candidate_key.Fill(0) &&
         m_candidate_direction.Fill(0) && m_candidate_scale.Fill(0) &&
         m_selected_correction.Fill(0) && m_selected_slot.Fill(-1.0f) &&
         m_selected_score.Fill(-3.402823e+38f) && m_diagnostics.Fill(0) &&
         m_utility_distribution.Fill(0) && m_usage_distribution.Fill(0) &&
         (m_retrieval_scores.Total() == 0 || m_retrieval_scores.Fill(0)) &&
          (m_retrieval_partial.Total() == 0 || m_retrieval_partial.Fill(0)) &&
          (m_retrieval_merge.Total() == 0 || m_retrieval_merge.Fill(0)) &&
          (m_retrieval_top.Total() == 0 || m_retrieval_top.Fill(0)) &&
          (m_free_scores.Total() == 0 || m_free_scores.Fill(0)) &&
          (m_eviction_scores.Total() == 0 || m_eviction_scores.Fill(0)) &&
          (m_free_top.Total() == 0 || m_free_top.Fill(0)) &&
          (m_eviction_top.Total() == 0 || m_eviction_top.Fill(0)) &&
          (m_metric_partials.Total() == 0 || m_metric_partials.Fill(0)) &&
          (m_metrics.Total() == 0 || m_metrics.Fill(0)) &&
          (m_decision.Total() == 0 || m_decision.Fill(0)) &&
          (m_diagnostics_partials.Total() == 0 || m_diagnostics_partials.Fill(0))))
      return(false);
   return(SetCorrectionBuffer(GetPointer(m_selected_correction)));
  }
#endif // NEURONET_D2SKILL_MQH
